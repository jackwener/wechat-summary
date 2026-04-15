"""
workflows.py — High-level workflows composing Locate + Act.

Each workflow is a sequence of Locate(element) → Act(action) steps.
Element.click() bridges Locate and Act — the element knows its position.
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from wechat import actions
from wechat.locator import Locator
from wechat.ocr import describe_target_type


def navigate_to_chat(
    locator: Locator,
    group_name: str,
    target_type: str = "group",
) -> None:
    """
    Navigate to a specific chat.

    Strategy:
    1. Locate(sidebar_item) → elem.click()
    2. If not found: Act(cmd_f) → Act(type) → Locate(search_result) → elem.click()
    3. Verify via Locate(chat_title)

    Raises RuntimeError if navigation fails.
    """

    # --- Strategy 1: Locate in sidebar, click ---
    elem = locator.sidebar_item(group_name)

    if elem is not None:
        print(f"  Found '{group_name}' in sidebar: {elem}")
        actions.activate_wechat()
        time.sleep(0.1)
        elem.click()
        # Retry verification twice to handle slow chat transitions.
        for wait in (1.0, 1.5):
            time.sleep(wait)
            if locator.verify_chat_title(group_name):
                print(f"  ✅ Chat verification passed: title bar matches '{group_name}'")
                print(f"Navigated to: {group_name}")
                return
            print(f"  ↩  Title not matched yet (waited {wait}s), retrying...")

        print(f"  ⚠️  Sidebar click landed on wrong chat, falling back to search...")
    else:
        print(f"  '{group_name}' not visible in sidebar, searching...")

    # --- Strategy 2: Search via Cmd+F ---
    _search_and_click(locator, group_name, target_type)

    # --- Dismiss search panel (restores sidebar) ---
    # After clicking a search result WeChat keeps the search overlay open.
    # Pressing Escape closes it so that layout re-detection sees the normal
    # sidebar/chat divider instead of the search panel's different boundary.
    actions.press_escape()
    time.sleep(0.4)

    # --- Verify title bar ---
    if locator.verify_chat_title(group_name):
        print(f"  ✅ Chat verification passed: title bar matches '{group_name}'")
    else:
        raise RuntimeError(
            f"Navigation failed: title bar does not match '{group_name}'. "
            "WeChat may have navigated to the wrong chat."
        )

    print(f"Navigated to: {group_name}")


def _search_and_click(
    locator: Locator,
    group_name: str,
    target_type: str,
) -> None:
    """Act(cmd_f) → Act(type) → verify → Locate(search_result) → elem.click()."""
    print(f"  Searching for '{group_name}' via Cmd+F...")

    # Act: open search
    actions.cmd_f()

    # Act: type search text
    actions.type_text(group_name)
    time.sleep(1.5)

    # Verify search results contain target (uses Snapshot internally)
    if locator.verify_search_results(group_name):
        print(f"  ✅ Search verification passed: found '{group_name}' in search results")
    else:
        print(f"  ⚠️  Search verification: '{group_name}' not found in results, retrying...")
        # Retry: clear and re-type
        actions.select_all_and_delete()
        time.sleep(0.3)
        actions.type_text(group_name)
        time.sleep(2.0)
        if locator.verify_search_results(group_name):
            print(f"  ✅ Search verification passed on retry")
        else:
            raise RuntimeError(
                f"Navigation failed: could not find '{group_name}' in search results. "
                "Please ensure the chat name is correct and you are a member."
            )

    # Locate: find the correct search result (reuses Snapshot from verify)
    elem = locator.search_result(group_name, target_type)

    if elem is None:
        raise RuntimeError(
            f"Navigation failed: no match for '{group_name}' found in "
            f"{describe_target_type(target_type)} section(s)."
        )

    # Act: click the search result
    print(f"  Clicking search result: {elem}")
    actions.activate_wechat()
    elem.click()
    time.sleep(1.0)


def send_message(
    locator: Locator,
    message: str,
    group_name: Optional[str] = None,
    target_type: str = "group",
) -> None:
    """
    Send a message in the current or specified chat.

    Workflow:
    1. (Optional) navigate_to_chat if group_name specified
    2. Locate(input_box) → elem.click()
    3. Act(type_text) — paste message
    4. Act(press_enter) — send
    """
    if not message.strip():
        raise ValueError("Message cannot be empty.")

    if group_name:
        navigate_to_chat(locator, group_name, target_type=target_type)
        time.sleep(0.5)
        locator.refresh()

    # Locate input box → click
    elem = locator.input_box()
    print(f"  Clicking input box: {elem}")
    actions.activate_wechat()
    elem.click()
    time.sleep(0.3)

    # Type and send
    print(f"  Typing message ({len(message)} chars)")
    actions.type_text(message)
    time.sleep(0.3)
    actions.press_enter()
    print(f"  ✅ Message sent")


def capture_screenshots(
    locator: Locator,
    max_pages: int = 30,
    output_dir: str = "screenshots",
    scroll_delay: float = 0.8,
) -> list[str]:
    """
    Capture chat screenshots by scrolling up page by page.

    Workflow:
    1. Locate(chat_area_center) → elem.click() for focus
    2. Loop: Act(screenshot) → Act(page_up)
    3. Detect duplicates (reached top) → stop

    Returns list of screenshot file paths in chronological order (oldest first).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    chat_area = locator.chat_area()
    print(f"Window: {locator.window}")
    print(f"Chat area: {chat_area}")

    # Click chat area for keyboard focus
    center = locator.chat_area_center()
    center.click()
    time.sleep(0.5)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshots = []
    prev_hash = None

    # Capture current (most recent) view
    filepath = str(out / f"{timestamp}_page_000.png")
    actions.screenshot(chat_area, filepath)
    screenshots.append(filepath)
    prev_hash = actions.image_hash(filepath)
    print(f"Captured page 0 (current view)")

    # Scroll up to capture older messages
    for i in range(1, max_pages):
        actions.page_up()
        time.sleep(scroll_delay)

        filepath = str(out / f"{timestamp}_page_{i:03d}.png")
        actions.screenshot(chat_area, filepath)

        curr_hash = actions.image_hash(filepath)
        if curr_hash == prev_hash:
            os.remove(filepath)
            print(f"Reached top of chat at page {i}. Stopping.")
            break

        screenshots.append(filepath)
        prev_hash = curr_hash
        print(f"Captured page {i}")

    # Reverse for chronological order (oldest first)
    screenshots.reverse()
    print(f"\nTotal pages captured: {len(screenshots)}")
    return screenshots

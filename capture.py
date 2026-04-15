"""
capture.py — Screenshot capture for macOS WeChat (backward-compatible wrapper).

Delegates to the Locate→Act framework in wechat/ package and workflows.py.

Public API preserved:
  - capture_chat(max_pages, output_dir, scroll_delay, group_name, target_type) → list[str]
  - navigate_to_chat(group_name, window, layout, target_type)
  - get_wechat_window_info() → dict
  - detect_layout(window) → dict
  - get_chat_area(window, layout) → dict
  - take_screenshot(region, filepath) → str
"""

import time
from typing import Optional

# Re-export low-level functions for backward compatibility
from wechat.actions import get_window_info as get_wechat_window_info
from wechat.actions import screenshot as take_screenshot
from wechat.layout import detect_layout, get_chat_area
from wechat.locator import Locator
from workflows import capture_screenshots, navigate_to_chat as _wf_navigate


def navigate_to_chat(
    group_name: str,
    window: dict,
    layout: Optional[dict] = None,
    target_type: str = "group",
) -> None:
    """
    Navigate to a specific chat. Backward-compatible wrapper.

    See workflows.navigate_to_chat for the Locate→Act implementation.
    """
    locator = Locator.__new__(Locator)
    locator.window = window
    locator.layout = layout or detect_layout(window)
    _wf_navigate(locator, group_name, target_type=target_type)


def capture_chat(
    max_pages: int = 30,
    output_dir: str = "screenshots",
    scroll_delay: float = 0.8,
    group_name: Optional[str] = None,
    target_type: str = "group",
) -> list[str]:
    """
    Capture WeChat chat screenshots by scrolling up page by page.

    Returns list of screenshot file paths in chronological order (oldest first).
    """
    locator = Locator()

    if group_name:
        _wf_navigate(locator, group_name, target_type=target_type)
        # Re-detect layout after navigation
        time.sleep(0.5)
        locator.refresh()

    return capture_screenshots(
        locator,
        max_pages=max_pages,
        output_dir=output_dir,
        scroll_delay=scroll_delay,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Capture WeChat chat screenshots")
    parser.add_argument("--pages", type=int, default=30, help="Max pages to capture")
    parser.add_argument("--output", default="screenshots", help="Output directory")
    parser.add_argument("--delay", type=float, default=0.8, help="Scroll delay (seconds)")
    parser.add_argument("--group", default=None, help="Chat name to navigate to")
    parser.add_argument(
        "--target-type", default="group", choices=["group", "contact", "any"],
        help="Type of chat to find: group (default), contact, or any",
    )
    args = parser.parse_args()

    files = capture_chat(
        max_pages=args.pages,
        output_dir=args.output,
        scroll_delay=args.delay,
        group_name=args.group,
        target_type=args.target_type,
    )
    print("\nScreenshots saved:")
    for f in files:
        print(f"  {f}")

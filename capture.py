"""
capture.py — Screenshot capture for macOS WeChat

Uses AppleScript to control the WeChat window:
1. Activate WeChat and get window geometry
2. Loop: screenshot the chat area → Page Up to scroll
3. Detect duplicate screenshots (reached top) → stop
4. Save screenshots in reverse order (oldest first)
"""

import os
import subprocess
import time
import hashlib
from pathlib import Path
from datetime import datetime

from PIL import Image


def run_applescript(script: str) -> str:
    """Run an AppleScript and return its output."""
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0:
        raise RuntimeError(f"AppleScript error: {result.stderr.strip()}")
    return result.stdout.strip()


def get_wechat_window_info() -> dict:
    """Get WeChat window position and size."""
    script = '''
    tell application "System Events"
        tell process "WeChat"
            set frontmost to true
            delay 0.3
            tell window 1
                set p to position
                set s to size
                return (item 1 of p as text) & "|" & (item 2 of p as text) & "|" & (item 1 of s as text) & "|" & (item 2 of s as text)
            end tell
        end tell
    end tell
    '''
    result = run_applescript(script)
    x, y, w, h = [int(v.strip()) for v in result.split("|")]
    return {"x": x, "y": y, "width": w, "height": h}


def _find_edge(pixels, coord_range, get_pair, threshold=50):
    """
    Scan pixel pairs along a coordinate range, return position of first edge
    where color difference exceeds threshold.
    get_pair(i) should return ((r1,g1,b1), (r2,g2,b2)) for the pair at position i.
    """
    for i in coord_range:
        c1, c2 = get_pair(i)
        diff = abs(c1[0]-c2[0]) + abs(c1[1]-c2[1]) + abs(c1[2]-c2[2])
        if diff >= threshold:
            return i
    return None


def detect_layout(window: dict) -> dict:
    """
    Screenshot the full WeChat window and detect UI regions by pixel scanning.
    Works in both light and dark mode by finding the largest color jumps
    rather than assuming specific color directions.

    Returns a dict with keys:
        icon_col_right, sidebar_right, titlebar_bottom, inputbox_top,
        sidebar, chat_area, search_box_center, sidebar_center_x, scale
    """
    path = "/tmp/_layout_detect.png"
    take_screenshot(window, path)
    img = Image.open(path)
    iw, ih = img.size
    px = img.load()

    ww, wh = window["width"], window["height"]
    scale = iw / ww  # typically 2.0 on Retina

    def _color_diff(c1, c2):
        return abs(c1[0]-c2[0]) + abs(c1[1]-c2[1]) + abs(c1[2]-c2[2])

    # --- Find vertical sidebar divider using multi-row voting ---
    # Scan multiple horizontal lines, find biggest edge in each, vote on x position.
    scan_x_start = int(iw * 0.05)
    scan_x_end = int(iw * 0.50)
    sample_ys = [int(ih * p) for p in [0.3, 0.4, 0.5, 0.6, 0.7]]

    edge_votes = {}  # x_px -> count
    for sy in sample_ys:
        best_x, best_diff = None, 0
        for x in range(scan_x_start, scan_x_end):
            d = _color_diff(px[x, sy][:3], px[x+1, sy][:3])
            if d > best_diff:
                best_diff = d
                best_x = x
        if best_x and best_diff > 30:
            # Group nearby x positions (within 4px) into same vote
            placed = False
            for vx in list(edge_votes.keys()):
                if abs(vx - best_x) <= 4:
                    edge_votes[vx] += 1
                    placed = True
                    break
            if not placed:
                edge_votes[best_x] = 1

    sidebar_right_px = None
    if edge_votes:
        # Pick the x with most votes (most consistent across rows)
        sidebar_right_px = max(edge_votes, key=edge_votes.get)
        votes = edge_votes[sidebar_right_px]
        if votes < 2:
            sidebar_right_px = None  # not confident enough

    if sidebar_right_px is None:
        sidebar_right_px = int(iw * 0.25)
        print(f"  ⚠️  Could not detect sidebar divider, using fallback: {sidebar_right_px/scale:.0f}pt")

    sidebar_right_pt = int(sidebar_right_px / scale)

    # --- Find icon column right edge ---
    # First significant edge from left side at y=50%
    mid_y = ih // 2
    icon_col_right_px = None
    for x in range(20, min(sidebar_right_px, int(iw * 0.15))):
        d = _color_diff(px[x, mid_y][:3], px[x+1, mid_y][:3])
        if d > 15:
            icon_col_right_px = x + 1
            break
    if icon_col_right_px is None:
        icon_col_right_px = int(40 * scale)
    icon_col_right_pt = int(icon_col_right_px / scale)

    # --- Find title bar bottom (horizontal divider) ---
    # In the sidebar column, find the largest color jump in top 15% of window.
    # This is the line between the search bar area and the chat list.
    sidebar_col_x = (icon_col_right_px + sidebar_right_px) // 2
    titlebar_bottom_px = None
    best_diff = 0
    for y in range(10, int(ih * 0.15)):
        d = _color_diff(px[sidebar_col_x, y][:3], px[sidebar_col_x, y+1][:3])
        if d > best_diff:
            best_diff = d
            titlebar_bottom_px = y + 1

    if titlebar_bottom_px is None or best_diff < 20:
        titlebar_bottom_px = int(60 * scale)
        print(f"  ⚠️  Could not detect title bar, using fallback: {titlebar_bottom_px/scale:.0f}pt")
    titlebar_bottom_pt = int(titlebar_bottom_px / scale)

    # --- Find input box top edge ---
    # In the chat area column, find the largest color jump in bottom 40%.
    chat_col_x = sidebar_right_px + (iw - sidebar_right_px) // 2
    inputbox_top_px = None
    best_diff = 0
    for y in range(ih - 10, int(ih * 0.6), -1):
        d = _color_diff(px[chat_col_x, y][:3], px[chat_col_x, y-1][:3])
        if d > best_diff:
            best_diff = d
            inputbox_top_px = y

    if inputbox_top_px is None or best_diff < 20:
        inputbox_top_px = int((wh - 160) * scale)
        print(f"  ⚠️  Could not detect input box, using fallback: {inputbox_top_px/scale:.0f}pt")
    inputbox_top_pt = int(inputbox_top_px / scale)

    # --- Build layout dict ---
    wx, wy = window["x"], window["y"]

    sidebar = {
        "x": wx + icon_col_right_pt,
        "y": wy + titlebar_bottom_pt,
        "width": sidebar_right_pt - icon_col_right_pt,
        "height": wh - titlebar_bottom_pt,
    }

    chat_area = {
        "x": wx + sidebar_right_pt,
        "y": wy + titlebar_bottom_pt,
        "width": ww - sidebar_right_pt,
        "height": inputbox_top_pt - titlebar_bottom_pt,
    }

    # Search box: centered in sidebar, near top
    search_box_center = (
        wx + icon_col_right_pt + (sidebar_right_pt - icon_col_right_pt) // 2,
        wy + titlebar_bottom_pt // 2,
    )

    sidebar_center_x = wx + icon_col_right_pt + (sidebar_right_pt - icon_col_right_pt) // 2

    layout = {
        "icon_col_right": icon_col_right_pt,
        "sidebar_right": sidebar_right_pt,
        "titlebar_bottom": titlebar_bottom_pt,
        "inputbox_top": inputbox_top_pt,
        "sidebar": sidebar,
        "chat_area": chat_area,
        "search_box_center": search_box_center,
        "sidebar_center_x": sidebar_center_x,
        "scale": scale,
    }

    print(f"  Layout detected: sidebar={sidebar_right_pt}pt, titlebar={titlebar_bottom_pt}pt, inputbox={inputbox_top_pt}pt, icon_col={icon_col_right_pt}pt, scale={scale:.1f}x")
    return layout


def get_chat_area(window: dict, layout: dict | None = None) -> dict:
    """
    Return the chat message area region.
    If layout is provided, use detected coordinates.
    Otherwise fall back to hardcoded estimates.
    """
    if layout:
        return layout["chat_area"]

    # Fallback: hardcoded estimates
    sidebar_width = int(window["width"] * 0.25)
    top_bar = 60
    bottom_bar = 160

    return {
        "x": window["x"] + sidebar_width,
        "y": window["y"] + top_bar,
        "width": window["width"] - sidebar_width,
        "height": window["height"] - top_bar - bottom_bar,
    }


def take_screenshot(region: dict, filepath: str) -> str:
    """Take a screenshot of a specific screen region."""
    x, y, w, h = region["x"], region["y"], region["width"], region["height"]
    subprocess.run(
        ["screencapture", "-R", f"{x},{y},{w},{h}", "-x", filepath],
        check=True, timeout=10
    )
    return filepath

def click_chat_area(chat_area: dict):
    """Click in the center of the chat area to ensure it has focus."""
    import Quartz
    center_x = chat_area["x"] + chat_area["width"] // 2
    center_y = chat_area["y"] + chat_area["height"] // 2

    point = Quartz.CGPointMake(center_x, center_y)
    click_down = Quartz.CGEventCreateMouseEvent(
        None, Quartz.kCGEventLeftMouseDown, point, Quartz.kCGMouseButtonLeft
    )
    click_up = Quartz.CGEventCreateMouseEvent(
        None, Quartz.kCGEventLeftMouseUp, point, Quartz.kCGMouseButtonLeft
    )
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, click_down)
    time.sleep(0.05)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, click_up)


def navigate_to_chat(group_name: str, window: dict, layout: dict | None = None) -> None:
    """
    Navigate to a specific group chat by OCR-scanning the sidebar.

    Strategy:
    1. Screenshot sidebar and OCR to find the target group
    2. If not visible, search for the group name first, then Escape + re-scan
    3. Click the detected position

    Requires: cliclick (brew install cliclick)
    """
    wx, wy = window["x"], window["y"]

    # Use detected layout or fall back to hardcoded estimates
    if layout:
        sidebar_region = layout["sidebar"]
        search_x, search_y = layout["search_box_center"]
        click_x = layout["sidebar_center_x"]
    else:
        sidebar_region = {
            "x": wx + 72,
            "y": wy + 60,
            "width": 220,
            "height": min(window.get("height", 890) - 60, 800),
        }
        search_x = wx + 121
        search_y = wy + 24
        click_x = wx + 180

    def _find_group_in_sidebar() -> int | None:
        """Screenshot sidebar, OCR it, return screen y of matched group or None."""
        path = "/tmp/_sidebar_scan.png"
        take_screenshot(sidebar_region, path)
        return _ocr_find_text_y(path, sidebar_region, group_name)

    # First try: check if the group is already visible in sidebar
    target_y = _find_group_in_sidebar()

    if target_y is None:
        # Group not visible — search for it to bring it into sidebar
        print(f"  Group '{group_name}' not visible in sidebar, searching...")
        subprocess.run(["cliclick", f"c:{search_x},{search_y}"], check=True, timeout=5)
        time.sleep(0.5)

        subprocess.run(["pbcopy"], input=group_name.encode("utf-8"), check=True, timeout=5)
        run_applescript('''
        tell application "System Events"
            tell process "WeChat"
                keystroke "v" using command down
            end tell
        end tell
        ''')
        time.sleep(1.5)

        # Press Escape to close dropdown — searched group may now appear in sidebar
        run_applescript('''
        tell application "System Events"
            key code 53
        end tell
        ''')
        time.sleep(0.5)

        target_y = _find_group_in_sidebar()

    if target_y is not None:
        click_y = target_y
        print(f"  Found '{group_name}' at y={click_y}, clicking ({click_x}, {click_y})")
        # Ensure WeChat is frontmost before clicking
        run_applescript('tell application "WeChat" to activate')
        time.sleep(0.3)
        subprocess.run(["cliclick", f"c:{click_x},{click_y}"], check=True, timeout=5)
    else:
        # Last resort fallback: click first chat in sidebar
        print(f"  ⚠️  Could not find '{group_name}' in sidebar via OCR, trying first chat")
        fallback_y = sidebar_region["y"] + 35
        subprocess.run(["cliclick", f"c:{click_x},{fallback_y}"], check=True, timeout=5)

    time.sleep(1.0)
    print(f"Navigated to: {group_name}")


def _ocr_find_text_y(image_path: str, region: dict, target: str) -> int | None:
    """
    OCR an image and return the screen y-coordinate of text matching target.
    Collects all matching candidates and prefers exact matches over fuzzy ones.
    Handles WeChat sidebar truncation (e.g. "golang runtim..").
    """
    import objc
    from Foundation import NSURL
    import Vision

    image_url = NSURL.fileURLWithPath_(image_path)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setRecognitionLanguages_(["zh-Hans", "en"])

    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(image_url, {})
    handler.performRequests_error_([request], None)

    results = request.results() or []
    img = Image.open(image_path)
    img_h = img.height

    target_lower = target.lower()
    target_clean = target_lower.rstrip('.…。').strip()

    # screencapture -R uses point coordinates but produces pixel-resolution images
    # On Retina displays, img_h (pixels) = region["height"] (points) * scale_factor
    scale_factor = img_h / region["height"] if region["height"] > 0 else 1

    def _get_screen_y(obs):
        bbox = obs.boundingBox()
        center_y_px = (1.0 - bbox.origin.y - bbox.size.height / 2) * img_h
        # Convert pixel coordinate back to screen points
        center_y_pts = center_y_px / scale_factor
        return region["y"] + int(center_y_pts)

    # Collect all candidates with priority: 0=exact, 1=full-substring, 2=prefix, 3=partial
    candidates = []  # list of (priority, screen_y, text)

    for obs in results:
        text = obs.topCandidates_(1)[0].string()
        text_lower = text.lower()
        text_clean = text_lower.rstrip('.…。').strip()

        # Priority 0: exact match (after stripping dots)
        if text_clean == target_clean:
            candidates.append((0, _get_screen_y(obs), text))
            continue

        # Priority 1: full substring match (target in text or text in target)
        if target_lower in text_lower or text_lower in target_lower:
            candidates.append((1, _get_screen_y(obs), text))
            continue

        # Priority 2: prefix match (for truncated names)
        if len(text_clean) >= 5 and (
            target_clean.startswith(text_clean) or text_clean.startswith(target_clean)
        ):
            candidates.append((2, _get_screen_y(obs), text))
            continue

        # Priority 3: word-level partial match
        target_parts = [p for p in target.split() if len(p) > 2]
        if target_parts and target_parts[0].lower() in text_lower:
            if len(target_parts) >= 2:
                second = target_parts[1].lower()
                if second[:4] in text_lower:
                    candidates.append((3, _get_screen_y(obs), text))

    if not candidates:
        return None

    # Sort by priority (lower = better), return best match
    candidates.sort(key=lambda c: c[0])
    best = candidates[0]
    print(f"  OCR matches: {[(p, t) for p, _, t in candidates]}, selected: priority={best[0]} text='{best[2]}'")
    return best[1]


def send_page_up():
    """Send Page Up key to WeChat via AppleScript to scroll up."""
    run_applescript('''
    tell application "System Events"
        tell process "WeChat"
            key code 116
        end tell
    end tell
    ''')


def image_hash(filepath: str) -> str:
    """Compute a perceptual hash to detect duplicate screenshots."""
    img = Image.open(filepath)
    # Resize to small size and convert to grayscale for comparison
    img = img.resize((64, 64)).convert("L")
    return hashlib.md5(img.tobytes()).hexdigest()


def capture_chat(
    max_pages: int = 30,
    output_dir: str = "screenshots",
    scroll_delay: float = 0.8,
    group_name: str | None = None,
) -> list[str]:
    """
    Capture WeChat chat screenshots by scrolling up page by page.

    Args:
        max_pages: Maximum number of pages to capture
        output_dir: Directory to save screenshots
        scroll_delay: Seconds to wait after each scroll for rendering
        group_name: If provided, navigate to this group chat first

    Returns:
        List of screenshot file paths in chronological order (oldest first)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Activate WeChat and get window geometry
    window = get_wechat_window_info()

    # Detect window layout by pixel scanning
    layout = detect_layout(window)

    # Navigate to target chat if specified
    if group_name:
        navigate_to_chat(group_name, window, layout)

    chat_area = get_chat_area(window, layout)
    print(f"Window: {window}")
    print(f"Chat area: {chat_area}")

    # Click in chat area to ensure it has keyboard focus for Page Up
    click_chat_area(chat_area)
    time.sleep(0.5)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshots = []
    prev_hash = None

    # First, take a screenshot of the current (most recent) view
    filepath = str(out / f"{timestamp}_page_000.png")
    take_screenshot(chat_area, filepath)
    screenshots.append(filepath)
    prev_hash = image_hash(filepath)
    print(f"Captured page 0 (current view)")

    # Then scroll up to capture older messages
    for i in range(1, max_pages):
        send_page_up()
        time.sleep(scroll_delay)

        filepath = str(out / f"{timestamp}_page_{i:03d}.png")
        take_screenshot(chat_area, filepath)

        # Check if we've reached the top (duplicate screenshot)
        curr_hash = image_hash(filepath)
        if curr_hash == prev_hash:
            os.remove(filepath)
            print(f"Reached top of chat at page {i}. Stopping.")
            break

        screenshots.append(filepath)
        prev_hash = curr_hash
        print(f"Captured page {i}")

    # Reverse so screenshots are in chronological order (oldest first)
    screenshots.reverse()
    print(f"\nTotal pages captured: {len(screenshots)}")
    return screenshots


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Capture WeChat chat screenshots")
    parser.add_argument("--pages", type=int, default=30, help="Max pages to capture")
    parser.add_argument("--output", default="screenshots", help="Output directory")
    parser.add_argument("--delay", type=float, default=0.8, help="Scroll delay (seconds)")
    parser.add_argument("--group", default=None, help="Group chat name to navigate to")
    args = parser.parse_args()

    files = capture_chat(
        max_pages=args.pages,
        output_dir=args.output,
        scroll_delay=args.delay,
        group_name=args.group,
    )
    print("\nScreenshots saved:")
    for f in files:
        print(f"  {f}")

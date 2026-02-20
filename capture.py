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


from typing import Optional, Dict

def get_chat_area(window: Dict, layout: Optional[Dict] = None) -> Dict:
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


def _ocr_normalize(text: str) -> str:
    """Normalize text to handle common OCR confusions (I/l/1, O/0, etc.)."""
    # Apply lowercase first, then normalize confusable characters
    t = text.lower()
    # l (lowercase L) and 1 → i (canonical form for I/l/1 confusion)
    t = t.replace("l", "i").replace("1", "i")
    # 0 → o (for O/0 confusion)
    t = t.replace("0", "o")
    return t


def _ocr_all_text(image_path: str) -> list[str]:
    """OCR an image and return all recognized text strings."""
    import Vision
    from Foundation import NSURL

    image_url = NSURL.fileURLWithPath_(image_path)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setRecognitionLanguages_(["zh-Hans", "en"])

    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(image_url, {})
    handler.performRequests_error_([request], None)

    texts = []
    for obs in (request.results() or []):
        top = obs.topCandidates_(1)
        if top:
            texts.append(top[0].string())
    return texts


def _verify_search_results(group_name: str, sidebar_region: dict) -> bool:
    """
    Screenshot the search results area and OCR to verify
    the search results contain the target group name.

    Uses an expanded region to cover WeChat's search results panel,
    which may extend beyond the normal sidebar area.
    """
    # Search results panel may be wider/taller than the sidebar — expand region
    search_region = {
        "x": sidebar_region["x"],
        "y": sidebar_region["y"],
        "width": sidebar_region["width"] + 100,  # search results can be wider
        "height": sidebar_region["height"],
    }
    path = "/tmp/_search_verify.png"
    take_screenshot(search_region, path)
    texts = _ocr_all_text(path)
    all_text = " ".join(texts).lower()
    # Also create a version with all whitespace stripped for fuzzy matching
    all_text_compact = all_text.replace(" ", "")
    # OCR-normalized versions (handles I/l/1 and O/0 confusion)
    all_text_norm = _ocr_normalize(" ".join(texts))
    all_text_compact_norm = all_text_norm.replace(" ", "")

    print(f"  Search OCR texts: {texts[:10]}")  # debug: show first 10 OCR blocks

    target_lower = group_name.lower()
    target_compact = target_lower.replace(" ", "")
    target_norm = _ocr_normalize(group_name)
    target_compact_norm = target_norm.replace(" ", "")

    # Match 1: exact substring (with spaces)
    if target_lower in all_text:
        return True

    # Match 2: compact match (ignore spaces — handles OCR splitting "AI coding" etc.)
    if target_compact in all_text_compact:
        return True

    # Match 3: OCR-normalized match (handles I/l, O/0 confusion)
    if target_norm in all_text_norm or target_compact_norm in all_text_compact_norm:
        return True

    # Match 4: key parts match (at least half of the parts found)
    target_parts = [p for p in target_lower.split() if len(p) >= 1]
    if target_parts:
        matched = sum(1 for part in target_parts if part in all_text or _ocr_normalize(part) in all_text_norm)
        if matched >= max(1, len(target_parts) * 0.5):
            return True

    return False


def _verify_chat_title(group_name: str, window: Dict, layout: Optional[Dict]) -> bool:
    """
    Screenshot the chat title bar area and OCR to verify
    we entered the correct group chat.
    Re-acquires window info to handle layout changes after search.
    """
    # Re-acquire window info — layout may have changed after search/navigation
    try:
        window = get_wechat_window_info()
    except Exception:  # noqa: BLE001
        pass  # Fall back to the passed-in window

    wx, wy = window["x"], window["y"]
    ww = window["width"]

    # Chat title is in the top ~50pt of the area right of the sidebar.
    # Use a conservative sidebar estimate since we may not have accurate layout.
    sidebar_right = layout["sidebar_right"] if layout else int(ww * 0.25)

    title_region = {
        "x": wx + sidebar_right,
        "y": wy,
        "width": ww - sidebar_right,
        "height": 50,
    }

    path = "/tmp/_title_verify.png"
    take_screenshot(title_region, path)
    texts = _ocr_all_text(path)
    all_text = " ".join(texts).lower()
    print(f"  Title bar OCR: {texts}")

    target_lower = group_name.lower()
    target_parts = [p for p in target_lower.split() if len(p) > 1]

    if target_lower in all_text:
        return True
    # Partial match: check key parts (handles truncated titles)
    matched = sum(1 for part in target_parts if part in all_text)
    if target_parts and matched >= len(target_parts) * 0.5:
        return True
    return False


def navigate_to_chat(group_name: str, window: Dict, layout: Optional[Dict] = None) -> None:
    """
    Navigate to a specific group chat with screenshot-based verification.

    Strategy:
    1. Screenshot sidebar and OCR to find the target group
    2. If not visible, Cmd+F to search, paste group name
    3. Verify search results contain the target group (screenshot + OCR)
    4. Press Return to enter the chat
    5. Verify title bar shows correct group name (screenshot + OCR)

    Raises RuntimeError if navigation cannot be verified.
    """
    wx, wy = window["x"], window["y"]

    if layout:
        sidebar_region = layout["sidebar"]
        click_x = layout["sidebar_center_x"]
    else:
        sidebar_region = {
            "x": wx + 72,
            "y": wy + 60,
            "width": 220,
            "height": min(window.get("height", 890) - 60, 800),
        }
        click_x = wx + 180

    def _find_group_in_sidebar() -> Optional[int]:
        """Screenshot sidebar, OCR it, return screen y of matched group or None."""
        path = "/tmp/_sidebar_scan.png"
        take_screenshot(sidebar_region, path)
        return _ocr_find_text_y(path, sidebar_region, group_name)

    def _search_and_enter() -> None:
        """Use Cmd+F search to find and enter the target group chat."""
        print(f"  Searching for '{group_name}' via Cmd+F...")

        # Cmd+F to activate search
        run_applescript('''
        tell application "System Events"
            tell process "WeChat"
                keystroke "f" using command down
            end tell
        end tell
        ''')
        time.sleep(0.5)

        # Paste group name
        subprocess.run(["pbcopy"], input=group_name.encode("utf-8"), check=True, timeout=5)
        run_applescript('''
        tell application "System Events"
            tell process "WeChat"
                keystroke "v" using command down
            end tell
        end tell
        ''')
        time.sleep(1.5)

        # --- Verification Step 1: search results contain target group ---
        if _verify_search_results(group_name, sidebar_region):
            print(f"  ✅ Search verification passed: found '{group_name}' in search results")
        else:
            print(f"  ⚠️  Search verification: '{group_name}' not found in results, retrying...")
            # Retry: clear and re-paste
            run_applescript('''
            tell application "System Events"
                tell process "WeChat"
                    keystroke "a" using command down
                    key code 51
                end tell
            end tell
            ''')
            time.sleep(0.3)
            run_applescript('''
            tell application "System Events"
                tell process "WeChat"
                    keystroke "v" using command down
                end tell
            end tell
            ''')
            time.sleep(2.0)
            if _verify_search_results(group_name, sidebar_region):
                print(f"  ✅ Search verification passed on retry")
            else:
                raise RuntimeError(
                    f"Navigation failed: could not find '{group_name}' in search results. "
                    "Please ensure the group chat name is correct and you are a member."
                )

        # Click on the search result instead of pressing Return —
        # Return doesn't reliably select group chat results in WeChat search.
        search_region = {
            "x": sidebar_region["x"],
            "y": sidebar_region["y"],
            "width": sidebar_region["width"] + 100,
            "height": sidebar_region["height"],
        }
        search_screenshot = "/tmp/_search_verify.png"
        take_screenshot(search_region, search_screenshot)
        result_y = _ocr_find_text_y(search_screenshot, search_region, group_name)

        if result_y is not None:
            result_x = search_region["x"] + search_region["width"] // 2
            print(f"  Clicking search result at ({result_x}, {result_y})")
            run_applescript('tell application "WeChat" to activate')
            time.sleep(0.2)
            subprocess.run(["cliclick", f"c:{result_x},{result_y}"], check=True, timeout=5)
        else:
            # Fallback: press Return (may not always work)
            print(f"  ⚠️  Could not locate search result position, pressing Return as fallback")
            run_applescript('''
            tell application "System Events"
                tell process "WeChat"
                    key code 36
                end tell
            end tell
            ''')
        time.sleep(1.0)

    # --- Strategy 1: check if the group is already visible in sidebar ---
    target_y = _find_group_in_sidebar()

    if target_y is not None:
        print(f"  Found '{group_name}' at y={target_y}, clicking ({click_x}, {target_y})")
        run_applescript('tell application "WeChat" to activate')
        time.sleep(0.3)
        subprocess.run(["cliclick", f"c:{click_x},{target_y}"], check=True, timeout=5)
        time.sleep(1.0)

        # Verify we landed in the right chat
        if _verify_chat_title(group_name, window, layout):
            print(f"  ✅ Chat verification passed: title bar matches '{group_name}'")
            print(f"Navigated to: {group_name}")
            return

        # Sidebar click landed on wrong chat — fallback to search
        print(f"  ⚠️  Sidebar click landed on wrong chat, falling back to search...")

    else:
        print(f"  Group '{group_name}' not visible in sidebar, searching...")

    # --- Strategy 2: search via Cmd+F ---
    _search_and_enter()

    # --- Verification Step 2: title bar shows correct group name ---
    if _verify_chat_title(group_name, window, layout):
        print(f"  ✅ Chat verification passed: title bar matches '{group_name}'")
    else:
        raise RuntimeError(
            f"Navigation failed: title bar does not match '{group_name}'. "
            "WeChat may have navigated to the wrong chat."
        )

    print(f"Navigated to: {group_name}")


def _ocr_find_text_y(image_path: str, region: Dict, target: str) -> Optional[int]:
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
    target_norm = _ocr_normalize(target)
    target_clean_norm = target_norm.rstrip('.…。').strip()
    # Space-stripped versions for handling OCR inserting/missing spaces
    target_compact = target_lower.replace(" ", "")
    target_compact_norm = target_norm.replace(" ", "")

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
        top_candidates = obs.topCandidates_(1)
        if not top_candidates:
            continue
        text = top_candidates[0].string()
        text_lower = text.lower()
        text_clean = text_lower.rstrip('.…。').strip()
        text_norm = _ocr_normalize(text)
        text_clean_norm = text_norm.rstrip('.…。').strip()
        # Space-stripped versions
        text_compact = text_lower.replace(" ", "")
        text_compact_norm = text_norm.replace(" ", "")

        # Priority 0: exact match (after stripping dots), including compact/normalized
        if (text_clean == target_clean or text_clean_norm == target_clean_norm or
                text_compact == target_compact or text_compact_norm == target_compact_norm):
            candidates.append((0, _get_screen_y(obs), text))
            continue

        # Priority 1: full substring match (target in text or text in target)
        # Check with original, normalized, and compact variants
        if (target_lower in text_lower or text_lower in target_lower or
                target_norm in text_norm or text_norm in target_norm or
                target_compact in text_compact or text_compact in target_compact or
                target_compact_norm in text_compact_norm or text_compact_norm in target_compact_norm):
            candidates.append((1, _get_screen_y(obs), text))
            continue

        # Priority 2: prefix match (for truncated names)
        if len(text_clean) >= 5 and (
            target_clean.startswith(text_clean) or text_clean.startswith(target_clean) or
            target_clean_norm.startswith(text_clean_norm) or text_clean_norm.startswith(target_clean_norm) or
            target_compact.startswith(text_compact) or text_compact.startswith(target_compact)
        ):
            candidates.append((2, _get_screen_y(obs), text))
            continue

        # Priority 3: word-level partial match
        target_parts = [p for p in target.split() if len(p) > 2]
        if target_parts and (target_parts[0].lower() in text_lower or _ocr_normalize(target_parts[0]) in text_norm):
            if len(target_parts) >= 2:
                second = target_parts[1].lower()
                if second[:4] in text_lower or _ocr_normalize(second[:4]) in text_norm:
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
    group_name: Optional[str] = None,
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
        # Re-detect layout after navigation — when no chat was previously
        # selected, the initial detect_layout may have used incorrect
        # fallback values (the right side is blank, so sidebar divider
        # detection fails). After clicking into a chat, the layout becomes
        # detectable.
        time.sleep(0.5)
        window = get_wechat_window_info()
        layout = detect_layout(window)

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

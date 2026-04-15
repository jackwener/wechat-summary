"""
actions.py — Pure action functions for WeChat automation.

Each function performs one atomic action (click, type, scroll, screenshot).
No locating logic here — positions are always passed in.
"""

import subprocess
import time

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


def activate_wechat() -> None:
    """Bring WeChat to front."""
    run_applescript('tell application "WeChat" to activate')
    time.sleep(0.2)


def get_window_info() -> dict:
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


def click(x: int, y: int) -> None:
    """Click at screen coordinates using CGEvent."""
    import Quartz

    point = Quartz.CGPointMake(x, y)
    down = Quartz.CGEventCreateMouseEvent(
        None, Quartz.kCGEventLeftMouseDown, point, Quartz.kCGMouseButtonLeft
    )
    up = Quartz.CGEventCreateMouseEvent(
        None, Quartz.kCGEventLeftMouseUp, point, Quartz.kCGMouseButtonLeft
    )
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, down)
    time.sleep(0.05)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, up)


def screenshot(region: dict, filepath: str) -> str:
    """Take a screenshot of a specific screen region. Returns filepath."""
    x, y, w, h = region["x"], region["y"], region["width"], region["height"]
    subprocess.run(
        ["screencapture", "-R", f"{x},{y},{w},{h}", "-x", filepath],
        check=True, timeout=10
    )
    return filepath


def cmd_f() -> None:
    """Open WeChat search (Cmd+F)."""
    run_applescript('''
    tell application "System Events"
        tell process "WeChat"
            keystroke "f" using command down
        end tell
    end tell
    ''')
    time.sleep(0.5)


def type_text(text: str) -> None:
    """Type text by pasting from clipboard (pbcopy + Cmd+V)."""
    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True, timeout=5)
    run_applescript('''
    tell application "System Events"
        tell process "WeChat"
            keystroke "v" using command down
        end tell
    end tell
    ''')


def select_all_and_delete() -> None:
    """Select all text and delete (Cmd+A, Backspace)."""
    run_applescript('''
    tell application "System Events"
        tell process "WeChat"
            keystroke "a" using command down
            key code 51
        end tell
    end tell
    ''')


def page_up() -> None:
    """Send Page Up key to WeChat."""
    run_applescript('''
    tell application "System Events"
        tell process "WeChat"
            key code 116
        end tell
    end tell
    ''')


def image_hash(filepath: str) -> str:
    """Compute a perceptual hash to detect duplicate screenshots."""
    import hashlib
    img = Image.open(filepath)
    img = img.resize((64, 64)).convert("L")
    return hashlib.md5(img.tobytes()).hexdigest()

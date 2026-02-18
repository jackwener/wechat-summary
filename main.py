#!/usr/bin/env python3
"""
main.py — WeChat Group Chat Summary Tool

Orchestrates the full pipeline:
  1. Capture screenshots from macOS WeChat (AppleScript scroll + screencapture)
  2. OCR screenshots using macOS Vision framework
  3. Summarize chat using Claude API

Usage:
  # Full pipeline: capture → OCR → summarize
  python main.py --pages 20

  # Capture only
  python main.py --capture-only --pages 20

  # OCR + summarize from existing screenshots
  python main.py --from-screenshots ./screenshots/

  # Summarize from existing text file
  python main.py --from-text ./chat.txt
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from capture import capture_chat
from ocr import ocr_screenshots
from summarize import summarize_chat, save_summary


def _check_command_exists(command: str, errors: list[str]) -> None:
    """Check if a shell command is available."""
    if shutil.which(command):
        print(f"  ✅ Command available: {command}")
    else:
        errors.append(f"Missing required command: {command}")


def _check_wechat_permissions(errors: list[str]) -> dict | None:
    """Check WeChat automation/accessibility permission via AppleScript."""
    script = """
    tell application "System Events"
        if not (exists process "WeChat") then
            return "WECHAT_NOT_RUNNING"
        end if
        tell process "WeChat"
            set frontmost to true
            delay 0.2
            tell window 1
                set p to position
                set s to size
                return (item 1 of p as text) & "|" & (item 2 of p as text) & "|" & (item 1 of s as text) & "|" & (item 2 of s as text)
            end tell
        end tell
    end tell
    """

    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        errors.append(
            "Cannot control WeChat via System Events. "
            "Grant Accessibility and Automation permissions to your terminal/Codex app. "
            f"AppleScript error: {stderr or 'unknown error'}"
        )
        return None
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Failed to run AppleScript preflight check: {exc}")
        return None

    output = result.stdout.strip()
    if output == "WECHAT_NOT_RUNNING":
        errors.append("WeChat is not running. Open WeChat and keep a chat window available.")
        return None

    try:
        x, y, w, h = [int(v.strip()) for v in output.split("|")]
    except Exception:  # noqa: BLE001
        errors.append(f"Unexpected AppleScript output during preflight: {output!r}")
        return None

    print("  ✅ Accessibility/Automation: can access WeChat window")
    return {"x": x, "y": y, "width": w, "height": h}


def _check_screen_recording(window: dict, errors: list[str]) -> None:
    """Check screen capture permission by taking a tiny probe screenshot."""
    probe = Path("/tmp/_wechat_screen_permission_probe.png")
    region = f"{window['x']},{window['y']},8,8"
    try:
        subprocess.run(
            ["screencapture", "-R", region, "-x", str(probe)],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        errors.append(
            "Screen recording check failed. Grant Screen Recording permission to your terminal/Codex app. "
            f"screencapture error: {stderr or 'unknown error'}"
        )
        return
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Failed to run screen recording preflight check: {exc}")
        return

    if not probe.exists() or probe.stat().st_size == 0:
        errors.append("Screen recording probe produced an empty screenshot.")
        return

    print("  ✅ Screen Recording: screenshot probe succeeded")
    probe.unlink(missing_ok=True)


def run_preflight_checks(
    require_capture: bool,
    require_navigation: bool,
    require_summarization: bool,
) -> None:
    """Run startup checks for commands, permissions, and required environment."""
    print("\n🔧 Preflight checks...")
    errors: list[str] = []

    if require_capture and sys.platform != "darwin":
        errors.append("Capture pipeline requires macOS (AppleScript + Vision + screencapture).")

    if require_capture:
        for command in ["osascript", "screencapture"]:
            _check_command_exists(command, errors)
        if require_navigation:
            for command in ["cliclick", "pbcopy"]:
                _check_command_exists(command, errors)

        window = _check_wechat_permissions(errors)
        if window:
            _check_screen_recording(window, errors)

    if require_summarization:
        has_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
        if has_key:
            print("  ✅ Anthropic API key detected")
        else:
            errors.append("Missing ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY.")

    if errors:
        print("\n❌ Preflight failed:")
        for idx, err in enumerate(errors, start=1):
            print(f"  {idx}. {err}")
        sys.exit(2)

    print("✅ Preflight passed")


def main():
    parser = argparse.ArgumentParser(
        description="WeChat Group Chat Summary Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --pages 20                    # Full pipeline
  python main.py --capture-only --pages 5      # Capture only
  python main.py --from-screenshots ./screenshots/  # Skip capture
  python main.py --from-text chat.txt          # Skip capture + OCR
        """,
    )

    # Pipeline control
    parser.add_argument(
        "--capture-only", action="store_true",
        help="Only capture screenshots, skip OCR and summarization"
    )
    parser.add_argument(
        "--from-screenshots", type=str, default=None,
        help="Skip capture, run OCR + summarization on existing screenshots"
    )
    parser.add_argument(
        "--from-text", type=str, default=None,
        help="Skip capture and OCR, summarize existing text file"
    )

    # Capture options
    parser.add_argument("--pages", type=int, default=30, help="Max pages to capture (default: 30)")
    parser.add_argument("--delay", type=float, default=0.8, help="Scroll delay in seconds (default: 0.8)")
    parser.add_argument("--screenshots-dir", default="screenshots", help="Screenshots directory")

    # Summarization options
    parser.add_argument("--group", default="群聊", help="Group chat name for output file")
    parser.add_argument("--output-dir", default="output", help="Summary output directory")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model")
    parser.add_argument("--save-text", default=None, help="Save OCR text to file")

    args = parser.parse_args()

    require_capture = not args.from_text and not args.from_screenshots
    require_navigation = require_capture and bool(args.group and args.group != "群聊")
    require_summarization = not args.capture_only
    run_preflight_checks(require_capture, require_navigation, require_summarization)

    print("=" * 60)
    print("  WeChat Group Chat Summary Tool")
    print("=" * 60)

    chat_text = None

    # Step 1: Capture (unless skipped)
    if args.from_text:
        print(f"\n📄 Loading text from {args.from_text}")
        chat_text = Path(args.from_text).read_text(encoding="utf-8")
        print(f"   Loaded {len(chat_text)} characters")

    elif args.from_screenshots:
        print(f"\n📸 Using existing screenshots from {args.from_screenshots}")
        screenshots_dir = args.from_screenshots
    else:
        print(f"\n📸 Step 1: Capturing screenshots (max {args.pages} pages)")
        if args.group and args.group != "群聊":
            print(f"   🔍 Auto-navigating to: {args.group}")
        else:
            print(f"   ⚠️  Make sure WeChat is open with the target group chat visible!")

        screenshots = capture_chat(
            max_pages=args.pages,
            output_dir=args.screenshots_dir,
            scroll_delay=args.delay,
            group_name=args.group if args.group != "群聊" else None,
        )

        if not screenshots:
            print("❌ No screenshots captured. Exiting.")
            sys.exit(1)

        screenshots_dir = args.screenshots_dir

        if args.capture_only:
            print(f"\n✅ Capture complete! Screenshots saved to {screenshots_dir}/")
            return

    # Step 2: OCR (unless text provided)
    if chat_text is None:
        print(f"\n🔍 Step 2: Running OCR on screenshots...")
        chat_text = ocr_screenshots(screenshots_dir)

        if not chat_text.strip():
            print("❌ OCR produced no text. Exiting.")
            sys.exit(1)

        # Optionally save OCR text
        if args.save_text:
            Path(args.save_text).write_text(chat_text, encoding="utf-8")
            print(f"   OCR text saved to {args.save_text}")

    # Step 3: Summarize
    print(f"\n🤖 Step 3: Generating summary with Claude ({args.model})...")
    summary = summarize_chat(chat_text, model=args.model)

    filepath = save_summary(summary, output_dir=args.output_dir, group_name=args.group)

    print(f"\n{'=' * 60}")
    print(f"  ✅ Done!")
    print(f"  Summary: {filepath}")
    print(f"{'=' * 60}")

    # Also print the summary
    print(f"\n{summary}")


if __name__ == "__main__":
    main()

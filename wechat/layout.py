"""
layout.py — WeChat window layout detection via pixel scanning.

Detects sidebar divider, title bar, input box, and icon column
by finding the largest color jumps in the screenshot.
Works in both light and dark mode.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

from PIL import Image

from wechat.actions import screenshot


def detect_layout(window: dict, retries: int = 2) -> dict:
    """
    Screenshot the full WeChat window and detect UI regions by pixel scanning.

    Returns a dict with keys:
        icon_col_right, sidebar_right, titlebar_bottom, inputbox_top,
        sidebar, chat_area, search_box_center, sidebar_center_x, scale

    Retries up to `retries` times (with a short sleep) to handle transient
    states where WeChat's window is still transitioning (e.g. search overlay).
    """
    import time as _time
    fd, path = tempfile.mkstemp(suffix=".png", prefix="_wechat_layout_")
    os.close(fd)
    try:
        layout = _detect_layout_from_file(window, path)
        # If fallback was used (sidebar looks wrong), retry once after a short pause
        ww = window["width"]
        for _ in range(retries):
            if layout["sidebar_right"] >= int(ww * 0.28):
                break
            _time.sleep(0.4)
            layout = _detect_layout_from_file(window, path)
        return layout
    finally:
        Path(path).unlink(missing_ok=True)


def _detect_layout_from_file(window: dict, path: str) -> dict:
    screenshot(window, path)
    img = Image.open(path)
    iw, ih = img.size
    px = img.load()

    ww, wh = window["width"], window["height"]
    scale = iw / ww  # typically 2.0 on Retina

    def _color_diff(c1, c2):
        return abs(c1[0]-c2[0]) + abs(c1[1]-c2[1]) + abs(c1[2]-c2[2])

    # --- Find vertical sidebar divider using multi-row voting ---
    scan_x_start = int(iw * 0.05)
    scan_x_end = int(iw * 0.50)
    sample_ys = [int(ih * p) for p in [0.3, 0.4, 0.5, 0.6, 0.7]]

    edge_votes = {}
    for sy in sample_ys:
        best_x, best_diff = None, 0
        for x in range(scan_x_start, scan_x_end):
            d = _color_diff(px[x, sy][:3], px[x+1, sy][:3])
            if d > best_diff:
                best_diff = d
                best_x = x
        if best_x and best_diff > 30:
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
        sidebar_right_px = max(edge_votes, key=edge_votes.get)
        if edge_votes[sidebar_right_px] < 2:
            sidebar_right_px = None

    # WeChat sidebar is typically 35–42 % of window width.
    # The old 0.25 fallback was too narrow (icon-column width, not sidebar right).
    _min_plausible_px = int(iw * 0.28)  # ~210pt for 756pt window
    if sidebar_right_px is None or sidebar_right_px < _min_plausible_px:
        sidebar_right_px = int(iw * 0.38)  # ~287pt — good WeChat default
        print(f"  ⚠️  Could not detect sidebar divider, using fallback: {sidebar_right_px/scale:.0f}pt")

    sidebar_right_pt = int(sidebar_right_px / scale)

    # --- Find icon column right edge ---
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

    # --- Find title bar bottom ---
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


def get_chat_area(window: dict, layout: Optional[dict] = None) -> dict:
    """Return the chat message area region."""
    if layout:
        return layout["chat_area"]

    sidebar_width = int(window["width"] * 0.25)
    top_bar = 60
    bottom_bar = 160

    return {
        "x": window["x"] + sidebar_width,
        "y": window["y"] + top_bar,
        "width": window["width"] - sidebar_width,
        "height": window["height"] - top_bar - bottom_bar,
    }

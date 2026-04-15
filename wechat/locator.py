"""
locator.py — Locator class for finding WeChat UI elements.

The Locator encapsulates all "where is this element?" logic.
It holds window geometry and detected layout, and provides
named methods that return Element objects.

Uses Snapshot internally to cache screenshots + OCR results,
avoiding redundant captures within the same operation.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

from wechat import actions
from wechat.element import Element, Snapshot
from wechat.layout import detect_layout, get_chat_area
from wechat.ocr import select_candidates_by_type


class Locator:
    """Finds UI element positions in the WeChat window."""

    def __init__(self, window: Optional[dict] = None, layout: Optional[dict] = None):
        self.window = window or actions.get_window_info()
        self.layout = layout or detect_layout(self.window)
        self._snapshots: dict[str, Snapshot] = {}

    def refresh(self) -> None:
        """Re-acquire window info and re-detect layout. Clears cached snapshots."""
        self.window = actions.get_window_info()
        self.layout = detect_layout(self.window)
        self._snapshots.clear()

    def snapshot(self, region_name: str) -> Snapshot:
        """
        Get a Snapshot for a named region. Returns cached version if fresh,
        otherwise takes a new screenshot + OCR.
        """
        cached = self._snapshots.get(region_name)
        if cached and not cached.is_stale:
            return cached

        region = self._region_for_name(region_name)
        snap = Snapshot(region, region_name=region_name)
        self._snapshots[region_name] = snap
        return snap

    def invalidate(self, region_name: Optional[str] = None) -> None:
        """Clear cached snapshot(s). If region_name is None, clear all."""
        if region_name:
            self._snapshots.pop(region_name, None)
        else:
            self._snapshots.clear()

    def _region_for_name(self, name: str) -> dict:
        """Map a region name to its screen coordinates."""
        if name == "sidebar":
            return self.sidebar_region()
        elif name == "search_panel":
            region = self.sidebar_region()
            return {
                "x": region["x"],
                "y": region["y"],
                "width": region["width"] + 100,
                "height": region["height"],
            }
        elif name == "title":
            return self.title_region()
        elif name == "chat_area":
            return self.chat_area()
        raise ValueError(f"Unknown region: {name}")

    # --- Named element locators ---

    def search_box(self) -> Element:
        """Return Element for the search box."""
        pos = self.layout["search_box_center"]
        return Element(text="search_box", position=pos, region="toolbar")

    def chat_area(self) -> dict:
        """Return chat area region dict."""
        return get_chat_area(self.window, self.layout)

    def chat_area_center(self) -> Element:
        """Return Element for the center of the chat area."""
        area = self.chat_area()
        pos = (area["x"] + area["width"] // 2, area["y"] + area["height"] // 2)
        return Element(text="chat_area", position=pos, region="chat_area")

    def input_box(self) -> Element:
        """Return Element for the message input box."""
        wx, wy = self.window["x"], self.window["y"]
        ww, wh = self.window["width"], self.window["height"]
        sidebar_right = self.layout["sidebar_right"] if self.layout else int(ww * 0.25)
        inputbox_top = self.layout["inputbox_top"] if self.layout else (wh - 160)

        x = wx + sidebar_right + (ww - sidebar_right) // 2
        y = wy + inputbox_top + (wh - inputbox_top) // 2
        return Element(text="input_box", position=(x, y), region="input")

    def sidebar_region(self) -> dict:
        """Return sidebar region dict."""
        if self.layout:
            return self.layout["sidebar"]
        wx, wy = self.window["x"], self.window["y"]
        return {
            "x": wx + 72,
            "y": wy + 60,
            "width": 220,
            "height": min(self.window.get("height", 890) - 60, 800),
        }

    def sidebar_click_x(self) -> int:
        """Return x coordinate for clicking sidebar items."""
        if self.layout:
            return self.layout["sidebar_center_x"]
        return self.window["x"] + 180

    def sidebar_item(self, name: str) -> Optional[Element]:
        """
        Find a named item in the sidebar via OCR.
        Returns Element or None if not found.
        """
        snap = self.snapshot("sidebar")
        elem = snap.find_text(name)
        if elem is None:
            return None
        # Override x to use sidebar center (more reliable click target)
        return Element(text=elem.text, position=(self.sidebar_click_x(), elem.position[1]), region="sidebar")

    def search_result(self, name: str, target_type: str = "group") -> Optional[Element]:
        """
        Find a search result matching name in the correct section.
        Must be called after search text has been entered.
        Returns Element or None.
        """
        snap = self.snapshot("search_panel")
        sections = snap.find_search_sections(name)

        candidates = select_candidates_by_type(
            {k: [(e.position[1], e.text) for e in v] for k, v in sections.items()},
            target_type,
        )
        if not candidates:
            return None

        result_y, result_text = candidates[0]
        result_x = snap.region["x"] + snap.region["width"] // 2
        print(f"  Selected '{result_text}' from {target_type} search at y={result_y}")
        return Element(text=result_text, position=(result_x, result_y), region=target_type)

    def title_region(self) -> dict:
        """Return the chat title bar region (right of sidebar, full header height)."""
        wx, wy = self.window["x"], self.window["y"]
        ww = self.window["width"]
        sidebar_right = self.layout["sidebar_right"] if self.layout else int(ww * 0.25)
        # Use detected titlebar_bottom so the group name text is fully covered.
        # Fixed height of 50pt can cut off the name if macOS title bar + WeChat
        # header together are taller than 50pt.
        title_height = (self.layout["titlebar_bottom"] + 10) if self.layout else 80
        return {
            "x": wx + sidebar_right,
            "y": wy,
            "width": ww - sidebar_right,
            "height": title_height,
        }

    def verify_chat_title(self, name: str, debug_save: Optional[str] = None) -> bool:
        """Check if the chat title bar shows the expected name."""
        # Ensure WeChat is frontmost before screenshotting the title bar.
        # Without this explicit activate, another window can cover WeChat
        # between the Quartz click and the screenshot (blank [] result).
        actions.activate_wechat()
        try:
            self.window = actions.get_window_info()
        except Exception:  # noqa: BLE001
            pass
        self.invalidate("title")
        snap = self.snapshot("title")
        return snap.contains_text(name)

    def verify_search_results(self, name: str) -> bool:
        """Check if search results contain the target name."""
        self.invalidate("search_panel")
        snap = self.snapshot("search_panel")
        return snap.contains_text(name)

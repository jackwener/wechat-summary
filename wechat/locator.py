"""
locator.py — Locator class for finding WeChat UI elements.

The Locator encapsulates all "where is this element?" logic.
It holds window geometry and detected layout, and provides
named methods that return screen positions.
"""

from typing import Optional

from PIL import Image

from wechat import actions
from wechat.layout import detect_layout, get_chat_area
from wechat.ocr import (
    ocr_find_text_y,
    parse_search_sections,
    select_candidates_by_type,
    verify_text_in_region,
)


class Locator:
    """Finds UI element positions in the WeChat window."""

    def __init__(self, window: Optional[dict] = None, layout: Optional[dict] = None):
        self.window = window or actions.get_window_info()
        self.layout = layout or detect_layout(self.window)

    def refresh(self) -> None:
        """Re-acquire window info and re-detect layout."""
        self.window = actions.get_window_info()
        self.layout = detect_layout(self.window)

    # --- Named element locators ---

    def search_box(self) -> tuple[int, int]:
        """Return (x, y) center of the search box."""
        return self.layout["search_box_center"]

    def chat_area(self) -> dict:
        """Return chat area region dict."""
        return get_chat_area(self.window, self.layout)

    def chat_area_center(self) -> tuple[int, int]:
        """Return (x, y) center of the chat area."""
        area = self.chat_area()
        return (area["x"] + area["width"] // 2, area["y"] + area["height"] // 2)

    def input_box(self) -> tuple[int, int]:
        """Return (x, y) center of the message input box."""
        wx, wy = self.window["x"], self.window["y"]
        ww, wh = self.window["width"], self.window["height"]
        sidebar_right = self.layout["sidebar_right"] if self.layout else int(ww * 0.25)
        inputbox_top = self.layout["inputbox_top"] if self.layout else (wh - 160)

        # Input box: horizontally centered in chat column, vertically centered between inputbox_top and window bottom
        x = wx + sidebar_right + (ww - sidebar_right) // 2
        y = wy + inputbox_top + (wh - inputbox_top) // 2
        return (x, y)

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

    def sidebar_item(self, name: str) -> Optional[tuple[int, int]]:
        """
        Find a named item in the sidebar via OCR.
        Returns (x, y) screen coordinates or None if not found.
        """
        region = self.sidebar_region()
        path = "/tmp/_sidebar_scan.png"
        actions.screenshot(region, path)
        y = ocr_find_text_y(path, region, name)
        if y is None:
            return None
        return (self.sidebar_click_x(), y)

    def search_result(self, name: str, target_type: str = "group") -> Optional[tuple[int, int]]:
        """
        Find a search result matching name in the correct section.
        Must be called after search text has been entered.
        Returns (x, y) screen coordinates or None.
        """
        import Vision
        from Foundation import NSURL

        search_region = self.sidebar_region()
        search_region = {
            "x": search_region["x"],
            "y": search_region["y"],
            "width": search_region["width"] + 100,
            "height": search_region["height"],
        }

        path = "/tmp/_search_click.png"
        actions.screenshot(search_region, path)

        # OCR the search results
        img_url = NSURL.fileURLWithPath_(path)
        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        req.setRecognitionLanguages_(["zh-Hans", "en"])
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(img_url, {})
        handler.performRequests_error_([req], None)

        img = Image.open(path)
        img_h = img.height
        scale = img_h / search_region["height"] if search_region["height"] > 0 else 1

        sections = parse_search_sections(
            req.results(), search_region, img_h, scale, name
        )

        candidates = select_candidates_by_type(sections, target_type)
        if not candidates:
            return None

        result_y = candidates[0][0]
        result_x = search_region["x"] + search_region["width"] // 2
        print(f"  Selected '{candidates[0][1]}' from {target_type} search at y={result_y}")
        return (result_x, result_y)

    def title_region(self) -> dict:
        """Return the chat title bar region (top ~50pt right of sidebar)."""
        wx, wy = self.window["x"], self.window["y"]
        ww = self.window["width"]
        sidebar_right = self.layout["sidebar_right"] if self.layout else int(ww * 0.25)
        return {
            "x": wx + sidebar_right,
            "y": wy,
            "width": ww - sidebar_right,
            "height": 50,
        }

    def verify_chat_title(self, name: str) -> bool:
        """Check if the chat title bar shows the expected name."""
        # Re-acquire window in case layout changed
        try:
            self.window = actions.get_window_info()
        except Exception:  # noqa: BLE001
            pass
        return verify_text_in_region(name, self.title_region())

    def verify_search_results(self, name: str) -> bool:
        """Check if search results contain the target name."""
        region = self.sidebar_region()
        return verify_text_in_region(name, region, expand_width=100)

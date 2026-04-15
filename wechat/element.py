"""
element.py — Element and Snapshot types for WeChat UI automation.

Element: a located UI element with text, position, and region info.
Snapshot: a cached screenshot + OCR result for a screen region.
"""

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class Element:
    """A located UI element on screen."""

    text: str
    position: tuple[int, int]  # (x, y) screen coordinates
    region: str = ""           # logical region name (e.g., "sidebar", "search_panel")

    def click(self) -> None:
        """Click this element."""
        from wechat import actions
        actions.click(*self.position)

    def __bool__(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"Element('{self.text}', pos={self.position}, region='{self.region}')"


# Stale threshold: snapshots older than this are automatically refreshed.
_SNAPSHOT_TTL = 3.0  # seconds


class Snapshot:
    """
    A cached screenshot + OCR results for a screen region.

    Take one screenshot, run OCR once, then query multiple times
    without re-capturing. Automatically expires after _SNAPSHOT_TTL seconds.
    """

    def __init__(self, region: dict, region_name: str = "", image_path: Optional[str] = None):
        self.region = region
        self.region_name = region_name
        self.timestamp = time.time()

        # Take screenshot (lazy import to avoid circular dependency)
        from wechat import actions as _actions
        self._image_path = image_path or f"/tmp/_snapshot_{region_name or 'unknown'}.png"
        _actions.screenshot(region, self._image_path)

        # Run OCR once
        self._ocr_observations = self._run_ocr()
        self._img_height, self._scale = self._compute_scale()

    @property
    def is_stale(self) -> bool:
        return time.time() - self.timestamp > _SNAPSHOT_TTL

    def _run_ocr(self) -> list:
        """Run Vision OCR and return raw observations."""
        import Vision
        from Foundation import NSURL

        img_url = NSURL.fileURLWithPath_(self._image_path)
        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        req.setRecognitionLanguages_(["zh-Hans", "en"])
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(img_url, {})
        handler.performRequests_error_([req], None)
        return list(req.results() or [])

    def _compute_scale(self) -> tuple[int, float]:
        """Compute image height and scale factor."""
        from PIL import Image
        img = Image.open(self._image_path)
        img_h = img.height
        scale = img_h / self.region["height"] if self.region["height"] > 0 else 1
        return img_h, scale

    def _obs_screen_y(self, obs) -> int:
        """Convert a Vision observation's bounding box to screen y coordinate."""
        bbox = obs.boundingBox()
        cy_px = (1.0 - bbox.origin.y - bbox.size.height / 2) * self._img_height
        return self.region["y"] + int(cy_px / self._scale)

    def all_texts(self) -> list[str]:
        """Return all OCR-recognized text strings."""
        texts = []
        for obs in self._ocr_observations:
            top = obs.topCandidates_(1)
            if top:
                texts.append(top[0].string())
        return texts

    def contains_text(self, target: str) -> bool:
        """Check if target text appears in this snapshot (fuzzy matching)."""
        from wechat.ocr import ocr_normalize

        texts = self.all_texts()
        print(f"  Verify OCR texts: {texts[:10]}")
        all_text = " ".join(texts).lower()
        all_text_compact = all_text.replace(" ", "")
        all_text_norm = ocr_normalize(" ".join(texts))
        all_text_compact_norm = all_text_norm.replace(" ", "")

        target_lower = target.lower()
        target_compact = target_lower.replace(" ", "")
        target_norm = ocr_normalize(target)
        target_compact_norm = target_norm.replace(" ", "")

        if target_lower in all_text:
            return True
        if target_compact in all_text_compact:
            return True
        if target_norm in all_text_norm or target_compact_norm in all_text_compact_norm:
            return True

        target_parts = [p for p in target_lower.split() if len(p) >= 1]
        if target_parts:
            matched = sum(1 for part in target_parts if part in all_text or ocr_normalize(part) in all_text_norm)
            if matched >= max(1, len(target_parts) * 0.5):
                return True

        # Handle truncated WeChat titles: "...流群（499）" for long group names.
        # Check if any trailing suffix of the target appears in the OCR text.
        if len(target_lower) >= 4:
            for suffix_len in range(min(len(target_lower), 10), 2, -1):
                suffix = target_lower[-suffix_len:].replace(" ", "")
                if len(suffix) >= 3 and suffix in all_text_compact:
                    return True

        return False

    def find_text(self, target: str) -> Optional[Element]:
        """
        Find text matching target in this snapshot.
        Returns the best-matching Element, or None.
        Uses priority-based matching (exact > substring > prefix > partial).
        """
        from wechat.ocr import ocr_normalize

        target_lower = target.lower()
        target_clean = target_lower.rstrip('.…。').strip()
        target_norm = ocr_normalize(target)
        target_clean_norm = target_norm.rstrip('.…。').strip()
        target_compact = target_lower.replace(" ", "")
        target_compact_norm = target_norm.replace(" ", "")

        candidates = []  # (priority, screen_y, text)

        for obs in self._ocr_observations:
            top = obs.topCandidates_(1)
            if not top:
                continue
            text = top[0].string()
            text_lower = text.lower()
            text_clean = text_lower.rstrip('.…。').strip()
            text_norm = ocr_normalize(text)
            text_clean_norm = text_norm.rstrip('.…。').strip()
            text_compact = text_lower.replace(" ", "")
            text_compact_norm = text_norm.replace(" ", "")

            sy = self._obs_screen_y(obs)

            # Priority 0: exact match
            if (text_clean == target_clean or text_clean_norm == target_clean_norm or
                    text_compact == target_compact or text_compact_norm == target_compact_norm):
                candidates.append((0, sy, text))
                continue

            # Priority 1: full substring match
            if (target_lower in text_lower or text_lower in target_lower or
                    target_norm in text_norm or text_norm in target_norm or
                    target_compact in text_compact or text_compact in target_compact or
                    target_compact_norm in text_compact_norm or text_compact_norm in target_compact_norm):
                candidates.append((1, sy, text))
                continue

            # Priority 2: prefix match
            if len(text_clean) >= 5 and (
                target_clean.startswith(text_clean) or text_clean.startswith(target_clean) or
                target_clean_norm.startswith(text_clean_norm) or text_clean_norm.startswith(target_clean_norm) or
                target_compact.startswith(text_compact) or text_compact.startswith(target_compact)
            ):
                candidates.append((2, sy, text))
                continue

            # Priority 3: word-level partial match
            target_parts = [p for p in target.split() if len(p) > 2]
            if target_parts and (target_parts[0].lower() in text_lower or ocr_normalize(target_parts[0]) in text_norm):
                if len(target_parts) >= 2:
                    second = target_parts[1].lower()
                    if second[:4] in text_lower or ocr_normalize(second[:4]) in text_norm:
                        candidates.append((3, sy, text))

        if not candidates:
            return None

        candidates.sort(key=lambda c: c[0])
        best = candidates[0]
        print(f"  OCR matches: {[(p, t) for p, _, t in candidates]}, selected: priority={best[0]} text='{best[2]}'")

        # Use region center x for click position (sidebar items are clicked at center x)
        click_x = self.region["x"] + self.region["width"] // 2
        return Element(text=best[2], position=(click_x, best[1]), region=self.region_name)

    def find_search_sections(self, target_name: str) -> dict[str, list]:
        """
        Parse OCR results into section-aware buckets for search results.
        Returns dict of section_name → list of Element.
        """
        from difflib import SequenceMatcher
        from wechat.ocr import ocr_normalize, _match_section_header, _find_containing_section, _SECTION_HEADER_PATTERNS

        target_norm = ocr_normalize(target_name)
        target_compact_norm = target_norm.replace(" ", "")

        # Collect all OCR blocks
        all_blocks = []
        for obs in self._ocr_observations:
            top = obs.topCandidates_(1)
            if not top:
                continue
            text = top[0].string()
            sy = self._obs_screen_y(obs)
            all_blocks.append((sy, text.lower().strip(), text))

        all_blocks.sort(key=lambda b: b[0])

        # Identify section headers
        section_headers = []
        used_indices = set()

        for i, (sy, tl, _orig) in enumerate(all_blocks):
            matched_section = _match_section_header(tl)

            if matched_section is None and i + 1 < len(all_blocks):
                next_sy, next_tl, _ = all_blocks[i + 1]
                if abs(next_sy - sy) <= 8:
                    merged = f"{tl} {next_tl}"
                    matched_section = _match_section_header(merged)
                    if matched_section is not None:
                        used_indices.add(i + 1)

            if matched_section is not None:
                section_headers.append((sy, matched_section))
                used_indices.add(i)

        section_headers.sort(key=lambda h: h[0])
        print(f"  Detected section headers: {[(name, y) for y, name in section_headers]}")

        # Classify matches into sections, returning Elements.
        # "unknown" bucket catches items before the first section header
        # (WeChat shows Group Chats at the top without a labelled header).
        sections: dict[str, list] = {name: [] for name in _SECTION_HEADER_PATTERNS}
        sections["unknown"] = []
        click_x = self.region["x"] + self.region["width"] // 2

        def _text_matches(text_norm_compact: str) -> bool:
            if (target_compact_norm == text_norm_compact or
                    target_compact_norm in text_norm_compact or
                    text_norm_compact in target_compact_norm):
                return True
            # Fuzzy match: tolerates OCR errors (O↔C, l↔I↔1, 交↔父, 群↔苟…)
            sim = SequenceMatcher(None, target_compact_norm, text_norm_compact).ratio()
            return sim >= 0.55

        for i, (sy, tl, text_orig) in enumerate(all_blocks):
            if i in used_indices:
                continue

            text_norm = ocr_normalize(text_orig)
            text_compact_norm = text_norm.replace(" ", "")

            if not _text_matches(text_compact_norm):
                continue

            containing = _find_containing_section(sy, section_headers)
            elem = Element(text=text_orig, position=(click_x, sy), region=containing)
            sections.setdefault(containing, []).append(elem)

        for section_name, matches in sections.items():
            if matches:
                print(f"  Section '{section_name}': {[e.text for e in matches]}")

        return sections

    @property
    def observations(self) -> list:
        """Raw Vision OCR observations (for advanced use)."""
        return self._ocr_observations

    @property
    def image_height(self) -> int:
        return self._img_height

    @property
    def scale(self) -> float:
        return self._scale

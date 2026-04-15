"""
ocr.py — OCR utilities for WeChat UI element recognition.

Uses macOS Vision framework for text recognition.
Provides text normalization, text search in images, and
section-aware parsing of WeChat search results.
"""

from typing import Optional

from PIL import Image

from wechat.actions import screenshot


def ocr_normalize(text: str) -> str:
    """Normalize text to handle common OCR confusions (I/l/1, O/0, etc.)."""
    t = text.lower()
    t = t.replace("l", "i").replace("1", "i")
    t = t.replace("0", "o")
    return t


def ocr_all_text(image_path: str) -> list[str]:
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


def ocr_find_text_y(image_path: str, region: dict, target: str) -> Optional[int]:
    """
    OCR an image and return the screen y-coordinate of text matching target.
    Collects all matching candidates and prefers exact matches over fuzzy ones.
    Handles WeChat sidebar truncation (e.g. "golang runtim..").
    """
    import Vision
    from Foundation import NSURL

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
    target_norm = ocr_normalize(target)
    target_clean_norm = target_norm.rstrip('.…。').strip()
    target_compact = target_lower.replace(" ", "")
    target_compact_norm = target_norm.replace(" ", "")

    scale_factor = img_h / region["height"] if region["height"] > 0 else 1

    def _get_screen_y(obs):
        bbox = obs.boundingBox()
        center_y_px = (1.0 - bbox.origin.y - bbox.size.height / 2) * img_h
        center_y_pts = center_y_px / scale_factor
        return region["y"] + int(center_y_pts)

    # Collect candidates with priority: 0=exact, 1=full-substring, 2=prefix, 3=partial
    candidates = []

    for obs in results:
        top_candidates = obs.topCandidates_(1)
        if not top_candidates:
            continue
        text = top_candidates[0].string()
        text_lower = text.lower()
        text_clean = text_lower.rstrip('.…。').strip()
        text_norm = ocr_normalize(text)
        text_clean_norm = text_norm.rstrip('.…。').strip()
        text_compact = text_lower.replace(" ", "")
        text_compact_norm = text_norm.replace(" ", "")

        # Priority 0: exact match
        if (text_clean == target_clean or text_clean_norm == target_clean_norm or
                text_compact == target_compact or text_compact_norm == target_compact_norm):
            candidates.append((0, _get_screen_y(obs), text))
            continue

        # Priority 1: full substring match
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
        if target_parts and (target_parts[0].lower() in text_lower or ocr_normalize(target_parts[0]) in text_norm):
            if len(target_parts) >= 2:
                second = target_parts[1].lower()
                if second[:4] in text_lower or ocr_normalize(second[:4]) in text_norm:
                    candidates.append((3, _get_screen_y(obs), text))

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[0])
    best = candidates[0]
    print(f"  OCR matches: {[(p, t) for p, _, t in candidates]}, selected: priority={best[0]} text='{best[2]}'")
    return best[1]


def verify_text_in_region(target: str, region: dict, expand_width: int = 0) -> bool:
    """
    Screenshot a region, OCR it, and check if target text is present.
    Used to verify search results and chat titles.
    """
    scan_region = dict(region)
    if expand_width:
        scan_region["width"] = region["width"] + expand_width

    path = "/tmp/_verify_text.png"
    screenshot(scan_region, path)
    texts = ocr_all_text(path)
    all_text = " ".join(texts).lower()
    all_text_compact = all_text.replace(" ", "")
    all_text_norm = ocr_normalize(" ".join(texts))
    all_text_compact_norm = all_text_norm.replace(" ", "")

    print(f"  Verify OCR texts: {texts[:10]}")

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

    return False


# --- Section-aware search result parsing ---

# Section header patterns for WeChat search results.
_SECTION_HEADER_PATTERNS: dict[str, list] = {
    "internet_search": ["internet search", "搜索"],
    "contacts": ["contacts", "联系人"],
    "group_chats": ["group chats", "group chat", "群聊"],
    "chat_history": ["chat history", "聊天记录"],
    "official_accounts": ["official accounts", "公众号"],
    "mini_programs": ["mini programs", "小程序"],
}

# Priority order when target_type is "any"
_SECTION_PRIORITY = ["group_chats", "contacts", "chat_history"]


def _match_section_header(text_lower: str) -> Optional[str]:
    """Match a lowercase text string against known section header patterns."""
    for section_name, patterns in _SECTION_HEADER_PATTERNS.items():
        for pattern in patterns:
            if pattern in text_lower:
                return section_name
    return None


def _find_containing_section(y: int, section_headers: list) -> str:
    """Find which section a given y coordinate falls into."""
    containing = "unknown"
    for header_y, section_name in section_headers:
        if header_y <= y:
            containing = section_name
        else:
            break
    return containing


def parse_search_sections(
    ocr_observations: list,
    search_region: dict,
    img_height: int,
    scale: float,
    target_name: str,
) -> dict[str, list]:
    """
    Parse OCR results from a search screenshot into section-aware buckets.

    Returns a dict mapping section name → list of (screen_y, text) tuples.
    """

    def _obs_screen_y(obs):
        bbox = obs.boundingBox()
        cy_px = (1.0 - bbox.origin.y - bbox.size.height / 2) * img_height
        return search_region["y"] + int(cy_px / scale)

    target_norm = ocr_normalize(target_name)
    target_compact_norm = target_norm.replace(" ", "")

    # Collect all OCR blocks with y positions
    all_blocks = []
    for obs in (ocr_observations or []):
        top = obs.topCandidates_(1)
        if not top:
            continue
        text = top[0].string()
        sy = _obs_screen_y(obs)
        all_blocks.append((sy, text.lower().strip(), text))

    all_blocks.sort(key=lambda b: b[0])

    # Identify section headers (merge adjacent blocks within 8px for split headers)
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

    # Classify name-matching blocks into sections
    sections: dict[str, list] = {name: [] for name in _SECTION_HEADER_PATTERNS}

    for i, (sy, tl, text_orig) in enumerate(all_blocks):
        if i in used_indices:
            continue

        text_norm = ocr_normalize(text_orig)
        text_compact_norm = text_norm.replace(" ", "")

        if not (target_compact_norm == text_compact_norm or
                target_compact_norm in text_compact_norm or
                text_compact_norm in target_compact_norm):
            continue

        containing_section = _find_containing_section(sy, section_headers)
        sections[containing_section].append((sy, text_orig))

    for section_name, matches in sections.items():
        if matches:
            print(f"  Section '{section_name}': {[(t, y) for y, t in matches]}")

    return sections


def select_candidates_by_type(sections: dict[str, list], target_type: str) -> list:
    """
    Select candidate matches from parsed sections based on target_type.

    Returns list of (screen_y, text) candidates from the appropriate section(s).
    """
    if target_type == "group":
        preferred = ["group_chats"]
    elif target_type == "contact":
        preferred = ["contacts"]
    else:
        preferred = _SECTION_PRIORITY

    for section_name in preferred:
        candidates = sections.get(section_name, [])
        if candidates:
            return candidates

    for section_name, matches in sections.items():
        if matches and section_name != "internet_search":
            print(f"  ⚠️  No match in preferred sections, falling back to '{section_name}'")
            return matches

    return []


def describe_target_type(target_type: str) -> str:
    """Human-readable description of target_type for error messages."""
    if target_type == "group":
        return "Group Chats"
    elif target_type == "contact":
        return "Contacts"
    return "Group Chats / Contacts / Chat History"

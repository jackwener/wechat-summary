"""
ocr.py — OCR utilities for WeChat UI element recognition.

Provides text normalization, section header matching,
and section-aware candidate selection.

Note: OCR execution and text search are now in Snapshot (element.py).
This module retains shared helpers used by both Snapshot and Locator.
"""

from typing import Optional


def ocr_normalize(text: str) -> str:
    """Normalize text to handle common OCR confusions (I/l/1, O/0, etc.)."""
    t = text.lower()
    t = t.replace("l", "i").replace("1", "i")
    t = t.replace("0", "o")
    return t


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

    # When the preferred section header wasn't OCR'd (WeChat omits headers for
    # single-match results), items land in "unknown". Use them as fallback for
    # any target type — the top result is almost always the right one.
    unknown = sections.get("unknown", [])
    if unknown:
        label = {"group": "Group Chats", "contact": "Contacts"}.get(target_type, "section")
        print(f"  ℹ️  {label} header not detected; using {len(unknown)} 'unknown' candidate(s) (likely top-of-results)")
        return unknown

    # Last resort: any non-internet, non-unknown section
    for section_name, matches in sections.items():
        if matches and section_name not in ("internet_search", "unknown"):
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

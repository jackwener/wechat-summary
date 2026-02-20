"""
ocr.py — OCR using macOS Vision framework

Uses VNRecognizeTextRequest to extract text from WeChat chat screenshots.
Supports Chinese (zh-Hans) and English, with deduplication across pages.
"""

import os
import re
from pathlib import Path

CAPTURE_FILENAME_RE = re.compile(r"^(?P<prefix>\d{8}_\d{6})_page_(?P<page>\d+)\.png$")


def _collect_screenshot_files(screenshot_dir: Path, file_paths: list[str] | None) -> list[Path]:
    """
    Collect screenshot files for OCR.

    If file_paths is provided, preserve that explicit order.
    If scanning a directory, auto-detect capture naming pattern and:
      1. keep only the latest run when multiple runs are present
      2. sort pages by descending page index (oldest -> newest)
    """
    if file_paths is not None:
        files = [Path(p) for p in file_paths]
        missing = [str(p) for p in files if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Screenshot files not found: {missing[:3]}")
        return files

    files = sorted(screenshot_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNG files found in {screenshot_dir}")

    parsed = []
    for f in files:
        m = CAPTURE_FILENAME_RE.match(f.name)
        if not m:
            return files
        parsed.append((f, m.group("prefix"), int(m.group("page"))))

    prefixes = sorted({prefix for _, prefix, _ in parsed})
    if len(prefixes) > 1:
        latest = prefixes[-1]
        print(
            f"Detected {len(prefixes)} capture runs in {screenshot_dir}. "
            f"Using latest run: {latest}"
        )
    else:
        latest = prefixes[0]

    run_files = [(f, page) for f, prefix, page in parsed if prefix == latest]
    run_files.sort(key=lambda item: item[1], reverse=True)
    return [f for f, _ in run_files]


def collect_screenshot_files(screenshot_dir: Path, file_paths: list[str] | None = None) -> list[Path]:
    """Public wrapper for screenshot file discovery and ordering."""
    return _collect_screenshot_files(screenshot_dir, file_paths)


def _import_vision_stack():
    """Import macOS OCR frameworks lazily with a clear error message."""
    try:
        import Quartz
        import Vision
        from Foundation import NSURL
    except ImportError as exc:
        raise RuntimeError(
            "OCR requires macOS Vision frameworks. "
            "Install dependencies from requirements.txt in a macOS environment."
        ) from exc
    return Quartz, Vision, NSURL


def recognize_text_in_image(image_path: str, languages: list[str] = None) -> list[dict]:
    """
    Recognize text in a single image using macOS Vision framework.

    Args:
        image_path: Path to the image file
        languages: OCR languages, default ["zh-Hans", "en-US"]

    Returns:
        List of dicts with keys: text, confidence, y_position (0=bottom, 1=top)
    """
    if languages is None:
        languages = ["zh-Hans", "en-US"]

    Quartz, Vision, NSURL = _import_vision_stack()

    # Load image
    image_url = NSURL.fileURLWithPath_(os.path.abspath(image_path))
    image_source = Quartz.CGImageSourceCreateWithURL(image_url, None)
    if image_source is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    cg_image = Quartz.CGImageSourceCreateImageAtIndex(image_source, 0, None)
    if cg_image is None:
        raise ValueError(f"Cannot create CGImage from: {image_path}")

    # Create and configure text recognition request
    results = []

    def completion_handler(request, error):
        if error:
            print(f"OCR error: {error}")
            return
        observations = request.results()
        if observations is None:
            return
        for obs in observations:
            candidates = obs.topCandidates_(1)
            if candidates and len(candidates) > 0:
                text = candidates[0].string()
                confidence = candidates[0].confidence()
                # Get bounding box (y: 0=bottom, 1=top in Vision coordinates)
                bbox = obs.boundingBox()
                y_pos = bbox.origin.y + bbox.size.height / 2
                results.append({
                    "text": text,
                    "confidence": confidence,
                    "y_position": y_pos,  # Vision coords: 0=bottom, 1=top
                })

    request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(
        completion_handler
    )
    request.setRecognitionLanguages_(languages)
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)

    # Perform recognition
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
        cg_image, None
    )
    success = handler.performRequests_error_([request], None)
    if not success[0]:
        print(f"OCR request failed for {image_path}")

    # Sort top-to-bottom (high y = top of image in Vision coords)
    results.sort(key=lambda r: r["y_position"], reverse=True)
    return results


def _similarity(a: str, b: str) -> float:
    """Compute similarity ratio between two strings."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()


def _normalize_line(text: str) -> str:
    """Normalize OCR line for matching."""
    return "".join(ch for ch in text.lower().strip() if not ch.isspace())


def _longest_true_streak(flags: list[bool]) -> int:
    """Return longest consecutive True streak length."""
    best = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return best


def _find_anchor_overlap(
    merged: list[str],
    current: list[str],
    max_overlap_lines: int,
    similarity_threshold: float,
) -> int:
    """
    Find suffix/prefix overlap length using anchor-based matching.

    Instead of requiring all line pairs to match, this method looks for:
    1. High-confidence anchor matches
    2. Sufficient overall match ratio
    3. Consecutive match streak to preserve sequence continuity
    """
    if not merged or not current:
        return 0

    check_range = min(max_overlap_lines, len(merged), len(current))
    if check_range <= 0:
        return 0

    normalized_merged = [_normalize_line(line) for line in merged]
    normalized_current = [_normalize_line(line) for line in current]
    high_conf_threshold = min(0.95, similarity_threshold + 0.1)

    for n in range(check_range, 0, -1):
        tail = normalized_merged[-n:]
        head = normalized_current[:n]
        similarities = [_similarity(a, b) for a, b in zip(tail, head)]
        matched = [sim >= similarity_threshold for sim in similarities]

        match_count = sum(matched)
        if match_count == 0:
            continue

        match_ratio = match_count / n
        anchor_count = sum(1 for sim in similarities if sim >= high_conf_threshold)
        streak = _longest_true_streak(matched)

        required_ratio = 0.6 if n >= 8 else 0.5
        required_streak = 2 if n >= 4 else 1
        required_anchors = 2 if n >= 8 else 1

        if (
            match_ratio >= required_ratio
            and streak >= required_streak
            and anchor_count >= required_anchors
        ):
            return n

    return 0


def deduplicate_pages(
    pages: list[list[dict]],
    max_overlap_lines: int = 15,
    similarity_threshold: float = 0.8,
) -> list[str]:
    """
    Merge OCR results from multiple pages, removing overlapping lines.

    Uses fuzzy matching since OCR may produce slightly different text
    for the same content across screenshots.

    Args:
        pages: List of per-page OCR results (each a list of text dicts)
        max_overlap_lines: Max lines to check for overlap
        similarity_threshold: Min similarity ratio to consider lines as matching

    Returns:
        Merged list of text lines (deduplicated)
    """
    if not pages:
        return []

    # Start with all lines from the first page
    merged = [r["text"] for r in pages[0]]

    for page_results in pages[1:]:
        current_lines = [r["text"] for r in page_results]
        if not current_lines:
            continue

        # Find overlap using anchors rather than full line-by-line equality
        best_overlap = _find_anchor_overlap(
            merged,
            current_lines,
            max_overlap_lines=max_overlap_lines,
            similarity_threshold=similarity_threshold,
        )

        if best_overlap > 0:
            print(f"  Detected anchor-based overlap: {best_overlap} lines")

        # Append non-overlapping lines
        merged.extend(current_lines[best_overlap:])

    return merged


def ocr_screenshots(
    screenshot_dir: str = "screenshots",
    languages: list[str] = None,
    file_paths: list[str] | None = None,
) -> str:
    """
    OCR all screenshots in a directory and merge into a single text.

    Args:
        screenshot_dir: Directory containing screenshot PNGs
        languages: OCR languages
        file_paths: Explicit screenshot file list (preserved order)

    Returns:
        Merged text from all screenshots
    """
    screenshot_dir = Path(screenshot_dir)
    files = _collect_screenshot_files(screenshot_dir, file_paths)

    print(f"Found {len(files)} screenshots to OCR...")

    all_pages = []
    for i, filepath in enumerate(files):
        print(f"OCR page {i + 1}/{len(files)}: {filepath.name}")
        page_results = recognize_text_in_image(str(filepath), languages)
        all_pages.append(page_results)
        print(f"  → {len(page_results)} text blocks recognized")

    # Deduplicate and merge
    merged_lines = deduplicate_pages(all_pages)
    text = "\n".join(merged_lines)

    print(f"\nTotal lines after dedup: {len(merged_lines)}")
    return text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OCR WeChat chat screenshots")
    parser.add_argument("--input", default="screenshots", help="Screenshot directory")
    parser.add_argument("--output", default=None, help="Output text file (default: stdout)")
    args = parser.parse_args()

    text = ocr_screenshots(args.input)

    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"\nSaved to {args.output}")
    else:
        print("\n" + "=" * 60)
        print(text)

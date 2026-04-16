"""
vision.py — Shared macOS Vision OCR entry point.

Both the page-level OCR pipeline (top-level ocr.py) and the Snapshot
UI-matching layer (wechat/element.py) use this module to run Vision
text recognition. Centralizing the Vision call here gives us:
  - one place to handle Vision framework import failures
  - one place to surface Vision errors (previously Snapshot silently
    swallowed them by passing None for the error pointer)
  - consistent request configuration (recognition level, languages,
    language correction)
"""

import os
from typing import Optional


def _import_vision_stack():
    """Import Vision frameworks with a clear error if missing."""
    try:
        import Vision
        from Foundation import NSURL
    except ImportError as exc:
        raise RuntimeError(
            "OCR requires macOS Vision frameworks. "
            "Install dependencies from requirements.txt in a macOS environment."
        ) from exc
    return Vision, NSURL


def run_text_recognition(
    image_path: str,
    languages: Optional[list[str]] = None,
    use_language_correction: bool = True,
) -> list:
    """Run Vision text recognition on an image file.

    Returns the raw list of VNRecognizedTextObservation. Callers read:
        obs.topCandidates_(1)[0].string()
        obs.topCandidates_(1)[0].confidence()
        obs.boundingBox()  # CGRect in normalized coords (0,0 = bottom-left)

    On failure, prints a warning and returns an empty list rather than
    raising — the caller typically treats "no text" the same as "OCR failed".
    """
    if languages is None:
        languages = ["zh-Hans", "en-US"]

    Vision, NSURL = _import_vision_stack()

    abspath = os.path.abspath(image_path)
    img_url = NSURL.fileURLWithPath_(abspath)

    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setRecognitionLanguages_(languages)
    req.setUsesLanguageCorrection_(use_language_correction)

    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(img_url, {})

    # PyObjC translates the NSError** out-parameter into a tuple return:
    # (BOOL success, NSError error_or_none). Unpack defensively so we
    # surface Vision errors instead of pretending OCR returned no text.
    result = handler.performRequests_error_([req], None)
    if isinstance(result, tuple):
        success, error = result[0], result[1] if len(result) > 1 else None
    else:
        success, error = bool(result), None

    if not success:
        err_msg = str(error) if error else "unknown Vision error"
        print(f"  ⚠️  Vision OCR failed for {image_path}: {err_msg}")
        return []

    return list(req.results() or [])


def observations_to_dicts(observations: list) -> list[dict]:
    """Convert Vision observations to dicts for the page-level OCR pipeline.

    Each dict has:
      - text: recognized string (top candidate)
      - confidence: float in [0, 1]
      - y_position: bbox vertical center in Vision coords (0=bottom, 1=top)
    """
    results: list[dict] = []
    for obs in observations:
        candidates = obs.topCandidates_(1)
        if not candidates or len(candidates) == 0:
            continue
        text = candidates[0].string()
        confidence = candidates[0].confidence()
        bbox = obs.boundingBox()
        y_pos = bbox.origin.y + bbox.size.height / 2
        results.append({
            "text": text,
            "confidence": confidence,
            "y_position": y_pos,
        })
    return results

"""OCR fallback using Tesseract (pytesseract).

Scope: page-level OCR only. We render the page to a high-DPI image and pass it
to Tesseract. Word-level bbox reconstruction is deferred to a future iteration —
M4 layout rebuild can do useful work with a single OCR text block per page.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import pymupdf
from PIL import Image

log = logging.getLogger(__name__)


def ocr_page(
    page: pymupdf.Page,
    *,
    page_number: int,
    languages: str,
    dpi: int,
) -> dict[str, Any] | None:
    """Render the page to an image and OCR it. Returns a normalized "ocr" block,
    or None if Tesseract is unavailable or returns no text."""
    try:
        import pytesseract  # local import: tool is optional in dev environments
    except ImportError:
        log.warning("pytesseract not installed; skipping OCR for page %d", page_number)
        return None

    try:
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(image, lang=languages).strip()
    except pytesseract.TesseractNotFoundError:
        log.warning(
            "tesseract binary not found on PATH; skipping OCR for page %d. "
            "Install with `brew install tesseract tesseract-lang`.",
            page_number,
        )
        return None
    except Exception:
        log.exception("OCR failed for page %d", page_number)
        return None

    if not text:
        return None

    return {
        "block_id": f"p{page_number}-ocr",
        "type": "ocr",
        "text": text,
        "bbox": [0.0, 0.0, float(page.rect.width), float(page.rect.height)],
        "dominant_font": None,
        "spans": [],
    }

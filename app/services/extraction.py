"""PyMuPDF-based PDF extraction.

Produces a normalized structured JSON document that downstream stages (chunking,
translation, layout reconstruction) consume. OCR fallback for scanned pages is
delegated to ``ocr.py``.

Output schema (per page):
    {
      "page_number": int,
      "width": float,          # PDF points
      "height": float,
      "is_scanned": bool,
      "blocks": [
        {
          "block_id": "p<page>-b<index>",
          "type": "paragraph" | "heading" | "image" | "ocr",
          "text": str,
          "bbox": [x0, y0, x1, y1],
          "dominant_font": {"family": str, "size": float, "bold": bool, "italic": bool} | None,
          "spans": [ {text, bbox, font, size, bold, italic}, ... ]  # for layout rebuild
        }
      ]
    }
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf

from app.config import get_settings
from app.services import ocr

# DPI for the per-page pixmap used only by the bg-color sampler. We don't
# need detail — just an averaged color per bbox — so 100 DPI keeps render
# time tiny (~10–30ms/page) while still putting plenty of pixels inside
# even a small text bbox.
_BG_SAMPLE_DPI = 100

# Confidence cutoff for the bg-color histogram. If the dominant color holds
# less than this share of bbox pixels, the bbox probably contains an image
# or gradient and we fall back to white. 0.25 is loose enough to handle a
# dense paragraph (where text covers ~40-50% of pixels) and tight enough
# to reject noisy bboxes.
_BG_CONFIDENCE_THRESHOLD = 0.25

log = logging.getLogger(__name__)

# PyMuPDF span "flags" bitmask: bold = bit 4, italic = bit 1.
_FLAG_ITALIC = 1 << 1
_FLAG_BOLD = 1 << 4


@dataclass(frozen=True)
class ExtractionResult:
    structured_json_path: Path
    total_pages: int
    total_blocks: int
    is_scanned: bool        # whole-doc scanned (every page below threshold)
    scanned_pages: list[int]
    plain_text_excerpt: str  # first N chars, for classification / debugging


def extract_pdf(pdf_path: Path, *, output_dir: Path) -> ExtractionResult:
    """Parse a PDF into normalized JSON, run OCR fallback where needed, and
    persist to ``output_dir/structure.json``. Returns a summary."""
    s = get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    try:
        median_size = _document_median_font_size(doc)
        pages_out: list[dict[str, Any]] = []
        scanned_pages: list[int] = []
        plain_text_chunks: list[str] = []
        total_blocks = 0

        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            page_dict = page.get_text("dict")
            # Render the page once at low DPI so the bg-color sampler can
            # detect the colored background of each text bbox (callout
            # panels, sidebars). 100 DPI is plenty for averaging colors;
            # higher would just burn time.
            page_pixmap: pymupdf.Pixmap | None = None
            try:
                page_pixmap = page.get_pixmap(dpi=_BG_SAMPLE_DPI, alpha=False)
            except Exception:
                # Don't block extraction on a render failure — fall back to
                # the white-fill behavior.
                log.exception("page %d: pixmap render failed; bg-color sampling disabled", page_index + 1)
            blocks = _normalize_blocks(
                page_dict.get("blocks", []),
                page_number=page_index + 1,
                heading_threshold=median_size * s.heading_size_multiplier
                if median_size
                else None,
                page_pixmap=page_pixmap,
                page_width=float(page.rect.width),
                page_height=float(page.rect.height),
            )
            char_count = sum(len(b["text"]) for b in blocks if b["type"] in {"paragraph", "heading"})
            is_scanned = char_count < s.scanned_page_char_threshold

            if is_scanned:
                scanned_pages.append(page_index + 1)
                ocr_block = ocr.ocr_page(
                    page,
                    page_number=page_index + 1,
                    languages=s.ocr_languages,
                    dpi=s.ocr_dpi,
                )
                if ocr_block is not None:
                    # Drop image-only blocks; OCR has captured the content.
                    blocks = [b for b in blocks if b["type"] != "image"] + [ocr_block]

            page_out = {
                "page_number": page_index + 1,
                "width": float(page.rect.width),
                "height": float(page.rect.height),
                "is_scanned": is_scanned,
                "blocks": blocks,
            }
            pages_out.append(page_out)
            total_blocks += len(blocks)

            for b in blocks:
                if b["text"]:
                    plain_text_chunks.append(b["text"])

        structured = {
            "schema_version": 1,
            "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "summary": {
                "total_pages": doc.page_count,
                "total_blocks": total_blocks,
                "scanned_pages": scanned_pages,
            },
            "pages": pages_out,
        }

        out_path = output_dir / "structure.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(structured, fh, ensure_ascii=False, indent=2)

        excerpt_limit = s.classification_excerpt_chars
        excerpt = "\n".join(plain_text_chunks)[:excerpt_limit]

        return ExtractionResult(
            structured_json_path=out_path,
            total_pages=doc.page_count,
            total_blocks=total_blocks,
            is_scanned=len(scanned_pages) == doc.page_count and doc.page_count > 0,
            scanned_pages=scanned_pages,
            plain_text_excerpt=excerpt,
        )
    finally:
        doc.close()


def _document_median_font_size(doc: pymupdf.Document) -> float | None:
    """Sample text spans across the document to compute a median font size used
    for heading detection. Returns None if the document has no text."""
    sizes: list[float] = []
    sample_pages = min(doc.page_count, 10)
    for i in range(sample_pages):
        page_dict = doc.load_page(i).get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = float(span.get("size", 0) or 0)
                    if size > 0:
                        sizes.append(size)
    return statistics.median(sizes) if sizes else None


def _normalize_blocks(
    raw_blocks: list[dict[str, Any]],
    *,
    page_number: int,
    heading_threshold: float | None,
    page_pixmap: pymupdf.Pixmap | None = None,
    page_width: float | None = None,
    page_height: float | None = None,
) -> list[dict[str, Any]]:
    """Convert PyMuPDF raw blocks into our normalized block list.

    When ``page_pixmap`` + ``page_width``/``page_height`` are provided, each
    block also gets a ``dominant_bg_color`` (#rrggbb) sampled from that
    bbox region of the rendered page. The rebuild uses this to fill the
    text-bbox with the right color (white on white pages, blue on blue
    callout panels) instead of always stamping white."""
    pix_array = _pixmap_to_rgb_array(page_pixmap) if page_pixmap is not None else None
    out: list[dict[str, Any]] = []
    for idx, block in enumerate(raw_blocks):
        block_id = f"p{page_number}-b{idx}"
        bbox = _to_bbox(block.get("bbox"))

        if block.get("type") == 1:
            out.append(
                {
                    "block_id": block_id,
                    "type": "image",
                    "text": "",
                    "bbox": bbox,
                    "dominant_font": None,
                    "spans": [],
                }
            )
            continue

        spans_out: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for line in block.get("lines", []):
            line_texts: list[str] = []
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text:
                    continue
                flags = int(span.get("flags", 0) or 0)
                # PyMuPDF encodes span color as a packed int (0xRRGGBB). White
                # text on a colored panel is common in formatted docs; if we
                # don't capture it, every translation renders as default black
                # and the colored callouts look broken.
                color_int = int(span.get("color", 0) or 0)
                color_hex = f"#{color_int & 0xFFFFFF:06x}"
                spans_out.append(
                    {
                        "text": text,
                        "bbox": _to_bbox(span.get("bbox")),
                        "font": span.get("font") or "",
                        "size": float(span.get("size", 0) or 0),
                        "bold": bool(flags & _FLAG_BOLD),
                        "italic": bool(flags & _FLAG_ITALIC),
                        "color": color_hex,
                    }
                )
                line_texts.append(text)
            if line_texts:
                text_parts.append("".join(line_texts))

        full_text = "\n".join(text_parts).strip()
        if not full_text and not spans_out:
            continue

        dominant = _dominant_font(spans_out)
        dominant_color = _dominant_color(spans_out)
        dominant_bg = _dominant_bg_color(
            bbox=bbox,
            pix_array=pix_array,
            page_width=page_width,
            page_height=page_height,
        )
        block_type = "paragraph"
        if (
            heading_threshold is not None
            and dominant is not None
            and dominant["size"] >= heading_threshold
        ):
            block_type = "heading"

        out.append(
            {
                "block_id": block_id,
                "type": block_type,
                "text": full_text,
                "bbox": bbox,
                "dominant_font": dominant,
                "dominant_color": dominant_color,
                "dominant_bg_color": dominant_bg,
                "spans": spans_out,
            }
        )
    return out


def _pixmap_to_rgb_array(pix: pymupdf.Pixmap) -> np.ndarray | None:
    """Materialize the page pixmap as an (H, W, 3) uint8 numpy array.

    Returns None if the pixmap isn't 3- or 4-channel (e.g. CMYK source);
    the bg sampler then short-circuits to None (→ white fill at rebuild)."""
    n = pix.n
    if n not in (3, 4):
        return None
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = arr.reshape(pix.height, pix.width, n)
    if n == 4:
        arr = arr[:, :, :3]  # drop alpha
    return arr


def _dominant_bg_color(
    *,
    bbox: tuple[float, float, float, float] | None,
    pix_array: np.ndarray | None,
    page_width: float | None,
    page_height: float | None,
) -> str | None:
    """Sample the dominant background color of a text bbox from the rendered
    page. Returns ``#rrggbb`` or None when sampling can't yield a confident
    answer (rebuild then falls back to white).

    Algorithm:
      1. Map bbox PDF coords → pixmap pixel coords by scale factor.
      2. Quantize each pixel to 8 levels per channel (3 bits per channel,
         512 bins total). Anti-aliased glyph-edge pixels collapse into
         the same bins as the surrounding background, so the bg color
         dominates the histogram even when text covers ~half the bbox.
      3. The most common bin wins. We require it to hold ≥25% of pixels
         (defensive against bboxes containing images/gradients) — otherwise
         return None and let the rebuild use white.
    """
    if pix_array is None or bbox is None:
        return None
    if page_width is None or page_height is None or page_width <= 0 or page_height <= 0:
        return None

    h, w, _ = pix_array.shape
    scale_x = w / page_width
    scale_y = h / page_height
    # floor() lower bounds + ceil() upper bounds so the sample covers every
    # pixel the bbox actually touches. Zero-area bboxes collapse to None.
    x0 = max(0, int(bbox[0] * scale_x))
    y0 = max(0, int(bbox[1] * scale_y))
    x1 = min(w, int(np.ceil(bbox[2] * scale_x)))
    y1 = min(h, int(np.ceil(bbox[3] * scale_y)))
    if x1 <= x0 or y1 <= y0:
        return None

    region = pix_array[y0:y1, x0:x1, :]
    if region.size == 0:
        return None

    # Quantize: top 4 bits per channel → 4096 bins. The histogram tells us
    # which BIN the bg lives in (clustering AA edge pixels together with
    # the true background).
    quantized = region >> 4  # uint8 values 0..15
    keys = (
        quantized[..., 0].astype(np.int32) * 256
        + quantized[..., 1].astype(np.int32) * 16
        + quantized[..., 2].astype(np.int32)
    )
    keys_flat = keys.ravel()
    counts = np.bincount(keys_flat, minlength=4096)
    total = int(counts.sum())
    if total == 0:
        return None
    mode_key = int(counts.argmax())
    mode_count = int(counts[mode_key])
    if mode_count / total < _BG_CONFIDENCE_THRESHOLD:
        return None

    # Reconstruct the color from the ACTUAL pixels in the dominant bin —
    # not the bin center. Bin-center math (q*16+8) maxed out at 248 even
    # for pure-white pixels (255, 255, 255), producing #f8f8f8 fill on
    # pages whose true background is #ffffff. The 7-unit difference
    # showed up as faint gray rectangles around every text block on plain
    # white pages. Median of bin pixels gives the exact original color.
    mask = keys_flat == mode_key
    matching = region.reshape(-1, region.shape[-1])[mask]
    if matching.size == 0:
        return None
    r = int(np.median(matching[:, 0]))
    g = int(np.median(matching[:, 1]))
    b = int(np.median(matching[:, 2]))
    return f"#{r:02x}{g:02x}{b:02x}"


def _dominant_color(spans: list[dict[str, Any]]) -> str | None:
    """Pick the color covering the most characters in the block. Used by the
    rebuilder to render the translated text in the original color (e.g.
    white callout text on a colored panel)."""
    if not spans:
        return None
    weighted: Counter[str] = Counter()
    for sp in spans:
        color = sp.get("color")
        if not color:
            continue
        weighted[color] += len(sp.get("text", ""))
    if not weighted:
        return None
    return weighted.most_common(1)[0][0]


def _dominant_font(spans: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not spans:
        return None
    # Pick the font/size combo covering the most characters in the block.
    by_key: Counter[tuple[str, float, bool, bool]] = Counter()
    for sp in spans:
        key = (sp["font"], round(sp["size"], 2), sp["bold"], sp["italic"])
        by_key[key] += len(sp["text"])
    family, size, bold, italic = by_key.most_common(1)[0][0]
    return {"family": family, "size": float(size), "bold": bool(bold), "italic": bool(italic)}


def _to_bbox(raw: Any) -> list[float] | None:
    if not raw:
        return None
    try:
        return [float(v) for v in raw]
    except (TypeError, ValueError):
        return None

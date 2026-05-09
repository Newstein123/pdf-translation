"""Layout-mode + book-metadata auto-detection for the M14 export pipeline.

Two related responsibilities:

  1. ``detect_book_metadata(pdf_path)`` — read the PDF's ``doc.metadata`` for
     title and author, with a filename fallback for title. Used at upload
     time to pre-fill the job's ``title_source`` and ``author_source``.

  2. ``detect_layout_mode(pdf_path)`` — sample a handful of pages and decide
     whether the document is "plain white-paper text" (TEMPLATE candidate)
     or "has color / imagery / decorative formatting" (PRESERVE candidate).
     Used by the AWAITING_CONFIRMATION step to recommend a default; the
     user can always override.

Both functions are deterministic and cheap. They don't call Gemini.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.config import get_settings
from app.db.models import ExportLayoutMode

log = logging.getLogger(__name__)

# Sample a small number of pages — these heuristics need a representative
# sample, not exhaustive coverage. 6 pages catches most "color page tucked
# in chapter 4" cases without slowing upload.
_DEFAULT_SAMPLE_PAGE_COUNT = 6
# DPI for the white-pixel sampling. We don't need quality, just histograms.
_SAMPLE_DPI = 72
# Pixel value above which a channel counts as "near-white". 240 catches
# anti-aliased edges around black text without falsely flagging slightly
# tinted backgrounds (which we WANT to detect as color).
_WHITE_CHANNEL_THRESHOLD = 240


# ---------------------------------------------------------------------------
# Title / author auto-detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectedBookMetadata:
    title: str | None
    author: str | None


def detect_book_metadata(pdf_path: Path) -> DetectedBookMetadata:
    """Read PDF metadata for title and author; fall back to filename for title.

    Returns ``DetectedBookMetadata`` with whichever fields could be inferred.
    Both fields are normalized: stripped, internal whitespace collapsed,
    obviously-junk values rejected (e.g. "untitled", "Microsoft Word", a
    bare path, a UUID).
    """
    try:
        doc = pymupdf.open(pdf_path)
    except Exception:
        log.exception("pymupdf open failed for metadata detection: %s", pdf_path)
        return DetectedBookMetadata(title=None, author=None)
    try:
        meta = doc.metadata or {}
    finally:
        doc.close()

    title = _clean_metadata_value(meta.get("title")) or _title_from_filename(pdf_path)
    author = _clean_metadata_value(meta.get("author"))
    return DetectedBookMetadata(title=title, author=author)


_JUNK_TITLE_PATTERNS = re.compile(
    r"^("
    r"untitled|microsoft word|word document|document\d*|"
    r"[a-f0-9-]{30,}|"  # UUIDs / hashes
    r"page \d+|cover|"
    r"output\b|.*\.(pdf|docx?|odt)$"
    r")$",
    re.IGNORECASE,
)
# Some PDF authoring tools store the document UUID with spaces between
# segments instead of hyphens (e.g. "2931B8B2 35A7 4139 A0F8 B30C3F78B0C0").
# Normalize-and-test against the same pattern so those slip through too.
_JUNK_TITLE_NORMALIZED_PATTERN = re.compile(
    r"^[A-F0-9]{4,}(\s+[A-F0-9]{4,}){2,}$", re.IGNORECASE
)


def _clean_metadata_value(raw: str | None) -> str | None:
    if not raw:
        return None
    cleaned = " ".join(str(raw).split()).strip()
    if not cleaned:
        return None
    # Reject obvious junk values that PDF authoring tools leave behind.
    if _JUNK_TITLE_PATTERNS.match(cleaned):
        return None
    if _JUNK_TITLE_NORMALIZED_PATTERN.match(cleaned):
        return None
    if len(cleaned) > 500:  # spam title
        return None
    return cleaned


def _title_from_filename(pdf_path: Path) -> str | None:
    stem = pdf_path.stem
    # Strip trailing timestamp / id segments common in scraped uploads
    # (e.g. "the-time-machine-1739810135861" → "the-time-machine").
    stem = re.sub(r"-\d{6,}$", "", stem)
    # Replace separators with spaces, collapse whitespace.
    stem = re.sub(r"[_-]+", " ", stem).strip()
    if not stem:
        return None
    # Title-case if all lowercase, leave alone otherwise (preserves Burmese).
    if stem == stem.lower() and stem.isascii():
        stem = stem.title()
    return stem


# ---------------------------------------------------------------------------
# Layout mode auto-detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayoutDetection:
    recommended_mode: ExportLayoutMode
    is_white_book: bool
    pages_sampled: int
    pages_with_image: int
    avg_white_pixel_ratio: float
    reason: str


def detect_layout_mode(
    pdf_path: Path, *, sample_count: int = _DEFAULT_SAMPLE_PAGE_COUNT
) -> LayoutDetection:
    """Recommend TEMPLATE or PRESERVE based on color/imagery sampling.

    Heuristic:
      * If any sampled page contains an embedded image → PRESERVE
        (figures, diagrams, photos all matter for the reader).
      * Otherwise compute the per-page ratio of near-white pixels. If the
        average across sampled pages is ≥ ``settings.auto_white_page_pixel_threshold``
        → TEMPLATE (it's clean text on white).
      * Otherwise PRESERVE (colored backgrounds, callouts, sidebars).

    Cheap: renders pages at 72 DPI, runs a histogram, doesn't load the
    full document. Defensive against malformed PDFs — any failure returns
    PRESERVE so we never accidentally erase color content with a template.
    """
    settings = get_settings()
    threshold = settings.auto_white_page_pixel_threshold

    try:
        doc = pymupdf.open(pdf_path)
    except Exception:
        log.exception("pymupdf open failed for layout detection: %s", pdf_path)
        return LayoutDetection(
            recommended_mode=ExportLayoutMode.PRESERVE,
            is_white_book=False,
            pages_sampled=0,
            pages_with_image=0,
            avg_white_pixel_ratio=0.0,
            reason="pdf open failed; defaulting to PRESERVE",
        )

    try:
        n = doc.page_count
        if n == 0:
            return LayoutDetection(
                recommended_mode=ExportLayoutMode.PRESERVE,
                is_white_book=False,
                pages_sampled=0,
                pages_with_image=0,
                avg_white_pixel_ratio=0.0,
                reason="empty document",
            )

        sample_indices = _stratified_sample_indices(n, sample_count)
        white_ratios: list[float] = []
        pages_with_image = 0

        for idx in sample_indices:
            page = doc.load_page(idx)
            if _page_has_image(page):
                pages_with_image += 1
                continue
            ratio = _page_white_pixel_ratio(page)
            white_ratios.append(ratio)
    finally:
        doc.close()

    avg_white = (
        sum(white_ratios) / len(white_ratios) if white_ratios else 0.0
    )

    if pages_with_image > 0:
        return LayoutDetection(
            recommended_mode=ExportLayoutMode.PRESERVE,
            is_white_book=False,
            pages_sampled=len(sample_indices),
            pages_with_image=pages_with_image,
            avg_white_pixel_ratio=avg_white,
            reason=(
                f"{pages_with_image} of {len(sample_indices)} sampled pages "
                "contain images — PRESERVE keeps figures intact"
            ),
        )

    if avg_white >= threshold:
        return LayoutDetection(
            recommended_mode=ExportLayoutMode.TEMPLATE,
            is_white_book=True,
            pages_sampled=len(sample_indices),
            pages_with_image=0,
            avg_white_pixel_ratio=avg_white,
            reason=(
                f"avg white-pixel ratio {avg_white:.3f} ≥ {threshold:.3f}, "
                "no images — TEMPLATE produces clean Burmese typography"
            ),
        )

    return LayoutDetection(
        recommended_mode=ExportLayoutMode.PRESERVE,
        is_white_book=False,
        pages_sampled=len(sample_indices),
        pages_with_image=0,
        avg_white_pixel_ratio=avg_white,
        reason=(
            f"avg white-pixel ratio {avg_white:.3f} < {threshold:.3f}, "
            "colored / decorated content present — PRESERVE keeps it intact"
        ),
    )


def _stratified_sample_indices(page_count: int, want: int) -> list[int]:
    """Even spread across the document so we don't sample only the first 6."""
    if page_count <= want:
        return list(range(page_count))
    stride = page_count / want
    return sorted({int(i * stride) for i in range(want)})


def _page_has_image(page: pymupdf.Page) -> bool:
    try:
        return bool(page.get_images(full=False))
    except Exception:
        # Defensive: failures here shouldn't drag the whole detection down.
        return False


def _page_white_pixel_ratio(page: pymupdf.Page) -> float:
    """Fraction of pixels that are "near-white" on all three RGB channels.

    1.0 = entirely white, 0.0 = entirely non-white. Returns 1.0 (i.e. treats
    as white) on render failure so a corrupt page doesn't push a clean book
    toward PRESERVE.
    """
    try:
        pix = page.get_pixmap(dpi=_SAMPLE_DPI, alpha=False)
    except Exception:
        return 1.0
    samples = pix.samples
    n_pixels = pix.width * pix.height
    if n_pixels <= 0:
        return 1.0
    # samples is a bytes object of length width * height * n_components.
    # For RGB (n=3) we count pixels where every channel is near-white.
    n = pix.n
    if n < 3:
        # Grayscale: single channel.
        white = sum(1 for v in samples if v >= _WHITE_CHANNEL_THRESHOLD)
        return white / n_pixels
    white = 0
    # Iterate in steps of n; check the first 3 channels.
    s = bytes(samples)
    threshold = _WHITE_CHANNEL_THRESHOLD
    for i in range(0, len(s), n):
        if s[i] >= threshold and s[i + 1] >= threshold and s[i + 2] >= threshold:
            white += 1
    return white / n_pixels

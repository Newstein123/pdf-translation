"""Layout-preserving PDF rebuild using HarfBuzz-shaped text.

Strategy:
  1. Open the original PDF.
  2. For each page, redact each text bbox with its DETECTED background
     color (sampled at extraction time from a rendered pixmap). On a plain
     white page this is white — same as before. On a colored callout
     panel, this is the panel color, so the decoration survives intact
     instead of getting stamped with white. Images and vector line-art
     are preserved via the ``*_NONE`` flags.
  3. Render the translated text into the same bbox via
     ``page.insert_htmlbox`` with an embedded Burmese font and the
     original text color (sampled from spans during extraction).
     ``insert_htmlbox`` uses an internal HTML/CSS engine with proper
     complex-script shaping (stacked consonants, reordered vowels,
     ligatures) — ``insert_textbox`` does NOT and produces broken Burmese
     glyphs.
  4. Auto-shrink font size if the translation overflows; expand the bbox
     vertically as a last resort to avoid dropping content.
  5. Save as a new PDF under ``storage/outputs/<job_id>/<basename>.my.pdf``.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from dataclasses import dataclass
from html import escape
from pathlib import Path

import pymupdf

from app.config import get_settings
from app.utils.fonts import get_burmese_fonts

log = logging.getLogger(__name__)

_DEFAULT_FONT_SIZE = 11.0
_MIN_FONT_SIZE = 6.0
# Tables and other tight-cell bboxes need to go smaller because Burmese
# tokenizes ~3× source — at 6pt with 1.6 line-height, a table cell that
# held two English lines can't even hold one Burmese line.
_MIN_FONT_SIZE_TIGHT = 4.0
# Bboxes shorter than this (in PDF points) are treated as "tight" — table
# cells, badge labels, etc. — and use _MIN_FONT_SIZE_TIGHT plus a much
# tighter line-height.
_TIGHT_BBOX_HEIGHT = 30.0
_FONT_SHRINK_STEP = 0.5
_MAX_VERTICAL_EXPANSION = 4.0  # multiples of original block height


@dataclass(frozen=True)
class RebuildResult:
    output_path: Path
    pages_written: int
    blocks_placed: int
    overflow_blocks: int


def rebuild_pdf(
    *,
    source_pdf: Path,
    structure_path: Path,
    translations_by_block: dict[str, str],
    output_dir: Path,
    output_name: str,
    skip_pages: set[int] | None = None,
    cover_image_path: Path | str | None = None,
) -> RebuildResult:
    """Layout-preserving rebuild — redact source text bboxes, fill with
    translated text, copy images / tables / non-text untouched.

    ``skip_pages`` (M15): set of 1-indexed page numbers to drop entirely
    from the output. We render the surviving pages first, then delete
    the skipped pages from the doc, so the output is shorter than the
    source by ``len(skip_pages)`` pages.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / output_name
    skip = skip_pages or set()

    fonts = get_burmese_fonts()
    archive: pymupdf.Archive | None = None
    css = _build_css(fonts, archive_ref=False)
    if fonts.regular_path is not None:
        archive = pymupdf.Archive()
        archive.add(str(fonts.regular_path), fonts.regular_path.name)
        if fonts.bold_path is not None and fonts.bold_path != fonts.regular_path:
            archive.add(str(fonts.bold_path), fonts.bold_path.name)
        css = _build_css(fonts, archive_ref=True)

    with structure_path.open("r", encoding="utf-8") as fh:
        structure = json.load(fh)

    doc = pymupdf.open(source_pdf)
    pages_written = doc.page_count
    blocks_placed = 0
    overflow = 0
    try:
        for page_idx, page_data in enumerate(structure.get("pages", [])):
            if page_idx >= pages_written:
                break
            # M15: pages on the skip list don't get text rendered into them.
            # We delete them after the render loop in one pass (deleting
            # mid-loop would shift the indices of pages we still need).
            page_no = int(page_data.get("page_number", page_idx + 1))
            if page_no in skip:
                continue
            page = doc.load_page(page_idx)
            translatable = [
                b for b in page_data.get("blocks", [])
                if b.get("type") in {"paragraph", "heading", "ocr"}
                and translations_by_block.get(b.get("block_id", ""))
            ]
            if not translatable:
                continue

            # 1. Redact original text regions. We use the SPAN bboxes
            #    (per-line tight rectangles) instead of the wider BLOCK
            #    bbox so that decorative icons sitting in the gutter that
            #    text wraps around survive untouched. Block bbox is the
            #    union of all spans — when text wraps around an image, the
            #    block bbox includes the image area, and a block-level
            #    colored fill would paint over the image. Span-level
            #    redaction draws the fill only where actual text glyphs
            #    are, leaving inter-span whitespace (where icons sit) alone.
            for block in translatable:
                fill_rgb = _parse_hex_rgb(block.get("dominant_bg_color"))
                spans = block.get("spans") or []
                if spans:
                    for span in spans:
                        sb = span.get("bbox")
                        if not sb:
                            continue
                        srect = pymupdf.Rect(sb)
                        if srect.is_empty or srect.is_infinite:
                            continue
                        page.add_redact_annot(srect, fill=fill_rgb)
                else:
                    # Fallback for blocks without span data (OCR blocks etc.).
                    bbox = block.get("bbox")
                    if not bbox:
                        continue
                    rect = pymupdf.Rect(bbox)
                    if rect.is_empty or rect.is_infinite:
                        continue
                    page.add_redact_annot(rect, fill=fill_rgb)
            page.apply_redactions(
                images=pymupdf.PDF_REDACT_IMAGE_NONE,
                graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
            )

            # 2. Place translated text via insert_htmlbox (HarfBuzz-shaped),
            #    in the original text color so light text on dark panels
            #    keeps reading correctly.
            for block in translatable:
                block_id = block.get("block_id")
                bbox = block.get("bbox")
                if not block_id or not bbox:
                    continue
                # Use the icon-aware rect so wrap-around-icon layouts (e.g.
                # the hourglass next to "Time Travel" body text) don't get
                # painted over by the new translated text. Falls through to
                # the original block bbox when no icon pattern is detected.
                rect = _icon_aware_insert_rect(bbox, block.get("spans"))
                translated = translations_by_block.get(block_id, "").strip()
                if not translated:
                    continue
                base_size = _starting_font_size(block)
                is_heading = block.get("type") == "heading"
                color_hex = block.get("dominant_color") or "#000000"
                fitted = _fit_html(
                    page,
                    rect,
                    translated,
                    base_size=base_size,
                    is_heading=is_heading,
                    color=color_hex,
                    css=css,
                    archive=archive,
                )
                if not fitted:
                    overflow += 1
                blocks_placed += 1

        # M15: drop user-skipped pages from the output. Done after the
        # render loop so deleting one page doesn't shift the indices of
        # pages we still need to render. PyMuPDF accepts a list and
        # deletes in descending order internally.
        if skip:
            page_indices_to_delete = sorted(
                {p - 1 for p in skip if 0 < p <= doc.page_count}, reverse=True
            )
            for idx in page_indices_to_delete:
                doc.delete_page(idx)
            pages_written = doc.page_count

        # M16: prepend a cover page. We aspect-fit the image into a page
        # the same size as page 1 of the source so the cover doesn't
        # look out of proportion next to the body. Failure is non-fatal
        # — the rest of the rebuild already succeeded.
        if cover_image_path:
            try:
                _prepend_cover_page(doc, Path(cover_image_path))
                pages_written = doc.page_count
            except Exception:
                log.exception("cover render failed; saving PDF without cover")

        doc.save(out_path, garbage=4, deflate=True, clean=True)
    finally:
        doc.close()

    return RebuildResult(
        output_path=out_path,
        pages_written=pages_written,
        blocks_placed=blocks_placed,
        overflow_blocks=overflow,
    )


def _prepend_cover_page(doc: pymupdf.Document, image_path: Path) -> None:
    """Insert a new page at index 0 with the cover image aspect-fitted to
    match the dimensions of the existing page 1. Used by both PRESERVE
    and (indirectly) TEMPLATE export paths to brand the output."""
    from PIL import Image as _Image

    if doc.page_count == 0:
        # Nothing to size against — skip silently.
        return
    page1 = doc.load_page(0)
    page_w = float(page1.rect.width)
    page_h = float(page1.rect.height)

    with _Image.open(image_path) as im:
        img_w, img_h = im.size
    if img_w <= 0 or img_h <= 0:
        return

    cover_page = doc.new_page(pno=0, width=page_w, height=page_h)
    scale = min(page_w / img_w, page_h / img_h)
    target_w = img_w * scale
    target_h = img_h * scale
    x0 = (page_w - target_w) / 2
    y0 = (page_h - target_h) / 2
    cover_page.insert_image(
        pymupdf.Rect(x0, y0, x0 + target_w, y0 + target_h),
        filename=str(image_path),
    )


def _parse_hex_rgb(hex_color: str | None) -> tuple[float, float, float]:
    """Convert ``#rrggbb`` into a (r, g, b) tuple of 0–1 floats for PyMuPDF.
    Falls back to white when the input is missing or malformed — old
    structure.json files (extracted before bg-sampling existed) get the
    same white-fill behavior they had originally."""
    if not hex_color or not isinstance(hex_color, str):
        return (1.0, 1.0, 1.0)
    s = hex_color.strip().lstrip("#")
    if len(s) != 6:
        return (1.0, 1.0, 1.0)
    try:
        r = int(s[0:2], 16) / 255.0
        g = int(s[2:4], 16) / 255.0
        b = int(s[4:6], 16) / 255.0
    except ValueError:
        return (1.0, 1.0, 1.0)
    return (r, g, b)


def _icon_aware_insert_rect(
    block_bbox: tuple[float, float, float, float],
    spans: list[dict] | None,
    *,
    threshold: float = 15.0,
    min_consecutive: int = 2,
) -> pymupdf.Rect:
    """Compute an insert rect that respects wrap-around-icon layouts.

    PyMuPDF reports a block bbox = the union of every span. When the
    original text wrapped around a left- or right-aligned icon at the top
    of the block, that icon's column is included in the bbox even though
    no span covers it. ``insert_htmlbox`` does plain rectangular layout,
    so the new (translated) text would draw OVER the icon area unless we
    shift the insert rect inward.

    Detection (conservative):
      * Find the "common" left/right of mid-block spans (where the icon, if
        any, has ended and text is at full width).
      * If the first ``min_consecutive`` spans are indented past the common
        left by ``threshold`` pt or more, treat that as a top-left icon and
        shift the rect's left edge to the indented spans' min x.
      * Symmetric check for top-right icons (top spans truncated short of
        the common right).

    Returns the original bbox unchanged when no clear icon pattern exists,
    so it's safe to call on any block — plain prose passes through.
    """
    rect = pymupdf.Rect(*block_bbox)
    if not spans or len(spans) < 4:
        return rect

    sorted_spans = sorted(spans, key=lambda s: (s.get("bbox") or [0, 0, 0, 0])[1])
    if len(sorted_spans) < 4:
        return rect

    # Use mid-block spans (skip the first quarter and last quarter) to
    # learn the "regular" text column. This avoids picking up the icon-
    # adjacent spans we're trying to detect.
    mid_lo = len(sorted_spans) // 4
    mid_hi = max(mid_lo + 1, (3 * len(sorted_spans)) // 4)
    middle_spans = sorted_spans[mid_lo:mid_hi]
    if not middle_spans:
        return rect

    common_x0 = min((s.get("bbox") or [0, 0, 0, 0])[0] for s in middle_spans)
    common_x1 = max((s.get("bbox") or [0, 0, 0, 0])[2] for s in middle_spans)

    new_x0 = rect.x0
    new_x1 = rect.x1

    # Top-left icon: consecutive starting spans indented past common_x0.
    consecutive_indented = 0
    for s in sorted_spans:
        x0 = (s.get("bbox") or [0, 0, 0, 0])[0]
        if x0 - common_x0 > threshold:
            consecutive_indented += 1
        else:
            break
    if consecutive_indented >= min_consecutive:
        indented_min = min(
            (sorted_spans[i].get("bbox") or [0, 0, 0, 0])[0]
            for i in range(consecutive_indented)
        )
        new_x0 = max(new_x0, indented_min)

    # Top-right icon: consecutive starting spans truncated short of common_x1.
    consecutive_short = 0
    for s in sorted_spans:
        x1 = (s.get("bbox") or [0, 0, 0, 0])[2]
        if common_x1 - x1 > threshold:
            consecutive_short += 1
        else:
            break
    if consecutive_short >= min_consecutive:
        short_max = max(
            (sorted_spans[i].get("bbox") or [0, 0, 0, 0])[2]
            for i in range(consecutive_short)
        )
        new_x1 = min(new_x1, short_max)

    # Sanity: if both adjustments collapsed the rect to nothing, give up
    # and return the original — better to slightly cover an icon than to
    # render text into a 0-width column.
    if new_x1 - new_x0 < 30:
        return rect
    return pymupdf.Rect(new_x0, rect.y0, new_x1, rect.y1)


def _starting_font_size(block: dict) -> float:
    dominant = block.get("dominant_font") or {}
    size = float(dominant.get("size") or 0) or _DEFAULT_FONT_SIZE
    return min(max(size, _MIN_FONT_SIZE), 48.0)


def _build_html(
    text: str,
    *,
    font_size: float,
    is_heading: bool,
    color: str = "#000000",
    line_height: float | None = None,
) -> str:
    tag = "h2" if is_heading else "p"
    # M10: NFC normalize so canonically-equivalent Burmese sequences (decomposed
    # vs precomposed vowel/medial combinations) shape consistently. Models
    # sometimes return decomposed forms even when sources are precomposed.
    normalized = unicodedata.normalize("NFC", text)
    body = escape(normalized).replace("\n", "<br/>")
    if line_height is None:
        line_height = get_settings().mm_line_height
    return (
        f'<{tag} style="margin:0; padding:0; font-size:{font_size:.1f}pt; '
        f'line-height:{line_height:.2f}; color:{escape(color)};">{body}</{tag}>'
    )


def _build_css(fonts, *, archive_ref: bool) -> str:
    """CSS injected into every insert_htmlbox call.

    When we have a bundled Burmese font, expose it as ``@font-face`` and set
    it as the default. ``insert_htmlbox`` resolves the ``url(...)`` against the
    ``Archive`` we pass alongside the call.

    ``word-break: break-all`` + ``overflow-wrap: anywhere`` are CRITICAL for
    Burmese: the script doesn't use spaces between words, so the renderer's
    default (break only at whitespace) keeps long sentences as one
    monolithic "word" that overflows tight bboxes (table cells, badges).
    With these rules the renderer wraps at any glyph boundary so the text
    fits inside the cell instead of falling outside.
    """
    common = (
        "* { word-break: break-all; overflow-wrap: anywhere; "
        "white-space: normal; hyphens: none; }"
    )
    if not archive_ref or fonts.regular_path is None:
        return f"* {{ font-family: sans-serif; }} {common}"
    family = fonts.family_name
    regular = fonts.regular_path.name
    bold = (fonts.bold_path or fonts.regular_path).name
    return (
        f'@font-face {{ font-family: "{family}"; '
        f'src: url("{regular}"); font-weight: normal; }}\n'
        f'@font-face {{ font-family: "{family}"; '
        f'src: url("{bold}"); font-weight: bold; }}\n'
        f'* {{ font-family: "{family}", sans-serif; }}\n'
        f"{common}"
    )


def _fit_html(
    page: pymupdf.Page,
    rect: pymupdf.Rect,
    text: str,
    *,
    base_size: float,
    is_heading: bool,
    color: str,
    css: str,
    archive: pymupdf.Archive | None,
) -> bool:
    """Render ``text`` into ``rect`` shrinking font until it fits.

    ``insert_htmlbox`` returns a tuple ``(spare_height, scale)``. ``spare_height``
    is positive when the text fits and negative when it overflows. We shrink
    the size until ``spare_height >= 0``, expanding the rect downward if even
    the minimum size won't fit.

    Bboxes shorter than ``_TIGHT_BBOX_HEIGHT`` (table cells, small badges)
    use a tighter line-height and a lower min font, because at default
    settings Burmese can't fit even a single line in a table cell."""
    bbox_h = rect.y1 - rect.y0
    is_tight = bbox_h < _TIGHT_BBOX_HEIGHT
    min_size = _MIN_FONT_SIZE_TIGHT if is_tight else _MIN_FONT_SIZE
    line_height = 1.05 if is_tight else None  # None → uses settings.mm_line_height (1.6)

    size = base_size
    while size >= min_size:
        html = _build_html(
            text,
            font_size=size,
            is_heading=is_heading,
            color=color,
            line_height=line_height,
        )
        spare, _scale = page.insert_htmlbox(rect, html, css=css, archive=archive)
        if spare >= 0:
            return True
        size -= _FONT_SHRINK_STEP

    page_h = float(page.rect.height)
    base_h = rect.y1 - rect.y0
    target_h = min(base_h * (1 + _MAX_VERTICAL_EXPANSION), page_h - rect.y0)
    expanded = pymupdf.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + target_h)
    html = _build_html(
        text,
        font_size=min_size,
        is_heading=is_heading,
        color=color,
        line_height=line_height,
    )
    spare, _ = page.insert_htmlbox(expanded, html, css=css, archive=archive)
    return spare >= 0

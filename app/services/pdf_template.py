"""Clean Burmese book PDF template (M14 — "TEMPLATE" export mode).

Renders translated text through a configurable book template — fixed page
geometry (A5 by default), embedded Burmese font, justified body with
first-line indent, running header (book title + chapter), page numbers,
chapter title styling. Used for plain-prose books; the layout-preserving
``rebuild_pdf`` path covers textbooks and illustrated content.

Public surface:

  * ``render_template_pdf(...)`` — produces the output PDF.
  * ``TemplateConfig`` — frozen dataclass of resolved template settings;
    built from global defaults in ``settings`` and merged with per-job
    overrides on ``Job.export_template_overrides_json``.

Implementation choice: **PyMuPDF Story API**. ReportLab — which the
earlier version used — does not apply OpenType GSUB/GPOS rules required
by Myanmar script, so consonant clusters and stacked diacritics rendered
incorrectly even when the right font was embedded. PyMuPDF uses HarfBuzz
under the hood and shapes Burmese correctly, the same way the
layout-preserving ``rebuild_pdf`` already does. See the side-by-side
A/B test we ran when diagnosing the bug — ``ပြန်လည်`` vs ``ပ¨ြ¨လည်``.

Layout strategy:
  1. Build an HTML document for the whole book (body + chapter headings).
  2. Use ``Story.place()`` to flow content across pages of fixed geometry.
  3. After each page is laid out, scan it for chapter heading text so we
     can update the running-header state for the *next* page.
  4. After the body is fully rendered, do a second pass to draw running
     headers and footers on every page (they live outside the Story's
     content rectangle).

The Auto-TOC feature from the ReportLab version is dropped for now —
PyMuPDF Story doesn't have a TableOfContents flowable. Re-adding it
later is straightforward (render body first → record chapter pages →
render TOC as separate Story → prepend pages).
"""
from __future__ import annotations

import html as html_lib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf

from app.config import get_settings
from app.utils.fonts import get_burmese_fonts

log = logging.getLogger(__name__)

# Millimeters → points helper (PyMuPDF uses points natively).
_MM_TO_PT = 72.0 / 25.4


# ---------------------------------------------------------------------------
# Result + config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemplateRenderResult:
    output_path: Path
    pages_written: int
    chapters_rendered: int
    blocks_placed: int


@dataclass(frozen=True)
class TemplateConfig:
    """Resolved template settings — global defaults merged with per-job overrides."""

    page_size: str = "A5"
    margin_mm: float = 18.0
    body_font: str = "Noto Sans Myanmar"
    body_font_size_pt: float = 11.0
    line_height: float = 1.6
    first_line_indent_mm: float = 6.0
    justify: bool = True
    page_number_style: str = "page"
    running_header: str = "title_and_chapter"
    header_alignment: str = "alternating"
    chapter_style: str = "large_centered"
    section_break: str = "stars"
    title_source: str | None = None
    title_target: str | None = None
    author_source: str | None = None
    author_target: str | None = None

    @classmethod
    def from_settings(
        cls,
        *,
        overrides: dict[str, Any] | None = None,
        title_source: str | None = None,
        title_target: str | None = None,
        author_source: str | None = None,
        author_target: str | None = None,
    ) -> "TemplateConfig":
        s = get_settings()
        base = {
            "page_size": s.template_page_size,
            "margin_mm": s.template_margin_mm,
            "body_font": s.template_body_font,
            "body_font_size_pt": s.template_body_font_size_pt,
            "line_height": s.template_line_height,
            "first_line_indent_mm": s.template_first_line_indent_mm,
            "justify": s.template_justify,
            "page_number_style": s.template_page_number_style,
            "running_header": s.template_running_header,
            "header_alignment": s.template_header_alignment,
            "chapter_style": s.template_chapter_style,
            "section_break": s.template_section_break,
        }
        if overrides:
            for k, v in overrides.items():
                if k in base and v is not None:
                    base[k] = v
        return cls(
            **base,
            title_source=title_source,
            title_target=title_target,
            author_source=author_source,
            author_target=author_target,
        )


# ---------------------------------------------------------------------------
# Source-block ingestion
# ---------------------------------------------------------------------------


@dataclass
class _RenderBlock:
    kind: str  # "heading" | "body"
    text: str = ""
    page_number_in_source: int | None = None


_TOC_HEADING_HINTS = ("contents", "မာတိကာ", "မာတိ ကာ")
_TOC_NUMERIC_LINE_RE = re.compile(r"^[\s\d၀-၉.\-–—]+$")


def _looks_like_toc_entry(text: str) -> bool:
    if not text:
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    return all(_TOC_NUMERIC_LINE_RE.match(ln) for ln in lines)


def _is_toc_heading(text: str) -> bool:
    t = text.strip().lower()
    return any(hint in t for hint in _TOC_HEADING_HINTS)


def _load_render_blocks(
    structure_path: Path,
    translations_by_block: dict[str, str],
    *,
    skip_pages: set[int] | None = None,
) -> list[_RenderBlock]:
    """Walk structure.json in page order; emit (heading | body) blocks.

    Filters dropped:
      * Source TOC ("Contents" / ``မာတိကာ``) heading + numeric paragraphs
        that follow it. The template can regenerate a TOC if needed; the
        source's page numbers don't apply to the re-flowed output.
      * Bare page-number paragraphs (single Burmese / Arabic digits).
      * (M15) Pages whose 1-indexed number is in ``skip_pages``.
    """
    skip = skip_pages or set()
    with structure_path.open("r", encoding="utf-8") as fh:
        structure = json.load(fh)

    out: list[_RenderBlock] = []
    in_toc = False
    for page in structure.get("pages") or []:
        page_number = page.get("page_number")
        if page_number in skip:
            continue
        for block in page.get("blocks") or []:
            block_type = block.get("type")
            if block_type not in {"paragraph", "heading", "ocr"}:
                continue
            block_id = block.get("block_id")
            if not block_id:
                continue
            translated = translations_by_block.get(block_id)
            if not translated or not translated.strip():
                continue

            text = translated.strip()
            if block_type == "heading" and _is_toc_heading(text):
                in_toc = True
                continue
            if in_toc:
                if block_type == "heading" and not _is_toc_heading(text):
                    in_toc = False
                elif _looks_like_toc_entry(text):
                    continue
                elif len(text) > 40:
                    in_toc = False
                else:
                    continue

            if _looks_like_toc_entry(text) and len(text) <= 8:
                continue

            kind = "heading" if block_type == "heading" else "body"
            out.append(
                _RenderBlock(
                    kind=kind,
                    text=text,
                    page_number_in_source=page_number,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Public render entrypoint
# ---------------------------------------------------------------------------


def render_template_pdf(
    *,
    structure_path: Path,
    translations_by_block: dict[str, str],
    output_dir: Path,
    output_name: str,
    config: TemplateConfig,
    skip_pages: set[int] | None = None,
    cover_image_path: Path | str | None = None,
) -> TemplateRenderResult:
    """Render translated text into a clean Burmese book PDF.

    Pipeline:
      1. Build the body HTML (chapters + paragraphs) and matching CSS.
      2. Stream the Story across pages of the configured size.
      3. After each page, scan its text for the next chapter heading so
         the running-header state is up to date for *that* page.
      4. After all pages are written, reopen the file and add running
         headers + page numbers.

    ``skip_pages`` (M15): set of source page numbers to drop. The
    template flows blocks naturally, so once a skipped page's blocks are
    excluded, the output simply doesn't include them — no page deletion
    needed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / output_name

    blocks = _load_render_blocks(
        structure_path, translations_by_block, skip_pages=skip_pages
    )
    if not blocks:
        out_path.write_bytes(b"%PDF-1.4\n%empty template render\n")
        return TemplateRenderResult(
            output_path=out_path, pages_written=0, chapters_rendered=0, blocks_placed=0
        )

    chapter_count = sum(1 for b in blocks if b.kind == "heading")

    # Resolve fonts honoring the configured body-font name.
    fonts = get_burmese_fonts(preferred_name=config.body_font)
    if fonts.regular_path is None:
        log.warning(
            "no Burmese font available for template render; output will be "
            "incorrect. Run scripts/download_fonts.sh to install Noto Sans Myanmar."
        )

    archive = pymupdf.Archive()
    if fonts.regular_path is not None:
        archive.add(str(fonts.regular_path), fonts.regular_path.name)
    if fonts.bold_path is not None and fonts.bold_path != fonts.regular_path:
        archive.add(str(fonts.bold_path), fonts.bold_path.name)

    body_html = _build_body_html(blocks, config)
    body_css = _build_body_css(fonts, config)

    page_size_pt = _resolve_page_size(config.page_size)
    margin_pt = config.margin_mm * _MM_TO_PT
    media_rect = pymupdf.Rect(0, 0, page_size_pt[0], page_size_pt[1])
    content_rect = pymupdf.Rect(
        margin_pt,
        margin_pt + 12,  # leave room for running header
        page_size_pt[0] - margin_pt,
        page_size_pt[1] - margin_pt - 12,  # leave room for footer
    )

    # Step 1: render body to a temp PDF. We can't pre-compute chapter
    # positions without doing this; do it once.
    body_tmp = out_path.with_suffix(".body.tmp.pdf")
    _render_story_to_pdf(
        html=body_html,
        css=body_css,
        archive=archive,
        out=body_tmp,
        media_rect=media_rect,
        content_rect=content_rect,
    )

    # Step 2: find which body-page each chapter heading lands on. Chapter
    # headings have ``page-break-before: always`` so each heading is at
    # the top of its page.
    chapter_titles = [b.text for b in blocks if b.kind == "heading"]
    body_chapter_pages = _find_chapter_pages(body_tmp, chapter_titles)

    # Step 3: render TOC. Iterate up to a few times so the TOC's own page
    # count converges (changing the offset shifts page numbers, which
    # might add or remove a TOC page). Two iterations almost always
    # suffices for ≤30 chapters.
    toc_tmp: Path | None = None
    toc_pages = 0
    if chapter_titles:
        toc_tmp = out_path.with_suffix(".toc.tmp.pdf")
        toc_pages_estimate = 1
        for _ in range(3):
            toc_html = _build_toc_html(
                chapter_titles=chapter_titles,
                body_chapter_pages=body_chapter_pages,
                offset=toc_pages_estimate,
                config=config,
            )
            toc_css = _build_toc_css(fonts, config)
            _render_story_to_pdf(
                html=toc_html,
                css=toc_css,
                archive=archive,
                out=toc_tmp,
                media_rect=media_rect,
                content_rect=content_rect,
            )
            d = pymupdf.open(toc_tmp)
            actual = d.page_count
            d.close()
            if actual == toc_pages_estimate:
                toc_pages = actual
                break
            toc_pages_estimate = actual
            toc_pages = actual

    # Step 3.5: render the cover (if provided) as a single page sized to
    # match the rest of the book. The cover image is fitted-to-page with
    # white margins so portrait covers land cleanly on portrait pages.
    cover_tmp: Path | None = None
    cover_pages = 0
    if cover_image_path:
        cover_tmp = out_path.with_suffix(".cover.tmp.pdf")
        try:
            _render_cover_pdf(
                image_path=Path(cover_image_path),
                out=cover_tmp,
                page_size_pt=page_size_pt,
            )
            cover_pages = 1
        except Exception:
            log.exception("cover render failed; producing PDF without cover")
            cover_tmp = None
            cover_pages = 0

    # Step 4: assemble final PDF — cover page (if any) → TOC pages → body.
    combined_tmp = out_path.with_suffix(".combined.tmp.pdf")
    _combine_pdfs_in_order([cover_tmp, toc_tmp, body_tmp], combined_tmp)
    pages_written = 0
    d = pymupdf.open(combined_tmp)
    pages_written = d.page_count
    d.close()

    # Step 5: add running header + page number to every page. The cover
    # and TOC pages skip the running header; only body pages get it.
    _add_page_furniture(
        body_pdf=combined_tmp,
        out_path=out_path,
        config=config,
        page_size_pt=page_size_pt,
        margin_pt=margin_pt,
        fonts=fonts,
        chapter_titles=chapter_titles,
        toc_page_count=toc_pages,
        cover_page_count=cover_pages,
    )
    for p in (body_tmp, toc_tmp, cover_tmp, combined_tmp):
        if p is None:
            continue
        try:
            p.unlink()
        except OSError:
            pass

    return TemplateRenderResult(
        output_path=out_path,
        pages_written=pages_written,
        chapters_rendered=chapter_count,
        blocks_placed=len(blocks),
    )


# ---------------------------------------------------------------------------
# HTML / CSS construction
# ---------------------------------------------------------------------------


def _build_body_html(blocks: list[_RenderBlock], config: TemplateConfig) -> str:
    """Assemble the full-book HTML.

    Each chapter heading gets ``page-break-before: always`` (except the
    very first) so chapters always start on a fresh page. Paragraphs are
    wrapped in ``<p class="body">`` for the first-line-indent CSS to apply.
    """
    parts: list[str] = ["<body>"]
    saw_first_heading = False
    chapter_index = 0
    for block in blocks:
        if block.kind == "heading":
            chapter_index += 1
            cls = "chapter" + (" first-chapter" if not saw_first_heading else "")
            parts.append(
                f'<h1 class="{cls}" data-chapter-index="{chapter_index}">'
                f"{html_lib.escape(block.text)}"
                "</h1>"
            )
            saw_first_heading = True
        else:
            parts.append(f'<p class="body">{html_lib.escape(block.text)}</p>')
    parts.append("</body>")
    return "".join(parts)


def _build_body_css(fonts, config: TemplateConfig) -> str:
    """CSS: font registration, body typography, chapter heading style."""
    family = "BookBody"
    css_parts: list[str] = []

    if fonts.regular_path is not None:
        css_parts.append(
            f"@font-face {{ font-family: '{family}'; "
            f"src: url({fonts.regular_path.name}); }}"
        )
    if fonts.bold_path is not None and fonts.bold_path != fonts.regular_path:
        css_parts.append(
            f"@font-face {{ font-family: '{family}'; font-weight: bold; "
            f"src: url({fonts.bold_path.name}); }}"
        )

    align = "justify" if config.justify else "left"
    css_parts.append(
        f"body {{ font-family: '{family}', sans-serif; "
        f"font-size: {config.body_font_size_pt}pt; "
        f"line-height: {config.line_height}; "
        f"text-align: {align}; }}"
    )

    css_parts.append(
        f"p.body {{ text-indent: {config.first_line_indent_mm}mm; "
        f"margin: 0; padding: 0; }}"
    )

    # Chapter heading: large + centered (or left, per config). The first
    # heading skips ``page-break-before`` so the book doesn't open with a
    # blank page.
    chapter_size = config.body_font_size_pt * (
        1.7 if "large" in config.chapter_style else 1.2
    )
    chapter_align = "center" if config.chapter_style.endswith("centered") else "left"
    css_parts.append(
        f"h1.chapter {{ font-size: {chapter_size:.1f}pt; "
        f"font-weight: bold; text-align: {chapter_align}; "
        f"page-break-before: always; "
        f"margin: 12mm 0 6mm 0; padding: 0; }}"
    )
    css_parts.append("h1.first-chapter { page-break-before: auto; }")

    return "\n".join(css_parts)


# ---------------------------------------------------------------------------
# Story → PDF + TOC helpers
# ---------------------------------------------------------------------------


def _render_story_to_pdf(
    *,
    html: str,
    css: str,
    archive: pymupdf.Archive,
    out: Path,
    media_rect: pymupdf.Rect,
    content_rect: pymupdf.Rect,
    safety_cap: int = 5000,
) -> None:
    """Run a PyMuPDF Story to a fresh PDF, one page at a time, until the
    Story signals it has placed all content."""
    story = pymupdf.Story(html=html, archive=archive, user_css=css)
    writer = pymupdf.DocumentWriter(str(out))
    pages = 0
    while pages < safety_cap:
        dev = writer.begin_page(media_rect)
        more, _filled = story.place(content_rect)
        story.draw(dev)
        writer.end_page()
        pages += 1
        if not more:
            break
    writer.close()


def _find_chapter_pages(body_pdf: Path, chapter_titles: list[str]) -> list[int]:
    """Return the body-page index (0-based) where each chapter title first
    appears at the top of a page. Order matches ``chapter_titles``.

    Chapters that don't appear on any page (shouldn't happen given
    ``page-break-before: always``) get the ``-1`` sentinel."""
    if not chapter_titles:
        return []
    doc = pymupdf.open(body_pdf)
    try:
        out: list[int] = [-1] * len(chapter_titles)
        next_idx = 0
        for page_index in range(doc.page_count):
            if next_idx >= len(chapter_titles):
                break
            page = doc.load_page(page_index)
            page_top_h = page.rect.height * 0.3
            top_blocks = [b for b in page.get_text("blocks") if b[1] < page_top_h]
            top_text = " ".join(b[4] for b in top_blocks)
            while (
                next_idx < len(chapter_titles)
                and chapter_titles[next_idx] in top_text
            ):
                out[next_idx] = page_index
                next_idx += 1
        return out
    finally:
        doc.close()


def _build_toc_html(
    *,
    chapter_titles: list[str],
    body_chapter_pages: list[int],
    offset: int,
    config: TemplateConfig,
) -> str:
    """Render the table-of-contents HTML.

    ``offset`` is the number of TOC pages we expect to consume — added to
    each chapter's body-page index so the displayed page numbers reflect
    the chapter's position in the FINAL combined PDF (TOC + body).

    Uses a 2-column ``<table>`` instead of flex layout because PyMuPDF
    Story's CSS subset does not support flex. Tables are well-supported
    and give us clean column alignment for chapter title + page number.
    """
    toc_title = "မာတိကာ" if config.title_target else "Contents"
    rows: list[str] = []
    for title, body_page in zip(chapter_titles, body_chapter_pages):
        if body_page < 0:
            continue
        # body_page is 0-based; combined page (1-based for humans) is
        # body_page + offset (TOC pages) + 1. Mirror what the footer
        # shows so the TOC page numbers match what the reader sees.
        final_page_num = body_page + offset + 1
        rows.append(
            "<tr>"
            f'<td class="toc-title">{html_lib.escape(title)}</td>'
            f'<td class="toc-page">{final_page_num}</td>'
            "</tr>"
        )
    return (
        "<body>"
        f'<h1 class="toc-heading">{html_lib.escape(toc_title)}</h1>'
        '<table class="toc">'
        + "".join(rows)
        + "</table>"
        "</body>"
    )


def _build_toc_css(fonts, config: TemplateConfig) -> str:
    """CSS for the TOC page. Uses the same body font as the rest of the
    book; the 2-column table gives clean alignment of titles vs. page
    numbers (PyMuPDF Story doesn't support flex layout)."""
    family = "BookBody"
    parts: list[str] = []
    if fonts.regular_path is not None:
        parts.append(
            f"@font-face {{ font-family: '{family}'; "
            f"src: url({fonts.regular_path.name}); }}"
        )
    if fonts.bold_path is not None and fonts.bold_path != fonts.regular_path:
        parts.append(
            f"@font-face {{ font-family: '{family}'; font-weight: bold; "
            f"src: url({fonts.bold_path.name}); }}"
        )
    parts.append(
        f"body {{ font-family: '{family}', sans-serif; "
        f"font-size: {config.body_font_size_pt}pt; "
        f"line-height: {config.line_height}; "
        f"text-align: left; }}"
    )
    parts.append(
        "h1.toc-heading { font-size: "
        f"{config.body_font_size_pt * 1.7:.1f}pt; "
        "font-weight: bold; text-align: center; "
        "margin: 0 0 8mm 0; padding: 0; }"
    )
    parts.append("table.toc { width: 100%; border-collapse: collapse; }")
    parts.append(
        "table.toc td { "
        f"font-size: {config.body_font_size_pt}pt; "
        "padding: 2pt 0; vertical-align: baseline; }"
    )
    parts.append(
        "table.toc td.toc-title { text-align: left; }"
        "table.toc td.toc-page { text-align: right; "
        "white-space: nowrap; padding-left: 8pt; }"
    )
    return "\n".join(parts)


def _combine_pdfs(toc_pdf: Path | None, body_pdf: Path, out_path: Path) -> None:
    """Backward-compat wrapper kept for callers that only pass TOC + body."""
    _combine_pdfs_in_order([toc_pdf, body_pdf], out_path)


def _combine_pdfs_in_order(parts: list[Path | None], out_path: Path) -> None:
    """Concatenate a list of PDFs in order. ``None`` and missing files are
    silently skipped — the caller can pass a fixed-shape list with optional
    parts (cover, TOC, body) without having to filter it first."""
    combined = pymupdf.open()
    try:
        for p in parts:
            if p is None or not p.exists():
                continue
            src = pymupdf.open(p)
            try:
                combined.insert_pdf(src)
            finally:
                src.close()
        combined.save(str(out_path))
    finally:
        combined.close()


def _render_cover_pdf(
    *, image_path: Path, out: Path, page_size_pt: tuple[float, float]
) -> None:
    """Render a single-page PDF where the cover image fills the page,
    aspect-fit (preserves aspect ratio; pads to white if needed).

    Pillow reads the image dimensions; PyMuPDF embeds it. We don't go
    through PyMuPDF for the dimension read because pymupdf.open() on a
    raw image file can be flaky depending on build options.
    """
    from PIL import Image

    page_w, page_h = page_size_pt
    with Image.open(image_path) as im:
        img_w, img_h = im.size
    if img_w <= 0 or img_h <= 0:
        raise RuntimeError(f"cover image has zero dimensions: {image_path}")

    doc = pymupdf.open()
    try:
        page = doc.new_page(width=page_w, height=page_h)
        # Compute the largest rect inside the page that preserves the
        # image's aspect ratio.
        scale = min(page_w / img_w, page_h / img_h)
        target_w = img_w * scale
        target_h = img_h * scale
        x0 = (page_w - target_w) / 2
        y0 = (page_h - target_h) / 2
        rect = pymupdf.Rect(x0, y0, x0 + target_w, y0 + target_h)
        page.insert_image(rect, filename=str(image_path))
        doc.save(str(out))
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Page furniture (running headers + footers)
# ---------------------------------------------------------------------------


def _add_page_furniture(
    *,
    body_pdf: Path,
    out_path: Path,
    config: TemplateConfig,
    page_size_pt: tuple[float, float],
    margin_pt: float,
    fonts,
    chapter_titles: list[str],
    toc_page_count: int = 0,
    cover_page_count: int = 0,
) -> None:
    """Open the body PDF, add running header + page number to each page.

    Chapter detection: we know the chapter heading strings up-front (the
    template emitted them as ``<h1>`` with ``page-break-before: always``).
    For each page we scan its text and check whether the next-unseen
    chapter heading appears at the top of the page (top 30%). When it
    does, that page begins that chapter; otherwise we inherit the
    current chapter from the previous page. This is robust: body text
    can never leak into the running header because we only ever match
    known heading strings.

    ``toc_page_count`` tells us how many leading pages are TOC pages.
    Those pages get a footer page number (matching what the TOC entries
    point to) but skip the running header — standard book convention.
    """
    doc = pymupdf.open(body_pdf)
    try:
        title = config.title_target or config.title_source or ""
        page_number_style = config.page_number_style
        running_header_mode = config.running_header
        header_alignment = config.header_alignment

        # Build a map: page index -> chapter title in effect on that page.
        # Walk pages in order (skipping cover + TOC pages — they don't
        # carry chapter state), advancing through ``chapter_titles`` only
        # when the next heading is found at the top of the current page.
        per_page_chapter: dict[int, str] = {}
        next_chapter_idx = 0
        current_chapter = ""
        prelude_pages = cover_page_count + toc_page_count
        for i in range(doc.page_count):
            if i < prelude_pages:
                # Cover or TOC page — no chapter label.
                per_page_chapter[i] = ""
                continue
            page = doc.load_page(i)
            top_y_threshold = page_size_pt[1] * 0.3
            top_blocks = [
                b for b in page.get_text("blocks")
                if b[1] < top_y_threshold
            ]
            top_text = " ".join(b[4] for b in top_blocks)

            # Only the next-unseen chapter heading is a candidate.
            while (
                next_chapter_idx < len(chapter_titles)
                and chapter_titles[next_chapter_idx] in top_text
            ):
                current_chapter = chapter_titles[next_chapter_idx]
                next_chapter_idx += 1
            per_page_chapter[i] = current_chapter

        # Choose font for furniture text. Use the same Burmese font so
        # Burmese book titles + chapter labels render properly. Fall back
        # to Helvetica when no Burmese font is available.
        if fonts.regular_path is not None:
            font_arg = {"fontfile": str(fonts.regular_path), "fontname": "BookFurn"}
        else:
            font_arg = {"fontname": "helv"}

        for i in range(doc.page_count):
            page = doc.load_page(i)
            page_num_human = i + 1
            chapter = per_page_chapter.get(i, "")
            is_cover_page = i < cover_page_count
            is_toc_page = cover_page_count <= i < cover_page_count + toc_page_count

            # Cover pages have no furniture at all — just the image.
            if is_cover_page:
                continue

            # --- Footer: page number ---
            footer_text = _page_number_text(page_num_human, page_number_style)
            if footer_text:
                _draw_centered_text(
                    page,
                    text=footer_text,
                    x_center=page_size_pt[0] / 2,
                    y=page_size_pt[1] - margin_pt / 2,
                    font_size=9,
                    **font_arg,
                )

            # Running header is skipped on TOC pages — clean book
            # convention. The TOC heading is already part of the
            # rendered page content.
            if is_toc_page:
                continue

            # --- Running header: title + chapter ---
            left_text, right_text = _header_pair(
                title=title,
                chapter=chapter,
                page_num=page_num_human,
                mode=running_header_mode,
                alignment=header_alignment,
            )
            top_y = margin_pt / 2
            if header_alignment == "centered":
                combined = " · ".join(t for t in (left_text, right_text) if t)
                if combined:
                    _draw_centered_text(
                        page,
                        text=combined,
                        x_center=page_size_pt[0] / 2,
                        y=top_y,
                        font_size=9,
                        **font_arg,
                    )
            else:
                if left_text:
                    _draw_text_at(
                        page,
                        text=left_text,
                        x=margin_pt,
                        y=top_y,
                        font_size=9,
                        **font_arg,
                    )
                if right_text:
                    _draw_right_aligned_text(
                        page,
                        text=right_text,
                        x_right=page_size_pt[0] - margin_pt,
                        y=top_y,
                        font_size=9,
                        **font_arg,
                    )

        # Save final PDF with metadata.
        doc.set_metadata(
            {
                "title": (config.title_target or config.title_source or "Translated"),
                "author": (config.author_target or config.author_source or ""),
            }
        )
        doc.save(out_path, garbage=4, deflate=True)
    finally:
        doc.close()


def _draw_text_at(page, *, text: str, x: float, y: float, font_size: float, **font_kwargs) -> None:
    """Draw text using insert_htmlbox so Burmese glyphs shape correctly."""
    # Use a tight horizontal rect and let HTML wrap. For Burmese we MUST
    # go through an HTML path so HarfBuzz shaping applies; ``page.insert_text``
    # uses the simple text engine which mis-shapes Myanmar.
    rect = pymupdf.Rect(x, y - font_size, x + 800, y + 4)
    fontfile = font_kwargs.get("fontfile")
    if fontfile:
        archive = pymupdf.Archive()
        font_path = Path(fontfile)
        archive.add(str(font_path), font_path.name)
        css = (
            f"@font-face {{ font-family: 'F'; src: url({font_path.name}); }}"
            f"body {{ font-family: 'F'; font-size: {font_size}pt; "
            f"line-height: 1; margin: 0; padding: 0; white-space: nowrap; }}"
        )
        page.insert_htmlbox(
            rect,
            f"<body>{html_lib.escape(text)}</body>",
            css=css,
            archive=archive,
        )
    else:
        # Latin-only fallback path (no Burmese font installed).
        page.insert_text((x, y), text, fontsize=font_size, fontname="helv")


def _draw_centered_text(page, *, text: str, x_center: float, y: float, font_size: float, **font_kwargs) -> None:
    # Approximate width via insert_htmlbox in a wide rect; centering is via
    # CSS ``text-align: center`` over a rectangle spanning the full width.
    fontfile = font_kwargs.get("fontfile")
    if fontfile:
        archive = pymupdf.Archive()
        font_path = Path(fontfile)
        archive.add(str(font_path), font_path.name)
        css = (
            f"@font-face {{ font-family: 'F'; src: url({font_path.name}); }}"
            f"body {{ font-family: 'F'; font-size: {font_size}pt; "
            f"line-height: 1; text-align: center; margin: 0; padding: 0; "
            f"white-space: nowrap; }}"
        )
        rect = pymupdf.Rect(0, y - font_size, page.rect.width, y + 4)
        page.insert_htmlbox(
            rect,
            f"<body>{html_lib.escape(text)}</body>",
            css=css,
            archive=archive,
        )
    else:
        page.insert_text(
            (x_center - len(text) * font_size * 0.25, y),
            text,
            fontsize=font_size,
            fontname="helv",
        )


def _draw_right_aligned_text(page, *, text: str, x_right: float, y: float, font_size: float, **font_kwargs) -> None:
    fontfile = font_kwargs.get("fontfile")
    if fontfile:
        archive = pymupdf.Archive()
        font_path = Path(fontfile)
        archive.add(str(font_path), font_path.name)
        css = (
            f"@font-face {{ font-family: 'F'; src: url({font_path.name}); }}"
            f"body {{ font-family: 'F'; font-size: {font_size}pt; "
            f"line-height: 1; text-align: right; margin: 0; padding: 0; "
            f"white-space: nowrap; }}"
        )
        rect = pymupdf.Rect(0, y - font_size, x_right, y + 4)
        page.insert_htmlbox(
            rect,
            f"<body>{html_lib.escape(text)}</body>",
            css=css,
            archive=archive,
        )
    else:
        page.insert_text(
            (x_right - len(text) * font_size * 0.5, y),
            text,
            fontsize=font_size,
            fontname="helv",
        )


def _header_pair(
    *,
    title: str,
    chapter: str,
    page_num: int,
    mode: str,
    alignment: str,
) -> tuple[str, str]:
    """Return (left, right) running-header text. Empty strings hide the side."""
    if not mode:
        return ("", "")
    if mode == "title":
        return (title, "") if page_num % 2 == 0 else ("", title)
    if mode == "chapter":
        return (chapter, "") if page_num % 2 == 0 else ("", chapter)
    if mode == "title_and_chapter":
        if page_num % 2 == 0:
            return (title, chapter)
        return (chapter, title)
    return ("", "")


def _page_number_text(page_num: int, style: str) -> str:
    if not style or style == "hidden":
        return ""
    if style == "page":
        return f"{page_num} | Page"
    if style == "dashed":
        return f"— {page_num} —"
    return str(page_num)


def _resolve_page_size(name: str) -> tuple[float, float]:
    """Return (width_pt, height_pt) for a named page size."""
    upper = (name or "").upper()
    rect = None
    if upper == "A4":
        rect = pymupdf.paper_rect("a4")
    elif upper == "LETTER":
        rect = pymupdf.paper_rect("letter")
    elif upper in {"6X9", "6×9"}:
        return (6 * 72, 9 * 72)
    else:
        rect = pymupdf.paper_rect("a5")
    return (rect.width, rect.height)

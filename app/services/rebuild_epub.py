"""EPUB generation from translated structure.

We split the document into chapters at heading boundaries: every "heading"
block starts a new chapter. Content between headings (or before the first
heading) becomes the chapter body. We render headings as <h1>/<h2> based on
relative font size, and paragraphs as <p>. OCR blocks become <p> wrapped in
a "ocr" class so the user can style them differently.

Burmese font (Noto Sans Myanmar) is embedded so the EPUB renders correctly on
e-readers without the font installed.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
import uuid
from dataclasses import dataclass
from html import escape
from pathlib import Path

from ebooklib import epub

from app.config import get_settings
from app.utils.fonts import BurmeseFontSet, get_burmese_fonts

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpubResult:
    output_path: Path
    chapters_written: int


_CSS_TEMPLATE = """\
@font-face {{
  font-family: "{family}";
  src: url("{regular_filename}") format("truetype");
  font-weight: normal;
  font-style: normal;
}}
@font-face {{
  font-family: "{family}";
  src: url("{bold_filename}") format("truetype");
  font-weight: bold;
  font-style: normal;
}}

body {{
  font-family: "{family}", sans-serif;
  line-height: {line_height};
  margin: 1em;
}}
h1 {{ font-size: 1.6em; margin: 1.2em 0 0.6em; }}
h2 {{ font-size: 1.3em; margin: 1em 0 0.5em; }}
p  {{ margin: 0.6em 0; text-align: justify; }}
.ocr {{ color: #444; font-size: 0.95em; }}
"""


@dataclass
class _Chapter:
    title: str
    file_name: str
    blocks: list[dict]  # [{"type": "heading"|"paragraph"|"ocr", "text": "..."}]


def rebuild_epub(
    *,
    structure_path: Path,
    translations_by_block: dict[str, str],
    output_dir: Path,
    output_name: str,
    title: str,
    job_id: str,
) -> EpubResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / output_name

    with structure_path.open("r", encoding="utf-8") as fh:
        structure = json.load(fh)

    chapters = _collect_chapters(structure, translations_by_block)

    book = epub.EpubBook()
    book.set_identifier(job_id or str(uuid.uuid4()))
    book.set_title(title)
    book.set_language("my")
    book.add_author("Translated by pdf-translation")

    fonts = get_burmese_fonts()
    css_filename = "style/main.css"
    _embed_fonts(book, fonts)
    book.add_item(
        epub.EpubItem(
            uid="main_css",
            file_name=css_filename,
            media_type="text/css",
            content=_build_css(fonts).encode("utf-8"),
        )
    )

    epub_chapters: list[epub.EpubHtml] = []
    for idx, ch in enumerate(chapters, start=1):
        item = epub.EpubHtml(title=ch.title, file_name=ch.file_name, lang="my")
        item.content = _render_chapter_html(ch, css_filename=css_filename)
        item.add_link(href=css_filename, rel="stylesheet", type="text/css")
        book.add_item(item)
        epub_chapters.append(item)

    book.toc = tuple(epub_chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *epub_chapters]

    epub.write_epub(str(out_path), book)
    return EpubResult(output_path=out_path, chapters_written=len(epub_chapters))


def _collect_chapters(structure: dict, translations: dict[str, str]) -> list[_Chapter]:
    """Walk pages → blocks; split into chapters at heading boundaries."""
    chapters: list[_Chapter] = []
    current = _Chapter(title="Untitled", file_name="chap_001.xhtml", blocks=[])

    for page in structure.get("pages", []):
        for block in page.get("blocks", []):
            block_id = block.get("block_id", "")
            btype = block.get("type", "")
            if btype not in {"paragraph", "heading", "ocr"}:
                continue
            translated = translations.get(block_id, "").strip()
            if not translated:
                continue
            if btype == "heading":
                # Flush prior chapter if it had content; start a new one.
                if current.blocks:
                    chapters.append(current)
                    next_idx = len(chapters) + 1
                    current = _Chapter(
                        title=translated,
                        file_name=f"chap_{next_idx:03d}.xhtml",
                        blocks=[{"type": "heading", "text": translated}],
                    )
                else:
                    # First heading establishes the title of the (still empty) chapter.
                    current.title = translated
                    current.blocks.append({"type": "heading", "text": translated})
            else:
                current.blocks.append({"type": btype, "text": translated})

    if current.blocks:
        chapters.append(current)

    if not chapters:
        # No translated blocks at all — emit one empty chapter so the EPUB is valid.
        chapters.append(
            _Chapter(title="Empty", file_name="chap_001.xhtml", blocks=[])
        )
    return chapters


def _render_chapter_html(ch: _Chapter, *, css_filename: str) -> str:
    """Return the inner body markup. ``ebooklib`` wraps it in a full XHTML doc."""
    parts: list[str] = []
    saw_first_heading = False
    for block in ch.blocks:
        text = block["text"]
        if block["type"] == "heading":
            tag = "h1" if not saw_first_heading else "h2"
            saw_first_heading = True
            parts.append(f"<{tag}>{escape(text)}</{tag}>")
        elif block["type"] == "ocr":
            parts.append(f'<p class="ocr">{_paragraph_html(text)}</p>')
        else:
            parts.append(f"<p>{_paragraph_html(text)}</p>")
    if not parts:
        # ebooklib chokes on empty body content — emit a placeholder.
        parts.append(f"<p>{escape(ch.title)}</p>")
    return "\n".join(parts)


def _paragraph_html(text: str) -> str:
    """Escape text and convert internal newlines to <br/>.

    M10: NFC normalize so canonically-equivalent Burmese sequences (decomposed
    vs precomposed combinations) shape consistently in the e-reader."""
    normalized = unicodedata.normalize("NFC", text)
    pieces = [escape(line) for line in re.split(r"\r?\n", normalized)]
    return "<br/>".join(pieces)


def _embed_fonts(book: epub.EpubBook, fonts: BurmeseFontSet) -> None:
    if fonts.regular_path is None:
        return
    book.add_item(
        epub.EpubItem(
            uid="font_regular",
            file_name=f"fonts/{fonts.regular_path.name}",
            media_type="font/ttf",
            content=fonts.regular_path.read_bytes(),
        )
    )
    if fonts.bold_path is not None:
        book.add_item(
            epub.EpubItem(
                uid="font_bold",
                file_name=f"fonts/{fonts.bold_path.name}",
                media_type="font/ttf",
                content=fonts.bold_path.read_bytes(),
            )
        )


def _build_css(fonts: BurmeseFontSet) -> str:
    line_height = f"{get_settings().mm_line_height:.2f}"
    if fonts.regular_path is None:
        return (
            f"body {{ font-family: sans-serif; line-height: {line_height}; margin: 1em; }}\n"
            "h1 { font-size: 1.6em; }\n"
            "h2 { font-size: 1.3em; }\n"
            ".ocr { color: #444; font-size: 0.95em; }\n"
        )
    return _CSS_TEMPLATE.format(
        family=fonts.family_name,
        regular_filename=f"../fonts/{fonts.regular_path.name}",
        bold_filename=f"../fonts/{(fonts.bold_path or fonts.regular_path).name}",
        line_height=line_height,
    )

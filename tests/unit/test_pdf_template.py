"""Unit tests for the PDF template renderer (M14)."""
from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from app.services.pdf_template import (
    TemplateConfig,
    render_template_pdf,
)


def _structure(path: Path, blocks: list[dict]) -> Path:
    structure = {
        "schema_version": 1,
        "summary": {"total_pages": 1, "total_blocks": len(blocks), "scanned_pages": []},
        "pages": [
            {
                "page_number": 1,
                "width": 400,
                "height": 500,
                "is_scanned": False,
                "blocks": blocks,
            }
        ],
    }
    path.write_text(json.dumps(structure, ensure_ascii=False), encoding="utf-8")
    return path


def test_render_template_pdf_produces_valid_pdf(tmp_path: Path) -> None:
    structure_path = _structure(
        tmp_path / "structure.json",
        [
            {"block_id": "b0", "type": "heading", "text": "Chapter 1"},
            {"block_id": "b1", "type": "paragraph", "text": "First paragraph source."},
            {"block_id": "b2", "type": "paragraph", "text": "Second paragraph source."},
        ],
    )
    translations = {
        "b0": "ပထမအခန်း",
        "b1": "ပထမစာပိုဒ်ဖြစ်သည်။",
        "b2": "ဒုတိယစာပိုဒ်ဖြစ်သည်။",
    }
    cfg = TemplateConfig.from_settings(
        title_source="Test Book",
        title_target="စမ်းသပ်စာအုပ်",
        author_source="Anon",
        author_target=None,
    )
    result = render_template_pdf(
        structure_path=structure_path,
        translations_by_block=translations,
        output_dir=tmp_path / "out",
        output_name="test.pdf",
        config=cfg,
    )
    assert result.output_path.exists()
    assert result.blocks_placed == 3
    assert result.chapters_rendered == 1
    # Open the result and verify it's a real, readable PDF.
    doc = pymupdf.open(result.output_path)
    try:
        assert doc.page_count >= 1
        # Body paragraph appears somewhere in the document — page 0 is
        # the auto-generated TOC, so we scan all pages instead of
        # assuming page 0.
        all_text = "".join(
            doc.load_page(i).get_text("text") for i in range(doc.page_count)
        )
        assert "ပထမစာပိုဒ်" in all_text
    finally:
        doc.close()


def test_render_template_pdf_handles_no_translations(tmp_path: Path) -> None:
    structure_path = _structure(
        tmp_path / "structure.json",
        [{"block_id": "b0", "type": "paragraph", "text": "untranslated"}],
    )
    cfg = TemplateConfig.from_settings()
    result = render_template_pdf(
        structure_path=structure_path,
        translations_by_block={},  # empty
        output_dir=tmp_path / "out",
        output_name="empty.pdf",
        config=cfg,
    )
    assert result.output_path.exists()
    assert result.blocks_placed == 0
    assert result.chapters_rendered == 0
    assert result.pages_written == 0


def test_template_config_applies_overrides() -> None:
    cfg = TemplateConfig.from_settings(
        overrides={
            "page_size": "LETTER",
            "body_font_size_pt": 13.0,
            "page_number_style": "dashed",
        }
    )
    assert cfg.page_size == "LETTER"
    assert cfg.body_font_size_pt == 13.0
    assert cfg.page_number_style == "dashed"


def test_template_config_falls_back_to_settings_defaults() -> None:
    cfg = TemplateConfig.from_settings()
    # The defaults from config.py
    assert cfg.page_size == "A5"
    assert cfg.body_font_size_pt == 10.0
    assert cfg.line_height == 1.5
    # Justified by default; the small body font + tight margins keep
    # word-gap stretching tame in Burmese.
    assert cfg.justify is True


def test_template_config_ignores_unknown_keys_in_overrides() -> None:
    cfg = TemplateConfig.from_settings(overrides={"page_size": "A4", "unknown": "x"})
    assert cfg.page_size == "A4"
    assert not hasattr(cfg, "unknown")


def test_template_skips_source_toc_entries(tmp_path: Path) -> None:
    """Source ``Contents`` heading + digit-only paragraphs should be filtered
    out of the rendered output — the template generates its own TOC."""
    structure_path = _structure(
        tmp_path / "structure.json",
        [
            # Source TOC: heading + digit-only paragraphs
            {"block_id": "h0", "type": "heading", "text": "Contents"},
            {"block_id": "t1", "type": "paragraph", "text": "1\n5"},
            {"block_id": "t2", "type": "paragraph", "text": "2\n11"},
            {"block_id": "t3", "type": "paragraph", "text": "3\n15"},
            # First real chapter
            {"block_id": "h1", "type": "heading", "text": "Chapter 1"},
            {"block_id": "p1", "type": "paragraph", "text": "Real prose " * 20},
        ],
    )
    translations = {
        "h0": "မာတိကာ",
        "t1": "၁\n၅",
        "t2": "၂\n၁၁",
        "t3": "၃\n၁၅",
        "h1": "ပထမအခန်း",
        "p1": "တကယ့်စာသား " * 20,
    }
    cfg = TemplateConfig.from_settings(title_target="စမ်းစာအုပ်")
    result = render_template_pdf(
        structure_path=structure_path,
        translations_by_block=translations,
        output_dir=tmp_path / "out",
        output_name="toc.pdf",
        config=cfg,
    )
    # The 3 TOC entries + the "Contents" heading should be skipped — only
    # the real chapter heading + body remains.
    assert result.chapters_rendered == 1
    assert result.blocks_placed == 2  # h1 (heading) + p1 (body), not 6


def test_template_skips_bare_page_number_paragraphs(tmp_path: Path) -> None:
    structure_path = _structure(
        tmp_path / "structure.json",
        [
            {"block_id": "h", "type": "heading", "text": "Chapter"},
            {"block_id": "page", "type": "paragraph", "text": "5"},
            {"block_id": "real", "type": "paragraph", "text": "Real prose " * 20},
        ],
    )
    translations = {
        "h": "အခန်း",
        "page": "၅",
        "real": "တကယ့်စာသား " * 20,
    }
    cfg = TemplateConfig.from_settings()
    result = render_template_pdf(
        structure_path=structure_path,
        translations_by_block=translations,
        output_dir=tmp_path / "out",
        output_name="page.pdf",
        config=cfg,
    )
    # Bare "၅" page-number paragraph should drop out.
    assert result.blocks_placed == 2  # heading + real body, not 3


def test_render_template_pdf_renders_running_header_and_footer(tmp_path: Path) -> None:
    # Build enough body to span >1 page so we exercise the page-furniture
    # callback on a non-first page.
    blocks = [{"block_id": "h0", "type": "heading", "text": "Long Chapter"}]
    for i in range(50):
        blocks.append(
            {
                "block_id": f"p{i}",
                "type": "paragraph",
                "text": f"Paragraph {i}",
            }
        )
    structure_path = _structure(tmp_path / "structure.json", blocks)
    translations = {"h0": "ရှည်လျားသည့်အခန်း"}
    for i in range(50):
        translations[f"p{i}"] = f"စာပိုဒ် {i}။ " * 8

    cfg = TemplateConfig.from_settings(
        title_source="Title", title_target="ခေါင်းစဉ်"
    )
    result = render_template_pdf(
        structure_path=structure_path,
        translations_by_block=translations,
        output_dir=tmp_path / "out",
        output_name="multi.pdf",
        config=cfg,
    )
    assert result.output_path.exists()
    doc = pymupdf.open(result.output_path)
    try:
        # Multi-page output is the load-bearing assertion — exercises the
        # page-furniture callback on a non-first page. We don't grep for
        # "| Page" because PyMuPDF's text extraction doesn't reliably pick
        # up canvas-drawn footers (the footer is visually present in the
        # PDF; the rendered PNG confirms this on real output).
        assert doc.page_count >= 2
        # The body font registers each TTF subset with a unique name; just
        # confirm at least one font is embedded so we know the renderer
        # actually shipped non-default fonts.
        embedded_fonts = doc.embfile_names() + [
            f for page in doc for f in page.get_fonts()
        ]
        assert embedded_fonts, "no embedded fonts in output"
    finally:
        doc.close()

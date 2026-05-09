"""Unit tests for M10 polish: chunk-size overrides + NFC normalization."""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from app.db.models import Category
from app.services import chunking


def _write_structure(path: Path, *, blocks_per_page: int, words_per_block: int) -> Path:
    """Build a structure.json with enough text that chunking will produce
    multiple chunks even at the smallest target sizes we test."""
    page_blocks = []
    for i in range(blocks_per_page):
        page_blocks.append(
            {
                "block_id": f"p1b{i}",
                "type": "paragraph",
                "text": ("word " * words_per_block).strip(),
                "bbox": [0, 0, 100, 20],
            }
        )
    structure = {"pages": [{"page_number": 1, "blocks": page_blocks, "is_scanned": False}]}
    path.write_text(json.dumps(structure), encoding="utf-8")
    return path


def test_chunking_respects_target_override(tmp_path: Path) -> None:
    structure_path = _write_structure(
        tmp_path / "s.json", blocks_per_page=20, words_per_block=80
    )
    # Tight cap → many small chunks; loose cap → fewer larger chunks.
    tight = chunking.build_chunks(
        structure_path,
        category=Category.NOVEL,
        target_tokens_override=200,
        hard_cap_tokens_override=400,
    )
    loose = chunking.build_chunks(
        structure_path,
        category=Category.NOVEL,
        target_tokens_override=2000,
        hard_cap_tokens_override=4000,
    )
    assert len(tight) > len(loose), "tighter cap should produce more chunks"


def test_chunking_override_beats_educational_ratio(tmp_path: Path) -> None:
    """When the user sets an explicit override, the EDUCATIONAL 0.7x scaling
    must NOT apply — explicit user intent wins."""
    structure_path = _write_structure(
        tmp_path / "s.json", blocks_per_page=20, words_per_block=80
    )
    educational_default = chunking.build_chunks(
        structure_path, category=Category.EDUCATIONAL
    )
    # An explicit override at the default target should produce chunks the
    # same size as a NOVEL run with the same target — i.e. the educational
    # downscale should NOT also kick in on top of the override.
    overridden = chunking.build_chunks(
        structure_path,
        category=Category.EDUCATIONAL,
        target_tokens_override=1500,
    )
    novel_baseline = chunking.build_chunks(structure_path, category=Category.NOVEL)
    # The overridden educational run should match the novel run (same target,
    # no extra scaling), and produce *fewer or equal* chunks than the
    # un-overridden educational run.
    assert len(overridden) == len(novel_baseline)
    assert len(overridden) <= len(educational_default)


def test_pdf_paragraph_html_normalizes_to_nfc() -> None:
    """The PDF rebuilder must NFC-normalize translated text before shaping.

    We pick a Latin glyph with both decomposed and precomposed forms because
    asserting on Burmese byte sequences here would be brittle for readers."""
    from app.services.rebuild_pdf import _build_html

    # 'café' with decomposed e + combining acute.
    decomposed = "café"
    precomposed = "café"
    assert decomposed != precomposed
    assert unicodedata.normalize("NFC", decomposed) == precomposed

    out = _build_html(decomposed, font_size=11.0, is_heading=False)
    # The HTML body should contain the precomposed form, not the decomposed.
    assert precomposed in out
    assert decomposed not in out


def test_epub_paragraph_html_normalizes_to_nfc() -> None:
    from app.services.rebuild_epub import _paragraph_html

    decomposed = "café"
    precomposed = "café"
    out = _paragraph_html(decomposed)
    assert precomposed in out
    assert decomposed not in out


def test_pdf_html_uses_configured_line_height(monkeypatch) -> None:
    """The PDF rebuilder reads ``mm_line_height`` from settings so admins can
    tune Burmese line spacing without redeploying."""
    from app.config import get_settings
    from app.services.rebuild_pdf import _build_html

    monkeypatch.setenv("MM_LINE_HEIGHT", "1.85")
    get_settings.cache_clear()
    try:
        html = _build_html("ဟယ်လို", font_size=11.0, is_heading=False)
        assert "line-height:1.85" in html
    finally:
        get_settings.cache_clear()

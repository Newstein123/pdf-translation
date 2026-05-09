"""Unit tests for the Book Brief service (M11 / Track B)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.book_brief import (
    compose_book_brief_block,
    sample_for_brief,
)


def _write_structure(tmp_path: Path, page_texts: list[str]) -> Path:
    pages = [
        {
            "page_number": i + 1,
            "blocks": [
                {"type": "paragraph", "text": text}
            ],
        }
        for i, text in enumerate(page_texts)
    ]
    p = tmp_path / "structure.json"
    p.write_text(json.dumps({"pages": pages}, ensure_ascii=False), encoding="utf-8")
    return p


def test_sample_for_brief_returns_full_when_book_fits_budget(tmp_path: Path) -> None:
    structure = _write_structure(
        tmp_path, ["Short opening paragraph.", "Short closing paragraph."]
    )
    samples = sample_for_brief(structure, target_tokens=10_000)
    assert len(samples) == 1
    assert samples[0].label == "full"
    assert "opening" in samples[0].text
    assert "closing" in samples[0].text


def test_sample_for_brief_stratifies_long_book(tmp_path: Path) -> None:
    # Build a book big enough to exceed the budget so the stratifier kicks in.
    pages = [f"This is page {i} with some prose to fill it out. " * 50 for i in range(50)]
    structure = _write_structure(tmp_path, pages)
    samples = sample_for_brief(structure, target_tokens=2_000, middle_sample_count=3)

    labels = [s.label for s in samples]
    assert "opening" in labels
    assert "closing" in labels
    # Should have at least one middle sample.
    assert any(label.startswith("middle:") for label in labels)


def test_sample_for_brief_returns_empty_for_missing_file(tmp_path: Path) -> None:
    samples = sample_for_brief(tmp_path / "does-not-exist.json")
    assert samples == []


def test_sample_for_brief_returns_empty_for_empty_pages(tmp_path: Path) -> None:
    p = tmp_path / "structure.json"
    p.write_text(json.dumps({"pages": []}), encoding="utf-8")
    assert sample_for_brief(p) == []


def test_compose_book_brief_block_returns_empty_when_no_brief() -> None:
    assert compose_book_brief_block(None) == ""
    assert compose_book_brief_block({}) == ""


def test_compose_book_brief_block_renders_known_fields() -> None:
    brief = {
        "title": "A Tale of Two Cities",
        "pov": "THIRD_OMNISCIENT",
        "tense": "PAST",
        "register": "LITERARY",
        "register_notes": "19th-century formal English; long sentences.",
        "narrator_voice": "Wry, distant, fond of subordinate clauses.",
        "key_characters": [
            {"name": "Sydney Carton", "role": "protagonist", "voice_notes": "self-deprecating"},
            {"name": "Madame Defarge", "role": "antagonist", "voice_notes": "cold, terse"},
            {"name": "", "role": "noise", "voice_notes": ""},  # ignored
        ],
        "themes": ["sacrifice", "revolution", "duality"],
        "do_not_translate": ["Old Bailey"],
        "translation_pitfalls": ["Watch for archaic legal terms."],
    }
    block = compose_book_brief_block(brief)
    assert "BOOK BRIEF" in block
    assert "A Tale of Two Cities" in block
    assert "THIRD_OMNISCIENT" in block
    assert "Sydney Carton" in block
    assert "Madame Defarge" in block
    assert "noise" not in block  # empty name dropped
    assert "Old Bailey" in block
    assert "archaic legal terms" in block


def test_compose_book_brief_block_skips_empty_strings() -> None:
    brief = {"pov": "", "tense": "PAST", "register": ""}
    block = compose_book_brief_block(brief)
    assert "POV" not in block
    assert "Register:" not in block
    assert "Tense: PAST" in block

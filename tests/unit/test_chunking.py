import json
from pathlib import Path

from app.db.models import Category
from app.services.chunking import build_chunks


def _write_structure(path: Path, blocks_per_page: list[list[dict]]) -> Path:
    pages = []
    for i, blocks in enumerate(blocks_per_page, start=1):
        pages.append(
            {
                "page_number": i,
                "width": 612.0,
                "height": 792.0,
                "is_scanned": False,
                "blocks": blocks,
            }
        )
    path.write_text(
        json.dumps({"schema_version": 1, "summary": {}, "pages": pages}),
        encoding="utf-8",
    )
    return path


def _block(bid: str, text: str) -> dict:
    return {
        "block_id": bid,
        "type": "paragraph",
        "text": text,
        "bbox": [0, 0, 100, 100],
        "dominant_font": None,
        "spans": [],
    }


def test_chunking_respects_block_boundaries(tmp_path: Path) -> None:
    structure = _write_structure(
        tmp_path / "s.json",
        [
            [_block("p1-b0", "Sentence one. " * 20), _block("p1-b1", "Sentence two. " * 20)],
            [_block("p2-b0", "Page two paragraph. " * 20)],
        ],
    )
    chunks = build_chunks(structure, category=Category.NOVEL)
    assert len(chunks) >= 1
    # Block IDs from the chunks must collectively cover all source blocks, in order.
    seen = [b["block_id"] for c in chunks for b in c.blocks]
    assert seen == ["p1-b0", "p1-b1", "p2-b0"]


def test_chunking_flushes_at_target(tmp_path: Path) -> None:
    big_text = "Word " * 1000  # roughly several hundred tokens
    blocks = [_block(f"p1-b{i}", big_text) for i in range(5)]
    structure = _write_structure(tmp_path / "s.json", [blocks])
    chunks = build_chunks(structure, category=Category.NOVEL)
    # With ~5 blocks of ~hundreds of tokens, we should get more than one chunk.
    assert len(chunks) > 1
    # No chunk should massively overshoot the hard cap (allow for one oversized
    # block to land alone, which is permitted).
    from app.config import get_settings

    cap = get_settings().chunk_hard_cap_tokens
    for c in chunks:
        if len(c.blocks) > 1:
            assert c.token_count <= cap


def test_educational_chunks_are_smaller(tmp_path: Path) -> None:
    blocks = [_block(f"p1-b{i}", "Word " * 200) for i in range(10)]
    structure = _write_structure(tmp_path / "s.json", [blocks])
    novel = build_chunks(structure, category=Category.NOVEL)
    edu = build_chunks(structure, category=Category.EDUCATIONAL)
    # Educational uses 0.7x target -> at least as many chunks as novel.
    assert len(edu) >= len(novel)


def test_oversized_block_becomes_solo_chunk(tmp_path: Path) -> None:
    # Single block that exceeds the hard cap.
    huge = "Word " * 5000
    structure = _write_structure(tmp_path / "s.json", [[_block("p1-b0", huge)]])
    chunks = build_chunks(structure, category=Category.NOVEL)
    assert len(chunks) == 1
    assert len(chunks[0].blocks) == 1
    assert chunks[0].blocks[0]["block_id"] == "p1-b0"

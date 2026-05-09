from pathlib import Path

import pymupdf

from app.services.extraction import extract_pdf


def _make_digital_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Big Heading Title", fontsize=24)
    page.insert_text((72, 120), "This is a normal paragraph of body text. " * 4, fontsize=11)
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Chapter Two The Second Heading", fontsize=24)
    page2.insert_text(
        (72, 120),
        "Another paragraph here with enough content to clear the scanned threshold. " * 2,
        fontsize=11,
    )
    doc.save(path)
    doc.close()
    return path


def test_extract_digital_pdf_emits_blocks(tmp_path: Path) -> None:
    pdf = _make_digital_pdf(tmp_path / "doc.pdf")
    result = extract_pdf(pdf, output_dir=tmp_path / "out")

    assert result.total_pages == 2
    assert result.total_blocks > 0
    assert not result.is_scanned
    assert result.scanned_pages == []
    assert result.structured_json_path.exists()
    assert "Big Heading Title" in result.plain_text_excerpt or "Heading" in result.plain_text_excerpt


def test_extract_writes_normalized_schema(tmp_path: Path) -> None:
    import json

    pdf = _make_digital_pdf(tmp_path / "doc.pdf")
    result = extract_pdf(pdf, output_dir=tmp_path / "out")

    data = json.loads(result.structured_json_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["summary"]["total_pages"] == 2
    assert len(data["pages"]) == 2

    first_page = data["pages"][0]
    assert first_page["page_number"] == 1
    assert first_page["is_scanned"] is False
    assert len(first_page["blocks"]) > 0
    block = first_page["blocks"][0]
    for key in ("block_id", "type", "text", "bbox", "dominant_font", "spans"):
        assert key in block
    types = {b["type"] for b in first_page["blocks"]}
    # We expect at least one of: heading or paragraph.
    assert types & {"heading", "paragraph"}

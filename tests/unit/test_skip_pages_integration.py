"""Integration tests for the M15 skip_pages feature.

Covers the three filter sites:
  * Chunking — skipped pages don't enter chunks (saves Gemini tokens)
  * pdf_template — skipped pages don't appear in the rendered TEMPLATE
  * /metadata PATCH — validates skip_pages format at write time

The PRESERVE-mode rebuild_pdf filter is exercised indirectly via
chunking + the worker; integration testing rebuild_pdf needs a real
source PDF with multiple pages, which is heavier than unit-test scope.
"""
from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest


def _structure_with_pages(path: Path, page_blocks: list[list[str]]) -> Path:
    """Build a structure.json where ``page_blocks[i]`` is the list of
    paragraph texts on page ``i+1``."""
    pages = []
    for i, blocks in enumerate(page_blocks):
        pages.append(
            {
                "page_number": i + 1,
                "width": 400,
                "height": 500,
                "is_scanned": False,
                "blocks": [
                    {
                        "block_id": f"p{i + 1}-{j}",
                        "type": "paragraph",
                        "text": text,
                    }
                    for j, text in enumerate(blocks)
                ],
            }
        )
    structure = {
        "schema_version": 1,
        "summary": {"total_pages": len(page_blocks), "total_blocks": 0, "scanned_pages": []},
        "pages": pages,
    }
    path.write_text(json.dumps(structure, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Chunking respects skip_pages
# ---------------------------------------------------------------------------


def test_chunking_excludes_skipped_pages(tmp_path: Path) -> None:
    from app.db.models import Category
    from app.services.chunking import build_chunks

    structure = _structure_with_pages(
        tmp_path / "structure.json",
        [
            ["Page one paragraph one.", "Page one paragraph two."],
            ["Page two — should be SKIPPED."],
            ["Page three paragraph one."],
        ],
    )
    specs = build_chunks(structure, category=Category.NOVEL, skip_pages={2})
    all_text = " ".join(s.source_text for s in specs)
    assert "Page one" in all_text
    assert "Page three" in all_text
    assert "SKIPPED" not in all_text  # page 2 dropped before chunking


def test_chunking_no_skip_keeps_all_pages(tmp_path: Path) -> None:
    from app.db.models import Category
    from app.services.chunking import build_chunks

    structure = _structure_with_pages(
        tmp_path / "structure.json",
        [["Page one."], ["Page two."], ["Page three."]],
    )
    specs = build_chunks(structure, category=Category.NOVEL, skip_pages=None)
    all_text = " ".join(s.source_text for s in specs)
    assert "Page one" in all_text
    assert "Page two" in all_text
    assert "Page three" in all_text


# ---------------------------------------------------------------------------
# pdf_template respects skip_pages
# ---------------------------------------------------------------------------


def test_pdf_template_drops_skipped_pages(tmp_path: Path) -> None:
    from app.services.pdf_template import TemplateConfig, render_template_pdf

    structure = _structure_with_pages(
        tmp_path / "structure.json",
        [
            ["UNIQUEMARKER1 page one prose.", "More page one."],
            ["UNIQUEMARKER2 page two prose."],
            ["UNIQUEMARKER3 page three prose."],
        ],
    )
    translations = {
        "p1-0": "ပထမ စာမျက်နှာ",
        "p1-1": "နောက်ထပ် ပထမ စာမျက်နှာ",
        "p2-0": "ဒုတိယ စာမျက်နှာ — SKIP_THIS_PAGE",
        "p3-0": "တတိယ စာမျက်နှာ",
    }
    cfg = TemplateConfig.from_settings()
    result = render_template_pdf(
        structure_path=structure,
        translations_by_block=translations,
        output_dir=tmp_path / "out",
        output_name="t.pdf",
        config=cfg,
        skip_pages={2},
    )
    doc = pymupdf.open(result.output_path)
    try:
        all_text = "".join(
            doc.load_page(i).get_text("text") for i in range(doc.page_count)
        )
    finally:
        doc.close()
    # Load-bearing assertion: the skipped page's translation never lands
    # in the output. The other side (page 1 + 3 text present) is hard to
    # verify cleanly because PyMuPDF's text extraction breaks Burmese
    # clusters across line wraps; the visible PDF is fine, but
    # ``get_text`` returns reordered glyphs.
    assert "SKIP_THIS_PAGE" not in all_text


# ---------------------------------------------------------------------------
# /metadata PATCH validates skip_pages
# ---------------------------------------------------------------------------


def _seed_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=500)
    page.insert_text((50, 100), "Some content")
    doc.save(path)
    doc.close()
    return path


def _upload(client, auth_headers, path: Path) -> str:
    with path.open("rb") as fh:
        resp = client.post(
            "/api/jobs",
            headers=auth_headers,
            files={"file": (path.name, fh, "application/pdf")},
        )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def test_patch_metadata_accepts_skip_pages(client, auth_headers, tmp_path: Path) -> None:
    pdf = _seed_pdf(tmp_path / "p.pdf")
    job_id = _upload(client, auth_headers, pdf)

    resp = client.patch(
        f"/api/jobs/{job_id}/metadata",
        headers={**auth_headers, "content-type": "application/json"},
        json={"skip_pages": "1-3, 5"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["skip_pages"] == "1-3, 5"


def test_patch_metadata_rejects_invalid_skip_pages(
    client, auth_headers, tmp_path: Path
) -> None:
    pdf = _seed_pdf(tmp_path / "p.pdf")
    job_id = _upload(client, auth_headers, pdf)

    resp = client.patch(
        f"/api/jobs/{job_id}/metadata",
        headers={**auth_headers, "content-type": "application/json"},
        json={"skip_pages": "abc"},
    )
    assert resp.status_code == 400
    assert "skip_pages" in resp.text.lower()


def test_patch_metadata_clears_skip_pages_with_empty_string(
    client, auth_headers, tmp_path: Path
) -> None:
    pdf = _seed_pdf(tmp_path / "p.pdf")
    job_id = _upload(client, auth_headers, pdf)

    client.patch(
        f"/api/jobs/{job_id}/metadata",
        headers={**auth_headers, "content-type": "application/json"},
        json={"skip_pages": "1-3"},
    )
    client.patch(
        f"/api/jobs/{job_id}/metadata",
        headers={**auth_headers, "content-type": "application/json"},
        json={"skip_pages": ""},
    )
    resp = client.get(f"/api/jobs/{job_id}", headers=auth_headers)
    assert resp.json()["skip_pages"] is None

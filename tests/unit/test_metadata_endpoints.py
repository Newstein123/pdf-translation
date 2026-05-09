"""Endpoint tests for M14 metadata + re-export API."""
from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest


def _seed_pdf(path: Path, *, pages: int = 2) -> Path:
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page(width=400, height=500)
        page.insert_text((50, 100), "Test source content. " * 4)
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


def test_patch_metadata_persists_fields(client, auth_headers, tmp_path: Path) -> None:
    pdf = _seed_pdf(tmp_path / "sample.pdf")
    job_id = _upload(client, auth_headers, pdf)

    resp = client.patch(
        f"/api/jobs/{job_id}/metadata",
        headers={**auth_headers, "content-type": "application/json"},
        json={
            "title_source": "The Time Machine",
            "title_target": "ကာလယန္တရား",
            "author_source": "H. G. Wells",
            "author_target": "အိပ်ချ် ဂျီ ဝဲလ်စ်",
            "export_layout_mode": "TEMPLATE",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["title_source"] == "The Time Machine"
    assert body["title_target"] == "ကာလယန္တရား"
    assert body["author_source"] == "H. G. Wells"
    assert body["author_target"] == "အိပ်ချ် ဂျီ ဝဲလ်စ်"
    assert body["export_layout_mode"] == "TEMPLATE"


def test_patch_metadata_only_updates_sent_fields(client, auth_headers, tmp_path: Path) -> None:
    pdf = _seed_pdf(tmp_path / "sample.pdf")
    job_id = _upload(client, auth_headers, pdf)

    # Set both title and author.
    client.patch(
        f"/api/jobs/{job_id}/metadata",
        headers={**auth_headers, "content-type": "application/json"},
        json={"title_source": "First Title", "author_source": "First Author"},
    )
    # Update only the title; author should remain.
    client.patch(
        f"/api/jobs/{job_id}/metadata",
        headers={**auth_headers, "content-type": "application/json"},
        json={"title_source": "Second Title"},
    )

    resp = client.get(f"/api/jobs/{job_id}", headers=auth_headers)
    body = resp.json()
    assert body["title_source"] == "Second Title"
    assert body["author_source"] == "First Author"


def test_patch_metadata_clears_field_with_empty_string(
    client, auth_headers, tmp_path: Path
) -> None:
    pdf = _seed_pdf(tmp_path / "sample.pdf")
    job_id = _upload(client, auth_headers, pdf)

    client.patch(
        f"/api/jobs/{job_id}/metadata",
        headers={**auth_headers, "content-type": "application/json"},
        json={"title_source": "to be cleared"},
    )
    client.patch(
        f"/api/jobs/{job_id}/metadata",
        headers={**auth_headers, "content-type": "application/json"},
        json={"title_source": ""},
    )
    resp = client.get(f"/api/jobs/{job_id}", headers=auth_headers)
    assert resp.json()["title_source"] is None


def test_patch_metadata_404_for_unknown_job(client, auth_headers) -> None:
    resp = client.patch(
        "/api/jobs/does-not-exist/metadata",
        headers={**auth_headers, "content-type": "application/json"},
        json={"title_source": "X"},
    )
    assert resp.status_code == 404


def test_re_export_409_when_not_completed(
    client, auth_headers, tmp_path: Path
) -> None:
    pdf = _seed_pdf(tmp_path / "sample.pdf")
    job_id = _upload(client, auth_headers, pdf)
    # Job is freshly UPLOADED, not COMPLETED
    resp = client.post(f"/api/jobs/{job_id}/re-export", headers=auth_headers)
    assert resp.status_code == 409
    assert "COMPLETED" in resp.text


def test_re_export_404_for_unknown_job(client, auth_headers) -> None:
    resp = client.post("/api/jobs/does-not-exist/re-export", headers=auth_headers)
    assert resp.status_code == 404


def test_create_job_auto_detects_metadata(client, auth_headers, tmp_path: Path) -> None:
    """The upload service auto-fills title/author from PDF metadata."""
    pdf_path = tmp_path / "auto-detect.pdf"
    doc = pymupdf.open()
    doc.new_page(width=400, height=500)
    doc.set_metadata({"title": "Detected Title", "author": "Detected Author"})
    doc.save(pdf_path)
    doc.close()

    job_id = _upload(client, auth_headers, pdf_path)
    resp = client.get(f"/api/jobs/{job_id}", headers=auth_headers)
    body = resp.json()
    assert body["title_source"] == "Detected Title"
    assert body["author_source"] == "Detected Author"

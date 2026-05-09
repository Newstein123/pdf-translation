"""Smoke tests for SSE event stream + UI surface."""

from pathlib import Path

import pymupdf

from app.db.models import Job, JobStatus
from app.db.session import SessionLocal
from app.services.jobs import record_event, set_job_status


def _make_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(72, 72, 540, 720), "Hello world. " * 8, fontsize=11)
    doc.save(path)
    doc.close()
    return path


def _seed_job(pdf_path: Path) -> str:
    with SessionLocal() as db:
        job = Job(
            original_filename=pdf_path.name,
            file_path=str(pdf_path),
            file_size_bytes=pdf_path.stat().st_size,
            status=JobStatus.UPLOADED,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job.id


def test_sse_streams_events_and_closes_on_terminal(client, tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_job(pdf)

    # Pre-seed an event and flip to a terminal status so the stream closes
    # quickly once the client connects.
    record_event(job_id, "extract_completed", {"total_blocks": 3})
    set_job_status(job_id, JobStatus.COMPLETED, progress_percent=100.0)
    record_event(job_id, "rebuild_completed", None)

    url = f"/api/jobs/{job_id}/events?token=test-token"
    with client.stream("GET", url) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = ""
        for chunk in resp.iter_text():
            body += chunk
            if "event: done" in body:
                break

    assert "event: hello" in body
    assert "event: extract_completed" in body
    assert "event: rebuild_completed" in body
    assert "event: done" in body
    assert '"status": "COMPLETED"' in body


def test_sse_rejects_missing_token(client) -> None:
    # job_id doesn't exist — but auth should fail before we even check.
    r = client.get("/api/jobs/anything/events")
    assert r.status_code == 401


def test_sse_rejects_bad_token(client, tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_job(pdf)
    set_job_status(job_id, JobStatus.COMPLETED)
    r = client.get(f"/api/jobs/{job_id}/events?token=wrong")
    assert r.status_code == 401


def test_sse_404_for_unknown_job(client) -> None:
    r = client.get("/api/jobs/00000000-0000-0000-0000-000000000000/events?token=test-token")
    assert r.status_code == 404


def test_index_html_served(client) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "PDF" in r.text
    assert "/static/app.js" in r.text


def test_static_assets_served(client) -> None:
    r_js = client.get("/static/app.js")
    r_css = client.get("/static/style.css")
    assert r_js.status_code == 200
    assert r_css.status_code == 200
    assert "EventSource" in r_js.text
    assert "--bg" in r_css.text


def test_download_accepts_query_token(client, auth_headers, tmp_path: Path) -> None:
    """Download endpoint must work without an Authorization header (?token= only)."""
    # Seed a fake completed job + Output row pointing at a real file on disk.
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_job(pdf)

    from app.db.models import Output, OutputFormat

    with SessionLocal() as db:
        db.add(
            Output(
                job_id=job_id,
                format=OutputFormat.PDF,
                file_path=str(pdf),
                size_bytes=pdf.stat().st_size,
            )
        )
        db.commit()

    # No Authorization header — only query token. Browsers can't set headers
    # on a plain anchor click, so this path matters for the UI.
    r = client.get(f"/api/jobs/{job_id}/outputs/pdf?token=test-token")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:4] == b"%PDF"

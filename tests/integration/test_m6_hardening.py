"""M6: cancel, resume, cost reporting, retention/purge."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pymupdf

from app.db.models import (
    Category,
    ChunkStatus,
    Job,
    JobStatus,
    Output,
    OutputFormat,
    TranslationChunk,
)
from app.db.session import SessionLocal
from app.services.translation import ChunkTranslationResult, DiscoveredName, TranslatedBlock


_TEXT = (
    "Alice walked through the Forest of Eldar to find Master Bell. "
    "The journey was long, and the night was cold. " * 8
)


def _make_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(72, 72, 540, 720), _TEXT, fontsize=11)
    doc.save(path)
    doc.close()
    return path


def _seed_job(pdf_path: Path, status: JobStatus = JobStatus.UPLOADED) -> str:
    with SessionLocal() as db:
        job = Job(
            original_filename=pdf_path.name,
            file_path=str(pdf_path),
            file_size_bytes=pdf_path.stat().st_size,
            status=status,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job.id


# ---------- cancel ----------


def test_cancel_endpoint_flips_status(client, auth_headers, tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_job(pdf, status=JobStatus.TRANSLATING)
    r = client.post(f"/api/jobs/{job_id}/cancel", headers=auth_headers)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "CANCELLED"


def test_cancel_rejects_terminal(client, auth_headers, tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_job(pdf, status=JobStatus.COMPLETED)
    r = client.post(f"/api/jobs/{job_id}/cancel", headers=auth_headers)
    assert r.status_code == 409


def test_translate_task_aborts_when_cancelled(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "src.pdf")
    # Drive job to AWAITING_CONFIRMATION via real extract+classify, set NOVEL,
    # then mark CANCELLED before we run translate_task.
    job_id = _seed_job(pdf, status=JobStatus.UPLOADED)

    from app.workers import tasks

    with patch("app.workers.tasks.classify_task.delay"):
        tasks.extract_task.run(job_id)
    tasks.classify_task.run(job_id)
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        job.category = Category.NOVEL
        job.status = JobStatus.CANCELLED  # cancel before translate runs
        db.commit()

    # The translate_task should bail before calling Gemini.
    with patch.object(tasks.translation, "translate_chunk") as mock_translate:
        tasks.translate_task.run(job_id)
    mock_translate.assert_not_called()

    with SessionLocal() as db:
        job = db.get(Job, job_id)
    assert job.status == JobStatus.CANCELLED


# ---------- resume ----------


def test_resume_reroutes_based_on_progress(client, auth_headers, tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_job(pdf, status=JobStatus.UPLOADED)

    from app.workers import tasks

    # Walk through extract+classify to land in AWAITING_CONFIRMATION (no chunks yet).
    with patch("app.workers.tasks.classify_task.delay"):
        tasks.extract_task.run(job_id)
    tasks.classify_task.run(job_id)
    with SessionLocal() as db:
        db.get(Job, job_id).category = Category.NOVEL
        db.get(Job, job_id).status = JobStatus.FAILED  # pretend translate crashed early
        db.commit()

    with patch("app.workers.tasks.translate_task.delay") as mock_delay:
        r = client.post(f"/api/jobs/{job_id}/resume", headers=auth_headers)
    assert r.status_code == 200
    mock_delay.assert_called_once_with(job_id)


def test_resume_resets_in_progress_chunks_to_pending(client, auth_headers, tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_job(pdf, status=JobStatus.FAILED)

    with SessionLocal() as db:
        db.add(
            TranslationChunk(
                job_id=job_id,
                chunk_index=0,
                page_range_start=1,
                page_range_end=1,
                source_text="x",
                status=ChunkStatus.IN_PROGRESS,
                attempt_count=2,
            )
        )
        db.add(
            TranslationChunk(
                job_id=job_id,
                chunk_index=1,
                page_range_start=1,
                page_range_end=1,
                source_text="y",
                status=ChunkStatus.FAILED,
                attempt_count=5,
            )
        )
        db.commit()

    with patch("app.workers.tasks.translate_task.delay"):
        r = client.post(f"/api/jobs/{job_id}/resume", headers=auth_headers)
    assert r.status_code == 200

    with SessionLocal() as db:
        chunks = (
            db.query(TranslationChunk)
            .filter(TranslationChunk.job_id == job_id)
            .order_by(TranslationChunk.chunk_index)
            .all()
        )
    assert all(c.status == ChunkStatus.PENDING for c in chunks)
    # Failed chunk's attempt count was zeroed for a clean retry budget.
    assert chunks[1].attempt_count == 0


def test_resume_rejects_completed(client, auth_headers, tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_job(pdf, status=JobStatus.COMPLETED)
    r = client.post(f"/api/jobs/{job_id}/resume", headers=auth_headers)
    assert r.status_code == 409


# ---------- cost reporting ----------


def test_cost_tokens_persisted_on_chunk_completion(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_job(pdf)

    from app.workers import tasks

    with patch("app.workers.tasks.classify_task.delay"):
        tasks.extract_task.run(job_id)
    tasks.classify_task.run(job_id)
    with SessionLocal() as db:
        db.get(Job, job_id).category = Category.NOVEL
        db.commit()

    def stub(*, blocks, **_kw):
        return ChunkTranslationResult(
            blocks=[
                TranslatedBlock(block_id=b["block_id"], translated="ဘာသာပြန်: " + b["source"][:20])
                for b in blocks
            ],
            new_names=[DiscoveredName(source="Alice", burmese="အဲလစ်")],
            summary_for_next_chunk="",
            total_tokens=1234,
        )

    with (
        patch.object(tasks.translation, "translate_chunk", side_effect=stub),
        patch("app.workers.tasks.rebuild_task.delay"),
    ):
        tasks.translate_task.run(job_id)

    with SessionLocal() as db:
        chunks = db.query(TranslationChunk).filter(TranslationChunk.job_id == job_id).all()
    assert chunks
    assert all(c.cost_tokens >= 1234 for c in chunks)


def test_total_tokens_in_status_response(client, auth_headers, tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_job(pdf, status=JobStatus.COMPLETED)
    with SessionLocal() as db:
        for i in range(3):
            db.add(
                TranslationChunk(
                    job_id=job_id,
                    chunk_index=i,
                    page_range_start=1,
                    page_range_end=1,
                    source_text="x",
                    status=ChunkStatus.COMPLETED,
                    cost_tokens=100,
                )
            )
        db.commit()
    r = client.get(f"/api/jobs/{job_id}", headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["total_tokens"] == 300


# ---------- retention ----------


def test_purge_dry_run_doesnt_delete(client, auth_headers, tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_job(pdf, status=JobStatus.COMPLETED)

    # Backdate updated_at so the purge picks it up.
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        old = datetime.now(timezone.utc) - timedelta(hours=99)
        job.updated_at = old.replace(tzinfo=None)
        job.completed_at = old.replace(tzinfo=None)
        db.commit()

    r = client.post(
        "/api/admin/purge?dry_run=true&override_hours=1", headers=auth_headers
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dry_run"] is True
    assert job_id in body["deleted_jobs"]
    # File still exists because dry_run=True.
    assert pdf.exists()


def test_purge_deletes_artifacts(client, auth_headers, tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_job(pdf, status=JobStatus.COMPLETED)

    # Add an Output row + file so we can verify it's removed.
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        old = datetime.now(timezone.utc) - timedelta(hours=99)
        job.updated_at = old.replace(tzinfo=None)
        job.completed_at = old.replace(tzinfo=None)
        db.add(
            Output(
                job_id=job_id,
                format=OutputFormat.PDF,
                file_path=str(pdf),
                size_bytes=pdf.stat().st_size,
            )
        )
        db.commit()

    r = client.post("/api/admin/purge?override_hours=1", headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["dry_run"] is False
    assert job_id in body["deleted_jobs"]
    assert not pdf.exists()
    with SessionLocal() as db:
        outputs = db.query(Output).filter(Output.job_id == job_id).all()
    assert outputs == []


def test_purge_skips_active_jobs(client, auth_headers, tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_job(pdf, status=JobStatus.TRANSLATING)
    with SessionLocal() as db:
        old = datetime.now(timezone.utc) - timedelta(hours=99)
        db.get(Job, job_id).updated_at = old.replace(tzinfo=None)
        db.commit()
    r = client.post("/api/admin/purge?override_hours=1", headers=auth_headers)
    assert r.status_code == 200
    assert job_id not in r.json()["deleted_jobs"]
    assert pdf.exists()

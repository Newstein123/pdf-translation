"""Integration tests that run the worker tasks synchronously (no Redis required).

We exercise the real extraction service and stub Gemini at the classifier
boundary, validating that:
  - extract_task writes structure.json + DocumentStructure row
  - classify_task transitions the job to AWAITING_CONFIRMATION when Gemini works
  - classify_task falls back to UNKNOWN when Gemini is unavailable
"""

from pathlib import Path
from unittest.mock import patch

import pymupdf

from app.db.models import Category, DocumentStructure, Job, JobStatus
from app.db.session import SessionLocal
from app.services.classification import ClassificationResult


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


_DEFAULT_TEXT = (
    "This is a sample sentence used to seed test PDFs with enough characters "
    "to clear the scanned-page detection threshold and let the extractor produce "
    "a meaningful normalized JSON structure for assertions."
)


def _make_pdf(path: Path, *, text: str = _DEFAULT_TEXT) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    doc.save(path)
    doc.close()
    return path


def test_extract_task_persists_structure(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_job(pdf)

    from app.workers.tasks import extract_task

    # Prevent automatic chain into classify_task; we test it separately.
    with patch("app.workers.tasks.classify_task.delay"):
        extract_task.run(job_id)

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        assert job is not None
        ds = db.query(DocumentStructure).filter_by(job_id=job_id).one()
        assert Path(ds.structured_json_path).exists()
        assert ds.total_blocks > 0
        assert job.is_scanned is False


def test_classify_task_uses_classifier_result(tmp_path: Path) -> None:
    pdf = _make_pdf(
        tmp_path / "src.pdf",
        text=(
            "Once upon a time there lived a brave knight in a kingdom by the sea. "
            "He set forth on a long journey to seek fortune and adventure."
        ),
    )
    job_id = _seed_job(pdf)

    from app.workers.tasks import classify_task, extract_task

    with patch("app.workers.tasks.classify_task.delay"):
        extract_task.run(job_id)

    fake_result = ClassificationResult(
        category=Category.NOVEL,
        confidence=0.92,
        reason="narrative prose",
        auto_detected=True,
    )
    with patch(
        "app.workers.tasks.classification.classify_document", return_value=fake_result
    ) as mock_classify:
        classify_task.run(job_id)

    mock_classify.assert_called_once()
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        assert job.category == Category.NOVEL
        assert job.category_auto_detected is True
        assert job.status == JobStatus.AWAITING_CONFIRMATION


def test_classify_task_falls_back_to_unknown_without_gemini(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_job(pdf)

    from app.workers.tasks import classify_task, extract_task

    with patch("app.workers.tasks.classify_task.delay"):
        extract_task.run(job_id)

    # No GEMINI_API_KEY in test env -> classifier returns UNKNOWN auto_detected=False.
    classify_task.run(job_id)

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        assert job.category == Category.UNKNOWN
        assert job.category_auto_detected is False
        assert job.status == JobStatus.AWAITING_CONFIRMATION


def test_patch_category_and_start(client, auth_headers, tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_job(pdf)

    # Move job through extract+classify so it lands in AWAITING_CONFIRMATION.
    from app.workers.tasks import classify_task, extract_task

    with patch("app.workers.tasks.classify_task.delay"):
        extract_task.run(job_id)
    classify_task.run(job_id)  # falls back to UNKNOWN without Gemini

    # Cannot start without a category set
    r = client.post(f"/api/jobs/{job_id}/start", headers=auth_headers)
    assert r.status_code == 400

    # PATCH category to NOVEL
    r = client.patch(
        f"/api/jobs/{job_id}/category", headers=auth_headers, json={"category": "NOVEL"}
    )
    assert r.status_code == 200, r.text
    assert r.json()["category"] == "NOVEL"
    assert r.json()["category_auto_detected"] is False

    # Now start translation (translate_task is still a stub for M3)
    with patch("app.workers.tasks.translate_task.delay") as mock_delay:
        r = client.post(f"/api/jobs/{job_id}/start", headers=auth_headers)
    assert r.status_code == 200, r.text
    mock_delay.assert_called_once_with(job_id)


def test_get_structure_returns_summary(client, auth_headers, tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_job(pdf)

    from app.workers.tasks import extract_task

    with patch("app.workers.tasks.classify_task.delay"):
        extract_task.run(job_id)

    r = client.get(f"/api/jobs/{job_id}/structure", headers=auth_headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == job_id
    assert body["total_blocks"] > 0
    assert body["total_pages"] >= 1
    assert isinstance(body["pages_preview"], list)

"""End-to-end translate_task integration tests with Gemini stubbed.

We seed a job through extraction, set its category, then run translate_task.
The Gemini call is replaced by a deterministic stub that returns a fixed
"translation" so we can assert chunk persistence, name-registry updates,
status transitions, and the merged output endpoint without hitting the API.
"""

from pathlib import Path
from unittest.mock import patch

import pymupdf

from app.db.models import Category, ChunkStatus, Job, JobStatus, NameRegistry, TranslationChunk
from app.db.session import SessionLocal
from app.services.translation import ChunkTranslationResult, DiscoveredName, TranslatedBlock


_TEXT = (
    "Alice walked through the Forest of Eldar to find Master Bell. "
    "The journey was long, and the night was cold. " * 8
)


def _make_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), _TEXT, fontsize=11)
    doc.save(path)
    doc.close()
    return path


def _seed_through_classify(pdf_path: Path) -> str:
    """Run extract+classify so the job lands in AWAITING_CONFIRMATION, then set
    category to NOVEL. Returns the job_id."""
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
        job_id = job.id

    from app.workers.tasks import classify_task, extract_task

    with patch("app.workers.tasks.classify_task.delay"):
        extract_task.run(job_id)
    classify_task.run(job_id)  # falls back to UNKNOWN without Gemini

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        job.category = Category.NOVEL
        db.commit()
    return job_id


def _stub_translate_chunk(*, blocks, **_kwargs) -> ChunkTranslationResult:
    """Return a fake but well-formed translation for every block."""
    translated = [
        TranslatedBlock(block_id=b["block_id"], translated="[BURMESE] " + b["source"])
        for b in blocks
    ]
    return ChunkTranslationResult(
        blocks=translated,
        new_names=[DiscoveredName(source="Alice", burmese="အဲလစ်")],
        summary_for_next_chunk="previous chunk happened",
    )


def test_translate_task_runs_full_pipeline(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_through_classify(pdf)

    from app.workers import tasks

    with (
        patch.object(tasks.translation, "translate_chunk", side_effect=_stub_translate_chunk),
        patch("app.workers.tasks.rebuild_task.delay"),  # don't auto-rebuild here
    ):
        tasks.translate_task.run(job_id)

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        chunks = (
            db.query(TranslationChunk)
            .filter(TranslationChunk.job_id == job_id)
            .order_by(TranslationChunk.chunk_index)
            .all()
        )
        names = db.query(NameRegistry).filter(NameRegistry.job_id == job_id).all()

    # translate_task hands off to rebuild_task; status should not yet be COMPLETED.
    assert job.status == JobStatus.POSTPROCESSING
    assert len(chunks) >= 1
    assert all(c.status == ChunkStatus.COMPLETED for c in chunks)
    assert all(c.translated_text and c.translated_text.startswith("[BURMESE]") for c in chunks)
    # Name registry was upserted (Alice should appear once due to unique index).
    sources = [n.source_name for n in names]
    assert sources.count("Alice") == 1


def test_translation_endpoint_returns_merged_text(client, auth_headers, tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_through_classify(pdf)

    from app.workers import tasks

    with (
        patch.object(tasks.translation, "translate_chunk", side_effect=_stub_translate_chunk),
        patch("app.workers.tasks.rebuild_task.delay"),
    ):
        tasks.translate_task.run(job_id)

    r = client.get(f"/api/jobs/{job_id}/translation", headers=auth_headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == job_id
    assert body["chunks_completed"] >= 1
    assert body["chunks_completed"] == body["chunks_total"]
    assert "[BURMESE]" in body["plain_text"]
    assert any("[BURMESE]" in p["translated_text"] for p in body["pages"])


def test_chunks_endpoint_lists_chunks(client, auth_headers, tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_through_classify(pdf)

    from app.workers import tasks

    with (
        patch.object(tasks.translation, "translate_chunk", side_effect=_stub_translate_chunk),
        patch("app.workers.tasks.rebuild_task.delay"),
    ):
        tasks.translate_task.run(job_id)

    r = client.get(f"/api/jobs/{job_id}/chunks", headers=auth_headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body, list) and len(body) >= 1
    assert all(c["status"] == "COMPLETED" for c in body)


def test_glossary_upload_and_list(client, auth_headers, tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_through_classify(pdf)

    glossary = '[{"term":"Eldar","translation":"အဲလ်ဒါ"}]'.encode("utf-8")
    r = client.post(
        f"/api/jobs/{job_id}/glossary",
        headers=auth_headers,
        files={"file": ("g.json", glossary, "application/json")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["saved"] == 1

    r = client.get(f"/api/jobs/{job_id}/glossary", headers=auth_headers)
    assert r.status_code == 200, r.text
    assert [g["term_source"] for g in r.json()] == ["Eldar"]


def test_translate_task_aborts_immediately_on_model_missing(tmp_path: Path) -> None:
    """A 404 model_not_found should fail the WHOLE job after one chunk attempt
    rather than retrying every chunk × max_retries (the bug we hit live)."""
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_through_classify(pdf)

    from app.workers import tasks
    from app.services.translation import TranslationModelMissing

    call_count = {"n": 0}

    def boom(**_kw):
        call_count["n"] += 1
        raise TranslationModelMissing(
            "model 'gemini-X-pro' not found: 404 NOT_FOUND. Update GEMINI_TRANSLATION_MODEL"
        )

    with (
        patch.object(tasks.translation, "translate_chunk", side_effect=boom),
        patch("app.workers.tasks.rebuild_task.delay"),
    ):
        tasks.translate_task.run(job_id)

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        chunks = (
            db.query(TranslationChunk)
            .filter(TranslationChunk.job_id == job_id)
            .order_by(TranslationChunk.chunk_index)
            .all()
        )

    assert job.status == JobStatus.FAILED
    # All chunks marked FAILED, not RETRY.
    assert all(c.status == ChunkStatus.FAILED for c in chunks)
    # Critical: translate_chunk called ONCE per chunk, not multiple times.
    assert call_count["n"] == len(chunks), (
        f"expected one call per chunk ({len(chunks)}), got {call_count['n']} "
        f"(retries should not happen for model-missing errors)"
    )
    assert job.error_message and "not found" in job.error_message.lower()


def test_translate_task_marks_failed_when_translation_errors(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_through_classify(pdf)

    from app.workers import tasks
    from app.services.translation import TranslationError

    def always_error(**_kw) -> ChunkTranslationResult:
        raise TranslationError("simulated failure")

    with patch.object(tasks.translation, "translate_chunk", side_effect=always_error):
        tasks.translate_task.run(job_id)

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        chunks = db.query(TranslationChunk).filter(TranslationChunk.job_id == job_id).all()

    assert job.status == JobStatus.FAILED
    assert all(c.status == ChunkStatus.FAILED for c in chunks)

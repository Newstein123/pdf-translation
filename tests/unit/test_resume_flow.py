"""Regression tests for the Resume flow.

Bug we're guarding against: ``reset_for_resume`` previously cleared the
``error_message`` but left ``job.status = CANCELLED`` intact. The worker's
``is_cancelled(job_id)`` check reads the DB status as its first action,
so the resumed task short-circuited and silently did nothing — which the
user saw as "Resume button doesn't work."
"""
from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from app.db.models import (
    Category,
    ChunkStatus,
    Job,
    JobStatus,
    TranslationChunk,
)
from app.services.jobs import is_cancelled, reset_for_resume


@pytest.fixture
def session(db_engine):
    with Session(bind=db_engine) as s:
        yield s
        s.rollback()


def _seed_job(session, **kwargs) -> Job:
    j = Job(
        original_filename=kwargs.pop("original_filename", "t.pdf"),
        file_path=kwargs.pop("file_path", "/tmp/t.pdf"),
        file_size_bytes=kwargs.pop("file_size_bytes", 1),
        status=kwargs.pop("status", JobStatus.UPLOADED),
        category=kwargs.pop("category", Category.NOVEL),
        **kwargs,
    )
    session.add(j)
    session.flush()
    return j


def test_reset_for_resume_clears_cancelled_status(session) -> None:
    """A cancelled job's status must be reset so ``is_cancelled`` won't
    short-circuit the worker."""
    job = _seed_job(
        session,
        status=JobStatus.CANCELLED,
        error_message="cancelled by user",
    )
    session.commit()
    # Pre-condition: is_cancelled returns True.
    assert is_cancelled(job.id) is True

    reset_for_resume(session, job)

    # Post-condition: status is no longer CANCELLED + is_cancelled is False
    # so the worker will actually do work when re-enqueued.
    assert job.status != JobStatus.CANCELLED
    assert is_cancelled(job.id) is False
    assert job.error_message is None


def test_reset_for_resume_clears_failed_status(session) -> None:
    job = _seed_job(
        session,
        status=JobStatus.FAILED,
        error_message="something exploded",
    )
    session.commit()
    reset_for_resume(session, job)
    assert job.status != JobStatus.FAILED
    assert job.error_message is None


def test_reset_for_resume_preserves_in_flight_status(session) -> None:
    """A job that's still TRANSLATING (worker crashed mid-run) shouldn't
    have its status changed — the resumed task will overwrite it anyway,
    and we don't want to flip it back to UPLOADED briefly."""
    job = _seed_job(session, status=JobStatus.TRANSLATING)
    session.commit()
    reset_for_resume(session, job)
    assert job.status == JobStatus.TRANSLATING


def test_reset_for_resume_reverts_in_progress_chunks_to_pending(session) -> None:
    job = _seed_job(session, status=JobStatus.CANCELLED)
    chunk = TranslationChunk(
        job_id=job.id,
        chunk_index=0,
        page_range_start=1,
        page_range_end=1,
        source_text="x",
        token_count=1,
        status=ChunkStatus.IN_PROGRESS,
        attempt_count=2,
    )
    session.add(chunk)
    session.commit()

    reset_for_resume(session, job)

    session.refresh(chunk)
    assert chunk.status == ChunkStatus.PENDING


def test_reset_for_resume_resets_failed_chunks(session) -> None:
    job = _seed_job(session, status=JobStatus.FAILED)
    chunk = TranslationChunk(
        job_id=job.id,
        chunk_index=0,
        page_range_start=1,
        page_range_end=1,
        source_text="x",
        token_count=1,
        status=ChunkStatus.FAILED,
        attempt_count=5,
    )
    session.add(chunk)
    session.commit()

    reset_for_resume(session, job)

    session.refresh(chunk)
    assert chunk.status == ChunkStatus.PENDING
    assert chunk.attempt_count == 0

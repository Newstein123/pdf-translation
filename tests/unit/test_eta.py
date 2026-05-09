"""ETA estimation on Job.estimated_seconds_remaining."""

from datetime import datetime, timedelta

from app.db.models import (
    ChunkStatus,
    Job,
    JobStatus,
    TranslationChunk,
)


def _job_with_chunks(*, status: JobStatus, chunks: list[TranslationChunk]) -> Job:
    job = Job(
        original_filename="x.pdf",
        file_path="/tmp/x.pdf",
        file_size_bytes=1,
        status=status,
    )
    # Manually attach chunks (bypass the relationship — no DB session here).
    job.chunks = chunks  # type: ignore[assignment]
    return job


def test_eta_none_when_not_translating() -> None:
    job = _job_with_chunks(status=JobStatus.UPLOADED, chunks=[])
    assert job.estimated_seconds_remaining is None

    job = _job_with_chunks(status=JobStatus.AWAITING_CONFIRMATION, chunks=[])
    assert job.estimated_seconds_remaining is None

    job = _job_with_chunks(status=JobStatus.COMPLETED, chunks=[])
    assert job.estimated_seconds_remaining is None


def test_eta_none_when_no_completed_chunks() -> None:
    chunks = [
        TranslationChunk(
            chunk_index=i,
            page_range_start=1,
            page_range_end=1,
            source_text="x",
            status=ChunkStatus.PENDING,
        )
        for i in range(5)
    ]
    job = _job_with_chunks(status=JobStatus.TRANSLATING, chunks=chunks)
    assert job.estimated_seconds_remaining is None


def test_eta_extrapolates_from_completed_chunks() -> None:
    now = datetime.utcnow()
    earliest_start = now - timedelta(seconds=60)
    last_complete = now - timedelta(seconds=10)
    chunks = [
        TranslationChunk(
            chunk_index=0,
            page_range_start=1,
            page_range_end=1,
            source_text="x",
            status=ChunkStatus.COMPLETED,
            started_at=earliest_start,
            completed_at=last_complete,
        ),
        TranslationChunk(
            chunk_index=1,
            page_range_start=2,
            page_range_end=2,
            source_text="x",
            status=ChunkStatus.COMPLETED,
            started_at=earliest_start + timedelta(seconds=10),
            completed_at=last_complete - timedelta(seconds=5),
        ),
        TranslationChunk(
            chunk_index=2,
            page_range_start=3,
            page_range_end=3,
            source_text="x",
            status=ChunkStatus.PENDING,
        ),
        TranslationChunk(
            chunk_index=3,
            page_range_start=4,
            page_range_end=4,
            source_text="x",
            status=ChunkStatus.PENDING,
        ),
    ]
    job = _job_with_chunks(status=JobStatus.TRANSLATING, chunks=chunks)

    # ~60s elapsed, 2/4 done → rate ~0.033/s → remaining (2) takes ~60s + 20s rebuild buffer.
    eta = job.estimated_seconds_remaining
    assert eta is not None
    assert 60 <= eta <= 120, f"unexpected ETA: {eta}s"


def test_eta_just_rebuild_buffer_when_translation_done() -> None:
    earlier = datetime.utcnow() - timedelta(seconds=120)
    chunks = [
        TranslationChunk(
            chunk_index=0,
            page_range_start=1,
            page_range_end=1,
            source_text="x",
            status=ChunkStatus.COMPLETED,
            started_at=earlier,
            completed_at=earlier + timedelta(seconds=60),
        ),
    ]
    job = _job_with_chunks(status=JobStatus.POSTPROCESSING, chunks=chunks)
    assert job.estimated_seconds_remaining == 20

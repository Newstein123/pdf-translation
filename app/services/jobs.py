from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import (
    Category,
    ChunkStatus,
    DocumentStructure,
    GlossaryEntry,
    Job,
    JobEvent,
    JobStatus,
    NameRegistry,
    Output,
    OutputFormat,
    StyleProfile,
    TranslationChunk,
)
from app.db.session import SessionLocal
from app.services.glossary import ParsedTerm


def create_job(
    db: Session,
    *,
    original_filename: str,
    file_path: str,
    file_size_bytes: int,
) -> Job:
    # M14: auto-fill title/author from PDF metadata so the AWAITING_CONFIRMATION
    # form can show a useful default. Defensive against malformed PDFs — any
    # detection failure leaves the fields NULL and the user fills them in.
    title_source: str | None = None
    author_source: str | None = None
    try:
        from pathlib import Path

        from app.services import layout_detection

        detected = layout_detection.detect_book_metadata(Path(file_path))
        title_source = detected.title
        author_source = detected.author
    except Exception:
        # Don't block upload on detection issues — log and move on.
        import logging

        logging.getLogger(__name__).exception(
            "metadata auto-detect failed for %s", file_path
        )

    job = Job(
        original_filename=original_filename,
        file_path=file_path,
        file_size_bytes=file_size_bytes,
        status=JobStatus.UPLOADED,
        title_source=title_source,
        author_source=author_source,
    )
    db.add(job)
    db.flush()
    db.add(JobEvent(job_id=job.id, event_type="uploaded", payload_json={"size": file_size_bytes}))
    db.commit()
    db.refresh(job)
    return job


def get_job(db: Session, job_id: str) -> Job | None:
    return db.get(Job, job_id)


def set_job_status(
    job_id: str,
    status: JobStatus,
    *,
    error_message: str | None = None,
    progress_percent: float | None = None,
) -> None:
    """Worker-safe status transition using its own session."""
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None:
            return
        job.status = status
        if error_message is not None:
            job.error_message = error_message
        if progress_percent is not None:
            job.progress_percent = progress_percent
        if status == JobStatus.COMPLETED:
            job.completed_at = datetime.now(timezone.utc)
        db.commit()


def record_event(job_id: str, event_type: str, payload: dict[str, Any] | None = None) -> None:
    """Persist a JobEvent with microsecond-precision timestamps.

    The default ``server_default=func.now()`` uses ``CURRENT_TIMESTAMP`` which
    is second-precision on SQLite. When two events fire within the same
    second (e.g. ``glossary_suggestions_started`` followed immediately by
    ``glossary_suggestions_ready`` on a no-op job), they share a created_at
    value and the secondary ``id`` sort is by random UUID — so SSE replays
    them out of order. Setting ``created_at`` from Python preserves
    microsecond precision and keeps replay chronological."""
    with SessionLocal() as db:
        db.add(
            JobEvent(
                job_id=job_id,
                event_type=event_type,
                payload_json=payload,
                created_at=datetime.utcnow(),
            )
        )
        db.commit()


def save_extraction_result(
    job_id: str,
    *,
    structured_json_path: str,
    total_blocks: int,
    is_scanned: bool,
) -> None:
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None:
            return
        job.is_scanned = is_scanned
        existing = db.query(DocumentStructure).filter_by(job_id=job_id).one_or_none()
        if existing is None:
            db.add(
                DocumentStructure(
                    job_id=job_id,
                    structured_json_path=structured_json_path,
                    total_blocks=total_blocks,
                )
            )
        else:
            existing.structured_json_path = structured_json_path
            existing.total_blocks = total_blocks
        db.commit()


def save_classification(
    job_id: str,
    *,
    category: Category,
    auto_detected: bool,
) -> None:
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None:
            return
        job.category = category
        job.category_auto_detected = auto_detected
        db.commit()


def is_cancelled(job_id: str) -> bool:
    """Cheap status check for cooperative cancellation inside long-running tasks."""
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        return job is not None and job.status == JobStatus.CANCELLED


def request_cancel(db: Session, job: Job) -> Job:
    job.status = JobStatus.CANCELLED
    job.error_message = "cancelled by user"
    db.commit()
    db.refresh(job)
    return job


def reset_for_resume(db: Session, job: Job) -> Job:
    """Prepare a job to be resumed: clear error, clear cancel state, and revert
    any IN_PROGRESS chunks to PENDING so the next translate sweep picks them up.

    Critically: when the job's status is CANCELLED or FAILED, we must
    reset it to a non-terminal value before the worker re-runs. The worker's
    ``is_cancelled(job_id)`` check reads the DB status as its first action;
    a leftover CANCELLED status would short-circuit the entire resume to
    a no-op (the symptom users see as "Resume button doesn't work").
    The downstream task overwrites status to TRANSLATING / EXTRACTING /
    REBUILDING immediately, so the value we reset to is just a transient
    placeholder — UPLOADED is the safest choice (covers all re-entry
    points: extract, translate, rebuild)."""
    job.error_message = None
    if job.status in {JobStatus.CANCELLED, JobStatus.FAILED}:
        job.status = JobStatus.UPLOADED
    db.query(TranslationChunk).filter(
        TranslationChunk.job_id == job.id,
        TranslationChunk.status.in_({ChunkStatus.IN_PROGRESS, ChunkStatus.RETRY}),
    ).update({TranslationChunk.status: ChunkStatus.PENDING}, synchronize_session=False)
    # FAILED chunks are also reset so retries past max get another chance on resume.
    db.query(TranslationChunk).filter(
        TranslationChunk.job_id == job.id,
        TranslationChunk.status == ChunkStatus.FAILED,
    ).update(
        {
            TranslationChunk.status: ChunkStatus.PENDING,
            TranslationChunk.attempt_count: 0,
        },
        synchronize_session=False,
    )
    db.commit()
    db.refresh(job)
    return job


def update_category(db: Session, job: Job, category: Category) -> Job:
    job.category = category
    job.category_auto_detected = False
    db.commit()
    db.refresh(job)
    return job


def update_style_profile(db: Session, job: Job, profile: StyleProfile | None) -> Job:
    """Set or clear the style profile on a job.

    Passing ``profile=None`` clears it — the translation pipeline then
    falls back to the category-specific base prompt + the resolved voice
    card, which is the recommended path for PERSONAL_DEVELOPMENT (no
    PD-scoped profile exists yet)."""
    job.style_profile_id = profile.id if profile is not None else None
    db.commit()
    db.refresh(job)
    return job


def set_default_profile_if_missing(job_id: str) -> None:
    """Worker-safe: assign the seeded default profile for the job's category
    if no profile has been chosen yet. Called after classify_task lands a
    category so the user sees a default in AWAITING_CONFIRMATION."""
    from app.services.style_profiles import get_default_profile_for_category

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None or job.style_profile_id is not None:
            return
        profile = get_default_profile_for_category(db, job.category)
        if profile is None:
            return
        job.style_profile_id = profile.id
        db.commit()


def replace_glossary(db: Session, job_id: str, terms: list[ParsedTerm]) -> int:
    """Replace the job's glossary with ``terms``. Returns count saved."""
    db.query(GlossaryEntry).filter(GlossaryEntry.job_id == job_id).delete()
    for t in terms:
        db.add(
            GlossaryEntry(
                job_id=job_id,
                term_source=t.term_source,
                term_target=t.term_target,
                case_sensitive=t.case_sensitive,
            )
        )
    db.commit()
    return len(terms)


def list_glossary(db: Session, job_id: str) -> list[GlossaryEntry]:
    return (
        db.query(GlossaryEntry)
        .filter(GlossaryEntry.job_id == job_id)
        .order_by(GlossaryEntry.term_source)
        .all()
    )


def upsert_name(
    job_id: str, *, source_name: str, burmese_phonetic: str, chunk_index: int
) -> None:
    """Idempotent name-registry upsert. Concurrent-safe via the unique index
    on (job_id, source_name): a duplicate insert will raise IntegrityError,
    which we swallow and treat as 'already known'."""
    from sqlalchemy.exc import IntegrityError

    with SessionLocal() as db:
        existing = (
            db.query(NameRegistry)
            .filter(NameRegistry.job_id == job_id, NameRegistry.source_name == source_name)
            .one_or_none()
        )
        if existing is not None:
            existing.occurrence_count += 1
            db.commit()
            return
        try:
            db.add(
                NameRegistry(
                    job_id=job_id,
                    source_name=source_name,
                    burmese_phonetic=burmese_phonetic,
                    first_seen_chunk_index=chunk_index,
                )
            )
            db.commit()
        except IntegrityError:
            db.rollback()


def list_names(job_id: str) -> list[NameRegistry]:
    with SessionLocal() as db:
        return (
            db.query(NameRegistry)
            .filter(NameRegistry.job_id == job_id)
            .order_by(NameRegistry.first_seen_chunk_index, NameRegistry.source_name)
            .all()
        )


def reset_chunks(db: Session, job_id: str) -> None:
    db.query(TranslationChunk).filter(TranslationChunk.job_id == job_id).delete()
    db.commit()


def upsert_output(
    job_id: str,
    *,
    fmt: OutputFormat,
    file_path: str,
    size_bytes: int,
) -> None:
    """Replace any prior output of this format with the new one."""
    with SessionLocal() as db:
        existing = (
            db.query(Output)
            .filter(Output.job_id == job_id, Output.format == fmt)
            .one_or_none()
        )
        if existing is not None:
            existing.file_path = file_path
            existing.size_bytes = size_bytes
        else:
            db.add(
                Output(
                    job_id=job_id,
                    format=fmt,
                    file_path=file_path,
                    size_bytes=size_bytes,
                )
            )
        db.commit()


def list_outputs(db: Session, job_id: str) -> list[Output]:
    return (
        db.query(Output)
        .filter(Output.job_id == job_id)
        .order_by(Output.format)
        .all()
    )


def insert_chunks(db: Session, job_id: str, specs: list) -> None:
    """``specs`` is a list of ``ChunkSpec`` from chunking.build_chunks."""
    for spec in specs:
        db.add(
            TranslationChunk(
                job_id=job_id,
                chunk_index=spec.chunk_index,
                page_range_start=spec.page_range_start,
                page_range_end=spec.page_range_end,
                source_text=spec.source_text,
                blocks_json=spec.blocks,
                token_count=spec.token_count,
            )
        )
    db.commit()

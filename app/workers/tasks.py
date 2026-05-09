from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.db.models import (
    Category,
    ChunkStatus,
    ChunkTranslationPass,
    ExportLayoutMode,
    GlossaryEntry,
    Job,
    JobStatus,
    NameRegistry,
    OutputFormat,
    PassKind,
    PassStatus,
    PromptPurpose,
    QualityScore,
    StyleProfile,
    TranslationChunk,
)
from app.db.session import SessionLocal
from app.services import (
    chunking,
    classification,
    extraction,
    glossary_suggester,
    layout_detection,
    pdf_template,
    postprocess,
    rebuild_epub,
    rebuild_pdf,
    translation,
    voice_card,
)
from app.services.prompt_versions import get_or_create_prompt_version
from app.services.running_context import merge_running_context
from app.services.jobs import (
    insert_chunks,
    is_cancelled,
    list_names,
    record_event,
    reset_chunks,
    save_classification,
    save_extraction_result,
    set_default_profile_if_missing,
    set_job_status,
    upsert_name,
    upsert_output,
)
from app.workers.celery_app import celery_app

log = logging.getLogger(__name__)


@celery_app.task(name="jobs.extract", bind=True)
def extract_task(self, job_id: str) -> str:
    """Stage 2: PDF extraction → normalized JSON, then chain into classification."""
    log.info("extract_task start job_id=%s", job_id)
    set_job_status(job_id, JobStatus.EXTRACTING, progress_percent=3.0)
    record_event(job_id, "extract_started", {"task_id": self.request.id})

    settings = get_settings()
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None:
            log.error("extract_task: job %s not found", job_id)
            return job_id
        pdf_path = Path(job.file_path)

    job_intermediate = settings.intermediate_dir / job_id
    try:
        result = extraction.extract_pdf(pdf_path, output_dir=job_intermediate)
    except Exception as e:
        log.exception("extraction failed job_id=%s", job_id)
        set_job_status(job_id, JobStatus.FAILED, error_message=f"extraction failed: {e}")
        record_event(job_id, "extract_failed", {"error": str(e)})
        return job_id

    save_extraction_result(
        job_id,
        structured_json_path=str(result.structured_json_path),
        total_blocks=result.total_blocks,
        is_scanned=result.is_scanned,
    )
    record_event(
        job_id,
        "extract_completed",
        {
            "total_pages": result.total_pages,
            "total_blocks": result.total_blocks,
            "is_scanned": result.is_scanned,
            "scanned_pages": result.scanned_pages,
        },
    )

    classify_task.delay(job_id)
    return job_id


@celery_app.task(name="jobs.classify", bind=True)
def classify_task(self, job_id: str) -> str:
    """Stage 3: classify document; on success, await user confirmation."""
    log.info("classify_task start job_id=%s", job_id)
    set_job_status(job_id, JobStatus.CLASSIFYING, progress_percent=14.0)
    record_event(job_id, "classify_started", {"task_id": self.request.id})

    settings = get_settings()
    structure_path = settings.intermediate_dir / job_id / "structure.json"
    if not structure_path.exists():
        msg = f"structure.json missing at {structure_path}"
        log.error(msg)
        set_job_status(job_id, JobStatus.FAILED, error_message=msg)
        record_event(job_id, "classify_failed", {"error": msg})
        return job_id

    excerpt = _excerpt_from_structure(structure_path, settings.classification_excerpt_chars)

    try:
        result = classification.classify_document(excerpt)
    except Exception as e:
        log.exception("classification failed job_id=%s", job_id)
        set_job_status(job_id, JobStatus.FAILED, error_message=f"classification failed: {e}")
        record_event(job_id, "classify_failed", {"error": str(e)})
        return job_id

    save_classification(job_id, category=result.category, auto_detected=result.auto_detected)
    # Pin a default style profile based on the resolved category so the user
    # sees one pre-selected at the AWAITING_CONFIRMATION stage. No-op if the
    # uploader already chose one.
    set_default_profile_if_missing(job_id)
    record_event(
        job_id,
        "classify_completed",
        {
            "category": result.category.value,
            "confidence": result.confidence,
            "reason": result.reason,
            "auto_detected": result.auto_detected,
        },
    )
    # M8: kick off glossary auto-suggest before AWAITING_CONFIRMATION so the
    # user reviews proposals + confirms category in the same pass.
    try:
        suggest_glossary_task.delay(job_id)
    except Exception:
        log.exception("failed to enqueue suggest_glossary_task for job_id=%s", job_id)
    set_job_status(job_id, JobStatus.AWAITING_CONFIRMATION, progress_percent=22.0)
    return job_id


@celery_app.task(name="jobs.suggest_glossary", bind=True)
def suggest_glossary_task(self, job_id: str) -> str:
    """Stage 3.5 (M8): propose glossary entries the user should approve before
    translation begins. Best-effort — failure here doesn't block the job."""
    log.info("suggest_glossary_task start job_id=%s", job_id)
    record_event(job_id, "glossary_suggestions_started", {"task_id": self.request.id})

    settings = get_settings()
    structure_path = settings.intermediate_dir / job_id / "structure.json"
    if not structure_path.exists():
        log.warning(
            "suggest_glossary_task: structure.json missing for job_id=%s; skipping", job_id
        )
        record_event(
            job_id,
            "glossary_suggestions_ready",
            {"sampled": 0, "proposed": 0, "kept": 0, "skipped": "no_structure"},
        )
        return job_id

    with SessionLocal() as db:
        try:
            counts = glossary_suggester.run_for_job(db, job_id, structure_path)
        except Exception as e:
            log.exception("glossary suggester raised for job_id=%s", job_id)
            record_event(job_id, "glossary_suggestions_failed", {"error": str(e)[:300]})
            return job_id

    record_event(job_id, "glossary_suggestions_ready", counts)
    return job_id


@celery_app.task(name="jobs.translate", bind=True)
def translate_task(self, job_id: str) -> str:
    """Stage 5–7: chunk + translate (parallel) + postprocess.

    The chunk pipeline is sequenced: we first build chunks deterministically and
    persist them, then dispatch a bounded thread pool to translate them. The
    name registry is updated in serialized fashion (per-chunk DB upserts behind
    the unique index), and each chunk's earlier ``summary_for_next_chunk``
    becomes the next chunk's rolling context.
    """
    log.info("translate_task start job_id=%s", job_id)
    if is_cancelled(job_id):
        record_event(job_id, "translate_cancelled", {"task_id": self.request.id})
        return job_id
    set_job_status(job_id, JobStatus.TRANSLATING, progress_percent=25.0)
    record_event(job_id, "translate_started", {"task_id": self.request.id})

    settings = get_settings()
    structure_path = settings.intermediate_dir / job_id / "structure.json"
    if not structure_path.exists():
        msg = f"structure.json missing at {structure_path}"
        log.error(msg)
        set_job_status(job_id, JobStatus.FAILED, error_message=msg)
        record_event(job_id, "translate_failed", {"error": msg})
        return job_id

    # Load category + glossary; build chunks; persist them.
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None:
            return job_id
        category = job.category
        if category not in {
            Category.NOVEL,
            Category.EDUCATIONAL,
            Category.PERSONAL_DEVELOPMENT,
        }:
            msg = f"category not set; current={category.value}"
            set_job_status(job_id, JobStatus.FAILED, error_message=msg)
            record_event(job_id, "translate_failed", {"error": msg})
            return job_id

        # Resolve user-supplied skip-pages. Parser is forgiving; failure
        # here means the user typed something nonsensical at upload time —
        # surface that as a failed start instead of guessing.
        from app.utils.page_ranges import parse_page_ranges

        try:
            skip_pages = parse_page_ranges(job.skip_pages)
        except ValueError as e:
            msg = f"invalid skip_pages: {e}"
            set_job_status(job_id, JobStatus.FAILED, error_message=msg)
            record_event(job_id, "translate_failed", {"error": msg})
            return job_id

        try:
            specs = chunking.build_chunks(
                structure_path,
                category=category,
                target_tokens_override=job.chunk_target_tokens_override,
                hard_cap_tokens_override=job.chunk_hard_cap_tokens_override,
                skip_pages=skip_pages,
            )
        except Exception as e:
            log.exception("chunking failed job_id=%s", job_id)
            set_job_status(job_id, JobStatus.FAILED, error_message=f"chunking failed: {e}")
            record_event(job_id, "translate_failed", {"error": str(e)})
            return job_id

        reset_chunks(db, job_id)
        insert_chunks(db, job_id, specs)
        record_event(job_id, "chunks_built", {"count": len(specs)})

    # Translate: bounded ThreadPoolExecutor over PENDING chunks. We sweep until
    # all chunks reach a terminal state or retries are exhausted.
    max_retries = settings.chunk_max_retries
    concurrency = max(1, settings.translation_concurrency)

    abort_msg: str | None = None
    abort_reason: str = ""
    while True:
        if is_cancelled(job_id):
            record_event(job_id, "translate_cancelled", {})
            return job_id

        with SessionLocal() as db:
            chunks = (
                db.query(TranslationChunk)
                .filter(TranslationChunk.job_id == job_id)
                .order_by(TranslationChunk.chunk_index)
                .all()
            )
            pending = [
                c for c in chunks
                if c.status in {ChunkStatus.PENDING, ChunkStatus.RETRY}
                and c.attempt_count <= max_retries
            ]
            chunk_ids = [c.id for c in pending]

        if not chunk_ids:
            break

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_translate_one_chunk, job_id, cid): cid for cid in chunk_ids}
            for fut in as_completed(futures):
                try:
                    fut.result()
                except translation.TranslationModelMissing as e:
                    abort_msg = str(e)
                    abort_reason = "model_missing"
                except translation.TranslationAccessDenied as e:
                    abort_msg = str(e)
                    abort_reason = "access_denied"
                except Exception:
                    log.exception("chunk translation worker raised")

        if abort_msg:
            # Don't retry; same error will hit every chunk.
            log.error("translate_task aborting (%s): %s", abort_reason, abort_msg)
            set_job_status(job_id, JobStatus.FAILED, error_message=abort_msg)
            record_event(
                job_id, "translate_failed", {"reason": abort_reason, "error": abort_msg}
            )
            return job_id

    # Post-processing: glossary validation across the translated text.
    with SessionLocal() as db:
        chunks = (
            db.query(TranslationChunk)
            .filter(TranslationChunk.job_id == job_id)
            .order_by(TranslationChunk.chunk_index)
            .all()
        )
        glossary = (
            db.query(GlossaryEntry).filter(GlossaryEntry.job_id == job_id).all()
        )
        any_failed = any(c.status == ChunkStatus.FAILED for c in chunks)
        completed = sum(1 for c in chunks if c.status == ChunkStatus.COMPLETED)

    set_job_status(job_id, JobStatus.POSTPROCESSING, progress_percent=92.0)
    record_event(
        job_id,
        "translate_chunks_done",
        {"completed": completed, "total": len(chunks), "failed_any": any_failed},
    )
    validations = postprocess.validate_chunks(chunks, glossary)
    if validations:
        record_event(
            job_id,
            "glossary_violations",
            {
                "count": sum(len(v.violations) for v in validations),
                "chunks": [v.chunk_index for v in validations],
            },
        )

    if any_failed:
        set_job_status(
            job_id,
            JobStatus.FAILED,
            error_message=f"{sum(1 for c in chunks if c.status == ChunkStatus.FAILED)} chunk(s) failed",
        )
        return job_id

    record_event(job_id, "translate_completed", {"chunks": len(chunks)})
    rebuild_task.delay(job_id)
    return job_id


def _translate_one_chunk(job_id: str, chunk_id: str) -> None:
    """Translate a single chunk + persist results. Runs inside the thread pool.

    Multi-pass flow:
      1. Resolve the active prompt body (base + style profile composition) and
         cache it as a ``prompt_versions`` row.
      2. Pass 1 (TRANSLATE) → persist as a ``chunk_translation_passes`` row.
      3. (M7+) Pass 2 (CRITIQUE) and pass 3 (REVISE) hook in here, gated by
         ``critique_enabled`` and ``quality_score_threshold``.
      4. Set ``translation_chunks.final_pass_id`` to the chosen pass and copy
         that pass's translated text/blocks onto the chunk for fast read paths.
    """
    if is_cancelled(job_id):
        return
    settings = get_settings()
    pass_started_at = datetime.now(timezone.utc)
    with SessionLocal() as db:
        chunk = db.get(TranslationChunk, chunk_id)
        if chunk is None or chunk.status == ChunkStatus.COMPLETED:
            return
        chunk.status = ChunkStatus.IN_PROGRESS
        chunk.started_at = pass_started_at
        chunk.attempt_count = (chunk.attempt_count or 0) + 1
        attempt = chunk.attempt_count
        chunk_index = chunk.chunk_index
        blocks = list(chunk.blocks_json or [])
        job = db.get(Job, job_id)
        category = job.category if job else Category.UNKNOWN
        # Resolve the style profile inside this session so attribute access is safe;
        # then detach the data we need to use across sessions.
        profile = (
            db.get(StyleProfile, job.style_profile_id)
            if job is not None and job.style_profile_id
            else None
        )
        voice_card_text, voice_card_translator = (
            voice_card.resolve_voice_card_text_for_job(db, job_id)
        )
        system_prompt = translation.compose_translate_system_prompt(
            category, profile, voice_card_text=voice_card_text or None
        )
        prompt_key = translation.prompt_key_for_translate(category, profile)
        prompt_version = get_or_create_prompt_version(
            db,
            purpose=PromptPurpose.TRANSLATE,
            key=prompt_key,
            body=system_prompt,
            metadata={
                "category": category.value,
                "profile_key": profile.key if profile is not None else None,
                "profile_version": profile.version if profile is not None else None,
                # Record the voice card on the prompt-version row so
                # different voices produce distinct prompt-version entries
                # — that's how we track "translated under voice X" later.
                "voice_card_translator": voice_card_translator,
            },
        )
        prompt_version_id = prompt_version.id
        glossary = list(
            db.query(GlossaryEntry).filter(GlossaryEntry.job_id == job_id).all()
        )
        # Pull the previous chunk's summary + structured running context.
        prev_summary = None
        prev_running_context: dict | None = None
        if chunk_index > 0:
            prev = (
                db.query(TranslationChunk)
                .filter(
                    TranslationChunk.job_id == job_id,
                    TranslationChunk.chunk_index == chunk_index - 1,
                    TranslationChunk.status == ChunkStatus.COMPLETED,
                )
                .one_or_none()
            )
            if prev is not None:
                if prev.context_summary:
                    prev_summary = prev.context_summary
                if prev.running_context_json:
                    prev_running_context = dict(prev.running_context_json)
        db.commit()

    name_registry: list[NameRegistry] = list_names(job_id)

    translate_model = settings.gemini_translation_model

    try:
        result = translation.translate_chunk(
            blocks=blocks,
            glossary=glossary,
            name_registry=name_registry,
            previous_summary=prev_summary,
            previous_running_context=prev_running_context,
            system_prompt=system_prompt,
            model=translate_model,
        )
    except (
        translation.TranslationModelMissing,
        translation.TranslationAccessDenied,
    ) as e:
        # Mark this chunk failed and re-raise so the outer loop aborts the job.
        # Both classes are job-fatal — same error will hit every other chunk.
        kind = (
            "model_missing"
            if isinstance(e, translation.TranslationModelMissing)
            else "access_denied"
        )
        with SessionLocal() as db:
            chunk = db.get(TranslationChunk, chunk_id)
            if chunk is not None:
                chunk.status = ChunkStatus.FAILED
                db.commit()
            _record_failed_pass(
                db,
                chunk_id=chunk_id,
                pass_kind=PassKind.TRANSLATE,
                pass_number=1,
                model=translate_model,
                prompt_version_id=prompt_version_id,
                started_at=pass_started_at,
                error_message=f"{kind}: {e}",
            )
        raise
    except translation.TranslationError as e:
        log.warning("chunk %s translation error attempt=%d: %s", chunk_id, attempt, e)
        with SessionLocal() as db:
            chunk = db.get(TranslationChunk, chunk_id)
            if chunk is None:
                return
            if chunk.attempt_count > settings.chunk_max_retries:
                chunk.status = ChunkStatus.FAILED
            else:
                chunk.status = ChunkStatus.RETRY
            db.commit()
            _record_failed_pass(
                db,
                chunk_id=chunk_id,
                pass_kind=PassKind.TRANSLATE,
                pass_number=1,
                model=translate_model,
                prompt_version_id=prompt_version_id,
                started_at=pass_started_at,
                error_message=str(e),
            )
        return

    # Persist translated blocks + name registry updates.
    for n in result.new_names:
        upsert_name(
            job_id,
            source_name=n.source,
            burmese_phonetic=n.burmese,
            chunk_index=chunk_index,
        )

    translated_blocks = [
        {"block_id": b.block_id, "translated": b.translated} for b in result.blocks
    ]
    plain = "\n\n".join(b.translated for b in result.blocks)
    pass_completed_at = datetime.now(timezone.utc)

    # Merge the model's structured-context update with the prior snapshot,
    # under fixed caps. Falls back to the prior snapshot unchanged when the
    # model didn't emit a running_context (older prompts, transient skip).
    merged_context = merge_running_context(
        prev_running_context, result.running_context_update
    )

    pass_one_id: str
    with SessionLocal() as db:
        # Record the TRANSLATE pass row first so we can point final_pass_id at it.
        pass_row = ChunkTranslationPass(
            chunk_id=chunk_id,
            pass_number=1,
            pass_kind=PassKind.TRANSLATE,
            model=translate_model,
            prompt_version_id=prompt_version_id,
            input_tokens=result.prompt_tokens or 0,
            output_tokens=result.output_tokens or 0,
            cached_input_tokens=result.cached_input_tokens or 0,
            translated_text=plain,
            translated_blocks_json=translated_blocks,
            started_at=pass_started_at,
            completed_at=pass_completed_at,
            status=PassStatus.OK,
        )
        db.add(pass_row)
        db.flush()
        pass_one_id = pass_row.id

        chunk = db.get(TranslationChunk, chunk_id)
        if chunk is None:
            return
        chunk.translated_text = plain
        chunk.translated_blocks_json = translated_blocks
        chunk.context_summary = result.summary_for_next_chunk
        chunk.running_context_json = merged_context
        chunk.status = ChunkStatus.COMPLETED
        chunk.completed_at = pass_completed_at
        chunk.final_pass_id = pass_row.id
        # Accumulate so retries add to prior attempts' cost rather than overwriting.
        chunk.cost_tokens = (chunk.cost_tokens or 0) + (result.total_tokens or 0)
        db.commit()

    # Bump the job progress bar so the UI reflects this chunk landing.
    _update_translate_progress(job_id)

    # M7 quality loop. Errors here must not fail the chunk — pass 1 is already
    # the chosen final translation and chunk.status == COMPLETED.
    revise_happened = False
    if _critique_enabled_for_job(job_id):
        try:
            revise_happened = _run_critique_and_revise(
                job_id=job_id,
                chunk_id=chunk_id,
                source_blocks=blocks,
                pass_one_blocks=result.blocks,
                pass_one_id=pass_one_id,
                pass_one_total_tokens=result.total_tokens or 0,
                glossary=glossary,
                name_registry=name_registry,
                previous_summary=prev_summary,
                previous_running_context=prev_running_context,
                category=category,
                style_profile_id=(
                    profile.id if profile is not None else None
                ),
            )
        except (translation.TranslationModelMissing, translation.TranslationAccessDenied):
            # Critique/revise model missing or denied should not abort the job
            # — pass-1 already succeeded; we just skip the quality loop here.
            log.exception(
                "critique/revise model missing on chunk_id=%s; keeping pass 1", chunk_id
            )
        except Exception:
            log.exception(
                "critique/revise pipeline raised on chunk_id=%s; keeping pass 1", chunk_id
            )

    # M9 sampled judge. Run on:
    #   - chunk_index 0 (always; calibrates the rest of the document),
    #   - any chunk whose final pass is the revise (uplift measurement),
    #   - every Nth chunk by sample rate.
    # Errors here also must not fail the chunk.
    if _should_judge_chunk(chunk_index, revise_happened):
        try:
            _run_judge(
                job_id=job_id,
                chunk_id=chunk_id,
                source_blocks=blocks,
                glossary=glossary,
                name_registry=name_registry,
                previous_summary=prev_summary,
                category=category,
                style_profile_id=(profile.id if profile is not None else None),
            )
        except Exception:
            log.exception("judge raised on chunk_id=%s; ignoring", chunk_id)


def _update_translate_progress(job_id: str) -> None:
    """Recompute progress_percent during the TRANSLATING stage.

    Maps completed_chunks/total_chunks to the 25–88% range — the rest of the
    bar is reserved for postprocess (92), rebuild (95), and complete (100).
    Called after each chunk transitions to COMPLETED so the UI bar moves in
    real time instead of jumping from 25 → 92 at the end."""
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None or job.status != JobStatus.TRANSLATING:
            return
        total = (
            db.query(TranslationChunk)
            .filter(TranslationChunk.job_id == job_id)
            .count()
        )
        if total == 0:
            return
        completed = (
            db.query(TranslationChunk)
            .filter(
                TranslationChunk.job_id == job_id,
                TranslationChunk.status == ChunkStatus.COMPLETED,
            )
            .count()
        )
        # 25% at translate-start → 88% just before postprocess.
        job.progress_percent = 25.0 + (completed / total) * 63.0
        db.commit()


def _critique_enabled_for_job(job_id: str) -> bool:
    """Per-job override beats the global default; default is True."""
    settings = get_settings()
    if not settings.critique_enabled:
        return False
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None:
            return False
        # ``critique_enabled`` is non-null with default True, but an old row
        # backfilled before the column existed could conceivably be null.
        return bool(job.critique_enabled if job.critique_enabled is not None else True)


def _run_critique_and_revise(
    *,
    job_id: str,
    chunk_id: str,
    source_blocks: list[dict],
    pass_one_blocks: list,  # list[translation.TranslatedBlock]
    pass_one_id: str,
    pass_one_total_tokens: int,
    glossary: list,
    name_registry: list,
    previous_summary: str | None,
    previous_running_context: dict | None,
    category: Category,
    style_profile_id: str | None,
) -> bool:
    """Returns True iff a revise pass was promoted to ``final_pass_id``.

    The caller uses the return value to bias judge sampling — every revised
    chunk gets judged so we can measure uplift."""
    """Run pass 2 (critique) and conditionally pass 3 (revise).

    Gating rules (from the M7 plan):
    - Skip revise if critique.overall >= ``quality_score_threshold`` AND no
      severity=high issues.
    - Skip revise if running it would push the chunk's combined token spend
      past ``max_passes_total_tokens`` (cost guard against adversarial chunks).
    - If revise breaks block fidelity or otherwise raises, keep pass 1.
    """
    settings = get_settings()
    threshold = (
        _job_quality_score_threshold(job_id) or settings.quality_score_threshold
    )
    critique_model = settings.gemini_critique_model
    revise_model = settings.gemini_translation_model

    # Resolve profile + critique prompt version inside a fresh session.
    critique_started_at = datetime.now(timezone.utc)
    with SessionLocal() as db:
        profile = (
            db.get(StyleProfile, style_profile_id) if style_profile_id else None
        )
        voice_card_text, voice_card_translator = (
            voice_card.resolve_voice_card_text_for_job(db, job_id)
        )
        critique_system = translation.compose_critique_system_prompt(
            category, profile, voice_card_text=voice_card_text or None
        )
        critique_key = translation.prompt_key_for_critique(category, profile)
        critique_prompt_version = get_or_create_prompt_version(
            db,
            purpose=PromptPurpose.CRITIQUE,
            key=critique_key,
            body=critique_system,
            metadata={
                "category": category.value,
                "profile_key": profile.key if profile is not None else None,
                "voice_card_translator": voice_card_translator,
            },
        )
        critique_prompt_version_id = critique_prompt_version.id

    try:
        critique = translation.critique_chunk(
            blocks=source_blocks,
            pass_one_blocks=pass_one_blocks,
            glossary=glossary,
            name_registry=name_registry,
            previous_summary=previous_summary,
            category=category,
            style_profile=profile,
            system_prompt=critique_system,
            model=critique_model,
        )
    except translation.TranslationError as e:
        log.warning("critique failed on chunk_id=%s: %s — keeping pass 1", chunk_id, e)
        with SessionLocal() as db:
            _record_failed_pass(
                db,
                chunk_id=chunk_id,
                pass_kind=PassKind.CRITIQUE,
                pass_number=2,
                model=critique_model,
                prompt_version_id=critique_prompt_version_id,
                started_at=critique_started_at,
                error_message=str(e),
            )
        record_event(
            job_id, "critique_failed", {"chunk_id": chunk_id, "error": str(e)[:300]}
        )
        return False

    critique_completed_at = datetime.now(timezone.utc)
    with SessionLocal() as db:
        db.add(
            ChunkTranslationPass(
                chunk_id=chunk_id,
                pass_number=2,
                pass_kind=PassKind.CRITIQUE,
                model=critique_model,
                prompt_version_id=critique_prompt_version_id,
                input_tokens=critique.prompt_tokens or 0,
                output_tokens=critique.output_tokens or 0,
                cached_input_tokens=critique.cached_input_tokens or 0,
                critique_text=None,
                critique_json=critique.raw,
                started_at=critique_started_at,
                completed_at=critique_completed_at,
                status=PassStatus.OK,
            )
        )
        # Critique tokens roll into cost_tokens too.
        chunk = db.get(TranslationChunk, chunk_id)
        if chunk is not None:
            chunk.cost_tokens = (chunk.cost_tokens or 0) + (critique.total_tokens or 0)
        db.commit()

    record_event(
        job_id,
        "chunk_critiqued",
        {
            "chunk_id": chunk_id,
            "naturalness": critique.naturalness,
            "fidelity": critique.fidelity,
            "register_match": critique.register_match,
            "consistency": critique.consistency,
            "overall": critique.overall,
            "issues_count": len(critique.issues),
            "high_severity_count": sum(1 for i in critique.issues if i.severity == "high"),
        },
    )

    # Gate: revise only when needed.
    has_high = critique.has_high_severity()
    if critique.overall >= threshold and not has_high:
        record_event(
            job_id,
            "revise_skipped",
            {"chunk_id": chunk_id, "reason": "above_threshold", "overall": critique.overall},
        )
        return False

    # Token-budget guard.
    projected = (
        pass_one_total_tokens
        + (critique.total_tokens or 0)
        # Revise typically lands within ~1.3x the original translate spend; pad slightly.
        + int(pass_one_total_tokens * 1.3)
    )
    if projected > settings.max_passes_total_tokens:
        record_event(
            job_id,
            "revise_skipped",
            {
                "chunk_id": chunk_id,
                "reason": "token_budget",
                "projected": projected,
                "limit": settings.max_passes_total_tokens,
            },
        )
        return False

    # Pass 3 — revise.
    revise_started_at = datetime.now(timezone.utc)
    with SessionLocal() as db:
        # Re-resolve profile inside the session for safe attribute access.
        profile_for_revise = (
            db.get(StyleProfile, style_profile_id) if style_profile_id else None
        )
        voice_card_text, voice_card_translator = (
            voice_card.resolve_voice_card_text_for_job(db, job_id)
        )
        revise_system = translation.compose_revise_system_prompt(
            category, profile_for_revise, voice_card_text=voice_card_text or None
        )
        revise_key = translation.prompt_key_for_revise(category, profile_for_revise)
        revise_prompt_version = get_or_create_prompt_version(
            db,
            purpose=PromptPurpose.REVISE,
            key=revise_key,
            body=revise_system,
            metadata={
                "category": category.value,
                "profile_key": profile_for_revise.key if profile_for_revise is not None else None,
                "voice_card_translator": voice_card_translator,
            },
        )
        revise_prompt_version_id = revise_prompt_version.id

    try:
        revised = translation.revise_chunk(
            blocks=source_blocks,
            pass_one_blocks=pass_one_blocks,
            critique=critique,
            glossary=glossary,
            name_registry=name_registry,
            previous_summary=previous_summary,
            previous_running_context=previous_running_context,
            category=category,
            style_profile=profile_for_revise,
            system_prompt=revise_system,
            model=revise_model,
        )
    except translation.TranslationError as e:
        log.warning("revise failed on chunk_id=%s: %s — keeping pass 1", chunk_id, e)
        with SessionLocal() as db:
            _record_failed_pass(
                db,
                chunk_id=chunk_id,
                pass_kind=PassKind.REVISE,
                pass_number=3,
                model=revise_model,
                prompt_version_id=revise_prompt_version_id,
                started_at=revise_started_at,
                error_message=str(e),
            )
        record_event(
            job_id,
            "revise_failed",
            {"chunk_id": chunk_id, "error": str(e)[:300]},
        )
        return False

    revise_completed_at = datetime.now(timezone.utc)
    revised_blocks_serialized = [
        {"block_id": b.block_id, "translated": b.translated} for b in revised.blocks
    ]
    revised_plain = "\n\n".join(b.translated for b in revised.blocks)

    # If revise emitted an updated structured context, re-merge against prior
    # snapshot. If it didn't, the pass-1 context already on the chunk wins.
    revised_running_context = (
        merge_running_context(previous_running_context, revised.running_context_update)
        if revised.running_context_update
        else None
    )

    # Promote the revise pass: set final_pass_id, update chunk text + summary.
    with SessionLocal() as db:
        revise_row = ChunkTranslationPass(
            chunk_id=chunk_id,
            pass_number=3,
            pass_kind=PassKind.REVISE,
            model=revise_model,
            prompt_version_id=revise_prompt_version_id,
            input_tokens=revised.prompt_tokens or 0,
            output_tokens=revised.output_tokens or 0,
            cached_input_tokens=revised.cached_input_tokens or 0,
            translated_text=revised_plain,
            translated_blocks_json=revised_blocks_serialized,
            started_at=revise_started_at,
            completed_at=revise_completed_at,
            status=PassStatus.OK,
        )
        db.add(revise_row)
        db.flush()

        chunk = db.get(TranslationChunk, chunk_id)
        if chunk is not None:
            chunk.translated_text = revised_plain
            chunk.translated_blocks_json = revised_blocks_serialized
            chunk.context_summary = revised.summary_for_next_chunk or chunk.context_summary
            if revised_running_context is not None:
                chunk.running_context_json = revised_running_context
            chunk.final_pass_id = revise_row.id
            chunk.cost_tokens = (chunk.cost_tokens or 0) + (revised.total_tokens or 0)
        db.commit()

    # Persist any new names the revise pass surfaced.
    for n in revised.new_names:
        upsert_name(
            job_id,
            source_name=n.source,
            burmese_phonetic=n.burmese,
            chunk_index=_chunk_index_for(chunk_id) or 0,
        )

    record_event(
        job_id,
        "chunk_revised",
        {
            "chunk_id": chunk_id,
            "trigger": "high_severity" if has_high else "below_threshold",
            "critique_overall": critique.overall,
        },
    )
    return True


def _should_judge_chunk(chunk_index: int, revise_happened: bool) -> bool:
    """Decide whether the M9 judge should run on this chunk.

    Always judge:
      - chunk_index 0 (anchors the score for the rest of the document)
      - any chunk that went through revise (so we measure uplift)
    Otherwise sample at ``judge_sample_rate``. Sample-rate of 0.0 disables
    everything except the always-judge cases."""
    if chunk_index == 0:
        return True
    if revise_happened:
        return True
    rate = max(0.0, min(1.0, get_settings().judge_sample_rate))
    if rate <= 0:
        return False
    if rate >= 1.0:
        return True
    # Deterministic stride: every Nth chunk where N = round(1/rate). Stable
    # across worker restarts and easy to reason about during incident review.
    stride = max(1, round(1.0 / rate))
    return (chunk_index % stride) == 0


def _run_judge(
    *,
    job_id: str,
    chunk_id: str,
    source_blocks: list[dict],
    glossary: list,
    name_registry: list,
    previous_summary: str | None,
    category: Category,
    style_profile_id: str | None,
) -> None:
    """Run the M9 judge against a chunk's *final* translation. Persists the
    result as a ``quality_scores`` row + emits ``chunk_scored``. Low-overall
    chunks emit a separate ``low_score_chunk`` event for the review queue."""
    settings = get_settings()
    judge_model = settings.gemini_judge_model

    # Resolve profile + read the final translation in a single session.
    with SessionLocal() as db:
        chunk = db.get(TranslationChunk, chunk_id)
        if chunk is None:
            return
        final_pass_id = chunk.final_pass_id
        final_blocks_data = chunk.translated_blocks_json or []
        if not final_blocks_data:
            return
        final_blocks = [
            translation.TranslatedBlock(
                block_id=b.get("block_id") or "", translated=b.get("translated") or ""
            )
            for b in final_blocks_data
            if b.get("block_id")
        ]
        if not final_blocks:
            return
        profile = (
            db.get(StyleProfile, style_profile_id) if style_profile_id else None
        )
        voice_card_text, voice_card_translator = (
            voice_card.resolve_voice_card_text_for_job(db, job_id)
        )
        judge_system = translation.compose_judge_system_prompt(
            category, profile, voice_card_text=voice_card_text or None
        )
        judge_key = translation.prompt_key_for_judge(category, profile)
        judge_prompt_version = get_or_create_prompt_version(
            db,
            purpose=PromptPurpose.JUDGE,
            key=judge_key,
            body=judge_system,
            metadata={
                "category": category.value,
                "profile_key": profile.key if profile is not None else None,
                "voice_card_translator": voice_card_translator,
            },
        )
        judge_prompt_version_id = judge_prompt_version.id

    try:
        score = translation.judge_chunk(
            blocks=source_blocks,
            final_blocks=final_blocks,
            glossary=glossary,
            name_registry=name_registry,
            previous_summary=previous_summary,
            category=category,
            style_profile=profile,
            system_prompt=judge_system,
            model=judge_model,
        )
    except translation.TranslationError as e:
        log.warning("judge failed on chunk_id=%s: %s", chunk_id, e)
        return

    with SessionLocal() as db:
        db.add(
            QualityScore(
                chunk_id=chunk_id,
                pass_id=final_pass_id,
                naturalness=score.naturalness,
                fidelity=score.fidelity,
                register_match=score.register_match,
                consistency=score.consistency,
                overall=score.overall,
                reasons_json={
                    "issues": [
                        {
                            "block_id": i.block_id,
                            "severity": i.severity,
                            "kind": i.kind,
                            "snippet": i.snippet,
                            "suggestion": i.suggestion,
                        }
                        for i in score.issues
                    ],
                },
                judge_model=judge_model,
                judge_prompt_version_id=judge_prompt_version_id,
            )
        )
        chunk = db.get(TranslationChunk, chunk_id)
        if chunk is not None:
            # Roll judge tokens into chunk cost for accurate /admin/metrics.
            chunk.cost_tokens = (chunk.cost_tokens or 0) + (score.total_tokens or 0)
        db.commit()

    record_event(
        job_id,
        "chunk_scored",
        {
            "chunk_id": chunk_id,
            "naturalness": score.naturalness,
            "fidelity": score.fidelity,
            "register_match": score.register_match,
            "consistency": score.consistency,
            "overall": score.overall,
            "issues_count": len(score.issues),
        },
    )

    if score.overall <= settings.low_score_threshold:
        record_event(
            job_id,
            "low_score_chunk",
            {"chunk_id": chunk_id, "overall": score.overall},
        )


def _job_quality_score_threshold(job_id: str) -> float | None:
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None:
            return None
        return job.quality_score_threshold


def _chunk_index_for(chunk_id: str) -> int | None:
    with SessionLocal() as db:
        chunk = db.get(TranslationChunk, chunk_id)
        return chunk.chunk_index if chunk is not None else None


def _record_failed_pass(
    db,
    *,
    chunk_id: str,
    pass_kind: PassKind,
    pass_number: int,
    model: str,
    prompt_version_id: str | None,
    started_at,
    error_message: str,
) -> None:
    """Best-effort: persist a FAILED pass row for forensics. Swallows any
    secondary error — the outer chunk-status update is the source of truth."""
    try:
        db.add(
            ChunkTranslationPass(
                chunk_id=chunk_id,
                pass_number=pass_number,
                pass_kind=pass_kind,
                model=model,
                prompt_version_id=prompt_version_id,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                status=PassStatus.FAILED,
                error_message=error_message[:1000],
            )
        )
        db.commit()
    except Exception:
        log.exception("failed to persist FAILED pass row for chunk_id=%s", chunk_id)


@celery_app.task(name="jobs.rebuild", bind=True)
def rebuild_task(self, job_id: str) -> str:
    """Stage 9: rebuild layout-preserved PDF + EPUB outputs."""
    log.info("rebuild_task start job_id=%s", job_id)
    if is_cancelled(job_id):
        record_event(job_id, "rebuild_cancelled", {})
        return job_id
    set_job_status(job_id, JobStatus.REBUILDING, progress_percent=95.0)
    record_event(job_id, "rebuild_started", {"task_id": self.request.id})

    settings = get_settings()
    structure_path = settings.intermediate_dir / job_id / "structure.json"
    if not structure_path.exists():
        msg = f"structure.json missing at {structure_path}"
        log.error(msg)
        set_job_status(job_id, JobStatus.FAILED, error_message=msg)
        record_event(job_id, "rebuild_failed", {"error": msg})
        return job_id

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None:
            return job_id
        source_pdf = Path(job.file_path)
        original_filename = job.original_filename
        export_mode = job.export_layout_mode
        export_overrides = job.export_template_overrides_json
        title_source = job.title_source
        title_target = job.title_target
        author_source = job.author_source
        author_target = job.author_target
        cover_image_path = job.cover_image_path
        # M15: parse skip-pages here too — the rebuild path drops pages even
        # when translation already excluded them, so re-renders on a job
        # whose skip_pages was edited post-translate work consistently.
        from app.utils.page_ranges import parse_page_ranges as _parse_pr

        try:
            rebuild_skip_pages = _parse_pr(job.skip_pages)
        except ValueError:
            rebuild_skip_pages = set()
        chunks = (
            db.query(TranslationChunk)
            .filter(TranslationChunk.job_id == job_id)
            .order_by(TranslationChunk.chunk_index)
            .all()
        )

    # Build a flat block_id → translated_text map across all chunks.
    translations_by_block: dict[str, str] = {}
    for c in chunks:
        for entry in c.translated_blocks_json or []:
            bid = entry.get("block_id")
            translated = entry.get("translated") or ""
            if bid and translated:
                translations_by_block[bid] = translated

    if not translations_by_block:
        msg = "no translated blocks available; cannot rebuild"
        set_job_status(job_id, JobStatus.FAILED, error_message=msg)
        record_event(job_id, "rebuild_failed", {"error": msg})
        return job_id

    output_dir = settings.outputs_dir / job_id
    base = Path(original_filename).stem or "translated"

    # M14: choose export path. AUTO defers to a layout-detection pass over
    # the source PDF; TEMPLATE/PRESERVE are explicit. Detection failure is
    # non-fatal — we fall back to PRESERVE because the existing pipeline is
    # known-good and a wrong-path TEMPLATE on a colored book would erase
    # important visual content.
    chosen_mode = export_mode or ExportLayoutMode.AUTO
    detection_reason: str | None = None
    if chosen_mode == ExportLayoutMode.AUTO:
        try:
            detection = layout_detection.detect_layout_mode(source_pdf)
            chosen_mode = detection.recommended_mode
            detection_reason = detection.reason
        except Exception as e:
            log.exception("layout detection failed; defaulting to PRESERVE")
            chosen_mode = ExportLayoutMode.PRESERVE
            detection_reason = f"detection failed: {e}"

    # 1. PDF
    try:
        if chosen_mode == ExportLayoutMode.TEMPLATE:
            template_config = pdf_template.TemplateConfig.from_settings(
                overrides=export_overrides,
                title_source=title_source,
                title_target=title_target,
                author_source=author_source,
                author_target=author_target,
            )
            template_result = pdf_template.render_template_pdf(
                structure_path=structure_path,
                translations_by_block=translations_by_block,
                output_dir=output_dir,
                output_name=f"{base}.my.pdf",
                config=template_config,
                skip_pages=rebuild_skip_pages,
                cover_image_path=cover_image_path,
            )
            pdf_output_path = template_result.output_path
            pdf_event_payload = {
                "path": str(pdf_output_path),
                "mode": "TEMPLATE",
                "pages": template_result.pages_written,
                "chapters": template_result.chapters_rendered,
                "blocks_placed": template_result.blocks_placed,
                "detection_reason": detection_reason,
                "skipped_pages_count": len(rebuild_skip_pages),
            }
        else:
            pdf_result = rebuild_pdf.rebuild_pdf(
                source_pdf=source_pdf,
                structure_path=structure_path,
                translations_by_block=translations_by_block,
                output_dir=output_dir,
                output_name=f"{base}.my.pdf",
                skip_pages=rebuild_skip_pages,
                cover_image_path=cover_image_path,
            )
            pdf_output_path = pdf_result.output_path
            pdf_event_payload = {
                "path": str(pdf_output_path),
                "mode": "PRESERVE",
                "blocks_placed": pdf_result.blocks_placed,
                "overflow_blocks": pdf_result.overflow_blocks,
                "detection_reason": detection_reason,
                "skipped_pages_count": len(rebuild_skip_pages),
            }
    except Exception as e:
        log.exception("PDF rebuild failed job_id=%s", job_id)
        set_job_status(job_id, JobStatus.FAILED, error_message=f"PDF rebuild failed: {e}")
        record_event(job_id, "rebuild_failed", {"stage": "pdf", "error": str(e)})
        return job_id
    upsert_output(
        job_id,
        fmt=OutputFormat.PDF,
        file_path=str(pdf_output_path),
        size_bytes=pdf_output_path.stat().st_size,
    )
    record_event(job_id, "rebuild_pdf_done", pdf_event_payload)

    if is_cancelled(job_id):
        record_event(job_id, "rebuild_cancelled", {})
        return job_id

    # 2. EPUB
    try:
        epub_result = rebuild_epub.rebuild_epub(
            structure_path=structure_path,
            translations_by_block=translations_by_block,
            output_dir=output_dir,
            output_name=f"{base}.my.epub",
            title=base,
            job_id=job_id,
        )
    except Exception as e:
        log.exception("EPUB rebuild failed job_id=%s", job_id)
        set_job_status(job_id, JobStatus.FAILED, error_message=f"EPUB rebuild failed: {e}")
        record_event(job_id, "rebuild_failed", {"stage": "epub", "error": str(e)})
        return job_id
    upsert_output(
        job_id,
        fmt=OutputFormat.EPUB,
        file_path=str(epub_result.output_path),
        size_bytes=epub_result.output_path.stat().st_size,
    )
    record_event(
        job_id,
        "rebuild_epub_done",
        {"path": str(epub_result.output_path), "chapters": epub_result.chapters_written},
    )

    set_job_status(job_id, JobStatus.COMPLETED, progress_percent=100.0)
    record_event(job_id, "rebuild_completed", {})
    return job_id


def _excerpt_from_structure(path: Path, limit: int) -> str:
    """Stream the structured JSON and concatenate block texts up to ``limit``."""
    import json

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    parts: list[str] = []
    total = 0
    for page in data.get("pages", []):
        for block in page.get("blocks", []):
            text = block.get("text", "")
            if not text:
                continue
            parts.append(text)
            total += len(text)
            if total >= limit:
                break
        if total >= limit:
            break
    return ("\n".join(parts))[:limit]

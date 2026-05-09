from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path

import pymupdf
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.api.auth import require_token
from app.config import Settings, get_settings
from app.db.models import Category, ExportLayoutMode, JobStatus, OutputFormat, SuggestionState
from app.db.session import get_db
from app.schemas.jobs import (
    CategoryUpdateRequest,
    ChunkSummary,
    CompareBlock,
    ComparePage,
    CompareResponse,
    CostEstimateResponse,
    GlossaryEntryItem,
    GlossarySuggestionItem,
    GlossaryUploadResponse,
    JobCreateResponse,
    JobMetadataUpdateRequest,
    JobMetricsSummary,
    JobStatusResponse,
    OutputItem,
    OutputListResponse,
    QualityScoreItem,
    ReExportResponse,
    StructureSummaryResponse,
    StyleProfileUpdateRequest,
    SuggestionPatchRequest,
    SuggestionsBulkRequest,
    SuggestionsRunResponse,
    TranslationOutputResponse,
)
from app.services import cost_estimator, glossary_suggester, metrics
from app.services.glossary import parse_glossary
from app.services.jobs import (
    create_job,
    get_job,
    list_glossary,
    list_outputs,
    replace_glossary,
    request_cancel,
    reset_for_resume,
    update_category,
    update_style_profile,
)
from app.services.style_profiles import get_profile_by_key

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"], dependencies=[Depends(require_token)])


def _validate_pdf(path: Path, max_pages: int) -> tuple[int, bool]:
    """Open the PDF to validate + return (page_count, is_encrypted_rejected)."""
    try:
        doc = pymupdf.open(path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"invalid PDF: {e}"
        ) from e

    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="password-protected PDFs are not supported",
            )
        page_count = doc.page_count
        if page_count > max_pages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"page count {page_count} exceeds max {max_pages}",
            )
        return page_count, False
    finally:
        doc.close()


@router.post("", response_model=JobCreateResponse, status_code=status.HTTP_201_CREATED)
async def upload_pdf(
    file: UploadFile = File(...),
    style_profile_key: str | None = Form(default=None),
    chunk_target_tokens: int | None = Form(default=None, ge=200, le=8000),
    chunk_hard_cap_tokens: int | None = Form(default=None, ge=400, le=12000),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> JobCreateResponse:
    if file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unsupported content_type: {file.content_type}",
        )

    # Resolve the optional profile up-front so the user gets immediate feedback
    # on a bad key rather than discovering it after the upload completes.
    chosen_profile = None
    if style_profile_key:
        chosen_profile = get_profile_by_key(db, style_profile_key)
        if chosen_profile is None or not chosen_profile.active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"unknown or inactive style_profile_key={style_profile_key!r}",
            )

    job_id = str(uuid.uuid4())
    dest = settings.uploads_dir / f"{job_id}.pdf"
    max_bytes = settings.max_upload_mb * 1024 * 1024

    size = 0
    with dest.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"file exceeds max_upload_mb={settings.max_upload_mb}",
                )
            f.write(chunk)

    try:
        page_count, _ = _validate_pdf(dest, settings.max_pages)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise

    job = create_job(
        db,
        original_filename=file.filename or f"{job_id}.pdf",
        file_path=str(dest),
        file_size_bytes=size,
    )
    job.page_count = page_count
    if chosen_profile is not None:
        job.style_profile_id = chosen_profile.id
    if chunk_target_tokens is not None:
        job.chunk_target_tokens_override = chunk_target_tokens
    if chunk_hard_cap_tokens is not None:
        job.chunk_hard_cap_tokens_override = chunk_hard_cap_tokens
    db.commit()
    db.refresh(job)

    # Kick off extraction. Import lazily to avoid forcing Celery at import time during tests.
    try:
        from app.workers.tasks import extract_task

        extract_task.delay(job.id)
    except Exception:
        log.exception("failed to enqueue extract_task for job_id=%s", job.id)

    return JobCreateResponse.model_validate(job)


@router.get("", response_model=list[JobStatusResponse])
def list_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[JobStatusResponse]:
    """Recent jobs, newest first. Used by the UI's history panel."""
    from app.db.models import Job

    rows = db.query(Job).order_by(Job.created_at.desc()).limit(limit).all()
    return [JobStatusResponse.model_validate(j) for j in rows]


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str, db: Session = Depends(get_db)) -> JobStatusResponse:
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    return JobStatusResponse.model_validate(job)


@router.patch("/{job_id}/category", response_model=JobStatusResponse)
def patch_category(
    job_id: str,
    body: CategoryUpdateRequest,
    db: Session = Depends(get_db),
) -> JobStatusResponse:
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    if body.category not in {
        Category.NOVEL,
        Category.EDUCATIONAL,
        Category.PERSONAL_DEVELOPMENT,
    }:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="category must be NOVEL, EDUCATIONAL, or PERSONAL_DEVELOPMENT",
        )
    if job.status not in {
        JobStatus.AWAITING_CONFIRMATION,
        JobStatus.CLASSIFYING,
        JobStatus.UPLOADED,
        JobStatus.EXTRACTING,
    }:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"cannot change category in status={job.status.value}",
        )
    update_category(db, job, body.category)
    return JobStatusResponse.model_validate(job)


@router.patch("/{job_id}/metadata", response_model=JobStatusResponse)
def patch_metadata(
    job_id: str,
    body: JobMetadataUpdateRequest,
    db: Session = Depends(get_db),
) -> JobStatusResponse:
    """Update title/author/layout-mode metadata for a job.

    Only the fields explicitly set on the request are applied. ``None``
    means "leave alone"; an empty string means "clear this field".
    Editable at any stage of the pipeline — including post-COMPLETED, since
    title/author edits drive the running header on re-exports."""
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")

    # Pydantic v2: ``model_fields_set`` tells us which fields were sent vs.
    # left at default. Only those fields update the row.
    sent = body.model_fields_set
    if "title_source" in sent:
        job.title_source = body.title_source or None
    if "title_target" in sent:
        job.title_target = body.title_target or None
    if "author_source" in sent:
        job.author_source = body.author_source or None
    if "author_target" in sent:
        job.author_target = body.author_target or None
    if "export_layout_mode" in sent and body.export_layout_mode is not None:
        job.export_layout_mode = body.export_layout_mode
    if "export_template_overrides_json" in sent:
        job.export_template_overrides_json = body.export_template_overrides_json
    if "skip_pages" in sent:
        # Validate at write time — an invalid format here would block the
        # job's translation start later, which is much harder to debug.
        from app.utils.page_ranges import parse_page_ranges as _parse_pr

        if body.skip_pages:
            try:
                _parse_pr(body.skip_pages)
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"invalid skip_pages: {e}",
                )
            job.skip_pages = body.skip_pages
        else:
            job.skip_pages = None

    db.commit()
    db.refresh(job)
    return JobStatusResponse.model_validate(job)


@router.post("/{job_id}/cover", response_model=JobStatusResponse)
async def upload_cover(
    job_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> JobStatusResponse:
    """Upload a cover image for the book.

    Accepts JPG/PNG/WEBP (and a few others). Image is normalized to JPEG,
    resized to a reasonable max dimension, and saved under
    ``storage/intermediate/<job_id>/cover.jpg``. The export pipeline picks
    it up automatically — no further action required."""
    from app.services.cover_image import CoverImageError, save_cover_image

    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    raw = await file.read()
    try:
        path = save_cover_image(job_id=job_id, raw_bytes=raw)
    except CoverImageError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"invalid cover image: {e}",
        )
    job.cover_image_path = str(path)
    db.commit()
    db.refresh(job)
    return JobStatusResponse.model_validate(job)


@router.delete("/{job_id}/cover", response_model=JobStatusResponse)
def delete_cover(
    job_id: str,
    db: Session = Depends(get_db),
) -> JobStatusResponse:
    """Remove the cover image. Output PDFs revert to no cover page on next
    re-export."""
    from app.services.cover_image import delete_cover_image

    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    delete_cover_image(cover_path=job.cover_image_path)
    job.cover_image_path = None
    db.commit()
    db.refresh(job)
    return JobStatusResponse.model_validate(job)


@router.post("/{job_id}/re-export", response_model=ReExportResponse)
def re_export_pdf(
    job_id: str,
    db: Session = Depends(get_db),
) -> ReExportResponse:
    """Re-render the PDF using the current Job state (title/author/layout-mode).

    Cheap — no Gemini, just template rendering or layout-preserving rebuild
    against existing translated chunks. Useful when the user edits the
    title/author/layout-mode on a COMPLETED job and wants the new values
    reflected in the output PDF.

    Only valid when the job is COMPLETED (has translated chunks); otherwise
    returns 409. The new PDF replaces the existing one and the corresponding
    ``outputs`` row is updated.
    """
    from pathlib import Path

    from app.db.models import TranslationChunk
    from app.services import layout_detection, pdf_template, rebuild_pdf
    from app.services.jobs import upsert_output

    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"re-export requires status=COMPLETED, got {job.status.value}",
        )

    settings = get_settings()
    structure_path = settings.intermediate_dir / job_id / "structure.json"
    if not structure_path.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"structure.json missing at {structure_path}",
        )

    chunks = (
        db.query(TranslationChunk)
        .filter(TranslationChunk.job_id == job_id)
        .order_by(TranslationChunk.chunk_index)
        .all()
    )
    translations_by_block: dict[str, str] = {}
    for c in chunks:
        for entry in c.translated_blocks_json or []:
            bid = entry.get("block_id")
            translated = entry.get("translated") or ""
            if bid and translated:
                translations_by_block[bid] = translated
    if not translations_by_block:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="no translated blocks found; cannot re-export",
        )

    # Resolve layout mode (AUTO defers to detection over the source PDF).
    chosen_mode = job.export_layout_mode or ExportLayoutMode.AUTO
    if chosen_mode == ExportLayoutMode.AUTO:
        try:
            chosen_mode = layout_detection.detect_layout_mode(
                Path(job.file_path)
            ).recommended_mode
        except Exception:
            log.exception("layout detection failed during re-export; defaulting to PRESERVE")
            chosen_mode = ExportLayoutMode.PRESERVE

    output_dir = settings.outputs_dir / job_id
    base = Path(job.original_filename).stem or "translated"

    # Resolve skip_pages — same parser as the worker. Bad input was already
    # rejected at /metadata PATCH time; here we just return empty on parse
    # failure so a stale value doesn't block re-export.
    from app.utils.page_ranges import parse_page_ranges as _parse_pr

    try:
        skip_set = _parse_pr(job.skip_pages)
    except ValueError:
        skip_set = set()

    if chosen_mode == ExportLayoutMode.TEMPLATE:
        cfg = pdf_template.TemplateConfig.from_settings(
            overrides=job.export_template_overrides_json,
            title_source=job.title_source,
            title_target=job.title_target,
            author_source=job.author_source,
            author_target=job.author_target,
        )
        result = pdf_template.render_template_pdf(
            structure_path=structure_path,
            translations_by_block=translations_by_block,
            output_dir=output_dir,
            output_name=f"{base}.my.pdf",
            config=cfg,
            skip_pages=skip_set,
            cover_image_path=job.cover_image_path,
        )
        out_path = result.output_path
        pages_written = result.pages_written
        chapters_rendered = result.chapters_rendered
        blocks_placed = result.blocks_placed
    else:
        rb = rebuild_pdf.rebuild_pdf(
            source_pdf=Path(job.file_path),
            structure_path=structure_path,
            translations_by_block=translations_by_block,
            output_dir=output_dir,
            output_name=f"{base}.my.pdf",
            skip_pages=skip_set,
            cover_image_path=job.cover_image_path,
        )
        out_path = rb.output_path
        pages_written = rb.pages_written
        chapters_rendered = 0
        blocks_placed = rb.blocks_placed

    upsert_output(
        job_id,
        fmt=OutputFormat.PDF,
        file_path=str(out_path),
        size_bytes=out_path.stat().st_size,
    )

    return ReExportResponse(
        job_id=job_id,
        output_path=str(out_path),
        pages_written=pages_written,
        chapters_rendered=chapters_rendered,
        blocks_placed=blocks_placed,
        size_bytes=out_path.stat().st_size,
        mode=chosen_mode.value,
    )


@router.patch("/{job_id}/style-profile", response_model=JobStatusResponse)
def patch_style_profile(
    job_id: str,
    body: StyleProfileUpdateRequest,
    db: Session = Depends(get_db),
) -> JobStatusResponse:
    """Change the style profile while the job is still pre-translation.

    Locked once translation starts so chunks within a job are translated under
    a single, consistent profile (and a single ``prompt_version``)."""
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    if job.status not in {
        JobStatus.UPLOADED,
        JobStatus.EXTRACTING,
        JobStatus.CLASSIFYING,
        JobStatus.AWAITING_CONFIRMATION,
    }:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"cannot change style profile in status={job.status.value}",
        )
    # Empty string = clear the profile. ``style_profile_id`` becomes NULL;
    # the translation pipeline then uses the category base prompt + voice
    # card directly, which is the recommended path for PERSONAL_DEVELOPMENT.
    if not body.style_profile_key:
        update_style_profile(db, job, None)
        return JobStatusResponse.model_validate(job)
    profile = get_profile_by_key(db, body.style_profile_key)
    if profile is None or not profile.active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unknown or inactive style_profile_key={body.style_profile_key!r}",
        )
    update_style_profile(db, job, profile)
    return JobStatusResponse.model_validate(job)


@router.get("/{job_id}/structure", response_model=StructureSummaryResponse)
def get_structure(
    job_id: str,
    db: Session = Depends(get_db),
) -> StructureSummaryResponse:
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    if job.structure is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="extraction has not produced a structure yet",
        )
    path = Path(job.structure.structured_json_path)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"structure file missing at {path}"
        )
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    summary = data.get("summary", {})
    pages = data.get("pages", [])
    preview: list[dict] = []
    for page in pages[:3]:
        truncated_blocks = []
        for block in page.get("blocks", [])[:5]:
            truncated_blocks.append(
                {
                    "block_id": block.get("block_id"),
                    "type": block.get("type"),
                    "bbox": block.get("bbox"),
                    "text": (block.get("text") or "")[:300],
                    "dominant_font": block.get("dominant_font"),
                }
            )
        preview.append(
            {
                "page_number": page.get("page_number"),
                "is_scanned": page.get("is_scanned"),
                "blocks": truncated_blocks,
                "block_count": len(page.get("blocks", [])),
            }
        )
    return StructureSummaryResponse(
        job_id=job_id,
        structured_json_path=str(path),
        total_blocks=summary.get("total_blocks", job.structure.total_blocks),
        total_pages=summary.get("total_pages", job.page_count or 0),
        is_scanned=job.is_scanned,
        scanned_pages=summary.get("scanned_pages", []),
        pages_preview=preview,
    )


@router.post("/{job_id}/glossary", response_model=GlossaryUploadResponse)
async def upload_glossary(
    job_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> GlossaryUploadResponse:
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    if job.status not in {
        JobStatus.UPLOADED,
        JobStatus.EXTRACTING,
        JobStatus.CLASSIFYING,
        JobStatus.AWAITING_CONFIRMATION,
    }:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"cannot upload glossary in status={job.status.value}",
        )
    raw = await file.read()
    try:
        terms = parse_glossary(raw, file.content_type)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    saved = replace_glossary(db, job_id, terms)
    return GlossaryUploadResponse(job_id=job_id, saved=saved)


@router.get("/{job_id}/glossary", response_model=list[GlossaryEntryItem])
def get_glossary(job_id: str, db: Session = Depends(get_db)) -> list[GlossaryEntryItem]:
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    return [GlossaryEntryItem.model_validate(g) for g in list_glossary(db, job_id)]


@router.post(
    "/{job_id}/glossary/suggestions/run",
    response_model=SuggestionsRunResponse,
)
def run_glossary_suggestions(
    job_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> SuggestionsRunResponse:
    """Manually (re)trigger the suggester. Synchronous on the API thread —
    the call hits Gemini and may take 5–15s. The classify_task auto-runs the
    suggester once; this endpoint is for re-runs after edits or failures."""
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    structure_path = settings.intermediate_dir / job_id / "structure.json"
    if not structure_path.exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="extraction has not produced a structure yet",
        )
    counts = glossary_suggester.run_for_job(db, job_id, structure_path)
    return SuggestionsRunResponse(**counts)


@router.get(
    "/{job_id}/glossary/suggestions",
    response_model=list[GlossarySuggestionItem],
)
def list_glossary_suggestions(
    job_id: str,
    state: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[GlossarySuggestionItem]:
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    state_enum: SuggestionState | None = None
    if state is not None:
        try:
            state_enum = SuggestionState(state.upper())
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"unknown state {state!r}"
            ) from e
    rows = glossary_suggester.list_suggestions(db, job_id, state=state_enum)
    return [
        GlossarySuggestionItem(
            id=r.id,
            term_source=r.term_source,
            term_target=r.term_target,
            kind=r.kind.value,
            confidence=r.confidence,
            evidence_json=r.evidence_json or [],
            state=r.state.value,
        )
        for r in rows
    ]


@router.patch(
    "/{job_id}/glossary/suggestions/{suggestion_id}",
    response_model=GlossarySuggestionItem,
)
def patch_glossary_suggestion(
    job_id: str,
    suggestion_id: str,
    body: SuggestionPatchRequest,
    db: Session = Depends(get_db),
) -> GlossarySuggestionItem:
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    target_state = body.state.upper()
    try:
        if target_state == "APPROVED":
            row = glossary_suggester.approve_suggestion(
                db, job_id, suggestion_id, term_target_override=body.term_target
            )
        elif target_state == "REJECTED":
            row = glossary_suggester.reject_suggestion(db, job_id, suggestion_id)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="state must be APPROVED or REJECTED",
            )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    return GlossarySuggestionItem(
        id=row.id,
        term_source=row.term_source,
        term_target=row.term_target,
        kind=row.kind.value,
        confidence=row.confidence,
        evidence_json=row.evidence_json or [],
        state=row.state.value,
    )


@router.post(
    "/{job_id}/glossary/suggestions/bulk",
    response_model=dict,
)
def bulk_glossary_suggestions(
    job_id: str,
    body: SuggestionsBulkRequest,
    db: Session = Depends(get_db),
) -> dict:
    """Approve or reject many suggestions in one call. Returns counts."""
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    action = body.action
    pending = glossary_suggester.list_suggestions(
        db, job_id, state=SuggestionState.PENDING
    )
    approved = rejected = 0
    if action == "approve_all":
        for s in pending:
            glossary_suggester.approve_suggestion(db, job_id, s.id)
            approved += 1
    elif action == "reject_all":
        for s in pending:
            glossary_suggester.reject_suggestion(db, job_id, s.id)
            rejected += 1
    elif action == "approve_ids":
        if not body.ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="approve_ids requires non-empty 'ids'",
            )
        for sid in body.ids:
            try:
                glossary_suggester.approve_suggestion(db, job_id, sid)
                approved += 1
            except ValueError:
                continue
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="action must be approve_all, reject_all, or approve_ids",
        )
    return {"approved": approved, "rejected": rejected}


@router.get("/{job_id}/quality-scores", response_model=list[QualityScoreItem])
def get_quality_scores(
    job_id: str, db: Session = Depends(get_db)
) -> list[QualityScoreItem]:
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    rows = metrics.quality_scores_for_job(db, job_id)
    chunks_by_id = {c.id: c for c in job.chunks}
    return [
        QualityScoreItem(
            chunk_id=r.chunk_id,
            chunk_index=chunks_by_id[r.chunk_id].chunk_index if r.chunk_id in chunks_by_id else -1,
            pass_id=r.pass_id,
            naturalness=r.naturalness,
            fidelity=r.fidelity,
            register_match=r.register_match,
            consistency=r.consistency,
            overall=r.overall,
            issues_count=len((r.reasons_json or {}).get("issues") or []),
            judge_model=r.judge_model,
            created_at=r.created_at,
        )
        for r in rows
    ]


@router.get("/{job_id}/review-queue", response_model=list[QualityScoreItem])
def get_review_queue(
    job_id: str,
    max_overall: float | None = Query(default=None, ge=1.0, le=5.0),
    db: Session = Depends(get_db),
) -> list[QualityScoreItem]:
    """Chunks below the low-score threshold, worst first.

    The default threshold falls back to ``settings.low_score_threshold`` so the
    UI can hit this endpoint with no params and get the queue an editor should
    review."""
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    rows = metrics.low_score_chunks(db, job_id, max_overall=max_overall)
    chunks_by_id = {c.id: c for c in job.chunks}
    return [
        QualityScoreItem(
            chunk_id=r.chunk_id,
            chunk_index=chunks_by_id[r.chunk_id].chunk_index if r.chunk_id in chunks_by_id else -1,
            pass_id=r.pass_id,
            naturalness=r.naturalness,
            fidelity=r.fidelity,
            register_match=r.register_match,
            consistency=r.consistency,
            overall=r.overall,
            issues_count=len((r.reasons_json or {}).get("issues") or []),
            judge_model=r.judge_model,
            created_at=r.created_at,
        )
        for r in rows
    ]


@router.get("/{job_id}/metrics", response_model=JobMetricsSummary)
def get_job_metrics(job_id: str, db: Session = Depends(get_db)) -> JobMetricsSummary:
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    return JobMetricsSummary(**metrics.job_metrics(db, job_id))


@router.get("/{job_id}/compare", response_model=CompareResponse)
def get_compare_view(
    job_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> CompareResponse:
    """Return source + translation aligned by ``block_id`` for the compare UI.

    Walks the structure.json pages in order so the compare view paginates the
    same way as the original PDF. Translated blocks come from
    ``translation_chunks.translated_blocks_json``; missing entries (chunks
    not yet translated) leave ``translated=null`` so the UI can render an
    empty cell instead of a blank page.
    """
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    if job.structure is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="extraction has not produced a structure yet",
        )
    structure_path = Path(job.structure.structured_json_path)
    if not structure_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"structure file missing at {structure_path}",
        )

    # Build a block_id → translated text map across all chunks.
    translated_by_block: dict[str, str] = {}
    for chunk in job.chunks:
        for entry in chunk.translated_blocks_json or []:
            bid = entry.get("block_id")
            txt = entry.get("translated")
            if isinstance(bid, str) and isinstance(txt, str):
                translated_by_block[bid] = txt

    with structure_path.open("r", encoding="utf-8") as fh:
        structure = json.load(fh)

    pages_out: list[ComparePage] = []
    blocks_total = 0
    blocks_translated = 0
    for page in structure.get("pages") or []:
        page_blocks: list[CompareBlock] = []
        for block in page.get("blocks") or []:
            btype = block.get("type") or ""
            if btype not in {"paragraph", "heading", "ocr"}:
                continue
            bid = block.get("block_id") or ""
            source = (block.get("text") or "").strip()
            if not bid or not source:
                continue
            translated = translated_by_block.get(bid)
            page_blocks.append(
                CompareBlock(
                    block_id=bid,
                    type=btype,
                    source=source,
                    translated=translated,
                )
            )
            blocks_total += 1
            if translated:
                blocks_translated += 1
        if page_blocks:
            pages_out.append(
                ComparePage(
                    page_number=int(page.get("page_number") or len(pages_out) + 1),
                    blocks=page_blocks,
                )
            )

    return CompareResponse(
        job_id=job_id,
        original_filename=job.original_filename,
        pages=pages_out,
        pages_total=len(pages_out),
        blocks_total=blocks_total,
        blocks_translated=blocks_translated,
    )


@router.get("/{job_id}/cost-estimate", response_model=CostEstimateResponse)
def get_cost_estimate(job_id: str, db: Session = Depends(get_db)) -> CostEstimateResponse:
    """Forward cost estimate (USD) plus rolling actual spend.

    Available as soon as extraction lands a structure.json. The three
    scenarios let the user choose whether the quality loop is worth the cost
    *before* clicking Start translation. Actuals come from the per-pass
    token rows so the running total during translation is exact, not
    projected."""
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    estimate = cost_estimator.estimate_job_cost(db, job_id)
    estimate["actual"] = cost_estimator.actual_job_cost(db, job_id)
    return CostEstimateResponse(**estimate)


@router.get("/{job_id}/chunks", response_model=list[ChunkSummary])
def get_chunks(job_id: str, db: Session = Depends(get_db)) -> list[ChunkSummary]:
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    return [
        ChunkSummary(
            chunk_index=c.chunk_index,
            page_range_start=c.page_range_start,
            page_range_end=c.page_range_end,
            token_count=c.token_count,
            status=c.status.value,
            attempt_count=c.attempt_count,
            cost_tokens=c.cost_tokens,
        )
        for c in sorted(job.chunks, key=lambda c: c.chunk_index)
    ]


@router.get("/{job_id}/translation", response_model=TranslationOutputResponse)
def get_translation(job_id: str, db: Session = Depends(get_db)) -> TranslationOutputResponse:
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    chunks = sorted(job.chunks, key=lambda c: c.chunk_index)
    completed = [c for c in chunks if c.translated_text]
    # Build page-keyed merged text (best-effort: chunk page_range_start as the key)
    by_page: dict[int, list[str]] = {}
    for c in completed:
        for block in c.translated_blocks_json or []:
            block_id = block.get("block_id") or ""
            page_no = _parse_page_from_block_id(block_id) or c.page_range_start
            text = block.get("translated") or ""
            if not text:
                continue
            by_page.setdefault(page_no, []).append(text)
    pages = [
        {"page_number": pn, "translated_text": "\n\n".join(by_page[pn])}
        for pn in sorted(by_page)
    ]
    plain = "\n\n".join(c.translated_text or "" for c in completed if c.translated_text)
    return TranslationOutputResponse(
        job_id=job_id,
        chunks_total=len(chunks),
        chunks_completed=len(completed),
        pages=pages,
        plain_text=plain,
    )


def _parse_page_from_block_id(block_id: str) -> int | None:
    if not block_id.startswith("p"):
        return None
    rest = block_id[1:]
    digits = ""
    for ch in rest:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else None


@router.get("/{job_id}/outputs", response_model=OutputListResponse)
def get_outputs(job_id: str, db: Session = Depends(get_db)) -> OutputListResponse:
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    items = [
        OutputItem(format=o.format.value, file_path=o.file_path, size_bytes=o.size_bytes)
        for o in list_outputs(db, job_id)
    ]
    return OutputListResponse(job_id=job_id, outputs=items)


@router.post("/{job_id}/preview-page")
def preview_page(
    job_id: str,
    page: int = Query(default=1, ge=1),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """Render one page of the translation as a PDF for typography QA.

    Useful for verifying Burmese shaping (Pyidaungsu vs Padauk vs Noto),
    line-height, and overflow before committing to a full rebuild. Returns
    the rendered PDF inline."""
    from fastapi.responses import FileResponse

    from app.services.rebuild_pdf import rebuild_pdf

    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    structure_path = settings.intermediate_dir / job_id / "structure.json"
    if not structure_path.exists() or job.structure is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="extraction has not produced a structure yet",
        )

    # Build the per-block translations dict (filtered to this page).
    chunks = [c for c in job.chunks if c.translated_blocks_json]
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="no translated chunks yet — run translation first",
        )
    translations: dict[str, str] = {}
    for c in chunks:
        for b in c.translated_blocks_json or []:
            bid = b.get("block_id")
            txt = b.get("translated") or ""
            if bid:
                translations[bid] = txt

    # Build a 1-page slice of the structure (only the requested page's
    # blocks). The rebuild walks pages in order and ignores pages outside
    # the source PDF's page count, so slicing the structure is sufficient.
    with structure_path.open("r", encoding="utf-8") as fh:
        full_structure = json.load(fh)
    page_idx = page - 1  # 1-indexed for the API; 0-indexed internally
    pages = full_structure.get("pages") or []
    if page_idx < 0 or page_idx >= len(pages):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"page {page} out of range (1..{len(pages)})",
        )
    sliced = {**full_structure, "pages": [pages[page_idx]]}
    sliced_path = settings.intermediate_dir / job_id / f"_preview_page_{page}.json"
    sliced_path.parent.mkdir(parents=True, exist_ok=True)
    with sliced_path.open("w", encoding="utf-8") as fh:
        json.dump(sliced, fh, ensure_ascii=False)

    out_dir = settings.outputs_dir / job_id / "previews"
    output_name = f"preview_page_{page}.pdf"
    rebuild_pdf(
        source_pdf=Path(job.file_path),
        structure_path=sliced_path,
        translations_by_block=translations,
        output_dir=out_dir,
        output_name=output_name,
    )
    out_path = out_dir / output_name
    return FileResponse(
        out_path,
        media_type="application/pdf",
        filename=f"{job_id}-preview-page-{page}.pdf",
    )


@router.post("/{job_id}/start", response_model=JobStatusResponse)
def start_translation(
    job_id: str,
    db: Session = Depends(get_db),
) -> JobStatusResponse:
    """Begin translation. Stub for M3 — sets status and enqueues the (still-stub) translate task."""
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    if job.status != JobStatus.AWAITING_CONFIRMATION:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"cannot start in status={job.status.value}; awaiting category confirmation",
        )
    if job.category not in {
        Category.NOVEL,
        Category.EDUCATIONAL,
        Category.PERSONAL_DEVELOPMENT,
    }:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="set category before starting translation",
        )
    try:
        from app.workers.tasks import translate_task

        translate_task.delay(job_id)
    except Exception:
        log.exception("failed to enqueue translate_task for job_id=%s", job_id)
    return JobStatusResponse.model_validate(job)


@router.post("/{job_id}/cancel", response_model=JobStatusResponse)
def cancel_job(job_id: str, db: Session = Depends(get_db)) -> JobStatusResponse:
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    if job.status in {JobStatus.COMPLETED, JobStatus.CANCELLED, JobStatus.FAILED}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"cannot cancel in status={job.status.value}",
        )
    request_cancel(db, job)
    return JobStatusResponse.model_validate(job)


@router.post("/{job_id}/resume", response_model=JobStatusResponse)
def resume_job(job_id: str, db: Session = Depends(get_db)) -> JobStatusResponse:
    """Resume a job that previously failed, was cancelled, or had its worker
    crash mid-translate/rebuild. Picks the right re-entry point based on what
    artifacts exist."""
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    if job.status == JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="job already COMPLETED; nothing to resume",
        )

    reset_for_resume(db, job)

    # Decide where to pick up. We re-import lazily so unit tests can patch.
    from app.workers.tasks import (  # noqa: F401  (rebuild_task imported lazily below)
        classify_task,
        extract_task,
        rebuild_task,
        translate_task,
    )

    has_chunks = len(job.chunks) > 0
    has_outputs = len(job.outputs) > 0

    if has_outputs:
        # All work done; rebuild produced outputs but the job didn't reach COMPLETED.
        # Re-run rebuild as a no-op-style refresh so status flips correctly.
        target = rebuild_task
    elif has_chunks:
        # Translation was in progress. Re-running translate_task picks up
        # PENDING/RETRY chunks (now reset) and chains into rebuild.
        target = translate_task
    elif job.structure is not None:
        # Extraction done but translation not yet started: behave like /start.
        target = translate_task
    else:
        # Nothing produced yet: restart from extraction.
        target = extract_task

    try:
        target.delay(job_id)
    except Exception:
        log.exception("failed to enqueue resume task for job_id=%s", job_id)
    return JobStatusResponse.model_validate(job)


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_job(
    job_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> None:
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    source = Path(job.file_path).resolve()
    storage_root = settings.storage_root.resolve()
    if source.exists() and source.is_relative_to(storage_root):
        source.unlink(missing_ok=True)
    job_intermediate = settings.intermediate_dir / job.id
    if job_intermediate.exists():
        shutil.rmtree(job_intermediate, ignore_errors=True)
    job_outputs = settings.outputs_dir / job.id
    if job_outputs.exists():
        shutil.rmtree(job_outputs, ignore_errors=True)
    db.delete(job)
    db.commit()

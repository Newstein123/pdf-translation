"""Admin routes (single-user local tool — no role separation).

Exposes retention/purge and the M9 metrics aggregator.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.auth import require_token
from app.db.session import get_db
from app.schemas.jobs import AdminMetricsResponse
from app.services import metrics
from app.services.retention import purge_old_artifacts

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_token)])


@router.post("/purge")
def purge(
    dry_run: bool = Query(default=False),
    override_hours: int | None = Query(default=None, ge=0),
) -> dict:
    """Delete artifacts (uploads, intermediate JSON, outputs) for jobs that
    completed/failed/cancelled more than ``ARTIFACT_RETENTION_HOURS`` ago.

    Pass ``dry_run=true`` to preview without deleting. ``override_hours`` lets
    you force a different threshold (e.g. ``0`` to purge everything terminal)."""
    result = purge_old_artifacts(dry_run=dry_run, override_hours=override_hours)
    return {
        "candidates": result.candidates,
        "deleted_jobs": result.deleted_jobs,
        "bytes_freed": result.bytes_freed,
        "dry_run": result.dry_run,
    }


@router.get("/metrics", response_model=AdminMetricsResponse)
def admin_metrics(db: Session = Depends(get_db)) -> AdminMetricsResponse:
    """Aggregated M9 metrics across all jobs.

    Tokens-by-pass-kind, average judge overall, revise rate, and counts.
    Used by the admin dashboard to spot prompt-version regressions and to
    keep an eye on the cost/quality tradeoff over time."""
    return AdminMetricsResponse(**metrics.admin_metrics(db))

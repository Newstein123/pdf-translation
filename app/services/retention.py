"""Artifact retention.

Removes uploads, intermediate JSON, and outputs older than
``ARTIFACT_RETENTION_HOURS``. The job DB row is kept for the audit trail
unless ``delete_db_rows=True``. Set retention to 0 to disable.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.config import get_settings
from app.db.models import Job, JobStatus
from app.db.session import SessionLocal

log = logging.getLogger(__name__)


@dataclass
class PurgeResult:
    candidates: int
    deleted_jobs: list[str]
    bytes_freed: int
    dry_run: bool


_TERMINAL = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}


def purge_old_artifacts(*, dry_run: bool = False, override_hours: int | None = None) -> PurgeResult:
    s = get_settings()
    hours = override_hours if override_hours is not None else s.artifact_retention_hours
    if hours <= 0:
        return PurgeResult(candidates=0, deleted_jobs=[], bytes_freed=0, dry_run=dry_run)

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    bytes_freed = 0
    deleted: list[str] = []

    with SessionLocal() as db:
        rows = (
            db.query(Job)
            .filter(Job.status.in_(_TERMINAL))
            .all()
        )
        candidates = [
            j for j in rows
            if (j.completed_at or j.updated_at).replace(tzinfo=timezone.utc) < cutoff
        ]

        for job in candidates:
            paths: list[Path] = []
            try:
                paths.append(Path(job.file_path))
            except Exception:
                pass
            paths.append(s.intermediate_dir / job.id)
            paths.append(s.outputs_dir / job.id)

            for p in paths:
                bytes_freed += _path_size(p)
                if not dry_run:
                    _remove_path(p)

            if not dry_run:
                # Reset structure / outputs / chunks pointers so a stale row doesn't
                # claim files that are now gone. We keep the Job row itself.
                job.error_message = (job.error_message or "") + " [artifacts purged]"
                if job.structure is not None:
                    job.structure.structured_json_path = ""
                for o in list(job.outputs):
                    db.delete(o)

            deleted.append(job.id)

        if not dry_run:
            db.commit()

    return PurgeResult(
        candidates=len(candidates),
        deleted_jobs=deleted,
        bytes_freed=bytes_freed,
        dry_run=dry_run,
    )


def _path_size(p: Path) -> int:
    if not p.exists():
        return 0
    if p.is_file():
        try:
            return p.stat().st_size
        except OSError:
            return 0
    total = 0
    for sub in p.rglob("*"):
        if sub.is_file():
            try:
                total += sub.stat().st_size
            except OSError:
                pass
    return total


def _remove_path(p: Path) -> None:
    if not p.exists():
        return
    try:
        if p.is_file():
            p.unlink(missing_ok=True)
        else:
            shutil.rmtree(p, ignore_errors=True)
    except Exception:
        log.exception("failed to remove %s", p)

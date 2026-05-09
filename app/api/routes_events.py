"""Server-Sent Events stream of per-job progress.

We poll ``job_events`` for new rows since the last cursor and emit each as an
SSE message. The stream auto-closes when the job hits a terminal status.

Auth uses the flexible ``Authorization`` header OR ``?token=`` query param so
plain ``EventSource`` (which can't set headers) can subscribe.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, status
from sse_starlette.sse import EventSourceResponse

from app.api.auth import require_token_flexible
from app.db.models import JobEvent, JobStatus
from app.db.session import SessionLocal

log = logging.getLogger(__name__)

_TERMINAL = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
_POLL_SECONDS = 1.0
_MAX_STREAM_SECONDS = 60 * 60 * 2  # 2 hours; safety bound for stuck streams

router = APIRouter(
    prefix="/api/jobs",
    tags=["events"],
    dependencies=[Depends(require_token_flexible)],
)


@router.get("/{job_id}/events")
async def stream_events(job_id: str) -> EventSourceResponse:
    # Validate the job exists before opening the stream.
    with SessionLocal() as db:
        from app.db.models import Job

        job = db.get(Job, job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="job not found"
            )

    async def gen() -> AsyncIterator[dict]:
        last_event_at: str | None = None
        elapsed = 0.0

        # Send a hello so the client knows the stream is open.
        yield {
            "event": "hello",
            "data": json.dumps({"job_id": job_id}),
        }

        while elapsed < _MAX_STREAM_SECONDS:
            with SessionLocal() as db:
                q = db.query(JobEvent).filter(JobEvent.job_id == job_id)
                if last_event_at is not None:
                    q = q.filter(JobEvent.created_at > last_event_at)
                rows = q.order_by(JobEvent.created_at, JobEvent.id).all()

                for row in rows:
                    last_event_at = row.created_at.isoformat()
                    yield {
                        "event": row.event_type,
                        "id": row.id,
                        "data": json.dumps(
                            {
                                "event_type": row.event_type,
                                "payload": row.payload_json,
                                "created_at": row.created_at.isoformat(),
                            },
                            ensure_ascii=False,
                        ),
                    }

                from app.db.models import Job  # local import to avoid cycle

                current = db.get(Job, job_id)
                terminal = current is not None and current.status in _TERMINAL
                status_payload = {
                    "status": current.status.value if current else "MISSING",
                    "progress_percent": current.progress_percent if current else 0,
                    "category": current.category.value if current else "UNKNOWN",
                }

            yield {"event": "status", "data": json.dumps(status_payload)}

            if terminal:
                yield {"event": "done", "data": json.dumps(status_payload)}
                return

            await asyncio.sleep(_POLL_SECONDS)
            elapsed += _POLL_SECONDS

        yield {"event": "timeout", "data": json.dumps({"job_id": job_id})}

    return EventSourceResponse(gen())

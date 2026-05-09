"""Output downloads.

These live on a separate router that accepts the bearer token via either the
``Authorization`` header OR the ``?token=`` query param. ``<a href>`` clicks
from the UI cannot attach headers, so query-param auth is required for direct
download links.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.auth import require_token_flexible
from app.db.models import OutputFormat
from app.db.session import get_db
from app.services.jobs import get_job, list_outputs

router = APIRouter(
    prefix="/api/jobs",
    tags=["downloads"],
    dependencies=[Depends(require_token_flexible)],
)


@router.get("/{job_id}/outputs/{fmt}")
def download_output(
    job_id: str,
    fmt: str,
    inline: bool = Query(default=False),
    db: Session = Depends(get_db),
):
    """Serve an output PDF/EPUB.

    By default sends ``Content-Disposition: attachment`` so anchor clicks
    trigger a download. Pass ``?inline=1`` to switch to ``inline`` so the
    browser previews the file in an iframe / new tab instead — used by the
    in-page PDF preview pane and the History "PDF" buttons. EPUBs are
    always sent as attachments because browsers can't preview them
    natively."""
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    try:
        target_fmt = OutputFormat(fmt.upper())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="format must be PDF or EPUB",
        ) from None
    output = next((o for o in list_outputs(db, job_id) if o.format == target_fmt), None)
    if output is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no {target_fmt.value} output for job",
        )
    path = Path(output.file_path)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"output file missing on disk: {path}",
        )
    media_type = (
        "application/pdf" if target_fmt == OutputFormat.PDF else "application/epub+zip"
    )
    # ``filename=`` makes FileResponse default to ``Content-Disposition:
    # attachment``; pass content_disposition_type explicitly when we want
    # the browser to render inline.
    if inline and target_fmt == OutputFormat.PDF:
        return FileResponse(
            path,
            media_type=media_type,
            filename=path.name,
            content_disposition_type="inline",
        )
    return FileResponse(path, media_type=media_type, filename=path.name)


@router.get("/{job_id}/cover")
def download_cover(job_id: str, db: Session = Depends(get_db)):
    """Serve the saved cover image so the UI can render a preview <img>.

    Lives on this router (not ``routes_jobs``) because the browser cannot
    attach an ``Authorization`` header to an ``<img src=>`` request — only
    a query-string token. ``require_token_flexible`` is the router default
    here, which accepts either."""
    from app.db.models import Job

    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    if not job.cover_image_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no cover image")
    path = Path(job.cover_image_path)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"cover file missing on disk: {path}",
        )
    return FileResponse(path, media_type="image/jpeg")

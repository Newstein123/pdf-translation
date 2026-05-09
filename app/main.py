import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes_admin import router as admin_router
from app.api.routes_downloads import router as downloads_router
from app.api.routes_events import router as events_router
from app.api.routes_jobs import router as jobs_router
from app.api.routes_style_profiles import router as style_profiles_router
from app.config import get_settings
from app.db.session import SessionLocal
from app.services.style_profiles import sync_profiles_to_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    get_settings().ensure_dirs()
    try:
        with SessionLocal() as db:
            sync_profiles_to_db(db)
    except Exception:
        # Profile sync failure must not block app boot — the existing single-pass
        # path still works without profiles. Surface as a warning so the operator
        # can fix the YAML/DB issue without losing service.
        logger.exception("style profile sync failed at boot; continuing without sync")
    yield


app = FastAPI(title="PDF Translation (Burmese)", version="0.1.0", lifespan=lifespan)
app.include_router(jobs_router)
app.include_router(events_router)
app.include_router(downloads_router)
app.include_router(admin_router)
app.include_router(style_profiles_router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_WEB_DIR / "index.html")


@app.get("/history", include_in_schema=False)
def history() -> FileResponse:
    return FileResponse(_WEB_DIR / "history.html")


@app.get("/compare", include_in_schema=False)
def compare() -> FileResponse:
    return FileResponse(_WEB_DIR / "compare.html")


# Static assets (JS, CSS).
if (_WEB_DIR / "static").is_dir():
    app.mount("/static", StaticFiles(directory=_WEB_DIR / "static"), name="static")

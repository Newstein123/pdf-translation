"""Test fixtures.

Critical safety note: tests MUST run against a temp SQLite DB, NOT the
production ``storage/pdf_translation.db``. We achieve this by:

  1. Mutating env vars BEFORE any app module reads ``settings``.
  2. Clearing ``get_settings.cache_clear()`` so pydantic-settings re-reads.
  3. Disabling ``.env`` loading so the user's local ``GEMINI_API_KEY`` /
     ``DATABASE_URL`` cannot leak into the test process.
  4. Calling ``reset_engine()`` so any module that imported ``SessionLocal``
     before us picks up the test DB on next call.
  5. Asserting the active database URL points at the temp dir before any
     test runs. If a future regression rebinds to prod, the assertion trips
     immediately rather than silently writing test rows into prod.
"""

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolate_env(tmp_path_factory: pytest.TempPathFactory) -> None:
    tmp = tmp_path_factory.mktemp("pdftrans")
    os.environ["STORAGE_ROOT"] = str(tmp)
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp}/test.db"
    os.environ["API_BEARER_TOKEN"] = "test-token"
    os.environ["CELERY_BROKER_URL"] = "memory://"
    os.environ["CELERY_RESULT_BACKEND"] = "cache+memory://"
    os.environ["GEMINI_API_KEY"] = ""
    # M7 quality loop calls Gemini; tests stub translate_chunk but not critique.
    # Disable globally so the existing stubbed pipeline tests don't try to call
    # an unstubbed critique path. Tests that exercise critique override this.
    os.environ["CRITIQUE_ENABLED"] = "false"

    from app.config import Settings, get_settings

    get_settings.cache_clear()
    Settings.model_config["env_file"] = None  # type: ignore[index]

    # Disconnect any pre-built engine that might have bound to prod settings.
    from app.db.session import reset_engine

    reset_engine()

    # Hard guard: the next session.engine call MUST point at our temp file.
    from app.db.session import get_engine

    engine = get_engine()
    expected = f"sqlite:///{tmp}/test.db"
    assert str(engine.url) == expected, (
        f"FATAL: tests bound to wrong database. "
        f"Expected {expected}, got {engine.url}. Aborting to prevent prod data loss."
    )


@pytest.fixture(scope="session")
def db_engine():
    from app.db.models import Base
    from app.db.session import get_engine

    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    yield engine


@pytest.fixture
def client(db_engine):
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


@pytest.fixture
def auth_headers() -> dict:
    return {"Authorization": "Bearer test-token"}


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    import pymupdf

    path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello, world. This is a test document.")
    doc.save(path)
    doc.close()
    return path

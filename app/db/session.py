"""SQLAlchemy session/engine management.

The engine is created lazily on first use so test fixtures can mutate the
``DATABASE_URL`` env var (and ``settings``) BEFORE any session is opened.
A previous version eagerly built the engine at import time, which silently
bound tests to the production SQLite file when conftest's env-var override
arrived too late. That meant pytest writes — including
``test_purge_deletes_artifacts`` — could destroy real user data.

Public surface (unchanged for callers):
  - ``SessionLocal()`` returns a new ``Session``.
  - ``get_engine()`` returns the underlying engine (for migrations / metadata).
  - ``get_db()`` is the FastAPI dependency yielding a session.
  - ``reset_engine()`` is a test helper that disposes and forgets the engine
    so the next call rebuilds it against current settings.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _initialize() -> None:
    global _engine, _session_factory
    if _engine is not None:
        return
    s = get_settings()
    connect_args: dict[str, Any] = (
        {"check_same_thread": False} if s.database_url.startswith("sqlite") else {}
    )
    _engine = create_engine(s.database_url, future=True, connect_args=connect_args)
    _session_factory = sessionmaker(
        bind=_engine, autoflush=False, autocommit=False, expire_on_commit=False
    )


def get_engine() -> Engine:
    _initialize()
    assert _engine is not None
    return _engine


def reset_engine() -> None:
    """Dispose and forget the engine. Tests call this after mutating settings."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None


class _SessionLocal:
    """Callable proxy that returns a Session, initializing on first call.

    Existing code uses ``with SessionLocal() as db:`` and ``SessionLocal()``
    interchangeably — both still work."""

    def __call__(self) -> Session:
        _initialize()
        assert _session_factory is not None
        return _session_factory()


SessionLocal = _SessionLocal()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

"""Idempotent registry for the exact prompt body that produced each translation.

Why this exists: chunks are translated under a particular *combination* of
(base prompt + profile goals override + few-shot examples). When a curator
edits a profile YAML, the next sync bumps the profile version — but we still
need every prior chunk's pass row to point at the prompt body that was used
*at translate time*. ``prompt_versions`` is the immutable, hash-keyed cache
that survives profile edits.

The helper is process-safe: under a UNIQUE-violation race two workers
producing the same body will land on the same row.
"""
from __future__ import annotations

import hashlib
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import PromptPurpose, PromptVersion


def hash_body(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def get_or_create_prompt_version(
    session: Session,
    *,
    purpose: PromptPurpose,
    key: str,
    body: str,
    metadata: dict[str, Any] | None = None,
) -> PromptVersion:
    """Return the existing row matching (purpose, key, hash) or create one.

    The triple (purpose, key, hash) is logically unique. We do not declare
    a DB-level unique constraint because curated edits over time will create
    multiple active versions intentionally; instead we treat the "latest body
    per (purpose, key)" as the active one for new translations and snapshot
    the rest for old jobs to point back at.
    """
    body_hash = hash_body(body)

    existing = session.execute(
        select(PromptVersion).where(
            PromptVersion.purpose == purpose,
            PromptVersion.key == key,
            PromptVersion.hash == body_hash,
        ).limit(1)
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    # Determine the next version number for this (purpose, key) so each row's
    # version is monotonic.
    latest_version = session.execute(
        select(PromptVersion.version).where(
            PromptVersion.purpose == purpose,
            PromptVersion.key == key,
        ).order_by(PromptVersion.version.desc()).limit(1)
    ).scalar_one_or_none()
    next_version = (latest_version or 0) + 1

    # Mark previous active rows for this (purpose, key) inactive so the cache
    # has a clear "current" entry. Old entries remain queryable for audit.
    session.execute(
        select(PromptVersion).where(
            PromptVersion.purpose == purpose,
            PromptVersion.key == key,
            PromptVersion.active.is_(True),
        )
    )
    for row in session.execute(
        select(PromptVersion).where(
            PromptVersion.purpose == purpose,
            PromptVersion.key == key,
            PromptVersion.active.is_(True),
        )
    ).scalars():
        row.active = False

    new_row = PromptVersion(
        purpose=purpose,
        key=key,
        version=next_version,
        hash=body_hash,
        body=body,
        metadata_json=metadata,
        active=True,
    )
    session.add(new_row)
    try:
        session.commit()
    except IntegrityError:
        # Another worker raced us. Roll back and re-fetch the row that won.
        session.rollback()
        winner = session.execute(
            select(PromptVersion).where(
                PromptVersion.purpose == purpose,
                PromptVersion.key == key,
                PromptVersion.hash == body_hash,
            ).limit(1)
        ).scalar_one_or_none()
        if winner is None:
            raise
        return winner
    return new_row

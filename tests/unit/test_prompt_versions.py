"""Unit tests for the prompt_versions hash-keyed cache.

Covers:
- Identical body returns the same row (idempotent).
- Edited body inserts a new row, bumps version, deactivates the prior active row.
- Same body resurrects the cached row even if a sibling version exists.
"""
from __future__ import annotations

from app.db.models import PromptPurpose, PromptVersion
from app.db.session import SessionLocal
from app.services.prompt_versions import get_or_create_prompt_version, hash_body


def _count(db, *, purpose, key) -> int:
    return (
        db.query(PromptVersion)
        .filter(PromptVersion.purpose == purpose, PromptVersion.key == key)
        .count()
    )


def test_get_or_create_idempotent_on_same_body(db_engine) -> None:
    body = "system prompt body for translate__t1"
    with SessionLocal() as db:
        a = get_or_create_prompt_version(
            db, purpose=PromptPurpose.TRANSLATE, key="translate__t1", body=body
        )
        b = get_or_create_prompt_version(
            db, purpose=PromptPurpose.TRANSLATE, key="translate__t1", body=body
        )
        assert a.id == b.id
        assert a.hash == hash_body(body)
        assert _count(db, purpose=PromptPurpose.TRANSLATE, key="translate__t1") == 1


def test_get_or_create_bumps_version_on_edit(db_engine) -> None:
    with SessionLocal() as db:
        v1 = get_or_create_prompt_version(
            db, purpose=PromptPurpose.TRANSLATE, key="translate__t2", body="v1"
        )
        assert v1.version == 1
        assert v1.active is True

        v2 = get_or_create_prompt_version(
            db, purpose=PromptPurpose.TRANSLATE, key="translate__t2", body="v2"
        )
        assert v2.version == 2
        assert v2.active is True
        # Old row was deactivated so the cache has a clear "current" entry.
        db.refresh(v1)
        assert v1.active is False
        assert _count(db, purpose=PromptPurpose.TRANSLATE, key="translate__t2") == 2


def test_get_or_create_resurrects_cached_body(db_engine) -> None:
    """If a curator reverts the YAML to a previously-used body, we should
    return the existing row (and not insert a duplicate)."""
    with SessionLocal() as db:
        get_or_create_prompt_version(
            db, purpose=PromptPurpose.TRANSLATE, key="translate__t3", body="v1"
        )
        get_or_create_prompt_version(
            db, purpose=PromptPurpose.TRANSLATE, key="translate__t3", body="v2"
        )
        # Revert to v1.
        v1_again = get_or_create_prompt_version(
            db, purpose=PromptPurpose.TRANSLATE, key="translate__t3", body="v1"
        )
        # Cached row reused (same id), no fourth row inserted.
        assert v1_again.hash == hash_body("v1")
        assert _count(db, purpose=PromptPurpose.TRANSLATE, key="translate__t3") == 2


def test_purpose_and_key_partition_cache(db_engine) -> None:
    with SessionLocal() as db:
        a = get_or_create_prompt_version(
            db, purpose=PromptPurpose.TRANSLATE, key="translate__t4", body="same"
        )
        b = get_or_create_prompt_version(
            db, purpose=PromptPurpose.CRITIQUE, key="critique__t4", body="same"
        )
        assert a.id != b.id
        assert _count(db, purpose=PromptPurpose.TRANSLATE, key="translate__t4") == 1
        assert _count(db, purpose=PromptPurpose.CRITIQUE, key="critique__t4") == 1

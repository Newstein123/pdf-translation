"""Style-profile loader and DB synchronizer.

Profiles live as YAML files under ``app/style_profiles/`` and are upserted
into the ``style_profiles`` table at app boot. Each YAML file declares a
profile by ``key``; the on-disk YAMLs are the source of truth, the DB row
is the runtime cache.

Sync semantics:
- New ``key`` on disk → insert with version=1, active=true.
- Existing ``key`` with unchanged content hash → no-op.
- Existing ``key`` with new content hash → update in place, bump version.
- Existing DB ``key`` no longer on disk → mark active=false (preserves
  reproducibility for old jobs that pinned a profile id).
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import (
    Category,
    ProfileCategoryScope,
    SentenceComplexity,
    StyleProfile,
)

logger = logging.getLogger(__name__)

PROFILES_DIR = Path(__file__).resolve().parent.parent / "style_profiles"


@dataclass
class ProfileYAML:
    key: str
    name: str
    description: str | None
    applies_to_category: ProfileCategoryScope
    sentence_complexity: SentenceComplexity
    is_default_for_category: bool
    system_prompt_template: str
    few_shot_examples: list[dict[str, Any]]
    vocabulary_preferences: dict[str, Any]

    def content_hash(self) -> str:
        payload = {
            "system_prompt_template": self.system_prompt_template,
            "few_shot_examples": self.few_shot_examples,
            "vocabulary_preferences": self.vocabulary_preferences,
            "sentence_complexity": self.sentence_complexity.value,
            "applies_to_category": self.applies_to_category.value,
        }
        body = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _coerce_enum(raw: str | None, enum_cls, default):
    if raw is None:
        return default
    try:
        return enum_cls(raw.upper())
    except ValueError as e:
        raise ValueError(f"unknown {enum_cls.__name__} value: {raw!r}") from e


def parse_profile_yaml(path: Path) -> ProfileYAML:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level must be a mapping")

    key = data.get("key")
    if not key or not isinstance(key, str):
        raise ValueError(f"{path}: 'key' is required")

    return ProfileYAML(
        key=key,
        name=data.get("name", key),
        description=data.get("description"),
        applies_to_category=_coerce_enum(
            data.get("applies_to_category"), ProfileCategoryScope, ProfileCategoryScope.BOTH
        ),
        sentence_complexity=_coerce_enum(
            data.get("sentence_complexity"), SentenceComplexity, SentenceComplexity.MEDIUM
        ),
        is_default_for_category=bool(data.get("is_default_for_category", False)),
        system_prompt_template=data.get("system_prompt_template", "").strip(),
        few_shot_examples=list(data.get("few_shot_examples") or []),
        vocabulary_preferences=dict(data.get("vocabulary_preferences") or {}),
    )


def load_profiles_from_disk(directory: Path = PROFILES_DIR) -> list[ProfileYAML]:
    if not directory.is_dir():
        return []
    out: list[ProfileYAML] = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            out.append(parse_profile_yaml(path))
        except Exception:
            logger.exception("failed to parse style profile %s; skipping", path.name)
    # Detect duplicate keys early so the curator gets a clear error.
    seen: dict[str, Path] = {}
    for p, src in zip(out, sorted(directory.glob("*.yaml")), strict=False):
        prior = seen.get(p.key)
        if prior is not None:
            raise ValueError(f"duplicate style profile key {p.key!r} in {src} and {prior}")
        seen[p.key] = src
    return out


def sync_profiles_to_db(session: Session, directory: Path = PROFILES_DIR) -> dict[str, int]:
    """Reconcile the on-disk YAMLs with the DB.

    Returns counts: ``{"inserted", "updated", "deactivated", "unchanged"}``.
    """
    on_disk = {p.key: p for p in load_profiles_from_disk(directory)}
    existing = {row.key: row for row in session.execute(select(StyleProfile)).scalars()}

    counts = {"inserted": 0, "updated": 0, "deactivated": 0, "unchanged": 0}

    for key, yaml_profile in on_disk.items():
        new_hash = yaml_profile.content_hash()
        row = existing.get(key)
        if row is None:
            session.add(
                StyleProfile(
                    key=yaml_profile.key,
                    name=yaml_profile.name,
                    description=yaml_profile.description,
                    applies_to_category=yaml_profile.applies_to_category,
                    system_prompt_template=yaml_profile.system_prompt_template,
                    few_shot_examples_json=yaml_profile.few_shot_examples,
                    vocabulary_preferences_json=yaml_profile.vocabulary_preferences,
                    sentence_complexity=yaml_profile.sentence_complexity,
                    version=1,
                    is_default_for_category=yaml_profile.is_default_for_category,
                    active=True,
                )
            )
            counts["inserted"] += 1
            continue

        existing_hash = ProfileYAML(
            key=row.key,
            name=row.name,
            description=row.description,
            applies_to_category=row.applies_to_category,
            sentence_complexity=row.sentence_complexity,
            is_default_for_category=row.is_default_for_category,
            system_prompt_template=row.system_prompt_template,
            few_shot_examples=list(row.few_shot_examples_json or []),
            vocabulary_preferences=dict(row.vocabulary_preferences_json or {}),
        ).content_hash()

        if existing_hash == new_hash and row.active and (
            row.is_default_for_category == yaml_profile.is_default_for_category
            and row.name == yaml_profile.name
            and row.description == yaml_profile.description
        ):
            counts["unchanged"] += 1
            continue

        row.name = yaml_profile.name
        row.description = yaml_profile.description
        row.applies_to_category = yaml_profile.applies_to_category
        row.system_prompt_template = yaml_profile.system_prompt_template
        row.few_shot_examples_json = yaml_profile.few_shot_examples
        row.vocabulary_preferences_json = yaml_profile.vocabulary_preferences
        row.sentence_complexity = yaml_profile.sentence_complexity
        row.is_default_for_category = yaml_profile.is_default_for_category
        if existing_hash != new_hash:
            row.version = (row.version or 1) + 1
        row.active = True
        counts["updated"] += 1

    for key, row in existing.items():
        if key not in on_disk and row.active:
            row.active = False
            counts["deactivated"] += 1

    session.commit()
    logger.info("style profile sync: %s", counts)
    return counts


def get_default_profile_for_category(
    session: Session, category: Category
) -> StyleProfile | None:
    """Pick the active default profile for a job's resolved category.

    Falls back through (matching category default → BOTH default → any
    matching → None) so a misconfigured set still yields *something*.
    """
    if category == Category.NOVEL:
        scope = ProfileCategoryScope.NOVEL
    elif category == Category.EDUCATIONAL:
        scope = ProfileCategoryScope.EDUCATIONAL
    elif category == Category.PERSONAL_DEVELOPMENT:
        scope = ProfileCategoryScope.PERSONAL_DEVELOPMENT
    else:
        scope = None

    base = select(StyleProfile).where(StyleProfile.active.is_(True))
    if scope is not None:
        row = session.execute(
            base.where(
                StyleProfile.applies_to_category.in_([scope, ProfileCategoryScope.BOTH]),
                StyleProfile.is_default_for_category.is_(True),
            ).limit(1)
        ).scalar_one_or_none()
        if row is not None:
            return row
        row = session.execute(
            base.where(
                StyleProfile.applies_to_category.in_([scope, ProfileCategoryScope.BOTH])
            ).limit(1)
        ).scalar_one_or_none()
        if row is not None:
            return row

    return session.execute(base.limit(1)).scalar_one_or_none()


def get_profile_by_key(session: Session, key: str) -> StyleProfile | None:
    return session.execute(
        select(StyleProfile).where(StyleProfile.key == key)
    ).scalar_one_or_none()


def list_active_profiles(
    session: Session, category: Category | None = None
) -> list[StyleProfile]:
    stmt = select(StyleProfile).where(StyleProfile.active.is_(True)).order_by(StyleProfile.name)
    if category == Category.NOVEL:
        stmt = stmt.where(
            StyleProfile.applies_to_category.in_(
                [ProfileCategoryScope.NOVEL, ProfileCategoryScope.BOTH]
            )
        )
    elif category == Category.EDUCATIONAL:
        stmt = stmt.where(
            StyleProfile.applies_to_category.in_(
                [ProfileCategoryScope.EDUCATIONAL, ProfileCategoryScope.BOTH]
            )
        )
    elif category == Category.PERSONAL_DEVELOPMENT:
        stmt = stmt.where(
            StyleProfile.applies_to_category.in_(
                [ProfileCategoryScope.PERSONAL_DEVELOPMENT, ProfileCategoryScope.BOTH]
            )
        )
    return list(session.execute(stmt).scalars())

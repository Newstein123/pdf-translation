"""Unit tests for the M7 style-profile YAML loader and DB sync.

Covers: parsing, hash stability, insert/update/no-op/deactivate transitions,
and the default-by-category resolver. Each test runs against a temp directory
so the seeded YAMLs in app/style_profiles/ aren't required."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.db.models import (
    Category,
    ProfileCategoryScope,
    SentenceComplexity,
    StyleProfile,
)
from app.db.session import SessionLocal
from app.services.style_profiles import (
    get_default_profile_for_category,
    list_active_profiles,
    parse_profile_yaml,
    sync_profiles_to_db,
)


def _write(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


def _yaml(
    *,
    key: str,
    name: str = "Demo",
    category: str = "NOVEL",
    is_default: bool = False,
    template: str = "Goals override.",
    examples: list[dict] | None = None,
    vocabulary: dict | None = None,
    sentence_complexity: str = "MEDIUM",
) -> str:
    parts = [
        f"key: {key}",
        f"name: {name!r}",
        f"applies_to_category: {category}",
        f"sentence_complexity: {sentence_complexity}",
        f"is_default_for_category: {str(is_default).lower()}",
        f"system_prompt_template: |\n  {template}",
    ]
    if examples:
        import json as _json
        parts.append("few_shot_examples:")
        for ex in examples:
            parts.append(f"  - source: {_json.dumps(ex.get('source', ''))}")
            parts.append(f"    target: {_json.dumps(ex.get('target', ''))}")
    if vocabulary:
        parts.append("vocabulary_preferences:")
        for k, v in vocabulary.items():
            parts.append(f"  {k}: {v!r}")
    return "\n".join(parts) + "\n"


def test_parse_profile_yaml_minimum_fields(tmp_path: Path) -> None:
    p = _write(tmp_path, "demo.yaml", _yaml(key="demo"))
    parsed = parse_profile_yaml(p)
    assert parsed.key == "demo"
    assert parsed.applies_to_category == ProfileCategoryScope.NOVEL
    assert parsed.sentence_complexity == SentenceComplexity.MEDIUM
    assert parsed.is_default_for_category is False


def test_parse_profile_yaml_missing_key_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, "bad.yaml", "name: foo\n")
    with pytest.raises(ValueError, match="'key' is required"):
        parse_profile_yaml(p)


def test_content_hash_stable(tmp_path: Path) -> None:
    body = _yaml(key="demo", template="Hello world", vocabulary={"formality": "high"})
    a = parse_profile_yaml(_write(tmp_path, "a.yaml", body))
    b = parse_profile_yaml(_write(tmp_path, "b.yaml", body))
    assert a.content_hash() == b.content_hash()


def test_content_hash_changes_with_template(tmp_path: Path) -> None:
    base = parse_profile_yaml(
        _write(tmp_path, "a.yaml", _yaml(key="demo", template="Original goals."))
    )
    edited = parse_profile_yaml(
        _write(tmp_path, "b.yaml", _yaml(key="demo", template="Edited goals."))
    )
    assert base.content_hash() != edited.content_hash()


def test_sync_inserts_then_no_ops(db_engine, tmp_path: Path) -> None:
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    _write(profiles_dir, "p1.yaml", _yaml(key="p1_inserts"))

    with SessionLocal() as db:
        counts = sync_profiles_to_db(db, profiles_dir)
        assert counts["inserted"] == 1
        assert counts["updated"] == 0
        assert counts["unchanged"] == 0

    with SessionLocal() as db:
        counts = sync_profiles_to_db(db, profiles_dir)
        assert counts["inserted"] == 0
        assert counts["unchanged"] == 1


def test_sync_bumps_version_on_template_edit(db_engine, tmp_path: Path) -> None:
    profiles_dir = tmp_path / "profiles2"
    profiles_dir.mkdir()
    path = _write(
        profiles_dir, "p2.yaml", _yaml(key="p2_versioned", template="v1 template")
    )

    with SessionLocal() as db:
        sync_profiles_to_db(db, profiles_dir)
        row = db.query(StyleProfile).filter_by(key="p2_versioned").one()
        assert row.version == 1

    # Edit the template — the next sync should bump version and update the row.
    path.write_text(_yaml(key="p2_versioned", template="v2 template"), encoding="utf-8")

    with SessionLocal() as db:
        counts = sync_profiles_to_db(db, profiles_dir)
        assert counts["updated"] == 1
        row = db.query(StyleProfile).filter_by(key="p2_versioned").one()
        assert row.version == 2
        assert "v2 template" in row.system_prompt_template


def test_sync_deactivates_yaml_removed_from_disk(
    db_engine, tmp_path: Path
) -> None:
    profiles_dir = tmp_path / "profiles3"
    profiles_dir.mkdir()
    _write(profiles_dir, "kept.yaml", _yaml(key="p3_kept"))
    _write(profiles_dir, "doomed.yaml", _yaml(key="p3_doomed"))

    with SessionLocal() as db:
        sync_profiles_to_db(db, profiles_dir)

    (profiles_dir / "doomed.yaml").unlink()

    with SessionLocal() as db:
        counts = sync_profiles_to_db(db, profiles_dir)
        assert counts["deactivated"] == 1
        # Old jobs that pinned this profile must still be able to read it,
        # so the row is preserved with active=False, not deleted.
        doomed = db.query(StyleProfile).filter_by(key="p3_doomed").one()
        assert doomed.active is False
        kept = db.query(StyleProfile).filter_by(key="p3_kept").one()
        assert kept.active is True


def test_get_default_profile_picks_category_default(
    db_engine, tmp_path: Path
) -> None:
    profiles_dir = tmp_path / "profiles4"
    profiles_dir.mkdir()
    _write(profiles_dir, "novel_def.yaml", _yaml(key="p4_novel_default", is_default=True))
    _write(profiles_dir, "novel_alt.yaml", _yaml(key="p4_novel_alt"))
    _write(
        profiles_dir,
        "edu_def.yaml",
        _yaml(key="p4_edu_default", category="EDUCATIONAL", is_default=True),
    )

    with SessionLocal() as db:
        sync_profiles_to_db(db, profiles_dir)

        novel_default = get_default_profile_for_category(db, Category.NOVEL)
        edu_default = get_default_profile_for_category(db, Category.EDUCATIONAL)

        assert novel_default is not None and novel_default.key == "p4_novel_default"
        assert edu_default is not None and edu_default.key == "p4_edu_default"


def test_list_active_profiles_filters_by_category(
    db_engine, tmp_path: Path
) -> None:
    profiles_dir = tmp_path / "profiles5"
    profiles_dir.mkdir()
    _write(profiles_dir, "n.yaml", _yaml(key="p5_novel"))
    _write(
        profiles_dir, "e.yaml", _yaml(key="p5_edu", category="EDUCATIONAL")
    )
    _write(profiles_dir, "b.yaml", _yaml(key="p5_both", category="BOTH"))

    with SessionLocal() as db:
        sync_profiles_to_db(db, profiles_dir)
        novel = {p.key for p in list_active_profiles(db, Category.NOVEL)}
        edu = {p.key for p in list_active_profiles(db, Category.EDUCATIONAL)}
        all_ = {p.key for p in list_active_profiles(db, None)}

    # NOVEL filter returns NOVEL + BOTH; EDUCATIONAL filter returns EDUCATIONAL + BOTH.
    assert {"p5_novel", "p5_both"}.issubset(novel)
    assert "p5_edu" not in novel
    assert {"p5_edu", "p5_both"}.issubset(edu)
    assert "p5_novel" not in edu
    assert {"p5_novel", "p5_edu", "p5_both"}.issubset(all_)

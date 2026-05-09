"""Unit tests for profile-aware prompt composition + critique parsing."""
from __future__ import annotations

import pytest

from app.db.models import (
    Category,
    ProfileCategoryScope,
    SentenceComplexity,
    StyleProfile,
)
from app.services.translation import (
    CritiqueIssue,
    TranslationError,
    _parse_critique_response,
    _render_few_shot_examples,
    compose_critique_system_prompt,
    compose_revise_system_prompt,
    compose_translate_system_prompt,
    prompt_key_for_critique,
    prompt_key_for_revise,
    prompt_key_for_translate,
)


def _profile(
    *,
    key: str = "demo",
    category: ProfileCategoryScope = ProfileCategoryScope.NOVEL,
    template: str = "Profile-specific goals.",
    examples: list[dict] | None = None,
    vocabulary: dict | None = None,
) -> StyleProfile:
    return StyleProfile(
        key=key,
        name="Demo",
        description=None,
        applies_to_category=category,
        sentence_complexity=SentenceComplexity.MEDIUM,
        is_default_for_category=False,
        system_prompt_template=template,
        few_shot_examples_json=examples,
        vocabulary_preferences_json=vocabulary,
        version=1,
        active=True,
    )


def test_compose_translate_falls_back_to_base_when_no_profile() -> None:
    body = compose_translate_system_prompt(Category.NOVEL, None)
    # Sanity: contains the base-prompt invariants.
    assert "block_id" in body
    assert "summary_for_next_chunk" in body


def test_compose_translate_includes_profile_template_and_vocab() -> None:
    profile = _profile(
        template="Aim for elevated literary cadence.",
        vocabulary={"formality": "high"},
    )
    body = compose_translate_system_prompt(Category.NOVEL, profile)
    assert "Aim for elevated literary cadence." in body
    assert "Vocabulary preferences" in body
    assert '"formality": "high"' in body


def test_few_shot_renderer_skips_todo_placeholders() -> None:
    rendered = _render_few_shot_examples(
        [
            {"source": "Hello world.", "target": "[TODO: rendering]"},
            {"source": "Real source.", "target": "Real target.", "note": "n"},
        ]
    )
    assert "Real source." in rendered
    assert "Real target." in rendered
    assert "[TODO" not in rendered  # placeholder must not poison the prompt
    # Examples block always lists indices starting at 1 — even if earlier
    # placeholder was skipped the surviving block becomes EXAMPLE 1.
    assert "EXAMPLE 1" in rendered
    assert "EXAMPLE 2" not in rendered


def test_few_shot_renderer_returns_empty_when_all_placeholders() -> None:
    rendered = _render_few_shot_examples(
        [{"source": "x", "target": "[TODO: y]"}]
    )
    assert rendered == ""


def test_compose_translate_omits_examples_block_when_only_todos() -> None:
    profile = _profile(
        examples=[{"source": "x", "target": "[TODO: y]"}],
    )
    body = compose_translate_system_prompt(Category.NOVEL, profile)
    assert "EXAMPLES" not in body  # placeholder examples must not surface
    assert "[TODO" not in body


def test_prompt_keys_include_profile_when_set() -> None:
    profile = _profile(key="novel_literary")
    assert prompt_key_for_translate(Category.NOVEL, profile) == "translate__novel_literary"
    assert prompt_key_for_critique(Category.NOVEL, profile) == "critique__novel_literary"
    assert prompt_key_for_revise(Category.NOVEL, profile) == "revise__novel_literary"


def test_prompt_keys_carry_category_when_no_profile() -> None:
    assert prompt_key_for_translate(Category.NOVEL, None) == "translate__novel__legacy"
    assert prompt_key_for_critique(Category.EDUCATIONAL, None) == "critique__educational__legacy"


def test_compose_critique_includes_profile_register() -> None:
    profile = _profile(template="Register: literary.")
    body = compose_critique_system_prompt(Category.NOVEL, profile)
    # The critic must know the profile so it scores against the right register.
    assert "Register: literary." in body
    # The critic must NOT include few-shot translations — we don't want to
    # bias the *judge* with example translations.
    assert "EXAMPLES (illustrative" not in body


def test_compose_revise_includes_few_shot_examples_for_target_register() -> None:
    profile = _profile(
        examples=[{"source": "Real source.", "target": "Real target.", "note": ""}],
    )
    body = compose_revise_system_prompt(Category.NOVEL, profile)
    # Revise produces actual translations; few-shot examples DO help.
    assert "Real source." in body
    assert "Real target." in body


def test_parse_critique_response_extracts_scores_and_issues() -> None:
    raw = {
        "naturalness": 4.2,
        "fidelity": 4.5,
        "register_match": 4.0,
        "consistency": 4.8,
        "overall": 4.3,
        "issues": [
            {
                "block_id": "p1b0",
                "severity": "medium",
                "kind": "naturalness",
                "snippet": "stilted phrase",
                "suggestion": "use idiomatic equivalent",
            },
            {
                "block_id": "p2b3",
                "severity": "high",
                "kind": "fidelity",
                "snippet": "missing sentence",
                "suggestion": "translate the elided clause",
            },
        ],
    }
    result = _parse_critique_response(raw, total_tokens=42, prompt_tokens=30, output_tokens=12)
    assert result.naturalness == 4.2
    assert result.overall == 4.3
    assert len(result.issues) == 2
    assert result.has_high_severity() is True
    assert result.total_tokens == 42 and result.prompt_tokens == 30


def test_parse_critique_response_clamps_out_of_range() -> None:
    raw = {
        "naturalness": -1.0,
        "fidelity": 7.0,
        "register_match": 4.0,
        "consistency": 4.0,
        "overall": 4.0,
    }
    result = _parse_critique_response(raw)
    assert result.naturalness == 1.0
    assert result.fidelity == 5.0


def test_parse_critique_response_drops_malformed_issues() -> None:
    raw = {
        "naturalness": 4.0,
        "fidelity": 4.0,
        "register_match": 4.0,
        "consistency": 4.0,
        "overall": 4.0,
        "issues": [
            {"block_id": "p1b0", "severity": "medium", "kind": "naturalness"},
            {"block_id": "p1b1", "severity": "totally-invalid", "kind": "x"},
            "not even a dict",
            {"severity": "high", "kind": "fidelity"},  # missing block_id
        ],
    }
    result = _parse_critique_response(raw)
    assert len(result.issues) == 1
    assert result.issues[0].block_id == "p1b0"


def test_parse_critique_response_raises_on_missing_score() -> None:
    raw = {"naturalness": 4.0}  # missing the rest
    with pytest.raises(TranslationError):
        _parse_critique_response(raw)

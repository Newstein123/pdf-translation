"""Verify voice card + book brief blocks slot into the four system prompts."""
from __future__ import annotations

from app.db.models import Category, StyleProfile
from app.services.translation import (
    compose_critique_system_prompt,
    compose_judge_system_prompt,
    compose_revise_system_prompt,
    compose_translate_system_prompt,
)


def _profile() -> StyleProfile:
    return StyleProfile(
        key="test_voice",
        name="Test Voice",
        system_prompt_template="Translate with literary register.",
        vocabulary_preferences_json={"prefer": "ကျနော်"},
        few_shot_examples_json=[],
    )


def test_translate_prompt_embeds_voice_card_and_brief() -> None:
    out = compose_translate_system_prompt(
        Category.NOVEL,
        _profile(),
        voice_card_text="TRANSLATOR-VOICE TARGET:\nMatch Min Thu Wun's voice.",
        book_brief_text="BOOK BRIEF:\nFirst-person past tense.",
    )
    assert "TRANSLATOR-VOICE TARGET" in out
    assert "Min Thu Wun" in out
    assert "BOOK BRIEF" in out
    assert "First-person past tense" in out
    # Profile content must still come through.
    assert "literary register" in out


def test_translate_prompt_unchanged_when_no_extras_or_profile() -> None:
    base = compose_translate_system_prompt(Category.NOVEL, None)
    # No voice card / brief; should match the base file content.
    from pathlib import Path

    expected = (
        Path("app/prompts/translate_novel.txt").read_text(encoding="utf-8")
    )
    assert base.strip() == expected.strip()


def test_critique_prompt_embeds_voice_card() -> None:
    out = compose_critique_system_prompt(
        Category.NOVEL,
        _profile(),
        voice_card_text="VOICE TARGET\n",
    )
    assert "VOICE TARGET" in out
    assert "literary register" in out


def test_revise_prompt_embeds_voice_card() -> None:
    out = compose_revise_system_prompt(
        Category.NOVEL,
        _profile(),
        voice_card_text="VOICE TARGET\n",
    )
    assert "VOICE TARGET" in out
    assert "literary register" in out


def test_judge_prompt_embeds_voice_card_and_brief() -> None:
    out = compose_judge_system_prompt(
        Category.NOVEL,
        _profile(),
        voice_card_text="VOICE TARGET\n",
        book_brief_text="BRIEF\n",
    )
    assert "VOICE TARGET" in out
    assert "BRIEF" in out


def test_critique_prompt_works_without_profile_when_extras_supplied() -> None:
    out = compose_critique_system_prompt(
        Category.NOVEL,
        None,
        voice_card_text="VOICE TARGET\n",
    )
    assert "VOICE TARGET" in out

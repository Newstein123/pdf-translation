"""Resolver tests: how the worker picks a voice card per job."""
from __future__ import annotations

import os

import pytest
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Category, Job, JobStatus, VoiceCard
from app.services.voice_card import resolve_voice_card_text_for_job


@pytest.fixture
def session(db_engine) -> Session:
    with Session(bind=db_engine) as s:
        yield s
        s.rollback()


def _job(session, **kwargs) -> Job:
    j = Job(
        original_filename=kwargs.pop("original_filename", "t.pdf"),
        file_path=kwargs.pop("file_path", "/tmp/t.pdf"),
        file_size_bytes=kwargs.pop("file_size_bytes", 1),
        status=kwargs.pop("status", JobStatus.UPLOADED),
        category=kwargs.pop("category", Category.NOVEL),
        **kwargs,
    )
    session.add(j)
    session.flush()
    return j


def _card(session, **kwargs) -> VoiceCard:
    c = VoiceCard(
        translator=kwargs.pop("translator"),
        card_text=kwargs.pop("card_text", "Match this voice."),
        active=kwargs.pop("active", True),
        source_passage_count=kwargs.pop("source_passage_count", 100),
        **kwargs,
    )
    session.add(c)
    session.flush()
    return c


def test_resolver_uses_explicit_job_override(session) -> None:
    _card(session, translator="ShweOoDaung", card_text="Formal classical voice.")
    _card(session, translator="OtherTranslator", card_text="Other voice.")

    job = _job(session, voice_card_translator="ShweOoDaung")
    text, translator = resolve_voice_card_text_for_job(session, job.id)
    assert translator == "ShweOoDaung"
    assert "Formal classical voice" in text
    assert "TRANSLATOR-VOICE TARGET" in text


def test_resolver_falls_back_to_settings_default(session, monkeypatch) -> None:
    _card(session, translator="DefaultGuy", card_text="Default voice.")
    monkeypatch.setenv("DEFAULT_VOICE_CARD_TRANSLATOR", "DefaultGuy")
    get_settings.cache_clear()

    # Use UNKNOWN category so category-specific defaults don't intercept
    # the resolution before it reaches the global default.
    job = _job(session, category=Category.UNKNOWN)
    text, translator = resolve_voice_card_text_for_job(session, job.id)
    assert translator == "DefaultGuy"
    assert "Default voice" in text


def test_resolver_returns_blank_when_translator_has_no_card(session, monkeypatch) -> None:
    monkeypatch.setenv("DEFAULT_VOICE_CARD_TRANSLATOR", "NoSuchTranslator")
    get_settings.cache_clear()
    job = _job(session, category=Category.UNKNOWN)
    text, translator = resolve_voice_card_text_for_job(session, job.id)
    assert text == ""
    # Translator name still surfaces so the worker can log "tried but missing".
    assert translator == "NoSuchTranslator"


def test_resolver_returns_blank_when_default_disabled(session, monkeypatch) -> None:
    monkeypatch.setenv("DEFAULT_VOICE_CARD_TRANSLATOR", "")
    get_settings.cache_clear()
    job = _job(session, category=Category.UNKNOWN)
    text, translator = resolve_voice_card_text_for_job(session, job.id)
    assert text == ""
    assert translator is None


def test_resolver_inactive_card_is_skipped(session, monkeypatch) -> None:
    _card(session, translator="Inactive", card_text="...", active=False)
    monkeypatch.setenv("DEFAULT_VOICE_CARD_TRANSLATOR", "Inactive")
    get_settings.cache_clear()
    job = _job(session, category=Category.UNKNOWN)
    text, translator = resolve_voice_card_text_for_job(session, job.id)
    assert text == ""
    assert translator == "Inactive"


def test_resolver_handles_missing_job(session) -> None:
    text, translator = resolve_voice_card_text_for_job(session, "no-such-job-id")
    assert text == ""
    assert translator is None


def test_resolver_uses_category_default_for_personal_development(
    session, monkeypatch
) -> None:
    """Job with category=PERSONAL_DEVELOPMENT and no per-job override should
    pick up the category-specific default voice."""
    _card(session, translator="HpeMyint", card_text="Self-help conversational tone.")
    monkeypatch.setenv("VOICE_CARD_FOR_PERSONAL_DEVELOPMENT", "HpeMyint")
    monkeypatch.setenv("DEFAULT_VOICE_CARD_TRANSLATOR", "ShouldNotBeUsed")
    get_settings.cache_clear()

    job = _job(session, category=Category.PERSONAL_DEVELOPMENT)
    text, translator = resolve_voice_card_text_for_job(session, job.id)
    assert translator == "HpeMyint"
    assert "Self-help conversational tone" in text


def test_resolver_explicit_override_beats_category_default(
    session, monkeypatch
) -> None:
    _card(session, translator="HpeMyint", card_text="PD voice.")
    _card(session, translator="OverrideGuy", card_text="Override voice.")
    monkeypatch.setenv("VOICE_CARD_FOR_PERSONAL_DEVELOPMENT", "HpeMyint")
    get_settings.cache_clear()

    job = _job(
        session,
        category=Category.PERSONAL_DEVELOPMENT,
        voice_card_translator="OverrideGuy",
    )
    text, translator = resolve_voice_card_text_for_job(session, job.id)
    assert translator == "OverrideGuy"
    assert "Override voice" in text


def test_resolver_falls_through_when_category_default_empty(
    session, monkeypatch
) -> None:
    """Empty category default should not block the global default fallback."""
    _card(session, translator="GlobalGuy", card_text="Global default voice.")
    monkeypatch.setenv("VOICE_CARD_FOR_EDUCATIONAL", "")  # no PD-specific
    monkeypatch.setenv("DEFAULT_VOICE_CARD_TRANSLATOR", "GlobalGuy")
    get_settings.cache_clear()

    job = _job(session, category=Category.EDUCATIONAL)
    text, translator = resolve_voice_card_text_for_job(session, job.id)
    assert translator == "GlobalGuy"


def test_explicit_override_wins_over_default(session, monkeypatch) -> None:
    _card(session, translator="ExplicitGuy", card_text="Explicit voice.")
    _card(session, translator="DefaultGuy", card_text="Default voice.")
    monkeypatch.setenv("DEFAULT_VOICE_CARD_TRANSLATOR", "DefaultGuy")
    get_settings.cache_clear()

    job = _job(session, voice_card_translator="ExplicitGuy")
    text, translator = resolve_voice_card_text_for_job(session, job.id)
    assert translator == "ExplicitGuy"
    assert "Explicit voice" in text
    assert "Default voice" not in text

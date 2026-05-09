"""Unit tests for the voice card service (M12 / Track B)."""
from __future__ import annotations

import pytest

from app.db.models import ReferenceMode, ReferencePassage, VoiceCard
from app.services import voice_card


@pytest.fixture
def session(db_engine):
    from sqlalchemy.orm import Session

    with Session(bind=db_engine) as s:
        yield s
        s.rollback()


def _seed(session, **kwargs) -> ReferencePassage:
    p = ReferencePassage(
        source_text="",
        target_text=kwargs.pop("target_text"),
        source_lang=kwargs.pop("source_lang", "en"),
        target_lang=kwargs.pop("target_lang", "my"),
        mode=kwargs.pop("mode", ReferenceMode.NARRATION),
        translator=kwargs.pop("translator", "TestTranslator"),
        book_title=kwargs.pop("book_title", "TestBook"),
        active=kwargs.pop("active", True),
        **kwargs,
    )
    session.add(p)
    session.flush()
    return p


# ---------------------------------------------------------------------------
# compute_baselines
# ---------------------------------------------------------------------------


def test_compute_baselines_handles_empty() -> None:
    b = voice_card.compute_baselines([])
    assert b.passage_count == 0
    assert b.char_count == 0
    assert b.avg_sentence_chars == 0.0
    assert b.dialogue_ratio == 0.0
    assert b.particle_rates_per_1000 == {}


def test_compute_baselines_counts_dialogue_ratio(session) -> None:
    a = _seed(session, target_text="ဇာတ်လမ်း သည်", mode=ReferenceMode.DIALOGUE)
    b = _seed(session, target_text="အခြား စာ", mode=ReferenceMode.NARRATION)
    c = _seed(session, target_text="နောက် ပိုဒ်", mode=ReferenceMode.DIALOGUE)
    baselines = voice_card.compute_baselines([a, b, c])
    assert baselines.passage_count == 3
    assert baselines.dialogue_ratio == pytest.approx(2 / 3, abs=1e-3)


def test_compute_baselines_particle_rates(session) -> None:
    # 100-char passage with 10 occurrences of "က" → 100 per 1000.
    text = "က" * 10 + "x" * 90
    p = _seed(session, target_text=text)
    baselines = voice_card.compute_baselines([p])
    assert baselines.particle_rates_per_1000["က"] == pytest.approx(100.0, abs=0.1)


def test_compute_baselines_avg_sentence_chars_uses_burmese_enders(session) -> None:
    # 3 sentences × 10 chars apiece, all ending in ။
    text = "abcdefghi။ jklmnopqr။ stuvwxyz။ "
    p = _seed(session, target_text=text)
    baselines = voice_card.compute_baselines([p])
    # avg ≈ char_count / 3
    assert baselines.avg_sentence_chars == pytest.approx(len(text) / 3, abs=1.0)


# ---------------------------------------------------------------------------
# compose_voice_card_block
# ---------------------------------------------------------------------------


def test_compose_voice_card_block_returns_empty_for_none() -> None:
    assert voice_card.compose_voice_card_block(None) == ""


def test_compose_voice_card_block_returns_empty_for_blank_text() -> None:
    card = VoiceCard(translator="X", card_text="")
    assert voice_card.compose_voice_card_block(card) == ""


def test_compose_voice_card_block_renders_target_text() -> None:
    card = VoiceCard(translator="Min Thu Wun", card_text="Use long literary sentences.")
    block = voice_card.compose_voice_card_block(card)
    assert "TRANSLATOR-VOICE TARGET" in block
    assert "Use long literary sentences." in block


# ---------------------------------------------------------------------------
# generate_voice_card error paths (Gemini stubbed)
# ---------------------------------------------------------------------------


def test_generate_voice_card_raises_when_no_passages(session) -> None:
    with pytest.raises(voice_card.VoiceCardError):
        voice_card.generate_voice_card(session, "NoSuchTranslator")


def test_generate_voice_card_returns_existing_active_when_not_forced(
    session, monkeypatch
) -> None:
    # Seed an existing active card.
    existing = VoiceCard(
        translator="Existing",
        card_text="An existing card.",
        source_passage_count=10,
        active=True,
    )
    session.add(existing)
    session.flush()

    # Even with no passages, idempotent path returns the existing card.
    result = voice_card.generate_voice_card(session, "Existing")
    assert result.id == existing.id


def test_generate_voice_card_with_force_calls_gemini(session, monkeypatch) -> None:
    # Seed enough passages so the analyzer doesn't trip on coverage.
    for i in range(12):
        _seed(
            session,
            target_text=f"ကောင်းမွန်သော စာသည် ဤသို့တည်ရှိ၏။ {i}",
            translator="Forced",
            mode=ReferenceMode.NARRATION,
        )

    fake_payload = {
        "summary": "Sparse, lyric prose.",
        "card_text": "Match the voice of Forced: short clauses, formal endings.",
        "register": "literary",
        "sentence_structure": "short, formal",
        "best_genre_match": ["classical", "lyrical"],
    }

    class _FakeResp:
        payload = fake_payload
        total_tokens = 1000
        prompt_tokens = 700
        output_tokens = 300
        cached_input_tokens = 0

    def _fake_generate_json(**_kwargs):
        return _FakeResp()

    monkeypatch.setattr(voice_card, "generate_json", _fake_generate_json)

    card = voice_card.generate_voice_card(session, "Forced", force=True)
    assert card.translator == "Forced"
    assert "short clauses" in card.card_text
    assert card.source_passage_count == 12
    assert card.genre_tags_json == ["classical", "lyrical"]
    assert (card.baselines_json or {}).get("passage_count") == 12


# ---------------------------------------------------------------------------
# Stratified sample
# ---------------------------------------------------------------------------


def test_stratified_sample_balances_modes(session) -> None:
    passages = []
    for i in range(5):
        passages.append(
            _seed(
                session,
                target_text="ပထမပိုဒ် " * 10 + str(i),
                mode=ReferenceMode.NARRATION,
                translator="Mixed",
            )
        )
        passages.append(
            _seed(
                session,
                target_text="“ဟောက်ခေါက်!”ဟု ပြောသည်။ " * 5 + str(i),
                mode=ReferenceMode.DIALOGUE,
                translator="Mixed",
            )
        )
    sample = voice_card._build_stratified_sample(passages, target_tokens=2000)
    assert "[mode: NARRATION]" in sample
    assert "[mode: DIALOGUE]" in sample

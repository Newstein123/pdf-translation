"""Voice cards: per-translator distilled style briefs.

A voice card is the output of an offline analysis pass over a translator's
corpus of published Burmese translations. Each card is a ~200-400 token
prose block describing the translator's voice (sentence rhythm, register,
dialogue style, particle preferences, vocabulary patterns) plus structured
fields that the linter and the Book Brief consume.

Cards are generated *once* per translator. At translation time the card is
spliced into the cached system-prompt prefix, so per-chunk cost is
near-zero — that's the point: voice anchoring without paying retrieval
tokens on every call.

Public surface:

  * ``generate_voice_card(session, translator)`` — build a card from the
    translator's existing ``ReferencePassage`` rows and persist it.
  * ``compute_baselines(passages)`` — pure function that produces the
    empirical numbers (avg sentence length, dialogue ratio, particle
    density) the linter uses as comparison baselines.
  * ``compose_voice_card_block(card)`` — render a card as a string ready
    to inject into a system prompt (returns "" when card is None).

The actual generation prompt lives in ``app/prompts/voice_card.txt``.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import ReferenceMode, ReferencePassage, VoiceCard
from app.services.gemini import GeminiUnavailable, generate_json
from app.utils.tokens import count_tokens

log = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "voice_card.txt"

# We sample up to this many tokens of corpus per card. Voice patterns repeat
# fast — beyond ~25k tokens we get diminishing returns and the prompt gets
# slow. Stratified to cover narration, dialogue, and exposition modes.
_DEFAULT_CARD_SAMPLE_TARGET_TOKENS = 25_000

# Minimum corpus size below which a card is suspect. We still allow it (with
# a logged warning) so a translator with sparse coverage doesn't block
# pipeline work — the card just gets a smaller evidence base.
_MIN_PASSAGES_FOR_CARD = 10


_VOICE_CARD_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "sentence_structure": {"type": "string"},
        "register": {"type": "string"},
        "dialogue_style": {"type": "string"},
        "particle_preferences": {"type": "array", "items": {"type": "string"}},
        "vocabulary_patterns": {"type": "array", "items": {"type": "string"}},
        "honorific_usage": {"type": "string"},
        "punctuation_and_layout": {"type": "string"},
        "card_text": {"type": "string"},
        "do_not_imitate": {"type": "array", "items": {"type": "string"}},
        "best_genre_match": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "card_text", "register", "sentence_structure"],
}


# Burmese script range for empirical baselines.
_BURMESE_RE = re.compile(r"[က-႟]")
# Common Burmese sentence enders. Translators vary; this list is conservative.
_SENTENCE_END_RE = re.compile(r"[။၏!?]")
# Particles we count for density baselines. These are overrepresented in
# Burmese prose generally, so absolute counts vary widely; we report rate
# (occurrences per 1000 chars) for cross-translator comparison.
_BASELINE_PARTICLES = ("က", "ကို", "မှာ", "ဖို့", "ရဲ့", "သည်", "၏", "တယ်", "မယ်")


class VoiceCardError(RuntimeError):
    pass


@dataclass(frozen=True)
class CorpusBaselines:
    passage_count: int
    char_count: int
    avg_sentence_chars: float
    dialogue_ratio: float  # fraction of passages tagged DIALOGUE
    particle_rates_per_1000: dict[str, float]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_voice_card(
    session: Session,
    translator: str,
    *,
    sample_target_tokens: int = _DEFAULT_CARD_SAMPLE_TARGET_TOKENS,
    force: bool = False,
    work_type: str | None = None,
) -> VoiceCard:
    """Build (or rebuild) the voice card for ``translator``.

    Reads the translator's active reference passages, builds a stratified
    sample (preferring a mix of narration / dialogue / exposition), calls
    Gemini to produce the structured card, computes empirical baselines,
    persists the result.

    Idempotent: returns the existing card if one is active and ``force`` is
    False. Raises ``VoiceCardError`` when no usable corpus exists or the
    Gemini call fails.
    """
    settings = get_settings()

    if not force:
        existing = (
            session.query(VoiceCard)
            .filter(VoiceCard.translator == translator, VoiceCard.active.is_(True))
            .one_or_none()
        )
        if existing is not None:
            return existing

    passages = list(
        session.execute(
            select(ReferencePassage)
            .where(
                ReferencePassage.translator == translator,
                ReferencePassage.active.is_(True),
            )
            .order_by(ReferencePassage.added_at)
        ).scalars()
    )
    if not passages:
        raise VoiceCardError(
            f"no active reference passages for translator {translator!r}"
        )
    if len(passages) < _MIN_PASSAGES_FOR_CARD:
        log.warning(
            "voice card for %r built from only %d passages — coverage is thin",
            translator,
            len(passages),
        )

    sample_text = _build_stratified_sample(passages, sample_target_tokens)
    if not sample_text:
        raise VoiceCardError(
            f"could not build sample for translator {translator!r}"
        )

    prompt_text = _PROMPT_PATH.read_text(encoding="utf-8")

    # Pass work_type so the prompt knows whether this is a native-Burmese
    # author (ORIGINAL) or a translator (TRANSLATION). The framing of the
    # voice card differs: original-author cards describe authentic Burmese
    # voice patterns; translator cards describe how the translator
    # negotiated foreign source into Burmese.
    work_type_hint = (
        f"Author work type: {work_type}\n"
        if work_type in {"ORIGINAL", "TRANSLATION"}
        else ""
    )
    user_payload = (
        f"Translator: {translator}\n"
        f"{work_type_hint}"
        f"Corpus passage count: {len(passages)}\n"
        f"=== SAMPLE ===\n{sample_text}\n=== END SAMPLE ==="
    )
    model = settings.gemini_utility_model

    try:
        resp = generate_json(
            model=model,
            system_instruction=prompt_text,
            user_text=user_payload,
            response_schema=_VOICE_CARD_RESPONSE_SCHEMA,
            temperature=0.1,
        )
    except GeminiUnavailable as e:
        raise VoiceCardError(f"gemini unavailable: {e}") from e

    payload = resp.payload
    baselines = compute_baselines(passages)

    existing = (
        session.query(VoiceCard).filter(VoiceCard.translator == translator).one_or_none()
    )
    if existing is None:
        existing = VoiceCard(translator=translator)
        session.add(existing)

    existing.target_lang = "my"
    existing.card_text = (payload.get("card_text") or "").strip() or _fallback_card_text(payload)
    existing.genre_tags_json = payload.get("best_genre_match") or []
    existing.baselines_json = _baselines_to_json(baselines)
    existing.source_passage_count = len(passages)
    existing.model = model
    existing.active = True
    existing.notes = json.dumps(
        {k: v for k, v in payload.items() if k != "card_text"},
        ensure_ascii=False,
    )
    existing.updated_at = datetime.now(timezone.utc)
    session.flush()
    return existing


def compute_baselines(passages: Iterable[ReferencePassage]) -> CorpusBaselines:
    """Empirical numbers for the linter to compare against.

    Pure function so the linter can call it on a hot path during a test
    without DB I/O. Counts are deterministic given the same inputs.
    """
    passages = list(passages)
    char_count = 0
    sentences = 0
    dialogue_count = 0
    particle_counts: dict[str, int] = {p: 0 for p in _BASELINE_PARTICLES}

    for p in passages:
        text = p.target_text or ""
        char_count += len(text)
        sentences += len(_SENTENCE_END_RE.findall(text)) or 1
        if p.mode == ReferenceMode.DIALOGUE:
            dialogue_count += 1
        for particle in _BASELINE_PARTICLES:
            particle_counts[particle] += text.count(particle)

    if not passages or char_count == 0:
        return CorpusBaselines(
            passage_count=len(passages),
            char_count=0,
            avg_sentence_chars=0.0,
            dialogue_ratio=0.0,
            particle_rates_per_1000={},
        )

    return CorpusBaselines(
        passage_count=len(passages),
        char_count=char_count,
        avg_sentence_chars=char_count / max(1, sentences),
        dialogue_ratio=dialogue_count / len(passages),
        particle_rates_per_1000={
            p: round(c / char_count * 1000, 3) for p, c in particle_counts.items()
        },
    )


def compose_voice_card_block(card: VoiceCard | None) -> str:
    """Render a voice card as a system-prompt block.

    Returns ``""`` when the card is None so callers can blindly concatenate.
    """
    if card is None or not card.card_text:
        return ""
    return (
        f"TRANSLATOR-VOICE TARGET:\n"
        f"{card.card_text.strip()}\n"
    )


def get_active_card(session: Session, translator: str) -> VoiceCard | None:
    """Return the active voice card for ``translator``, or None."""
    return (
        session.query(VoiceCard)
        .filter(VoiceCard.translator == translator, VoiceCard.active.is_(True))
        .one_or_none()
    )


def resolve_voice_card_text_for_job(
    session: Session, job_id: str
) -> tuple[str, str | None]:
    """Resolve the voice-card text and translator name for a translation job.

    Resolution order:
      1. ``Job.voice_card_translator`` — explicit per-job override.
      2. Per-category default (``voice_card_for_novel`` / ``..._educational`` /
         ``..._personal_development`` settings) when the job's category
         matches one of them.
      3. ``settings.default_voice_card_translator`` — global default.
      4. Empty string ("disabled") — no voice card injected.

    Returns ``(card_text, translator_name)``. ``card_text`` is "" when
    voice cards are disabled or the resolved translator has no active card
    (e.g. card not yet generated, or marked inactive). ``translator_name``
    is the resolved name (or None if disabled / unresolved) — surfaced for
    logging and event records.

    Failures are non-fatal: any unexpected error returns ``("", None)``
    so the translation pipeline never breaks because of a missing card."""
    from app.config import get_settings
    from app.db.models import Category, Job

    try:
        job = session.get(Job, job_id)
        if job is None:
            return "", None
        settings = get_settings()
        explicit = (job.voice_card_translator or "").strip()

        # Category-specific default — applied only when the job has no
        # explicit per-job override.
        category_default = ""
        if not explicit:
            if job.category == Category.NOVEL:
                category_default = (settings.voice_card_for_novel or "").strip()
            elif job.category == Category.EDUCATIONAL:
                category_default = (settings.voice_card_for_educational or "").strip()
            elif job.category == Category.PERSONAL_DEVELOPMENT:
                category_default = (
                    settings.voice_card_for_personal_development or ""
                ).strip()

        translator = (
            explicit
            or category_default
            or (settings.default_voice_card_translator or "").strip()
        )
        if not translator:
            return "", None
        card = get_active_card(session, translator)
        if card is None:
            log.info(
                "voice card requested for translator %r but no active card found",
                translator,
            )
            return "", translator
        return compose_voice_card_block(card), translator
    except Exception:
        log.exception("voice-card resolution failed for job %s", job_id)
        return "", None


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _build_stratified_sample(
    passages: list[ReferencePassage], target_tokens: int
) -> str:
    """Build a sample favoring even coverage across modes.

    Strategy:
      1. Bucket passages by mode.
      2. Distribute the budget evenly across populated buckets.
      3. Within each bucket, take passages in insertion order until the
         per-bucket budget is hit.
    """
    by_mode: dict[ReferenceMode, list[ReferencePassage]] = {}
    for p in passages:
        by_mode.setdefault(p.mode, []).append(p)

    populated_modes = [m for m, ps in by_mode.items() if ps]
    if not populated_modes:
        return ""
    per_bucket = target_tokens // len(populated_modes)

    chunks: list[str] = []
    for mode in populated_modes:
        bucket_text: list[str] = []
        used = 0
        for p in by_mode[mode]:
            text = (p.target_text or "").strip()
            if not text:
                continue
            t = count_tokens(text)
            if used + t > per_bucket:
                # Truncate the last passage to fit the bucket cleanly.
                remaining = max(0, per_bucket - used)
                if remaining > 0:
                    approx_chars = max(1, remaining * 4)
                    bucket_text.append(text[:approx_chars])
                break
            bucket_text.append(text)
            used += t
        if bucket_text:
            chunks.append(f"[mode: {mode.value}]\n" + "\n\n".join(bucket_text))

    return "\n\n=========\n\n".join(chunks)


def _fallback_card_text(payload: dict[str, Any]) -> str:
    """Build a plain-text card by stitching the structured fields when the
    model failed to emit a free-form ``card_text``. Better than empty."""
    parts: list[str] = []
    summary = (payload.get("summary") or "").strip()
    if summary:
        parts.append(summary)
    for key in (
        "sentence_structure",
        "register",
        "dialogue_style",
        "honorific_usage",
        "punctuation_and_layout",
    ):
        v = (payload.get(key) or "").strip()
        if v:
            parts.append(v)
    return " ".join(parts)


def _baselines_to_json(b: CorpusBaselines) -> dict[str, Any]:
    return {
        "passage_count": b.passage_count,
        "char_count": b.char_count,
        "avg_sentence_chars": round(b.avg_sentence_chars, 2),
        "dialogue_ratio": round(b.dialogue_ratio, 3),
        "particle_rates_per_1000": b.particle_rates_per_1000,
    }

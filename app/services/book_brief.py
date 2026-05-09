"""Whole-book pre-analysis (Book Brief).

Runs once per job, immediately after extraction and before chunking. Reads a
stratified sample of the book (first chapter + last chapter + a handful of
mid-book spans) and asks Gemini to emit a structured "translator's brief":
POV, tense, register, narrator voice, key characters, themes, dialogue and
idiom density, cultural / honorific guidance, do-not-translate list,
translation pitfalls.

The brief becomes a system-prompt block injected into every subsequent
TRANSLATE / CRITIQUE / REVISE / JUDGE pass — so end-to-end coherence comes
from a single shared frame rather than each chunk re-deriving the book's
voice from a 1500-token window.

This module is a Track-B foundation skeleton:

  * ``analyze_book(session, job_id)`` — orchestrates: build sample → call
    Gemini → persist BookBrief row.
  * ``compose_book_brief_block(brief)`` — renders the brief as a string
    block ready to splice into a system prompt.
  * ``sample_for_brief(structure_path, *, target_tokens)`` — builds the
    stratified sample from ``structure.json`` (deterministic so the same
    book yields the same sample).

The actual prompt template lives in ``app/prompts/book_brief.txt``. The
response schema mirrors the prompt's JSON contract exactly so we can reject
malformed responses early.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import BookBrief, BriefStatus, Job
from app.services.gemini import GeminiUnavailable, generate_json
from app.utils.tokens import count_tokens

log = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "book_brief.txt"

# Default sample budget. Big enough to cover first/last chapter + middle
# spans on a typical novel, small enough to fit a Flash 1M call cleanly.
_DEFAULT_SAMPLE_TARGET_TOKENS = 30_000
# How many evenly-spaced middle samples to take on top of first + last.
_MIDDLE_SAMPLE_COUNT = 4


_BRIEF_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": ["string", "null"]},
        "category_hint": {"type": "string"},
        "pov": {"type": "string"},
        "tense": {"type": "string"},
        "register": {"type": "string"},
        "register_notes": {"type": "string"},
        "narrator_voice": {"type": "string"},
        "key_characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "voice_notes": {"type": "string"},
                },
                "required": ["name"],
            },
        },
        "themes": {"type": "array", "items": {"type": "string"}},
        "setting": {"type": "string"},
        "dialogue_density": {"type": "string"},
        "idiom_density": {"type": "string"},
        "cultural_notes": {"type": "string"},
        "honorific_guidance": {"type": "string"},
        "do_not_translate": {"type": "array", "items": {"type": "string"}},
        "translation_pitfalls": {"type": "array", "items": {"type": "string"}},
        "target_reader": {"type": "string"},
    },
    "required": [
        "category_hint",
        "pov",
        "tense",
        "register",
        "register_notes",
        "narrator_voice",
        "dialogue_density",
        "idiom_density",
    ],
}


class BookBriefError(RuntimeError):
    pass


@dataclass(frozen=True)
class BriefSample:
    """A stratified text sample fed to the brief analyzer.

    ``label`` documents which slice of the book this came from
    (``"opening"``, ``"middle:1/4"``, ``"closing"``). The label is included
    in the prompt so the analyzer can weight the opening/closing more
    heavily for voice without us hard-coding that logic."""

    label: str
    text: str
    token_count: int


def sample_for_brief(
    structure_path: Path,
    *,
    target_tokens: int = _DEFAULT_SAMPLE_TARGET_TOKENS,
    middle_sample_count: int = _MIDDLE_SAMPLE_COUNT,
) -> list[BriefSample]:
    """Read ``structure.json`` and return a stratified sample.

    Strategy:
      1. Concatenate text per page in document order.
      2. The opening sample = first ~25% of the budget, drawn from the start.
      3. The closing sample = next ~25% of the budget, drawn from the end.
      4. The middle samples = remaining ~50% split evenly across N spans.

    Returns an empty list if the structure file is missing or unreadable —
    the caller should treat that as "no brief possible" and skip the stage.
    """
    if not structure_path.exists():
        return []
    try:
        with structure_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        log.exception("failed to read %s for brief sample", structure_path)
        return []

    pages = data.get("pages") or []
    if not pages:
        return []

    page_texts: list[str] = []
    for page in pages:
        parts: list[str] = []
        for block in page.get("blocks") or []:
            text = (block.get("text") or "").strip()
            if text:
                parts.append(text)
        page_texts.append("\n\n".join(parts))

    full_text = "\n\n".join(t for t in page_texts if t)
    if not full_text:
        return []

    total_tokens = count_tokens(full_text)
    if total_tokens <= target_tokens:
        # Whole book fits — one sample, labeled "full".
        return [BriefSample(label="full", text=full_text, token_count=total_tokens)]

    opening_budget = target_tokens // 4
    closing_budget = target_tokens // 4
    middle_budget = max(0, target_tokens - opening_budget - closing_budget)
    per_middle = middle_budget // max(1, middle_sample_count)

    samples: list[BriefSample] = []

    opening = _take_first_n_tokens(full_text, opening_budget)
    if opening:
        samples.append(
            BriefSample(
                label="opening", text=opening, token_count=count_tokens(opening)
            )
        )

    if per_middle > 0 and middle_sample_count > 0:
        # Even spans across the middle 60% so we don't overlap with opening/closing.
        middle_start_pos = len(full_text) // 5  # skip first 20%
        middle_end_pos = (len(full_text) * 4) // 5  # stop at 80%
        span_total = middle_end_pos - middle_start_pos
        if span_total > 0:
            stride = span_total // middle_sample_count
            for i in range(middle_sample_count):
                center = middle_start_pos + stride * i + stride // 2
                excerpt = _take_around_position(full_text, center, per_middle)
                if excerpt:
                    samples.append(
                        BriefSample(
                            label=f"middle:{i + 1}/{middle_sample_count}",
                            text=excerpt,
                            token_count=count_tokens(excerpt),
                        )
                    )

    closing = _take_last_n_tokens(full_text, closing_budget)
    if closing:
        samples.append(
            BriefSample(
                label="closing", text=closing, token_count=count_tokens(closing)
            )
        )

    return samples


def compose_user_payload(samples: list[BriefSample]) -> str:
    """Render samples for the brief analyzer.

    We use plain text with delimiters rather than JSON because the analyzer
    is reading prose: large strings escape ugly inside JSON, and delimiters
    are easier to skim during prompt audits."""
    parts = []
    for s in samples:
        parts.append(f"=== SAMPLE: {s.label} ({s.token_count} tokens) ===")
        parts.append(s.text.strip())
    return "\n\n".join(parts)


def analyze_book(session: Session, job_id: str) -> BookBrief:
    """Generate (or regenerate) the BookBrief for a job.

    Idempotent in spirit: if a brief already exists and is READY, returns it
    without re-calling Gemini. Pass ``force=True`` (TODO: add when wired
    into a re-analyze admin endpoint) to override.

    Raises ``BookBriefError`` on any failure that should halt the job before
    chunking starts. The caller (extract_task downstream) decides whether
    a missing brief is fatal or just a quality regression worth proceeding
    without.
    """
    settings = get_settings()
    job = session.get(Job, job_id)
    if job is None:
        raise BookBriefError(f"unknown job {job_id}")

    existing = (
        session.query(BookBrief).filter(BookBrief.job_id == job_id).one_or_none()
    )
    if existing is not None and existing.status == BriefStatus.READY:
        return existing

    structure_path = settings.intermediate_dir / job_id / "structure.json"
    samples = sample_for_brief(structure_path)
    if not samples:
        raise BookBriefError(
            f"no extractable text for job {job_id} — cannot build brief"
        )

    sample_total = sum(s.token_count for s in samples)
    prompt_text = _PROMPT_PATH.read_text(encoding="utf-8")
    user_payload = compose_user_payload(samples)
    model = settings.gemini_utility_model

    if existing is None:
        existing = BookBrief(
            job_id=job_id,
            status=BriefStatus.PENDING,
            sample_token_count=sample_total,
            model=model,
        )
        session.add(existing)
        session.flush()

    try:
        resp = generate_json(
            model=model,
            system_instruction=prompt_text,
            user_text=user_payload,
            response_schema=_BRIEF_RESPONSE_SCHEMA,
            temperature=0.1,
        )
    except GeminiUnavailable as e:
        existing.status = BriefStatus.FAILED
        existing.error_message = str(e)[:1000]
        existing.updated_at = datetime.now(timezone.utc)
        session.flush()
        raise BookBriefError(f"gemini unavailable: {e}") from e

    existing.brief_json = resp.payload
    existing.status = BriefStatus.READY
    existing.error_message = None
    existing.sample_token_count = sample_total
    existing.model = model
    existing.updated_at = datetime.now(timezone.utc)
    session.flush()
    return existing


def compose_book_brief_block(brief: BookBrief | dict[str, Any] | None) -> str:
    """Render the brief as a system-prompt block.

    Returns an empty string when no brief is available so callers can blindly
    concatenate the result. Skips empty fields so the prompt isn't padded
    with "POV: ".
    """
    if brief is None:
        return ""
    payload: dict[str, Any] | None
    if isinstance(brief, BookBrief):
        payload = brief.brief_json or None
    else:
        payload = brief
    if not payload:
        return ""

    lines: list[str] = ["BOOK BRIEF (use as authoritative context for translation):"]

    def _add(label: str, key: str) -> None:
        v = payload.get(key)
        if isinstance(v, str) and v.strip():
            lines.append(f"  {label}: {v.strip()}")

    _add("Title", "title")
    _add("POV", "pov")
    _add("Tense", "tense")
    _add("Register", "register")
    _add("Register notes", "register_notes")
    _add("Narrator voice", "narrator_voice")
    _add("Setting", "setting")
    _add("Dialogue density", "dialogue_density")
    _add("Idiom density", "idiom_density")
    _add("Cultural notes", "cultural_notes")
    _add("Honorific guidance", "honorific_guidance")
    _add("Target reader", "target_reader")

    chars = payload.get("key_characters") or []
    if isinstance(chars, list) and chars:
        lines.append("  Key characters:")
        for ch in chars[:8]:
            if not isinstance(ch, dict):
                continue
            name = (ch.get("name") or "").strip()
            if not name:
                continue
            role = (ch.get("role") or "").strip()
            voice = (ch.get("voice_notes") or "").strip()
            tail = " — ".join(p for p in (role, voice) if p)
            lines.append(f"    - {name}{' (' + tail + ')' if tail else ''}")

    themes = payload.get("themes") or []
    if isinstance(themes, list) and themes:
        lines.append("  Themes: " + ", ".join(str(t) for t in themes[:5] if t))

    dnt = payload.get("do_not_translate") or []
    if isinstance(dnt, list) and dnt:
        lines.append("  Do not translate: " + ", ".join(str(t) for t in dnt if t))

    pitfalls = payload.get("translation_pitfalls") or []
    if isinstance(pitfalls, list) and pitfalls:
        lines.append("  Pitfalls to avoid:")
        for p in pitfalls[:5]:
            if isinstance(p, str) and p.strip():
                lines.append(f"    - {p.strip()}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _take_first_n_tokens(text: str, target_tokens: int) -> str:
    """Greedy character-window approximation. Token counters are slow; we
    over-shoot in chars then trim to the token budget. Good enough for a
    coarse sample — exact alignment isn't required."""
    if target_tokens <= 0 or not text:
        return ""
    # Heuristic: 1 token ≈ 4 chars for English prose. Take 5x as a starting
    # window then trim down by token measurement.
    approx_chars = target_tokens * 5
    head = text[:approx_chars]
    return _trim_to_token_budget(head, target_tokens)


def _take_last_n_tokens(text: str, target_tokens: int) -> str:
    if target_tokens <= 0 or not text:
        return ""
    approx_chars = target_tokens * 5
    tail = text[-approx_chars:] if len(text) > approx_chars else text
    # Trim leading partial sentence if we cut mid-word.
    nl = tail.find("\n\n")
    if 0 < nl < len(tail) // 4:
        tail = tail[nl + 2 :]
    return _trim_to_token_budget(tail, target_tokens, from_end=True)


def _take_around_position(text: str, center: int, target_tokens: int) -> str:
    if target_tokens <= 0 or not text:
        return ""
    approx_chars = target_tokens * 5
    half = approx_chars // 2
    start = max(0, center - half)
    end = min(len(text), center + half)
    excerpt = text[start:end]
    return _trim_to_token_budget(excerpt, target_tokens)


def _trim_to_token_budget(text: str, target_tokens: int, *, from_end: bool = False) -> str:
    """Shrink text until it fits the token budget. Prefer paragraph boundaries
    so we don't ship dangling half-sentences to the analyzer.

    When the text has no useful paragraph boundary (one giant blob), fall
    back to a character-budget truncation rather than returning empty —
    a coarse cut is better than dropping the sample entirely."""
    if not text:
        return ""
    if count_tokens(text) <= target_tokens:
        return text
    paragraphs = text.split("\n\n")
    while len(paragraphs) > 1 and count_tokens("\n\n".join(paragraphs)) > target_tokens:
        if from_end:
            paragraphs.pop(0)
        else:
            paragraphs.pop()
    joined = "\n\n".join(paragraphs)
    if count_tokens(joined) <= target_tokens or not joined:
        return joined
    # Single huge paragraph that still overflows. Approximate with a 4-chars-
    # per-token slice; we deliberately undershoot a touch so we don't end up
    # one token over the budget.
    approx_chars = max(1, target_tokens * 4 - 50)
    if from_end:
        return joined[-approx_chars:]
    return joined[:approx_chars]

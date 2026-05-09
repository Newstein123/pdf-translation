"""Per-chunk translation via Gemini.

Responsibilities:
- Compose the system prompt from the base rules + the job's style profile
  (goals override + few-shot examples + vocabulary preferences).
- Inject glossary, name registry, and rolling context summary.
- Call Gemini with a structured JSON response contract.
- Parse + validate the response shape; surface errors as ``TranslationError``.

This module is deliberately stateless. The caller (translate_task) owns the
order of chunk execution, name-registry persistence, prompt-version
resolution, pass-row recording, and retries.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.db.models import Category, GlossaryEntry, NameRegistry, StyleProfile
from app.services.gemini import (
    GeminiAccessDenied,
    GeminiModelMissing,
    GeminiUnavailable,
    generate_json,
)
from app.services.running_context import serialize_for_prompt

log = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).parent.parent / "prompts"


class TranslationError(RuntimeError):
    pass


class TranslationModelMissing(TranslationError):
    """Surface ``GeminiModelMissing`` to the worker so it can fail the whole job."""


class TranslationAccessDenied(TranslationError):
    """Surface ``GeminiAccessDenied`` (403) to the worker so it can fail the whole
    job. Same fast-fail rationale as ``TranslationModelMissing``: the auth
    error affects every chunk, retrying just delays the inevitable."""


@dataclass(frozen=True)
class TranslatedBlock:
    block_id: str
    translated: str


@dataclass(frozen=True)
class DiscoveredName:
    source: str
    burmese: str


@dataclass(frozen=True)
class ChunkTranslationResult:
    blocks: list[TranslatedBlock]
    new_names: list[DiscoveredName]
    summary_for_next_chunk: str
    total_tokens: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    # M11: tokens billed at the cached input rate (subset of prompt_tokens).
    cached_input_tokens: int = 0
    # M8: structured running-context update emitted by the model. Empty dict
    # when the model didn't include the field (older prompts, errors, etc.) —
    # callers should fall back to ``summary_for_next_chunk`` in that case.
    running_context_update: dict[str, Any] | None = None


@dataclass(frozen=True)
class CritiqueIssue:
    block_id: str
    severity: str  # "low" | "medium" | "high"
    kind: str  # naturalness | fidelity | register | consistency | glossary | name
    snippet: str
    suggestion: str


@dataclass(frozen=True)
class CritiqueResult:
    naturalness: float
    fidelity: float
    register_match: float
    consistency: float
    overall: float
    issues: list[CritiqueIssue]
    raw: dict[str, Any]
    total_tokens: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    # M11: tokens billed at the cached input rate (subset of prompt_tokens).
    cached_input_tokens: int = 0

    def has_high_severity(self) -> bool:
        return any(i.severity == "high" for i in self.issues)


_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "block_id": {"type": "string"},
                    "translated": {"type": "string"},
                },
                "required": ["block_id", "translated"],
            },
        },
        "new_names": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "burmese": {"type": "string"},
                },
                "required": ["source", "burmese"],
            },
        },
        "summary_for_next_chunk": {"type": "string"},
        # M8: structured running-context snapshot. Optional so older prompt
        # versions and revise responses without it still validate.
        "running_context": {"type": "object"},
    },
    "required": ["blocks"],
}


_CRITIQUE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "naturalness": {"type": "number"},
        "fidelity": {"type": "number"},
        "register_match": {"type": "number"},
        "consistency": {"type": "number"},
        "overall": {"type": "number"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "block_id": {"type": "string"},
                    "severity": {"type": "string"},
                    "kind": {"type": "string"},
                    "snippet": {"type": "string"},
                    "suggestion": {"type": "string"},
                },
                "required": ["block_id", "severity", "kind"],
            },
        },
    },
    "required": ["naturalness", "fidelity", "register_match", "consistency", "overall"],
}


def translate_chunk(
    *,
    blocks: list[dict],  # [{"block_id": str, "source": str}]
    glossary: list[GlossaryEntry],
    name_registry: list[NameRegistry],
    previous_summary: str | None,
    previous_running_context: dict[str, Any] | None = None,
    category: Category | None = None,
    style_profile: StyleProfile | None = None,
    system_prompt: str | None = None,
    model: str | None = None,
) -> ChunkTranslationResult:
    """Translate one chunk. Raises ``TranslationError`` on any failure.

    Either ``system_prompt`` or ``category`` must be provided; the prompt is
    composed from ``style_profile`` when given, otherwise falls back to the
    base category prompt. ``model`` overrides the default translation model
    (used by the revise pass which reuses this call shape).
    ``previous_running_context`` is the structured snapshot from chunk N-1
    (M8); when present it ships to the model in the user payload.
    """
    if not blocks:
        return ChunkTranslationResult(blocks=[], new_names=[], summary_for_next_chunk="")

    s = get_settings()
    if system_prompt is None:
        if category is None:
            raise TranslationError("translate_chunk requires either system_prompt or category")
        system_prompt = compose_translate_system_prompt(category, style_profile)

    user_payload = _build_user_payload(
        blocks=blocks,
        glossary=glossary,
        name_registry=name_registry,
        previous_summary=previous_summary,
        previous_running_context=previous_running_context,
    )

    try:
        resp = generate_json(
            model=model or s.gemini_translation_model,
            system_instruction=system_prompt,
            user_text=json.dumps(user_payload, ensure_ascii=False),
            response_schema=_RESPONSE_SCHEMA,
            temperature=0.2,
        )
    except GeminiModelMissing as e:
        raise TranslationModelMissing(str(e)) from e
    except GeminiAccessDenied as e:
        raise TranslationAccessDenied(str(e)) from e
    except GeminiUnavailable as e:
        raise TranslationError(f"gemini unavailable: {e}") from e

    return _parse_response(
        resp.payload,
        expected_block_ids=[b["block_id"] for b in blocks],
        total_tokens=resp.total_tokens,
        prompt_tokens=resp.prompt_tokens,
        output_tokens=resp.output_tokens,
        cached_input_tokens=resp.cached_input_tokens,
    )


def compose_translate_system_prompt(
    category: Category,
    profile: StyleProfile | None,
    *,
    voice_card_text: str | None = None,
    book_brief_text: str | None = None,
) -> str:
    """Compose: base category prompt + profile goals override + vocabulary
    preferences + book brief + voice-card target + curated few-shot examples.

    ``voice_card_text`` and ``book_brief_text`` are optional system-prompt
    blocks resolved once per job (Track B / M11). They land in the cached
    portion of the prefix so per-chunk cost stays flat.

    Returns the full text passed to Gemini as ``system_instruction``. Stable
    output for the same inputs — the prompt-version cache hashes the result.
    """
    base = _load_base_translate_prompt(category)
    if profile is None and not voice_card_text and not book_brief_text:
        return base

    parts: list[str] = [base.rstrip()]

    if profile is not None:
        override = (profile.system_prompt_template or "").strip()
        if override:
            parts.append("---")
            parts.append(override)

    if book_brief_text:
        parts.append("---")
        parts.append(book_brief_text.strip())

    if voice_card_text:
        parts.append("---")
        parts.append(voice_card_text.strip())

    if profile is not None and profile.vocabulary_preferences_json:
        parts.append("---")
        parts.append("Vocabulary preferences (machine-readable hints):")
        parts.append(
            json.dumps(profile.vocabulary_preferences_json, ensure_ascii=False, indent=2)
        )

    examples_block = (
        _render_few_shot_examples(profile.few_shot_examples_json or [])
        if profile is not None
        else ""
    )
    if examples_block:
        parts.append("---")
        parts.append(examples_block)

    return "\n\n".join(parts) + "\n"


def prompt_key_for_translate(category: Category, profile: StyleProfile | None) -> str:
    """Stable key for ``prompt_versions``: identifies the *role* (translate)
    plus the profile/category. Multiple versions per key accumulate over time
    as the curator edits the YAML."""
    if profile is None:
        return f"translate__{category.value.lower()}__legacy"
    return f"translate__{profile.key}"


def _load_base_translate_prompt(category: Category) -> str:
    if category == Category.EDUCATIONAL:
        path = _PROMPT_DIR / "translate_educational.txt"
    elif category == Category.PERSONAL_DEVELOPMENT:
        path = _PROMPT_DIR / "translate_personal_development.txt"
    else:
        # NOVEL is the default; UNKNOWN should never reach here (the start
        # endpoint blocks it) but if it does, treat as novel.
        path = _PROMPT_DIR / "translate_novel.txt"
    return path.read_text(encoding="utf-8")


def _render_few_shot_examples(examples: list[dict[str, Any]]) -> str:
    """Render the examples block. Skip rows whose target is a ``[TODO …]``
    placeholder — handing the model "[TODO …]" as a target would teach it to
    emit "[TODO …]" rather than a real translation."""
    rendered: list[str] = []
    idx = 1
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        target = (ex.get("target") or "").strip()
        source = (ex.get("source") or "").strip()
        if not target or not source or target.startswith("[TODO"):
            continue
        note = (ex.get("note") or "").strip()
        block = f"EXAMPLE {idx}\nSOURCE:\n{source}\n\nTRANSLATION:\n{target}"
        if note:
            block += f"\n\n(Note: {note})"
        rendered.append(block)
        idx += 1

    if not rendered:
        return ""

    sep = "\n\n=========\n\n"
    return (
        "EXAMPLES (illustrative — match this register and approach):\n"
        "============================================================\n"
        + sep.join(rendered)
        + "\n============================================================"
    )


def _build_user_payload(
    *,
    blocks: list[dict],
    glossary: list[GlossaryEntry],
    name_registry: list[NameRegistry],
    previous_summary: str | None,
    previous_running_context: dict[str, Any] | None = None,
) -> dict:
    payload = {
        "glossary": [
            {"source": g.term_source, "target": g.term_target} for g in glossary
        ],
        "name_registry": [
            {"source": n.source_name, "burmese": n.burmese_phonetic} for n in name_registry
        ],
        "previous_chunk_summary": previous_summary or "",
        "blocks": blocks,
    }
    if previous_running_context:
        payload["running_context"] = serialize_for_prompt(previous_running_context)
    return payload


def _parse_response(
    raw: dict,
    *,
    expected_block_ids: list[str],
    total_tokens: int = 0,
    prompt_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
) -> ChunkTranslationResult:
    blocks_raw = raw.get("blocks") or []
    if not isinstance(blocks_raw, list):
        raise TranslationError("response 'blocks' must be a list")

    by_id: dict[str, str] = {}
    for entry in blocks_raw:
        if not isinstance(entry, dict):
            continue
        bid = entry.get("block_id")
        translated = entry.get("translated")
        if not isinstance(bid, str) or not isinstance(translated, str):
            continue
        by_id[bid] = translated

    missing = [bid for bid in expected_block_ids if bid not in by_id]
    if missing:
        raise TranslationError(f"model omitted blocks: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    blocks_out = [
        TranslatedBlock(block_id=bid, translated=by_id[bid]) for bid in expected_block_ids
    ]

    names_raw = raw.get("new_names") or []
    new_names: list[DiscoveredName] = []
    if isinstance(names_raw, list):
        for n in names_raw:
            if not isinstance(n, dict):
                continue
            src = n.get("source")
            burmese = n.get("burmese")
            if isinstance(src, str) and isinstance(burmese, str) and src.strip() and burmese.strip():
                new_names.append(DiscoveredName(source=src.strip(), burmese=burmese.strip()))

    summary = raw.get("summary_for_next_chunk") or ""
    if not isinstance(summary, str):
        summary = ""

    rc_raw = raw.get("running_context")
    running_context_update = rc_raw if isinstance(rc_raw, dict) else None

    return ChunkTranslationResult(
        blocks=blocks_out,
        new_names=new_names,
        summary_for_next_chunk=summary,
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        running_context_update=running_context_update,
    )


# ---------------------------------------------------------------------------
# Critique + revise (M7 quality loop)
# ---------------------------------------------------------------------------


def critique_chunk(
    *,
    blocks: list[dict],
    pass_one_blocks: list[TranslatedBlock],
    glossary: list[GlossaryEntry],
    name_registry: list[NameRegistry],
    previous_summary: str | None,
    category: Category,
    style_profile: StyleProfile | None,
    system_prompt: str | None = None,
    model: str | None = None,
) -> CritiqueResult:
    """Pass 2: critique the prior translation. Raises ``TranslationError`` on
    malformed responses. The caller decides whether to ignore the failure
    (keep pass 1) or retry."""
    if not pass_one_blocks:
        raise TranslationError("critique requires a non-empty prior translation")

    s = get_settings()
    if system_prompt is None:
        system_prompt = compose_critique_system_prompt(category, style_profile)

    payload = {
        "glossary": [
            {"source": g.term_source, "target": g.term_target} for g in glossary
        ],
        "name_registry": [
            {"source": n.source_name, "burmese": n.burmese_phonetic} for n in name_registry
        ],
        "previous_chunk_summary": previous_summary or "",
        "source_blocks": blocks,
        "translated_blocks": [
            {"block_id": b.block_id, "translated": b.translated} for b in pass_one_blocks
        ],
    }

    try:
        resp = generate_json(
            model=model or s.gemini_critique_model,
            system_instruction=system_prompt,
            user_text=json.dumps(payload, ensure_ascii=False),
            response_schema=_CRITIQUE_RESPONSE_SCHEMA,
            temperature=0.1,
        )
    except GeminiModelMissing as e:
        raise TranslationModelMissing(str(e)) from e
    except GeminiAccessDenied as e:
        raise TranslationAccessDenied(str(e)) from e
    except GeminiUnavailable as e:
        raise TranslationError(f"gemini unavailable: {e}") from e

    return _parse_critique_response(
        resp.payload,
        total_tokens=resp.total_tokens,
        prompt_tokens=resp.prompt_tokens,
        output_tokens=resp.output_tokens,
        cached_input_tokens=resp.cached_input_tokens,
    )


def revise_chunk(
    *,
    blocks: list[dict],
    pass_one_blocks: list[TranslatedBlock],
    critique: CritiqueResult,
    glossary: list[GlossaryEntry],
    name_registry: list[NameRegistry],
    previous_summary: str | None,
    category: Category,
    style_profile: StyleProfile | None,
    previous_running_context: dict[str, Any] | None = None,
    system_prompt: str | None = None,
    model: str | None = None,
) -> ChunkTranslationResult:
    """Pass 3: produce a revised translation given source + pass-1 + critique."""
    if not blocks:
        return ChunkTranslationResult(blocks=[], new_names=[], summary_for_next_chunk="")

    s = get_settings()
    if system_prompt is None:
        system_prompt = compose_revise_system_prompt(category, style_profile)

    payload = {
        "glossary": [
            {"source": g.term_source, "target": g.term_target} for g in glossary
        ],
        "name_registry": [
            {"source": n.source_name, "burmese": n.burmese_phonetic} for n in name_registry
        ],
        "previous_chunk_summary": previous_summary or "",
        "source_blocks": blocks,
        "prior_translation": [
            {"block_id": b.block_id, "translated": b.translated} for b in pass_one_blocks
        ],
        "critique": critique.raw,
    }
    if previous_running_context:
        payload["running_context"] = serialize_for_prompt(previous_running_context)

    try:
        resp = generate_json(
            model=model or s.gemini_translation_model,
            system_instruction=system_prompt,
            user_text=json.dumps(payload, ensure_ascii=False),
            response_schema=_RESPONSE_SCHEMA,
            temperature=0.2,
        )
    except GeminiModelMissing as e:
        raise TranslationModelMissing(str(e)) from e
    except GeminiAccessDenied as e:
        raise TranslationAccessDenied(str(e)) from e
    except GeminiUnavailable as e:
        raise TranslationError(f"gemini unavailable: {e}") from e

    return _parse_response(
        resp.payload,
        expected_block_ids=[b["block_id"] for b in blocks],
        total_tokens=resp.total_tokens,
        prompt_tokens=resp.prompt_tokens,
        output_tokens=resp.output_tokens,
        cached_input_tokens=resp.cached_input_tokens,
    )


def compose_critique_system_prompt(
    category: Category,
    profile: StyleProfile | None,
    *,
    voice_card_text: str | None = None,
    book_brief_text: str | None = None,
) -> str:
    """Critique prompt = base critique rubric + the profile context the
    translator was working under (so the critic scores against the right
    register) + optional book brief and voice-card target (so the critic
    knows what voice the translation should be matching).
    """
    base = (_PROMPT_DIR / "critique.txt").read_text(encoding="utf-8")
    if profile is None and not voice_card_text and not book_brief_text:
        return base
    parts = [base.rstrip()]
    if profile is not None:
        parts.append("---")
        parts.append("Style profile the translator was working under:")
        parts.append(profile.system_prompt_template.strip())
    if book_brief_text:
        parts.append("---")
        parts.append(book_brief_text.strip())
    if voice_card_text:
        parts.append("---")
        parts.append(voice_card_text.strip())
    if profile is not None and profile.vocabulary_preferences_json:
        parts.append("Vocabulary preferences:")
        parts.append(
            json.dumps(profile.vocabulary_preferences_json, ensure_ascii=False, indent=2)
        )
    return "\n\n".join(parts) + "\n"


def compose_revise_system_prompt(
    category: Category,
    profile: StyleProfile | None,
    *,
    voice_card_text: str | None = None,
    book_brief_text: str | None = None,
) -> str:
    """Revise prompt = base revise rules + same profile composition as
    translate (override + vocabulary preferences + few-shot examples) plus
    optional book brief and voice-card target, so the revised output keeps
    matching the same voice the translator was aiming for."""
    base = (_PROMPT_DIR / "revise.txt").read_text(encoding="utf-8")
    if profile is None and not voice_card_text and not book_brief_text:
        return base
    parts = [base.rstrip()]
    if profile is not None:
        parts.append("---")
        parts.append(profile.system_prompt_template.strip())
    if book_brief_text:
        parts.append("---")
        parts.append(book_brief_text.strip())
    if voice_card_text:
        parts.append("---")
        parts.append(voice_card_text.strip())
    if profile is not None and profile.vocabulary_preferences_json:
        parts.append("Vocabulary preferences (machine-readable hints):")
        parts.append(
            json.dumps(profile.vocabulary_preferences_json, ensure_ascii=False, indent=2)
        )
    examples_block = (
        _render_few_shot_examples(profile.few_shot_examples_json or [])
        if profile is not None
        else ""
    )
    if examples_block:
        parts.append(examples_block)
    return "\n\n".join(parts) + "\n"


def prompt_key_for_critique(category: Category, profile: StyleProfile | None) -> str:
    if profile is None:
        return f"critique__{category.value.lower()}__legacy"
    return f"critique__{profile.key}"


def prompt_key_for_revise(category: Category, profile: StyleProfile | None) -> str:
    if profile is None:
        return f"revise__{category.value.lower()}__legacy"
    return f"revise__{profile.key}"


def prompt_key_for_judge(category: Category, profile: StyleProfile | None) -> str:
    if profile is None:
        return f"judge__{category.value.lower()}__legacy"
    return f"judge__{profile.key}"


def compose_judge_system_prompt(
    category: Category,
    profile: StyleProfile | None,
    *,
    voice_card_text: str | None = None,
    book_brief_text: str | None = None,
) -> str:
    """Judge prompt = base rubric + the profile context the translation was
    produced under + optional book brief and voice-card target, so the judge
    scores against the right voice. The judge is *not* given few-shot
    examples — the rubric anchors are enough, and we don't want to bias
    the score toward a specific exemplar."""
    base = (_PROMPT_DIR / "judge.txt").read_text(encoding="utf-8")
    if profile is None and not voice_card_text and not book_brief_text:
        return base
    parts = [base.rstrip()]
    if profile is not None:
        parts.append("---")
        parts.append("Style profile the translation was produced under:")
        parts.append(profile.system_prompt_template.strip())
    if book_brief_text:
        parts.append("---")
        parts.append(book_brief_text.strip())
    if voice_card_text:
        parts.append("---")
        parts.append(voice_card_text.strip())
    if profile is not None and profile.vocabulary_preferences_json:
        parts.append("Vocabulary preferences:")
        parts.append(
            json.dumps(profile.vocabulary_preferences_json, ensure_ascii=False, indent=2)
        )
    return "\n\n".join(parts) + "\n"


def judge_chunk(
    *,
    blocks: list[dict],
    final_blocks: list[TranslatedBlock],
    glossary: list[GlossaryEntry],
    name_registry: list[NameRegistry],
    previous_summary: str | None,
    category: Category,
    style_profile: StyleProfile | None,
    system_prompt: str | None = None,
    model: str | None = None,
) -> CritiqueResult:
    """Independent score of a finished translation against its source.

    Returns a ``CritiqueResult`` since the output shape is structurally
    identical to a critique (per-axis 1–5 scores + issues). The caller
    persists the result as a ``quality_scores`` row, not a critique pass.
    """
    if not final_blocks:
        raise TranslationError("judge requires a non-empty final translation")

    s = get_settings()
    if system_prompt is None:
        system_prompt = compose_judge_system_prompt(category, style_profile)

    payload = {
        "glossary": [
            {"source": g.term_source, "target": g.term_target} for g in glossary
        ],
        "name_registry": [
            {"source": n.source_name, "burmese": n.burmese_phonetic} for n in name_registry
        ],
        "previous_chunk_summary": previous_summary or "",
        "source_blocks": blocks,
        "translated_blocks": [
            {"block_id": b.block_id, "translated": b.translated} for b in final_blocks
        ],
    }

    try:
        resp = generate_json(
            model=model or s.gemini_judge_model,
            system_instruction=system_prompt,
            user_text=json.dumps(payload, ensure_ascii=False),
            response_schema=_CRITIQUE_RESPONSE_SCHEMA,
            temperature=0.0,
        )
    except GeminiModelMissing as e:
        raise TranslationModelMissing(str(e)) from e
    except GeminiAccessDenied as e:
        raise TranslationAccessDenied(str(e)) from e
    except GeminiUnavailable as e:
        raise TranslationError(f"gemini unavailable: {e}") from e

    return _parse_critique_response(
        resp.payload,
        total_tokens=resp.total_tokens,
        prompt_tokens=resp.prompt_tokens,
        output_tokens=resp.output_tokens,
        cached_input_tokens=resp.cached_input_tokens,
    )


def _parse_critique_response(
    raw: dict[str, Any],
    *,
    total_tokens: int = 0,
    prompt_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
) -> CritiqueResult:
    def _score(key: str) -> float:
        v = raw.get(key)
        if not isinstance(v, (int, float)):
            raise TranslationError(f"critique missing numeric '{key}'")
        f = float(v)
        # Clamp into [1, 5] — the model occasionally returns 0 or > 5.
        return max(1.0, min(5.0, f))

    naturalness = _score("naturalness")
    fidelity = _score("fidelity")
    register_match = _score("register_match")
    consistency = _score("consistency")
    overall = _score("overall")

    issues: list[CritiqueIssue] = []
    raw_issues = raw.get("issues") or []
    if isinstance(raw_issues, list):
        for it in raw_issues:
            if not isinstance(it, dict):
                continue
            block_id = it.get("block_id")
            severity = (it.get("severity") or "").lower()
            kind = (it.get("kind") or "").lower()
            if not isinstance(block_id, str) or severity not in {"low", "medium", "high"}:
                continue
            issues.append(
                CritiqueIssue(
                    block_id=block_id,
                    severity=severity,
                    kind=kind or "other",
                    snippet=str(it.get("snippet") or ""),
                    suggestion=str(it.get("suggestion") or ""),
                )
            )

    return CritiqueResult(
        naturalness=naturalness,
        fidelity=fidelity,
        register_match=register_match,
        consistency=consistency,
        overall=overall,
        issues=issues,
        raw=raw,
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
    )

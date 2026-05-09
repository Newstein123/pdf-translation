"""Pre-translation glossary auto-suggester (M8).

Samples chunks of extracted text proportionally across the document, asks
Gemini to surface high-value terms (proper nouns, repeated terminology,
idioms), deduplicates against existing glossary entries and the name
registry, and persists the survivors as ``glossary_suggestions`` rows in
PENDING state for the user to approve/reject before translation begins.

Designed to run after extraction + classification, before
AWAITING_CONFIRMATION. Best-effort: if Gemini is unavailable, the suggester
no-ops and the user can still use the manual glossary upload path.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import (
    GlossaryEntry,
    GlossarySuggestion,
    NameRegistry,
    SuggestionKind,
    SuggestionState,
)
from app.services.gemini import GeminiUnavailable, generate_json

log = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "glossary_suggest.txt"


_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term_source": {"type": "string"},
                    "candidate_burmese": {"type": "string"},
                    "kind": {"type": "string"},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["term_source", "candidate_burmese"],
            },
        }
    },
    "required": ["suggestions"],
}


@dataclass(frozen=True)
class SuggestedTerm:
    term_source: str
    term_target: str
    kind: SuggestionKind
    confidence: float
    evidence: list[str]


def sample_excerpts_from_structure(
    structure_path: Path, *, max_excerpts: int = 6, max_chars_per_excerpt: int = 1500
) -> list[str]:
    """Sample text excerpts proportionally across the document.

    Sampling start, middle, and end (rather than just the first N pages) means
    the suggester sees terminology that appears late in long books — exactly
    where consistency starts to drift in single-pass translation.
    """
    if not structure_path.exists():
        return []
    try:
        with structure_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        log.exception("failed to read structure.json at %s", structure_path)
        return []

    pages = data.get("pages") or []
    if not pages:
        return []

    if len(pages) <= max_excerpts:
        sampled_pages = pages
    else:
        # Evenly spaced indices into the page list.
        step = len(pages) / max_excerpts
        sampled_pages = [pages[min(int(i * step), len(pages) - 1)] for i in range(max_excerpts)]

    excerpts: list[str] = []
    for page in sampled_pages:
        blocks = page.get("blocks") or []
        text = "\n".join(b.get("text") or "" for b in blocks).strip()
        if not text:
            continue
        if len(text) > max_chars_per_excerpt:
            text = text[:max_chars_per_excerpt]
        excerpts.append(text)
    return excerpts


def call_suggester(excerpts: list[str], *, model: str | None = None) -> list[SuggestedTerm]:
    """Call Gemini once with the sampled excerpts and return parsed suggestions.

    Raises ``GeminiUnavailable`` upstream; the caller decides whether that is
    fatal (it usually isn't — the user can still upload a manual glossary)."""
    if not excerpts:
        return []
    s = get_settings()
    system = _PROMPT_PATH.read_text(encoding="utf-8")
    payload = {"excerpts": excerpts}
    resp = generate_json(
        model=model or s.gemini_utility_model,
        system_instruction=system,
        user_text=json.dumps(payload, ensure_ascii=False),
        response_schema=_RESPONSE_SCHEMA,
        temperature=0.0,
    )
    return _parse_suggestions(resp.payload)


def _parse_suggestions(raw: dict[str, Any]) -> list[SuggestedTerm]:
    out: list[SuggestedTerm] = []
    rows = raw.get("suggestions") or []
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        src = (row.get("term_source") or "").strip()
        tgt = (row.get("candidate_burmese") or "").strip()
        if not src or not tgt:
            continue
        kind_raw = (row.get("kind") or "").upper()
        try:
            kind = SuggestionKind(kind_raw)
        except ValueError:
            kind = SuggestionKind.OTHER
        confidence = row.get("confidence")
        try:
            confidence_f = float(confidence) if confidence is not None else 0.0
        except (TypeError, ValueError):
            confidence_f = 0.0
        confidence_f = max(0.0, min(1.0, confidence_f))
        evidence_raw = row.get("evidence") or []
        evidence = [str(e).strip() for e in evidence_raw if isinstance(e, str) and e.strip()]
        out.append(
            SuggestedTerm(
                term_source=src,
                term_target=tgt,
                kind=kind,
                confidence=confidence_f,
                evidence=evidence[:3],
            )
        )
    return out


def deduplicate_against_existing(
    session: Session, job_id: str, terms: list[SuggestedTerm]
) -> list[SuggestedTerm]:
    """Drop suggestions whose source already exists in glossary_entries or
    name_registry for this job (case-insensitive match).

    The case-insensitive comparison is intentional: the model often capitalizes
    "alice" → "Alice"; we don't want to surface that as a fresh suggestion if
    the user already approved "alice" earlier.
    """
    existing_terms = {
        row.lower()
        for row in session.execute(
            select(GlossaryEntry.term_source).where(GlossaryEntry.job_id == job_id)
        ).scalars()
    }
    existing_names = {
        row.lower()
        for row in session.execute(
            select(NameRegistry.source_name).where(NameRegistry.job_id == job_id)
        ).scalars()
    }
    existing = existing_terms | existing_names

    # Also dedupe within the suggestion list itself in case the model returns
    # the same term twice with slightly different casing.
    seen_in_batch: set[str] = set()
    out: list[SuggestedTerm] = []
    for t in terms:
        key = t.term_source.lower()
        if key in existing or key in seen_in_batch:
            continue
        seen_in_batch.add(key)
        out.append(t)
    return out


def persist_suggestions(
    session: Session, job_id: str, terms: list[SuggestedTerm]
) -> int:
    """Replace any prior PENDING suggestions for this job with the new batch.

    Already-actioned (APPROVED/REJECTED/MERGED) rows are preserved so the
    user's prior decisions don't get reversed by a re-run."""
    session.execute(
        GlossarySuggestion.__table__.delete().where(
            (GlossarySuggestion.job_id == job_id)
            & (GlossarySuggestion.state == SuggestionState.PENDING)
        )
    )
    for t in terms:
        session.add(
            GlossarySuggestion(
                job_id=job_id,
                term_source=t.term_source,
                term_target=t.term_target,
                kind=t.kind,
                confidence=t.confidence,
                evidence_json=t.evidence,
                state=SuggestionState.PENDING,
            )
        )
    session.commit()
    return len(terms)


def list_suggestions(
    session: Session, job_id: str, *, state: SuggestionState | None = None
) -> list[GlossarySuggestion]:
    stmt = (
        select(GlossarySuggestion)
        .where(GlossarySuggestion.job_id == job_id)
        .order_by(GlossarySuggestion.confidence.desc(), GlossarySuggestion.term_source)
    )
    if state is not None:
        stmt = stmt.where(GlossarySuggestion.state == state)
    return list(session.execute(stmt).scalars())


def approve_suggestion(
    session: Session,
    job_id: str,
    suggestion_id: str,
    *,
    term_target_override: str | None = None,
    actioned_by: str = "user",
) -> GlossarySuggestion:
    """Promote a PENDING suggestion to APPROVED + create a glossary_entries row.

    The optional ``term_target_override`` lets the user edit the proposed
    Burmese term before approving it, which is the common case once they
    review the suggester's output."""
    from datetime import datetime, timezone

    suggestion = session.get(GlossarySuggestion, suggestion_id)
    if suggestion is None or suggestion.job_id != job_id:
        raise ValueError(f"unknown suggestion id {suggestion_id} for job {job_id}")
    if suggestion.state in {SuggestionState.APPROVED, SuggestionState.MERGED}:
        return suggestion

    final_target = (term_target_override or suggestion.term_target).strip()
    if not final_target:
        raise ValueError("term_target cannot be empty")

    # If the user already added the same source manually, mark as MERGED rather
    # than creating a duplicate glossary entry.
    existing = session.execute(
        select(GlossaryEntry)
        .where(
            GlossaryEntry.job_id == job_id,
            GlossaryEntry.term_source == suggestion.term_source,
        )
        .limit(1)
    ).scalar_one_or_none()

    if existing is not None:
        suggestion.state = SuggestionState.MERGED
        suggestion.approved_glossary_entry_id = existing.id
    else:
        entry = GlossaryEntry(
            job_id=job_id,
            term_source=suggestion.term_source,
            term_target=final_target,
            case_sensitive=False,
        )
        session.add(entry)
        session.flush()
        suggestion.state = SuggestionState.APPROVED
        suggestion.approved_glossary_entry_id = entry.id
        suggestion.term_target = final_target

    suggestion.actioned_by = actioned_by
    suggestion.actioned_at = datetime.now(timezone.utc)
    session.commit()
    return suggestion


def reject_suggestion(
    session: Session,
    job_id: str,
    suggestion_id: str,
    *,
    actioned_by: str = "user",
) -> GlossarySuggestion:
    from datetime import datetime, timezone

    suggestion = session.get(GlossarySuggestion, suggestion_id)
    if suggestion is None or suggestion.job_id != job_id:
        raise ValueError(f"unknown suggestion id {suggestion_id} for job {job_id}")
    suggestion.state = SuggestionState.REJECTED
    suggestion.actioned_by = actioned_by
    suggestion.actioned_at = datetime.now(timezone.utc)
    session.commit()
    return suggestion


def run_for_job(
    session: Session, job_id: str, structure_path: Path
) -> dict[str, int]:
    """Top-level: sample → call model → dedupe → persist. Returns counts.

    Empty result on transient failure is fine — the user can still use the
    manual glossary path."""
    excerpts = sample_excerpts_from_structure(structure_path)
    if not excerpts:
        return {"sampled": 0, "proposed": 0, "kept": 0}
    try:
        proposed = call_suggester(excerpts)
    except GeminiUnavailable as e:
        log.warning("glossary suggester unavailable: %s", e)
        return {"sampled": len(excerpts), "proposed": 0, "kept": 0}
    kept = deduplicate_against_existing(session, job_id, proposed)
    persist_suggestions(session, job_id, kept)
    return {"sampled": len(excerpts), "proposed": len(proposed), "kept": len(kept)}

"""Deterministic post-translation linter for Burmese output.

Catches the long tail of small errors that make machine-translated Burmese
feel "off": Zawgyi residue, glossary violations, missing/extra digits,
inconsistent honorific levels. The linter runs on every translation pass
and writes findings to the ``linter_findings`` table; severity drives flow
(BLOCKER → revise, WARNING → QA report, INFO → telemetry only).

Design principles:

  * **Deterministic.** No model calls — the linter is cheap so it can run
    on every pass without thinking about cost.
  * **Rule-based and registered.** Rules are plain functions registered in
    ``_RULES``. Adding a rule = one function + one registration line.
  * **Diff-aware where useful.** Rules that compare source vs. translation
    (e.g. number preservation) take both texts; pure-output rules (e.g.
    Zawgyi detection) take only the translation.
  * **Ship as WARNING first.** New rules default to WARNING severity
    while we validate against real books; promote to BLOCKER once we
    trust the false-positive rate.

Initial rule set (v1):

  * ``zawgyi_residue``     — codepoints that should never appear in Unicode
                             Burmese output; almost always indicates a
                             Zawgyi/Unicode mix-up. BLOCKER.
  * ``number_drift``       — counts of digit groups in source vs. target
                             differ. WARNING (occasional legitimate cases:
                             "12" → "ဆယ့်နှစ်"; flagged for human review).
  * ``glossary_violation`` — source contains a glossary term but the
                             expected target translation doesn't appear
                             in the output. WARNING.
  * ``empty_output``       — translation is empty / whitespace-only despite
                             non-empty source. BLOCKER.
  * ``ascii_only_output``  — output contains no Burmese codepoints despite
                             non-empty source — almost always means the
                             model echoed English. BLOCKER.

Future rules (Phase 2 of Track B): pronoun consistency, particle balance,
register drift, honorific level drift, quote-mark conventions.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable

from app.db.models import GlossaryEntry, LinterSeverity

log = logging.getLogger(__name__)

# Burmese script Unicode block: U+1000–U+109F. Anything inside is "Burmese
# text"; anything outside (Latin, digits, punctuation, etc.) isn't.
_BURMESE_RE = re.compile(r"[က-႟]")

# Codepoints common in Zawgyi-encoded text but invalid in correct Unicode
# Burmese. Presence of these in output strongly suggests a Zawgyi/Unicode
# mix-up. Conservative list — only the most diagnostic codepoints, to keep
# false positives low.
_ZAWGYI_DIAGNOSTIC_CODEPOINTS = {
    "ၠ",  # MYANMAR LETTER KA WITH LOWER STROKE (Zawgyi only)
    "ၡ",
    "ၢ",
    "ၣ",
    "ၤ",
    "ၥ",
    "ၦ",
    "ၧ",
    "ၨ",
    "ၩ",
    "ၪ",
    "ၫ",
    "ၬ",
    "ၭ",
    "ၮ",
    "ၯ",
    "ၰ",
    "ၱ",
    "ၲ",
    "ၳ",
    "ၴ",
    "ၵ",
    "ၶ",
    "ၷ",
    "ၸ",
    "ၹ",
    "ၺ",
    "ၻ",
    "ၼ",
    "ၽ",
    "ၾ",
    "ၿ",
    "ႀ",
    "ႁ",
    "ႂ",
    "ႃ",
    "ႄ",
    "ႅ",
    "ႆ",
    "ႇ",
    "ႈ",
    "ႉ",
    "ႊ",
    "ႋ",
    "ႌ",
    "ႍ",
    "ႎ",
    "ႏ",
}

# Digit groups: contiguous runs of ASCII digits OR Burmese digits (၀–၉,
# U+1040–U+1049). We compare counts of these between source and translation
# rather than literal characters — Burmese rendering of "1925" can be either
# "၁၉၂၅" or "1925" depending on house style; what matters is *one* digit
# group per source digit group.
#
# Important: we explicitly use ``[0-9]+`` for ASCII rather than ``\d+``.
# Python's ``\d`` matches all Unicode decimal digits — including Burmese ၀-၉
# — which would double-count Burmese digits (once via ``\d``, once via
# ``[၀-၉]``) and trip false-positive drift warnings on faithful translations.
_ASCII_DIGITS_RE = re.compile(r"[0-9]+")
_BURMESE_DIGITS_RE = re.compile(r"[၀-၉]+")


@dataclass(frozen=True)
class LintInput:
    """Bundle passed to every rule.

    Source is optional — pure-output rules ignore it. Glossary is the
    job-scoped list (already filtered by chunk if you've adopted glossary
    scoping in tasks.py)."""

    source_text: str
    target_text: str
    source_blocks: list[dict] | None = None  # [{"block_id": str, "text": str}]
    target_blocks: list[dict] | None = None  # [{"block_id": str, "translated": str}]
    glossary: list[GlossaryEntry] = field(default_factory=list)


@dataclass(frozen=True)
class LintFinding:
    """One finding from a rule. Persisted as a ``LinterFinding`` row by the
    caller — keeping the dataclass DB-free lets unit tests run without a
    session."""

    rule_key: str
    severity: LinterSeverity
    message: str
    block_id: str | None = None
    snippet: str | None = None
    payload: dict | None = None
    position_start: int | None = None
    position_end: int | None = None


Rule = Callable[[LintInput], list[LintFinding]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lint_translation(payload: LintInput, *, rule_keys: list[str] | None = None) -> list[LintFinding]:
    """Run all registered rules (or the subset named in ``rule_keys``) and
    return findings in registration order.

    Findings are returned, not persisted — the caller decides whether to
    write them to ``linter_findings`` or just log them (e.g. tests pass
    over the deterministic output without touching the DB).
    """
    selected = _RULES if not rule_keys else [
        (k, fn) for k, fn in _RULES if k in rule_keys
    ]
    out: list[LintFinding] = []
    for key, rule_fn in selected:
        try:
            out.extend(rule_fn(payload))
        except Exception:
            log.exception("linter rule %r raised; skipping", key)
    return out


def list_rules() -> list[str]:
    """Names of registered rules, in the order they run."""
    return [k for k, _ in _RULES]


# ---------------------------------------------------------------------------
# Rule: zawgyi_residue
# ---------------------------------------------------------------------------


def _rule_zawgyi_residue(payload: LintInput) -> list[LintFinding]:
    text = payload.target_text or ""
    if not text:
        return []
    hits: list[tuple[int, str]] = []
    for i, ch in enumerate(text):
        if ch in _ZAWGYI_DIAGNOSTIC_CODEPOINTS:
            hits.append((i, ch))
            if len(hits) >= 10:  # cap to keep payload small
                break
    if not hits:
        return []
    snippets = [
        text[max(0, pos - 20) : pos + 20] for pos, _ in hits[:3]
    ]
    return [
        LintFinding(
            rule_key="zawgyi_residue",
            severity=LinterSeverity.BLOCKER,
            message=(
                f"output contains {len(hits)} Zawgyi-diagnostic codepoint(s); "
                "input may have been Zawgyi-encoded or model leaked Zawgyi"
            ),
            snippet=" … ".join(snippets),
            payload={"codepoint_hits": [hex(ord(ch)) for _, ch in hits[:10]]},
            position_start=hits[0][0],
            position_end=hits[-1][0] + 1,
        )
    ]


# ---------------------------------------------------------------------------
# Rule: number_drift
# ---------------------------------------------------------------------------


def _rule_number_drift(payload: LintInput) -> list[LintFinding]:
    src = payload.source_text or ""
    tgt = payload.target_text or ""
    if not src or not tgt:
        return []
    src_count = len(_ASCII_DIGITS_RE.findall(src)) + len(_BURMESE_DIGITS_RE.findall(src))
    tgt_count = len(_ASCII_DIGITS_RE.findall(tgt)) + len(_BURMESE_DIGITS_RE.findall(tgt))
    if src_count == tgt_count:
        return []
    return [
        LintFinding(
            rule_key="number_drift",
            severity=LinterSeverity.WARNING,
            message=(
                f"digit-group count mismatch: source has {src_count}, target has {tgt_count}; "
                "verify dates, page references, and quantities round-trip"
            ),
            payload={"source_count": src_count, "target_count": tgt_count},
        )
    ]


# ---------------------------------------------------------------------------
# Rule: glossary_violation
# ---------------------------------------------------------------------------


def _rule_glossary_violation(payload: LintInput) -> list[LintFinding]:
    src = payload.source_text or ""
    tgt = payload.target_text or ""
    if not src or not tgt or not payload.glossary:
        return []
    findings: list[LintFinding] = []
    src_lower = src.lower()
    for entry in payload.glossary:
        term = (entry.term_source or "").strip()
        target = (entry.term_target or "").strip()
        if not term or not target:
            continue
        # Match insensitive unless the entry says case matters.
        haystack = src if entry.case_sensitive else src_lower
        needle = term if entry.case_sensitive else term.lower()
        if needle not in haystack:
            continue
        # Term appears in source. Did the expected target translation make it
        # into the output? Allow a couple of light variants by ignoring
        # surrounding whitespace.
        if target in tgt:
            continue
        findings.append(
            LintFinding(
                rule_key="glossary_violation",
                severity=LinterSeverity.WARNING,
                message=(
                    f"source contains glossary term {term!r} but expected "
                    f"target {target!r} not found in translation"
                ),
                payload={"term_source": term, "term_target": target},
            )
        )
    return findings


# ---------------------------------------------------------------------------
# Rule: empty_output
# ---------------------------------------------------------------------------


def _rule_empty_output(payload: LintInput) -> list[LintFinding]:
    src = (payload.source_text or "").strip()
    tgt = (payload.target_text or "").strip()
    if src and not tgt:
        return [
            LintFinding(
                rule_key="empty_output",
                severity=LinterSeverity.BLOCKER,
                message="translation is empty despite non-empty source",
            )
        ]
    return []


# ---------------------------------------------------------------------------
# Rule: ascii_only_output
# ---------------------------------------------------------------------------


def _rule_ascii_only_output(payload: LintInput) -> list[LintFinding]:
    src = (payload.source_text or "").strip()
    tgt = payload.target_text or ""
    if not src or not tgt:
        return []
    # Tolerate translations that are intentionally English (proper nouns,
    # do-not-translate). We trip the rule only when there is *zero* Burmese
    # in the output and the source was meaningfully long.
    if len(src) < 40:
        return []
    if _BURMESE_RE.search(tgt) is None:
        return [
            LintFinding(
                rule_key="ascii_only_output",
                severity=LinterSeverity.BLOCKER,
                message=(
                    "translation contains no Burmese codepoints; "
                    "model may have echoed source instead of translating"
                ),
                snippet=tgt[:200],
            )
        ]
    return []


# ---------------------------------------------------------------------------
# Rule registry — order is run order (and surface order in the QA report).
# ---------------------------------------------------------------------------


_RULES: list[tuple[str, Rule]] = [
    ("empty_output", _rule_empty_output),
    ("ascii_only_output", _rule_ascii_only_output),
    ("zawgyi_residue", _rule_zawgyi_residue),
    ("number_drift", _rule_number_drift),
    ("glossary_violation", _rule_glossary_violation),
]

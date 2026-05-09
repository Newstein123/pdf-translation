"""Glossary handling.

Two layers per the plan:
  1. Prompt-level — hand the model a list of source→target term mappings as
     hard constraints (handled by translation.build_prompt).
  2. Post-check — verify that every source term that *appears in the source*
     also appears (translated correctly) in the output. Violations are returned
     so the caller can retry the chunk or apply deterministic find/replace.

Glossary upload formats accepted by ``parse_glossary``:
  - JSON array: ``[{"term": "...", "translation": "..."}, ...]``
  - JSON object: ``{"term": "translation", ...}``
  - CSV / TSV / newline-delimited: ``term,translation`` (or tab) per line
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass

from app.db.models import GlossaryEntry


@dataclass(frozen=True)
class ParsedTerm:
    term_source: str
    term_target: str
    case_sensitive: bool = False


@dataclass(frozen=True)
class GlossaryViolation:
    term_source: str
    expected_target: str
    found_in_source: bool
    found_in_translation: bool


def parse_glossary(raw: bytes, content_type: str | None) -> list[ParsedTerm]:
    """Parse uploaded bytes into a list of ParsedTerm. Raises ``ValueError`` on bad input."""
    text = raw.decode("utf-8-sig")
    ct = (content_type or "").lower()
    if "json" in ct or text.lstrip().startswith(("{", "[")):
        return _parse_json(text)
    return _parse_delimited(text)


def _parse_json(text: str) -> list[ParsedTerm]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e
    out: list[ParsedTerm] = []
    if isinstance(data, dict):
        for k, v in data.items():
            if not isinstance(v, str) or not v:
                continue
            out.append(ParsedTerm(term_source=str(k).strip(), term_target=v.strip()))
    elif isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            src = entry.get("term") or entry.get("source")
            tgt = entry.get("translation") or entry.get("target")
            if not src or not tgt:
                continue
            out.append(
                ParsedTerm(
                    term_source=str(src).strip(),
                    term_target=str(tgt).strip(),
                    case_sensitive=bool(entry.get("case_sensitive", False)),
                )
            )
    else:
        raise ValueError("glossary JSON must be an object or array")
    return [t for t in out if t.term_source and t.term_target]


def _parse_delimited(text: str) -> list[ParsedTerm]:
    sample = text[:1024]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except csv.Error:
        dialect = csv.excel  # default to comma
    reader = csv.reader(io.StringIO(text), dialect)
    out: list[ParsedTerm] = []
    for row in reader:
        if len(row) < 2:
            continue
        src, tgt = row[0].strip(), row[1].strip()
        if src and tgt and src.lower() not in {"term", "source"}:
            out.append(ParsedTerm(term_source=src, term_target=tgt))
    return out


def find_violations(
    *,
    source_text: str,
    translated_text: str,
    glossary: list[GlossaryEntry] | list[ParsedTerm],
) -> list[GlossaryViolation]:
    """Return any glossary terms that appeared in source but whose target is
    missing from the translation. Longest-match-first.
    """
    if not source_text or not translated_text or not glossary:
        return []

    pairs = [(_term_src(g), _term_tgt(g), _term_cs(g)) for g in glossary]
    pairs.sort(key=lambda p: len(p[0]), reverse=True)

    violations: list[GlossaryViolation] = []
    for src, tgt, case_sensitive in pairs:
        if not src:
            continue
        flags = 0 if case_sensitive else re.IGNORECASE
        # Word boundaries help avoid false matches on substrings.
        pattern = re.compile(rf"(?<!\w){re.escape(src)}(?!\w)", flags)
        in_source = pattern.search(source_text) is not None
        if not in_source:
            continue
        in_translation = tgt in translated_text
        if not in_translation:
            violations.append(
                GlossaryViolation(
                    term_source=src,
                    expected_target=tgt,
                    found_in_source=True,
                    found_in_translation=False,
                )
            )
    return violations


def deterministic_replace(translated_text: str, violations: list[GlossaryViolation]) -> str:
    """Last-resort fix: insert glossary targets where they're missing, marked
    so a reviewer can spot them. We append `[expected_target]` rather than
    blindly substituting, which would risk corrupting natural sentences."""
    if not violations:
        return translated_text
    tail = "\n\n[Glossary corrections]: " + "; ".join(
        f"{v.term_source} → {v.expected_target}" for v in violations
    )
    return translated_text + tail


def _term_src(g) -> str:
    return getattr(g, "term_source", "")


def _term_tgt(g) -> str:
    return getattr(g, "term_target", "")


def _term_cs(g) -> bool:
    return bool(getattr(g, "case_sensitive", False))

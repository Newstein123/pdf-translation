"""Unit tests for the deterministic Burmese linter (M11 / Track B)."""
from __future__ import annotations

import pytest

from app.db.models import GlossaryEntry, LinterSeverity
from app.services.burmese_linter import (
    LintInput,
    lint_translation,
    list_rules,
)


def _glossary(*pairs: tuple[str, str], case_sensitive: bool = False) -> list[GlossaryEntry]:
    out: list[GlossaryEntry] = []
    for src, tgt in pairs:
        out.append(
            GlossaryEntry(
                term_source=src, term_target=tgt, case_sensitive=case_sensitive
            )
        )
    return out


def test_list_rules_returns_v1_rule_set() -> None:
    keys = list_rules()
    assert "empty_output" in keys
    assert "ascii_only_output" in keys
    assert "zawgyi_residue" in keys
    assert "number_drift" in keys
    assert "glossary_violation" in keys


def test_clean_translation_yields_no_findings() -> None:
    findings = lint_translation(
        LintInput(
            source_text="Hello, world. This is a clean sentence.",
            target_text="မင်္ဂလာပါ ကမ္ဘာ။ ဒါက သန့်ရှင်းတဲ့ဝါကျတစ်ခုပါ။",
        )
    )
    assert findings == []


def test_empty_output_blocker() -> None:
    findings = lint_translation(
        LintInput(
            source_text="A non-empty source paragraph.",
            target_text="   ",
        )
    )
    rules = {f.rule_key for f in findings}
    assert "empty_output" in rules
    blocker = next(f for f in findings if f.rule_key == "empty_output")
    assert blocker.severity == LinterSeverity.BLOCKER


def test_ascii_only_output_blocker_when_source_long_enough() -> None:
    findings = lint_translation(
        LintInput(
            source_text="A reasonably long source paragraph with several words to translate.",
            target_text="A reasonably long source paragraph with several words to translate.",
        )
    )
    assert any(
        f.rule_key == "ascii_only_output" and f.severity == LinterSeverity.BLOCKER
        for f in findings
    )


def test_ascii_only_output_skipped_for_short_source() -> None:
    # Short sources legitimately translate to short outputs that may have no
    # Burmese (e.g. proper noun do-not-translate). Don't trip the rule.
    findings = lint_translation(
        LintInput(source_text="John", target_text="John")
    )
    assert all(f.rule_key != "ascii_only_output" for f in findings)


def test_zawgyi_residue_detected() -> None:
    findings = lint_translation(
        LintInput(
            source_text="Hello world",
            target_text="မင်္ဂလာပါ ၸ ၹ ၺ",  # contains Zawgyi-diagnostic codepoints
        )
    )
    z = [f for f in findings if f.rule_key == "zawgyi_residue"]
    assert z, "expected zawgyi_residue finding"
    assert z[0].severity == LinterSeverity.BLOCKER
    assert "codepoint_hits" in (z[0].payload or {})


def test_number_drift_warning_when_counts_diverge() -> None:
    findings = lint_translation(
        LintInput(
            source_text="In 1925 there were 12 ships and 3 captains.",
            # Translation drops "12" and "3" — a real failure mode
            target_text="၁၉၂၅ ခုနှစ်တွင် သင်္ဘောအချို့နှင့် ကပ္ပတိန်များရှိခဲ့သည်။",
        )
    )
    drift = [f for f in findings if f.rule_key == "number_drift"]
    assert drift, "expected number_drift finding"
    assert drift[0].severity == LinterSeverity.WARNING
    assert drift[0].payload["source_count"] == 3
    assert drift[0].payload["target_count"] == 1


def test_number_drift_silent_when_arabic_to_burmese_digits_preserved() -> None:
    findings = lint_translation(
        LintInput(
            source_text="In 1925 there were 12 ships.",
            target_text="၁၉၂၅ ခုနှစ်တွင် သင်္ဘော ၁၂ စီးရှိခဲ့သည်။",
        )
    )
    assert all(f.rule_key != "number_drift" for f in findings)


def test_glossary_violation_when_translation_misses_canonical_target() -> None:
    findings = lint_translation(
        LintInput(
            source_text="The protagonist met Sherlock at Baker Street.",
            target_text="ဇာတ်လိုက်သည် ဘေကာလမ်းတွင် တွေ့ဆုံခဲ့သည်။",  # no "ရှာလော့ခ်"
            glossary=_glossary(("Sherlock", "ရှာလော့ခ်")),
        )
    )
    violations = [f for f in findings if f.rule_key == "glossary_violation"]
    assert violations, "expected glossary_violation"
    assert violations[0].severity == LinterSeverity.WARNING


def test_glossary_violation_silent_when_target_present() -> None:
    findings = lint_translation(
        LintInput(
            source_text="Sherlock arrived.",
            target_text="ရှာလော့ခ် ရောက်လာသည်။",
            glossary=_glossary(("Sherlock", "ရှာလော့ခ်")),
        )
    )
    assert all(f.rule_key != "glossary_violation" for f in findings)


def test_lint_subset_runs_only_named_rules() -> None:
    findings = lint_translation(
        LintInput(source_text="abc", target_text=""),
        rule_keys=["zawgyi_residue"],  # would otherwise also fire empty_output
    )
    assert all(f.rule_key == "zawgyi_residue" for f in findings)


def test_unknown_rule_in_subset_is_safely_ignored() -> None:
    findings = lint_translation(
        LintInput(source_text="abc", target_text="abc"),
        rule_keys=["does_not_exist"],
    )
    assert findings == []

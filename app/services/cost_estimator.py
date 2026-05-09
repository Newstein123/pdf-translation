"""Pre-translation cost estimation + actual-spend reconciliation.

Two flavors:

1. ``estimate_job_cost(session, job_id)`` — runs as soon as extraction is
   done. Reads the source-text token count from ``structure.json`` and
   applies empirical multipliers (calibrated against observed runs) to
   project Pro vs Flash spend across the M7 quality loop. Returns a low /
   current / high range so the user can see the price *before* clicking
   Start translation.

2. ``actual_job_cost(session, job_id)`` — sums every persisted
   ``chunk_translation_passes`` row by ``model`` × (input_tokens,
   output_tokens) × the configured rate card. This is the ground-truth
   receipt — what was actually billed. Available during translation
   (running total) and after completion.

Multipliers calibrated against the H. G. Wells run (8 pages → ~148k tokens)
which works out to ~18.5k tokens/page across the full pipeline. The
multipliers below decompose that into per-pass shares so we can split
between Pro and Flash for accurate pricing.
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
    ChunkTranslationPass,
    Job,
    PassStatus,
    TranslationChunk,
)
from app.utils.tokens import count_tokens

log = logging.getLogger(__name__)


# ----- Per-pass multipliers (per source-token, amortized) -----
#
# Numbers reflect the prompt envelope each pass ships (system prompt + glossary
# + name registry + running context) on top of the source/translation payload.
# Calibrated against the 8-page H.G. Wells run (5126 src tokens → 77k actual).
# Burmese is denser in Gemini's tokenizer than English — output multipliers
# are ~3× source, not the 1.1× I initially guessed.
#
# The actual receipt comes from ``actual_job_cost``; these are forward
# estimates only.

# Translate pass (always runs).
_TRANSLATE_INPUT_PER_SRC = 2.1  # source + ~3k system overhead per chunk averaged
_TRANSLATE_OUTPUT_PER_SRC = 2.9  # Burmese tokenizes ~3× source in Gemini

# Critique pass (runs whenever critique_enabled).
_CRITIQUE_INPUT_PER_SRC = 4.5  # sees source + translation + system prompt
_CRITIQUE_OUTPUT_PER_SRC = 0.45  # JSON of scores + per-block issues

# Revise pass — only when critique flags issues.
_REVISE_INPUT_PER_SRC = 4.2  # source + pass-1 + critique + system overhead
_REVISE_OUTPUT_PER_SRC = 2.65
_REVISE_RATE_DEFAULT = 0.6  # observed: 75% on the Wells run; conservative 60%

# Judge pass — sampled by ``judge_sample_rate`` + always-judge first chunk +
# always-judge revised chunks. Average sample rate effectively ~0.3 with the
# default settings on a typical book.
_JUDGE_INPUT_PER_SRC = 1.8
_JUDGE_OUTPUT_PER_SRC = 0.15

# Glossary suggester (one-shot, fixed cost regardless of book size).
_SUGGESTER_INPUT_TOKENS = 8_000  # ~6 sampled excerpts + system prompt
_SUGGESTER_OUTPUT_TOKENS = 2_000  # JSON of ~10-20 suggestions

# Classifier (one-shot).
_CLASSIFIER_INPUT_TOKENS = 4_500
_CLASSIFIER_OUTPUT_TOKENS = 200


@dataclass(frozen=True)
class CostBreakdown:
    pro_input_tokens: int
    pro_output_tokens: int
    flash_input_tokens: int
    flash_output_tokens: int
    # M11: cached-input buckets. ``pro_input_tokens`` / ``flash_input_tokens``
    # contain only the *uncached* portion; the cached portion is billed at
    # the discounted rate via these dedicated fields. ``actual_job_cost``
    # populates these from ``cached_input_tokens`` on each pass.
    pro_cached_input_tokens: int = 0
    flash_cached_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.pro_input_tokens
            + self.pro_output_tokens
            + self.flash_input_tokens
            + self.flash_output_tokens
            + self.pro_cached_input_tokens
            + self.flash_cached_input_tokens
        )

    def usd(self) -> float:
        s = get_settings()
        return (
            self.pro_input_tokens * s.gemini_pro_input_usd_per_m / 1_000_000
            + self.pro_output_tokens * s.gemini_pro_output_usd_per_m / 1_000_000
            + self.flash_input_tokens * s.gemini_flash_input_usd_per_m / 1_000_000
            + self.flash_output_tokens * s.gemini_flash_output_usd_per_m / 1_000_000
            + self.pro_cached_input_tokens
            * s.gemini_pro_cached_input_usd_per_m
            / 1_000_000
            + self.flash_cached_input_tokens
            * s.gemini_flash_cached_input_usd_per_m
            / 1_000_000
        )


def _empty_breakdown() -> CostBreakdown:
    return CostBreakdown(0, 0, 0, 0, 0, 0)


def _add(a: CostBreakdown, b: CostBreakdown) -> CostBreakdown:
    return CostBreakdown(
        pro_input_tokens=a.pro_input_tokens + b.pro_input_tokens,
        pro_output_tokens=a.pro_output_tokens + b.pro_output_tokens,
        flash_input_tokens=a.flash_input_tokens + b.flash_input_tokens,
        flash_output_tokens=a.flash_output_tokens + b.flash_output_tokens,
        pro_cached_input_tokens=a.pro_cached_input_tokens + b.pro_cached_input_tokens,
        flash_cached_input_tokens=a.flash_cached_input_tokens + b.flash_cached_input_tokens,
    )


def _scale(breakdown: CostBreakdown, factor: float) -> CostBreakdown:
    return CostBreakdown(
        pro_input_tokens=int(breakdown.pro_input_tokens * factor),
        pro_output_tokens=int(breakdown.pro_output_tokens * factor),
        flash_input_tokens=int(breakdown.flash_input_tokens * factor),
        flash_output_tokens=int(breakdown.flash_output_tokens * factor),
        pro_cached_input_tokens=int(breakdown.pro_cached_input_tokens * factor),
        flash_cached_input_tokens=int(breakdown.flash_cached_input_tokens * factor),
    )


def source_token_count(structure_path: Path) -> int:
    """Sum tokens across every translatable block in structure.json. Returns 0
    if the file is missing — the estimator will then return zero cost."""
    if not structure_path.exists():
        return 0
    try:
        with structure_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        log.exception("failed to read %s for cost estimate", structure_path)
        return 0
    total = 0
    for page in data.get("pages") or []:
        for block in page.get("blocks") or []:
            text = block.get("text") or ""
            if text:
                total += count_tokens(text)
    return total


def estimate_job_cost(
    session: Session, job_id: str
) -> dict[str, Any]:
    """Forward estimate. Returns three scenarios:
      * ``cost_low`` — critique disabled, translate only
      * ``cost_current`` — given the job's current settings
      * ``cost_high`` — full quality loop (critique + always-revise + judge=1.0)

    Each scenario carries the projected Pro/Flash token breakdown so the UI
    can show *why* one scenario costs more than another."""
    job = session.get(Job, job_id)
    if job is None:
        raise ValueError(f"unknown job {job_id}")

    settings = get_settings()
    structure_path = settings.intermediate_dir / job_id / "structure.json"
    src_tokens = source_token_count(structure_path)

    # Project the chunk count so the suggester/classifier fixed costs are
    # only added once. Effective chunks come from the actual chunking
    # constraints — but we don't run chunking here for speed; estimate via
    # the configured target.
    target = job.chunk_target_tokens_override or settings.chunk_target_tokens
    est_chunks = max(1, (src_tokens // max(1, target)) + (1 if src_tokens % target else 0))

    critique_enabled = (
        bool(job.critique_enabled)
        if job.critique_enabled is not None
        else settings.critique_enabled
    )
    judge_rate = max(0.0, min(1.0, settings.judge_sample_rate))

    fixed = _fixed_cost_breakdown()
    cost_low = _add(fixed, _translate_only(src_tokens))
    cost_current = _add(
        fixed,
        _full_pipeline(
            src_tokens,
            critique_enabled=critique_enabled,
            revise_rate=_REVISE_RATE_DEFAULT,
            judge_rate=judge_rate,
        ),
    )
    cost_high = _add(
        fixed,
        _full_pipeline(
            src_tokens,
            critique_enabled=True,
            revise_rate=1.0,
            judge_rate=1.0,
        ),
    )

    return {
        "source_tokens": src_tokens,
        "estimated_chunks": est_chunks,
        "scenarios": {
            "translate_only": {
                "label": "Translate only (no quality loop)",
                "tokens": cost_low.total_tokens,
                "usd": round(cost_low.usd(), 2),
            },
            "current_settings": {
                "label": (
                    "Current settings"
                    if critique_enabled
                    else "Translate only (critique disabled in settings)"
                ),
                "tokens": cost_current.total_tokens,
                "usd": round(cost_current.usd(), 2),
            },
            "full_loop": {
                "label": "Full quality loop (always revise + always judge)",
                "tokens": cost_high.total_tokens,
                "usd": round(cost_high.usd(), 2),
            },
        },
    }


def actual_job_cost(session: Session, job_id: str) -> dict[str, Any]:
    """Ground-truth receipt: walk every successful pass row + sum by model."""
    settings = get_settings()
    breakdown = _empty_breakdown()
    by_pass_kind: dict[str, dict[str, int]] = {}

    rows = list(
        session.execute(
            select(ChunkTranslationPass)
            .join(TranslationChunk, TranslationChunk.id == ChunkTranslationPass.chunk_id)
            .where(
                TranslationChunk.job_id == job_id,
                ChunkTranslationPass.status == PassStatus.OK,
            )
        ).scalars()
    )

    for row in rows:
        in_t = int(row.input_tokens or 0)
        out_t = int(row.output_tokens or 0)
        cached_t = int(getattr(row, "cached_input_tokens", 0) or 0)
        # The schema enforces cached <= input, but be defensive in case an old
        # SDK over-reports — clamp to the bucket so we never invoice negative
        # uncached tokens.
        cached_t = max(0, min(cached_t, in_t))
        uncached_t = in_t - cached_t
        is_pro = "pro" in (row.model or "").lower()
        if is_pro:
            breakdown = _add(
                breakdown,
                CostBreakdown(
                    pro_input_tokens=uncached_t,
                    pro_output_tokens=out_t,
                    flash_input_tokens=0,
                    flash_output_tokens=0,
                    pro_cached_input_tokens=cached_t,
                    flash_cached_input_tokens=0,
                ),
            )
        else:
            breakdown = _add(
                breakdown,
                CostBreakdown(
                    pro_input_tokens=0,
                    pro_output_tokens=0,
                    flash_input_tokens=uncached_t,
                    flash_output_tokens=out_t,
                    pro_cached_input_tokens=0,
                    flash_cached_input_tokens=cached_t,
                ),
            )
        kind_bucket = by_pass_kind.setdefault(
            row.pass_kind.value,
            {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0, "calls": 0},
        )
        kind_bucket["input_tokens"] += in_t
        kind_bucket["output_tokens"] += out_t
        kind_bucket["cached_input_tokens"] += cached_t
        kind_bucket["calls"] += 1

    return {
        "pro_input_tokens": breakdown.pro_input_tokens,
        "pro_output_tokens": breakdown.pro_output_tokens,
        "flash_input_tokens": breakdown.flash_input_tokens,
        "flash_output_tokens": breakdown.flash_output_tokens,
        "pro_cached_input_tokens": breakdown.pro_cached_input_tokens,
        "flash_cached_input_tokens": breakdown.flash_cached_input_tokens,
        "total_tokens": breakdown.total_tokens,
        "usd": round(breakdown.usd(), 4),
        "by_pass_kind": by_pass_kind,
        "rates": {
            "gemini_pro_input_usd_per_m": settings.gemini_pro_input_usd_per_m,
            "gemini_pro_output_usd_per_m": settings.gemini_pro_output_usd_per_m,
            "gemini_flash_input_usd_per_m": settings.gemini_flash_input_usd_per_m,
            "gemini_flash_output_usd_per_m": settings.gemini_flash_output_usd_per_m,
            "gemini_pro_cached_input_usd_per_m": settings.gemini_pro_cached_input_usd_per_m,
            "gemini_flash_cached_input_usd_per_m": settings.gemini_flash_cached_input_usd_per_m,
        },
    }


# ----- internals -----


def _fixed_cost_breakdown() -> CostBreakdown:
    """Suggester + classifier are one-shots regardless of book size, both on
    Flash. Sums to a fraction of a cent — included for completeness."""
    return CostBreakdown(
        pro_input_tokens=0,
        pro_output_tokens=0,
        flash_input_tokens=_SUGGESTER_INPUT_TOKENS + _CLASSIFIER_INPUT_TOKENS,
        flash_output_tokens=_SUGGESTER_OUTPUT_TOKENS + _CLASSIFIER_OUTPUT_TOKENS,
    )


def _translate_only(src_tokens: int) -> CostBreakdown:
    return CostBreakdown(
        pro_input_tokens=int(src_tokens * _TRANSLATE_INPUT_PER_SRC),
        pro_output_tokens=int(src_tokens * _TRANSLATE_OUTPUT_PER_SRC),
        flash_input_tokens=0,
        flash_output_tokens=0,
    )


def _full_pipeline(
    src_tokens: int,
    *,
    critique_enabled: bool,
    revise_rate: float,
    judge_rate: float,
) -> CostBreakdown:
    bd = _translate_only(src_tokens)

    if not critique_enabled:
        return bd

    # Critique runs on every chunk → flash.
    bd = _add(
        bd,
        CostBreakdown(
            pro_input_tokens=0,
            pro_output_tokens=0,
            flash_input_tokens=int(src_tokens * _CRITIQUE_INPUT_PER_SRC),
            flash_output_tokens=int(src_tokens * _CRITIQUE_OUTPUT_PER_SRC),
        ),
    )
    # Revise → pro, gated.
    revise = CostBreakdown(
        pro_input_tokens=int(src_tokens * _REVISE_INPUT_PER_SRC),
        pro_output_tokens=int(src_tokens * _REVISE_OUTPUT_PER_SRC),
        flash_input_tokens=0,
        flash_output_tokens=0,
    )
    bd = _add(bd, _scale(revise, max(0.0, min(1.0, revise_rate))))
    # Judge → flash, sampled. Effective rate adds the always-judge cases
    # (chunk 0 + every revised chunk); approximate as max(judge_rate, revise_rate).
    judge = CostBreakdown(
        pro_input_tokens=0,
        pro_output_tokens=0,
        flash_input_tokens=int(src_tokens * _JUDGE_INPUT_PER_SRC),
        flash_output_tokens=int(src_tokens * _JUDGE_OUTPUT_PER_SRC),
    )
    effective_judge = max(judge_rate, revise_rate)
    bd = _add(bd, _scale(judge, max(0.0, min(1.0, effective_judge))))
    return bd

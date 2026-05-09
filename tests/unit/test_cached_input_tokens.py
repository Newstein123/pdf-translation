"""Cached-token plumbing (M11 / Track A).

Exercises the new ``cached_input_tokens`` field across:
  * GeminiJsonResponse extraction
  * ChunkTranslationResult / CritiqueResult propagation
  * cost_estimator.actual_job_cost discounted-rate billing
"""
from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import (
    Category,
    ChunkStatus,
    ChunkTranslationPass,
    Job,
    JobStatus,
    PassKind,
    PassStatus,
    TranslationChunk,
)
from app.services.gemini import _extract_token_counts
from app.services.cost_estimator import actual_job_cost


# ---------------------------------------------------------------------------
# Gemini token extraction
# ---------------------------------------------------------------------------


class _FakeUsageMeta:
    def __init__(self, prompt: int, output: int, cached: int = 0, total: int | None = None):
        self.prompt_token_count = prompt
        self.candidates_token_count = output
        self.cached_content_token_count = cached
        self.total_token_count = total if total is not None else prompt + output


class _FakeResponse:
    def __init__(self, meta: _FakeUsageMeta | None):
        self.usage_metadata = meta


def test_extract_token_counts_parses_cached_field() -> None:
    resp = _FakeResponse(_FakeUsageMeta(prompt=1000, output=500, cached=400))
    prompt, output, total, cached = _extract_token_counts(resp)
    assert prompt == 1000
    assert output == 500
    assert total == 1500
    assert cached == 400


def test_extract_token_counts_clamps_cached_to_prompt() -> None:
    # A misbehaving SDK returning cached > prompt would otherwise push
    # uncached negative — clamp prevents that.
    resp = _FakeResponse(_FakeUsageMeta(prompt=100, output=50, cached=999))
    _, _, _, cached = _extract_token_counts(resp)
    assert cached == 100


def test_extract_token_counts_handles_missing_meta() -> None:
    resp = _FakeResponse(None)
    assert _extract_token_counts(resp) == (0, 0, 0, 0)


def test_extract_token_counts_handles_missing_cached_field() -> None:
    # Older SDK responses without cached_content_token_count.
    class _BareMeta:
        prompt_token_count = 200
        candidates_token_count = 100
        total_token_count = 300

    prompt, output, total, cached = _extract_token_counts(_FakeResponse(_BareMeta()))
    assert (prompt, output, total) == (200, 100, 300)
    assert cached == 0


# ---------------------------------------------------------------------------
# actual_job_cost discounts cached input
# ---------------------------------------------------------------------------


@pytest.fixture
def session(db_engine) -> Session:
    with Session(bind=db_engine) as s:
        yield s
        s.rollback()


def _seed_job_with_pass(
    session: Session,
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int,
) -> str:
    job = Job(
        original_filename="t.pdf",
        file_path="/tmp/t.pdf",
        file_size_bytes=1,
        status=JobStatus.TRANSLATING,
        category=Category.NOVEL,
    )
    session.add(job)
    session.flush()
    chunk = TranslationChunk(
        job_id=job.id,
        chunk_index=0,
        page_range_start=1,
        page_range_end=1,
        source_text="hi",
        token_count=2,
        status=ChunkStatus.COMPLETED,
    )
    session.add(chunk)
    session.flush()
    session.add(
        ChunkTranslationPass(
            chunk_id=chunk.id,
            pass_number=1,
            pass_kind=PassKind.TRANSLATE,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            status=PassStatus.OK,
        )
    )
    session.commit()
    return job.id


def test_actual_job_cost_splits_cached_and_uncached_buckets(session: Session) -> None:
    job_id = _seed_job_with_pass(
        session,
        model="gemini-2.5-pro",
        input_tokens=1000,
        output_tokens=500,
        cached_input_tokens=400,
    )
    s = get_settings()
    report = actual_job_cost(session, job_id)

    assert report["pro_input_tokens"] == 600  # uncached portion only
    assert report["pro_cached_input_tokens"] == 400
    assert report["flash_input_tokens"] == 0
    assert report["pro_output_tokens"] == 500
    # USD: 600*1.25e-6 + 400*0.31e-6 + 500*10e-6 ≈ 0.000750 + 0.000124 + 0.005
    expected_usd = round(
        600 * s.gemini_pro_input_usd_per_m / 1_000_000
        + 400 * s.gemini_pro_cached_input_usd_per_m / 1_000_000
        + 500 * s.gemini_pro_output_usd_per_m / 1_000_000,
        4,
    )
    assert report["usd"] == pytest.approx(expected_usd, abs=1e-4)


def test_actual_job_cost_zero_cached_matches_legacy_behavior(session: Session) -> None:
    job_id = _seed_job_with_pass(
        session,
        model="gemini-2.5-flash",
        input_tokens=500,
        output_tokens=200,
        cached_input_tokens=0,
    )
    report = actual_job_cost(session, job_id)
    assert report["flash_input_tokens"] == 500
    assert report["flash_cached_input_tokens"] == 0


def test_actual_job_cost_pass_kind_breakdown_includes_cached(session: Session) -> None:
    job_id = _seed_job_with_pass(
        session,
        model="gemini-2.5-pro",
        input_tokens=800,
        output_tokens=300,
        cached_input_tokens=500,
    )
    report = actual_job_cost(session, job_id)
    bucket = report["by_pass_kind"]["TRANSLATE"]
    assert bucket["input_tokens"] == 800
    assert bucket["cached_input_tokens"] == 500
    assert bucket["output_tokens"] == 300
    assert bucket["calls"] == 1

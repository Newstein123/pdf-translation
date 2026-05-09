"""Unit tests for the M9 judge sampling gate."""
from __future__ import annotations

import pytest

from app.config import get_settings


@pytest.fixture
def cleared_settings():
    """Drop the cached settings before/after so each test mutates env vars
    in isolation."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_chunk_zero_always_judged(monkeypatch, cleared_settings) -> None:
    monkeypatch.setenv("JUDGE_SAMPLE_RATE", "0.0")
    get_settings.cache_clear()
    from app.workers.tasks import _should_judge_chunk
    assert _should_judge_chunk(0, revise_happened=False) is True


def test_revised_chunk_always_judged(monkeypatch, cleared_settings) -> None:
    monkeypatch.setenv("JUDGE_SAMPLE_RATE", "0.0")
    get_settings.cache_clear()
    from app.workers.tasks import _should_judge_chunk
    assert _should_judge_chunk(7, revise_happened=True) is True


def test_zero_rate_skips_unrevised_non_zero_chunks(monkeypatch, cleared_settings) -> None:
    monkeypatch.setenv("JUDGE_SAMPLE_RATE", "0.0")
    get_settings.cache_clear()
    from app.workers.tasks import _should_judge_chunk
    assert _should_judge_chunk(5, revise_happened=False) is False


def test_full_rate_judges_every_chunk(monkeypatch, cleared_settings) -> None:
    monkeypatch.setenv("JUDGE_SAMPLE_RATE", "1.0")
    get_settings.cache_clear()
    from app.workers.tasks import _should_judge_chunk
    assert all(
        _should_judge_chunk(i, revise_happened=False) for i in range(20)
    )


def test_quarter_rate_strides_every_four(monkeypatch, cleared_settings) -> None:
    monkeypatch.setenv("JUDGE_SAMPLE_RATE", "0.25")
    get_settings.cache_clear()
    from app.workers.tasks import _should_judge_chunk
    judged = [i for i in range(20) if _should_judge_chunk(i, revise_happened=False)]
    assert judged == [0, 4, 8, 12, 16]

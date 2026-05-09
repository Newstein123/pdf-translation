"""Unit tests for the structured running-context merger (M8)."""
from __future__ import annotations

from app.services.running_context import (
    empty_context,
    merge_running_context,
    normalize_context,
)


def test_normalize_handles_garbage_input() -> None:
    assert normalize_context(None) == empty_context()
    assert normalize_context("not a dict") == empty_context()
    assert normalize_context({"characters": "wrong type"})["characters"] == []


def test_merge_carries_forward_when_update_empty() -> None:
    prev = {
        "characters": [{"source": "Alice", "burmese": "အဲလစ်"}],
        "tone": "wistful",
        "rolling_summary": "Alice journeys.",
    }
    merged = merge_running_context(prev, {})
    # Carry-forward: prior entries survive even though update is empty.
    assert merged["characters"] == [{"source": "Alice", "burmese": "အဲလစ်"}]
    assert merged["tone"] == "wistful"
    assert merged["rolling_summary"] == "Alice journeys."


def test_merge_appends_new_characters() -> None:
    prev = {"characters": [{"source": "Alice", "burmese": "အဲလစ်"}]}
    update = {"characters": [{"source": "Bob", "burmese": "ဘော်ဘ်"}]}
    merged = merge_running_context(prev, update)
    sources = [c["source"] for c in merged["characters"]]
    assert sources == ["Alice", "Bob"]


def test_merge_overwrites_on_key_collision() -> None:
    """If the model updates a character's role/social_rank, the new entry wins."""
    prev = {"characters": [{"source": "Alice", "burmese": "အဲလစ်", "role": "stranger"}]}
    update = {"characters": [{"source": "Alice", "burmese": "အဲလစ်", "role": "protagonist"}]}
    merged = merge_running_context(prev, update)
    assert len(merged["characters"]) == 1
    assert merged["characters"][0]["role"] == "protagonist"


def test_merge_caps_characters_at_30() -> None:
    prev_chars = [{"source": f"Char{i}"} for i in range(40)]
    merged = merge_running_context({"characters": prev_chars}, {})
    assert len(merged["characters"]) == 30
    # FIFO eviction: oldest entries dropped first.
    assert merged["characters"][0]["source"] == "Char10"
    assert merged["characters"][-1]["source"] == "Char39"


def test_merge_relationships_dedupe_unordered_pair() -> None:
    """{a: Alice, b: Bob} and {a: Bob, b: Alice} are the same relationship."""
    prev = {"relationships": [{"a": "Alice", "b": "Bob", "kind": "mentor"}]}
    update = {"relationships": [{"a": "Bob", "b": "Alice", "kind": "rival"}]}
    merged = merge_running_context(prev, update)
    assert len(merged["relationships"]) == 1
    # Newer wins on conflict.
    assert merged["relationships"][0]["kind"] == "rival"


def test_merge_scalar_keeps_prior_when_update_empty_string() -> None:
    prev = {"tone": "wistful", "pov": "third"}
    update = {"tone": "", "pov": "first"}
    merged = merge_running_context(prev, update)
    assert merged["tone"] == "wistful"  # carried forward
    assert merged["pov"] == "first"  # overwritten


def test_merge_terminology_dedupe_case_insensitive() -> None:
    prev = {"terminology": [{"source": "Cell", "burmese": "ဆဲလ်"}]}
    update = {"terminology": [{"source": "cell", "burmese": "ဆဲလ် (cell)"}]}
    merged = merge_running_context(prev, update)
    assert len(merged["terminology"]) == 1
    # Newer renderings win.
    assert merged["terminology"][0]["burmese"] == "ဆဲလ် (cell)"

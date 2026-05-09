"""Tests for the page-range parser used by the M15 skip-pages feature."""
from __future__ import annotations

import pytest

from app.utils.page_ranges import format_page_ranges, parse_page_ranges


# ---------------------------------------------------------------------------
# parse_page_ranges
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty_set() -> None:
    assert parse_page_ranges(None) == set()
    assert parse_page_ranges("") == set()
    assert parse_page_ranges("   ") == set()


def test_single_number() -> None:
    assert parse_page_ranges("5") == {5}


def test_simple_range() -> None:
    assert parse_page_ranges("1-3") == {1, 2, 3}


def test_mixed_singles_and_ranges() -> None:
    assert parse_page_ranges("1-3, 5, 8-10") == {1, 2, 3, 5, 8, 9, 10}


def test_extra_whitespace_is_tolerated() -> None:
    assert parse_page_ranges("  1 - 3 ,  5  ;  8-10 ") == {1, 2, 3, 5, 8, 9, 10}


def test_mixed_dash_characters() -> None:
    # ASCII hyphen, en-dash, em-dash all accepted.
    assert parse_page_ranges("1–3, 5—7, 9-9") == {1, 2, 3, 5, 6, 7, 9}


def test_reversed_range_normalizes() -> None:
    assert parse_page_ranges("5-3") == {3, 4, 5}


def test_duplicates_and_overlaps_dedupe() -> None:
    assert parse_page_ranges("1-3, 2-4, 4, 4") == {1, 2, 3, 4}


def test_zero_rejected() -> None:
    with pytest.raises(ValueError, match="1-indexed"):
        parse_page_ranges("0")


def test_negative_rejected() -> None:
    with pytest.raises(ValueError, match="1-indexed"):
        parse_page_ranges("-3")


def test_non_numeric_rejected() -> None:
    with pytest.raises(ValueError, match="not a number"):
        parse_page_ranges("foo")


def test_too_many_dashes_rejected() -> None:
    with pytest.raises(ValueError, match="too many"):
        parse_page_ranges("1-2-3")


# ---------------------------------------------------------------------------
# format_page_ranges (round-trip)
# ---------------------------------------------------------------------------


def test_format_empty() -> None:
    assert format_page_ranges(set()) == ""


def test_format_collapses_consecutive() -> None:
    assert format_page_ranges({1, 2, 3, 5, 8, 9, 10}) == "1-3, 5, 8-10"


def test_format_keeps_singletons_as_bare_numbers() -> None:
    assert format_page_ranges({5}) == "5"
    assert format_page_ranges({3, 5, 7}) == "3, 5, 7"


def test_round_trip() -> None:
    spec = "1-3, 5, 8-10, 15"
    assert format_page_ranges(parse_page_ranges(spec)) == spec

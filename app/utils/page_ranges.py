"""Parse user-friendly page-range strings into a set of page numbers.

Used by the "skip these pages" feature (M15): user enters something like
``"1-3, 5, 8-10"`` in the AWAITING_CONFIRMATION block, the worker calls
``parse_page_ranges()`` to turn that into ``{1, 2, 3, 5, 8, 9, 10}``,
and downstream code filters blocks / pages by membership in that set.

Tolerant of:
  * Mixed separators (``,``, ``;``, whitespace, line breaks)
  * Mixed dash characters (``-``, ``–``, ``—``)
  * Empty / None input → empty set
  * Reversed ranges like ``"5-3"`` (treated as ``3-5``)
  * Duplicates and overlaps (deduped automatically by set semantics)

Rejects:
  * Non-numeric tokens → ``ValueError``
  * Negative numbers → ``ValueError``
  * Zero → ``ValueError`` (pages are 1-indexed)
"""
from __future__ import annotations

import re

# Tokens are separated by commas, semicolons, or whitespace. We split on any
# of those then trim each token independently.
_TOKEN_SEPARATORS_RE = re.compile(r"[,;\s]+")
# Range delimiters: ASCII hyphen, en-dash, em-dash. Allow optional whitespace
# around them for forgiveness ("5 - 10").
_RANGE_DELIM_RE = re.compile(r"\s*[-–—]\s*")


def parse_page_ranges(spec: str | None) -> set[int]:
    """Parse a page-range string into a set of 1-indexed page numbers.

    Empty / None input returns an empty set. Raises ``ValueError`` with a
    user-readable message on malformed input — the API endpoint surfaces
    this as a 400 so the user sees what went wrong.
    """
    if spec is None:
        return set()
    cleaned = spec.strip()
    if not cleaned:
        return set()

    # Normalize whitespace AROUND dashes so "1 - 3" doesn't get split
    # into separate tokens by the whitespace-tolerant separator regex.
    cleaned = re.sub(r"\s*([-–—])\s*", r"\1", cleaned)
    # Reject leading dash on a token — that would parse as an empty
    # left-side range bound, but it's almost always a user typo for a
    # negative number, which we don't accept.
    if cleaned.startswith(("-", "–", "—")):
        raise ValueError(
            "page numbers are 1-indexed; leading dash is not valid"
        )

    out: set[int] = set()
    for token in _TOKEN_SEPARATORS_RE.split(cleaned):
        if not token:
            continue
        parts = _RANGE_DELIM_RE.split(token)
        if len(parts) == 1:
            # Bare page number.
            n = _parse_int(parts[0], context=token)
            out.add(n)
        elif len(parts) == 2:
            lo = _parse_int(parts[0], context=token)
            hi = _parse_int(parts[1], context=token)
            if lo > hi:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            raise ValueError(
                f"page range {token!r} has too many '-' delimiters"
            )
    return out


def _parse_int(s: str, *, context: str) -> int:
    s = s.strip()
    if not s:
        raise ValueError(f"empty number in page range {context!r}")
    try:
        n = int(s)
    except ValueError as e:
        raise ValueError(
            f"not a number: {s!r} in page range {context!r}"
        ) from e
    if n <= 0:
        raise ValueError(
            f"page numbers are 1-indexed; {n} is not valid (in {context!r})"
        )
    return n


def format_page_ranges(pages: set[int]) -> str:
    """Inverse of ``parse_page_ranges``: collapse a set of page numbers
    back into a compact range string. Useful for showing the user what
    they really asked for after parsing — and for tests.
    """
    if not pages:
        return ""
    sorted_pages = sorted(pages)
    runs: list[tuple[int, int]] = []
    start = end = sorted_pages[0]
    for n in sorted_pages[1:]:
        if n == end + 1:
            end = n
        else:
            runs.append((start, end))
            start = end = n
    runs.append((start, end))
    return ", ".join(
        str(lo) if lo == hi else f"{lo}-{hi}" for lo, hi in runs
    )

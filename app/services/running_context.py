"""Structured running-context memory across chunks (M8).

The translator carried a 2–3 sentence rolling summary forward as the only
inter-chunk memory before M8. This module replaces that with a structured
JSON snapshot containing the characters, relationships, terminology, tone,
and open plot threads — exactly the state a careful human translator
mentally tracks across a long book.

Each translated chunk persists its full snapshot in
``translation_chunks.running_context_json``; the next chunk reads that
snapshot, ships it to the model, and the model returns an updated snapshot
which we merge (server-side) under fixed caps so the context can't grow
unboundedly.
"""
from __future__ import annotations

from typing import Any

# Per-list caps. Eviction is FIFO by appearance order. Caps are deliberate:
# a 300-page novel might mention 60 minor characters but the *active cast*
# the translator needs to consult is much smaller. Keep the snapshot small
# enough that shipping it with every chunk doesn't blow the context window.
_CAP_CHARACTERS = 30
_CAP_RELATIONSHIPS = 50
_CAP_OPEN_THREADS = 20
_CAP_TERMINOLOGY = 50

# The keys that identify a "same item" across chunks. Used during merge for
# dedupe — case-insensitive on the source string.
_CHARACTER_KEY = "source"
_TERM_KEY = "source"


def empty_context() -> dict[str, Any]:
    """Initial snapshot for chunk 0."""
    return {
        "characters": [],
        "relationships": [],
        "pov": "",
        "tone": "",
        "mood": "",
        "open_threads": [],
        "terminology": [],
        "rolling_summary": "",
    }


def normalize_context(raw: Any) -> dict[str, Any]:
    """Coerce a model-emitted (or DB-stored) context object into the canonical
    shape. Missing or wrong-typed fields fall back to empty defaults. Defensive
    against malformed model output."""
    if not isinstance(raw, dict):
        return empty_context()
    out = empty_context()
    if isinstance(raw.get("characters"), list):
        out["characters"] = [c for c in raw["characters"] if isinstance(c, dict)]
    if isinstance(raw.get("relationships"), list):
        out["relationships"] = [r for r in raw["relationships"] if isinstance(r, dict)]
    if isinstance(raw.get("open_threads"), list):
        out["open_threads"] = [t for t in raw["open_threads"] if isinstance(t, dict)]
    if isinstance(raw.get("terminology"), list):
        out["terminology"] = [t for t in raw["terminology"] if isinstance(t, dict)]
    for scalar_key in ("pov", "tone", "mood", "rolling_summary"):
        v = raw.get(scalar_key)
        if isinstance(v, str):
            out[scalar_key] = v
    return out


def merge_running_context(
    previous: dict[str, Any] | None, update: dict[str, Any] | None
) -> dict[str, Any]:
    """Combine the previous chunk's snapshot with the model's update.

    Strategy:
    - Lists merge by key (case-insensitive) — newer entry wins on conflict so
      the model can correct its own earlier renderings.
    - Order preserves appearance: previously-seen items keep their slot,
      new items append at the tail.
    - Caps are enforced FIFO once exceeded; the *oldest* entries are evicted.
    - Scalars (pov, tone, mood, rolling_summary) take the update's value
      whenever the update provides a non-empty one; otherwise carry forward.
    """
    prev = normalize_context(previous)
    new = normalize_context(update)

    merged = empty_context()

    merged["characters"] = _merge_keyed_list(
        prev["characters"], new["characters"], key=_CHARACTER_KEY, cap=_CAP_CHARACTERS
    )
    merged["terminology"] = _merge_keyed_list(
        prev["terminology"], new["terminology"], key=_TERM_KEY, cap=_CAP_TERMINOLOGY
    )
    # Relationships are keyed by an unordered pair (a, b). We canonicalize.
    merged["relationships"] = _merge_relationships(
        prev["relationships"], new["relationships"], cap=_CAP_RELATIONSHIPS
    )
    # Open threads: identify by ``id`` if present, else by ``summary``.
    merged["open_threads"] = _merge_keyed_list(
        prev["open_threads"],
        new["open_threads"],
        key=lambda t: (t.get("id") or t.get("summary") or "").strip().lower(),
        cap=_CAP_OPEN_THREADS,
    )

    for scalar_key in ("pov", "tone", "mood", "rolling_summary"):
        merged[scalar_key] = new[scalar_key] or prev[scalar_key]

    return merged


def _merge_keyed_list(
    prev_items: list[dict[str, Any]],
    new_items: list[dict[str, Any]],
    *,
    key,
    cap: int,
) -> list[dict[str, Any]]:
    """Merge two keyed lists. ``key`` may be a string field name or a callable.

    New items override on key collision (so the model can self-correct);
    items only in ``prev_items`` carry forward. FIFO eviction once over cap.
    """
    key_fn = (lambda item, k=key: str(item.get(k) or "").strip().lower()) if isinstance(key, str) else key

    by_key: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in prev_items:
        k = key_fn(item)
        if not k:
            continue
        if k not in by_key:
            order.append(k)
        by_key[k] = item
    for item in new_items:
        k = key_fn(item)
        if not k:
            continue
        if k not in by_key:
            order.append(k)
        by_key[k] = item  # newer wins

    if len(order) > cap:
        # Drop the oldest first.
        order = order[-cap:]

    return [by_key[k] for k in order]


def _merge_relationships(
    prev_items: list[dict[str, Any]],
    new_items: list[dict[str, Any]],
    *,
    cap: int,
) -> list[dict[str, Any]]:
    def _key(rel: dict[str, Any]) -> str:
        a = (rel.get("a") or "").strip().lower()
        b = (rel.get("b") or "").strip().lower()
        if not a or not b:
            return ""
        # Order-independent: (a, b) and (b, a) share a key.
        lo, hi = sorted([a, b])
        return f"{lo}|{hi}"

    return _merge_keyed_list(prev_items, new_items, key=_key, cap=cap)


def serialize_for_prompt(context: dict[str, Any]) -> dict[str, Any]:
    """Slim the snapshot to only the fields the model needs to *read*.

    Avoids shipping internal-only fields. Currently a passthrough on a copy,
    but kept as an explicit boundary so we can prune fields later without
    touching the prompt schema."""
    safe = normalize_context(context)
    return safe

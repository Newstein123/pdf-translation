"""Reference library: curated source↔Burmese aligned passages.

The library is the moat between "AI prose" and "feels like a published
translation." For each chunk we retrieve the top-K most similar reference
passages (filtered by genre + register + mode) and inject them as few-shot
examples in the translate / critique / revise prompts.

This module is the Track-B foundation skeleton: it covers the **shape** of
the library — ingest, embedding, search — without committing to a specific
embedding backend. Once we pick one (Gemini ``text-embedding-004`` is the
likely default), the only part that changes is ``_compute_embedding``.

Public surface:

  * ``ingest_passage(...)`` — add a single curated passage to the library.
  * ``ingest_aligned_pairs(...)`` — bulk ingest from a list of dicts.
  * ``search(query_text, ...)`` — cosine-similarity retrieval with optional
    genre / register / mode filters; returns top-K with similarity scores.
  * ``compose_examples_block(passages)`` — render retrieved passages as a
    few-shot examples block for prompt insertion.

Storage strategy:
  Embeddings live in ``ReferencePassage.embedding_json`` as a JSON array of
  floats. SQLite has no native vector type and we don't want to require
  pgvector for local dev. Cosine similarity is computed in Python — fine
  for the corpus size we're targeting (~200 passages). Migration to
  pgvector when the corpus grows past ~5k entries is a one-shot copy.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import ReferenceMode, ReferencePassage
from app.services.gemini import GeminiUnavailable

log = logging.getLogger(__name__)

_DEFAULT_TOP_K = 5
_DEFAULT_MIN_SIMILARITY = 0.55  # below this, retrieval returns nothing
# Gemini's recommended retrieval embedding model. Keep configurable so we can
# swap to a newer model without a migration; persisted as ``embedding_model``
# on each row so we can detect mismatches and re-embed selectively.
_DEFAULT_EMBEDDING_MODEL = "text-embedding-004"


@dataclass(frozen=True)
class RetrievedPassage:
    """One ranked result from ``search``."""

    passage: ReferencePassage
    similarity: float


class ReferenceLibraryError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def ingest_passage(
    session: Session,
    *,
    source_text: str | None,
    target_text: str,
    source_lang: str = "en",
    target_lang: str = "my",
    genre: str | None = None,
    register: str | None = None,
    mode: ReferenceMode | str = ReferenceMode.OTHER,
    book_title: str | None = None,
    translator: str | None = None,
    notes: str | None = None,
    embedding: list[float] | None = None,
    embedding_model: str | None = None,
) -> ReferencePassage:
    """Insert a single curated passage.

    Two flavors are supported:

      * **Aligned pair** — pass both ``source_text`` and ``target_text``.
        Embedding is computed over the SOURCE so retrieval ranks by
        "this English passage looks like that English passage."
      * **Target-only style exemplar** — pass ``source_text=None`` (or
        empty) and ``target_text`` only. Embedding is computed over the
        TARGET text so retrieval ranks by "this Burmese passage looks
        like that Burmese passage." Used when the corpus is a curated
        body of published Burmese translations whose source originals we
        don't have aligned. ``translator`` + ``mode`` filters carry most
        of the retrieval signal in this mode.

    Validation:
      * target_text is always required.
      * mode coerces from string for ergonomic CLI/JSON ingestion.
    """
    if not target_text or not target_text.strip():
        raise ReferenceLibraryError("target_text is required")

    if isinstance(mode, str):
        try:
            mode = ReferenceMode(mode.upper())
        except ValueError:
            mode = ReferenceMode.OTHER

    has_source = bool(source_text and source_text.strip())
    text_to_embed = (source_text if has_source else target_text)

    if embedding is None:
        embedding = _compute_embedding(text_to_embed)
        embedding_model = embedding_model or _DEFAULT_EMBEDDING_MODEL

    passage = ReferencePassage(
        source_text=source_text.strip() if has_source else "",
        target_text=target_text.strip(),
        source_lang=source_lang,
        target_lang=target_lang,
        genre=genre,
        register=register,
        mode=mode,
        book_title=book_title,
        translator=translator,
        notes=notes,
        embedding_json=embedding,
        embedding_model=embedding_model,
        active=True,
    )
    session.add(passage)
    session.flush()
    return passage


def ingest_aligned_pairs(
    session: Session,
    pairs: Iterable[dict[str, Any]],
    *,
    default_genre: str | None = None,
    default_register: str | None = None,
    default_mode: ReferenceMode | str = ReferenceMode.OTHER,
) -> int:
    """Bulk-ingest. Each pair is a dict with at least ``source`` and ``target``;
    optional keys: ``genre``, ``register``, ``mode``, ``book_title``,
    ``translator``, ``notes``.

    Returns the number of passages added. Skips invalid entries with a log
    line — bulk imports shouldn't fail the whole batch on one bad row."""
    added = 0
    for pair in pairs:
        try:
            ingest_passage(
                session,
                source_text=pair["source"],
                target_text=pair["target"],
                genre=pair.get("genre", default_genre),
                register=pair.get("register", default_register),
                mode=pair.get("mode", default_mode),
                book_title=pair.get("book_title"),
                translator=pair.get("translator"),
                notes=pair.get("notes"),
            )
            added += 1
        except Exception:
            log.exception("skipping malformed reference pair: %r", pair)
    return added


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def search(
    session: Session,
    query_text: str,
    *,
    top_k: int = _DEFAULT_TOP_K,
    min_similarity: float = _DEFAULT_MIN_SIMILARITY,
    genre: str | None = None,
    register: str | None = None,
    mode: ReferenceMode | str | None = None,
    translator: str | None = None,
    target_lang: str = "my",
) -> list[RetrievedPassage]:
    """Cosine-similarity retrieval over active passages.

    Filters tighten the candidate pool *before* similarity scoring so we
    don't waste cycles ranking irrelevant rows. ``min_similarity`` blocks
    low-quality matches that would mislead the translator more than help.

    The ``translator`` filter is the primary lever for the target-only
    voice corpus: pick a translator (e.g. "Banmaw Tin Aung") and pull
    exemplars in their voice, optionally further narrowed by mode (dialogue
    vs. narration vs. exposition).

    Returns up to ``top_k`` results, sorted by similarity descending.
    Returns ``[]`` if the embedding service is unavailable — never raises
    for retrieval failures because translation must keep working without
    references."""
    if not query_text.strip():
        return []

    try:
        query_emb = _compute_embedding(query_text)
    except GeminiUnavailable:
        log.warning("reference-library search skipped: embedding service unavailable")
        return []
    except Exception:
        log.exception("reference-library embedding failed")
        return []

    stmt = select(ReferencePassage).where(
        ReferencePassage.active.is_(True),
        ReferencePassage.target_lang == target_lang,
    )
    if genre:
        stmt = stmt.where(ReferencePassage.genre == genre)
    if register:
        stmt = stmt.where(ReferencePassage.register == register)
    if translator:
        stmt = stmt.where(ReferencePassage.translator == translator)
    if mode is not None:
        if isinstance(mode, str):
            try:
                mode = ReferenceMode(mode.upper())
            except ValueError:
                mode = None
        if mode is not None:
            stmt = stmt.where(ReferencePassage.mode == mode)

    rows = list(session.execute(stmt).scalars())
    if not rows:
        return []

    scored: list[RetrievedPassage] = []
    for row in rows:
        emb = row.embedding_json
        if not isinstance(emb, list) or not emb:
            continue
        sim = _cosine(query_emb, emb)
        if sim >= min_similarity:
            scored.append(RetrievedPassage(passage=row, similarity=sim))

    scored.sort(key=lambda r: r.similarity, reverse=True)
    return scored[:top_k]


def list_translators(session: Session) -> list[str]:
    """Distinct, alphabetized list of translators with at least one active
    passage. Used by the Book Brief stage to constrain the
    ``recommended_translator_voice`` field to translators we actually have
    a corpus for."""
    rows = list(
        session.execute(
            select(ReferencePassage.translator)
            .where(
                ReferencePassage.active.is_(True),
                ReferencePassage.translator.is_not(None),
            )
            .distinct()
            .order_by(ReferencePassage.translator)
        ).scalars()
    )
    return [r for r in rows if r]


def compose_examples_block(retrieved: list[RetrievedPassage]) -> str:
    """Render retrieved passages as a few-shot examples block.

    Format mirrors the existing ``_render_few_shot_examples`` in
    ``translation.py`` so the prompt structure is consistent between
    profile-defined examples and library-retrieved ones."""
    if not retrieved:
        return ""

    parts: list[str] = [
        "REFERENCE EXAMPLES (from published Burmese translations — match this register and approach):",
        "============================================================",
    ]
    sep = "\n\n=========\n\n"
    blocks: list[str] = []
    for i, r in enumerate(retrieved, start=1):
        p = r.passage
        attribution = []
        if p.book_title:
            attribution.append(p.book_title)
        if p.translator:
            attribution.append(f"trans. {p.translator}")
        attr_str = " — " + ", ".join(attribution) if attribution else ""
        block = (
            f"EXAMPLE {i} (similarity {r.similarity:.2f}{attr_str})\n"
            f"SOURCE:\n{p.source_text}\n\n"
            f"TRANSLATION:\n{p.target_text}"
        )
        blocks.append(block)
    parts.append(sep.join(blocks))
    parts.append("============================================================")
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _compute_embedding(text: str) -> list[float]:
    """Compute an embedding via Gemini's embedding API.

    Implementation kept thin so we can swap providers later. Returns a
    plain list of floats; SQLite stores it as JSON. Raises
    ``GeminiUnavailable`` on any failure (caller decides whether to fall
    back).
    """
    settings = get_settings()
    if not settings.gemini_api_key:
        raise GeminiUnavailable("GEMINI_API_KEY is not set; reference embeddings unavailable")

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=settings.gemini_api_key)
        result = client.models.embed_content(
            model=_DEFAULT_EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
    except Exception as e:
        raise GeminiUnavailable(f"embedding failed: {e}") from e

    # SDK shape: result.embeddings is a list of objects with `.values`.
    embeddings = getattr(result, "embeddings", None) or []
    if not embeddings:
        raise GeminiUnavailable("embedding response empty")
    values = getattr(embeddings[0], "values", None)
    if not values:
        raise GeminiUnavailable("embedding response missing values")
    return [float(v) for v in values]


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in pure Python. Returns 0.0 on dimension mismatch
    rather than raising, so a stale-embedding row never breaks retrieval."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))

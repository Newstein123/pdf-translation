"""Unit tests for the reference library cosine retrieval (M11 / Track B).

The embedding API is stubbed out so tests run offline. We exercise the
ranking + filtering logic against passages with hand-set embeddings.
"""
from __future__ import annotations

import math
from typing import Iterable

import pytest

from app.db.models import Base, ReferenceMode, ReferencePassage
from app.services import reference_library


@pytest.fixture
def session(db_engine):
    from sqlalchemy.orm import Session

    with Session(bind=db_engine) as s:
        yield s
        s.rollback()


def _seed(session, *, embedding: list[float], **kwargs) -> ReferencePassage:
    p = ReferencePassage(
        source_text=kwargs.pop("source_text", "src"),
        target_text=kwargs.pop("target_text", "tgt"),
        embedding_json=embedding,
        embedding_model="test-stub",
        active=kwargs.pop("active", True),
        **kwargs,
    )
    session.add(p)
    session.flush()
    return p


def test_search_returns_empty_when_query_blank(session, monkeypatch) -> None:
    monkeypatch.setattr(
        reference_library, "_compute_embedding", lambda _t: [1.0, 0.0]
    )
    assert reference_library.search(session, "   ") == []


def test_search_ranks_by_cosine_similarity(session, monkeypatch) -> None:
    # Seed: closer to query (1,0) than the other.
    near = _seed(session, embedding=[0.99, 0.14], target_lang="my")
    far = _seed(session, embedding=[0.0, 1.0], target_lang="my")

    monkeypatch.setattr(
        reference_library, "_compute_embedding", lambda _t: [1.0, 0.0]
    )
    results = reference_library.search(session, "anything", min_similarity=0.0)
    assert [r.passage.id for r in results] == [near.id, far.id]
    assert results[0].similarity > results[1].similarity


def test_search_filters_inactive(session, monkeypatch) -> None:
    active = _seed(session, embedding=[1.0, 0.0], target_lang="my")
    _seed(session, embedding=[1.0, 0.0], target_lang="my", active=False)

    monkeypatch.setattr(
        reference_library, "_compute_embedding", lambda _t: [1.0, 0.0]
    )
    results = reference_library.search(session, "x", min_similarity=0.0)
    assert [r.passage.id for r in results] == [active.id]


def test_search_filters_genre_register_mode(session, monkeypatch) -> None:
    match = _seed(
        session,
        embedding=[1.0, 0.0],
        genre="literary_fiction",
        register="LITERARY",
        mode=ReferenceMode.NARRATION,
        target_lang="my",
    )
    _seed(
        session,
        embedding=[1.0, 0.0],
        genre="thriller",
        register="GENRE_FICTION",
        mode=ReferenceMode.DIALOGUE,
        target_lang="my",
    )

    monkeypatch.setattr(
        reference_library, "_compute_embedding", lambda _t: [1.0, 0.0]
    )
    results = reference_library.search(
        session,
        "x",
        genre="literary_fiction",
        register="LITERARY",
        mode=ReferenceMode.NARRATION,
        min_similarity=0.0,
    )
    assert [r.passage.id for r in results] == [match.id]


def test_search_drops_below_min_similarity(session, monkeypatch) -> None:
    _seed(session, embedding=[0.0, 1.0], target_lang="my")  # cos=0 vs query (1,0)

    monkeypatch.setattr(
        reference_library, "_compute_embedding", lambda _t: [1.0, 0.0]
    )
    results = reference_library.search(session, "x", min_similarity=0.5)
    assert results == []


def test_search_returns_empty_when_embedding_unavailable(session, monkeypatch) -> None:
    from app.services.gemini import GeminiUnavailable

    def _raise(_t):
        raise GeminiUnavailable("offline")

    monkeypatch.setattr(reference_library, "_compute_embedding", _raise)
    # Don't raise; retrieval failures are non-fatal.
    assert reference_library.search(session, "x") == []


def test_compose_examples_block_renders_attribution() -> None:
    p = ReferencePassage(
        source_text="The fog rolled in over the harbor.",
        target_text="ဆိပ်ကမ်းပေါ်တွင် မြူပွေးနေသည်။",
        book_title="Bleak House",
        translator="U Hla",
        embedding_json=[1.0],
    )
    block = reference_library.compose_examples_block(
        [reference_library.RetrievedPassage(passage=p, similarity=0.91)]
    )
    assert "REFERENCE EXAMPLES" in block
    assert "0.91" in block
    assert "Bleak House" in block
    assert "U Hla" in block
    assert "fog rolled" in block


def test_compose_examples_block_empty_when_no_results() -> None:
    assert reference_library.compose_examples_block([]) == ""


def test_cosine_helpers() -> None:
    cos = reference_library._cosine
    assert cos([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cos([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cos([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]) == pytest.approx(-1.0)
    # Dimension mismatch returns 0 instead of raising.
    assert cos([1.0, 0.0], [1.0]) == 0.0
    # Zero vector → 0, no division-by-zero crash.
    assert cos([0.0, 0.0], [1.0, 0.0]) == 0.0

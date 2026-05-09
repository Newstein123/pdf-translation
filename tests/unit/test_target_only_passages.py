"""Tests for target-only passages and translator-keyed retrieval (M12)."""
from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from app.db.models import ReferenceMode, ReferencePassage
from app.services import reference_library


@pytest.fixture
def session(db_engine):
    with Session(bind=db_engine) as s:
        yield s
        s.rollback()


def test_ingest_passage_accepts_target_only(session, monkeypatch) -> None:
    # Stub the embedding so we don't hit Gemini during tests; the embedding
    # for a target-only passage should be computed over the TARGET text.
    captured: dict[str, str] = {}

    def _fake_embed(text):
        captured["text"] = text
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(reference_library, "_compute_embedding", _fake_embed)

    p = reference_library.ingest_passage(
        session,
        source_text=None,
        target_text="ဤသည်ကား ပုံပြင်တစ်ပုဒ်၏ အစ ဖြစ်သည်။",
        translator="TestTranslator",
        mode=ReferenceMode.NARRATION,
    )
    assert p.source_text == ""
    assert p.translator == "TestTranslator"
    assert p.embedding_json == [0.1, 0.2, 0.3]
    assert "ပုံပြင်" in captured["text"], "embedding should run over target text"


def test_ingest_passage_rejects_missing_target(session) -> None:
    with pytest.raises(reference_library.ReferenceLibraryError):
        reference_library.ingest_passage(
            session,
            source_text="x",
            target_text="",
        )


def test_search_filters_by_translator(session, monkeypatch) -> None:
    a = ReferencePassage(
        source_text="",
        target_text="passage A",
        target_lang="my",
        translator="Author1",
        embedding_json=[1.0, 0.0],
        active=True,
    )
    b = ReferencePassage(
        source_text="",
        target_text="passage B",
        target_lang="my",
        translator="Author2",
        embedding_json=[1.0, 0.0],
        active=True,
    )
    session.add_all([a, b])
    session.flush()

    monkeypatch.setattr(
        reference_library, "_compute_embedding", lambda _t: [1.0, 0.0]
    )
    results = reference_library.search(
        session, "anything", translator="Author1", min_similarity=0.0
    )
    assert [r.passage.translator for r in results] == ["Author1"]


def test_list_translators_returns_distinct_active(session) -> None:
    session.add_all(
        [
            ReferencePassage(
                source_text="", target_text="x", translator="A", embedding_json=[1.0]
            ),
            ReferencePassage(
                source_text="", target_text="x", translator="A", embedding_json=[1.0]
            ),
            ReferencePassage(
                source_text="", target_text="x", translator="B", embedding_json=[1.0]
            ),
            ReferencePassage(
                source_text="",
                target_text="x",
                translator="C",
                embedding_json=[1.0],
                active=False,  # excluded
            ),
            ReferencePassage(
                source_text="", target_text="x", translator=None, embedding_json=[1.0]
            ),  # excluded
        ]
    )
    session.flush()
    assert reference_library.list_translators(session) == ["A", "B"]

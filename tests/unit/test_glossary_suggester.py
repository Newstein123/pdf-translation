"""Unit tests for the M8 glossary suggester (parser + dedupe + approve)."""
from __future__ import annotations

from app.db.models import (
    GlossaryEntry,
    GlossarySuggestion,
    Job,
    JobStatus,
    NameRegistry,
    SuggestionKind,
    SuggestionState,
)
from app.db.session import SessionLocal
from app.services.glossary_suggester import (
    SuggestedTerm,
    _parse_suggestions,
    approve_suggestion,
    deduplicate_against_existing,
    persist_suggestions,
    reject_suggestion,
)


def _make_job(db) -> str:
    job = Job(
        original_filename="t.pdf",
        file_path="/tmp/t.pdf",
        file_size_bytes=1,
        status=JobStatus.AWAITING_CONFIRMATION,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job.id


def test_parse_suggestions_handles_minimum_fields() -> None:
    raw = {
        "suggestions": [
            {"term_source": "Eldar", "candidate_burmese": "အဲလ်ဒါ"},
            {"term_source": "", "candidate_burmese": "skipped"},  # blank source
            {"term_source": "skipme", "candidate_burmese": ""},  # blank target
            "not a dict",
        ]
    }
    parsed = _parse_suggestions(raw)
    assert len(parsed) == 1
    assert parsed[0].term_source == "Eldar"
    assert parsed[0].kind == SuggestionKind.OTHER  # default when missing


def test_parse_suggestions_clamps_confidence() -> None:
    raw = {
        "suggestions": [
            {"term_source": "Hi", "candidate_burmese": "x", "confidence": 7.0},
            {"term_source": "Lo", "candidate_burmese": "x", "confidence": -2.0},
        ]
    }
    parsed = _parse_suggestions(raw)
    assert parsed[0].confidence == 1.0
    assert parsed[1].confidence == 0.0


def test_dedupe_drops_existing_glossary_terms(db_engine) -> None:
    with SessionLocal() as db:
        job_id = _make_job(db)
        db.add(
            GlossaryEntry(job_id=job_id, term_source="Alice", term_target="အဲလစ်")
        )
        db.commit()
        proposed = [
            SuggestedTerm("alice", "x", SuggestionKind.PROPER_NOUN, 0.9, []),  # dup
            SuggestedTerm("Bob", "ဘော်ဘ်", SuggestionKind.PROPER_NOUN, 0.8, []),
        ]
        kept = deduplicate_against_existing(db, job_id, proposed)
        assert [t.term_source for t in kept] == ["Bob"]


def test_dedupe_drops_existing_name_registry(db_engine) -> None:
    with SessionLocal() as db:
        job_id = _make_job(db)
        db.add(
            NameRegistry(
                job_id=job_id,
                source_name="Newton",
                burmese_phonetic="နယူတန်",
                first_seen_chunk_index=0,
            )
        )
        db.commit()
        proposed = [
            SuggestedTerm("NEWTON", "x", SuggestionKind.PROPER_NOUN, 0.9, []),
            SuggestedTerm("Einstein", "အိုင်းစတိုင်း", SuggestionKind.PROPER_NOUN, 0.9, []),
        ]
        kept = deduplicate_against_existing(db, job_id, proposed)
        assert [t.term_source for t in kept] == ["Einstein"]


def test_persist_replaces_only_pending(db_engine) -> None:
    with SessionLocal() as db:
        job_id = _make_job(db)
        # Pre-existing approved row should NOT be wiped on re-run.
        db.add(
            GlossarySuggestion(
                job_id=job_id, term_source="Approved", term_target="x",
                kind=SuggestionKind.OTHER, confidence=0.9,
                state=SuggestionState.APPROVED,
            )
        )
        db.add(
            GlossarySuggestion(
                job_id=job_id, term_source="OldPending", term_target="x",
                kind=SuggestionKind.OTHER, confidence=0.5,
                state=SuggestionState.PENDING,
            )
        )
        db.commit()

        persist_suggestions(
            db,
            job_id,
            [SuggestedTerm("New", "y", SuggestionKind.OTHER, 0.7, [])],
        )

        rows = db.query(GlossarySuggestion).filter_by(job_id=job_id).all()
        sources = sorted(r.term_source for r in rows)
        # Approved survived; old PENDING wiped; new PENDING inserted.
        assert sources == ["Approved", "New"]


def test_approve_creates_glossary_entry_and_links(db_engine) -> None:
    with SessionLocal() as db:
        job_id = _make_job(db)
        s = GlossarySuggestion(
            job_id=job_id, term_source="Eldar", term_target="အဲလ်ဒါ",
            kind=SuggestionKind.LOCATION, confidence=0.9,
        )
        db.add(s)
        db.commit()
        suggestion_id = s.id

        approve_suggestion(db, job_id, suggestion_id, term_target_override="အဲလ်ဒါ (edited)")

        approved = db.get(GlossarySuggestion, suggestion_id)
        assert approved.state == SuggestionState.APPROVED
        assert approved.term_target == "အဲလ်ဒါ (edited)"
        entry = db.get(GlossaryEntry, approved.approved_glossary_entry_id)
        assert entry is not None
        assert entry.term_target == "အဲလ်ဒါ (edited)"


def test_approve_when_manual_glossary_already_exists_marks_merged(db_engine) -> None:
    with SessionLocal() as db:
        job_id = _make_job(db)
        manual = GlossaryEntry(job_id=job_id, term_source="Alice", term_target="manual")
        db.add(manual)
        db.add(
            GlossarySuggestion(
                job_id=job_id, term_source="Alice", term_target="auto-prop",
                kind=SuggestionKind.PROPER_NOUN, confidence=0.9,
            )
        )
        db.commit()
        suggestion = (
            db.query(GlossarySuggestion).filter_by(job_id=job_id, term_source="Alice").one()
        )

        approve_suggestion(db, job_id, suggestion.id)

        suggestion = db.get(GlossarySuggestion, suggestion.id)
        # Manual entry wins; suggestion records as MERGED with link to manual row.
        assert suggestion.state == SuggestionState.MERGED
        assert suggestion.approved_glossary_entry_id == manual.id


def test_reject_marks_state(db_engine) -> None:
    with SessionLocal() as db:
        job_id = _make_job(db)
        s = GlossarySuggestion(
            job_id=job_id, term_source="x", term_target="y",
            kind=SuggestionKind.OTHER, confidence=0.5,
        )
        db.add(s)
        db.commit()
        reject_suggestion(db, job_id, s.id)
        assert db.get(GlossarySuggestion, s.id).state == SuggestionState.REJECTED

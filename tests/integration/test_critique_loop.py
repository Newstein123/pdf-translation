"""Integration tests for the M7 critique + revise loop.

These tests stub Gemini at the service layer (translate_chunk, critique_chunk,
revise_chunk) so we can verify the worker's gate logic and pass-row
persistence without hitting the API. The seed mirrors the existing
translation_pipeline test setup."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pymupdf

from app.db.models import (
    Category,
    ChunkStatus,
    ChunkTranslationPass,
    Job,
    JobStatus,
    PassKind,
    PromptVersion,
    StyleProfile,
    TranslationChunk,
)
from app.db.session import SessionLocal
from app.services.style_profiles import sync_profiles_to_db
from app.services.translation import (
    ChunkTranslationResult,
    CritiqueIssue,
    CritiqueResult,
    DiscoveredName,
    TranslatedBlock,
)


_TEXT = (
    "Alice walked through the Forest of Eldar to find Master Bell. " * 6
)


def _make_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), _TEXT, fontsize=11)
    doc.save(path)
    doc.close()
    return path


def _seed_job_at_translating(pdf_path: Path) -> str:
    """Drive a job through extraction so chunks exist, then mark category."""
    with SessionLocal() as db:
        job = Job(
            original_filename=pdf_path.name,
            file_path=str(pdf_path),
            file_size_bytes=pdf_path.stat().st_size,
            status=JobStatus.UPLOADED,
            category=Category.NOVEL,
            critique_enabled=True,
            quality_score_threshold=4.0,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id

    from app.workers.tasks import classify_task, extract_task

    with patch("app.workers.tasks.classify_task.delay"):
        extract_task.run(job_id)
    classify_task.run(job_id)

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        job.category = Category.NOVEL
        db.commit()
    return job_id


def _seed_default_profile() -> None:
    """Make sure the seeded YAMLs are synced into the test DB so the worker
    has a profile to compose prompts from."""
    seed_dir = Path(__file__).resolve().parent.parent.parent / "app" / "style_profiles"
    with SessionLocal() as db:
        sync_profiles_to_db(db, seed_dir)


def _stub_translate(*, blocks, **_kwargs) -> ChunkTranslationResult:
    return ChunkTranslationResult(
        blocks=[
            TranslatedBlock(block_id=b["block_id"], translated="[T1] " + b["source"])
            for b in blocks
        ],
        new_names=[DiscoveredName(source="Alice", burmese="အဲလစ်")],
        summary_for_next_chunk="pass-1 summary",
        total_tokens=100,
        prompt_tokens=70,
        output_tokens=30,
    )


def _make_critique(*, overall: float, high_severity_block: str | None = None) -> CritiqueResult:
    issues: list[CritiqueIssue] = []
    if high_severity_block is not None:
        issues.append(
            CritiqueIssue(
                block_id=high_severity_block,
                severity="high",
                kind="fidelity",
                snippet="missed clause",
                suggestion="translate the elided fragment",
            )
        )
    return CritiqueResult(
        naturalness=overall,
        fidelity=overall,
        register_match=overall,
        consistency=overall,
        overall=overall,
        issues=issues,
        raw={"overall": overall, "issues_synthetic": True},
        total_tokens=20,
        prompt_tokens=15,
        output_tokens=5,
    )


def _stub_revise(*, blocks, **_kwargs) -> ChunkTranslationResult:
    return ChunkTranslationResult(
        blocks=[
            TranslatedBlock(block_id=b["block_id"], translated="[T3-revised] " + b["source"])
            for b in blocks
        ],
        new_names=[],
        summary_for_next_chunk="revised summary",
        total_tokens=80,
        prompt_tokens=55,
        output_tokens=25,
    )


def _set_critique_enabled_for_test(value: str) -> None:
    """Override the conftest CRITIQUE_ENABLED=false so individual cases can
    exercise the loop. We mutate the cached settings directly because env-var
    reads were memoized at process start."""
    os.environ["CRITIQUE_ENABLED"] = value
    from app.config import get_settings as _gs
    _gs.cache_clear()


def test_critique_skipped_when_above_threshold(db_engine, tmp_path: Path) -> None:
    _set_critique_enabled_for_test("true")
    _seed_default_profile()
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_job_at_translating(pdf)

    from app.workers import tasks

    with (
        patch.object(tasks.translation, "translate_chunk", side_effect=_stub_translate),
        patch.object(
            tasks.translation, "critique_chunk",
            return_value=_make_critique(overall=4.5),
        ) as critique_spy,
        patch.object(tasks.translation, "revise_chunk") as revise_spy,
        patch("app.workers.tasks.rebuild_task.delay"),
    ):
        tasks.translate_task.run(job_id)

    # Critique ran; revise must have been skipped because overall >= threshold (4.0).
    assert critique_spy.called
    assert not revise_spy.called

    with SessionLocal() as db:
        chunks = db.query(TranslationChunk).filter_by(job_id=job_id).all()
        chunk_ids = [c.id for c in chunks]
        passes = (
            db.query(ChunkTranslationPass)
            .filter(ChunkTranslationPass.chunk_id.in_(chunk_ids))
            .all()
        )
        kinds_per_chunk: dict[str, set[PassKind]] = {}
        for p in passes:
            kinds_per_chunk.setdefault(p.chunk_id, set()).add(p.pass_kind)

        for chunk in chunks:
            assert chunk.status == ChunkStatus.COMPLETED
            assert chunk.final_pass_id is not None
            kinds = kinds_per_chunk.get(chunk.id, set())
            assert PassKind.TRANSLATE in kinds
            assert PassKind.CRITIQUE in kinds
            assert PassKind.REVISE not in kinds  # gated out
            # The chunk's translated text still comes from pass 1.
            assert chunk.translated_text and chunk.translated_text.startswith("[T1]")


def test_revise_runs_when_high_severity_even_above_threshold(db_engine, tmp_path: Path) -> None:
    _set_critique_enabled_for_test("true")
    _seed_default_profile()
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_job_at_translating(pdf)

    from app.workers import tasks

    # Find any block_id we'll receive at translate time so the high-severity
    # finding is concrete. We snapshot it via the stub.
    seen_block_ids: list[str] = []

    def _capture_translate(*, blocks, **_kwargs):
        seen_block_ids.extend(b["block_id"] for b in blocks)
        return _stub_translate(blocks=blocks)

    def _make_critique_high(*_args, **_kwargs):
        # Attach a high-severity finding to the first block of this chunk.
        first = seen_block_ids[-1] if seen_block_ids else "p1b0"
        return _make_critique(overall=4.7, high_severity_block=first)

    with (
        patch.object(tasks.translation, "translate_chunk", side_effect=_capture_translate),
        patch.object(tasks.translation, "critique_chunk", side_effect=_make_critique_high),
        patch.object(tasks.translation, "revise_chunk", side_effect=_stub_revise) as revise_spy,
        patch("app.workers.tasks.rebuild_task.delay"),
    ):
        tasks.translate_task.run(job_id)

    # High-severity must override the threshold gate.
    assert revise_spy.called

    with SessionLocal() as db:
        chunks = db.query(TranslationChunk).filter_by(job_id=job_id).all()
        for chunk in chunks:
            # final_pass_id now points at the revise pass, not the translate pass.
            final = db.get(ChunkTranslationPass, chunk.final_pass_id)
            assert final is not None
            assert final.pass_kind == PassKind.REVISE
            assert chunk.translated_text.startswith("[T3-revised]")
            assert chunk.context_summary == "revised summary"


def test_revise_runs_when_below_threshold(db_engine, tmp_path: Path) -> None:
    _set_critique_enabled_for_test("true")
    _seed_default_profile()
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_job_at_translating(pdf)

    from app.workers import tasks

    with (
        patch.object(tasks.translation, "translate_chunk", side_effect=_stub_translate),
        patch.object(
            tasks.translation, "critique_chunk", return_value=_make_critique(overall=3.0)
        ),
        patch.object(tasks.translation, "revise_chunk", side_effect=_stub_revise) as revise_spy,
        patch("app.workers.tasks.rebuild_task.delay"),
    ):
        tasks.translate_task.run(job_id)

    assert revise_spy.called

    with SessionLocal() as db:
        chunks = db.query(TranslationChunk).filter_by(job_id=job_id).all()
        for chunk in chunks:
            assert chunk.translated_text.startswith("[T3-revised]")


def test_revise_failure_keeps_pass_one(db_engine, tmp_path: Path) -> None:
    """If revise raises, pass-1 must remain the final translation and the
    chunk must still complete cleanly."""
    from app.services.translation import TranslationError

    _set_critique_enabled_for_test("true")
    _seed_default_profile()
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_job_at_translating(pdf)

    from app.workers import tasks

    with (
        patch.object(tasks.translation, "translate_chunk", side_effect=_stub_translate),
        patch.object(
            tasks.translation, "critique_chunk", return_value=_make_critique(overall=2.5)
        ),
        patch.object(
            tasks.translation, "revise_chunk",
            side_effect=TranslationError("revise broke block fidelity"),
        ),
        patch("app.workers.tasks.rebuild_task.delay"),
    ):
        tasks.translate_task.run(job_id)

    with SessionLocal() as db:
        chunks = db.query(TranslationChunk).filter_by(job_id=job_id).all()
        for chunk in chunks:
            assert chunk.status == ChunkStatus.COMPLETED
            # final_pass_id still points at the TRANSLATE pass.
            final = db.get(ChunkTranslationPass, chunk.final_pass_id)
            assert final is not None
            assert final.pass_kind == PassKind.TRANSLATE
            assert chunk.translated_text.startswith("[T1]")

        # A FAILED revise pass row was recorded for forensics.
        chunk_ids = [c.id for c in chunks]
        failed = (
            db.query(ChunkTranslationPass)
            .filter(
                ChunkTranslationPass.chunk_id.in_(chunk_ids),
                ChunkTranslationPass.pass_kind == PassKind.REVISE,
            )
            .all()
        )
        assert failed, "expected forensic FAILED revise pass rows"
        assert all(p.status.value == "FAILED" for p in failed)


def test_translate_records_prompt_version_per_pass(db_engine, tmp_path: Path) -> None:
    _set_critique_enabled_for_test("true")
    _seed_default_profile()
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_job_at_translating(pdf)

    from app.workers import tasks

    with (
        patch.object(tasks.translation, "translate_chunk", side_effect=_stub_translate),
        patch.object(
            tasks.translation, "critique_chunk", return_value=_make_critique(overall=4.5)
        ),
        patch("app.workers.tasks.rebuild_task.delay"),
    ):
        tasks.translate_task.run(job_id)

    with SessionLocal() as db:
        translate_versions = (
            db.query(PromptVersion)
            .filter(PromptVersion.purpose.in_(["TRANSLATE"]))
            .all()
        )
        critique_versions = (
            db.query(PromptVersion)
            .filter(PromptVersion.purpose.in_(["CRITIQUE"]))
            .all()
        )
    # At least one row each — the cache is populated on first chunk and reused.
    assert translate_versions, "expected a translate prompt_version row"
    assert critique_versions, "expected a critique prompt_version row"

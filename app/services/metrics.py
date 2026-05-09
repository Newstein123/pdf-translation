"""Aggregate metrics for the M9 admin dashboard.

The endpoints stay thin: this module owns the SQL so route handlers don't
have to learn the schema. Aggregates are computed on read; for higher-volume
deployments these would move to a materialized view, but at single-user
scale the chunks-per-job count is tractable for live SQL.
"""
from __future__ import annotations

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import (
    ChunkStatus,
    ChunkTranslationPass,
    Job,
    PassKind,
    PassStatus,
    QualityScore,
    TranslationChunk,
)


def job_metrics(session: Session, job_id: str) -> dict:
    chunks_total = session.execute(
        select(func.count(TranslationChunk.id)).where(TranslationChunk.job_id == job_id)
    ).scalar_one()
    chunks_completed = session.execute(
        select(func.count(TranslationChunk.id)).where(
            TranslationChunk.job_id == job_id,
            TranslationChunk.status == ChunkStatus.COMPLETED,
        )
    ).scalar_one()

    judged_q = (
        select(QualityScore)
        .join(TranslationChunk, TranslationChunk.id == QualityScore.chunk_id)
        .where(TranslationChunk.job_id == job_id)
    )
    judged = list(session.execute(judged_q).scalars())
    chunks_judged = len({s.chunk_id for s in judged})
    avg_overall = (
        sum(s.overall for s in judged) / len(judged) if judged else None
    )
    settings = get_settings()
    low_score_count = sum(1 for s in judged if s.overall <= settings.low_score_threshold)

    # Revise rate = chunks where final pass is REVISE / chunks_completed.
    revise_count = session.execute(
        select(func.count(ChunkTranslationPass.id))
        .join(TranslationChunk, TranslationChunk.final_pass_id == ChunkTranslationPass.id)
        .where(
            TranslationChunk.job_id == job_id,
            ChunkTranslationPass.pass_kind == PassKind.REVISE,
        )
    ).scalar_one()
    revise_rate = (revise_count / chunks_completed) if chunks_completed else None

    tokens_by_kind = _tokens_by_pass_kind_for_job(session, job_id)

    return {
        "job_id": job_id,
        "chunks_total": int(chunks_total),
        "chunks_completed": int(chunks_completed),
        "chunks_judged": chunks_judged,
        "avg_overall": avg_overall,
        "low_score_count": low_score_count,
        "revise_rate": revise_rate,
        "tokens_by_pass_kind": tokens_by_kind,
    }


def admin_metrics(session: Session) -> dict:
    jobs_total = session.execute(select(func.count(Job.id))).scalar_one()
    chunks_total = session.execute(
        select(func.count(TranslationChunk.id))
    ).scalar_one()
    judged_count = session.execute(
        select(func.count(func.distinct(QualityScore.chunk_id)))
    ).scalar_one()
    avg_overall_row = session.execute(select(func.avg(QualityScore.overall))).scalar_one()
    avg_overall = float(avg_overall_row) if avg_overall_row is not None else None

    completed = session.execute(
        select(func.count(TranslationChunk.id)).where(
            TranslationChunk.status == ChunkStatus.COMPLETED
        )
    ).scalar_one()
    revise_count = session.execute(
        select(func.count(ChunkTranslationPass.id))
        .join(TranslationChunk, TranslationChunk.final_pass_id == ChunkTranslationPass.id)
        .where(ChunkTranslationPass.pass_kind == PassKind.REVISE)
    ).scalar_one()
    revise_rate = (revise_count / completed) if completed else None

    tokens_by_kind = _tokens_by_pass_kind_global(session)

    return {
        "jobs_total": int(jobs_total),
        "chunks_total": int(chunks_total),
        "chunks_judged": int(judged_count),
        "avg_overall": avg_overall,
        "revise_rate": revise_rate,
        "tokens_by_pass_kind": tokens_by_kind,
        # avg_revise_uplift requires a paired pre/post-revise judge call
        # which is M9 future work. Plumbed but currently None.
        "avg_revise_uplift": None,
    }


def _tokens_by_pass_kind_for_job(session: Session, job_id: str) -> dict[str, int]:
    rows = session.execute(
        select(
            ChunkTranslationPass.pass_kind,
            func.coalesce(func.sum(ChunkTranslationPass.input_tokens), 0)
            + func.coalesce(func.sum(ChunkTranslationPass.output_tokens), 0),
        )
        .join(TranslationChunk, TranslationChunk.id == ChunkTranslationPass.chunk_id)
        .where(
            TranslationChunk.job_id == job_id,
            ChunkTranslationPass.status == PassStatus.OK,
        )
        .group_by(ChunkTranslationPass.pass_kind)
    ).all()
    return {kind.value: int(total) for kind, total in rows}


def _tokens_by_pass_kind_global(session: Session) -> dict[str, int]:
    rows = session.execute(
        select(
            ChunkTranslationPass.pass_kind,
            func.coalesce(func.sum(ChunkTranslationPass.input_tokens), 0)
            + func.coalesce(func.sum(ChunkTranslationPass.output_tokens), 0),
        )
        .where(ChunkTranslationPass.status == PassStatus.OK)
        .group_by(ChunkTranslationPass.pass_kind)
    ).all()
    return {kind.value: int(total) for kind, total in rows}


def low_score_chunks(
    session: Session, job_id: str, *, max_overall: float | None = None
) -> list[QualityScore]:
    """Chunks below ``max_overall`` (or settings.low_score_threshold), sorted
    ascending so the worst chunk surfaces first."""
    threshold = max_overall if max_overall is not None else get_settings().low_score_threshold
    return list(
        session.execute(
            select(QualityScore)
            .join(TranslationChunk, TranslationChunk.id == QualityScore.chunk_id)
            .where(
                TranslationChunk.job_id == job_id,
                QualityScore.overall <= threshold,
            )
            .order_by(QualityScore.overall.asc())
        ).scalars()
    )


def quality_scores_for_job(session: Session, job_id: str) -> list[QualityScore]:
    return list(
        session.execute(
            select(QualityScore)
            .join(TranslationChunk, TranslationChunk.id == QualityScore.chunk_id)
            .where(TranslationChunk.job_id == job_id)
            .order_by(TranslationChunk.chunk_index.asc())
        ).scalars()
    )


# Reserved for future SQL-aggregate refactor; keeps imports stable.
__all__ = [
    "job_metrics",
    "admin_metrics",
    "low_score_chunks",
    "quality_scores_for_job",
    "case",
]

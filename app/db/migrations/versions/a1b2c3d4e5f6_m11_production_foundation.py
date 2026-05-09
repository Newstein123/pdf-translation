"""m11 production foundation

Adds the four foundation tables for the production roadmap:

  * book_briefs        — whole-book pre-analysis (POV, tone, voice, register).
  * reference_passages — curated source↔Burmese aligned passages.
  * linter_findings    — deterministic post-translation rule findings.
  * gemini_caches      — Gemini context-cache handles for prefix reuse.

Plus two columns on chunk_translation_passes so we can attribute spend to
cached vs. uncached input tokens once context caching ships:

  * cached_input_tokens (subset of input_tokens that hit the cache)
  * gemini_cache_id     (FK to gemini_caches; SET NULL on cache deletion)

Revision ID: a1b2c3d4e5f6
Revises: f253b0841f2f
Create Date: 2026-05-03

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "f253b0841f2f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. gemini_caches first — chunk_translation_passes will FK into it.
    op.create_table(
        "gemini_caches",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "job_id",
            sa.String(length=36),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("purpose", sa.String(length=32), nullable=False),
        sa.Column("prefix_hash", sa.String(length=64), nullable=False),
        sa.Column("gemini_cache_name", sa.String(length=255), nullable=False),
        sa.Column("model", sa.String(length=64), nullable=False),
        sa.Column("ttl_seconds", sa.Integer(), nullable=False, server_default="3600"),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("cached_token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("hit_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_used_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index(
        "ix_gemini_caches_prefix_hash_purpose",
        "gemini_caches",
        ["prefix_hash", "purpose"],
    )
    op.create_index("ix_gemini_caches_expires_at", "gemini_caches", ["expires_at"])

    # 2. cached_input_tokens + gemini_cache_id on chunk_translation_passes.
    with op.batch_alter_table("chunk_translation_passes", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "cached_input_tokens",
                sa.Integer(),
                nullable=False,
                server_default="0",
            )
        )
        batch_op.add_column(
            sa.Column(
                "gemini_cache_id",
                sa.String(length=36),
                nullable=True,
            )
        )
        batch_op.create_foreign_key(
            "fk_passes_gemini_cache_id",
            "gemini_caches",
            ["gemini_cache_id"],
            ["id"],
            ondelete="SET NULL",
        )

    # 3. book_briefs.
    op.create_table(
        "book_briefs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "job_id",
            sa.String(length=36),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="PENDING"),
        sa.Column("brief_json", sa.JSON(), nullable=True),
        sa.Column("sample_token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("model", sa.String(length=64), nullable=True),
        sa.Column(
            "prompt_version_id",
            sa.String(length=36),
            sa.ForeignKey("prompt_versions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "edited_by_human", sa.Boolean(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )

    # 4. reference_passages.
    op.create_table(
        "reference_passages",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("source_text", sa.Text(), nullable=False),
        sa.Column("target_text", sa.Text(), nullable=False),
        sa.Column("source_lang", sa.String(length=16), nullable=False, server_default="en"),
        sa.Column("target_lang", sa.String(length=16), nullable=False, server_default="my"),
        sa.Column("genre", sa.String(length=64), nullable=True),
        sa.Column("register", sa.String(length=64), nullable=True),
        sa.Column("mode", sa.String(length=32), nullable=False, server_default="OTHER"),
        sa.Column("book_title", sa.String(length=512), nullable=True),
        sa.Column("translator", sa.String(length=255), nullable=True),
        sa.Column("embedding_json", sa.JSON(), nullable=True),
        sa.Column("embedding_model", sa.String(length=64), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.text("1")),
        sa.Column("added_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index(
        "ix_reference_passages_genre_mode",
        "reference_passages",
        ["genre", "mode", "active"],
    )

    # 5. linter_findings.
    op.create_table(
        "linter_findings",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "chunk_id",
            sa.String(length=36),
            sa.ForeignKey("translation_chunks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "pass_id",
            sa.String(length=36),
            sa.ForeignKey("chunk_translation_passes.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("rule_key", sa.String(length=64), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False, server_default="WARNING"),
        sa.Column("block_id", sa.String(length=64), nullable=True),
        sa.Column("position_start", sa.Integer(), nullable=True),
        sa.Column("position_end", sa.Integer(), nullable=True),
        sa.Column("snippet", sa.Text(), nullable=True),
        sa.Column("payload_json", sa.JSON(), nullable=True),
        sa.Column(
            "resolved_in_pass_id",
            sa.String(length=36),
            sa.ForeignKey("chunk_translation_passes.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_linter_findings_rule_key", "linter_findings", ["rule_key"])
    op.create_index(
        "ix_linter_findings_chunk_severity", "linter_findings", ["chunk_id", "severity"]
    )
    op.create_index(
        "ix_linter_findings_chunk_rule", "linter_findings", ["chunk_id", "rule_key"]
    )


def downgrade() -> None:
    op.drop_index("ix_linter_findings_chunk_rule", table_name="linter_findings")
    op.drop_index("ix_linter_findings_chunk_severity", table_name="linter_findings")
    op.drop_index("ix_linter_findings_rule_key", table_name="linter_findings")
    op.drop_table("linter_findings")

    op.drop_index("ix_reference_passages_genre_mode", table_name="reference_passages")
    op.drop_table("reference_passages")

    op.drop_table("book_briefs")

    with op.batch_alter_table("chunk_translation_passes", schema=None) as batch_op:
        batch_op.drop_constraint("fk_passes_gemini_cache_id", type_="foreignkey")
        batch_op.drop_column("gemini_cache_id")
        batch_op.drop_column("cached_input_tokens")

    op.drop_index("ix_gemini_caches_expires_at", table_name="gemini_caches")
    op.drop_index("ix_gemini_caches_prefix_hash_purpose", table_name="gemini_caches")
    op.drop_table("gemini_caches")

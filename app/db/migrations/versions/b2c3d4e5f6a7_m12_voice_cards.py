"""m12 voice cards

Adds a ``voice_cards`` table: per-translator distilled style brief.

Generated once (offline) by analyzing a translator's corpus of published
Burmese translations. The resulting card is a 200–400 token natural-language
description of the translator's voice — sentence rhythm, register,
dialogue style, particle preferences, vocabulary patterns. Cards are then
injected into the cached system-prompt prefix on each translation job, so
the per-chunk cost of using them is near zero (cached tokens bill at ~25%
of standard input rate).

This is the cheap alternative to per-chunk ref-passage retrieval: a single
~300-token card replaces ~2000 tokens of raw passage injection while
preserving most of the voice-anchoring effect.

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-05-03

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "voice_cards",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("translator", sa.String(length=255), nullable=False, unique=True),
        sa.Column("target_lang", sa.String(length=16), nullable=False, server_default="my"),
        sa.Column("genre_tags_json", sa.JSON(), nullable=True),
        # The prose card itself — ready to drop into a system prompt.
        sa.Column("card_text", sa.Text(), nullable=False),
        # Empirical baselines computed from the corpus: avg sentence length,
        # particle density, dialogue ratio. Used by the linter and the brief.
        sa.Column("baselines_json", sa.JSON(), nullable=True),
        sa.Column("source_passage_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("model", sa.String(length=64), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.text("1")),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_voice_cards_translator_active", "voice_cards", ["translator", "active"])


def downgrade() -> None:
    op.drop_index("ix_voice_cards_translator_active", table_name="voice_cards")
    op.drop_table("voice_cards")

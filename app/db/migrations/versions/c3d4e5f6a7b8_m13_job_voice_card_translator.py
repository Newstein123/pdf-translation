"""m13 job voice card translator

Adds ``Job.voice_card_translator`` — an optional per-job override that picks
which translator's voice card to inject into the cached prompt prefix on
every translate / critique / revise / judge call. When NULL, the worker
falls back to ``settings.default_voice_card_translator`` (typically the
modern conversational voice).

Storing the *translator name* rather than the voice-card id lets us rebuild
voice cards offline (force-regenerate) without invalidating in-flight job
configurations.

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-05-04

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("voice_card_translator", sa.String(length=255), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.drop_column("voice_card_translator")

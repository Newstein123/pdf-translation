"""m15 job skip pages

Adds ``Job.skip_pages`` — a user-supplied range string like
``"1-3, 5, 8-10"`` that lists pages to skip translation AND skip from the
final exported PDF. Saves Gemini tokens on copyright pages, blank pages,
publisher boilerplate, etc.

Stored as a free-form string (rather than a parsed JSON list) because:
  * it round-trips cleanly with the UI input
  * the parser tolerates comma / space variations
  * a future "tell me which pages will be skipped" preview can re-parse
    on demand

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-05-04
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e5f6a7b8c9d0"
down_revision: Union[str, None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("skip_pages", sa.String(length=512), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.drop_column("skip_pages")

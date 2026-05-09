"""m14 book metadata + export layout mode

Adds book-metadata fields to ``Job`` (title and author, source + target) plus
the ``export_layout_mode`` selector and a per-job JSON override for template
settings. Together these support the two-path export design:

  * AUTO     — system picks based on color/imagery analysis (default).
  * TEMPLATE — clean Burmese book template (configurable, defaults in
               ``settings``; per-job overrides in ``export_template_overrides_json``).
  * PRESERVE — existing redact-and-fill path that keeps source layout.

Title and author are editable at any stage (upload, AWAITING_CONFIRMATION,
post-translation) and feed the template's running headers.

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-05-04
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("title_source", sa.String(length=512), nullable=True))
        batch_op.add_column(sa.Column("title_target", sa.String(length=512), nullable=True))
        batch_op.add_column(sa.Column("author_source", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("author_target", sa.String(length=255), nullable=True))
        batch_op.add_column(
            sa.Column(
                "export_layout_mode",
                sa.String(length=16),
                nullable=False,
                server_default="AUTO",
            )
        )
        batch_op.add_column(
            sa.Column("export_template_overrides_json", sa.JSON(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.drop_column("export_template_overrides_json")
        batch_op.drop_column("export_layout_mode")
        batch_op.drop_column("author_target")
        batch_op.drop_column("author_source")
        batch_op.drop_column("title_target")
        batch_op.drop_column("title_source")

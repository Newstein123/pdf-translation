"""m16 job cover image

Adds ``Job.cover_image_path`` — local filesystem path to the user-uploaded
cover image (normalized to JPEG by ``services.cover_image``). When set,
both the TEMPLATE and PRESERVE export paths render the cover as page 1
of the output PDF.

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-05-04
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f6a7b8c9d0e1"
down_revision: Union[str, None] = "e5f6a7b8c9d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("cover_image_path", sa.String(length=1024), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.drop_column("cover_image_path")

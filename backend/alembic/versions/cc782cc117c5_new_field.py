"""new field

Revision ID: cc782cc117c5
Revises: ca37b1238e0b
Create Date: 2024-10-13 01:23:31.742629

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "cc782cc117c5"
down_revision: Union[str, None] = "ca37b1238e0b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass

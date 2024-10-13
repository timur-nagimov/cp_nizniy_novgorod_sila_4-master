"""new field

Revision ID: 3ec6067a8282
Revises: cc782cc117c5
Create Date: 2024-10-13 01:24:27.949504

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3ec6067a8282'
down_revision: Union[str, None] = 'cc782cc117c5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass

"""new field

Revision ID: 7a160339fcc4
Revises: 3ec6067a8282
Create Date: 2024-10-13 01:27:03.064693

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "7a160339fcc4"
down_revision: Union[str, None] = "3ec6067a8282"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column(
        "prompt_history", sa.Column("rewritten_question", sa.Text(), nullable=True)
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("prompt_history", "rewritten_question")
    # ### end Alembic commands ###
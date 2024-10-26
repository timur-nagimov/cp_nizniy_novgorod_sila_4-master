from sqlalchemy import Column, Text, Integer, Identity

from backend import Base


class PromptHistory(Base):
    __tablename__ = "prompt_history"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, Identity(always=True), primary_key=True)
    question = Column(Text)
    rewritten_question = Column(Text)
    context = Column(Text)
    answer = Column(Text)

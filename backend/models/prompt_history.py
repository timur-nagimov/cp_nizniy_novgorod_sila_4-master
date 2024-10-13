from sqlalchemy import Column, Text, Integer, Identity

from db.init import Base


class PromptHistory(Base):
    __tablename__ = "prompt_history"

    id = Column(Integer, Identity(always=True), primary_key=True)
    question = Column(Text)
    rewritten_question = Column(Text)
    context = Column(Text)
    answer = Column(Text)
    
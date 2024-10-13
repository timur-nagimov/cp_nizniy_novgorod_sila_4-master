from backend.db.init import session
from backend.models.prompt_history import PromptHistory


for prompt in session.query(PromptHistory).all():
    print(vars(prompt))
    break

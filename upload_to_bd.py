from backend.db.init import session
from backend.models.prompt_history import PromptHistory
import pickle

# Открываем файл в режиме записи бинарных данных
with open('presets.pkl', 'rb') as f:
        data = pickle.load(f)

for key in data.keys():
    info = PromptHistory(question=key, context='', answer=data[key])
    session.add(info)
session.commit()
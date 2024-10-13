from difflib import SequenceMatcher
import joblib

from backend.db.init import session
from backend.models.prompt_history import PromptHistory


qa_database = {}
for prompt in session.query(PromptHistory).all():
    qa_database[prompt.question] = prompt.answer
# База данных вопросов и ответов в виде словаря
# qa_database = joblib.load("")

def find_best_match(query, database):
    """
    Находит наиболее подходящий вопрос из базы данных по степени схожести.

    :param query: Вопрос пользователя.
    :param database: Словарь с вопросами (ключи) и ответами (значения).
    :return: Кортеж (лучший совпадающий вопрос, степень схожести).
    """
    highest_ratio = 0
    best_question = None
    for question in database.keys():
        ratio = SequenceMatcher(None, query.lower(), question.lower()).ratio()
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_question = question
    return best_question, highest_ratio

def get_answer(query, database=qa_database, threshold=0.7):
    """
    Возвращает ответ на вопрос пользователя, если степень совпадения превышает порог.

    :param query: Вопрос пользователя.
    :param database: Словарь с вопросами и ответами.
    :param threshold: Порог схожести для принятия решения.
    :return: Строка с ответом или None.
    """
    best_question, similarity = find_best_match(query, database)
    if similarity >= threshold:
        return database[best_question]
    else:
        return None
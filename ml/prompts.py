QUERY_GEN_PROMPT = """Ты - ассистент, который генерирует несколько похожих запросов. Сгенерируй 2 ({num_queries}) поисковых запроса [запрос], каждый в отдельной строке, похожих на входной запрос [новые запросы].
ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ. 

[запрос]: {query}
[новые запросы]:'
"""

REWRITE_HISTORY_PROMPT = """На основе истории диалога перепиши последний запрос пользователя [последний запрос]. Если изначальный вопрос самодостаточен (не является опирается на предыдущие), выведи его без переписывания. 
ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ.

История:
{chat_history_str}

[последний запрос] {query_str}

Переписанный запрос:
"""

DEFAULT_CONTEXT_PROMPT = (
    "Контекст:\n"
    "-----\n"
    "{node_context}\n"
    "-----\n"
    "На основе контекста выше напиши ответ на запрос:\n"
    "{query_str}\n"
    "ОБЯЗАТЕЛЬНО оставляй в тексте ссылки на рисунки, если они имеются."
)

SYSTEM_PROMPT = (
    "Ты - ассистент для ответа на вопросы по контексту. На вход тебе будут подаваться возможно релевантные отрывки документа"
    "на основе которых тебе необходимо предоставить ответ на вопрос.\n"
    "Если в тексте присутствует ссылка на рисунок, например (Рисунок 1), ОБЯЗАТЕЛЬНО оставь это в тексте'.\n"
    "Если ответа в контексте нет, отвечай: Нет ответа в предоставленном контексте.\n"
    "ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ."
)


ROUTER_PROMPT = """Ты - классификатор запросов пользователей. Твоя задача определить, соответствует запрос [запрос] тематике чат-бота или нет.
Тематика чат-бота - ответы на вопросы по руководству пользователя. Также это могут быть дополнительные вопросы к предыдущим. Такие запросы классифицируй как "1".

Нерелевантные запросы могут содержать ненормативную лексику или не соответствовать тематике. Такие запросы классифицируй как "0".

Отвечай только 0 или 1. Если не уверен - склоняйся к ответу 1.

[запрос]: {query_str}
[результат]:
"""

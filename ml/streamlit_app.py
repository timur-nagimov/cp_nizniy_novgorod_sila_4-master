import os
import sys
import streamlit as st

# ADD BACKEND TO PATH
path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path)

from backend.db.init import session
from backend.models.prompt_history import PromptHistory

from rag import get_response_with_routing
from yandex_gpt import load_yandex_gpt
from retriever import load_embedder
from preset import get_answer


st.title('Чат-бот ОАО "РЖД"')


@st.cache_resource
def initialize_models():
    """
    Инициализирует и кэширует необходимые модели для чата.
    Загружает модели Qwen и эмбеддер при запуске приложения.
    """
    load_yandex_gpt()
    load_embedder()


# Инициализация моделей при запуске приложения
initialize_models()


def display_collapsible_docs(context):
    """
    Отображает секцию с контекстом документов, используемым в ответе.

    Параметры:
    context (str): Текст контекста (использованные документы), который будет отображен в сворачиваемом виде.
    """
    with st.expander("Использованные отрывки документа:", expanded=False):
        st.markdown(context)


# Проверка, если сессия сообщений не инициализирована, то создаём её
if "messages" not in st.session_state:
    st.session_state.messages = []

# Отображение истории сообщений в сессии
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    else:
        if message.get("context"):
            display_collapsible_docs(message["context"])
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


log = None
if prompt := st.chat_input("Введите свой запрос"):

    with st.chat_message("user"):
        st.markdown(prompt)

    # Сначала ищем похожий вопрос в базе данных
    preset_answer = get_answer(prompt, threshold=0.9)
    if preset_answer:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(preset_answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": preset_answer}
        )
    # Если ответа в базе не нашлось - запускаем пайплайн
    else:
        history_str = "\n".join(
            [
                f"{msg['role']}: {msg['content']}"
                for msg in st.session_state.messages[-5:]
            ]
        )

        with st.spinner("Обработка запроса"):
            with st.chat_message("assistant"):
                # Получаем ответ
                response = get_response_with_routing(history_str, prompt)
                if isinstance(response, str):
                    st.markdown(response)
                    st.session_state.messages.append(
                        {"role": "user", "content": prompt}
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

                else:
                    # Отображаем полученный контекст
                    display_collapsible_docs(response["context"])
                    st.markdown(response["response"].text)
                    st.session_state.messages.append(
                        {"role": "user", "content": prompt}
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "context": response["context"],
                            "content": response["response"].text,
                        }
                    )
                    # Сохраняем историю запроса в лог для базы данных
                    log = PromptHistory(
                        question=prompt,
                        rewritten_question=response["rewritten_query"],
                        context=response["context"],
                        answer=response["response"].text,
                    )

    # Если лог был сформирован, сохраняем его в базу данных
    if log:
        session.add(log)
        session.commit()

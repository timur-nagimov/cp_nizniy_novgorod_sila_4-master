import os
import sys
import joblib
import ast
import re
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


st.title('Чат-бот по руководству пользователя')


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


def find_images(image_ref, image_folder='./images/'):
    image_ref = image_ref.replace(')', '').replace('(', '')
    found_images = [f for f in os.listdir(image_folder) if f.startswith(image_ref)]

    return found_images[0]


def display_collapsible_docs(context):
    """
    Отображает секцию с контекстом документов, используемым в ответе.

    Параметры:
    context (str): Текст контекста (использованные документы), который будет отображен в сворачиваемом виде.
    """
    with st.expander("Использованные отрывки документа:", expanded=False):
        context_list = context.split('\n\n\n')[:-1]
        for cont in context_list:
            print(cont.split('РИСУНКИ: '))
            text, images = cont.split('РИСУНКИ: ')
            img_list = ast.literal_eval(images.strip())
            # print(img_list)

            st.markdown(text)
            for img in img_list:
                try:
                    st.image(f'./images/{img}.png', caption=img)
                except:
                    print(f'COULDNT SHOW IMAGE {img}')



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

        image_refs = re.findall(r'\(Рисунок\s\d+\)', preset_answer)
        for image_ref in list(dict.fromkeys(image_refs)):
            img_name = find_images(image_ref=image_ref)
            st.image(f'./images/{img_name}', caption=img_name.split('.png')[0])

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

        
        with st.chat_message("assistant"):
            with st.spinner("Обработка запроса"):
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
                    response_text = response["response"].text
                    st.markdown(response_text)

                    image_refs = re.findall(r'\(Рисунок\s\d+\)', response_text)
                    for image_ref in list(dict.fromkeys(image_refs)):
                        img_name = find_images(image_ref=image_ref)
                        st.image(f'./images/{img_name}', caption=img_name.split('.png')[0])

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

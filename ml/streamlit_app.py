import os
import sys

# ADD BACKEND TO PATH
path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path)

from backend.db.init import session
from backend.models.prompt_history import PromptHistory


import streamlit as st

from rag import init_global_settings, get_response
from preset import get_answer


st.title('Чат-бот ОАО "РЖД"')


@st.cache_resource
def initialize_models():
    init_global_settings()


initialize_models()


def display_collapsible_docs(context):
    with st.expander("Использованные отрывки документа:", expanded=False):
        st.markdown(context)


if "messages" not in st.session_state:
    st.session_state.messages = []

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

    preset_answer = get_answer(prompt, threshold=0.9)
    if preset_answer:
        # print("PRESET ANSWER")
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(preset_answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": preset_answer}
        )

    else:
        history_str = "\n".join(
            [
                f"{msg['role']}: {msg['content']}"
                for msg in st.session_state.messages[-5:]
            ]
        )
        print(history_str)

        with st.spinner("Обработка запроса"):
            with st.chat_message("assistant"):
                response = get_response(history_str, prompt)
                if isinstance(response, str):
                    # print("CLASSIFIED RESPONSE")
                    st.markdown(response)
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                    # log = PromptHistory(question=prompt, context='', answer=response)
                else:
                    display_collapsible_docs(response["context"])
                    st.markdown(response["response"].text)
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "context": response["context"],
                            "content": response["response"].text,
                        }
                    )
                    log = PromptHistory(
                        question=prompt,
                        rewritten_question=response["rewritten_query"],
                        context=response["context"],
                        answer=response["response"].text,
                    )

    if log:
        session.add(log)
        session.commit()

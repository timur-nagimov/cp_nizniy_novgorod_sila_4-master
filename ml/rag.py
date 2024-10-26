from llama_index.core import Settings
from typing import Any, Dict, List, Optional
from llama_index.core.bridge.pydantic import Field

from llama_index.core import PromptTemplate
from dataclasses import dataclass

from llama_index.core.query_pipeline import QueryPipeline, InputComponent

from custom_response import ResponseWithChatHistory
from retriever import load_index, get_fusion_retriever

import prompts


def get_rag_pipeline(retriever):
    """
    Создает и возвращает RAG пайплайн, который включает компоненты для ввода запроса,
    переписывания истории чатов, выполнения запроса через LLM и генерации ответа с учетом истории.

    Параметры:
    retriever (Any): Объект, используемый для поиска информации в базе данных или индексе.

    Возвращает:
    QueryPipeline: Пайплайн, состоящий из связанных между собой модулей для обработки входного запроса.
    """
    input_component = InputComponent()
    rewrite_template = PromptTemplate(prompts.REWRITE_HISTORY_PROMPT)

    response_component = ResponseWithChatHistory(
        llm=Settings.llm, system_prompt=prompts.SYSTEM_PROMPT
    )

    pipeline = QueryPipeline(
        modules={
            "input": input_component,
            "rewrite_template": rewrite_template,
            "rewrite_retriever": retriever,
            "llm": Settings.llm,
            "response_component": response_component,
        },
        verbose=False,
    )

    pipeline.add_link(
        "input", "rewrite_template", src_key="query_str", dest_key="query_str"
    )
    pipeline.add_link(
        "input",
        "rewrite_template",
        src_key="chat_history_str",
        dest_key="chat_history_str",
    )
    pipeline.add_link("rewrite_template", "llm")
    pipeline.add_link("llm", "rewrite_retriever")

    pipeline.add_link("rewrite_retriever", "response_component", dest_key="nodes")
    pipeline.add_link("llm", "response_component", dest_key="query_str")

    return pipeline


@dataclass
class PipelineDB:
    """
    Класс для хранения данных о выполнении запроса при прохождении через пайплайн.

    Атрибуты:
    input_text (str): Входной запрос от пользователя.
    question_variant (str): Переформулированный запрос для использования в модели LLM.
    context (List[str]): Контекст, переданный в модель LLM.
    answer (str): Сгенерированный ответ.
    """

    input_text: str = Field("None", description="input query")
    question_variant: str = Field("None", description="rewrited query")
    context: List[str] = Field(["None"], description="context for llm")
    answer: str = Field("None", description="answer from pipeline")


def get_response_with_routing(history_str, query_str):
    """
    Получает ответ от пайплайна с роутером на основе входного запроса и истории чата.

    Если роутер определяет, что запрос требует работы с LLM (результат не равен "0"),
    то создается пайплайн, и ответ формируется на основе его работы. В противном случае
    возвращается предопределенное сообщение, если запрос не относится к поддерживаемой тематике.

    Параметры:
    history_str (str): История чата с пользователем.
    query_str (str): Входной запрос пользователя.

    Возвращает:
    str: Ответ, сгенерированный системой (либо из пайплайна, либо предопределенный ответ).
    """
    router_prompt = prompts.ROUTER_PROMPT.format(query_str=query_str)
    router_res = Settings.llm.complete(router_prompt)

    if str(router_res) != "0":
        index = load_index()
        retriever = get_fusion_retriever(index)
        pipeline = get_rag_pipeline(retriever)

        response, intermediates = pipeline.run_with_intermediates(
            query_str=query_str, chat_history_str=history_str
        )

        to_bd = PipelineDB(
            input_text=intermediates["input"].inputs["query_str"],
            question_variant=intermediates["llm"].outputs["output"].text,
            context=[
                inter.get_content()
                for inter in intermediates["rewrite_retriever"].outputs["output"]
            ],
            answer=response["response"].text,
        )

        response["rewritten_query"] = to_bd.question_variant

    else:
        response = """Я - чат-бот для ответов на вопросы по руководству пользователя и не могу ответить на данный вопрос.
Если Ваш запрос соответствует тематике, попробуйте переформулировать его или обратиться к специалисту-человеку."""

    return response

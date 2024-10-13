from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.retrievers.fusion_retriever import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer

import torch
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import Any, Dict, List, Optional
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms import ChatMessage
from llama_index.core.query_pipeline import CustomQueryComponent
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import RecursiveRetriever

from llama_index.core import PromptTemplate
from dataclasses import dataclass

from llama_index.core.query_pipeline import QueryPipeline, InputComponent

import prompts
import joblib
import re


def completion_to_prompt(completion):
    return f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{completion}<|im_end|>\n<|im_start|>assistant\n"


def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|im_start|>system\n{message.content}<|im_end|>\n"
        elif message.role == "user":
            prompt += f"<|im_start|>user\n{message.content}<|im_end|>\n"
        elif message.role == "assistant":
            prompt += f"<|im_start|>assistant\n{message.content}<|im_end|>\n"

    if not prompt.startswith("<|im_start|>system"):
        prompt = "<|im_start|>system\n" + prompt

    prompt = prompt + "<|im_start|>assistant\n"

    return prompt


def init_global_settings():
    Settings.llm = HuggingFaceLLM(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
        context_window=30000,
        max_new_tokens=1000,
        generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        device_map="auto",
        model_kwargs={"torch_dtype": torch.float16},
    )

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-large", device="cuda"
    )


def load_index(index_path="ml/base_index"):
    storage_context = StorageContext.from_defaults(persist_dir=index_path)

    index_simple = load_index_from_storage(storage_context, use_async=True)

    return index_simple


def get_fusion_retriever(index, vector_top_k=10, bm25_top_k=10, total_top_k=3):
    vector_retriever_chunk = index.as_retriever(similarity_top_k=vector_top_k)

    all_nodes_dict = joblib.load("ml/all_nodes_dict.pkl")
    retriever_chunk = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever_chunk},
        node_dict=all_nodes_dict,
        verbose=True,
    )

    bm25_retriever = BM25Retriever.from_defaults(
        docstore=index.docstore,
        similarity_top_k=bm25_top_k,
        language="ru",
        stemmer=Stemmer.Stemmer("russian"),
    )

    retriever = QueryFusionRetriever(
        [retriever_chunk, bm25_retriever],
        similarity_top_k=total_top_k,
        num_queries=2,
        mode="reciprocal_rerank",
        use_async=True,
        verbose=True,
        retriever_weights=[0.8, 0.2],
        query_gen_prompt=prompts.QUERY_GEN_PROMPT,
    )

    return retriever


class ResponseWithChatHistory(CustomQueryComponent):
    llm: HuggingFaceLLM = Field(..., description="Local LLM")
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt to use for the LLM"
    )
    context_prompt: str = Field(
        default=prompts.DEFAULT_CONTEXT_PROMPT,
        description="Context prompt to use for the LLM",
    )

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        return input

    @property
    def _input_keys(self) -> set:
        """Input keys dict."""
        return {"nodes", "query_str"}

    @property
    def _output_keys(self) -> set:
        return {"response", "context"}

    def _prepare_context(
        self,
        nodes: List[NodeWithScore],
        query_str: str,
    ) -> List[ChatMessage]:
        node_context = ""
        for idx, node in enumerate(nodes):
            node_text = node.get_text()
            node_meta = node.dict()["node"]["metadata"]
            node_context += f"Контекст {idx + 1}: \n Источник: {node_meta['source']} \n {node_text}\n\n"

        formatted_context = self.context_prompt.format(
            node_context=node_context, query_str=query_str
        )

        return formatted_context, node_context

    def _run_component(self, **kwargs) -> Dict[str, Any]:
        """Run the component."""
        nodes = kwargs["nodes"]
        query_str = kwargs["query_str"]

        prepared_context, node_context = self._prepare_context(nodes, query_str)

        response = self.llm.complete(prepared_context)

        return {"response": response, "context": node_context}

    async def _arun_component(self, **kwargs: Any) -> Dict[str, Any]:
        """Run the component asynchronously."""
        nodes = kwargs["nodes"]
        query_str = kwargs["query_str"]

        prepared_context, node_context = self._prepare_context(nodes, query_str)

        response = await self.llm.acomplete(prepared_context)

        return {"response": response, "context": node_context}


def get_rag_pipeline(retriever):
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
    input_text: str = Field("None", description="input query")
    question_variant: str = Field("None", description="rewrited query")
    context: List[str] = Field(["None"], description="context for llm")
    answer: str = Field("None", description="answer from pipeline")


def contains_chinese(text):
    chinese_characters = re.compile(r"[\u4e00-\u9fff]+")
    return bool(chinese_characters.search(text))


def get_response(history_str, query_str):
    router_prompt = prompts.ROUTER_PROMPT.format(query_str=query_str)
    router_res = Settings.llm.complete(router_prompt)

    # print("ROUTER RESULT", router_res)
    if str(router_res) != "0":
        index = load_index()
        retriever = get_fusion_retriever(index)
        pipeline = get_rag_pipeline(retriever)

        response, intermediates = pipeline.run_with_intermediates(
            query_str=query_str, chat_history_str=history_str
        )

        if contains_chinese(response["response"].text):
            # print("CHINESE DETECTED")
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
        response = """Я - чат-бот для ответов на вопросы сотрудников ОАО РЖД по документу коллективного договора и не могу ответить на данный вопрос.
Если Ваш запрос соответствует тематике, попробуйте переформулировать его или обратиться к специалисту-человеку."""

    return response

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.retrievers.fusion_retriever import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import RecursiveRetriever

import prompts
import joblib


def load_embedder():
    """
    Загружает и конфигурирует энкодер модель с hugging face.

    Модель используется для создания эмбеддингов текста, которые будут применяться для поиска и извлечения информации.

    Возвращает:
    HuggingFaceEmbedding: Объект модели эмбеддингов.
    """
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-large", device="cpu"
    )

    return Settings.embed_model


def load_index(index_path="ml/base_index"):
    """
    Загружает индекс из хранилища для последующего использования в поисковых запросах.

    Параметры:
    index_path (str): Путь к директории с сохранённым индексом. По умолчанию: "ml/base_index".

    Возвращает:
    Index: Загруженный индекс для выполнения запросов.
    """
    storage_context = StorageContext.from_defaults(persist_dir=index_path)

    index_simple = load_index_from_storage(storage_context, use_async=True)

    return index_simple


def get_fusion_retriever(index, vector_top_k=10, bm25_top_k=10, total_top_k=3):
    """
    Создает и настраивает комбинированный ретривер, который объединяет методы векторного поиска и BM25 для более точного поиска информации.

    Параметры:
    index (Index): Индекс, из которого извлекается информация.
    vector_top_k (int): Количество топ-результатов для векторного поиска. По умолчанию: 10.
    bm25_top_k (int): Количество топ-результатов для поиска BM25. По умолчанию: 10.
    total_top_k (int): Общее количество результатов после объединения методов. По умолчанию: 3.

    Возвращает:
    QueryFusionRetriever: Ретривер, объединяющий результаты векторного поиска и BM25 для извлечения данных.
    """
    vector_retriever_chunk = index.as_retriever(similarity_top_k=vector_top_k)

    retriever_chunk = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever_chunk},
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
        retriever_weights=[0.85, 0.15],
        query_gen_prompt=prompts.QUERY_GEN_PROMPT,
    )

    return retriever

from langchain_community.llms import YandexGPT
from llama_index.core import Settings
from llama_index.llms.langchain import LangChainLLM


def load_yandex_gpt():
    Settings.llm = LangChainLLM(llm=YandexGPT(model_uri="gpt://b1gior9ks9q2nh5h1vgg/yandexgpt/rc",
                                              api_key='AQVN39tCt1NuCteBGxeF15d0G3fdYKuKW-OR-uP8'))

    return Settings.llm

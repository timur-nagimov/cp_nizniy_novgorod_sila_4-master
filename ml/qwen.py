import torch
from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM


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


def load_qwen():
    """
    Загружает и настраивает модель Qwen с помощью HuggingFaceLLM.
    Конфигурирует модель с заданными параметрами, включая название модели, окно контекста,
    количество новых токенов и параметры генерации.

    Возвращает:
    HuggingFaceLLM: Загруженная модель LLM, готовая для использования.
    """
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

    return Settings.llm

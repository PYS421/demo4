"""统一的本地 Ollama API 调用接口，兼容原有评估代码。"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(
    dotenv_path=Path(__file__).resolve().parent.parent / ".env",
    override=True,
)

DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_K_M"
OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"


def create_client():
    return OpenAI(
        api_key="ollama",
        base_url=OLLAMA_BASE_URL,
        timeout=300.0,
        max_retries=1,
    )


def call_llm(messages, model=None, temperature=0):
    model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)

    client = create_client()

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=int(os.getenv("OLLAMA_MAX_TOKENS", "2048")),
    )

    return {
        "requested_model": model,
        "actual_model": response.model,
        "content": response.choices[0].message.content or "",
        "usage": (
            response.usage.model_dump()
            if response.usage is not None
            else None
        ),
    }

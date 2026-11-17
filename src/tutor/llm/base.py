"""Абстракция провайдеров: ядро знает только эти интерфейсы, провайдер выбирается в конфиге."""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from tutor.config import Settings


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FakeEmbedder:
    """Детерминированные «эмбеддинги» по набору слов — для тестов без модели."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"[a-zа-яё0-9]+", text.lower()):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embedder(settings: Settings) -> Embedder:
    provider = settings.embeddings_provider.lower()
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=settings.embeddings_model, base_url=settings.ollama_base_url)
    if provider == "fake":
        return FakeEmbedder()
    raise ValueError(f"Неизвестный провайдер эмбеддингов: {settings.embeddings_provider}")


def get_chat_model(settings: Settings, *, schema: dict | None = None, temperature: float | None = None,
                   max_tokens: int | None = None, top_p: float | None = None, top_k: int | None = None,
                   repeat_penalty: float | None = None, seed: int | None = None):
    """Чат-модель LangChain. schema — JSON-схема ответа (structured output), если провайдер её умеет.
    Параметры сэмплинга, равные None, не передаются: тогда действуют значения Ollama по умолчанию."""
    provider = settings.llm_provider.lower()
    temp = settings.llm_temperature if temperature is None else temperature
    limit = max_tokens or settings.llm_max_tokens
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm_model, base_url=settings.ollama_base_url, temperature=temp,
            num_ctx=settings.llm_num_ctx, num_predict=limit, keep_alive=settings.llm_keep_alive, format=schema,
            top_p=top_p, top_k=top_k, repeat_penalty=repeat_penalty, seed=seed,
        )
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Для LLM_PROVIDER=openai установите пакет langchain-openai") from e
        model = ChatOpenAI(model=settings.llm_model, base_url=settings.openai_base_url,
                           api_key=settings.openai_api_key, temperature=temp, max_tokens=limit,
                           top_p=top_p, seed=seed)  # top_k и repeat_penalty в OpenAI-совместимом API нет
        if schema:
            return model.bind(response_format={"type": "json_schema",
                                               "json_schema": {"name": "answer", "schema": schema}})
        return model
    raise ValueError(f"Неизвестный провайдер LLM: {settings.llm_provider}")

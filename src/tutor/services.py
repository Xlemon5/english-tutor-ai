"""Сборка ядра агента из конфигурации — общая точка входа для CLI, API и eval."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from tutor.agent.quiz import QuizAgent
from tutor.agent.review import ReviewAgent
from tutor.agent.theory import TheoryAgent
from tutor.config import Settings, get_settings
from tutor.llm.base import get_chat_model, get_embedder
from tutor.llm.structured import StructuredCaller
from tutor.retrieval.hybrid import Retriever
from tutor.retrieval.store import ChunkStore

RESUME_TEMPERATURE = 0.3


@dataclass
class Services:
    settings: Settings
    store: ChunkStore
    retriever: Retriever
    theory: TheoryAgent
    review: ReviewAgent
    quiz: QuizAgent


def build_services(settings: Settings) -> Services:
    store = ChunkStore(settings)
    retriever = Retriever(store, get_embedder(settings), settings)
    # для JSON-сценариев температура 0 и запас по длине ответа
    structured = StructuredCaller(
        lambda schema: get_chat_model(settings, schema=schema, temperature=0.0, max_tokens=1500))
    return Services(
        settings=settings, store=store, retriever=retriever,
        # если ответ перешёл на китайский, его дописывает модель с небольшой температурой:
        # при температуре 0 она на том же месте снова выбрала бы тот же иероглиф
        theory=TheoryAgent(retriever, get_chat_model(settings),
                           resume_llm=get_chat_model(settings, temperature=RESUME_TEMPERATURE)),
        review=ReviewAgent(retriever, structured),
        quiz=QuizAgent(retriever, structured, use_fragments=settings.quiz_use_fragments),
    )


@lru_cache
def get_services() -> Services:
    return build_services(get_settings())

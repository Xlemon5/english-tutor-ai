"""Сборка ядра агента из конфигурации — общая точка входа для CLI, API и eval."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from tutor.agent.compose import ComposeAgent
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
    compose: ComposeAgent
    review: ReviewAgent
    quiz: QuizAgent


def build_services(settings: Settings) -> Services:
    store = ChunkStore(settings)
    retriever = Retriever(store, get_embedder(settings), settings)
    # для JSON-сценариев температура 0 и запас по длине ответа
    debug_dir = settings.debug_llm_dir if settings.debug_llm else None
    # проверка работ и судья свободных ответов — детерминированно (T=0)
    structured = StructuredCaller(
        lambda schema, **opts: get_chat_model(settings, schema=schema, temperature=0.0, max_tokens=1500, **opts),
        debug_dir=debug_dir)
    # генерация тестов — сэмплинг из конфига; seed задаёт агент на каждый раунд
    quiz_sampling = {"temperature": settings.quiz_temperature, "top_p": settings.quiz_top_p,
                     "top_k": settings.quiz_top_k, "repeat_penalty": settings.quiz_repeat_penalty,
                     "max_tokens": settings.quiz_max_tokens}
    quiz_structured = StructuredCaller(  # параметры вызова (seed, temperature) переопределяют значения из конфига
        lambda schema, **opts: get_chat_model(settings, schema=schema, **(quiz_sampling | opts)),
        debug_dir=debug_dir)
    return Services(
        settings=settings, store=store, retriever=retriever,
        # если ответ перешёл на китайский, его дописывает модель с небольшой температурой:
        # при температуре 0 она на том же месте снова выбрала бы тот же иероглиф
        theory=TheoryAgent(retriever, get_chat_model(settings),
                           resume_llm=get_chat_model(settings, temperature=RESUME_TEMPERATURE),
                           can_resume=settings.llm_provider.lower() == "ollama"),
        compose=ComposeAgent(retriever, get_chat_model(settings, temperature=settings.compose_temperature,
                                                       top_p=settings.quiz_top_p, top_k=settings.quiz_top_k),
                             can_resume=settings.llm_provider.lower() == "ollama"),
        review=ReviewAgent(retriever, structured),
        quiz=QuizAgent(retriever, quiz_structured, use_fragments=settings.quiz_use_fragments, judge=structured,
                       temperature=settings.quiz_temperature),
    )


@lru_cache
def get_services() -> Services:
    return build_services(get_settings())

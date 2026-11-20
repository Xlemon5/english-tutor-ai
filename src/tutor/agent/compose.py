"""Сценарий «пример текста»: модель пишет короткий текст на английском под уровень студента.

Это не ответ по материалам, поэтому правило «только по фрагментам» здесь не действует. Если поиск
нашёл подходящий текст в материалах (например, «Writing Passages › Weekend Activities»), он идёт
в промпт как образец длины и лексики и показывается студенту как источник.
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Literal

from pydantic import BaseModel, Field

from tutor.agent.theory import TheoryAgent, format_context
from tutor.llm.structured import ContextBudgetError, estimate_tokens
from tutor.prompts import load_prompt
from tutor.retrieval.citations import Citation, build_citations

DEFAULT_LEVEL = "A2"
MODEL_CHARS = 900  # образец целиком не нужен: хватает начала текста
COMPOSE_RULE = ("Текст пиши только по-английски, строку «Грамматика» — по-русски. "
                "Китайский язык и иероглифы не используй никогда.")
_MARK = re.compile(r"[ \t]*\[\d+\]")
_RULE_LINE = re.compile(r"^[ \t]*[-*_]{3,}[ \t]*$\n?", re.MULTILINE)  # разделитель «---» перед строкой о грамматике


class ComposeAnswer(BaseModel):
    status: Literal["answered"] = "answered"
    answer: str
    level: str
    citations: list[Citation] = Field(default_factory=list)  # текст-образец из материалов, если нашёлся


class ComposeEvent(BaseModel):
    type: Literal["token", "final"]
    text: str = ""
    result: ComposeAnswer | None = None


class ComposeAgent(TheoryAgent):
    """Берёт у TheoryAgent потоковую генерацию с защитой от перехода на китайский; промпт свой."""
    user_prompt = "compose_user"
    resume_rule = COMPOSE_RULE

    def stream(self, request: str, level: str | None = None) -> Iterator[ComposeEvent]:
        level = level or DEFAULT_LEVEL
        # образец ищем среди всех видов материалов: тексты для чтения и письма бывают не только в теории
        result = self.retriever.search(request, kinds=None, level=level)
        citations = build_citations(result.hits[:1]) if result.relevant else []
        s = self.retriever.settings
        system = load_prompt("compose_system").replace("{level}", level)
        template = load_prompt(self.user_prompt)
        context = ("Текст-образец из учебных материалов:\n" + format_context(citations, MODEL_CHARS)
                   if citations else "Текста-образца в материалах нет.")
        if estimate_tokens(system, template, context, request) + s.llm_max_tokens > s.llm_num_ctx:
            raise ContextBudgetError("Просьба не помещается в окно модели — сократите её.")
        text = ""
        for piece in self._pieces({"system": system, "context": context, "question": request}):
            text += piece
            yield ComposeEvent(type="token", text=piece)
        # номера ссылок модель ставить не должна; если поставила — убираем, образец показывается отдельно
        answer = re.sub(r"\n{3,}", "\n\n", _RULE_LINE.sub("", _MARK.sub("", text))).strip()
        yield ComposeEvent(type="final", result=ComposeAnswer(answer=answer, level=level, citations=citations))

    def compose(self, request: str, level: str | None = None) -> ComposeAnswer:
        final = None
        for event in self.stream(request, level):
            if event.type == "final":
                final = event.result
        assert final is not None
        return final

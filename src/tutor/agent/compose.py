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
MAX_EXAMPLES = 3  # примеров в строке «Грамматика»: модель иногда перечисляет весь текст с повторами
MODEL_CHARS = 900  # образец целиком не нужен: хватает начала текста
COMPOSE_RULE = ("Текст пиши только по-английски, строку «Грамматика» — по-русски. "
                "Китайский язык и иероглифы не используй никогда.")
_MARK = re.compile(r"[ \t]*\[\d+\]")
_RULE_LINE = re.compile(r"^[ \t]*[-*_]{3,}[ \t]*$\n?", re.MULTILINE)  # разделитель «---» перед строкой о грамматике
_SPEAKER = re.compile(r"^\**([A-Z][A-Za-z .'-]{0,24}?)\**:\**[ \t]+")  # «Customer: …», «**Anna:** …»
_GRAMMAR = re.compile(r"^\**Грамматика\**:", re.IGNORECASE)
_EXAMPLE = re.compile(r"\*([^*\n]+)\*")


def tidy_grammar(line: str, body: str) -> str:
    """Оставляет в строке «Грамматика» только примеры, которые есть в самом тексте: модель иногда
    приводит слова из примера в промпте (в диалоге в кафе — *wanted*, *noticed*, которых там нет)."""
    rule, sep, examples = line.partition(" — ")
    if not sep:
        return line
    found: list[str] = []
    for m in _EXAMPLE.finditer(examples):
        word = m.group(1).strip()
        in_text = re.search(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", body, re.IGNORECASE)
        if in_text and f"*{word}*" not in found and len(found) < MAX_EXAMPLES:
            found.append(f"*{word}*")
    return f"{rule} — {', '.join(found)}." if found else rule.rstrip(".") + "."


def format_answer(text: str) -> str:
    """Markdown для показа: в Markdown одиночный перенос строки склеивает строки в абзац, поэтому
    реплики диалога и строки письма шли сплошным текстом. Каждая строка остаётся отдельной,
    имя говорящего выделяется."""
    lines = text.strip().split("\n")
    grammar = next((i for i in range(len(lines) - 1, -1, -1) if _GRAMMAR.match(lines[i].strip())), None)
    body_lines = lines[:grammar] if grammar is not None else lines
    body = "\n".join(body_lines)
    out = [_SPEAKER.sub(r"**\1:** ", ln.strip()) for ln in body_lines]
    md = "\n".join(f"{ln}  " if ln and nxt else ln for ln, nxt in zip(out, [*out[1:], ""], strict=True)).strip()
    if grammar is not None:
        md += "\n\n" + tidy_grammar(lines[grammar].strip(), body)
    return md


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
        answer = format_answer(re.sub(r"\n{3,}", "\n\n", _RULE_LINE.sub("", _MARK.sub("", text))))
        yield ComposeEvent(type="final", result=ComposeAnswer(answer=answer, level=level, citations=citations))

    def compose(self, request: str, level: str | None = None) -> ComposeAnswer:
        final = None
        for event in self.stream(request, level):
            if event.type == "final":
                final = event.result
        assert final is not None
        return final

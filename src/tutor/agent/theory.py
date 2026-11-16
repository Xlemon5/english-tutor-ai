"""Сценарий «вопрос по теории»: ответ только по найденным фрагментам, иначе честный отказ."""
from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from tutor.prompts import load_prompt
from tutor.retrieval.citations import Citation, build_citations
from tutor.retrieval.hybrid import Retriever, SearchResult

NO_MATERIAL_TEXT = "В материалах этого нет."
NO_ANSWER = "NO_ANSWER"
_MARK = re.compile(r"\[(\d+)\]")
FRAGMENT_CHARS = 1000


class TheoryAnswer(BaseModel):
    status: Literal["answered", "no_material"]
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)  # близкие темы, которые в материалах есть
    top_score: float = 0.0


class TheoryEvent(BaseModel):
    type: Literal["token", "final"]
    text: str = ""
    result: TheoryAnswer | None = None


def format_context(citations: list[Citation], chars: int = FRAGMENT_CHARS) -> str:
    blocks = []
    for c in citations:
        head = " › ".join(p for p in (c.title, c.section) if p)
        blocks.append(f"[{c.n}] ({head})\n{c.text[:chars]}")
    return "\n\n".join(blocks)


def renumber(answer: str, citations: list[Citation], used: list[int]) -> tuple[str, list[Citation]]:
    """Перенумеровывает использованные ссылки подряд: [5] и [2] в ответе станут [1] и [2]."""
    mapping = {old: new for new, old in enumerate(used, 1)}
    answer = _MARK.sub(lambda m: f"[{mapping[int(m.group(1))]}]", answer)
    by_n = {c.n: c for c in citations}
    return answer, [by_n[old].model_copy(update={"n": new}) for old, new in mapping.items()]


def clean_markers(text: str, valid: set[int]) -> tuple[str, list[int]]:
    """Убирает ссылки на несуществующие фрагменты, возвращает использованные номера по порядку."""
    used: list[int] = []

    def repl(m: re.Match) -> str:
        n = int(m.group(1))
        if n not in valid:
            return ""
        if n not in used:
            used.append(n)
        return m.group(0)

    cleaned = _MARK.sub(repl, text)
    return re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned).strip(), used


def suggest_topics(result: SearchResult, limit: int = 3, threshold: float = 0.0) -> list[str]:
    """Близкие темы предлагаем, только если поиск нашёл что-то почти релевантное (скор от 90% порога)."""
    if result.top_score < threshold * 0.9:
        return []
    seen: list[str] = []
    for h in result.hits:
        name = " › ".join(p for p in (h.chunk.doc_title, h.chunk.section) if p)
        if name not in seen:
            seen.append(name)
    return seen[:limit]


class TheoryAgent:
    def __init__(self, retriever: Retriever, llm) -> None:
        self.retriever = retriever
        prompt = ChatPromptTemplate.from_messages(
            [("system", "{system}"), ("human", load_prompt("theory_user"))])
        self.chain = prompt | llm | StrOutputParser()

    def _refuse(self, result: SearchResult, with_suggestions: bool) -> TheoryAnswer:
        return TheoryAnswer(
            status="no_material", answer=NO_MATERIAL_TEXT, top_score=result.top_score,
            suggestions=suggest_topics(result, threshold=self.retriever.settings.relevance_threshold)
            if with_suggestions else [],
        )

    def stream(self, question: str, level: str | None = None) -> Iterator[TheoryEvent]:
        result = self.retriever.search(question, level=level)
        if not result.relevant:
            yield TheoryEvent(type="final", result=self._refuse(result, True))
            return
        citations = build_citations(result.hits)
        context = format_context(citations, self.retriever.settings.context_fragment_chars)
        stream = self.chain.stream({
            "system": load_prompt("theory_system"), "context": context, "question": question})
        text, emitted = "", 0
        hold = len(NO_ANSWER) + 2  # не показываем начало ответа, пока не ясно, что это не отказ
        for piece in stream:
            text += piece
            if NO_ANSWER.startswith(text.strip()[:len(NO_ANSWER)]) and len(text.strip()) < hold:
                continue
            if NO_ANSWER in text[:hold + 4]:
                break
            yield TheoryEvent(type="token", text=text[emitted:])
            emitted = len(text)
        if NO_ANSWER in text or not text.strip():
            yield TheoryEvent(type="final", result=self._refuse(result, True))
            return
        answer, used = clean_markers(text, {c.n for c in citations})
        # если модель не поставила ни одной ссылки, показываем все фрагменты, на которых строился ответ
        cited = citations
        if used:
            answer, cited = renumber(answer, citations, used)
        yield TheoryEvent(type="final", result=TheoryAnswer(
            status="answered", answer=answer, citations=cited, top_score=result.top_score))

    def answer(self, question: str, level: str | None = None) -> TheoryAnswer:
        final = None
        for event in self.stream(question, level):
            if event.type == "final":
                final = event.result
        assert final is not None
        return final

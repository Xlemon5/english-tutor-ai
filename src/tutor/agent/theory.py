"""Сценарий «вопрос по теории»: ответ только по найденным фрагментам, иначе честный отказ."""
from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field

from tutor.llm.language import FOREIGN_SCRIPT, LANGUAGE_RULE
from tutor.llm.structured import ContextBudgetError, estimate_tokens
from tutor.prompts import load_prompt
from tutor.retrieval.citations import Citation, build_citations
from tutor.retrieval.hybrid import Retriever, SearchResult

NO_MATERIAL_TEXT = "В материалах этого нет."
NO_ANSWER = "NO_ANSWER"
_MARK = re.compile(r"\[(\d+)\]")
FRAGMENT_CHARS = 1000
MAX_RESUMES = 2  # сколько раз продолжать ответ после перехода модели на китайский
TRUNCATED_NOTE = "\n\n(Ответ оборвался: модель упёрлась в лимит длины. Задайте вопрос уже.)"


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
    def __init__(self, retriever: Retriever, llm, resume_llm=None, can_resume: bool = True) -> None:
        """resume_llm — модель, которая дописывает ответ, если основная перешла на китайский
        (по умолчанию та же; в сервисах — с небольшой температурой, чтобы не повторить тот же сбой)."""
        self.retriever = retriever
        # продолжение начатого ответа умеет Ollama; OpenAI-совместимые серверы (vLLM) начали бы ответ заново
        self.resumes = MAX_RESUMES if can_resume else 0
        messages = [("system", "{system}"), ("human", load_prompt("theory_user"))]
        # без StrOutputParser: по последнему куску видно done_reason (обрыв по лимиту длины)
        self.chain = ChatPromptTemplate.from_messages(messages) | llm
        # последнее сообщение — начало ответа ассистента: модель продолжает его, а не начинает заново
        self.resume_chain = ChatPromptTemplate.from_messages([*messages, ("ai", "{prefix}")]) | (resume_llm or llm)

    def _pieces(self, inputs: dict) -> Iterator[str]:
        """Текст ответа модели по кускам, без иероглифов. Если модель перешла на китайский, генерация
        обрывается и продолжается с последнего чистого места с явным запретом (до MAX_RESUMES раз);
        не получилось — ответ заканчивается на этом месте."""
        text = ""
        for attempt in range(self.resumes + 1):
            if attempt == 0:
                source = self.chain.stream(inputs)
            else:
                source = self.resume_chain.stream(
                    {**inputs, "system": inputs["system"] + "\n" + LANGUAGE_RULE, "prefix": text.rstrip()})
            dangling = text[len(text.rstrip()):]  # пробел в конце уже показан — второй не нужен
            tail = ""  # недописанное слово держим, пока не ясно, что за ним не иероглифы
            switched = None
            truncated = False
            for chunk in source:
                piece = chunk.content if hasattr(chunk, "content") else str(chunk)
                meta = getattr(chunk, "response_metadata", None) or {}
                truncated = truncated or (meta.get("done_reason") or meta.get("finish_reason")) == "length"
                if dangling and piece:
                    piece, dangling = piece.lstrip(" "), ""
                tail += piece
                if m := FOREIGN_SCRIPT.search(tail):
                    switched = tail[:m.start()]
                    break
                cut = max(tail.rfind(" "), tail.rfind("\n")) + 1
                if cut:
                    text += tail[:cut]
                    yield tail[:cut]
                    tail = tail[cut:]
            if switched is None:
                if tail:
                    yield tail
                if truncated:  # лимит длины — не нормальный конец ответа, говорим об этом прямо
                    logger.warning("ответ по теории оборвался на лимите длины")
                    yield TRUNCATED_NOTE
                return
            source.close()  # обрываем генерацию: дальше модель продолжила бы по-китайски
            logger.warning("ответ перешёл на китайский после «{}» (сбой {} из {})",
                           (text + switched)[-60:], attempt + 1, self.resumes + 1)
            if switched:
                text += switched
                yield switched
        logger.warning("модель не вернулась к русскому — ответ закончен на последнем чистом месте")

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
        s = self.retriever.settings
        system, template = load_prompt("theory_system"), load_prompt("theory_user")
        context = format_context(citations, s.context_fragment_chars)
        # окно: промпт плюс максимальный ответ должны поместиться до вызова; лишние фрагменты отбрасываются
        while len(citations) > 1 and estimate_tokens(system, template, context, question) + s.llm_max_tokens > s.llm_num_ctx:
            citations = citations[:-1]
            context = format_context(citations, s.context_fragment_chars)
        if estimate_tokens(system, template, context, question) + s.llm_max_tokens > s.llm_num_ctx:
            raise ContextBudgetError("Вопрос вместе с материалами не помещается в окно модели — сократите вопрос.")
        stream = self._pieces({"system": system, "context": context, "question": question})
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

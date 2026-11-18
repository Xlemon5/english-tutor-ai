"""Сценарий «проверка работы»: LLM находит ошибки, код привязывает их к тексту и к теории из материалов."""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from tutor.llm.language import FOREIGN_SCRIPT_ERROR, TEXT_PATTERN, has_foreign_script
from tutor.llm.structured import StructuredCaller
from tutor.prompts import load_prompt
from tutor.retrieval.citations import Citation, build_citations
from tutor.retrieval.hybrid import Retriever

# Текст попадает в промпт дважды (целиком и по предложениям); вместе с системным промптом и ответом до 1500
# токенов это должно помещаться в окно 4096. B1-эссе — 150–250 слов, ~1400 символов.
MAX_TEXT_CHARS = 2000
NOT_IN_MATERIALS = "в материалах не найдено"
ErrorType = Literal["grammar", "vocabulary", "spelling", "punctuation", "style"]
ERROR_TYPE_RU = {"grammar": "грамматика", "vocabulary": "лексика", "spelling": "орфография",
                 "punctuation": "пунктуация", "style": "стиль"}


class LLMError(BaseModel):
    fragment: str = Field(min_length=1, max_length=200)
    correction: str = Field(max_length=200)
    type: ErrorType
    explanation: str = Field(min_length=3, max_length=400)
    rule: str = Field(min_length=2, max_length=80)


class LLMReview(BaseModel):
    """То, что возвращает модель (JSON по схеме)."""

    errors: list[LLMError] = Field(max_length=30)
    summary: str = Field(min_length=5, max_length=600)
    cefr_level: Literal["A1", "A2", "B1", "B2"]
    review_topics: list[str] = Field(max_length=3)

    @classmethod
    def llm_schema(cls) -> dict:
        """Схема для ограниченного декодирования с теми же разрешёнными символами, что у тестов: при T=0 на
        текстах уровня A1 модель писала объяснения по-китайски в 7 вызовах из 8 (eval/other_paths_cjk.py)."""
        text = {"type": "string", "pattern": TEXT_PATTERN}
        error = {"type": "object", "properties": {
            "fragment": {"type": "string"}, "correction": {"type": "string"},  # цитаты из текста ученика "type": {"type": "string", "enum": list(ERROR_TYPE_RU)},
            "explanation": text, "rule": text}, "required": ["fragment", "correction", "type", "explanation", "rule"]}
        return {"type": "object", "properties": {
            "errors": {"type": "array", "items": error, "maxItems": 30}, "summary": text,
            "cefr_level": {"type": "string", "enum": ["A1", "A2", "B1", "B2"]},
            "review_topics": {"type": "array", "items": text, "maxItems": 3}},
            "required": ["errors", "summary", "cefr_level", "review_topics"]}

    @model_validator(mode="after")
    def check_language(self) -> LLMReview:
        if has_foreign_script(self.model_dump()):  # ошибка уйдёт модели, и она перепишет ответ
            raise ValueError(FOREIGN_SCRIPT_ERROR)
        return self


class ReviewedError(LLMError):
    type_ru: str
    start: int | None = None  # позиция фрагмента в тексте студента (для подсветки)
    end: int | None = None
    citation: Citation | None = None
    theory_note: str = ""  # «в материалах не найдено», если ссылки нет


class ReviewResult(BaseModel):
    text: str
    task: str = ""
    errors: list[ReviewedError]
    summary: str
    grade: int  # 2–5, считается кодом по плотности ошибок
    cefr_level: str
    review_topics: list[str]


def locate(text: str, fragment: str, taken: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Ищет фрагмент в тексте: точно, затем без учёта регистра и лишних пробелов; занятые места пропускает."""
    pattern = r"\s+".join(re.escape(w) for w in fragment.split())
    for flags in (0, re.IGNORECASE):
        for m in re.finditer(pattern, text, flags):
            if all(m.end() <= s or m.start() >= e for s, e in taken):
                return m.start(), m.end()
    return None


def split_error(text: str, e: LLMError) -> list[tuple[LLMError, tuple[int, int] | None]]:
    """Модель часто цитирует предложение целиком. Сравниваем его с исправлением по словам и
    оставляем только различающиеся места — каждое становится отдельной подсвеченной ошибкой."""
    span = locate(text, e.fragment, [])
    src = text[span[0]:span[1]] if span else e.fragment
    toks = [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", src)]
    a, b = [t[0] for t in toks], e.correction.split()
    ops = [op for op in SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op[0] != "equal"]
    if not ops or len(a) <= 3 or not b:
        return [(e, span)]
    out = []
    for _tag, i1, i2, j1, j2 in ops:
        if i1 == i2 or j1 == j2:  # вставка или удаление слова: добавляем соседнее слово как опору
            if i2 < len(a) and j2 < len(b):
                i2, j2 = i2 + 1, j2 + 1
            elif i1 > 0 and j1 > 0:
                i1, j1 = i1 - 1, j1 - 1
        if i1 == i2:
            continue
        start, end = toks[i1][1], toks[i2 - 1][2]
        frag, corr = src[start:end], " ".join(b[j1:j2])
        while frag and corr and frag[-1] in ".,!?;:" and frag[-1] == corr[-1]:
            frag, corr, end = frag[:-1], corr[:-1], end - 1
        if not frag or frag == corr:
            continue
        sub = (span[0] + start, span[0] + end) if span else None
        out.append((e.model_copy(update={"fragment": frag, "correction": corr}), sub))
    return out or [(e, span)]


def grade_for(errors: int, text: str) -> int:
    words = max(len(re.findall(r"[A-Za-z']+", text)), 1)
    per_100 = errors / words * 100
    return 5 if per_100 <= 2 else 4 if per_100 <= 6 else 3 if per_100 <= 12 else 2


class ReviewAgent:
    def __init__(self, retriever: Retriever, structured: StructuredCaller) -> None:
        self.retriever = retriever
        self.structured = structured

    def _theory(self, rule: str, cache: dict[str, Citation | None]) -> Citation | None:
        key = rule.lower().strip()
        if key not in cache:
            result = self.retriever.search(rule, k=3)
            cache[key] = build_citations(result.hits[:1])[0] if result.relevant and result.hits else None
        return cache[key]

    def review(self, text: str, task: str = "") -> ReviewResult:
        text = text.strip()
        if not text:
            raise ValueError("Текст работы пуст.")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"Текст слишком длинный: {len(text)} символов при лимите {MAX_TEXT_CHARS}.")
        sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", text) if x.strip()]
        user = load_prompt("review_user").format(
            task=task.strip() or "(не указано)", text=text,
            sentences="\n".join(f"{i}. {x}" for i, x in enumerate(sentences, 1)))
        raw = self.structured(LLMReview, load_prompt("review_system"), user)

        errors: list[ReviewedError] = []
        taken: list[tuple[int, int]] = []
        cache: dict[str, Citation | None] = {}
        for item in raw.errors:
            if item.fragment.strip() == item.correction.strip():
                continue  # модель «исправила» правильное на то же самое — это не ошибка
            for e, span in split_error(text, item):
                if span and any(span[0] < t[1] and t[0] < span[1] for t in taken):
                    span = locate(text, e.fragment, taken)
                if span:
                    taken.append(span)
                citation = self._theory(e.rule, cache)
                errors.append(ReviewedError(
                    **e.model_dump(), type_ru=ERROR_TYPE_RU[e.type],
                    start=span[0] if span else None, end=span[1] if span else None,
                    citation=citation, theory_note="" if citation else NOT_IN_MATERIALS,
                ))
        errors.sort(key=lambda x: (x.start is None, x.start or 0))
        for n, err in enumerate(c for c in errors if c.citation):
            err.citation = err.citation.model_copy(update={"n": n + 1})  # type: ignore[union-attr]
        return ReviewResult(
            text=text, task=task.strip(), errors=errors, summary=raw.summary,
            grade=grade_for(len(errors), text), cefr_level=raw.cefr_level, review_topics=raw.review_topics[:3],
        )

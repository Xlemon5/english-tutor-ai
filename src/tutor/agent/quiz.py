"""Сценарий «тест»: генерация по материалам, проверка ответов студента и разбор."""
from __future__ import annotations

import re
import uuid
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field, model_validator

from tutor.agent.theory import NO_MATERIAL_TEXT, format_context, suggest_topics
from tutor.lexicon import load_vocab, words_above_level
from tutor.llm.structured import StructuredCaller, StructuredOutputError
from tutor.prompts import load_prompt
from tutor.retrieval.citations import Citation, build_citations
from tutor.retrieval.hybrid import Retriever

QuestionType = Literal["multiple_choice", "fill_gap", "error_correction", "translation"]
TYPE_RU = {"multiple_choice": "выбор ответа", "fill_gap": "заполнение пропусков",
           "error_correction": "исправление ошибок", "translation": "перевод предложений"}
MAX_QUESTIONS = 15
BATCH = 4
MAX_ROUNDS = 5
MAX_HARD_WORDS = 1  # сколько слов выше уровня допускается в одном задании


class LLMQuestion(BaseModel):
    type: QuestionType
    prompt: str = Field(min_length=3, max_length=300)
    options: list[str] = Field(default_factory=list, max_length=4)
    answer: str = Field(min_length=1, max_length=300)
    explanation: str = Field(min_length=3, max_length=400)
    fragment: int = 0

    @model_validator(mode="after")
    def check_format(self) -> LLMQuestion:
        self.prompt = re.sub(r"_{3,}", "____", self.prompt).strip()
        self.options = [o.strip() for o in self.options if o.strip()]
        self.answer = self.answer.strip()
        if self.type == "multiple_choice":
            if not 3 <= len(self.options) <= 4 or len({o.lower() for o in self.options}) != len(self.options):
                raise ValueError("multiple_choice: нужно 3–4 разных варианта")
            if self.answer.lower() not in {o.lower() for o in self.options}:
                raise ValueError("multiple_choice: ответ должен совпадать с одним из вариантов")
        else:
            self.options = []
        if self.type in ("multiple_choice", "fill_gap") and self.prompt.count("____") != 1:
            raise ValueError("в задании с пропуском должен быть ровно один пропуск ____")
        if self.type == "error_correction" and normalize(self.prompt) == normalize(self.answer):
            raise ValueError("error_correction: исправленное предложение совпадает с исходным")
        if self.type == "translation" and not re.search(r"[а-яё]", self.prompt, re.IGNORECASE):
            raise ValueError("translation: задание должно быть на русском")
        return self


class LLMQuiz(BaseModel):
    questions: list[dict]  # вопросы проверяются по одному, чтобы один плохой не губил весь ответ


class Question(LLMQuestion):
    id: int
    type_ru: str
    citation: Citation | None = None


class Quiz(BaseModel):
    id: str
    status: Literal["ready", "no_material"] = "ready"
    topic: str
    level: str
    grounded: bool = True  # False — тест составлен без опоры на материалы (по явному согласию)
    questions: list[Question] = Field(default_factory=list)
    message: str = ""
    suggestions: list[str] = Field(default_factory=list)


class Judgement(BaseModel):
    correct: bool
    comment: str = Field(max_length=300)


class QuestionResult(BaseModel):
    question: Question
    given: str
    correct: bool
    comment: str = ""


class QuizResult(BaseModel):
    quiz_id: str
    topic: str
    level: str
    score: int
    total: int
    results: list[QuestionResult]


_CONTRACTIONS = (
    (r"\bwon't\b", "will not"), (r"\bcan't\b", "cannot"), (r"\bcan not\b", "cannot"), (r"\bshan't\b", "shall not"),
    (r"n't\b", " not"), (r"\bi'm\b", "i am"), (r"'re\b", " are"), (r"'ve\b", " have"), (r"'ll\b", " will"),
    (r"\b(he|she|it|that|there|what|who|where)'s\b", r"\1 is"), (r"\blet's\b", "let us"),
)


def normalize(text: str) -> str:
    """Приводит ответ к виду для сравнения: регистр, знаки препинания, лишние пробелы и
    сокращения (don't = do not) на результат проверки не влияют."""
    text = text.casefold().replace("’", "'").replace("‘", "'").replace("`", "'").replace("ё", "е")
    for pattern, full in _CONTRACTIONS:
        text = re.sub(pattern, full, text)
    text = re.sub(r"[^\w' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def plan_types(n: int, types: list[str]) -> list[str]:
    return [types[i % len(types)] for i in range(n)]


class QuizAgent:
    def __init__(self, retriever: Retriever, structured: StructuredCaller) -> None:
        self.retriever = retriever
        self.structured = structured

    def generate(self, topic: str, level: str = "A2", n: int = 5, types: list[str] | None = None,
                 allow_without_materials: bool = False) -> Quiz:
        topic = topic.strip()
        types = [t for t in (types or list(TYPE_RU)) if t in TYPE_RU] or list(TYPE_RU)
        n = max(1, min(n, MAX_QUESTIONS))
        quiz_id = uuid.uuid4().hex[:12]
        search = self.retriever.search(topic, k=4, level=level)
        if not search.relevant:  # на запрошенном уровне нет — пробуем любые уровни
            search = self.retriever.search(topic, k=4)
        citations: list[Citation] = []
        if search.relevant:
            citations = build_citations(search.hits)
        elif not allow_without_materials:
            return Quiz(
                id=quiz_id, status="no_material", topic=topic, level=level, grounded=False,
                message=f"{NO_MATERIAL_TEXT} По теме «{topic}» материалов нет. Сгенерировать тест без опоры на материалы?",
                suggestions=suggest_topics(search, threshold=self.retriever.settings.relevance_threshold),
            )
        context = format_context(citations) if citations else "(фрагментов нет — составь вопросы по теме сам)"
        vocab = load_vocab(str(self.retriever.settings.materials_dir))

        accepted: list[LLMQuestion] = []
        seen: set[str] = set()
        wanted = plan_types(n, types)
        for _ in range(MAX_ROUNDS):
            missing = list(wanted)
            for q in accepted:  # чего не хватает — считаем по типам, а не по позициям
                missing.remove(q.type)
            missing = missing[:BATCH]
            if not missing:
                break
            avoid = ""
            if accepted:
                avoid = "Не повторяй эти задания:\n" + "\n".join(f"- {q.prompt}" for q in accepted) + "\n"
            user = load_prompt("quiz_user").format(
                topic=topic, level=level, plan="\n".join(f"- {t}" for t in missing), avoid=avoid, context=context)
            try:
                raw = self.structured(LLMQuiz, load_prompt("quiz_system"), user)
            except StructuredOutputError as e:
                logger.warning("генерация теста: {}", e)
                continue
            pending = list(missing)
            for item in raw.questions:
                try:
                    q = LLMQuestion.model_validate(item)
                except Exception as e:  # noqa: BLE001
                    logger.info("вопрос отброшен: {}", str(e)[:200])
                    continue
                key = normalize(q.prompt)
                english = q.answer + " " + " ".join(q.options) + (" " + q.prompt if q.type != "translation" else "")
                if q.type not in pending or key in seen:
                    continue
                if len(words_above_level(english, level, vocab)) > MAX_HARD_WORDS:
                    logger.info("вопрос отброшен (лексика выше {}): {}", level, q.prompt)
                    continue
                pending.remove(q.type)
                seen.add(key)
                accepted.append(q)

        by_n = {c.n: c for c in citations}
        questions = []
        accepted.sort(key=lambda q: wanted.index(q.type))  # порядок типов как в запросе (сортировка устойчива)
        for i, q in enumerate(accepted[:n], 1):
            citation = by_n.get(q.fragment) or (citations[0] if citations else None)
            questions.append(Question(**q.model_dump(), id=i, type_ru=TYPE_RU[q.type], citation=citation))
        if not questions:
            raise StructuredOutputError("Не удалось составить ни одного корректного вопроса.")
        return Quiz(id=quiz_id, topic=topic, level=level, grounded=bool(citations), questions=questions)

    def check(self, quiz: Quiz, answers: dict[int, str]) -> QuizResult:
        results = []
        for q in quiz.questions:
            given = (answers.get(q.id) or "").strip()
            variants = {normalize(v) for v in q.answer.split(" / ")} | {normalize(q.answer)}
            correct = bool(given) and normalize(given) in variants
            comment = ""
            if given and not correct and q.type in ("error_correction", "translation"):
                # свободный ответ может быть верным, даже если не совпал с эталоном
                user = f"Задание: {q.prompt}\nЭталон: {q.answer}\nОтвет студента: {given}\nОтвет:"
                try:
                    verdict = self.structured(Judgement, load_prompt("judge_system"), user)
                    correct, comment = verdict.correct, verdict.comment
                except StructuredOutputError:
                    comment = "Автоматическая проверка не удалась, ответ сравнён с эталоном."
            results.append(QuestionResult(question=q, given=given, correct=correct, comment=comment))
        return QuizResult(quiz_id=quiz.id, topic=quiz.topic, level=quiz.level,
                          score=sum(r.correct for r in results), total=len(results), results=results)

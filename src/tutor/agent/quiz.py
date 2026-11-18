"""Сценарий «тест»: генерация по материалам, проверка ответов студента и разбор."""
from __future__ import annotations

import json
import random
import re
import uuid
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field, ValidationError, model_validator

from tutor.agent.theory import FRAGMENT_CHARS, NO_MATERIAL_TEXT, format_context, suggest_topics
from tutor.agent.topics import above_level, match_topic
from tutor.lexicon import load_vocab, words_above_level
from tutor.llm.language import FOREIGN_SCRIPT_ERROR, TEXT_PATTERN, has_foreign_script
from tutor.llm.structured import ContextBudgetError, StructuredCaller, StructuredOutputError, estimate_tokens
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
REPEAT_TEMPERATURE_STEP = 0.15  # на сколько поднять температуру, если модель повторила ответ прошлого раунда
MAX_TEMPERATURE = 1.0
WRITING_TYPES = ("multiple_choice", "error_correction")  # перевод и пропуски плохо проверяют темы по письму
SEARCH_K = 8  # кандидатов из поиска; после подъёма фрагментов нужной темы в промпт идут 4
MAX_AVOID = 10  # сколько уже принятых заданий перечислять в «не повторяй»
MIN_FRAGMENT_CHARS = 400

_TEXT = {"type": "string", "pattern": TEXT_PATTERN}
QUIZ_LLM_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {
        "type": "object",
        "properties": {"type": {"type": "string", "enum": ["multiple_choice", "fill_gap", "error_correction",
                                                          "translation"]},
                       "prompt": _TEXT, "options": {"type": "array", "items": _TEXT, "maxItems": 4},
                       "answer": _TEXT, "explanation": _TEXT, "fragment": {"type": "integer"}},
        "required": ["type", "prompt", "options", "answer", "explanation"]}}},
    "required": ["questions"],
}


class LLMQuestion(BaseModel):
    type: QuestionType
    prompt: str = Field(min_length=3, max_length=300)
    options: list[str] = Field(default_factory=list, max_length=4)
    answer: str = Field(min_length=1, max_length=300)
    explanation: str = Field(min_length=3, max_length=400)
    fragment: int = 0

    @model_validator(mode="after")
    def check_format(self) -> LLMQuestion:
        if type(self) is not LLMQuestion:
            return self  # Question из сохранённого теста: правила генерации к нему не применяются
        if has_foreign_script([self.prompt, self.options, self.answer, self.explanation]):
            raise ValueError(FOREIGN_SCRIPT_ERROR)
        self.prompt = re.sub(r"_{3,}", "____", self.prompt).strip()
        self.prompt = re.sub(r"\*\*(.+?)\*\*", r"\1", self.prompt)  # выделение из фрагментов Википедии
        self.options = [o.strip() for o in self.options if o.strip()]
        self.answer = self.answer.strip()
        if self.type == "multiple_choice" and "____" not in self.prompt and self.answer:
            # модель часто копирует пример из фрагмента вместе с ответом — ставим пропуск на его место;
            # варианты остаются подсказкой, поэтому вопрос не портится
            pattern = re.compile(r"(?<!\w)" + re.escape(self.answer) + r"(?!\w)", re.IGNORECASE)
            if len(pattern.findall(self.prompt)) == 1:
                self.prompt = pattern.sub("____", self.prompt)
        if self.type == "multiple_choice":
            if not 3 <= len(self.options) <= 4 or len({normalize(o) for o in self.options}) != len(self.options):
                raise ValueError("multiple_choice: нужно 3–4 разных варианта")
            same = [o for o in self.options if normalize(o) == normalize(self.answer)]
            if not same:
                raise ValueError("multiple_choice: ответ должен совпадать с одним из вариантов")
            self.answer = same[0]  # «Has been waiting.» и «has been waiting» — один вариант
        else:
            self.options = []
        if self.type in ("multiple_choice", "fill_gap") and self.prompt.count("____") != 1:
            raise ValueError("в задании с пропуском должен быть ровно один пропуск ____")
        if self.type == "error_correction" and normalize(self.prompt) == normalize(self.answer):
            raise ValueError("error_correction: исправленное предложение совпадает с исходным")
        if self.type == "translation" and not re.search(r"[а-яё]", self.prompt, re.IGNORECASE):
            raise ValueError("translation: задание должно быть на русском")
        check_content(self)
        return self


_CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)


def outside_brackets(text: str) -> str:
    return re.sub(r"\([^)]*\)", " ", text)


def check_content(q: LLMQuestion) -> None:
    """Дефекты, которые Qwen 7B допускает чаще всего и которые видны без модели (по разбору тестов на 16 темах)."""
    if q.type == "translation":
        if _CYRILLIC.search(q.answer):
            raise ValueError("translation: ответ — перевод на английский, без русских слов")
    elif any(_CYRILLIC.search(t) for t in (outside_brackets(q.prompt), q.answer, *q.options)):
        # подсказка в скобках может быть русской: (дождь)
        raise ValueError(f"{q.type}: задание и ответ на английском, по-русски только перевод и объяснение")
    if not _CYRILLIC.search(q.explanation):
        raise ValueError("explanation: объяснение пиши по-русски")
    if q.type == "fill_gap" and not re.search(r"\([^)]+\)", q.prompt):
        raise ValueError("fill_gap: после пропуска нужна подсказка в скобках, например (go)")
    if q.type in ("multiple_choice", "fill_gap"):
        before, _, after = outside_brackets(q.prompt).partition("____")
        if len(q.answer) >= 4 and re.search(r"(?<!\w)" + re.escape(q.answer) + r"(?!\w)", before + " " + after, re.IGNORECASE):
            raise ValueError("ответ уже написан в задании — в нём должен остаться только пропуск ____")
        words = q.answer.split()
        prev, nxt = before.split()[-1:], after.split()[:1]
        if words and ((prev and prev[0].lower() == words[0].lower())
                      or (nxt and nxt[0].strip(".,!?").lower() == words[-1].strip(".,!?").lower())):
            raise ValueError("после подстановки ответа слово повторяется дважды — пропуск стоит не там")


class LLMQuiz(BaseModel):
    questions: list[dict]  # вопросы проверяются по одному, чтобы один плохой не губил весь ответ

    @classmethod
    def llm_schema(cls) -> dict:
        """Схема для ограниченного декодирования: поля вопроса и разрешённые символы (см. TEXT_PATTERN)."""
        return QUIZ_LLM_SCHEMA


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

    @model_validator(mode="after")
    def check_language(self) -> Judgement:
        if has_foreign_script(self.comment):
            raise ValueError(FOREIGN_SCRIPT_ERROR)
        return self


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


def question_key(prompt: str) -> str:
    """Ключ для поиска повторов: одно и то же предложение с разными пояснениями в скобках — повтор."""
    return normalize(re.sub(r"\([^)]*\)", " ", prompt))


def example_keys(system: str) -> set[str]:
    """Ключи заданий из примера в инструкции: маленькая модель иногда переносит их в тест на другую тему."""
    return {question_key(p) for p in re.findall(r'"prompt": "([^"]+)"', system)}


def covered(search, info: dict | None) -> bool:
    """Тема есть в материалах: поиск уверен, или среди найденного есть фрагменты с пометкой этой темы по карте."""
    return search.relevant or bool(info and any(info["id"] in h.chunk.topics for h in search.hits))


def rank_by_topic(hits: list, info: dict | None) -> list:
    """Сначала фрагменты, чей заголовок подходит под тему карты, потом помеченные ею файлы, потом остальное."""
    if not info:
        return hits
    heading = [h for h in hits if info["re"].search(f"{h.chunk.doc_title} {h.chunk.section}")]
    tagged = [h for h in hits if info["id"] in h.chunk.topics and h not in heading]
    return heading + tagged + [h for h in hits if h not in heading and h not in tagged]


def plan_types(n: int, types: list[str]) -> list[str]:
    return [types[i % len(types)] for i in range(n)]


class QuizAgent:
    def __init__(self, retriever: Retriever, structured: StructuredCaller, use_fragments: bool = True,
                 judge: StructuredCaller | None = None, temperature: float = 0.7) -> None:
        """structured — генерация вопросов (с сэмплингом), judge — оценка свободных ответов студента
        (по умолчанию тот же вызывающий; в приложении — детерминированный).
        use_fragments=False — модель составляет предложения сама, без фрагментов в промпте;
        поиск всё равно проверяет, что тема есть в материалах, и даёт ссылку на теорию."""
        self.retriever = retriever
        self.structured = structured
        self.judge = judge or structured
        self.use_fragments = use_fragments
        self.temperature = temperature  # температура генерации; от неё считается повышение при повторах

    def generate(self, topic: str, level: str = "A2", n: int = 5, types: list[str] | None = None,
                 allow_without_materials: bool = False, seed: int | None = None) -> Quiz:
        """seed — база для seed раундов (в eval фиксированная, в приложении случайная). У каждого раунда свой
        seed, поэтому повторный раунд никогда не отправляет модели побайтно тот же запрос."""
        topic = topic.strip()
        types = [t for t in (types or list(TYPE_RU)) if t in TYPE_RU] or list(TYPE_RU)
        n = max(1, min(n, MAX_QUESTIONS))
        quiz_id = uuid.uuid4().hex[:12]
        info = match_topic(topic, self.retriever.settings.materials_dir)
        notes = []
        if info and above_level(info, level):
            notes.append(f"Тема «{info['title']}» относится к уровню {info['level']}: "
                         f"на уровне {level} часть заданий может не получиться.")
        if info and info["area"] == "writing":
            fit = [t for t in types if t in WRITING_TYPES] or list(WRITING_TYPES)
            if fit != types:
                notes.append("Для темы по письму оставлены выбор ответа и исправление ошибок.")
                types = fit
        # название из карты тем помогает поиску на коротких запросах («Articles» → «Артикли a/an/the»)
        query = topic if not info or info["title"].lower() in topic.lower() else f"{topic} ({info['title']})"
        search = self.retriever.search(query, k=SEARCH_K, level=level)
        if not covered(search, info):  # на запрошенном уровне нет — пробуем любые уровни
            search = self.retriever.search(query, k=SEARCH_K)
        citations: list[Citation] = []
        if covered(search, info):
            citations = build_citations(rank_by_topic(search.hits, info)[:4])
        elif not allow_without_materials:
            return Quiz(
                id=quiz_id, status="no_material", topic=topic, level=level, grounded=False,
                message=f"{NO_MATERIAL_TEXT} По теме «{topic}» материалов нет. Сгенерировать тест без опоры на материалы?",
                suggestions=suggest_topics(search, threshold=self.retriever.settings.relevance_threshold),
            )
        grounded_prompt = self.use_fragments and bool(citations)
        system = load_prompt("quiz_system" if grounded_prompt else "quiz_scratch_system")
        user_template = load_prompt("quiz_user" if grounded_prompt else "quiz_scratch_user")
        examples = example_keys(system)
        vocab = load_vocab(str(self.retriever.settings.materials_dir))

        accepted: list[LLMQuestion] = []
        seen: set[str] = set()
        rejected: list[str] = []  # причины отказа — подсказка модели в следующем раунде
        wanted = plan_types(n, types)
        base_seed = seed if seed is not None else random.randrange(1, 2**31 - MAX_ROUNDS)
        logger.info("генерация теста «{}» ({}): seed {}", topic, level, base_seed)
        responses: set[str] = set()
        repeats = 0  # сколько раз модель вернула ответ, уже полученный в прошлом раунде
        for round_no in range(MAX_ROUNDS):
            missing = list(wanted)
            for q in accepted:  # чего не хватает — считаем по типам, а не по позициям
                missing.remove(q.type)
            missing = missing[:BATCH]
            if not missing:
                break
            avoid = ""
            if accepted:
                avoid = ("Не повторяй эти задания:\n"
                         + "\n".join(f"- {q.prompt}" for q in accepted[-MAX_AVOID:]) + "\n")
            if rejected:
                # без этого при температуре 0 модель на тот же запрос вернёт те же ошибки
                avoid += ("Прошлые задания отклонены, исправь ошибки:\n"
                          + "\n".join(f"- {r}" for r in dict.fromkeys(rejected)) + "\n")
                rejected.clear()
            user = self._fit_prompt(system, user_template, citations if grounded_prompt else [], topic=topic,
                                    level=level, plan="\n".join(f"- {t}" for t in missing), avoid=avoid)
            try:
                options = {"seed": base_seed + round_no}
                if repeats:  # тот же ответ уже был: одного нового seed не хватило — выбираем свободнее
                    options["temperature"] = min(MAX_TEMPERATURE, self.temperature + REPEAT_TEMPERATURE_STEP * repeats)
                raw = self.structured(LLMQuiz, system, user, **options)
            except ContextBudgetError:
                if accepted:  # больше раундов не поместится — отдаём то, что уже принято
                    logger.warning("генерация теста: окно исчерпано, вопросов {}", len(accepted))
                    break
                raise  # не поместилось даже сокращённое — повторять бессмысленно
            except StructuredOutputError as e:
                logger.warning("генерация теста: {}", e)
                continue
            fingerprint = json.dumps(raw.questions, ensure_ascii=False, sort_keys=True)
            if fingerprint in responses:
                repeats += 1
                logger.info("модель повторила ответ прошлого раунда (повтор {}), поднимаю температуру", repeats)
            responses.add(fingerprint)
            pending = list(missing)
            for item in raw.questions:
                try:
                    q = LLMQuestion.model_validate(item)
                except ValidationError as e:
                    reasons = [err["msg"].removeprefix("Value error, ") for err in e.errors()]
                    logger.info("вопрос отброшен (формат): {}", "; ".join(reasons))
                    rejected += reasons
                    continue
                if q.type not in pending:
                    continue  # вопрос сверх плана — не ошибка модели
                key = question_key(q.prompt)
                if key in examples:
                    logger.info("вопрос отброшен (пример из инструкции): {}", q.prompt)
                    rejected.append(f"это пример из инструкции, а не задание по теме «{topic}»: {q.prompt}")
                    continue
                if key in seen:
                    logger.info("вопрос отброшен (повтор): {}", q.prompt)
                    rejected.append(f"задание повторяет уже составленное: {q.prompt}")
                    continue
                english = q.answer + " " + " ".join(q.options) + (" " + q.prompt if q.type != "translation" else "")
                hard = words_above_level(english, level, vocab)
                if len(hard) > MAX_HARD_WORDS:
                    logger.info("вопрос отброшен (лексика выше {}: {}): {}", level, ", ".join(hard), q.prompt)
                    rejected.append(f"слова сложнее уровня {level} ({', '.join(hard)}) — замени их простыми")
                    continue
                pending.remove(q.type)
                seen.add(key)
                accepted.append(q)

        by_n = {c.n: c for c in citations}
        questions = []
        accepted.sort(key=lambda q: wanted.index(q.type))  # порядок типов как в запросе (сортировка устойчива)
        rng = random.Random(base_seed)  # тот же seed — тот же порядок (воспроизводимо в eval)
        for i, q in enumerate(accepted[:n], 1):
            if q.type == "multiple_choice":
                # модель ставит верный ответ почти всегда на A или B (замер: A 47%, B 47%, C 6%) — перемешиваем сами
                rng.shuffle(q.options)
            citation = by_n.get(q.fragment) or (citations[0] if citations else None)
            questions.append(Question(**q.model_dump(), id=i, type_ru=TYPE_RU[q.type], citation=citation))
        if not questions:
            raise StructuredOutputError("Не удалось составить ни одного корректного вопроса.")
        return Quiz(id=quiz_id, topic=topic, level=level, grounded=bool(citations), questions=questions,
                    message=" ".join(notes))

    def _fit_prompt(self, system: str, template: str, citations: list[Citation], **fields: str) -> str:
        """Промпт раунда. Если вместе с максимальным ответом он не помещается в окно, сначала укорачиваются
        фрагменты, потом их становится меньше; не помогло — StructuredCaller откажет с понятной ошибкой."""
        s = self.retriever.settings
        chars, k = FRAGMENT_CHARS, len(citations)

        def render() -> str:
            return template.format(**fields, context=format_context(citations[:k], chars) if k else "")

        user = render()
        while k and estimate_tokens(system, user) + s.quiz_max_tokens > s.llm_num_ctx:
            if chars > MIN_FRAGMENT_CHARS:
                chars = max(MIN_FRAGMENT_CHARS, int(chars * 0.7))
            elif k > 1:
                k -= 1
            else:
                break
            user = render()
            logger.info("промпт теста не помещается в окно: фрагментов {} по {} символов", k, chars)
        return user

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
                    verdict = self.judge(Judgement, load_prompt("judge_system"), user)
                    correct, comment = verdict.correct, verdict.comment
                except StructuredOutputError:
                    comment = "Автоматическая проверка не удалась, ответ сравнён с эталоном."
            results.append(QuestionResult(question=q, given=given, correct=correct, comment=comment))
        return QuizResult(quiz_id=quiz.id, topic=quiz.topic, level=quiz.level,
                          score=sum(r.correct for r in results), total=len(results), results=results)

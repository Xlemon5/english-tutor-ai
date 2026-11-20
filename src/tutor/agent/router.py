"""Маршрутизация намерений в чате: теория / перевод / пример текста / проверка работы / тест / программа / приветствие.

Правила вместо отдельного вызова LLM: это мгновенно и предсказуемо. В разделах интерфейса
сценарий выбран явно, роутер нужен только для свободного чата.

Всё, что не распознано, уходит в теорию. Если теория не нашла ответа, `plan_fallback` решает,
не был ли это вопрос о составе уровня, — тогда вместо отказа показывается программа обучения.
Набор формулировок, на которых проверены правила, — tests/test_router.py.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from loguru import logger
from pydantic import BaseModel

from tutor.models import LEVELS

Intent = Literal["theory", "translate", "compose", "review", "quiz", "plan", "smalltalk"]

_I = re.IGNORECASE

# ---------------------------------------------------------------- уровни
# A1 / А1 (кириллица) / a1 / B-1, а также названия уровней: beginner — A1, elementary — A2, intermediate — B1
_LEVEL = re.compile(
    r"(?<![\w-])(?:([AaАа])-?([12])|([BbВв])-?(1))(?!\w)"
    r"|\b(pre-?\s?intermediate|intermediate|elementary|beginners?|starter|"
    r"начальн\w*|элементарн\w*|средн\w* уров\w*|уров\w* средн\w*)\b", _I)
_LEVEL_WORDS = [(re.compile(p, _I), lv) for p, lv in (
    (r"pre-?\s?intermediate", "A2"), (r"intermediate|средн", "B1"), (r"elementary|элементарн", "A2"),
    (r"beginner|starter|начальн", "A1"))]


def find_levels(text: str) -> list[str]:
    """Все упомянутые уровни по порядку, приведённые к A1/A2/B1."""
    found = []
    for m in _LEVEL.finditer(text):
        if m.group(1):
            lv = ("A" + m.group(2))
        elif m.group(3):
            lv = "B1"
        else:
            lv = next(v for p, v in _LEVEL_WORDS if p.search(m.group(5)))
        found.append(lv)
    return found


# ---------------------------------------------------------------- приветствия и вопросы о боте
# Сообщение целиком состоит из таких фраз (плюс знаки препинания) — это не учебный вопрос.
_SMALLTALK_PHRASES = re.compile(
    r"привет\w*|здравствуй\w*|здрасьте|добр\w+ (день|вечер|утро|ночи)|доброе утро|хай|хей|салют|"
    r"спасибо\w*|благодар\w*|пока|до свидания|до встречи|"
    r"как (у тебя |у вас )?дела|как ты|как поживаешь|"
    r"кто ты( такой)?|ты кто( такой)?|кто вы( такой)?|вы кто|"
    r"что (ты |вы )?(умеешь|умеете|можешь|можете)( делать)?|"
    r"как (тобой|вами|этим|бот\w*) пользоваться|как (ты |это )?работа\w+|"
    r"помоги(те)?( мне)?|помощь|нужна помощь|"
    r"ок|окей|хорошо|понятно|ясно|отлично|супер|класс|круто|"
    r"hello|hi|hey|there|good (morning|afternoon|evening)|thanks|thank you|thx|bye|goodbye|"
    r"who are you|what can you do|how are you|help( me)?|ok|okay|great|cool|"
    r"большое|огромное|очень|мне|тебе|вам|бот|друг|репетитор|пожалуйста|please", _I)
_THANKS = re.compile(r"спасибо|благодар|thank|thx", _I)
_BYE = re.compile(r"\b(пока|до свидания|до встречи|bye|goodbye)\b", _I)


def is_smalltalk(text: str) -> bool:
    if not re.search(r"\w", text) or len(text.split()) > 8:
        return False
    rest = _SMALLTALK_PHRASES.sub(" ", text.lower())
    return not re.search(r"\w", rest)


def smalltalk_kind(text: str) -> str:
    """greeting / thanks / bye — чтобы на «спасибо» не отвечать «здравствуйте»."""
    if _THANKS.search(text):
        return "thanks"
    if _BYE.search(text):
        return "bye"
    return "greeting"


# ---------------------------------------------------------------- программа обучения
# Явная просьба о программе — всегда программа.
_PLAN = re.compile(
    r"программ\w* (обучени|заняти|курс|изучени|подготовк)\w*|план\w* (обучени|заняти|учеб|изучени|подготовк)\w*|"
    r"учебн\w+ (план|программ)\w*|дорожн\w+ карт\w*|roadmap|road map|syllabus|curriculum|"
    r"(study|learning|course) (plan|path|program\w*)|"
    r"с чего (мне |лучше |стоит |нужно |надо )?(нач[аи]\w*|старт\w*)|where (should I |do I |to )?start|"
    r"как (мне )?(выучить|учить|изучать|изучить|освоить|подтянуть|улучшить|прокачать|подучить) "
    r"(свой |мой )?(английск\w*|язык|english)|"
    r"как (мне )?(повысить|поднять|улучшить|прокачать|подтянуть) (свой |мой )?уров\w*|"
    r"(how (can|do|should) I|how to) (learn|improve) (my )?english", _I)
# Вопрос, на который отвечает программа, но только если речь о уровне в целом:
# «что учить для A1», «какие темы входят в A2», «как сдать на B1», «what should I learn for A1».
_PLAN_HINT = re.compile(
    r"\b(тем[аыуе]?|список|входит|входят|включа\w*|состо\w+|требовани\w*|получ\w*|сда\w*|достич\w*|достиг\w*|"
    r"дойти|добраться|выйти|перейти|подготов\w*|проход\w*|что делать|программ\w*|план\w*|нужн\w*|надо|"
    r"необходим\w*|следует|стоит|знать|уметь|учить|изучать|изучить|выучить|освоить|подтянуть|"
    r"дальше|после|следующ\w*|learn|know|need|study|next|after|reach|get to|achieve|pass|topics?|cover\w*|include\w*|what'?s in|requirements?)\b", _I)
# «что учить», «что изучить дальше» — программа и без уровня
_WHAT_TO_LEARN = re.compile(
    r"\bчто (мне |ещё |еще |дальше |теперь |сначала |потом )*(нужно |надо |следует |стоит |лучше )?"
    r"(учить|изучать|выучить|изучить|освоить|проходить|пройти)\b|"
    r"\bwhat (should|do|must) I (need to |have to )?(learn|study|know)( next)?\b", _I)
# «о Present Perfect», «про артикли», «слова по теме «еда»» — вопрос о конкретном правиле, а не об уровне
# («про уровень A1», «о программе» — это как раз вопрос об уровне)
_ABOUT = re.compile(r"\b(о|об|обо|про|about|насчёт|насчет|по теме|на тему)\b(?!\s+(уров\w*|level|программ\w*|[AaBbАаВв]-?[12]\b))", _I)
# для страховки после отказа достаточно одного из этих слов (или уровня)
_LEVEL_TALK = re.compile(r"\b(уров\w*|level|программ\w*|план\w*|курс\w*|учить|изуч\w*|выуч\w*|освои\w*|"
                         r"learn\w*|study|studying)\b", _I)


@lru_cache
def _topic_patterns() -> tuple[re.Pattern, ...]:
    """Названия тем: русские термины из глоссария поиска и выражения карты тем materials/topics.yaml."""
    patterns: list[str] = []
    glossary = Path(__file__).parents[1] / "retrieval" / "glossary.yaml"
    # «вопрос» слишком общее слово: «у меня вопрос: что учить для A1» — не тема
    patterns += [k for k in (yaml.safe_load(glossary.read_text("utf-8")) or {}) if not k.startswith("вопрос")]
    try:
        from tutor.config import get_settings
        from tutor.ingestion.manifest import load_topics
        for t in load_topics(get_settings().materials_dir):
            patterns += [p for p in t["match"] if len(p) > 3]
    except Exception as e:  # noqa: BLE001 — без карты тем роутер работает по глоссарию
        logger.warning("карта тем для роутера не загрузилась: {}", e)
    patterns += [r"\b(present|past|future|perfect|continuous|progressive|simple)\b", r"\bвремен", r"\bглагол",
                 r"\bслов[оа]?\s+[«\"']?[A-Za-z]"]
    return tuple(re.compile(p, _I) for p in patterns)


def names_topic(text: str) -> bool:
    """Есть ли в вопросе конкретная тема (правило, время, часть речи, лексическая тема)."""
    return any(p.search(text) for p in _topic_patterns())


# «что учить после A1», «мой уровень A2, что дальше» — программа следующего уровня
_NEXT = re.compile(r"\b(после|дальше|следующ\w*|выше|after|next|beyond)\b", _I)


def _plan_route(text: str) -> Route:
    found = find_levels(text)
    if len(found) == 1 and _NEXT.search(text) and found[0] != LEVELS[-1]:
        return Route(intent="plan", level=LEVELS[LEVELS.index(found[0]) + 1], start_level=found[0])
    return Route(intent="plan", level=found[-1] if found else "", start_level=found[0] if len(found) > 1 else "")


def asks_for_plan(text: str) -> bool:
    """Вопрос об уровне в целом: «что учить» или уровень плюс «какие темы / что нужно / как сдать»,
    и при этом без конкретной темы («что нужно знать о Present Perfect для B1» — это теория).
    «what should I learn» вырезается перед поиском темы: should — тоже тема карты (модальные глаголы)."""
    if _ABOUT.search(text) or names_topic(_WHAT_TO_LEARN.sub(" ", text)):
        return False
    return bool(_WHAT_TO_LEARN.search(text) or (find_levels(text) and _PLAN_HINT.search(text)))


def plan_fallback(message: str) -> Route | None:
    """Страховка на случай формулировок, которые роутер не узнал заранее: теория не нашла ответа,
    а вопрос похож на вопрос об уровне или обучении в целом (упомянут уровень или «уровень / программа /
    учить», конкретной темы нет) — вместо отказа показываем программу. Условие мягче, чем в route():
    ошибиться здесь не страшно, иначе пользователь всё равно получил бы отказ."""
    text = message.strip()
    if _ABOUT.search(text) or names_topic(_WHAT_TO_LEARN.sub(" ", text)):
        return None
    return _plan_route(text) if find_levels(text) or _LEVEL_TALK.search(text) else None


# ---------------------------------------------------------------- перевод
# Явная просьба: «переведи…», «перевод слова…», «translate…» — перевод при любом остальном содержании.
_TRANSLATE = re.compile(r"\b(перевед\w*|перевест\w*|перевод\w*|translat\w*)\b", _I)
# «как будет кошка по-английски», «как сказать «я устал» на английском», «кошка по-английски»,
# «how do you say кошка in English» — перевод всегда
_LANG = r"(по[- ]?английск\w*|по[- ]?русск\w*|на английск\w*|на русск\w*|in english|in russian)"
_TRANSLATE_HOW = re.compile(
    r"\bкак (будет|сказать|говорят|называется|назвать|звучит|пишется)\b.{0,60}?" + _LANG + "|"
    r"\b" + _LANG + r"\s*(будет|это|значит|как)\b|"
    r"\bhow (do you|do I|to|can I) say\b|\bwhat is .{1,40}? in (english|russian)\b", _I)
# «что значит …» — перевод, если это не название правила («что значит Present Perfect» — теория)
_TRANSLATE_SOFT = re.compile(
    r"\bчто (значит|означает|обозначает) (слово|фраза|фразу|выражение|глагол|существительное|прилагательное)\b|"
    r"\bкак (пишется|произносится|переводится) (слово|фраза)\b|"
    r"\bчто (значит|означает|обозначает)\s*[«\"'“]?[A-Za-z]|"
    r"\bзначение слова\b|\bwhat (does|do|is) .{1,40}? mean\b|\bmeaning of\b", _I)
# «кошка по-английски», «собака — по-английски?» — только в коротком сообщении:
# «как написать письмо другу на английском?» — вопрос о письме, а не перевод
_WORD_IN_LANG = re.compile(r"^\W*\w+(\W+\w+){0,2}\s*(—|-|–)?\s*" + _LANG + r"\s*\??\s*$", _I)
_SHORT_WORD = re.compile(r"^[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё' -]*[?.!]?$")


def asks_for_translation(text: str) -> bool:
    if _TRANSLATE.search(text) or _TRANSLATE_HOW.search(text) or _WORD_IN_LANG.match(text):
        return True
    # «что значит Present Perfect» — вопрос о правиле, а не перевод
    if _TRANSLATE_SOFT.search(text):
        return not names_topic(re.sub(r"\b(слово|фраза|фразу|выражение|глагол|существительное|прилагательное)\b",
                                      " ", text, flags=_I))
    # одно-два слова без вопроса — «кошка», «butterfly»: просят перевод; «артикли», «Past Simple» — тема
    return bool(_SHORT_WORD.match(text) and len(text.split()) <= 2 and not names_topic(text) and not find_levels(text))


# ---------------------------------------------------------------- проверка, тест, текст
_QUIZ = re.compile(
    r"\b(тест\w*|квиз\w*|quiz\w*|упражнени\w*|задани[яйе]\w*|задач[уиа]\b|потренир\w*|тренировк\w*|вопрос\w* на|"
    r"проверочн\w*|контрольн\w*|погоня\w*|exercises?|practice)\b", _I)
# «проверь мои знания по Past Simple», «проверь меня» — тест, а не проверка текста
_KNOWLEDGE_CHECK = re.compile(
    r"\b(провер\w* (меня|себя)|провер\w* (мои |свои |моих |мой |свой |моё |мое )?(знани\w*|уровень)|test me|quiz me)\b", _I)
_REVIEW = re.compile(
    r"(провер[ьи]\w*|исправ[ьи]\w*|найди ошибк\w*|найти ошибк\w*|есть ли ошибк\w*|ошибки в|"
    r"правильно ли|верно ли|так (правильно|верно|можно сказать)|это правильно|это верно|"
    r"check my|check this|correct my|correct this|proofread|is (this|it|that) (correct|right|ok)|any (mistakes|errors))", _I)
# «напиши мне короткий текст про выходные», «составь письмо другу», «write a short story about my family»
_COMPOSE = re.compile(
    r"\b(напиш\w*|состав\w*|придума\w*|сочин\w*|write|compose|make up)\b.{0,40}?"
    r"\b(текст\w*|рассказ\w*|истори[юяи]|письм\w*|сочинени\w*|эссе|диалог\w*|абзац\w*|предложени\w*|"
    r"text|story|stories|letter|e-?mail|essay|dialogu?e|paragraph|sentences?)\b", _I | re.DOTALL)
_COUNT = re.compile(r"\b(\d{1,2})\s*(вопрос\w*|задани\w*|question\w*)|\bиз\s+(\d{1,2})\b", _I)
_QUIZ_NOISE = re.compile(
    r"\b(сделай|составь|сгенерируй|дай|хочу|мне|меня|мои|свои|пожалуйста|небольшой|короткий|на тему|по теме|теме|"
    r"уровн\w*|для уровня|тест\w*|квиз\w*|quiz\w*|упражнени\w*|задани\w*|задач\w*|вопрос\w*|потренир\w*|"
    r"тренировк\w*|проверочн\w*|контрольн\w*|экзамен\w*|погоня\w*|провер\w*|знани\w*|себя|test|me|"
    r"exercises?|practice|on|for|из|на|по|про|с|о)\b", _I)


class Route(BaseModel):
    intent: Intent
    topic: str = ""
    level: str = ""
    start_level: str = ""  # для программы обучения: «с A2 до B1»
    count: int = 0
    text: str = ""  # текст работы для проверки


def english_share(text: str) -> float:
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text)
    return sum(ch.isascii() for ch in letters) / len(letters) if letters else 0.0


def split_review(message: str) -> str:
    """Отделяет английский текст работы от русской просьбы («Проверь: I goes to school»)."""
    parts = re.split(r"[:\n]", message, maxsplit=1)
    if len(parts) == 2 and english_share(parts[1]) > 0.7 and english_share(parts[0]) < 0.5:
        return parts[1].strip()
    english = re.findall(r"[A-Za-z][A-Za-z0-9 ,.;:'\"!?()\-\n]{10,}", message)
    return max(english, key=len).strip() if english else message.strip()


def route(message: str) -> Route:
    text = message.strip()
    if is_smalltalk(text):
        return Route(intent="smalltalk")
    words = len(text.split())
    levels = find_levels(text)
    if _PLAN.search(text):
        return _plan_route(text)
    if _TRANSLATE.search(text):  # «переведи: I goes to school» — перевод, а не проверка
        return Route(intent="translate")
    knowledge_check = _KNOWLEDGE_CHECK.search(text)
    quiz = _QUIZ.search(text) or knowledge_check
    compose, review = _COMPOSE.search(text), _REVIEW.search(text) and not knowledge_check
    long_english = words >= 25 and english_share(text) > 0.9  # длинный английский текст без вопроса — это работа
    if not (quiz or compose or review or long_english) and asks_for_plan(text):
        return _plan_route(text)
    if review and english_share(text) > 0.4:
        return Route(intent="review", text=split_review(text))
    if long_english:
        return Route(intent="review", text=text)
    if compose:
        return Route(intent="compose", level=levels[0] if levels else "")
    if asks_for_translation(text):
        return Route(intent="translate")
    if quiz and english_share(text) < 0.9:
        count = _COUNT.search(text)
        topic = _QUIZ_NOISE.sub(" ", _COUNT.sub(" ", _LEVEL.sub(" ", text)))
        topic = re.sub(r"[^\w' /-]+", " ", topic)
        return Route(
            intent="quiz", topic=re.sub(r"\s+", " ", topic).strip(),
            level=levels[0] if levels else "",
            count=int(count.group(1) or count.group(3)) if count else 0,
        )
    return Route(intent="theory")

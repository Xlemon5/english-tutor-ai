"""Маршрутизация намерений в чате: теория / проверка работы / тест / приветствие.

Правила вместо отдельного вызова LLM: это мгновенно и предсказуемо. В разделах интерфейса
сценарий выбран явно, роутер нужен только для свободного чата.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

Intent = Literal["theory", "review", "quiz", "smalltalk"]

_QUIZ = re.compile(r"\b(тест\w*|квиз\w*|quiz|упражнени\w*|задани[яйе]\w*|потренир\w*|вопрос\w* на)\b", re.IGNORECASE)
_REVIEW = re.compile(r"(провер[ьи]\w*|исправ[ьи]\w*|найди ошибк\w*|check my|correct my|есть ли ошибк\w*)", re.IGNORECASE)
_SMALLTALK = re.compile(r"^\s*(привет\w*|здравствуй\w*|добрый (день|вечер)|доброе утро|спасибо\w*|"
                        r"пока|hello|hi|thanks|thank you)[\s!.,)]*$", re.IGNORECASE)
_LEVEL = re.compile(r"\b([AА][12]|[BВ]1)\b", re.IGNORECASE)
_COUNT = re.compile(r"\b(\d{1,2})\s*(вопрос\w*|задани\w*|question\w*)|\bиз\s+(\d{1,2})\b", re.IGNORECASE)
_QUIZ_NOISE = re.compile(
    r"\b(сделай|составь|сгенерируй|дай|хочу|мне|пожалуйста|небольшой|короткий|на тему|по теме|уровн\w*|"
    r"для уровня|тест\w*|квиз\w*|quiz|упражнени\w*|задани\w*|вопрос\w*|потренир\w*|из|на|по|про|с|о)\b", re.IGNORECASE)


class Route(BaseModel):
    intent: Intent
    topic: str = ""
    level: str = ""
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
    if _SMALLTALK.match(text):
        return Route(intent="smalltalk")
    words = len(text.split())
    if _REVIEW.search(text) and english_share(text) > 0.4:
        return Route(intent="review", text=split_review(text))
    if words >= 25 and english_share(text) > 0.9:  # длинный английский текст без вопроса — это работа
        return Route(intent="review", text=text)
    if _QUIZ.search(text) and english_share(text) < 0.9:
        level = _LEVEL.search(text)
        count = _COUNT.search(text)
        topic = _QUIZ_NOISE.sub(" ", _COUNT.sub(" ", _LEVEL.sub(" ", text)))
        topic = re.sub(r"[^\w' /-]+", " ", topic)
        return Route(
            intent="quiz", topic=re.sub(r"\s+", " ", topic).strip(),
            level=level.group(1).upper().replace("А", "A").replace("В", "B") if level else "",
            count=int(count.group(1) or count.group(3)) if count else 0,
        )
    return Route(intent="theory")

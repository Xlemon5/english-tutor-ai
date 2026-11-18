"""Сопоставление темы из запроса с картой тем (materials/topics.yaml).

Поиск по одному лишь тексту темы промахивается на коротких названиях: «Articles» не проходит порог близости,
хотя материалы об артиклях есть, а «Глагол to be» первым находит раздел про to have. Карта тем знает, какие
файлы покрывают тему (пометка topics у фрагментов), её уровень и раздел (грамматика, лексика, письмо).
"""
from __future__ import annotations

import difflib
from functools import lru_cache
from pathlib import Path

from tutor.ingestion.manifest import load_topics
from tutor.models import LEVELS

FUZZY_CUTOFF = 0.8  # «Pressent Simpel» → «Present Simple», но не «Past Simple»


@lru_cache
def _topics(materials_dir: str) -> tuple[dict, ...]:
    return tuple(load_topics(Path(materials_dir)))


def _names(topic: dict) -> list[str]:
    """Как тему могут назвать: заголовок из карты и id без уровня («a1-present-simple» → «present simple»)."""
    return [topic["title"].lower(), topic["id"].split("-", 1)[-1].replace("-", " ")]


def match_topic(text: str, materials_dir: str | Path) -> dict | None:
    """Тема карты для запроса: точное название, затем регулярные выражения карты, затем близкое написание."""
    topics = _topics(str(materials_dir))
    query = " ".join(text.lower().split())
    for t in topics:
        if query in _names(t):
            return t
    for t in topics:
        if t["re"].search(text):
            return t
    names = {name: t for t in topics for name in _names(t)}
    close = difflib.get_close_matches(query, list(names), n=1, cutoff=FUZZY_CUTOFF)
    return names[close[0]] if close else None


def above_level(topic: dict, level: str) -> bool:
    return topic["level"] in LEVELS and level in LEVELS and LEVELS.index(topic["level"]) > LEVELS.index(level)

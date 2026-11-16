"""Сценарий «программа обучения»: план по уровням, собранный из карты тем и библиотеки материалов.

LLM здесь не участвует. Темы берутся из materials/topics.yaml, материалы к ним — из манифеста
sources.yaml, поэтому в программе нет ничего, чего нет в библиотеке; темы без материалов честно
помечаются как пробелы.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from tutor.ingestion.manifest import AREAS, load_manifest, load_topics
from tutor.models import LEVELS, SourceMeta, levels_in

MATERIALS_PER_TOPIC = 3
_KIND_ORDER = {"theory": 0, "worksheet": 1, "wordlist": 2, "profile": 3}
KIND_RU = {"theory": "теория", "worksheet": "упражнения", "wordlist": "список слов", "profile": "справочник"}


class PlanMaterial(BaseModel):
    file: str
    title: str
    kind: str
    kind_ru: str
    source: str = ""
    url: str = ""
    author: str = ""
    license: str = ""
    level: str = ""
    language: str = ""


class PlanTopic(BaseModel):
    id: str
    title: str
    level: str
    area: str
    area_ru: str
    materials: list[PlanMaterial] = Field(default_factory=list)
    materials_total: int = 0  # сколько всего файлов в библиотеке раскрывают тему

    @property
    def covered(self) -> bool:
        return bool(self.materials)


class PlanStage(BaseModel):
    level: str
    topics: list[PlanTopic]


class StudyPlan(BaseModel):
    start_level: str
    target_level: str
    stages: list[PlanStage]
    total_topics: int
    covered_topics: int
    gaps: list[str] = Field(default_factory=list)  # темы без материалов


def normalize_levels(start: str | None, target: str | None) -> tuple[str, str]:
    """Приводит запрошенные уровни к допустимым и к правильному порядку."""
    target = (target or LEVELS[-1]).upper()
    start = (start or LEVELS[0]).upper()
    if target not in LEVELS:
        raise ValueError(f"Уровень должен быть одним из: {', '.join(LEVELS)}")
    if start not in LEVELS:
        raise ValueError(f"Уровень должен быть одним из: {', '.join(LEVELS)}")
    if LEVELS.index(start) > LEVELS.index(target):
        start, target = target, start
    return start, target


def _rank(entry: SourceMeta, topic: dict) -> tuple:
    """Какие материалы показывать первыми: тема в названии, подходящий уровень, теория раньше
    упражнений, для начинающих — сначала объяснения на русском."""
    in_title = bool(topic["re"].search(entry.title))
    right_level = topic["level"] in levels_in(entry.level)
    russian_first = "ru" in entry.language and topic["level"] in ("A1", "A2")
    return (not in_title, not right_level, _KIND_ORDER.get(entry.kind, 9), not russian_first, entry.title.lower())


def build_plan(materials_dir: Path, target: str | None = None, start: str | None = None) -> StudyPlan:
    start, target = normalize_levels(start, target)
    wanted = LEVELS[LEVELS.index(start):LEVELS.index(target) + 1]
    topics = load_topics(materials_dir)
    by_topic: dict[str, list[SourceMeta]] = {}
    for entry in load_manifest(materials_dir):
        if entry.kind == "examples":
            continue
        for topic_id in entry.topics:
            by_topic.setdefault(topic_id, []).append(entry)

    stages, gaps, covered, total = [], [], 0, 0
    for level in wanted:
        items = []
        for t in (t for t in topics if t["level"] == level):
            entries = sorted(by_topic.get(t["id"], []), key=lambda e: _rank(e, t))
            materials = [
                PlanMaterial(file=e.file, title=e.title, kind=e.kind, kind_ru=KIND_RU.get(e.kind, e.kind),
                             source=e.source, url=e.url, author=e.author, license=e.license, level=e.level,
                             language=e.language)
                for e in entries[:MATERIALS_PER_TOPIC]
            ]
            topic = PlanTopic(id=t["id"], title=t["title"], level=level, area=t["area"],
                              area_ru=AREAS.get(t["area"], t["area"]), materials=materials,
                              materials_total=len(entries))
            items.append(topic)
            total += 1
            covered += topic.covered
            if not topic.covered:
                gaps.append(f"{level} · {topic.area_ru} · {topic.title}")
        if items:
            stages.append(PlanStage(level=level, topics=items))
    return StudyPlan(start_level=start, target_level=target, stages=stages, total_topics=total,
                     covered_topics=covered, gaps=gaps)


def plan_summary(plan: StudyPlan) -> str:
    scope = plan.target_level if plan.start_level == plan.target_level else f"{plan.start_level}–{plan.target_level}"
    text = f"Программа обучения, уровень {scope}. Тем: {plan.total_topics}, с материалами в библиотеке: {plan.covered_topics}."
    if plan.gaps:
        text += f" Без материалов: {len(plan.gaps)}."
    return text

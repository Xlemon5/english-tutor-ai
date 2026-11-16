"""Общие модели данных ядра."""
from __future__ import annotations

from pydantic import BaseModel, Field

LEVELS = ("A1", "A2", "B1")


class SourceMeta(BaseModel):
    """Запись манифеста materials/sources.yaml."""

    file: str
    title: str
    source: str = ""
    url: str = ""
    author: str = ""
    license: str = ""
    license_url: str = ""
    retrieved: str = ""
    language: str = ""
    kind: str = "theory"
    level: str = ""
    topics: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    id: str
    text: str
    order: int = 0
    file: str
    doc_title: str
    section: str = ""
    page: int | None = None
    level: str = ""
    topics: list[str] = Field(default_factory=list)
    language: str = ""
    kind: str = "theory"
    source: str = ""
    url: str = ""
    author: str = ""
    license: str = ""
    license_url: str = ""

    def to_metadata(self) -> dict:
        """Плоские метаданные для Chroma (None и списки там недопустимы)."""
        meta = self.model_dump(exclude={"id", "text", "topics", "page"})
        meta["topics"] = ",".join(self.topics)
        if self.page is not None:
            meta["page"] = self.page
        for lvl in LEVELS:
            meta[f"level_{lvl.lower()}"] = lvl in levels_in(self.level)
        return meta

    @classmethod
    def from_store(cls, id_: str, text: str, meta: dict) -> Chunk:
        data = {k: v for k, v in meta.items() if not k.startswith("level_")}
        data["topics"] = [t for t in str(data.get("topics", "")).split(",") if t]
        return cls(id=id_, text=text, **data)


def levels_in(level: str) -> list[str]:
    """«A2–B1» → [A2, B1]; «A1» → [A1]; пустая или неизвестная строка → все уровни."""
    found = [lvl for lvl in LEVELS if lvl in level]
    if len(found) == 2:
        lo, hi = LEVELS.index(found[0]), LEVELS.index(found[1])
        return list(LEVELS[lo:hi + 1])
    return found or list(LEVELS)

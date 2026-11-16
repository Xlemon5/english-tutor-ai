"""Ссылки на источники строятся кодом из метаданных найденных чанков, а не моделью."""
from __future__ import annotations

from pydantic import BaseModel

from tutor.retrieval.hybrid import Hit


class Citation(BaseModel):
    n: int
    chunk_id: str
    file: str
    title: str
    section: str = ""
    page: int | None = None
    url: str = ""
    source: str = ""
    author: str = ""
    license: str = ""
    license_url: str = ""
    text: str
    score: float = 0.0

    @property
    def needs_attribution(self) -> bool:
        """CC BY и условия CEFR-J требуют указывать автора и лицензию."""
        return self.license.startswith("CC BY") or "citation" in self.license.lower()

    def label(self) -> str:
        parts = [self.title]
        if self.section and self.section.lower() != self.title.lower():
            parts.append(self.section)
        if self.page is not None:
            parts.append(f"с. {self.page}")
        text = f"[{self.n}] " + " › ".join(parts) + f" — {self.file}"
        if self.url:
            text += f" ({self.url})"
        if self.needs_attribution and (self.author or self.license):
            text += " · " + ", ".join(p for p in (self.author, self.license) if p)
        return text


def build_citations(hits: list[Hit]) -> list[Citation]:
    return [
        Citation(
            n=n, chunk_id=h.chunk.id, file=h.chunk.file, title=h.chunk.doc_title, section=h.chunk.section,
            page=h.chunk.page, url=h.chunk.url, source=h.chunk.source, author=h.chunk.author,
            license=h.chunk.license, license_url=h.chunk.license_url, text=h.chunk.text, score=round(h.score, 4),
        )
        for n, h in enumerate(hits, 1)
    ]

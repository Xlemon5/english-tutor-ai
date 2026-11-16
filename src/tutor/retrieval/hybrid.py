"""Гибридный поиск: dense (ChromaDB) + BM25, объединение рангов (RRF), отбор MMR, порог релевантности."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from tutor.config import Settings
from tutor.llm.base import Embedder
from tutor.models import Chunk, levels_in
from tutor.retrieval.bm25 import BM25Index
from tutor.retrieval.store import ChunkStore

RRF_K = 60


@lru_cache
def _glossary() -> list[tuple[re.Pattern, str]]:
    data = yaml.safe_load((Path(__file__).parent / "glossary.yaml").read_text("utf-8")) or {}
    return [(re.compile(k, re.IGNORECASE), v) for k, v in data.items()]


def expand_query(query: str) -> str:
    """Дополняет русский запрос английскими грамматическими терминами из глоссария."""
    padded = query + " "
    extra: list[str] = []
    for pattern, terms in _glossary():
        if pattern.search(padded):
            extra += [t for t in terms.split() if t not in extra and t.lower() not in query.lower()]
    return f"{query} ({' '.join(extra)})" if extra else query
# Примеры предложений не являются теорией и в ответы на вопросы по умолчанию не попадают.
THEORY_KINDS = ("theory", "worksheet", "wordlist", "profile")


@dataclass
class Hit:
    chunk: Chunk
    score: float  # косинусная близость запроса и чанка
    fused: float = 0.0
    dense_rank: int | None = None  # место в семантической выдаче (с 1), None — там не найден
    lexical_rank: int | None = None  # место в выдаче BM25

    @property
    def found_by(self) -> str:
        if self.dense_rank and self.lexical_rank:
            return "семантика+BM25"
        return "семантика" if self.dense_rank else "BM25"


@dataclass
class SearchResult:
    query: str
    hits: list[Hit] = field(default_factory=list)
    top_score: float = 0.0
    relevant: bool = False


class Retriever:
    def __init__(self, store: ChunkStore, embedder: Embedder, settings: Settings) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings
        self._bm25: BM25Index | None = None

    def refresh(self) -> None:
        self._bm25 = BM25Index(self.store.all_chunks())

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self.refresh()
        return self._bm25  # type: ignore[return-value]

    def search(self, query: str, k: int | None = None, kinds: tuple[str, ...] | None = THEORY_KINDS,
               level: str | None = None, mode: str = "hybrid") -> SearchResult:
        """mode: hybrid — семантический и лексический поиск вместе (по умолчанию);
        dense — только семантический; lexical — только BM25 (нужны для сравнения в eval)."""
        s = self.settings
        k = k or s.retrieval_top_k
        n = max(s.retrieval_candidates, k)
        conditions: list[dict] = []
        if kinds:
            conditions.append({"kind": {"$in": list(kinds)}})
        if level:
            conditions.append({f"level_{level.lower()}": True})
        where = None if not conditions else conditions[0] if len(conditions) == 1 else {"$and": conditions}

        def allowed(c: Chunk) -> bool:
            return (not kinds or c.kind in kinds) and (not level or level in levels_in(c.level))

        expanded = expand_query(query)
        q = np.asarray(self.embedder.embed_query(expanded), dtype=float)
        dense = self.store.query(q.tolist(), n, where) if mode in ("hybrid", "dense") else []
        lexical = self.bm25.search(expanded, n, allowed) if mode in ("hybrid", "lexical") else []

        chunks: dict[str, Chunk] = {}
        vectors: dict[str, np.ndarray] = {}
        fused: dict[str, float] = {}
        dense_rank: dict[str, int] = {}
        lexical_rank: dict[str, int] = {}
        for rank, (chunk, _, emb) in enumerate(dense, 1):
            chunks[chunk.id] = chunk
            vectors[chunk.id] = np.asarray(emb, dtype=float)
            dense_rank[chunk.id] = rank
            fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (RRF_K + rank)
        for rank, (chunk, _) in enumerate(lexical, 1):
            chunks.setdefault(chunk.id, chunk)
            lexical_rank[chunk.id] = rank
            weight = s.lexical_weight if mode == "hybrid" else 1.0
            fused[chunk.id] = fused.get(chunk.id, 0.0) + weight / (RRF_K + rank)
        missing = [i for i in chunks if i not in vectors]
        for chunk, emb in self.store.get(missing):
            vectors[chunk.id] = np.asarray(emb, dtype=float)
        ids = [i for i in chunks if i in vectors]
        if not ids:
            return SearchResult(query)

        def cos(a: np.ndarray, b: np.ndarray) -> float:
            return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0))

        sim = {i: cos(q, vectors[i]) for i in ids}
        top = max(fused[i] for i in ids)
        selected: list[str] = []
        pool = set(ids)
        while pool and len(selected) < k:  # MMR: релевантность против повторов
            def mmr(i: str) -> float:
                redundancy = max((cos(vectors[i], vectors[j]) for j in selected), default=0.0)
                return s.mmr_lambda * fused[i] / top - (1 - s.mmr_lambda) * redundancy
            best = max(pool, key=mmr)
            selected.append(best)
            pool.discard(best)

        hits = sorted((Hit(chunks[i], sim[i], fused[i], dense_rank.get(i), lexical_rank.get(i)) for i in selected),
                      key=lambda h: h.fused, reverse=True)
        top_score = max(h.score for h in hits)
        floor = s.relevance_threshold * 0.85
        relevant = top_score >= s.relevance_threshold
        if relevant:
            hits = [h for h in hits if h.score >= floor]
        return SearchResult(query, hits, top_score, relevant)

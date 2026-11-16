"""Лексический поиск BM25 — ловит точные термины («Present Perfect», «some/any»)."""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from tutor.models import Chunk

# Служебные и вопросительные слова. В корпусе, где большинство текстов английские, русские слова
# вроде «когда» и «используется» редки, получают высокий вес BM25 и уводят поиск в сторону.
_STOP = frozenset(
    ["a", "an", "the", "of", "to", "in", "on", "at", "for", "and", "or", "is", "are", "was", "were", "be", "it", "this", "that", "with", "as", "by", "from", "what", "when", "how", "why", "which", "who", "where", "do", "does", "did", "use", "used", "using", "between", "difference", "и", "в", "во", "на", "с", "со", "по", "к", "у", "о", "об", "от", "для", "что", "как", "это", "а", "но", "или", "не", "же", "ли", "бы", "то", "когда", "какой", "какая", "какое", "какие", "чем", "где", "почему", "зачем", "такое", "такой", "нужно", "можно", "между", "разница", "используется", "используются", "образуется", "образуются", "ставится", "ставятся", "пишется", "пишутся", "строится", "строятся", "отличается", "отличаются", "бывают", "бывает", "относятся", "относится", "расскажи", "объясни"]
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-zа-яё0-9]+", text.lower()) if t not in _STOP and len(t) > 1]


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        corpus = [tokenize(f"{c.doc_title} {c.section} {c.text}") for c in chunks]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, n: int, allow=None) -> list[tuple[Chunk, float]]:
        tokens = tokenize(query)
        if self.bm25 is None or not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in ranked:
            if scores[i] <= 0 or len(out) >= n:
                break
            if allow is None or allow(self.chunks[i]):
                out.append((self.chunks[i], float(scores[i])))
        return out

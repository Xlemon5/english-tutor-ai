"""Метрики для оценки RAG. Чистые функции — их проверяют юнит-тесты, а считает eval/run_eval.py."""
from __future__ import annotations

import math
import re
from statistics import median

_MARK = re.compile(r"\[\d+\]")


def strip_markers(text: str) -> str:
    text = re.sub(r"\s+", " ", _MARK.sub("", text))
    return re.sub(r" ([.,;:!?])", r"\1", text).strip()


def has_markers(text: str) -> bool:
    return bool(_MARK.search(text))


def sentences(text: str, min_chars: int = 20) -> list[str]:
    parts = re.split(r"(?<=[.!?:])\s+|\n+", strip_markers(text).replace("*", ""))
    return [p.strip(" -•\t") for p in parts if len(p.strip(" -•\t")) >= min_chars]


def first_match_rank(expect: str, texts: list[str]) -> int | None:
    """Место (с 1) первого найденного фрагмента, подходящего под ожидаемый раздел."""
    for rank, text in enumerate(texts, 1):
        if re.search(expect, text, re.IGNORECASE):
            return rank
    return None


def fact_coverage(answer: str, facts: list[str]) -> float:
    """Доля ключевых фактов эталона, которые встретились в ответе."""
    if not facts:
        return 1.0
    return sum(bool(re.search(f, answer, re.IGNORECASE)) for f in facts) / len(facts)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def support_scores(sentence_vecs: list[list[float]], fragment_vecs: list[list[float]]) -> list[float]:
    """Для каждого предложения ответа — близость к самому похожему фрагменту-источнику.
    Низкое значение означает, что утверждение не опирается на найденные материалы."""
    if not fragment_vecs:
        return [0.0 for _ in sentence_vecs]
    return [max(cosine(s, f) for f in fragment_vecs) for s in sentence_vecs]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def med(values: list[float]) -> float:
    return float(median(values)) if values else 0.0


def pct(part: int, total: int) -> str:
    return f"{part}/{total} ({100 * part / total:.0f}%)" if total else "—"

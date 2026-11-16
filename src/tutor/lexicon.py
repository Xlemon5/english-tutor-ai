"""Проверка уровня лексики по словарю CEFR-J (если он есть в материалах)."""
from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

RANK = {"A1": 1, "A2": 2, "B1": 3, "B2": 4}
VOCAB_FILE = "web/cefr-j/data/cefrj-vocabulary-profile-1.5.csv"
_SUFFIXES = (("ies", ("y",)), ("es", ("",)), ("s", ("",)), ("ied", ("y",)), ("ed", ("", "e")),
             ("ing", ("", "e")), ("er", ("", "e")), ("est", ("", "e")), ("ly", ("",)))


@lru_cache
def load_vocab(materials_dir: str) -> dict[str, int]:
    path = Path(materials_dir) / VOCAB_FILE
    vocab: dict[str, int] = {}
    if not path.exists():
        return vocab
    with path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rank = RANK.get(row["CEFR"])
            for word in row["headword"].lower().split("/"):
                word = word.strip()
                if rank and word and rank < vocab.get(word, 9):
                    vocab[word] = rank
    return vocab


def _forms(word: str):
    yield word
    if "'" in word:
        yield word.split("'")[0]
    for suffix, endings in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            stem = word[: -len(suffix)]
            for ending in endings:
                yield stem + ending
            if len(stem) > 2 and stem[-1] == stem[-2]:
                yield stem[:-1]


def words_above_level(text: str, level: str, vocab: dict[str, int]) -> list[str]:
    """Английские слова из текста, которые по CEFR-J выше уровня. Неизвестные слова не считаются."""
    limit = RANK.get(level, 3)
    above = []
    for token in re.findall(r"[A-Za-z]+(?:'[a-z]+)?", text):
        ranks = [vocab[f] for f in _forms(token.lower()) if f in vocab]
        if ranks and min(ranks) > limit and token.lower() not in above:
            above.append(token.lower())
    return above

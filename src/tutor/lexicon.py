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


LIST_FILES = ("web/cefr-j/vocabulary-a1.md", "web/cefr-j/vocabulary-a2.md", "web/cefr-j/vocabulary-b1.md")
_SECTION = re.compile(r"^## (A1|A2|B1): (.+)$")


def _rows(materials: Path):
    """(слово, уровень, часть речи) из таблицы CEFR-J; если её нет — из списков vocabulary-*.md.
    Таблица лежит в каталоге data/, который не попадает в git, а списки в репозитории есть."""
    path = materials / VOCAB_FILE
    if path.exists():
        with path.open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                yield row["headword"], row["CEFR"], row["pos"]
        return
    for name in LIST_FILES:
        list_path = materials / name
        if not list_path.exists():
            continue
        level = pos = ""
        for line in list_path.read_text("utf-8").splitlines():
            if m := _SECTION.match(line):
                level, pos = m.groups()
            elif level and line.strip() and not line.startswith(("#", "---")):
                for headword in line.split(", "):
                    yield headword, level, pos


@lru_cache
def load_entries(materials_dir: str) -> dict[str, list[tuple[str, str]]]:
    """слово → [(уровень, часть речи), …] по возрастанию уровня."""
    entries: dict[str, list[tuple[str, str]]] = {}
    for headword, level, pos in _rows(Path(materials_dir)):
        if level not in RANK:
            continue
        for word in headword.lower().split("/"):
            word = word.strip()
            if word and (level, pos) not in entries.setdefault(word, []):
                entries[word].append((level, pos))
    for found in entries.values():
        found.sort(key=lambda e: RANK[e[0]])
    return entries


@lru_cache
def load_vocab(materials_dir: str) -> dict[str, int]:
    return {word: RANK[found[0][0]] for word, found in load_entries(materials_dir).items()}


def word_levels(word: str, materials_dir: str) -> list[tuple[str, str]]:
    """Уровни слова (или словосочетания) по CEFR-J; для форм вроде cats, went → по исходной форме."""
    entries = load_entries(materials_dir)
    word = word.lower().strip()
    if word in entries:
        return entries[word]
    for form in _forms(word):
        if form in entries:
            return entries[form]
    return []


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

"""Уровни слов по CEFR-J. Таблица materials/web/cefr-j/data/*.csv лежит в репозитории; раньше её
скрывало правило `data/` в .gitignore, и на другой машине проверка лексики молча отключалась.
Если таблицы нет, уровни берутся из списков vocabulary-*.md."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tutor.agent.review import LLMError, fix_spelling_type
from tutor.agent.translate import fix_parts_of_speech
from tutor.lexicon import LIST_FILES, VOCAB_FILE, load_entries, load_vocab, word_levels, words_above_level

MATERIALS = Path("materials")


def test_vocabulary_table_is_in_repository():
    assert (MATERIALS / VOCAB_FILE).exists()
    vocab = load_vocab(str(MATERIALS))
    assert len(vocab) > 6000  # вместе с уровнем B2
    assert words_above_level("I negotiate a contract", "A1", vocab) == ["negotiate", "contract"]


def test_lists_are_used_without_table(tmp_path):
    for name in LIST_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(MATERIALS / name, tmp_path / name)
    load_entries.cache_clear()
    vocab = load_vocab(str(tmp_path))
    assert len(vocab) > 4000 and vocab["cat"] == 1 and vocab["however"] == 2
    assert word_levels("cats", str(tmp_path)) == [("A1", "noun")]
    assert load_vocab(str(tmp_path / "missing")) == {}


@pytest.mark.parametrize(("fragment", "correction", "given", "expected"), [
    ("frend", "friend", "grammar", "spelling"),  # живой случай: опечатка помечалась грамматикой
    ("freinds", "friends", "grammar", "spelling"),
    ("beautifull", "beautiful", "vocabulary", "spelling"),
    ("go", "went", "grammar", "grammar"),
    ("goed", "went", "grammar", "grammar"),  # неверная форма известного глагола — грамматика
    ("writed", "wrote", "grammar", "grammar"),
    ("childs", "children", "grammar", "grammar"),
    ("make", "do", "vocabulary", "vocabulary"),
    ("Yesterday I go", "Yesterday I went", "grammar", "grammar"),
])
def test_review_spelling_type(fragment, correction, given, expected):
    e = LLMError(fragment=fragment, correction=correction, type=given, explanation="Пояснение.", rule="rule")
    assert fix_spelling_type(e, str(MATERIALS)).type == expected


def test_translation_part_of_speech_from_cefr():
    answer = "**however** — однако (союз)\n\nПримеры:\n- However, it rained. — Однако шёл дождь."
    assert fix_parts_of_speech(answer, str(MATERIALS)).startswith("**however** — однако (наречие)\n\nПримеры:")
    # у favourite в CEFR-J две части речи, у pussycat — ни одной: ответ модели не трогаем
    for line in ("**favourite** — любимый (существительное)", "**pussycat** — киска (сущ.)"):
        assert fix_parts_of_speech(line, str(MATERIALS)) == line

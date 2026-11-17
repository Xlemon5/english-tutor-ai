from __future__ import annotations

import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import ValidationError

from tutor.agent.review import (
    NOT_IN_MATERIALS,
    LLMError,
    LLMReview,
    ReviewAgent,
    grade_for,
    locate,
    split_error,
)
from tutor.ingestion.indexer import ingest
from tutor.llm.structured import StructuredCaller, StructuredOutputError, extract_json
from tutor.retrieval.hybrid import Retriever

TEXT = "My brother work in a bank. He have a apple every day. She is a good persone."
GOOD = {
    "errors": [
        {"fragment": "brother work", "correction": "brother works", "type": "grammar",
         "explanation": "В третьем лице единственного числа нужен -s.", "rule": "third person singular ending -s"},
        {"fragment": "a apple", "correction": "an apple", "type": "grammar",
         "explanation": "Перед гласным звуком ставится an.", "rule": "indefinite article a an vowel sounds"},
        {"fragment": "persone", "correction": "person", "type": "spelling",
         "explanation": "Слово пишется person.", "rule": "зоология беспозвоночных"},
        {"fragment": "every day", "correction": "every day", "type": "style",
         "explanation": "Здесь всё правильно.", "rule": "adverbs of frequency"},
    ],
    "summary": "Текст понятный, но есть ошибки в формах глагола.",
    "cefr_level": "A1",
    "review_topics": ["Present Simple", "Артикли a/an"],
}


def caller(*responses: str) -> StructuredCaller:
    return StructuredCaller(lambda schema, **_: FakeListChatModel(responses=list(responses)), retries=2)


@pytest.fixture
def retriever(settings, embedder, store) -> Retriever:
    ingest(settings, embedder, store)
    return Retriever(store, embedder, settings)


def test_review_is_structured_and_linked_to_theory(retriever):
    result = ReviewAgent(retriever, caller(json.dumps(GOOD))).review(TEXT, "Write about your family.")
    assert [e.fragment for e in result.errors] == ["brother work", "a apple", "persone"]  # «не ошибка» отброшена
    first, second, third = result.errors
    assert TEXT[first.start:first.end] == "brother work" and first.type_ru == "грамматика"
    assert first.citation.file == "web/book/present-simple.md" and first.theory_note == ""
    assert second.citation.file == "web/book/articles.md"
    assert third.citation is None and third.theory_note == NOT_IN_MATERIALS
    assert [e.citation.n for e in result.errors if e.citation] == [1, 2]
    assert result.cefr_level == "A1" and result.grade == 2 and len(result.review_topics) == 2


def test_invalid_json_is_retried_then_accepted(retriever):
    bad = json.dumps({**GOOD, "cefr_level": "C2"})  # уровень вне схемы
    result = ReviewAgent(retriever, caller("не JSON", bad, "```json\n" + json.dumps(GOOD) + "\n```")).review(TEXT)
    assert result.summary.startswith("Текст понятный")


def test_gives_up_after_retries(retriever):
    with pytest.raises(StructuredOutputError):
        ReviewAgent(retriever, caller("плохо", "плохо", "плохо")).review(TEXT)


def test_schema_rejects_unknown_error_type():
    broken = {**GOOD, "errors": [{**GOOD["errors"][0], "type": "logic"}]}
    with pytest.raises(ValidationError):
        LLMReview.model_validate(broken)


def test_empty_and_too_long_text_rejected(retriever):
    agent = ReviewAgent(retriever, caller(json.dumps(GOOD)))
    with pytest.raises(ValueError):
        agent.review("   ")
    with pytest.raises(ValueError):
        agent.review("word " * 2000)


def test_locate_handles_case_whitespace_and_repeats():
    text = "I go home. Then i  go to bed."
    first = locate(text, "I go", [])
    assert first == (0, 4)
    assert locate(text, "I go", [first]) == (16, 21)  # второе вхождение: другой регистр и двойной пробел
    assert locate(text, "absent words", []) is None


def test_grade_and_extract_json():
    assert grade_for(0, "one two three four five") == 5
    assert grade_for(3, "word " * 10) == 2
    assert extract_json('Вот ответ: {"a": 1} Готово') == '{"a": 1}'


def test_whole_sentence_fragment_is_split_into_word_level_errors():
    text = "Hi. My sister like comedies, but she don't like horror films. We ate a popcorn and drinked cola."
    def err(fragment, correction):
        return LLMError(fragment=fragment, correction=correction, type="grammar", explanation="Объяснение.", rule="rule")
    parts = split_error(text, err("My sister like comedies, but she don't like horror films.",
                                  "My sister likes comedies, but she doesn't like horror films."))
    assert [(e.fragment, e.correction) for e, _ in parts] == [("like", "likes"), ("don't", "doesn't")]
    assert [text[s:e_] for _, (s, e_) in parts] == ["like", "don't"]
    parts = split_error(text, err("We ate a popcorn and drinked cola.", "We ate popcorn and drank cola."))
    assert [(e.fragment, e.correction) for e, _ in parts] == [("a popcorn", "popcorn"), ("drinked", "drank")]
    short = split_error(text, err("drinked", "drank"))
    assert len(short) == 1 and short[0][0].fragment == "drinked"


def test_chinese_in_review_is_rejected_and_retried(retriever):
    chinese = json.dumps({**GOOD, "summary": "Текст понятный, но 动词形式有错误."})
    result = ReviewAgent(retriever, caller(chinese, json.dumps(GOOD))).review(TEXT)
    assert result.summary == GOOD["summary"]

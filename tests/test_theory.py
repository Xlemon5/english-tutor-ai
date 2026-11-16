from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from tutor.agent.theory import NO_MATERIAL_TEXT, TheoryAgent, clean_markers
from tutor.ingestion.indexer import ingest
from tutor.retrieval.hybrid import Retriever


class ExplodingLLM(FakeListChatModel):
    """LLM, обращение к которой — ошибка: при отказе по порогу модель вызываться не должна."""

    def _call(self, *args, **kwargs):
        raise AssertionError("LLM не должна вызываться")

    def _stream(self, *args, **kwargs):
        raise AssertionError("LLM не должна вызываться")


@pytest.fixture
def retriever(settings, embedder, store) -> Retriever:
    ingest(settings, embedder, store)
    return Retriever(store, embedder, settings)


def test_answer_has_citations_built_from_chunks(retriever):
    llm = FakeListChatModel(responses=["Окончание -s добавляется в третьем лице: *he works* [1]. Ещё факт [9]."])
    result = TheoryAgent(retriever, llm).answer("third person singular ending -s or -es")
    assert result.status == "answered"
    assert "[9]" not in result.answer and "[1]" in result.answer  # выдуманная ссылка удалена
    assert [c.n for c in result.citations] == [1]
    c = result.citations[0]
    assert c.file == "web/book/present-simple.md" and c.section == "Third person singular › Spelling"
    assert c.url == "https://example.org/present-simple" and "third person singular" in c.text


def test_refuses_without_calling_llm_when_nothing_relevant(retriever):
    result = TheoryAgent(retriever, ExplodingLLM(responses=["x"])).answer("квантовая хромодинамика кварков")
    assert result.status == "no_material"
    assert result.answer == NO_MATERIAL_TEXT
    assert result.citations == []


def test_refuses_when_llm_says_no_answer(retriever):
    agent = TheoryAgent(retriever, FakeListChatModel(responses=["NO_ANSWER"]))
    events = list(agent.stream("third person singular ending -s or -es"))
    assert [e.type for e in events] == ["final"]  # ни одного токена отказа студент не увидел
    result = events[0].result
    assert result.status == "no_material" and result.citations == []
    assert result.suggestions and result.suggestions[0].startswith("Present Simple")


def test_stream_tokens_add_up_to_raw_answer(retriever):
    raw = "Артикль an ставится перед гласным звуком: an apple [1]."
    agent = TheoryAgent(retriever, FakeListChatModel(responses=[raw]))
    events = list(agent.stream("indefinite article a an before vowel sounds"))
    assert "".join(e.text for e in events if e.type == "token") == raw
    assert events[-1].type == "final" and events[-1].result.status == "answered"


def test_answer_without_markers_keeps_all_used_fragments(retriever):
    agent = TheoryAgent(retriever, FakeListChatModel(responses=["Окончание -s добавляется в третьем лице."]))
    result = agent.answer("third person singular ending -s or -es")
    assert result.status == "answered" and len(result.citations) >= 1


def test_clean_markers():
    text, used = clean_markers("Факт [2] , другой [5]. И ещё [1][2].", {1, 2})
    assert text == "Факт [2], другой. И ещё [1][2]."
    assert used == [2, 1]

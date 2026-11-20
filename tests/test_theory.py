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


def test_switch_to_chinese_is_cut_and_answer_continued(retriever):
    # Qwen иногда переходит на китайский посреди ответа; продолжение берётся из второго ответа модели
    llm = FakeListChatModel(responses=["Окончание -s добавляется 形成的句子应该是 после he [1].",
                                       " в третьем лице: *he works* [1]."])
    events = list(TheoryAgent(retriever, llm).stream("third person singular ending -s or -es"))
    shown = "".join(e.text for e in events if e.type == "token")
    assert shown == "Окончание -s добавляется в третьем лице: *he works* [1]."
    assert events[-1].result.answer == shown and events[-1].result.status == "answered"


def test_answer_stops_at_last_clean_place_if_model_never_recovers(retriever):
    llm = FakeListChatModel(responses=["Окончание -s добав形成", "形成的", "继续"])
    result = TheoryAgent(retriever, llm).answer("third person singular ending -s or -es")
    assert result.answer == "Окончание -s добав" and result.status == "answered"


def test_theory_marks_answer_cut_by_length(retriever):
    from langchain_core.outputs import ChatGenerationChunk

    from tutor.agent.theory import TRUNCATED_NOTE

    class CutModel(FakeListChatModel):
        def _stream(self, *args, **kwargs):
            chunks = list(super()._stream(*args, **kwargs))
            chunks[-1] = ChatGenerationChunk(message=chunks[-1].message.model_copy(
                update={"response_metadata": {"done_reason": "length"}}))
            yield from chunks

    result = TheoryAgent(retriever, CutModel(responses=["Окончание -s добавляется в третьем лице [1]"])).answer(
        "third person singular ending -s or -es")
    assert result.answer.endswith(TRUNCATED_NOTE.strip()) or TRUNCATED_NOTE.strip() in result.answer


def test_theory_refuses_question_that_does_not_fit_the_window(retriever):
    from tutor.llm.structured import ContextBudgetError

    retriever.settings.llm_num_ctx = 100
    with pytest.raises(ContextBudgetError):
        TheoryAgent(retriever, ExplodingLLM(responses=["x"])).answer("third person singular ending -s or -es")


def test_no_resume_without_assistant_prefill_support(retriever):
    llm = FakeListChatModel(responses=["Окончание -s добав形成", "не должно вызываться"])
    result = TheoryAgent(retriever, llm, can_resume=False).answer("third person singular ending -s or -es")
    assert result.answer == "Окончание -s добав"


def test_compose_writes_text_with_model_from_materials(retriever):
    from tutor.agent.compose import ComposeAgent
    raw = "He works every day [1].\n\n---\n\nГрамматика: Present Simple — *works*."
    result = ComposeAgent(retriever, FakeListChatModel(responses=[raw])).compose(
        "third person singular ending -s or -es", "A1")
    assert result.answer == "He works every day.\n\nГрамматика: Present Simple — *works*."  # номер ссылки убран
    assert result.level == "A1" and len(result.citations) == 1  # образец — один фрагмент из материалов


def test_compose_writes_text_even_without_model(retriever):
    from tutor.agent.compose import ComposeAgent
    result = ComposeAgent(retriever, FakeListChatModel(responses=["I like cats."])).compose("квантовая хромодинамика")
    assert result.answer == "I like cats." and result.citations == [] and result.level == "A2"

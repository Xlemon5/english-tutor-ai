from __future__ import annotations

import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import ValidationError

from tutor.agent.quiz import LLMQuestion, QuizAgent, normalize, plan_types
from tutor.ingestion.indexer import ingest
from tutor.lexicon import words_above_level
from tutor.llm.structured import StructuredCaller
from tutor.retrieval.hybrid import Retriever

MC = {"type": "multiple_choice", "prompt": "She ____ in a bank.", "options": ["work", "works", "working"],
      "answer": "works", "explanation": "После she нужен глагол с -s.", "fragment": 1}
GAP = {"type": "fill_gap", "prompt": "He ____ (watch) TV every evening.", "options": [], "answer": "watches",
       "explanation": "После -ch добавляется -es.", "fragment": 1}
FIX = {"type": "error_correction", "prompt": "He go to school every day.", "options": [],
       "answer": "He goes to school every day.", "explanation": "После he нужна форма goes.", "fragment": 7}
TR = {"type": "translation", "prompt": "Она работает в банке.", "options": [], "answer": "She works in a bank.",
      "explanation": "Регулярное действие — Present Simple.", "fragment": 1}
TOPIC = "present simple third person singular ending -s"


def agent_with(retriever, *responses) -> QuizAgent:
    queue = [json.dumps(r) if not isinstance(r, str) else r for r in responses]
    llm = FakeListChatModel(responses=queue)  # один экземпляр: ответы выдаются по очереди между вызовами
    return QuizAgent(retriever, StructuredCaller(lambda schema: llm, retries=1))


@pytest.fixture
def retriever(settings, embedder, store) -> Retriever:
    ingest(settings, embedder, store)
    return Retriever(store, embedder, settings)


def test_generates_grounded_quiz_with_citations(retriever):
    quiz = agent_with(retriever, {"questions": [MC, GAP, FIX, TR]}).generate(TOPIC, "A1", 4)
    assert quiz.status == "ready" and quiz.grounded
    assert [q.type for q in quiz.questions] == ["multiple_choice", "fill_gap", "error_correction", "translation"]
    assert [q.id for q in quiz.questions] == [1, 2, 3, 4]
    for q in quiz.questions:
        assert q.answer and q.explanation and q.citation is not None
        assert q.citation.file == "web/book/present-simple.md"  # ссылка взята из найденных чанков
    assert quiz.questions[2].citation.n == 1  # несуществующий фрагмент 7 заменён на реальный


def test_no_material_asks_before_generating(retriever):
    agent = agent_with(retriever, {"questions": [MC]})
    quiz = agent.generate("квантовая хромодинамика кварков", "B1", 1)
    assert quiz.status == "no_material" and quiz.questions == [] and "без опоры на материалы" in quiz.message
    forced = agent.generate("квантовая хромодинамика кварков", "B1", 1, ["multiple_choice"], allow_without_materials=True)
    assert forced.status == "ready" and not forced.grounded
    assert forced.questions[0].citation is None  # ссылок нет и они не выдуманы


def test_invalid_questions_are_dropped_and_regenerated(retriever):
    bad_mc = {**MC, "answer": "worked"}  # ответа нет среди вариантов
    no_gap = {**GAP, "prompt": "He watches TV."}  # нет пропуска
    quiz = agent_with(retriever, {"questions": [bad_mc, no_gap, GAP]}, {"questions": [MC]}).generate(
        TOPIC, "A1", 2, ["multiple_choice", "fill_gap"])
    assert sorted(q.type for q in quiz.questions) == ["fill_gap", "multiple_choice"]
    assert {q.answer for q in quiz.questions} == {"works", "watches"}


def test_question_schema_validation():
    for broken in ({**MC, "options": ["works", "works", "work"]}, {**FIX, "answer": "He go to school every day"},
                   {**TR, "prompt": "She works in a bank."}, {**MC, "type": "essay"}):
        with pytest.raises(ValidationError):
            LLMQuestion.model_validate(broken)
    assert LLMQuestion.model_validate({**GAP, "options": ["x"]}).options == []


def test_check_scores_and_uses_judge_for_free_answers(retriever):
    agent = agent_with(retriever, {"questions": [MC, GAP, FIX, TR]},
                       {"correct": True, "comment": "Предлог at тоже подходит."})
    quiz = agent.generate(TOPIC, "A1", 4)
    result = agent.check(quiz, {1: "Works", 2: "watch", 3: "he goes to school every day", 4: "She works at a bank"})
    assert [r.correct for r in result.results] == [True, False, True, True]
    assert (result.score, result.total) == (3, 4)
    assert result.results[3].comment == "Предлог at тоже подходит."
    assert result.results[1].question.answer == "watches" and result.results[1].question.citation is not None


def test_helpers():
    assert normalize("He’s  happy!") == "he is happy"
    assert plan_types(5, ["a", "b"]) == ["a", "b", "a", "b", "a"]
    vocab = {"cat": 1, "abandon": 3, "sophisticated": 4, "run": 1}
    assert words_above_level("The cats abandoned a sophisticated zzz running", "A1", vocab) == ["abandoned", "sophisticated"]
    assert words_above_level("The cats abandoned it", "B1", vocab) == []


def test_text_answers_ignore_case_punctuation_and_contractions(retriever):
    agent = agent_with(retriever, {"questions": [MC, GAP, FIX, TR]})
    quiz = agent.generate(TOPIC, "A1", 4)
    # judge не вызывается: все ответы совпадают с эталоном после нормализации
    result = agent.check(quiz, {1: "WORKS", 2: "  Watches ", 3: "HE GOES TO SCHOOL EVERY DAY", 4: "she works in a bank!"})
    assert [r.correct for r in result.results] == [True, True, True, True]
    assert normalize("He DOESN'T like it.") == normalize("he does not like it")
    assert normalize("I'm here, it’s OK") == normalize("I am here it is ok")
    assert normalize("They can't swim") == normalize("they cannot swim")

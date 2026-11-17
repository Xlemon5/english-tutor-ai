from __future__ import annotations

import itertools
import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import ValidationError

from tutor.agent.quiz import LLMQuestion, QuizAgent, normalize, plan_types
from tutor.ingestion.indexer import ingest
from tutor.lexicon import words_above_level
from tutor.llm.structured import StructuredCaller
from tutor.retrieval.hybrid import Retriever

MC = {"type": "multiple_choice", "prompt": "She ____ in a shop.", "options": ["work", "works", "working"],
      "answer": "works", "explanation": "После she нужен глагол с -s.", "fragment": 1}
GAP = {"type": "fill_gap", "prompt": "He ____ (watch) TV every evening.", "options": [], "answer": "watches",
       "explanation": "После -ch добавляется -es.", "fragment": 1}
FIX = {"type": "error_correction", "prompt": "He go to school every morning.", "options": [],
       "answer": "He goes to school every morning.", "explanation": "После he нужна форма goes.", "fragment": 7}
TR = {"type": "translation", "prompt": "Она работает в магазине.", "options": [], "answer": "She works in a shop.",
      "explanation": "Регулярное действие — Present Simple.", "fragment": 1}
TOPIC = "present simple third person singular ending -s"


def agent_with(retriever, *responses) -> QuizAgent:
    queue = [json.dumps(r) if not isinstance(r, str) else r for r in responses]
    llm = FakeListChatModel(responses=queue)  # один экземпляр: ответы выдаются по очереди между вызовами
    return QuizAgent(retriever, StructuredCaller(lambda schema, **_: llm, retries=1))


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
    for broken in ({**MC, "options": ["works", "works", "work"]}, {**FIX, "answer": "He go to school every morning"},
                   {**TR, "prompt": "She works in a bank."}, {**MC, "type": "essay"}):
        with pytest.raises(ValidationError):
            LLMQuestion.model_validate(broken)
    assert LLMQuestion.model_validate({**GAP, "options": ["x"]}).options == []


def test_answer_copied_into_choice_prompt_becomes_gap():
    # так отвечает Qwen 7B на Present Perfect: пример из Википедии с ответом, выделенным **
    copied = {**MC, "prompt": "I **have written** a letter this morning.",
              "options": ["have written", "wrote", "written"], "answer": "have written"}
    assert LLMQuestion.model_validate(copied).prompt == "I ____ a letter this morning."
    with pytest.raises(ValidationError):  # у заполнения пропуска так нельзя — пропала бы подсказка в скобках
        LLMQuestion.model_validate({**GAP, "prompt": "He watches TV."})
    with pytest.raises(ValidationError):  # ответа в предложении нет — чинить нечего
        LLMQuestion.model_validate({**MC, "prompt": "She is in a shop."})


def test_check_scores_and_uses_judge_for_free_answers(retriever):
    agent = agent_with(retriever, {"questions": [MC, GAP, FIX, TR]},
                       {"correct": True, "comment": "Предлог at тоже подходит."})
    quiz = agent.generate(TOPIC, "A1", 4)
    result = agent.check(quiz, {1: "Works", 2: "watch", 3: "he goes to school every morning", 4: "She works at a shop"})
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
    result = agent.check(quiz, {1: "WORKS", 2: "  Watches ", 3: "HE GOES TO SCHOOL EVERY MORNING", 4: "she works in a shop!"})
    assert [r.correct for r in result.results] == [True, True, True, True]
    assert normalize("He DOESN'T like it.") == normalize("he does not like it")
    assert normalize("I'm here, it’s OK") == normalize("I am here it is ok")
    assert normalize("They can't swim") == normalize("they cannot swim")


def test_chinese_in_question_or_judgement_is_rejected():
    from tutor.agent.quiz import Judgement
    from tutor.llm.language import has_foreign_script

    with pytest.raises(ValidationError):
        LLMQuestion.model_validate({**MC, "explanation": "После she нужен 第三人称 глагол."})
    with pytest.raises(ValidationError):
        Judgement.model_validate({"correct": True, "comment": "Верно，"})  # полноширинная запятая
    assert not has_foreign_script(["Он сказал: «It's — fine…» № 5, 50 °C", {"a": "café"}])
    assert has_foreign_script({"x": ["ひらがな"]}) and has_foreign_script("한국어")


def test_scratch_mode_keeps_fragments_out_of_prompt_but_links_theory(retriever):
    seen_prompts = []
    llm = FakeListChatModel(responses=[json.dumps({"questions": [MC, GAP]})])
    caller = StructuredCaller(lambda schema, **_: llm, retries=1)

    def spy(schema, system, user, **options):
        seen_prompts.append(system + user)
        return caller(schema, system, user, **options)

    agent = QuizAgent(retriever, spy, use_fragments=False)
    quiz = agent.generate(TOPIC, "A1", 2, ["multiple_choice", "fill_gap"])
    assert quiz.status == "ready" and quiz.grounded  # тема найдена в материалах
    assert "Фрагменты:" not in seen_prompts[0] and "[1]" not in seen_prompts[0]  # нумерованных фрагментов нет
    assert all(q.citation and q.citation.file == "web/book/present-simple.md" for q in quiz.questions)


def test_instruction_example_is_not_accepted_as_question(retriever):
    example = {**MC, "prompt": "She ____ in a bank."}  # слово в слово пример из quiz_system.md
    quiz = agent_with(retriever, {"questions": [example]}, {"questions": [MC]}).generate(
        TOPIC, "A1", 1, ["multiple_choice"])
    assert [q.prompt for q in quiz.questions] == ["She ____ in a shop."]


def test_content_checks_reject_typical_defects():
    # дефекты из ручной разметки 120 вопросов Qwen 7B
    cases = {
        "задание по-русски": {**MC, "prompt": "Она ____ в магазине.", "options": ["работает", "работать", "работал"],
                              "answer": "работает"},
        "перевод с русским ответом": {**TR, "answer": "Она работает в shop."},
        "объяснение не по-русски": {**MC, "explanation": "Third person singular takes -s."},
        "пропуск без подсказки": {**GAP, "prompt": "He ____ TV every evening."},
        "ответ уже в задании": {**GAP, "prompt": "She was born in 2010. She ____ (be born) in 2010.",
                                "answer": "was born"},
        "слово повторяется у пропуска": {**MC, "prompt": "If it rains, I ____ stay at home.",
                                         "options": ["will stay", "stay", "stays"], "answer": "will stay"},
    }
    for name, broken in cases.items():
        with pytest.raises(ValidationError):
            LLMQuestion.model_validate(broken)
            pytest.fail(name)
    # русская подсказка в скобках и короткий ответ, который встречается в предложении, допустимы
    assert LLMQuestion.model_validate({**GAP, "prompt": "It is ____ (дождь) today.", "answer": "rainy"})
    assert LLMQuestion.model_validate({**MC, "prompt": "It is a cat and ____ dog.", "options": ["a", "an", "the"],
                                       "answer": "a"})


def test_regeneration_rounds_never_send_identical_requests(retriever):
    # при T=0 одинаковый запрос давал одинаковый брак во всех раундах (Present Simple / A1)
    chinese = {**MC, "explanation": "第三人称单数主语需要动词的第三人称单数形式。"}
    llm = FakeListChatModel(responses=[json.dumps({"questions": [chinese]})] * 2 + [json.dumps({"questions": [MC]})])
    requests = []

    def make_llm(schema, **options):
        requests.append(options)
        return llm

    agent = QuizAgent(retriever, StructuredCaller(make_llm, retries=0))
    quiz = agent.generate(TOPIC, "A1", 1, ["multiple_choice"], seed=7)
    assert [q.prompt for q in quiz.questions] == [MC["prompt"]]
    assert [r["seed"] for r in requests] == [7, 8, 9]  # каждый раунд — свой seed
    # раунды 2 и 3 получают одинаковую обратную связь, но запрос всё равно разный
    assert all(a != b for a, b in itertools.pairwise(requests))

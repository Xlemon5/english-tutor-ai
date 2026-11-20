"""Роутер чата на живых формулировках. Случай, с которого всё началось (20.11): «что мне нужно изучить
для получения уровня A1» ушёл в теорию и получил отказ «в материалах этого нет», хотя на такие вопросы
отвечает программа обучения. Новую формулировку, на которой роутер ошибся, добавлять сюда."""
from __future__ import annotations

import pytest

from tutor.agent.router import find_levels, plan_fallback, route, smalltalk_kind
from tutor.agent.translate import cefr_levels, is_sentence

CASES = {
 "plan": [
  "что мне нужно изучить для получения уровня A1", "что учить для A2", "что надо знать для уровня A1",
  "какие темы нужны для B1", "какие темы входят в уровень A2", "что входит в уровень A1", "что включает уровень B1",
  "список тем для A1", "темы уровня A2", "требования к уровню B1", "как получить уровень A2", "как сдать на B1",
  "как достичь уровня B1", "как дойти до B1", "как перейти с A1 на A2", "как подготовиться к уровню A2",
  "как повысить уровень английского", "как улучшить свой английский", "с чего начать учить английский",
  "я новичок, с чего начать?", "я начинающий, что учить?", "составь программу обучения до a2", "план занятий на месяц",
  "дай план изучения английского", "дорожная карта до B1", "roadmap to B1", "what should I learn for A1",
  "what do I need to know for B1", "how to reach A2", "topics for A1", "study plan for B1", "A1 syllabus",
  "что нужно уметь на уровне A2", "какую грамматику нужно знать на A1?", "что изучать дальше после A2",
  "что учить после A1", "хочу выучить английский до B1", "хочу уровень B1, что делать", "программа A1",
  "что нужно для A1", "уровень A1 что учить", "что проходят на A2", "какие слова нужно знать для A1",
  "что надо выучить для уровня а2", "как выучить английский с нуля", "учебный план B1", "what's in level A2",
  "что учить для уровня beginner", "что нужно знать для intermediate",
 ],
 "theory": [
  "Когда используется Present Perfect?", "что нужно знать о Present Perfect для B1", "когда использовать present continious",
  "What is the difference between a and an?", "как образуется Past Simple", "какие модальные глаголы есть на A2",
  "чем отличается some от any", "как правильно ставить артикли", "что такое неправильные глаголы",
  "Как написать письмо другу?", "Как написать письмо другу на английском?", "что значит Present Perfect",
  "артикли", "Past Simple", "present continious",
  "какие предлоги места нужно знать", "разница между much и many", "как задать вопрос в Present Simple",
  # из eval/dataset.yaml: перехватывались новыми правилами
  "Какие слова по теме «еда» относятся к уровню A1?", "Что такое экзамен TOEFL и как он оценивается?",
 ],
 "translate": [
  "как будет кошка по-английски", "кошка по-английски", "собака — по-английски?", "как по-английски будет собака",
  "как сказать «я устал» по-английски", "как сказать по-английски сколько времени", "как называется ложка на английском",
  "что означает слово however", "что значит However?", "как пишется слово beautiful", "перевод слова дом",
  "переведи: I goes to school", "переведи на английский: я люблю кошек", "translate: good morning, teacher",
  "what does however mean", "how do you say кошка in English", "what is кошка in English", "кошка", "butterfly",
 ],
 "smalltalk": ["Привет!", "привет, как дела?", "кто ты?", "что ты умеешь?", "помоги", "спасибо большое!", "hi there",
  "добрый вечер!", "здравствуйте, вы кто", "как тобой пользоваться?", "help"],
 "quiz": ["Сделай тест из 7 вопросов по теме Past Simple уровня A2", "Сделай тест по Past Simple, A2",
  "дай упражнение на артикли", "хочу потренировать Present Continuous", "проверь мои знания по Past Simple",
  "проверь меня по теме артикли", "погоняй меня по неправильным глаголам", "дай задания на some/any"],
 "review": ["Проверь, пожалуйста: Yesterday I go to the park with my frend.", "исправь ошибки: She don't like apples",
  "найди ошибки в тексте: I has two brother", "is this correct: he go to school every day?",
  "правильно ли: I am agree with you"],
 "compose": ["напиши мне короткий текст про выходные", "составь письмо другу", "write a short story about my family",
  "придумай диалог в кафе", "Придумай рассказ про мою семью, A1"],
}


@pytest.mark.parametrize(("want", "text"), [(w, t) for w, qs in CASES.items() for t in qs])
def test_route_intent(want, text):
    assert route(text).intent == want


@pytest.mark.parametrize(("text", "start", "target"), [
    ("что мне нужно изучить для получения уровня A1", "", "A1"),
    ("как перейти с A1 на A2", "A1", "A2"),
    ("составь план обучения с A2 до B1", "A2", "B1"),
    ("что учить после A1", "A1", "A2"),
    ("мой уровень A2, что дальше", "A2", "B1"),
    ("что надо выучить для уровня а2", "", "A2"),  # кириллическая «а»
    ("что нужно знать для intermediate", "", "B1"),
    ("что учить для уровня beginner", "", "A1"),
    ("как выучить английский с нуля", "", ""),
])
def test_plan_levels(text, start, target):
    r = route(text)
    assert (r.intent, r.start_level, r.level) == ("plan", start, target)


def test_find_levels_ignores_letters_inside_words():
    assert find_levels("I have a cat and a dog") == []
    assert find_levels("A1, а2 и B-1") == ["A1", "A2", "B1"]


def test_quiz_topic_from_knowledge_check():
    r = route("проверь мои знания по Past Simple")
    assert (r.intent, r.topic) == ("quiz", "Past Simple")


def test_plan_fallback_after_refusal():
    # роутер такие формулировки не узнаёт, но если теория не нашла ответа — показываем программу
    for text, level in (("расскажи про уровень A1", "A1"), ("уровень B1 — это много?", "B1")):
        assert route(text).intent == "theory"
        fb = plan_fallback(text)
        assert fb is not None and (fb.intent, fb.level) == ("plan", level), text
    # конкретная тема или вопрос не об обучении — честный отказ остаётся отказом
    for text in ("как будет кошка по-английски", "квантовая хромодинамика кварков",
                 "что нужно знать о Present Perfect для B1", "как выучить неправильные глаголы быстро"):
        assert plan_fallback(text) is None, text


def test_smalltalk_kind():
    assert smalltalk_kind("спасибо большое!") == "thanks"
    assert smalltalk_kind("пока") == "bye"
    assert smalltalk_kind("привет, как дела?") == "greeting"


def test_cefr_levels_from_translation():
    answer = "**cat** — кошка (существительное)\n**pussycat** — киска\n\nПримеры:\n- I have a **black** cat."
    levels = [(w.word, w.level, w.pos) for w in cefr_levels(answer, "как будет кошка", "materials")]
    assert levels[0] == ("cat", "A1", "существительное")
    assert "pussycat" not in [w for w, _, _ in levels]  # слова нет в списках — уровень не выдумываем
    assert [w.word for w in cefr_levels("Это наречие «однако».", "что значит «however»", "materials")] == ["however"]


def test_translate_mode():
    for text in ("как будет кошка по-английски", "что означает слово however", "butterfly", "высокий по-английски",
                 "how do you say кошка in English", "перевод слова дом", "как называется ложка на английском"):
        assert not is_sentence(text), text
    for text in ("переведи на английский: я люблю читать книги", "как сказать «я очень устал» по-английски",
                 "translate: good morning, teacher"):
        assert is_sentence(text), text

"""Китайский в других путях, где модель пишет пользователю по-русски: ответы по теории (RAG), проверка работ,
судья свободных ответов. Контекст — тот же, на котором генерация теста стабильно уходила в китайский:
Present Simple / A1. Настройки — рабочие (T=0 там, где он задан в приложении). Пишутся СЫРЫЕ ответы модели,
до защит, чтобы увидеть, воспроизводится ли переход.

    uv run python eval/other_paths_cjk.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from tutor.agent.quiz import Question, Quiz, QuizAgent
from tutor.agent.review import ReviewAgent
from tutor.agent.theory import format_context
from tutor.config import get_settings
from tutor.llm.base import get_chat_model
from tutor.llm.structured import StructuredCaller
from tutor.prompts import load_prompt
from tutor.retrieval.citations import build_citations
from tutor.services import get_services

ROOT = Path(__file__).resolve().parent.parent
CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯＀-￯]")
THEORY_QUESTIONS = ["Present Simple", "Когда к глаголу в Present Simple добавляется окончание -s?",
                    "Как образуется отрицание в Present Simple?", "Как задать вопрос в Present Simple?",
                    "Чем Present Simple отличается от Present Continuous?"]
REVIEW_TEXTS = ["My sister go to school every day. She like books. He don't play football. They watches TV.",
                "I has a dog. My dog play in the garden every morning. My mother cook dinner at six.",
                "Every day I wakes up at 7. My brother don't like milk. Does she plays tennis?"]
JUDGE_CASES = [("Он работает в банке.", "He works in a bank.", "He is working at the bank."),
               ("Она не любит кофе.", "She doesn't like coffee.", "She not like coffee."),
               ("Мы играем в футбол по субботам.", "We play football on Saturdays.", "We plays football on Saturday."),
               ("Мой брат читает каждый вечер.", "My brother reads every evening.", "My brother read every evening.")]


def main() -> None:
    settings = get_settings()
    svc = get_services()
    raw: list[tuple[str, str]] = []

    class Rec:
        def __init__(self, llm, path):
            self.llm, self.path = llm, path

        def invoke(self, messages, *a, **k):
            r = self.llm.invoke(messages, *a, **k)
            raw.append((self.path, r.content))
            return r

    # 1) ответы по теории: тот же поиск по уровню A1, рабочая модель чата (температура из .env), без защиты потока
    theory_llm = get_chat_model(settings)
    for q in THEORY_QUESTIONS:
        result = svc.retriever.search(q, level="A1")
        if not result.relevant:
            print(f"теория «{q}»: материалов нет, модель не вызывается")
            continue
        context = format_context(build_citations(result.hits), settings.context_fragment_chars)
        chain = svc.theory.chain.first | theory_llm  # тот же промпт, что в приложении
        text = chain.invoke({"system": load_prompt("theory_system"), "context": context, "question": q}).content
        raw.append(("теория", text))
    # 2) проверка работ и 3) судья — как в приложении: T=0, лимит 1500, но с записью сырых ответов
    caller = StructuredCaller(lambda schema, **opts: Rec(
        get_chat_model(settings, schema=schema, temperature=0.0, max_tokens=1500, **opts), "?"))
    review = ReviewAgent(svc.retriever, caller)
    for text in REVIEW_TEXTS:
        before = len(raw)
        try:
            review.review(text)
        except Exception as e:  # noqa: BLE001
            print("проверка работы: ошибка", str(e)[:120])
        raw[before:] = [("проверка работ", t) for _, t in raw[before:]]
    quiz_agent = QuizAgent(svc.retriever, caller)
    for prompt, answer, given in JUDGE_CASES:
        q = Question(id=1, type="translation", prompt=prompt, options=[], answer=answer,
                     explanation="Регулярное действие — Present Simple.", type_ru="перевод предложений")
        quiz = Quiz(id="x", topic="Present Simple", level="A1", questions=[q])
        before = len(raw)
        quiz_agent.check(quiz, {1: given})
        raw[before:] = [("судья", t) for _, t in raw[before:]]

    out = ROOT / "eval/results/other_paths_cjk.json"
    out.write_text(json.dumps([{"path": p, "cjk": bool(CJK.search(t)), "text": t} for p, t in raw],
                              ensure_ascii=False, indent=1))
    for path in ("теория", "проверка работ", "судья"):
        items = [t for p, t in raw if p == path]
        bad = [t for t in items if CJK.search(t)]
        print(f"{path}: вызовов {len(items)}, с иероглифами {len(bad)}"
              + (f" (≤{300 / len(items):.0f}% при 0/{len(items)})" if items and not bad else ""))
        for t in bad[:3]:
            m = CJK.search(t)
            print("   …" + t[max(0, m.start() - 80):m.start() + 60].replace("\n", " ") + "…")


if __name__ == "__main__":
    main()

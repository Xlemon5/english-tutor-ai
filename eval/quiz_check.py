"""Проверка генерации тестов на всех темах из materials/topics.yaml.

Для каждой темы составляется тест с параметрами веб-формы по умолчанию (5 вопросов, все типы,
уровень темы). Если по теме нет материалов, проверяется и путь «сгенерировать без опоры на материалы».
Результат по каждой теме дописывается в JSONL сразу, итог печатается в конце.

    uv run python eval/quiz_check.py                       # все темы
    uv run python eval/quiz_check.py --only a2-past-simple b1-passive --level A1
    uv run python eval/quiz_check.py --mode scratch     # модель составляет предложения без фрагментов
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import yaml
from loguru import logger

from tutor.agent.quiz import TYPE_RU
from tutor.services import get_services

ROOT = Path(__file__).resolve().parent.parent


def source(citation) -> str | None:
    return " › ".join(x for x in (citation.title, citation.section) if x) if citation else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="id тем из topics.yaml")
    ap.add_argument("--level", help="уровень для всех тем (по умолчанию — уровень темы)")
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--types", nargs="*", default=list(TYPE_RU))
    ap.add_argument("--mode", choices=["settings", "fragments", "scratch"], default="settings",
                    help="fragments — фрагменты в промпте, scratch — без них; settings — как в .env")
    ap.add_argument("--out", default=str(ROOT / "eval/results/quiz_check.jsonl"))
    args = ap.parse_args()

    topics = [t for group in yaml.safe_load((ROOT / "materials/topics.yaml").read_text()).values() for t in group]
    if args.only:
        only = [x for item in args.only for x in item.split()]  # «a b» одной строкой тоже понимаем
        unknown = sorted(set(only) - {t["id"] for t in topics})
        if unknown:
            raise SystemExit(f"Нет таких тем в topics.yaml: {', '.join(unknown)}")
        topics = [t for t in topics if t["id"] in only]

    agent = get_services().quiz
    if args.mode != "settings":
        agent.use_fragments = args.mode == "fragments"
    print(f"Режим: {'фрагменты в промпте' if agent.use_fragments else 'с нуля, без фрагментов'}", flush=True)
    calls = Counter()
    inner = agent.structured

    def counted(schema, system, user):
        calls["llm"] += 1
        return inner(schema, system, user)

    agent.structured = counted
    rejects: Counter = Counter()
    reasons: Counter = Counter()  # тексты отказов по формату — чтобы понять, что именно не так

    def sink(message) -> None:  # причины отказа из журнала агента
        m = re.match(r"вопрос отброшен \(([^):]+)\)?:? ?(.*)", message.record["message"])
        if m:
            rejects[m.group(1)] += 1
            if m.group(1) == "формат":
                reasons.update(r.strip()[:90] for r in m.group(2).split(";") if r.strip())
        elif message.record["message"].startswith("генерация теста:"):
            rejects["невалидный JSON"] += 1

    logger.remove()
    logger.add(sink, level="INFO")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("")
    rows = []
    for i, t in enumerate(topics, 1):
        level = args.level or t["level"]
        row = {"id": t["id"], "topic": t["title"], "level": level}
        for forced in (False, True):
            calls.clear()
            rejects.clear()
            reasons.clear()
            start = time.time()
            try:
                quiz = agent.generate(t["title"], level, args.count, args.types, allow_without_materials=forced)
                status, got = quiz.status, len(quiz.questions)
                questions = [{"type": q.type, "prompt": q.prompt, "options": q.options, "answer": q.answer,
                              "explanation": q.explanation, "source": source(q.citation)} for q in quiz.questions]
            except Exception as e:  # noqa: BLE001
                status, got, questions = f"ошибка: {e}"[:200], 0, []
            row.update(status=status, questions=got, seconds=round(time.time() - start), llm_calls=calls["llm"],
                       rejects=dict(rejects), reasons=dict(reasons.most_common(6)), forced=forced, items=questions)
            if status != "no_material":
                break
        rows.append(row)
        with out.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        mark = "OK " if row["questions"] == args.count else ("ЧАСТЬ" if row["questions"] else "СБОЙ")
        print(f"[{i}/{len(topics)}] {mark} {t['id']} ({level}): {row['questions']}/{args.count}, "
              f"{row['seconds']} с, вызовов LLM {row['llm_calls']}, без материалов: {row['forced']}, "
              f"отказы {row['rejects']}", flush=True)

    full = sum(r["questions"] == args.count for r in rows)
    empty = [r["id"] for r in rows if not r["questions"]]
    partial = [f"{r['id']} ({r['questions']})" for r in rows if 0 < r["questions"] < args.count]
    total_rejects = Counter()
    for r in rows:
        total_rejects.update(r["rejects"])
    print(f"\nИтого: полный тест {full}/{len(rows)}, неполный {len(partial)}, ни одного вопроса {len(empty)}")
    print(f"Среднее время {sum(r['seconds'] for r in rows) / max(len(rows), 1):.0f} с, "
          f"максимум {max((r['seconds'] for r in rows), default=0)} с")
    print("Без материалов:", [r["id"] for r in rows if r["forced"]])
    print("Неполные:", partial)
    print("Пустые:", empty)
    print("Причины отказов:", dict(total_rejects))


if __name__ == "__main__":
    main()

"""Оценка RAG-агента на датасете eval/dataset.yaml.

Что считает:
1. Поиск (без LLM) в трёх режимах — семантический, лексический (BM25), гибридный:
   находит ли нужный раздел (Hit@k, MRR) и отсекает ли порог вопросы вне материалов.
2. Ответы агента при одной или нескольких температурах LLM: доля ответов и отказов, ссылки,
   покрытие ключевых фактов, близость к эталону, опора на источники, время.

Запуск:
  uv run python eval/run_eval.py                       # поиск + ответы при температуре из .env
  uv run python eval/run_eval.py --no-llm              # только поиск, несколько секунд
  uv run python eval/run_eval.py --temperatures 0,0.5,1.0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from loguru import logger

from tutor.agent.theory import TheoryAgent
from tutor.evaluation import (
    cosine,
    fact_coverage,
    first_match_rank,
    has_markers,
    mean,
    med,
    pct,
    sentences,
    strip_markers,
    support_scores,
)
from tutor.llm.base import get_chat_model
from tutor.services import get_services

HERE = Path(__file__).parent
MODES = {"dense": "семантический", "lexical": "лексический (BM25)", "hybrid": "гибридный"}
LOW_SUPPORT = 0.5  # предложение ответа считается «без опоры», если близость к источникам ниже


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def eval_retrieval(s, data: list[dict]) -> tuple[list[list[str]], dict]:
    inside = [d for d in data if d["in_materials"]]
    outside = [d for d in data if not d["in_materials"]]
    rows, details = [], {}
    for mode, name in MODES.items():
        ranks, refused = [], 0
        for d in inside:
            r = s.retriever.search(d["q"], mode=mode)
            texts = [f"{h.chunk.file} {h.chunk.doc_title} {h.chunk.section}" for h in r.hits] if r.relevant else []
            ranks.append(first_match_rank(d["expect"], texts))
        for d in outside:
            refused += not s.retriever.search(d["q"], mode=mode).relevant
        found = [r for r in ranks if r]
        details[mode] = {"ranks": ranks, "refused": refused}
        rows.append([name, pct(len(found), len(inside)), pct(sum(r == 1 for r in found), len(inside)),
                     f"{mean([1 / r if r else 0.0 for r in ranks]):.2f}", pct(refused, len(outside))])
    return rows, details


def eval_generation(s, data: list[dict], temperature: float) -> tuple[dict, list[dict]]:
    agent = TheoryAgent(s.retriever, get_chat_model(s.settings, temperature=temperature))
    embed = s.retriever.embedder.embed_documents
    chars = s.settings.context_fragment_chars
    records = []
    for n, d in enumerate(data, 1):
        start, first, result = time.time(), None, None
        for event in agent.stream(d["q"]):
            if event.type == "token" and first is None:
                first = time.time() - start
            elif event.type == "final":
                result = event.result
        rec = {"q": d["q"], "in_materials": d["in_materials"], "kind": d.get("kind", "in"),
               "status": result.status, "answer": result.answer, "top_score": round(result.top_score, 3),
               "citations": [f"{c.file} › {c.section}" for c in result.citations],
               "first_token_s": round(first, 1) if first else None, "total_s": round(time.time() - start, 1)}
        if result.status == "answered":
            clean = strip_markers(result.answer)
            sents = sentences(result.answer) or [clean]
            vecs = embed(sents + [c.text[:chars] for c in result.citations])
            support = support_scores(vecs[:len(sents)], vecs[len(sents):])
            rec |= {"markers": has_markers(result.answer), "support": round(mean(support), 3),
                    "unsupported_sentences": sum(v < LOW_SUPPORT for v in support), "sentences": len(sents)}
            if d["in_materials"]:
                texts = [f"{c.file} {c.title} {c.section}" for c in result.citations]
                a, b = embed([clean, d["reference"]])
                rec |= {"cited_expected": first_match_rank(d["expect"], texts) is not None,
                        "fact_coverage": round(fact_coverage(result.answer, d.get("facts", [])), 2),
                        "similarity": round(cosine(a, b), 3)}
        records.append(rec)
        mark = "✓" if (rec["status"] == "answered") == d["in_materials"] else "✗"
        print(f"  {mark} T={temperature} [{n}/{len(data)}] {rec['status']:11} {rec['total_s']:5.1f}с  {d['q'][:60]}",
              flush=True)

    inside = [r for r in records if r["in_materials"]]
    outside = [r for r in records if not r["in_materials"]]
    answered = [r for r in inside if r["status"] == "answered"]
    leaked = [r for r in outside if r["status"] == "answered"]
    summary = {
        "temperature": temperature,
        "answered": pct(len(answered), len(inside)),
        "false_refusal": pct(len(inside) - len(answered), len(inside)),
        "cited_expected": pct(sum(r["cited_expected"] for r in answered), len(answered)),
        "with_markers": pct(sum(r["markers"] for r in answered), len(answered)),
        "fact_coverage": f"{mean([r['fact_coverage'] for r in answered]):.2f}",
        "similarity": f"{mean([r['similarity'] for r in answered]):.2f}",
        "support": f"{mean([r['support'] for r in answered]):.2f}",
        "unsupported": pct(sum(r["unsupported_sentences"] for r in answered), sum(r["sentences"] for r in answered)),
        "refused_far": pct(sum(r["status"] == "no_material" for r in outside if r["kind"] == "far"),
                           sum(r["kind"] == "far" for r in outside)),
        "refused_near": pct(sum(r["status"] == "no_material" for r in outside if r["kind"] == "near"),
                            sum(r["kind"] == "near" for r in outside)),
        "answered_with_link_outside": len(leaked),
        "first_token_s": f"{med([r['first_token_s'] for r in answered if r['first_token_s']]):.1f}",
        "total_s": f"{med([r['total_s'] for r in answered]):.1f}",
    }
    return summary, records


GEN_ROWS = [
    ("answered", "Ответ дан (вопросы по материалам)"),
    ("false_refusal", "Ложный отказ"),
    ("cited_expected", "Ссылка ведёт на ожидаемый раздел"),
    ("with_markers", "В ответе есть номера ссылок [n]"),
    ("fact_coverage", "Покрытие ключевых фактов эталона (0–1)"),
    ("similarity", "Близость к эталонному ответу (0–1)"),
    ("support", "Опора на источники (0–1)"),
    ("unsupported", f"Предложений без опоры (близость < {LOW_SUPPORT})"),
    ("refused_far", "Отказ: далёкие темы"),
    ("refused_near", "Отказ: близкие темы (другие языки, экзамены)"),
    ("answered_with_link_outside", "Ответов со ссылкой на вопросы вне материалов"),
    ("first_token_s", "Первый токен, медиана, с"),
    ("total_s", "Полный ответ, медиана, с"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-llm", action="store_true", help="оценить только поиск")
    ap.add_argument("--temperatures", default="", help="температуры LLM через запятую, например 0,0.5,1.0")
    ap.add_argument("--dataset", default=str(HERE / "dataset.yaml"))
    ap.add_argument("--limit", type=int, default=0, help="взять только первые N вопросов каждого типа")
    args = ap.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    data = yaml.safe_load(Path(args.dataset).read_text("utf-8"))
    if args.limit:
        data = [d for d in data if d["in_materials"]][:args.limit] + [d for d in data if not d["in_materials"]][:args.limit]
    s = get_services()
    n_in, n_out = sum(d["in_materials"] for d in data), sum(not d["in_materials"] for d in data)
    intro = (f"Датасет: {len(data)} вопросов — {n_in} по материалам, {n_out} вне материалов. "
             f"Модель: `{s.settings.llm_model}`, эмбеддинги: `{s.settings.embeddings_model}`, "
             f"порог релевантности: {s.settings.relevance_threshold}, фрагментов в контексте: {s.settings.retrieval_top_k}.")
    report = [f"# Отчёт eval ({datetime.now():%Y-%m-%d %H:%M})", "", intro, ""]

    print("== Поиск: сравнение режимов", flush=True)
    rows, retrieval = eval_retrieval(s, data)
    block = table(["Режим поиска", f"Раздел найден (Hit@{s.settings.retrieval_top_k})", "На первом месте (Hit@1)",
                   "MRR", "Отсечено порогом (вне материалов)"], rows)
    print(block, "\n", flush=True)
    report += ["## Поиск", "", block, ""]

    results = {"retrieval": retrieval, "generation": {}}
    if not args.no_llm:
        temps = [float(t) for t in args.temperatures.split(",") if t.strip()] or [s.settings.llm_temperature]
        summaries = []
        for t in temps:
            print(f"== Ответы агента, температура {t}", flush=True)
            summary, records = eval_generation(s, data, t)
            summaries.append(summary)
            results["generation"][str(t)] = {"summary": summary, "records": records}
        block = table(["Метрика", *[f"T={x['temperature']}" for x in summaries]],
                      [[name, *[x[key] for x in summaries]] for key, name in GEN_ROWS])
        print("\n" + block, flush=True)
        report += ["## Ответы агента", "", block, ""]
        leaks = [(t, r) for t, g in results["generation"].items() for r in g["records"]
                 if not r["in_materials"] and r["status"] == "answered"]
        report += ["## Ответы на вопросы вне материалов", ""]
        report += [f"- T={t}: «{r['q']}» → {r['answer'][:200]}… (ссылки: {'; '.join(r['citations'][:2])})"
                   for t, r in leaks] or ["Таких ответов нет: на все вопросы вне материалов агент ответил отказом."]
        report.append("")

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out_dir / f"run-{stamp}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), "utf-8")
    (HERE / "report.md").write_text("\n".join(report), "utf-8")
    print(f"\nОтчёт: {HERE / 'report.md'}\nПодробности по каждому вопросу: {out_dir / f'run-{stamp}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

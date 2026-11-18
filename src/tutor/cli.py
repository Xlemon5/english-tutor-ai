"""CLI для отладки: tutor manifest | ingest | search | stats."""
from __future__ import annotations

import sys
from collections import Counter

import typer
from loguru import logger

from tutor.config import get_settings
from tutor.ingestion.indexer import ingest as run_ingest
from tutor.ingestion.manifest import build_manifest, coverage_gaps
from tutor.llm.base import get_embedder
from tutor.retrieval.citations import build_citations
from tutor.retrieval.hybrid import THEORY_KINDS, Retriever
from tutor.retrieval.store import ChunkStore

app = typer.Typer(add_completion=False, help="Репетитор по английскому: служебные команды.")
logger.remove()
logger.add(sys.stderr, level="INFO", format="<level>{level: <7}</level> {message}")


@app.command()
def manifest() -> None:
    """Пересобрать materials/sources.yaml и отчёт о покрытии."""
    s = get_settings()
    entries = build_manifest(s.materials_dir)
    by_license = Counter(e.license for e in entries)
    typer.echo(f"Файлов в манифесте: {len(entries)}")
    for lic, n in by_license.most_common():
        typer.echo(f"  {n:4d}  {lic}")
    gaps = coverage_gaps(s.materials_dir)
    typer.echo(f"Пробелов в карте тем: {len(gaps)}")
    for g in gaps:
        typer.echo(f"  нет: {g}")


@app.command()
def ingest(full: bool = typer.Option(False, "--full", help="Переиндексировать все файлы.")) -> None:
    """Проиндексировать materials/ (только изменившиеся файлы)."""
    s = get_settings()
    store = ChunkStore(s)
    report = run_ingest(s, get_embedder(s), store, full=full)
    typer.echo(report.summary())
    typer.echo(f"Всего чанков в индексе: {store.count()}")
    for file, err in report.failed.items():
        typer.echo(f"  ошибка: {file}: {err}")


@app.command()
def search(
    query: str,
    k: int = typer.Option(0, "-k", help="Сколько фрагментов показать."),
    level: str = typer.Option("", "--level", help="A1, A2 или B1."),
    all_kinds: bool = typer.Option(False, "--all", help="Искать и среди примеров предложений."),
    mode: str = typer.Option("hybrid", "--mode", help="hybrid, dense (только семантика) или lexical (только BM25)."),
) -> None:
    """Найти фрагменты материалов по запросу и показать ссылки."""
    s = get_settings()
    retriever = Retriever(ChunkStore(s), get_embedder(s), s)
    result = retriever.search(query, k or None, None if all_kinds else THEORY_KINDS, level or None, mode)
    verdict = "релевантно" if result.relevant else "В МАТЕРИАЛАХ ЭТОГО НЕТ"
    typer.echo(f"Лучший скор: {result.top_score:.3f} (порог {s.relevance_threshold}) — {verdict}\n")
    for c, h in zip(build_citations(result.hits), result.hits, strict=True):
        ranks = f"семантика №{h.dense_rank or '—'}, BM25 №{h.lexical_rank or '—'}"
        typer.echo(f"{c.label()}  [скор {c.score:.3f}; {ranks}]")
        typer.echo("    " + c.text[:300].replace("\n", "\n    ") + ("…" if len(c.text) > 300 else "") + "\n")


@app.command()
def ask(question: str, level: str = typer.Option("", "--level", help="A1, A2 или B1.")) -> None:
    """Задать вопрос по теории: ответ со ссылками или «В материалах этого нет»."""
    from tutor.llm.structured import StructuredOutputError
    from tutor.services import get_services

    try:
        events = list(get_services().theory.stream(question, level or None))
    except StructuredOutputError as e:
        typer.echo(f"⚠ {e}")
        raise typer.Exit(1) from e
    for event in events:
        if event.type == "token":
            typer.echo(event.text, nl=False)
            continue
        result = event.result
        if result.status == "no_material":
            typer.echo(f"\n⚠ {result.answer} (лучший скор {result.top_score:.3f})")
            if result.suggestions:
                typer.echo("Близкие темы в материалах:")
                for t in result.suggestions:
                    typer.echo(f"  • {t}")
        else:
            typer.echo("\n\nИсточники:")
            for c in result.citations:
                typer.echo("  " + c.label())


@app.command()
def review(
    text: str = typer.Argument("", help="Текст работы (или --file)."),
    file: str = typer.Option("", "--file", help="Файл с текстом работы."),
    task: str = typer.Option("", "--task", help="Формулировка задания."),
) -> None:
    """Проверить работу студента: ошибки, объяснения, ссылки на теорию."""
    from pathlib import Path

    from tutor.services import get_services

    body = Path(file).read_text("utf-8") if file else text
    result = get_services().review.review(body, task)
    for n, e in enumerate(result.errors, 1):
        typer.echo(f"{n}. «{e.fragment}» → «{e.correction}»  [{e.type_ru}]")
        typer.echo(f"   {e.explanation}")
        typer.echo("   Теория: " + (e.citation.label() if e.citation else e.theory_note))
    typer.echo(f"\nОценка: {result.grade}/5, примерный уровень: {result.cefr_level}")
    typer.echo(result.summary)
    typer.echo("Повторить: " + "; ".join(result.review_topics))


@app.command()
def quiz(
    topic: str,
    level: str = typer.Option("A2", "--level"),
    n: int = typer.Option(5, "-n", help="Количество вопросов."),
    types: str = typer.Option("", "--types", help="Через запятую: multiple_choice,fill_gap,error_correction,translation"),
    take: bool = typer.Option(False, "--take", help="Пройти тест в терминале."),
    force: bool = typer.Option(False, "--force", help="Генерировать, даже если материалов по теме нет."),
) -> None:
    """Сгенерировать тест по теме и (с --take) пройти его."""
    from tutor.services import get_services

    agent = get_services().quiz
    result = agent.generate(topic, level, n, [t for t in types.split(",") if t] or None, force)
    if result.status == "no_material":
        typer.echo("⚠ " + result.message)
        for t in result.suggestions:
            typer.echo(f"  • {t}")
        raise typer.Exit(1)
    if not result.grounded:
        typer.echo("⚠ Тест составлен без опоры на материалы.")
    answers: dict[int, str] = {}
    for q in result.questions:
        typer.echo(f"\n{q.id}. [{q.type_ru}] {q.prompt}")
        for opt in q.options:
            typer.echo(f"     - {opt}")
        if take:
            answers[q.id] = typer.prompt("   Ваш ответ", default="", show_default=False)
        else:
            typer.echo(f"   Ответ: {q.answer}\n   {q.explanation}")
            typer.echo("   Теория: " + (q.citation.label() if q.citation else "нет ссылки"))
    if take:
        checked = agent.check(result, answers)
        typer.echo(f"\nРезультат: {checked.score} из {checked.total}")
        for r in checked.results:
            mark = "✓" if r.correct else "✗"
            typer.echo(f"{mark} {r.question.id}. ваш ответ: «{r.given}», правильно: «{r.question.answer}»")
            typer.echo(f"   {r.comment or r.question.explanation}")
            if r.question.citation:
                typer.echo("   Теория: " + r.question.citation.label())


@app.command()
def plan(
    to: str = typer.Option("B1", "--to", help="До какого уровня: A1, A2 или B1."),
    start: str = typer.Option("A1", "--from", help="С какого уровня."),
) -> None:
    """Программа обучения: темы по уровням и материалы библиотеки к каждой."""
    from tutor.agent.plan import build_plan, plan_summary

    result = build_plan(get_settings().materials_dir, to, start)
    typer.echo(plan_summary(result))
    for stage in result.stages:
        typer.echo(f"\n=== Уровень {stage.level}")
        area = ""
        for t in stage.topics:
            if t.area_ru != area:
                area = t.area_ru
                typer.echo(f"\n  {area}")
            typer.echo(f"    {'•' if t.materials else '○'} {t.title}" + ("" if t.materials else "  — материалов нет"))
            for m in t.materials:
                typer.echo(f"        {m.kind_ru}: {m.title} — {m.file}")


@app.command()
def stats() -> None:
    """Сводка по индексу."""
    s = get_settings()
    chunks = ChunkStore(s).all_chunks()
    typer.echo(f"Чанков: {len(chunks)}, файлов: {len({c.file for c in chunks})}")
    for title, counter in (("По типу", Counter(c.kind for c in chunks)),
                           ("По уровню", Counter(c.level for c in chunks)),
                           ("По лицензии", Counter(c.license for c in chunks))):
        typer.echo(f"{title}:")
        for key, n in counter.most_common():
            typer.echo(f"  {n:5d}  {key}")


if __name__ == "__main__":
    app()

from __future__ import annotations

from pathlib import Path

import pytest

from tutor.agent.plan import build_plan, normalize_levels, plan_summary
from tutor.ingestion.manifest import build_manifest


def test_plan_is_built_only_from_library(materials: Path):
    (materials / "web" / "book" / "articles-practice.md").write_text(
        "---\ntitle: Articles practice\nkind: worksheet\nlicense: CC BY 4.0\n---\n\n# Articles practice\n\ntext\n", "utf-8")
    (materials / "web" / "book" / "examples.md").write_text(
        "---\ntitle: Articles examples\nkind: examples\n---\n\n# Articles examples\n\ntext\n", "utf-8")
    manifest = {e.file for e in build_manifest(materials)}
    plan = build_plan(materials, "B1")
    assert (plan.start_level, plan.target_level) == ("A1", "B1")
    topics = {t.id: t for stage in plan.stages for t in stage.topics}
    # каждая ссылка программы ведёт на файл из манифеста; примеры предложений в программу не попадают
    for topic in topics.values():
        assert all(m.file in manifest for m in topic.materials)
    articles = topics["a1-articles"]
    assert [m.file for m in articles.materials] == ["web/book/articles.md", "web/book/articles-practice.md"]
    assert [m.kind_ru for m in articles.materials] == ["теория", "упражнения"]  # теория раньше упражнений
    assert not topics["b1-passive"].covered and plan.gaps == ["B1 · Грамматика · Пассивный залог"]
    assert plan_summary(plan) == "Программа обучения, уровень A1–B1. Тем: 3, с материалами в библиотеке: 2. Без материалов: 1."


def test_plan_level_range(materials: Path):
    build_manifest(materials)
    assert [s.level for s in build_plan(materials, "A1").stages] == ["A1"]
    assert [s.level for s in build_plan(materials, "B1", "B1").stages] == ["B1"]
    assert normalize_levels("B1", "A1") == ("A1", "B1")  # перепутанный порядок исправляется
    assert normalize_levels(None, "a2") == ("A1", "A2")
    with pytest.raises(ValueError):
        build_plan(materials, "C1")

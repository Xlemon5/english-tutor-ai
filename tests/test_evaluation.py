from __future__ import annotations

import re
from pathlib import Path

import yaml

from tutor.evaluation import (
    cosine,
    fact_coverage,
    first_match_rank,
    has_markers,
    pct,
    sentences,
    strip_markers,
    support_scores,
)

DATASET = Path(__file__).parents[1] / "eval" / "dataset.yaml"


def test_markers_and_sentences():
    answer = "Окончание -s добавляется после he и she [1]. Например: *he works* [2].\n- Это третье лицо единственного числа."
    assert has_markers(answer) and not has_markers("Без ссылок.")
    assert strip_markers(answer).count("[") == 0
    assert sentences(answer) == ["Окончание -s добавляется после he и she.", "Это третье лицо единственного числа."]


def test_rank_and_fact_coverage():
    texts = ["web/a/articles.md Articles", "web/b/present-perfect.md Present perfect › Usage"]
    assert first_match_rank(r"present[ -_]perfect", texts) == 2
    assert first_match_rank("passive", texts) is None
    assert fact_coverage("A ставится перед согласным звуком.", ["согласн|consonant", "(?<!со)гласн|vowel"]) == 0.5
    assert fact_coverage("anything", []) == 1.0


def test_support_and_cosine():
    assert cosine([1, 0], [1, 0]) == 1.0 and cosine([1, 0], [0, 1]) == 0.0
    assert support_scores([[1, 0], [0, 1]], [[1, 0]]) == [1.0, 0.0]
    assert support_scores([[1, 0]], []) == [0.0]
    assert pct(3, 4) == "3/4 (75%)" and pct(0, 0) == "—"


def test_dataset_is_well_formed():
    data = yaml.safe_load(DATASET.read_text("utf-8"))
    inside = [d for d in data if d["in_materials"]]
    outside = [d for d in data if not d["in_materials"]]
    assert len(inside) >= 25 and len(outside) >= 10
    assert len({d["q"] for d in data}) == len(data)
    for d in inside:
        assert len(d["reference"]) > 40 and d["facts"]
        re.compile(d["expect"])
        # эталонный ответ сам должен содержать все свои ключевые факты
        assert fact_coverage(d["reference"], d["facts"]) == 1.0, d["q"]
    assert {d["kind"] for d in outside} == {"far", "near"}

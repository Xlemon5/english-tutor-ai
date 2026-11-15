"""Calibrate the pinned NLI model on a labeled JSONL validation set.

Each line must contain: {"premise": "...", "hypothesis": "...",
"label": "entailment" | "neutral" | "contradiction"}.

Example:
    python calibrate_nli_threshold.py data/nli_validation.jsonl --min-precision 0.90
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verification import NLIModel


def load_examples(path: Path) -> list[dict[str, str]]:
    examples = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"Line {line_number}: each JSONL entry must be an object")
        if not all(isinstance(item.get(key), str) and item[key].strip() for key in ("premise", "hypothesis")):
            raise ValueError(f"Line {line_number}: premise and hypothesis are required strings")
        if item.get("label") not in {"entailment", "neutral", "contradiction"}:
            raise ValueError(f"Line {line_number}: label must be entailment, neutral, or contradiction")
        examples.append(item)
    if not examples:
        raise ValueError("Validation file contains no examples")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation_jsonl", type=Path)
    parser.add_argument("--min-precision", type=float, default=0.90)
    parser.add_argument(
        "--target",
        choices=("groundedness", "strict"),
        default="groundedness",
        help="Choose the chat groundedness threshold or the stricter verification threshold.",
    )
    args = parser.parse_args()
    if not 0 < args.min_precision <= 1:
        parser.error("--min-precision must be in (0, 1]")

    examples = load_examples(args.validation_jsonl)
    model = NLIModel()
    probabilities = model.score_pairs([
        (item["premise"], item["hypothesis"]) for item in examples
    ])
    gold = [item["label"] == "entailment" for item in examples]

    candidates = []
    for step in range(30, 96):
        threshold = step / 100
        predicted = [score["entailment"] >= threshold for score in probabilities]
        true_positive = sum(actual and guess for actual, guess in zip(gold, predicted))
        false_positive = sum((not actual) and guess for actual, guess in zip(gold, predicted))
        false_negative = sum(actual and (not guess) for actual, guess in zip(gold, predicted))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        if precision >= args.min_precision:
            candidates.append((recall, precision, threshold))

    if not candidates:
        raise SystemExit("No threshold reaches the requested precision; add or review validation examples.")
    recall, precision, threshold = max(candidates)
    print(json.dumps({
        "model": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "examples": len(examples),
        "min_precision": args.min_precision,
        "recommended_threshold": threshold,
        "validation_precision": round(precision, 4),
        "validation_recall": round(recall, 4),
        "target": args.target,
        "environment_variable": (
            "RAG_GROUNDEDNESS_THRESHOLD"
            if args.target == "groundedness"
            else "RAG_STRICT_NLI_THRESHOLD"
        ),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Post-generation checks for answer grounding, citations, and quiz structure."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
import onnxruntime as ort


NLI_REPOSITORY = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
NLI_REVISION = "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"
DEFAULT_GROUNDEDNESS_THRESHOLD = 0.50
DEFAULT_STRICT_VERIFICATION_THRESHOLD = 0.70


def normalize_text(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def local_evidence_span(source: str, quote: str, radius: int = 400) -> str:
    """Return a bounded source window around an exact quote for contextual NLI."""
    normalized_source = normalize_text(source)
    normalized_quote = normalize_text(quote)
    position = normalized_source.find(normalized_quote)
    if position < 0:
        return normalized_quote
    return normalized_source[max(0, position - radius):position + len(normalized_quote) + radius]


def is_russian_text(value: str) -> bool:
    letters = [char for char in value if char.isalpha()]
    if not letters:
        return False
    cyrillic = sum("\u0400" <= char <= "\u04FF" for char in letters)
    return cyrillic / len(letters) >= 0.35


class NLIModel:
    """CPU ONNX runner for multilingual entailment/neutral/contradiction scores."""

    def __init__(self) -> None:
        model_path = hf_hub_download(
            repo_id=NLI_REPOSITORY,
            filename="onnx/model_quantized.onnx",
            revision=NLI_REVISION,
        )
        tokenizer_path = hf_hub_download(
            repo_id=NLI_REPOSITORY,
            filename="tokenizer.json",
            revision=NLI_REVISION,
        )
        config_path = hf_hub_download(
            repo_id=NLI_REPOSITORY,
            filename="config.json",
            revision=NLI_REVISION,
        )

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(os.cpu_count() or 1, 4))
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=512, strategy="longest_first")
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        if self.pad_id is None:
            raise RuntimeError("NLI tokenizer has no [PAD] token")

        with open(config_path, encoding="utf-8") as config_file:
            config = json.load(config_file)
        id2label = config.get("id2label", {})
        self.labels = {int(index): label.casefold() for index, label in id2label.items()}
        if not {"entailment", "neutral", "contradiction"}.issubset(set(self.labels.values())):
            raise RuntimeError("NLI model config is missing expected labels")

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        if not pairs:
            return []
        encodings = [self.tokenizer.encode(premise, hypothesis) for premise, hypothesis in pairs]
        max_length = max(len(item.ids) for item in encodings)
        input_ids = np.full((len(encodings), max_length), self.pad_id, dtype=np.int64)
        attention_mask = np.zeros((len(encodings), max_length), dtype=np.int64)
        for row, encoding in enumerate(encodings):
            length = len(encoding.ids)
            input_ids[row, :length] = encoding.ids
            attention_mask[row, :length] = encoding.attention_mask

        logits = self.session.run(
            ["logits"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )[0]
        logits = logits - np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        return [
            {self.labels[index]: float(probability[index]) for index in self.labels}
            for probability in probabilities
        ]


class VerificationLayer:
    """Fails closed when claims, citations, self-consistency, or quiz checks fail."""

    FAILURE_TYPES = {
        "unsupported_claim",
        "internal_contradiction",
        "bad_citation",
        "wrong_count",
        "citation_drift",
    }

    def __init__(self, threshold: float = DEFAULT_GROUNDEDNESS_THRESHOLD) -> None:
        self.groundedness_threshold = threshold
        # Keep translation, self-consistency, and quiz answer checks stricter.
        self.translation_threshold = DEFAULT_STRICT_VERIFICATION_THRESHOLD
        self.threshold = DEFAULT_STRICT_VERIFICATION_THRESHOLD
        self._nli: NLIModel | None = None
        self._load_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self.log_path = Path(__file__).resolve().parent / "data" / "verification_failures.jsonl"

    def record_failure(self, failure_type: str, operation: str, **details: Any) -> None:
        if failure_type not in self.FAILURE_TYPES:
            failure_type = "unsupported_claim"
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": failure_type,
            "operation": operation,
            "details": details,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_lock, self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _model(self) -> NLIModel:
        if self._nli is None:
            with self._load_lock:
                if self._nli is None:
                    self._nli = NLIModel()
        return self._nli

    def _scores(self, pairs: list[tuple[str, str]], operation: str) -> list[dict[str, float]] | None:
        try:
            return self._model().score_pairs(pairs)
        except Exception as error:
            self.record_failure(
                "unsupported_claim",
                operation,
                reason="nli_unavailable",
                error=type(error).__name__,
            )
            return None

    def verify_claims(
        self,
        claims: list[dict[str, Any]],
        retrieved: dict[str, dict[str, Any]],
        operation: str = "chat",
    ) -> tuple[bool, float]:
        if not claims:
            self.record_failure("unsupported_claim", operation, reason="no_atomic_claims")
            return False, 0.0

        pairs: list[tuple[str, str]] = []
        russian_sources: list[bool] = []
        for claim in claims:
            source_id = claim.get("source_id")
            quote = claim.get("quote")
            text = claim.get("text")
            claim_english = claim.get("claim_english")
            claim_source = claim.get("claim_source")
            if not isinstance(source_id, str) or source_id not in retrieved:
                self.record_failure("bad_citation", operation, reason="source_id_not_retrieved")
                return False, 0.0
            if not all(isinstance(value, str) and value.strip() for value in (quote, text, claim_english, claim_source)):
                self.record_failure("unsupported_claim", operation, reason="claim_missing_text_or_evidence")
                return False, 0.0

            source = retrieved[source_id].get("content", "")
            normalized_quote = normalize_text(quote)
            if len(normalized_quote) < 30 or normalized_quote not in normalize_text(source):
                self.record_failure("citation_drift", operation, reason="quote_not_in_cited_source")
                return False, 0.0

            # Ground against the exact cited span. For Russian sources, render
            # the verified Russian claim itself; only English/other sources
            # need a separate Russian translation check.
            pairs.append((quote, claim_source))
            russian_source = is_russian_text(quote)
            russian_sources.append(russian_source)
            if not russian_source:
                pairs.append((claim_source, text))

        scores = self._scores(pairs, operation)
        if scores is None:
            return False, 0.0
        groundedness = 1.0
        cursor = 0
        for claim_index, claim in enumerate(claims):
            source_entailment = scores[cursor].get("entailment", 0.0)
            cursor += 1
            if russian_sources[claim_index]:
                translation_entailment = 1.0
                claim["_verified_text"] = claim["claim_source"].strip()
            else:
                translation_entailment = scores[cursor].get("entailment", 0.0)
                cursor += 1
                claim["_verified_text"] = claim["text"].strip()
            claim_score = min(source_entailment, translation_entailment)
            groundedness = min(groundedness, claim_score)
            if source_entailment < self.groundedness_threshold or translation_entailment < self.translation_threshold:
                self.record_failure(
                    "unsupported_claim",
                    operation,
                    source_entailment=round(source_entailment, 4),
                    translation_entailment=round(translation_entailment, 4),
                    groundedness_threshold=self.groundedness_threshold,
                    translation_threshold=self.translation_threshold,
                    quote_preview=claim.get("quote", "")[:240],
                    source_claim_preview=claim.get("claim_source", "")[:240],
                    answer_claim_preview=claim.get("text", "")[:240],
                )
                return False, groundedness
        return True, groundedness

    def verify_consistency(
        self,
        candidates: list[dict[str, Any]],
        operation: str = "chat",
    ) -> tuple[bool, list[dict[str, Any]]]:
        abstentions = [bool(candidate.get("abstain")) for candidate in candidates]
        if all(abstentions):
            return True, []
        if any(abstentions):
            self.record_failure("internal_contradiction", operation, reason="mixed_abstention")
            return False, []

        claim_sets = [candidate.get("claims", []) for candidate in candidates]
        pairs: list[tuple[str, str]] = []
        for left_index in range(len(claim_sets)):
            for right_index in range(left_index + 1, len(claim_sets)):
                for left_claim in claim_sets[left_index]:
                    for right_claim in claim_sets[right_index]:
                        pairs.append((left_claim["claim_english"], right_claim["claim_english"]))
                        pairs.append((right_claim["claim_english"], left_claim["claim_english"]))

        scores = self._scores(pairs, operation)
        if scores is None:
            return False, []

        cursor = 0
        for left_index in range(len(claim_sets)):
            for right_index in range(left_index + 1, len(claim_sets)):
                for _left_claim in claim_sets[left_index]:
                    for _right_claim in claim_sets[right_index]:
                        left_to_right = scores[cursor]
                        right_to_left = scores[cursor + 1]
                        cursor += 2
                        if left_to_right.get("contradiction", 0.0) >= self.threshold or right_to_left.get("contradiction", 0.0) >= self.threshold:
                            self.record_failure("internal_contradiction", operation, reason="nli_contradiction")
                            return False, []
        return True, claim_sets[0]

    def verify_quiz_items(
        self,
        questions: list[dict[str, Any]],
        retrieved: dict[str, dict[str, Any]],
        requested_count: int,
    ) -> bool:
        if len(questions) != requested_count:
            self.record_failure(
                "wrong_count",
                "quiz",
                requested=requested_count,
                received=len(questions),
            )
            return False

        pairs: list[tuple[str, str]] = []
        question_positions: list[tuple[int, str]] = []
        for index, question in enumerate(questions):
            source_id = question.get("source_id")
            evidence = question.get("evidence")
            question_claim = question.get("question_claim")
            options = question.get("options")
            correct = question.get("correct")
            option_claims = question.get("option_claims")
            if not isinstance(source_id, str) or source_id not in retrieved:
                self.record_failure("bad_citation", "quiz", question=index, reason="source_id_not_retrieved")
                return False
            if not isinstance(evidence, str) or not isinstance(question_claim, str) or not isinstance(options, dict) or not isinstance(option_claims, dict):
                self.record_failure("bad_citation", "quiz", question=index, reason="missing_evidence_or_options")
                return False
            if set(options) != {"A", "B", "C", "D"} or set(option_claims) != {"A", "B", "C", "D"}:
                self.record_failure("unsupported_claim", "quiz", question=index, reason="invalid_option_set")
                return False
            if correct not in {"A", "B", "C", "D"}:
                self.record_failure("unsupported_claim", "quiz", question=index, reason="invalid_correct_option")
                return False
            source = retrieved[source_id].get("content", "")
            normalized_evidence = normalize_text(evidence)
            if len(normalized_evidence) < 30 or normalized_evidence not in normalize_text(source):
                self.record_failure("citation_drift", "quiz", question=index, reason="quote_not_in_cited_source")
                return False
            if any(not isinstance(value, str) or not value.strip() for value in option_claims.values()):
                self.record_failure("unsupported_claim", "quiz", question=index, reason="empty_option_claim")
                return False
            evidence_span = local_evidence_span(source, evidence)
            pairs.append((evidence_span, question_claim))
            question_positions.append((index, "QUESTION"))
            for option in ("A", "B", "C", "D"):
                option_text = options[option]
                option_claim = option_claims[option]
                if normalize_text(option_text) not in normalize_text(option_claim):
                    self.record_failure(
                        "unsupported_claim",
                        "quiz",
                        question=index,
                        option=option,
                        reason="visible_option_not_reflected_in_verified_claim",
                    )
                    return False
                pairs.append((evidence_span, option_claim))
                question_positions.append((index, option))

        scores = self._scores(pairs, "quiz")
        if scores is None:
            return False
        score_by_question: dict[int, dict[str, float]] = {}
        for (question_index, option), score in zip(question_positions, scores):
            score_by_question.setdefault(question_index, {})[option] = score.get("entailment", 0.0)

        for index, question in enumerate(questions):
            question_score = score_by_question[index].get("QUESTION", 0.0)
            if question_score < self.threshold:
                self.record_failure(
                    "unsupported_claim",
                    "quiz",
                    question=index,
                    reason="question_not_entailed_by_evidence",
                    entailment_score=round(question_score, 4),
                    threshold=self.threshold,
                )
                return False
            entailed = [
                option for option, score in score_by_question[index].items()
                if option != "QUESTION" and score >= self.threshold
            ]
            if entailed != [question["correct"]]:
                self.record_failure(
                    "unsupported_claim",
                    "quiz",
                    question=index,
                    reason="correct_option_not_uniquely_entailed",
                    entailed_options=entailed,
                    expected=question["correct"],
                    threshold=self.threshold,
                )
                return False
        return True

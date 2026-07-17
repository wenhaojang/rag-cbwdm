"""Classification and selection diagnostics."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any


class ClassificationMetrics:
    def __init__(self, labels: list[str] | None = None) -> None:
        self.labels = list(labels or [])
        self.num_examples = 0
        self.num_correct = 0
        self.evidence_chars: list[int] = []
        self.num_docs: list[int] = []
        self.by_gold: dict[str, Counter[str]] = defaultdict(Counter)
        self.prediction_distribution: Counter[str] = Counter()
        self.nan_inf_count = 0
        self.missing_prediction_count = 0

    def update(
        self,
        gold: str | None,
        pred: str | None,
        num_docs: int = 0,
        evidence_chars: int = 0,
        probs: list[float] | None = None,
    ) -> None:
        self.num_examples += 1
        if pred is None:
            self.missing_prediction_count += 1
        else:
            self.prediction_distribution[str(pred)] += 1
        if probs and any(not math.isfinite(float(value)) for value in probs):
            self.nan_inf_count += 1
        if gold is not None and pred is not None:
            self.num_correct += int(pred == gold)
            self.by_gold[str(gold)][str(pred)] += 1
        self.num_docs.append(int(num_docs))
        self.evidence_chars.append(int(evidence_chars))

    def compute(self) -> dict[str, Any]:
        labels = self.labels or sorted(
            set(self.by_gold) | set(self.prediction_distribution)
        )
        per_class = {}
        confusion = {}
        for gold in labels:
            counts = self.by_gold.get(gold, Counter())
            total = sum(counts.values())
            per_class[gold] = {
                "num_examples": total,
                "num_correct": counts.get(gold, 0),
                "accuracy": counts.get(gold, 0) / total if total else 0.0,
            }
            confusion[gold] = {pred: counts.get(pred, 0) for pred in labels}
        hist = Counter(str(value) for value in self.num_docs)
        return {
            "schema_version": "rag_cbwdm_metrics.v2",
            "num_examples": self.num_examples,
            "num_correct": self.num_correct,
            "accuracy": self.num_correct / self.num_examples if self.num_examples else 0.0,
            "per_class": per_class,
            "confusion_matrix": confusion,
            "prediction_distribution": {
                label: self.prediction_distribution.get(label, 0) for label in labels
            },
            "num_docs": {
                "average": statistics.fmean(self.num_docs) if self.num_docs else 0.0,
                "median": statistics.median(self.num_docs) if self.num_docs else 0.0,
                "min": min(self.num_docs) if self.num_docs else 0,
                "max": max(self.num_docs) if self.num_docs else 0,
                "histogram": dict(sorted(hist.items(), key=lambda item: int(item[0]))),
            },
            "avg_num_docs": statistics.fmean(self.num_docs) if self.num_docs else 0.0,
            "avg_evidence_chars": (
                statistics.fmean(self.evidence_chars) if self.evidence_chars else 0.0
            ),
            "nan_inf_count": self.nan_inf_count,
            "missing_prediction_count": self.missing_prediction_count,
        }


def compute_accuracy(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ClassificationMetrics()
    for row in predictions:
        metrics.update(
            gold=row.get("gold"),
            pred=row.get("pred"),
            num_docs=int(row.get("num_docs", 0)),
            evidence_chars=int(row.get("evidence_chars", 0)),
            probs=row.get("probs"),
        )
    return metrics.compute()

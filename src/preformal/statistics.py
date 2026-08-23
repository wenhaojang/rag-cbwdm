"""Fairness, summary, and paired statistics for preformal_eval."""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from typing import Any, Iterable

from src.metrics import ClassificationMetrics


def indexed_rows(rows: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = str(row.get("id"))
        if not identifier or identifier == "None": raise ValueError(f"{label} row lacks id")
        if identifier in result: raise ValueError(f"Duplicate {label} id: {identifier}")
        result[identifier] = row
    return result


def require_identical_ids(left: dict[str, Any], right: dict[str, Any], left_name: str, right_name: str) -> list[str]:
    if set(left) != set(right):
        raise ValueError(f"Paired IDs differ: {left_name}-only={sorted(set(left)-set(right))[:5]} {right_name}-only={sorted(set(right)-set(left))[:5]}")
    return sorted(left)


def _macro_f1(rows: list[dict[str, Any]], labels: list[str]) -> float:
    metric = ClassificationMetrics(labels=labels)
    for row in rows: metric.update(gold=str(row["gold"]), pred=str(row["pred"]))
    return float(metric.compute()["macro_f1"])


def exact_mcnemar_p_value(signed_only: int, baseline_only: int) -> float:
    discordant = signed_only + baseline_only
    if discordant == 0: return 1.0
    tail = sum(math.comb(discordant, index) for index in range(0, min(signed_only, baseline_only) + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered: raise ValueError("Cannot take percentile of empty values")
    position = (len(ordered) - 1) * probability; lower = int(math.floor(position)); upper = int(math.ceil(position))
    if lower == upper: return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def paired_comparison(signed_rows: Iterable[dict[str, Any]], baseline_rows: Iterable[dict[str, Any]], *,
                      bootstrap_seed: int = 130421, samples: int = 10000) -> dict[str, Any]:
    signed = indexed_rows(signed_rows, "signed"); baseline = indexed_rows(baseline_rows, "baseline")
    ids = require_identical_ids(signed, baseline, "signed", "baseline")
    if samples < 1: raise ValueError("bootstrap samples must be positive")
    labels = sorted({str(signed[item]["gold"]) for item in ids})
    for identifier in ids:
        if signed[identifier].get("gold") != baseline[identifier].get("gold"):
            raise ValueError(f"Gold label mismatch for paired id {identifier}")
    cells = Counter()
    for identifier in ids:
        signed_correct = str(signed[identifier]["pred"]) == str(signed[identifier]["gold"])
        baseline_correct = str(baseline[identifier]["pred"]) == str(baseline[identifier]["gold"])
        cells[(signed_correct, baseline_correct)] += 1
    rng = random.Random(bootstrap_seed); accuracy_diffs: list[float] = []; macro_diffs: list[float] = []
    n = len(ids)
    for _ in range(samples):
        sampled = [ids[rng.randrange(n)] for _ in range(n)]
        signed_sample = [signed[item] for item in sampled]; baseline_sample = [baseline[item] for item in sampled]
        signed_acc = sum(str(row["pred"]) == str(row["gold"]) for row in signed_sample) / n
        baseline_acc = sum(str(row["pred"]) == str(row["gold"]) for row in baseline_sample) / n
        accuracy_diffs.append(signed_acc - baseline_acc)
        macro_diffs.append(_macro_f1(signed_sample, labels) - _macro_f1(baseline_sample, labels))
    signed_only = cells[(True, False)]; baseline_only = cells[(False, True)]
    return {"num_examples": n, "paired_ids_sha_order": __import__("hashlib").sha256("\n".join(ids).encode()).hexdigest(),
        "correctness_table": {"both_correct": cells[(True, True)], "signed_only_correct": signed_only,
            "baseline_only_correct": baseline_only, "both_wrong": cells[(False, False)]},
        "mcnemar_exact_two_sided_p": exact_mcnemar_p_value(signed_only, baseline_only),
        "bootstrap": {"seed": bootstrap_seed, "samples": samples,
            "accuracy_difference_signed_minus_baseline": {"mean": statistics.fmean(accuracy_diffs),
                "ci95": [percentile(accuracy_diffs, .025), percentile(accuracy_diffs, .975)]},
            "macro_f1_difference_signed_minus_baseline": {"mean": statistics.fmean(macro_diffs),
                "ci95": [percentile(macro_diffs, .025), percentile(macro_diffs, .975)]}}}


def mean_sd(values: list[float]) -> dict[str, float | None]:
    return {"mean": statistics.fmean(values), "sd": statistics.stdev(values) if len(values) > 1 else None}

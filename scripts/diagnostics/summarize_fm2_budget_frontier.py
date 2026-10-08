"""Summarize an FM2 evidence-budget frontier plan without interpolation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnostics.run_fm2_budget_frontier import (
    PLAN_SCHEMA_VERSION,
    FrontierPlanError,
    validate_metrics_reference,
)
from src.run_manifest import atomic_write_json, stable_hash


SUMMARY_SCHEMA_VERSION = "rag_cbwdm_fm2_budget_frontier_summary.v1"
ROW_FIELDS = (
    "generator_id",
    "frontier_method_id",
    "underlying_method",
    "variant",
    "frontier_role",
    "frontier_status",
    "nominal_cap",
    "source",
    "effective_equivalence",
    "contract_preserving",
    "accuracy",
    "macro_f1",
    "num_correct",
    "num_examples",
    "avg_num_docs",
    "avg_evidence_chars",
    "avg_original_retrieval_rank",
    "metrics_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize an FM2 budget-frontier plan.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def _load_plan(path: str | Path) -> dict[str, Any]:
    try:
        plan = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrontierPlanError(f"Invalid frontier plan {path}: {exc}") from exc
    if not isinstance(plan, dict) or plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise FrontierPlanError(f"Frontier plan schema mismatch: {path}")
    if not isinstance(plan.get("points"), list):
        raise FrontierPlanError(f"Frontier plan lacks points: {path}")
    fingerprint_payload = dict(plan)
    recorded_fingerprint = fingerprint_payload.pop("plan_fingerprint", None)
    if recorded_fingerprint != stable_hash(fingerprint_payload):
        raise FrontierPlanError(f"Frontier plan fingerprint mismatch: {path}")
    return plan


def _metrics_path(point: dict[str, Any]) -> str:
    if point.get("source") == "evaluation":
        path = point.get("output_metrics_path")
    else:
        path = point.get("native_metrics_path")
    if not isinstance(path, str) or not path:
        raise FrontierPlanError(
            f"Logical point lacks a readable metrics path: {point.get('generator_id')} "
            f"{point.get('frontier_method_id')} cap={point.get('nominal_cap')}"
        )
    return path


def build_summary(plan: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    expected_rows = int(plan["expected_rows"])
    for point in plan["points"]:
        metrics_path = _metrics_path(point)
        metrics = validate_metrics_reference(
            metrics_path,
            expected_rows=expected_rows,
            split=str(plan["split"]),
        )
        rows.append(
            {
                "generator_id": point["generator_id"],
                "frontier_method_id": point["frontier_method_id"],
                "underlying_method": point["underlying_method"],
                "variant": point["variant"],
                "frontier_role": point["frontier_role"],
                "frontier_status": point["frontier_status"],
                "nominal_cap": point["nominal_cap"],
                "source": point["source"],
                "effective_equivalence": point["effective_equivalence"],
                "contract_preserving": point["contract_preserving"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "num_correct": metrics["num_correct"],
                "num_examples": metrics["num_examples"],
                "avg_num_docs": metrics["avg_num_docs"],
                "avg_evidence_chars": metrics["avg_evidence_chars"],
                "avg_original_retrieval_rank": metrics["avg_original_retrieval_rank"],
                "metrics_path": metrics_path,
            }
        )

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["frontier_method_id"],
            row["underlying_method"],
            row["variant"],
            row["frontier_role"],
            row["frontier_status"],
            row["nominal_cap"],
            row["contract_preserving"],
        )
        grouped[key].append(row)

    aggregates: list[dict[str, Any]] = []
    for key, members in grouped.items():
        if len(members) != len(plan["generators"]):
            raise FrontierPlanError(
                f"Aggregate operating point is missing generators: key={key!r} "
                f"expected={len(plan['generators'])} actual={len(members)}"
            )
        (
            frontier_method_id,
            underlying_method,
            variant,
            frontier_role,
            frontier_status,
            nominal_cap,
            contract_preserving,
        ) = key
        pooled_correct = sum(int(member["num_correct"]) for member in members)
        pooled_n = sum(int(member["num_examples"]) for member in members)
        sources = sorted({str(member["source"]) for member in members})
        aggregates.append(
            {
                "frontier_method_id": frontier_method_id,
                "underlying_method": underlying_method,
                "variant": variant,
                "frontier_role": frontier_role,
                "frontier_status": frontier_status,
                "nominal_cap": nominal_cap,
                "source": sources[0] if len(sources) == 1 else "mixed",
                "contract_preserving": contract_preserving,
                "generator_count": len(members),
                "pooled_correct": pooled_correct,
                "pooled_n": pooled_n,
                "pooled_accuracy": pooled_correct / pooled_n if pooled_n else 0.0,
                "mean_macro_f1": fmean(float(member["macro_f1"]) for member in members),
                "mean_avg_num_docs": fmean(float(member["avg_num_docs"]) for member in members),
                "mean_avg_evidence_chars": fmean(
                    float(member["avg_evidence_chars"]) for member in members
                ),
            }
        )
    aggregates.sort(
        key=lambda row: (
            str(row["frontier_method_id"]),
            str(row["frontier_role"]),
            -1 if row["nominal_cap"] is None else int(row["nominal_cap"]),
        )
    )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "dataset_id": plan["dataset_id"],
        "split": plan["split"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "rows": rows,
        "aggregates": aggregates,
    }


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)


def main() -> None:
    args = parse_args()
    plan = _load_plan(args.plan)
    summary = build_summary(plan)
    atomic_write_json(args.output_json, summary)
    write_csv(args.output_csv, summary["rows"])
    print(
        f"[fm2_budget_frontier_summary] rows={len(summary['rows'])} "
        f"aggregates={len(summary['aggregates'])}"
    )


if __name__ == "__main__":
    main()

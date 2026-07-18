from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_manifest import atomic_write_json, stable_hash, utc_now

DEFAULT_METHODS = [
    "no_evidence",
    "naive_topm",
    "bge",
    "infogain_fever",
    "rag_cbwdm",
    "cbwdm_oracle",
]
ORDER = {method: index for index, method in enumerate(DEFAULT_METHODS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize existing FEVER baseline metrics without running models.")
    parser.add_argument("--run-dir")
    parser.add_argument("--metrics", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-method", action="append", default=[])
    parser.add_argument("--fairness-audit")
    return parser.parse_args()


def absolute(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def discover_metrics(run_dir: Path | None, explicit: list[str]) -> list[Path]:
    paths = [absolute(value) for value in explicit]
    if run_dir:
        paths.extend(run_dir.glob("artifacts/eval/*_metrics.json"))
        paths.extend(run_dir.glob("artifacts/*_metrics.json"))
    return sorted(set(path for path in paths if path.is_file()))


def summarize(metrics_paths: list[Path], expected: list[str]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    fingerprints: dict[str, set[str]] = {}
    for path in metrics_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        method = str(payload.get("method") or path.stem.removesuffix("_metrics"))
        payload["_path"] = str(path.resolve())
        grouped.setdefault(method, []).append(payload)
        manifest_path = path.with_suffix(".manifest.json")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            fingerprints.setdefault(method, set()).add(str(manifest.get("fingerprint")))
    rows = []
    for method in expected:
        items = grouped.get(method, [])
        if not items:
            rows.append(
                {
                    "method": method,
                    "status": "missing",
                    "deployable": method != "cbwdm_oracle",
                    "diagnostic_only": method == "cbwdm_oracle",
                    "accuracy": None,
                    "macro_f1": None,
                    "avg_num_docs": None,
                    "avg_evidence_chars": None,
                    "num_examples": None,
                    "num_seeds": 0,
                    "run_fingerprint": None,
                }
            )
            continue
        if len(fingerprints.get(method, set())) > 1:
            raise ValueError(f"Method {method} has incompatible run fingerprints")
        accuracies = [float(item["accuracy"]) for item in items]
        macro_f1s = [float(item["macro_f1"]) for item in items]
        row = {
            "method": method,
            "status": "completed",
            "deployable": all(bool(item.get("deployable", True)) for item in items),
            "diagnostic_only": any(bool(item.get("diagnostic_only")) for item in items),
            "accuracy": statistics.fmean(accuracies),
            "accuracy_std": statistics.stdev(accuracies) if len(items) > 1 else None,
            "macro_f1": statistics.fmean(macro_f1s),
            "macro_f1_std": statistics.stdev(macro_f1s) if len(items) > 1 else None,
            "avg_num_docs": statistics.fmean(float(item["avg_num_docs"]) for item in items),
            "avg_evidence_chars": statistics.fmean(
                float(item["avg_evidence_chars"]) for item in items
            ),
            "num_examples": items[0]["num_examples"],
            "num_seeds": len(items),
            "generator": items[0].get("model_name"),
            "run_fingerprint": next(iter(fingerprints.get(method, set())), None),
        }
        rows.append(row)
    deployable = [
        row
        for row in rows
        if row["status"] == "completed" and row["deployable"] and not row["diagnostic_only"]
    ]
    best = max(deployable, key=lambda row: row["accuracy"])["method"] if deployable else None
    return {"methods": rows, "deployable_best": best}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_markdown(path: Path, rows: list[dict[str, Any]], comparable: bool) -> None:
    lines = [
        "# FEVER baseline summary",
        "",
        f"Comparable: {'YES' if comparable else 'NO'}",
        "",
        "| Method | Status | Deployable | Accuracy | Macro-F1 | Avg docs | N |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def show(key: str) -> str:
            value = row.get(key)
            return "missing" if value is None else (
                f"{value:.6f}" if isinstance(value, float) else str(value)
            )
        method = row["method"] + ("†" if row.get("diagnostic_only") else "")
        lines.append(
            f"| {method} | {row['status']} | {row['deployable']} | "
            f"{show('accuracy')} | {show('macro_f1')} | {show('avg_num_docs')} | "
            f"{show('num_examples')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = absolute(args.run_dir) if args.run_dir else None
    expected = args.expected_method or DEFAULT_METHODS
    summary = summarize(discover_metrics(run_dir, args.metrics), expected)
    fairness = None
    if args.fairness_audit:
        fairness = json.loads(absolute(args.fairness_audit).read_text(encoding="utf-8"))
    comparable = fairness is None or fairness.get("status") == "comparable"
    summary.update(
        {
            "schema_version": "rag_cbwdm_baseline_summary.v1",
            "status": "completed" if comparable else "not_comparable",
            "comparable": comparable,
            "fairness_audit": str(absolute(args.fairness_audit)) if args.fairness_audit else None,
            "summary_fingerprint": stable_hash(summary["methods"]),
            "created_at": utc_now(),
        }
    )
    output = absolute(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "baseline_summary.json", summary)
    write_csv(output / "baseline_summary.csv", summary["methods"])
    write_markdown(output / "baseline_summary.md", summary["methods"], comparable)
    print(f"[baseline_summary] methods={len(summary['methods'])} comparable={comparable} output={output}")


if __name__ == "__main__":
    main()

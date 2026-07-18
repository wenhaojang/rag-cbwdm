from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_manifest import atomic_write_json, stable_hash, utc_now

CANONICAL_METHODS: dict[str, dict[str, bool]] = {
    "no_evidence": {"deployable": True, "diagnostic_only": False},
    "naive_topm": {"deployable": True, "diagnostic_only": False},
    "bge": {"deployable": True, "diagnostic_only": False},
    "infogain_fever": {"deployable": True, "diagnostic_only": False},
    "rag_cbwdm": {"deployable": True, "diagnostic_only": False},
    "cbwdm_oracle": {"deployable": False, "diagnostic_only": True},
}
DEFAULT_METHODS = list(CANONICAL_METHODS)
ORDER = {method: index for index, method in enumerate(DEFAULT_METHODS)}
SUMMARY_FILENAMES = (
    "baseline_summary.json",
    "baseline_summary.csv",
    "baseline_summary.md",
)
METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "accuracy": ("accuracy", "acc"),
    "macro_f1": ("macro_f1", "f1_macro"),
    "avg_num_docs": ("avg_num_docs", "average_num_docs"),
    "avg_evidence_chars": ("avg_evidence_chars", "average_evidence_chars"),
    "num_examples": ("num_examples", "n_examples", "example_count"),
}
NESTED_METRIC_CONTAINERS = (
    "metrics",
    "classification_metrics",
    "evaluation",
    "results",
    "aggregate",
    "summary",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize canonical FEVER baseline evaluations without running models."
    )
    parser.add_argument("--run-dir")
    parser.add_argument(
        "--metrics",
        action="append",
        default=[],
        help="Explicit evaluation metrics path (must have a completed evaluation manifest).",
    )
    parser.add_argument(
        "--evaluation-manifest",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help="Canonical method and its completed evaluation manifest.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-method", action="append", default=[])
    parser.add_argument("--fairness-audit")
    return parser.parse_args()


def absolute(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def parse_manifest_assignments(values: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected METHOD=PATH, got {value!r}")
        method, raw_path = value.split("=", 1)
        if method not in CANONICAL_METHODS:
            raise ValueError(f"Unknown canonical baseline method: {method!r}")
        if method in result:
            raise ValueError(f"Duplicate evaluation manifest for {method!r}")
        result[method] = absolute(raw_path)
    return result


def manifest_metrics_path(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    candidates: list[Any] = [
        manifest.get("metrics_path"),
        manifest.get("metrics"),
    ]
    for container_name in ("paths", "outputs", "artifacts"):
        container = manifest.get(container_name)
        if isinstance(container, dict):
            candidates.extend((container.get("metrics_path"), container.get("metrics")))
    for value in candidates:
        if not isinstance(value, str) or not value:
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = manifest_path.parent / candidate
        if candidate.is_file():
            return candidate.resolve()

    suffix = ".manifest.json"
    if manifest_path.name.endswith(suffix):
        sidecar = manifest_path.with_name(
            manifest_path.name.removesuffix(suffix) + ".json"
        )
        if sidecar.is_file():
            return sidecar.resolve()
    raise FileNotFoundError(
        f"Evaluation manifest does not identify an existing metrics file: {manifest_path}"
    )


def validate_evaluation_manifest(
    manifest_path: Path,
    *,
    assigned_method: str | None = None,
) -> tuple[str, Path, dict[str, Any]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Evaluation manifest missing: {manifest_path}")
    manifest = load_json_object(manifest_path, "evaluation manifest")
    if manifest.get("stage") != "evaluation":
        raise ValueError(
            f"Not an evaluation manifest (stage={manifest.get('stage')!r}): {manifest_path}"
        )
    if manifest.get("status") != "completed" or manifest.get("completed") is False:
        raise ValueError(f"Evaluation manifest is not completed: {manifest_path}")
    method = manifest.get("method")
    if method not in CANONICAL_METHODS:
        raise ValueError(
            f"Evaluation manifest method is not canonical ({method!r}): {manifest_path}"
        )
    if assigned_method is not None and method != assigned_method:
        raise ValueError(
            f"Evaluation manifest method mismatch: assigned={assigned_method!r} "
            f"manifest={method!r} path={manifest_path}"
        )
    return str(method), manifest_metrics_path(manifest_path, manifest), manifest


def discover_metric_artifacts(
    run_dir: Path | None,
    explicit: list[str],
    evaluation_manifests: list[str] | None = None,
) -> tuple[list[Path], list[dict[str, str]]]:
    """Locate only canonical, completed evaluation metrics through their manifests."""
    selected: dict[Path, str] = {}
    excluded: list[dict[str, str]] = []

    assigned = parse_manifest_assignments(evaluation_manifests or [])
    for method, manifest_path in assigned.items():
        manifest_method, metrics_path, _ = validate_evaluation_manifest(
            manifest_path, assigned_method=method
        )
        selected[metrics_path] = manifest_method

    if run_dir is not None and not assigned:
        eval_dir = run_dir / "artifacts" / "eval"
        for manifest_path in sorted(eval_dir.glob("*.manifest.json")):
            try:
                method, metrics_path, _ = validate_evaluation_manifest(manifest_path)
            except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                excluded.append({"path": str(manifest_path.resolve()), "reason": str(exc)})
                continue
            selected[metrics_path] = method

    for raw_path in explicit:
        metrics_path = absolute(raw_path).resolve()
        manifest_path = metrics_path.with_suffix(".manifest.json")
        try:
            method, declared_metrics_path, _ = validate_evaluation_manifest(manifest_path)
            if declared_metrics_path != metrics_path:
                raise ValueError(
                    f"Manifest metrics path {declared_metrics_path} does not match {metrics_path}"
                )
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            excluded.append({"path": str(metrics_path), "reason": str(exc)})
            continue
        selected[metrics_path] = method

    return sorted(selected, key=lambda path: (ORDER[selected[path]], str(path))), excluded


def discover_metrics(
    run_dir: Path | None,
    explicit: list[str],
    evaluation_manifests: list[str] | None = None,
) -> list[Path]:
    paths, _ = discover_metric_artifacts(run_dir, explicit, evaluation_manifests)
    return paths


def evaluation_manifest_for_metrics(
    metrics_path: Path,
) -> tuple[str, Path, dict[str, Any]]:
    manifest_path = metrics_path.with_suffix(".manifest.json")
    method, declared_metrics_path, manifest = validate_evaluation_manifest(manifest_path)
    if declared_metrics_path != metrics_path.resolve():
        raise ValueError(
            f"Evaluation manifest points to {declared_metrics_path}, not {metrics_path.resolve()}"
        )
    return method, manifest_path.resolve(), manifest


def nested_values(payload: Any, aliases: tuple[str, ...], prefix: str = "") -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else key
            if key in aliases:
                found.append((path, value))
            if isinstance(value, (dict, list)):
                found.extend(nested_values(value, aliases, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            if isinstance(value, (dict, list)):
                found.extend(nested_values(value, aliases, f"{prefix}[{index}]"))
    return found


def metric_value(
    payload: dict[str, Any], canonical_name: str
) -> tuple[Any | None, str | None, str | None]:
    aliases = METRIC_ALIASES[canonical_name]
    for alias in aliases:
        if alias in payload:
            source = alias
            mapping = None if source == canonical_name else f"{canonical_name}<-{source}"
            return payload[alias], source, mapping
    for container_name in NESTED_METRIC_CONTAINERS:
        container = payload.get(container_name)
        if not isinstance(container, dict):
            continue
        for alias in aliases:
            if alias in container:
                source = f"{container_name}.{alias}"
                return container[alias], source, f"{canonical_name}<-{source}"
    matches = nested_values(payload, aliases)
    if not matches:
        return None, None, None
    values = {json.dumps(value, sort_keys=True, default=str) for _, value in matches}
    if len(values) > 1:
        locations = ", ".join(path for path, _ in matches)
        return None, None, f"ambiguous {canonical_name} at {locations}"
    source, value = matches[0]
    return value, source, f"{canonical_name}<-{source}"


def canonical_number(
    payload: dict[str, Any],
    canonical_name: str,
    *,
    integer: bool = False,
) -> tuple[int | float | None, str | None, str | None]:
    value, source, mapping = metric_value(payload, canonical_name)
    if source is None:
        return None, None, mapping or f"{canonical_name} is missing"
    if isinstance(value, bool):
        return None, source, f"{canonical_name} at {source} is boolean"
    try:
        number = int(value) if integer else float(value)
    except (TypeError, ValueError):
        return None, source, f"{canonical_name} at {source} is not numeric: {value!r}"
    if not integer and not math.isfinite(float(number)):
        return None, source, f"{canonical_name} at {source} is not finite: {value!r}"
    return number, source, mapping


def normalize_metrics(
    path: Path,
    method: str,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    payload = load_json_object(path, "evaluation metrics")
    payload_method = payload.get("method")
    if payload_method not in (None, method):
        raise ValueError(
            f"Metrics method mismatch: manifest={method!r} metrics={payload_method!r} path={path}"
        )
    normalized: dict[str, Any] = {
        "method": method,
        "_path": str(path.resolve()),
        "_manifest_path": str(manifest_path.resolve()),
        "_fingerprint": manifest.get("fingerprint"),
        "_mappings": [],
        "_reasons": [],
    }
    for field in METRIC_ALIASES:
        value, _, note = canonical_number(
            payload, field, integer=field == "num_examples"
        )
        normalized[field] = value
        if value is None:
            normalized["_reasons"].append(note or f"{field} is missing")
        elif note:
            normalized["_mappings"].append(note)
    normalized["model_name"] = payload.get("model_name")
    normalized["deployable"] = payload.get("deployable")
    normalized["diagnostic_only"] = payload.get("diagnostic_only")
    return normalized


def missing_row(method: str, reason: str) -> dict[str, Any]:
    metadata = CANONICAL_METHODS[method]
    return {
        "method": method,
        "status": "missing",
        "reason": reason,
        "missing_fields": list(METRIC_ALIASES),
        "schema_mappings": [],
        "metrics_path": None,
        "manifest_path": None,
        "deployable": metadata["deployable"],
        "diagnostic_only": metadata["diagnostic_only"],
        "accuracy": None,
        "accuracy_std": None,
        "macro_f1": None,
        "macro_f1_std": None,
        "avg_num_docs": None,
        "avg_evidence_chars": None,
        "num_examples": None,
        "num_seeds": 0,
        "generator": None,
        "run_fingerprint": None,
    }


def aggregate_field(items: list[dict[str, Any]], field: str) -> float | None:
    values = [item[field] for item in items]
    if any(value is None for value in values):
        return None
    return statistics.fmean(float(value) for value in values)


def summarize(metrics_paths: list[Path], expected: list[str]) -> dict[str, Any]:
    unknown = [method for method in expected if method not in CANONICAL_METHODS]
    if unknown:
        raise ValueError(f"Expected methods are not canonical: {unknown}")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in metrics_paths:
        method, manifest_path, manifest = evaluation_manifest_for_metrics(path)
        grouped.setdefault(method, []).append(
            normalize_metrics(path, method, manifest_path, manifest)
        )

    rows: list[dict[str, Any]] = []
    for method in expected:
        items = grouped.get(method, [])
        if not items:
            rows.append(
                missing_row(method, "canonical completed evaluation artifact not found")
            )
            continue
        fingerprints = {
            str(item["_fingerprint"])
            for item in items
            if item.get("_fingerprint") is not None
        }
        if len(fingerprints) > 1:
            raise ValueError(f"Method {method} has incompatible run fingerprints")

        missing_fields = [
            field for field in METRIC_ALIASES if any(item[field] is None for item in items)
        ]
        reasons = [
            f"{item['_path']}: {reason}"
            for item in items
            for reason in item["_reasons"]
        ]
        metadata = CANONICAL_METHODS[method]
        accuracy = aggregate_field(items, "accuracy")
        macro_f1 = aggregate_field(items, "macro_f1")
        row = {
            "method": method,
            "status": "missing_metrics" if missing_fields else "completed",
            "reason": "; ".join(reasons) if reasons else None,
            "missing_fields": missing_fields,
            "schema_mappings": sorted(
                {mapping for item in items for mapping in item["_mappings"]}
            ),
            "metrics_path": (
                items[0]["_path"] if len(items) == 1 else [item["_path"] for item in items]
            ),
            "manifest_path": (
                items[0]["_manifest_path"]
                if len(items) == 1
                else [item["_manifest_path"] for item in items]
            ),
            "deployable": metadata["deployable"],
            "diagnostic_only": metadata["diagnostic_only"],
            "accuracy": accuracy,
            "accuracy_std": (
                statistics.stdev(float(item["accuracy"]) for item in items)
                if len(items) > 1 and accuracy is not None
                else None
            ),
            "macro_f1": macro_f1,
            "macro_f1_std": (
                statistics.stdev(float(item["macro_f1"]) for item in items)
                if len(items) > 1 and macro_f1 is not None
                else None
            ),
            "avg_num_docs": aggregate_field(items, "avg_num_docs"),
            "avg_evidence_chars": aggregate_field(items, "avg_evidence_chars"),
            "num_examples": (
                items[0]["num_examples"]
                if all(item["num_examples"] == items[0]["num_examples"] for item in items)
                else None
            ),
            "num_seeds": len(items),
            "generator": items[0].get("model_name"),
            "run_fingerprint": next(iter(fingerprints), None),
        }
        rows.append(row)
    deployable = [
        row
        for row in rows
        if row["status"] == "completed"
        and row["deployable"]
        and not row["diagnostic_only"]
        and row["accuracy"] is not None
    ]
    best = max(deployable, key=lambda row: row["accuracy"])["method"] if deployable else None
    return {"methods": rows, "deployable_best": best}


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: csv_value(value) for key, value in row.items()} for row in rows
        )
    temporary.replace(path)


def write_markdown(path: Path, rows: list[dict[str, Any]], comparable: bool) -> None:
    lines = [
        "# FEVER baseline summary",
        "",
        f"Comparable: {'YES' if comparable else 'NO'}",
        "",
        "| Method | Status | Deployable | Accuracy | Macro-F1 | Avg docs | N | Reason |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:

        def show(key: str) -> str:
            value = row.get(key)
            return "missing" if value is None else (
                f"{value:.6f}" if isinstance(value, float) else str(value)
            )

        method = row["method"] + ("†" if row.get("diagnostic_only") else "")
        reason = str(row.get("reason") or "").replace("|", r"\|")
        lines.append(
            f"| {method} | {row['status']} | {row['deployable']} | "
            f"{show('accuracy')} | {show('macro_f1')} | {show('avg_num_docs')} | "
            f"{show('num_examples')} | {reason} |"
        )
    lines.extend(("", "† Diagnostic only; excluded from deployable best.", ""))
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def load_comparable_fairness(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    fairness = load_json_object(path, "fairness audit")
    overall = fairness.get("status", fairness.get("overall"))
    method_failures = []
    methods = fairness.get("methods")
    if isinstance(methods, dict):
        for method in DEFAULT_METHODS:
            details = methods.get(method)
            status = details.get("status") if isinstance(details, dict) else None
            if status != "comparable":
                method_failures.append(f"{method}={status!r}")
    if overall != "comparable" or method_failures:
        details = f"; methods: {', '.join(method_failures)}" if method_failures else ""
        raise ValueError(
            f"Refusing formal baseline summary: fairness audit is not comparable "
            f"(overall={overall!r}){details}"
        )
    return fairness


def fixed_output_dir(run_dir: Path | None, requested: str) -> Path:
    output = absolute(requested).resolve()
    if run_dir is not None:
        expected = (run_dir / "artifacts" / "baselines" / "summary").resolve()
        if output != expected:
            raise ValueError(
                f"Baseline summary output must be {expected}, got {output}"
            )
    return output


def main() -> None:
    args = parse_args()
    run_dir = absolute(args.run_dir).resolve() if args.run_dir else None
    expected = args.expected_method or DEFAULT_METHODS
    fairness_path = absolute(args.fairness_audit).resolve() if args.fairness_audit else None
    load_comparable_fairness(fairness_path)
    output = fixed_output_dir(run_dir, args.output_dir)
    metrics_paths, excluded = discover_metric_artifacts(
        run_dir, args.metrics, args.evaluation_manifest
    )
    summary = summarize(metrics_paths, expected)
    summary.update(
        {
            "schema_version": "rag_cbwdm_baseline_summary.v2",
            "status": "completed",
            "comparable": True,
            "fairness_audit": str(fairness_path) if fairness_path else None,
            "excluded_artifacts": excluded,
            "summary_fingerprint": stable_hash(summary["methods"]),
            "created_at": utc_now(),
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / SUMMARY_FILENAMES[0], summary)
    write_csv(output / SUMMARY_FILENAMES[1], summary["methods"])
    write_markdown(output / SUMMARY_FILENAMES[2], summary["methods"], True)
    print(
        f"[baseline_summary] methods={len(summary['methods'])} comparable=True output={output}"
    )


if __name__ == "__main__":
    main()

"""Deterministic validation-only selection of FEVER method parameters."""

from __future__ import annotations

import csv
import io
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

from src.formal_provenance import atomic_write_text, sha256_path
from src.formal_splits import validate_split_manifest
from src.io_utils import load_yaml, read_jsonl
from src.run_manifest import atomic_write_json, git_state, sha256_file, stable_hash, utc_now

CALIBRATION_SCHEMA_VERSION = "rag_cbwdm_fever_calibration.v1"
CALIBRATED_METHODS = ("infogain_fever", "rag_cbwdm")
OUTPUT_NAMES = (
    "calibration_results.json",
    "calibration_results.csv",
    "calibration_report.md",
    "frozen_parameters.yaml",
    "calibration.manifest.json",
)


def _product(mapping: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(mapping)
    if not keys:
        return [{}]
    return [
        dict(zip(keys, values))
        for values in itertools.product(*(mapping[key] for key in keys))
    ]


def enumerate_parameter_grid(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Return stable grids containing only parameters consumed by current code."""
    calibration = config.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("Config must define calibration grids")
    info = calibration.get("infogain")
    cbwdm = calibration.get("cbwdm")
    if not isinstance(info, dict) or not isinstance(cbwdm, dict):
        raise ValueError("Config must define calibration.infogain and calibration.cbwdm")

    quantiles = info.get("teacher_quantiles")
    if not isinstance(quantiles, list) or not quantiles:
        raise ValueError("calibration.infogain.teacher_quantiles must be a non-empty list")
    teacher_options = []
    for pair in quantiles:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Each teacher_quantiles entry must be [negative, positive]")
        negative, positive = map(float, pair)
        if not 0 <= negative < positive <= 1:
            raise ValueError("InfoGain teacher quantiles must satisfy 0 <= neg < pos <= 1")
        teacher_options.append(
            {"negative_quantile": negative, "positive_quantile": positive}
        )
    info_rest = {
        "filter_threshold": list(info.get("filter_thresholds", [])),
        "min_docs": list(info.get("min_docs", [])),
        "top_m": list(info.get("top_m", [])),
        "beta": list(info.get("beta", [])),
    }
    if any(not values for values in info_rest.values()):
        raise ValueError(
            "InfoGain grid requires filter_thresholds, min_docs, top_m, and beta"
        )
    info_grid = [
        {**teacher, **rest}
        for teacher in teacher_options
        for rest in _product(info_rest)
    ]

    cbwdm_mapping = {
        "stop_threshold": list(cbwdm.get("stop_thresholds", [])),
        "score_threshold": list(cbwdm.get("score_thresholds", [])),
        "min_docs": list(cbwdm.get("min_docs", [])),
        "top_m": list(cbwdm.get("top_m", [])),
        "beta": list(cbwdm.get("beta", [])),
        "gamma": list(cbwdm.get("gamma", [])),
        "b_plus": list(cbwdm.get("b_plus", [])),
        "b_minus": list(cbwdm.get("b_minus", [])),
    }
    if any(not values for values in cbwdm_mapping.values()):
        raise ValueError(
            "CBWDM grid requires stop_thresholds, score_thresholds, min_docs, top_m, "
            "beta, gamma, b_plus, and b_minus"
        )
    cbwdm_grid = _product(cbwdm_mapping)
    for parameters in cbwdm_grid:
        if float(parameters["b_plus"]) <= float(parameters["b_minus"]) or float(
            parameters["b_minus"]
        ) < 0:
            raise ValueError("CBWDM grid requires b_plus > b_minus >= 0")
        if int(parameters["min_docs"]) > int(parameters["top_m"]):
            raise ValueError("CBWDM grid requires min_docs <= top_m")
    return {"infogain_fever": info_grid, "rag_cbwdm": cbwdm_grid}


def load_candidate_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    source_text = str(source).casefold()
    if "held_out_test" in source_text or "held-out-test" in source_text:
        raise ValueError("Calibration input path must not reference held_out_test")
    if source.suffix.casefold() == ".jsonl":
        return list(read_jsonl(source))
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("candidates", payload.get("records"))
    else:
        records = None
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError("Calibration metrics must be a JSON list/JSONL or contain candidates")
    return records


def _canonical_metric(metrics: dict[str, Any], name: str) -> tuple[float | None, str | None]:
    aliases = {
        "macro_f1": ("macro_f1", "f1_macro"),
        "accuracy": ("accuracy",),
        "avg_num_docs": ("avg_num_docs",),
        "avg_evidence_chars": ("avg_evidence_chars",),
    }
    for key in aliases[name]:
        value = metrics.get(key)
        if value is not None:
            try:
                return float(value), None
            except (TypeError, ValueError):
                return None, f"{key} is not numeric"
    nested = metrics.get("metrics")
    if isinstance(nested, dict):
        return _canonical_metric(nested, name)
    return None, f"missing {name}"


def _normalize_record(
    row: dict[str, Any],
    *,
    validation_sha: str,
    objective: str,
    allowed_grid: dict[str, set[str]],
) -> dict[str, Any]:
    method = str(row.get("method", ""))
    if method not in CALIBRATED_METHODS:
        raise ValueError(f"Calibration candidate has unsupported method {method!r}")
    split = str(row.get("split", row.get("data_role", "")))
    if split != "validation":
        raise ValueError(
            f"Calibration candidate for {method} must use split=validation, got {split!r}"
        )
    candidate_sha = row.get("split_sha256", row.get("validation_sha256"))
    if candidate_sha != validation_sha:
        raise ValueError(
            f"Calibration candidate for {method} does not reference frozen validation SHA"
        )
    forbidden = json.dumps(row, ensure_ascii=False).casefold()
    if "held_out_test" in forbidden or "held-out-test" in forbidden:
        raise ValueError("Calibration candidate contains a held_out_test reference")
    parameters = row.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"Calibration candidate for {method} lacks parameters")
    parameter_key = stable_hash(parameters)
    if parameter_key not in allowed_grid[method]:
        raise ValueError(
            f"Calibration candidate parameters for {method} are outside the declared grid: "
            f"{parameters}"
        )
    metrics = row.get("metrics", row)
    if not isinstance(metrics, dict):
        metrics = {}
    normalized: dict[str, Any] = {
        "method": method,
        "split": "validation",
        "split_sha256": validation_sha,
        "parameters": parameters,
        "parameter_fingerprint": parameter_key,
        "candidate_fingerprint": row.get("candidate_fingerprint"),
        "training_fingerprint": row.get("training_fingerprint"),
        "selection_fingerprint": row.get("selection_fingerprint"),
        "source_status": row.get("status", "completed"),
    }
    reasons: list[str] = []
    for name in ("macro_f1", "accuracy", "avg_num_docs", "avg_evidence_chars"):
        value, reason = _canonical_metric(metrics, name)
        normalized[name] = value
        if reason:
            reasons.append(reason)
    if row.get("status", "completed") != "completed":
        reasons.append(str(row.get("reason") or f"candidate status={row.get('status')}"))
    normalized["eligible"] = normalized[objective] is not None and row.get(
        "status", "completed"
    ) == "completed"
    normalized["reason"] = "; ".join(reasons) if reasons else None
    return normalized


def _selection_key(row: dict[str, Any], objective: str) -> tuple[Any, ...]:
    huge = float("inf")
    return (
        -float(row[objective]),
        row["avg_num_docs"] if row["avg_num_docs"] is not None else huge,
        row["avg_evidence_chars"] if row["avg_evidence_chars"] is not None else huge,
        json.dumps(row["parameters"], sort_keys=True, separators=(",", ":")),
    )


def calibrate(
    config: dict[str, Any],
    split_manifest: dict[str, Any],
    records: Iterable[dict[str, Any]],
    *,
    objective: str = "macro_f1",
) -> dict[str, Any]:
    if objective not in {"macro_f1", "accuracy"}:
        raise ValueError("Calibration objective must be macro_f1 or accuracy")
    validation = split_manifest.get("splits", {}).get("validation", {})
    validation_sha = validation.get("sha256")
    if not validation_sha:
        raise ValueError("Split manifest lacks validation SHA")
    grid = enumerate_parameter_grid(config)
    allowed = {
        method: {stable_hash(parameters) for parameters in candidates}
        for method, candidates in grid.items()
    }
    normalized = [
        _normalize_record(
            row,
            validation_sha=validation_sha,
            objective=objective,
            allowed_grid=allowed,
        )
        for row in records
    ]
    selected: dict[str, Any] = {}
    for method in CALIBRATED_METHODS:
        candidates = [row for row in normalized if row["method"] == method and row["eligible"]]
        if not candidates:
            selected[method] = {
                "status": "missing",
                "parameters": None,
                "reason": f"No completed {method} candidate has {objective}",
            }
            continue
        winner = min(candidates, key=lambda row: _selection_key(row, objective))
        selected[method] = {
            "status": "selected",
            "parameters": winner["parameters"],
            "metrics": {
                key: winner[key]
                for key in ("macro_f1", "accuracy", "avg_num_docs", "avg_evidence_chars")
            },
            "reason": (
                f"highest validation {objective}; ties use lower avg_num_docs, then "
                "lower avg_evidence_chars, then canonical parameter JSON"
            ),
            "parameter_fingerprint": winner["parameter_fingerprint"],
            "candidate_fingerprint": winner.get("candidate_fingerprint"),
            "training_fingerprint": winner.get("training_fingerprint"),
            "selection_fingerprint": winner.get("selection_fingerprint"),
        }
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "status": (
            "completed"
            if all(item["status"] == "selected" for item in selected.values())
            else "blocked"
        ),
        "data_role": "validation",
        "validation_sha256": validation_sha,
        "held_out_test_used": False,
        "objective": objective,
        "selection_rule": {
            "primary": f"higher_{objective}",
            "secondary": ["lower_avg_num_docs", "lower_avg_evidence_chars"],
            "stable_final_tie_break": "canonical_parameter_json",
        },
        "parameter_audit": {
            "infogain_fever": {
                "negative_quantile": "12a --negative-quantile",
                "positive_quantile": "12a --positive-quantile",
                "filter_threshold": "12c --filter-threshold",
                "min_docs": "12c --min-docs",
                "top_m": "12c --top-m",
                "beta": "12b --beta",
            },
            "rag_cbwdm": {
                "stop_threshold": "04 --stop-threshold",
                "score_threshold": "11 --score-threshold",
                "min_docs": "11 --min-docs",
                "top_m": "04/11 --top-m",
                "beta": "10 --beta",
                "gamma": "10 --gamma",
                "b_plus": "10 --b-plus",
                "b_minus": "10 --b-minus",
                "gain_normalization": "not implemented; intentionally absent",
            },
        },
        "declared_grid": grid,
        "candidates": normalized,
        "selected": selected,
    }


def _csv_text(rows: list[dict[str, Any]]) -> str:
    fields = [
        "method",
        "eligible",
        "macro_f1",
        "accuracy",
        "avg_num_docs",
        "avg_evidence_chars",
        "parameters",
        "reason",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                **{key: row.get(key) for key in fields},
                "parameters": json.dumps(row["parameters"], sort_keys=True),
            }
        )
    return output.getvalue()


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# FEVER Validation Calibration",
        "",
        f"- Status: `{result['status']}`",
        f"- Data role: `validation`",
        f"- Held-out test used: `{str(result['held_out_test_used']).lower()}`",
        f"- Objective: `{result['objective']}`",
        "",
        "| Method | Status | Parameters | Reason |",
        "|---|---|---|---|",
    ]
    for method in CALIBRATED_METHODS:
        item = result["selected"][method]
        parameters = (
            json.dumps(item["parameters"], sort_keys=True) if item["parameters"] else "null"
        )
        lines.append(
            f"| {method} | {item['status']} | `{parameters}` | {item['reason']} |"
        )
    return "\n".join(lines) + "\n"


def publish_calibration(
    config_path: str | Path,
    split_manifest_path: str | Path,
    metrics_path: str | Path,
    output_dir: str | Path,
    *,
    objective: str = "macro_f1",
    artifacts: dict[str, str | Path] | None = None,
    resume: bool = False,
    overwrite: bool = False,
    project_root: str | Path | None = None,
) -> tuple[dict[str, Any], bool]:
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "calibration.manifest.json"
    config_file = Path(config_path).resolve()
    split_file = Path(split_manifest_path).resolve()
    metrics_file = Path(metrics_path).resolve()
    split_manifest = validate_split_manifest(split_file)
    artifact_hashes = {
        name: {"path": str(Path(path).resolve()), "sha256": sha256_path(path)}
        for name, path in sorted((artifacts or {}).items())
    }
    contract = {
        "config_path": str(config_file),
        "config_sha256": sha256_file(config_file),
        "split_manifest_path": str(split_file),
        "split_manifest_sha256": sha256_file(split_file),
        "split_manifest_fingerprint": split_manifest["fingerprint"],
        "validation_sha256": split_manifest["splits"]["validation"]["sha256"],
        "metrics_path": str(metrics_file),
        "metrics_sha256": sha256_file(metrics_file),
        "objective": objective,
        "artifacts": artifact_hashes,
    }
    fingerprint = stable_hash(contract)
    if resume and manifest_path.is_file() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint:
            raise ValueError("Cannot resume calibration: inputs or parameters changed")
        for name in OUTPUT_NAMES:
            path = directory / name
            if not path.is_file():
                raise ValueError(f"Cannot resume calibration: missing {path}")
            if name != "calibration.manifest.json":
                expected_sha = existing.get("output_sha256", {}).get(name)
                if expected_sha != sha256_file(path):
                    raise ValueError(
                        f"Cannot resume calibration: output SHA changed for {name}"
                    )
        if existing.get("status") != "completed":
            raise ValueError("Cannot resume an incomplete calibration")
        return existing, True
    existing = [directory / name for name in OUTPUT_NAMES if (directory / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Calibration output exists ({existing[0]}). Use --resume or --overwrite."
        )

    config = load_yaml(config_file)
    result = calibrate(
        config,
        split_manifest,
        load_candidate_records(metrics_file),
        objective=objective,
    )
    result.update(
        {
            "contract": contract,
            "fingerprint": fingerprint,
            "generator_contract": config.get("generator"),
            "git": git_state(project_root or Path(__file__).resolve().parents[2]),
            "created_at": utc_now(),
        }
    )
    atomic_write_json(directory / "calibration_results.json", result)
    atomic_write_text(directory / "calibration_results.csv", _csv_text(result["candidates"]))
    atomic_write_text(directory / "calibration_report.md", _markdown(result))
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to write frozen_parameters.yaml") from exc
    frozen = {
        method: item["parameters"] if item["status"] == "selected" else None
        for method, item in result["selected"].items()
    }
    atomic_write_text(
        directory / "frozen_parameters.yaml",
        yaml.safe_dump(frozen, sort_keys=True, allow_unicode=True),
    )
    output_hashes = {
        name: sha256_file(directory / name)
        for name in OUTPUT_NAMES
        if name != "calibration.manifest.json"
    }
    manifest = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "status": result["status"],
        "fingerprint": fingerprint,
        "contract": contract,
        "split_manifest_sha256": contract["split_manifest_sha256"],
        "validation_sha256": contract["validation_sha256"],
        "held_out_test_used": False,
        "selected": result["selected"],
        "output_sha256": output_hashes,
        "git": result["git"],
        "created_at": result["created_at"],
    }
    atomic_write_json(manifest_path, manifest)
    return manifest, False

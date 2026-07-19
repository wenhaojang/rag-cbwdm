"""Plan and execute the validation-only FEVER calibration grid sequentially."""

from __future__ import annotations

import csv
import io
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from src.calibration.fever import CALIBRATED_METHODS, enumerate_parameter_grid
from src.formal_provenance import atomic_write_text
from src.formal_splits import validate_split_manifest
from src.io_utils import load_yaml, read_jsonl
from src.prompts import fever_prompt_hash
from src.run_manifest import (
    atomic_write_json,
    git_state,
    sha256_file,
    stable_hash,
    utc_now,
)

GRID_SCHEMA_VERSION = "rag_cbwdm_fever_calibration_grid.v1"
CANDIDATE_SCHEMA_VERSION = "rag_cbwdm_calibration_candidate.v1"

PARAMETER_DEPENDENCIES = {
    "infogain_fever": {
        "negative_quantile": ["teacher", "training"],
        "positive_quantile": ["teacher", "training"],
        "beta": ["training"],
        "filter_threshold": ["selection"],
        "min_docs": ["selection"],
        "top_m": ["selection"],
    },
    "rag_cbwdm": {
        "stop_threshold": ["teacher", "training"],
        "top_m": ["teacher", "training", "selection"],
        "beta": ["training"],
        "gamma": ["training"],
        "b_plus": ["training"],
        "b_minus": ["training"],
        "score_threshold": ["selection"],
        "min_docs": ["selection"],
    },
}

TEACHER_KEYS = {
    "infogain_fever": ("negative_quantile", "positive_quantile"),
    "rag_cbwdm": ("stop_threshold", "top_m"),
}
TRAINING_KEYS = {
    "infogain_fever": ("negative_quantile", "positive_quantile", "beta"),
    "rag_cbwdm": (
        "stop_threshold",
        "top_m",
        "beta",
        "gamma",
        "b_plus",
        "b_minus",
    ),
}
SELECTION_KEYS = {
    "infogain_fever": ("filter_threshold", "min_docs", "top_m"),
    "rag_cbwdm": ("score_threshold", "min_docs", "top_m"),
}


def canonical_parameters(parameters: dict[str, Any]) -> str:
    return json.dumps(
        parameters, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _subset(parameters: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    return {key: parameters[key] for key in keys}


def _artifact(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Calibration grid input does not exist: {resolved}")
    text = str(resolved).casefold()
    if "held_out_test" in text or "held-out-test" in text:
        raise ValueError(f"Calibration grid input references held_out_test: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _role_artifact(path: str | Path, expected_role: str) -> dict[str, Any]:
    identity = _artifact(path)
    roles = {str(row.get("split")) for row in read_jsonl(identity["path"])}
    if roles != {expected_role}:
        raise ValueError(
            f"Calibration grid input must contain only split={expected_role}, "
            f"got {sorted(roles)} in {identity['path']}"
        )
    return identity


def build_grid_plan(
    *,
    config_path: str | Path,
    split_manifest_path: str | Path,
    train_retrieval_path: str | Path,
    validation_retrieval_path: str | Path,
    train_posteriors_path: str | Path,
    validation_posteriors_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
    generator_model: str | None = None,
    selector_model: str | None = None,
    infogain_model: str | None = None,
) -> dict[str, Any]:
    """Expand the sole YAML grid into reusable teacher/training/selection nodes."""
    config_file = Path(config_path).resolve()
    split_file = Path(split_manifest_path).resolve()
    config = load_yaml(config_file)
    split = validate_split_manifest(split_file)
    inputs = {
        "config": _artifact(config_file),
        "split_manifest": _artifact(split_file),
        "train_retrieval": _role_artifact(
            train_retrieval_path, "train_core"
        ),
        "validation_retrieval": _role_artifact(
            validation_retrieval_path, "validation"
        ),
        "train_posteriors": _role_artifact(
            train_posteriors_path, "train_core"
        ),
        "validation_posteriors": _role_artifact(
            validation_posteriors_path, "validation"
        ),
    }
    validation_sha = split["splits"]["validation"]["sha256"]
    train_sha = split["splits"]["train_core"]["sha256"]
    git = git_state(project_root)
    base_contract = {
        "config_sha256": inputs["config"]["sha256"],
        "split_manifest_sha256": inputs["split_manifest"]["sha256"],
        "train_core_sha256": train_sha,
        "validation_sha256": validation_sha,
        "train_retrieval_sha256": inputs["train_retrieval"]["sha256"],
        "validation_retrieval_sha256": inputs["validation_retrieval"]["sha256"],
        "train_posteriors_sha256": inputs["train_posteriors"]["sha256"],
        "validation_posteriors_sha256": inputs["validation_posteriors"]["sha256"],
        "generator_contract": {
            **config.get("generator", {}),
            "model_name": generator_model
            or config.get("generator", {}).get("model_name"),
        },
        "models": {
            "infogain": infogain_model
            or config.get("baselines", {})
            .get("infogain_fever", {})
            .get("model_name"),
            "rag_cbwdm": selector_model
            or config.get("selector", {}).get("model_name"),
        },
        "prompt_and_verbalizers": {
            "labels": config.get("task", {}).get("labels"),
            "verbalizers": config.get("task", {}).get("verbalizers"),
        },
        "retrieval_top_n": config.get("retrieval", {}).get("top_n"),
        "git_head": git.get("commit"),
    }
    plan_fingerprint = stable_hash(base_contract)
    root = Path(output_dir).resolve()
    grid = enumerate_parameter_grid(config)
    methods: dict[str, Any] = {}
    all_selection_fingerprints: set[str] = set()
    for method in CALIBRATED_METHODS:
        training_nodes: dict[str, dict[str, Any]] = {}
        teacher_fingerprints: set[str] = set()
        for parameters in grid[method]:
            teacher_parameters = _subset(parameters, TEACHER_KEYS[method])
            training_parameters = _subset(parameters, TRAINING_KEYS[method])
            selection_parameters = _subset(parameters, SELECTION_KEYS[method])
            teacher_fingerprint = stable_hash(
                {
                    "method": method,
                    "stage": "teacher",
                    "parameters": teacher_parameters,
                    "plan": plan_fingerprint,
                }
            )
            training_fingerprint = stable_hash(
                {
                    "method": method,
                    "stage": "training",
                    "teacher_fingerprint": teacher_fingerprint,
                    "parameters": training_parameters,
                    "plan": plan_fingerprint,
                }
            )
            selection_fingerprint = stable_hash(
                {
                    "method": method,
                    "stage": "selection",
                    "training_fingerprint": training_fingerprint,
                    "parameters": selection_parameters,
                    "validation_retrieval_sha256": inputs["validation_retrieval"][
                        "sha256"
                    ],
                    "validation_posteriors_sha256": inputs[
                        "validation_posteriors"
                    ]["sha256"],
                }
            )
            candidate_fingerprint = stable_hash(
                {
                    "method": method,
                    "parameters": parameters,
                    "training_fingerprint": training_fingerprint,
                    "selection_fingerprint": selection_fingerprint,
                    "generator_contract": base_contract["generator_contract"],
                    "validation_sha256": validation_sha,
                    "git_head": git.get("commit"),
                }
            )
            teacher_fingerprints.add(teacher_fingerprint)
            all_selection_fingerprints.add(selection_fingerprint)
            training_dir = root / method / training_fingerprint
            node = training_nodes.setdefault(
                training_fingerprint,
                {
                    "method": method,
                    "teacher_fingerprint": teacher_fingerprint,
                    "training_fingerprint": training_fingerprint,
                    "teacher_parameters": teacher_parameters,
                    "training_parameters": training_parameters,
                    "canonical_training_parameters": canonical_parameters(
                        training_parameters
                    ),
                    "directory": str(training_dir),
                    "selections": [],
                },
            )
            node["selections"].append(
                {
                    "method": method,
                    "candidate_fingerprint": candidate_fingerprint,
                    "training_fingerprint": training_fingerprint,
                    "teacher_fingerprint": teacher_fingerprint,
                    "selection_fingerprint": selection_fingerprint,
                    "parameters": parameters,
                    "canonical_parameter_json": canonical_parameters(parameters),
                    "selection_parameters": selection_parameters,
                    "selection_path": str(
                        training_dir
                        / "selections"
                        / f"{selection_fingerprint}.jsonl"
                    ),
                    "evaluation_dir": str(
                        training_dir / "evaluations" / selection_fingerprint
                    ),
                }
            )
        for node in training_nodes.values():
            node["selections"].sort(key=lambda item: item["candidate_fingerprint"])
        methods[method] = {
            "teacher_candidate_count": len(teacher_fingerprints),
            "training_candidate_count": len(training_nodes),
            "selection_candidate_count": sum(
                len(node["selections"]) for node in training_nodes.values()
            ),
            "training_candidates": sorted(
                training_nodes.values(), key=lambda item: item["training_fingerprint"]
            ),
        }
    return {
        "schema_version": GRID_SCHEMA_VERSION,
        "status": "planned",
        "fingerprint": plan_fingerprint,
        "base_contract": base_contract,
        "inputs": inputs,
        "split_manifest_sha256": inputs["split_manifest"]["sha256"],
        "train_core_sha256": train_sha,
        "validation_sha256": validation_sha,
        "generator_contract": base_contract["generator_contract"],
        "parameter_dependencies": PARAMETER_DEPENDENCIES,
        "methods": methods,
        "totals": {
            "teacher_candidates": sum(
                value["teacher_candidate_count"] for value in methods.values()
            ),
            "training_candidates": sum(
                value["training_candidate_count"] for value in methods.values()
            ),
            "selection_candidates": len(all_selection_fingerprints),
        },
        "output_dir": str(root),
        "git": git,
    }


def iter_training_nodes(
    plan: dict[str, Any], methods: set[str] | None = None
) -> Iterable[dict[str, Any]]:
    allowed = methods or set(CALIBRATED_METHODS)
    for method in CALIBRATED_METHODS:
        if method not in allowed:
            continue
        yield from plan["methods"][method]["training_candidates"]


def selected_plan_nodes(
    plan: dict[str, Any],
    *,
    methods: set[str] | None,
    candidate_limit: int | None,
    candidate_fingerprint: str | None,
    max_training_candidates: int | None,
) -> list[dict[str, Any]]:
    nodes = list(iter_training_nodes(plan, methods))
    if candidate_fingerprint:
        nodes = [
            {
                **node,
                "selections": [
                    item
                    for item in node["selections"]
                    if item["candidate_fingerprint"] == candidate_fingerprint
                ],
            }
            for node in nodes
            if any(
                item["candidate_fingerprint"] == candidate_fingerprint
                for item in node["selections"]
            )
        ]
        if not nodes:
            raise ValueError(
                f"Unknown calibration candidate fingerprint: {candidate_fingerprint}"
            )
    if max_training_candidates is not None:
        if max_training_candidates < 0:
            raise ValueError("--max-training-candidates must be non-negative")
        nodes = nodes[:max_training_candidates]
    if candidate_limit is not None:
        if candidate_limit < 0:
            raise ValueError("--candidate-limit must be non-negative")
        remaining = candidate_limit
        limited = []
        for node in nodes:
            if remaining <= 0:
                break
            selections = node["selections"][:remaining]
            if selections:
                limited.append({**node, "selections": selections})
                remaining -= len(selections)
        nodes = limited
    return nodes


def dry_run_text(plan: dict[str, Any], nodes: list[dict[str, Any]]) -> str:
    lines = [
        "[calibration_grid][dry-run] no models will be loaded",
        (
            "[calibration_grid][dry-run] "
            f"training_candidates={len(nodes)} "
            f"selection_candidates={sum(len(node['selections']) for node in nodes)}"
        ),
    ]
    seen_teachers: set[str] = set()
    for node in nodes:
        teacher = node["teacher_fingerprint"]
        lines.append(
            "[calibration_grid][dry-run] "
            f"method={node['method']} training={node['training_fingerprint']} "
            f"teacher={teacher} teacher_reuse={teacher in seen_teachers} "
            f"params={node['canonical_training_parameters']} "
            f"output={node['directory']}"
        )
        seen_teachers.add(teacher)
        for selection in node["selections"]:
            lines.append(
                "[calibration_grid][dry-run] "
                f"candidate={selection['candidate_fingerprint']} "
                f"checkpoint_reuse=true selection_only=true "
                f"params={selection['canonical_parameter_json']} "
                f"output={selection['selection_path']}"
            )
    return "\n".join(lines) + "\n"


def _stage_manifest_valid(
    manifest_path: Path, fingerprint: str, output_paths: list[Path]
) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("status") != "completed" or payload.get("fingerprint") != fingerprint:
        return False
    expected = payload.get("output_sha256", {})
    return all(
        path.is_file() and expected.get(str(path.resolve())) == sha256_file(path)
        for path in output_paths
    )


def _write_stage_manifest(
    manifest_path: Path,
    *,
    stage: str,
    fingerprint: str,
    contract: dict[str, Any],
    outputs: list[Path],
    project_root: Path,
    started_at: str,
    elapsed_seconds: float,
    diagnostics: dict[str, Any] | None = None,
) -> None:
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "rag_cbwdm_calibration_grid_stage.v1",
            "status": "completed",
            "stage": stage,
            "fingerprint": fingerprint,
            "contract": contract,
            "output_sha256": {
                str(path.resolve()): sha256_file(path) for path in outputs
            },
            "git": git_state(project_root),
            "started_at": started_at,
            "completed_at": utc_now(),
            "elapsed_seconds": elapsed_seconds,
            "diagnostics": diagnostics,
        },
    )


def _write_failed_stage_manifest(
    manifest_path: Path,
    *,
    stage: str,
    fingerprint: str,
    contract: dict[str, Any],
    reason: str,
    project_root: Path,
) -> None:
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "rag_cbwdm_calibration_grid_stage.v1",
            "status": "failed",
            "stage": stage,
            "fingerprint": fingerprint,
            "contract": contract,
            "reason": reason,
            "git": git_state(project_root),
            "failed_at": utc_now(),
        },
    )


def _teacher_diagnostics(path: Path, method: str) -> dict[str, Any]:
    if method == "infogain_fever":
        manifest = json.loads(
            path.with_suffix(".manifest.json").read_text(encoding="utf-8")
        )
        distribution = (
            manifest.get("thresholds", {}).get("label_distribution", {})
        )
        return {"label_distribution": distribution}
    positive = negative = zero = 0
    stops: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            reason = str(row.get("stop_reason", "missing"))
            stops[reason] = stops.get(reason, 0) + 1
            tolerance = float(row.get("gain_tolerance", 1e-10))
            for step in row.get("steps", []):
                for item in step.get("candidate_gains", []):
                    value = float(item.get("gain", 0.0))
                    positive += int(value > tolerance)
                    negative += int(value < -tolerance)
                    zero += int(abs(value) <= tolerance)
    return {
        "gain_sign_counts": {
            "positive": positive,
            "negative": negative,
            "zero": zero,
        },
        "stop_reason_counts": dict(sorted(stops.items())),
    }


def _run_command(
    command: list[str],
    *,
    log_path: Path,
    env: dict[str, str],
    project_root: Path,
) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"$ {shlex.join(command)}\n")
        handle.write(f"[stage] started_at={utc_now()}\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
        elapsed = time.monotonic() - started
        handle.write(
            f"[stage] completed_at={utc_now()} returncode={completed.returncode} "
            f"elapsed_seconds={elapsed:.6f}\n"
        )
        handle.flush()
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return elapsed


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return


def _read_metrics(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"cannot read evaluation metrics: {exc}"
    metrics = payload.get("metrics", payload)
    required = ("accuracy", "macro_f1", "avg_num_docs", "avg_evidence_chars")
    missing = [key for key in required if metrics.get(key) is None]
    if missing:
        return None, f"missing evaluation metrics: {missing}"
    return {key: float(metrics[key]) for key in required}, None


def _teacher_paths(root: Path, method: str, fingerprint: str) -> tuple[Path, Path]:
    path = root / method / "teachers" / f"{fingerprint}.jsonl"
    return path, path.with_suffix(".grid.manifest.json")


def _evaluation_paths(selection: dict[str, Any]) -> tuple[Path, Path, Path]:
    directory = Path(selection["evaluation_dir"])
    return (
        directory / "predictions.jsonl",
        directory / "metrics.json",
        directory / "evaluation.grid.manifest.json",
    )


def _common_record(
    plan: dict[str, Any],
    node: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    predictions, metrics, evaluation_manifest = _evaluation_paths(selection)
    teacher_path, teacher_manifest = _teacher_paths(
        Path(plan["output_dir"]),
        node["method"],
        node["teacher_fingerprint"],
    )
    training_dir = Path(node["directory"])
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "method": node["method"],
        "split": "validation",
        "candidate_fingerprint": selection["candidate_fingerprint"],
        "training_fingerprint": node["training_fingerprint"],
        "selection_fingerprint": selection["selection_fingerprint"],
        "parameters": selection["parameters"],
        "canonical_parameter_json": selection["canonical_parameter_json"],
        "teacher_manifest": str(teacher_manifest),
        "checkpoint_manifest": str(
            training_dir / "training.grid.manifest.json"
        ),
        "selection_manifest": str(
            Path(selection["selection_path"]).with_suffix(".grid.manifest.json")
        ),
        "evaluation_manifest": str(evaluation_manifest),
        "prediction_path": str(predictions),
        "metrics_path": str(metrics),
        "selection_path": selection["selection_path"],
        "split_manifest_sha256": plan["split_manifest_sha256"],
        "split_sha256": plan["validation_sha256"],
        "validation_sha256": plan["validation_sha256"],
        "retrieval_sha256": plan["inputs"]["validation_retrieval"]["sha256"],
        "posterior_sha256": plan["inputs"]["validation_posteriors"]["sha256"],
        "generator_contract": plan["generator_contract"],
        "git_head": plan["git"].get("commit"),
        "teacher_path": str(teacher_path),
    }


def _commands_for_node(
    plan: dict[str, Any],
    node: dict[str, Any],
    selection: dict[str, Any] | None,
    *,
    config: dict[str, Any],
    project_root: Path,
    generator_model: str | None,
    selector_model: str | None,
    infogain_model: str | None,
    generator_device: str,
    selector_device: str,
    infogain_device: str,
    seed: int,
    resume: bool,
) -> dict[str, list[str]]:
    py = sys.executable
    inputs = plan["inputs"]
    root = Path(plan["output_dir"])
    teacher_path, _ = _teacher_paths(
        root, node["method"], node["teacher_fingerprint"]
    )
    training_dir = Path(node["directory"])
    common_resume = ["--resume"] if resume else []
    if node["method"] == "infogain_fever":
        teacher = [
            py,
            str(project_root / "scripts/12a_build_infogain_teacher.py"),
            "--posteriors",
            inputs["train_posteriors"]["path"],
            "--output",
            str(teacher_path),
            "--threshold-mode",
            "train_quantile",
            "--negative-quantile",
            str(node["teacher_parameters"]["negative_quantile"]),
            "--positive-quantile",
            str(node["teacher_parameters"]["positive_quantile"]),
            "--generator-model",
            str(generator_model or config["generator"]["model_name"]),
            "--prompt-hash",
            fever_prompt_hash(
                list(config["task"]["labels"]),
                dict(config["task"]["verbalizers"]),
            ),
            "--verbalizer-hash",
            stable_hash(config["task"]["verbalizers"]),
            *(
                [
                    "--generator-revision",
                    str(config["generator"]["revision"]),
                ]
                if config["generator"].get("revision")
                else []
            ),
            *common_resume,
        ]
        info = config["baselines"]["infogain_fever"]
        training = [
            py,
            str(project_root / "scripts/12b_train_infogain_reranker.py"),
            "--teacher",
            str(teacher_path),
            "--output-dir",
            str(training_dir),
            "--model-name-or-path",
            str(infogain_model or info["model_name"]),
            "--device",
            infogain_device,
            "--max-length",
            str(info.get("max_length", 512)),
            "--epochs",
            str(info.get("epochs", 1)),
            "--lr",
            str(info.get("lr", 2e-5)),
            "--beta",
            str(node["training_parameters"]["beta"]),
            "--seed",
            str(seed),
            *(
                ["--revision", str(info["revision"])]
                if info.get("revision")
                else []
            ),
            *common_resume,
        ]
        commands = {"teacher": teacher, "training": training}
        if selection:
            params = selection["selection_parameters"]
            commands["selection"] = [
                py,
                str(project_root / "scripts/12c_select_infogain_reranker.py"),
                "--retrieval",
                inputs["validation_retrieval"]["path"],
                "--checkpoint-dir",
                str(training_dir / "checkpoint"),
                "--output",
                selection["selection_path"],
                "--device",
                infogain_device,
                "--batch-size",
                str(info.get("candidate_batch_size", 16)),
                "--top-m",
                str(params["top_m"]),
                "--min-docs",
                str(params["min_docs"]),
                "--filter-threshold",
                str(params["filter_threshold"]),
                *common_resume,
            ]
    else:
        teacher = [
            py,
            str(project_root / "scripts/04_build_cbwdm_teacher.py"),
            "--config",
            inputs["config"]["path"],
            "--split",
            "train_core",
            "--posteriors",
            inputs["train_posteriors"]["path"],
            "--output",
            str(teacher_path),
            "--stop-threshold",
            str(node["teacher_parameters"]["stop_threshold"]),
            "--top-m",
            str(node["teacher_parameters"]["top_m"]),
            *common_resume,
        ]
        selector = config["selector"]
        training = [
            py,
            str(project_root / "scripts/10_train_cross_encoder_selector.py"),
            "--config",
            inputs["config"]["path"],
            "--posteriors",
            inputs["train_posteriors"]["path"],
            "--teacher",
            str(teacher_path),
            "--retrieval",
            inputs["train_retrieval"]["path"],
            "--output-dir",
            str(training_dir),
            "--model-name",
            str(selector_model or selector["model_name"]),
            *(
                ["--model-revision", str(selector["revision"])]
                if selector.get("revision")
                else []
            ),
            *(
                [
                    "--tokenizer-revision",
                    str(selector["tokenizer_revision"]),
                ]
                if selector.get("tokenizer_revision")
                else []
            ),
            "--device",
            selector_device,
            "--batch-size",
            str(selector.get("batch_size", 1)),
            "--b-plus",
            str(node["training_parameters"]["b_plus"]),
            "--b-minus",
            str(node["training_parameters"]["b_minus"]),
            "--gamma",
            str(node["training_parameters"]["gamma"]),
            "--beta",
            str(node["training_parameters"]["beta"]),
            "--seed",
            str(seed),
        ]
        commands = {"teacher": teacher, "training": training}
        if selection:
            params = selection["selection_parameters"]
            commands["selection"] = [
                py,
                str(project_root / "scripts/11_select_with_cross_encoder.py"),
                "--config",
                inputs["config"]["path"],
                "--posteriors",
                inputs["validation_posteriors"]["path"],
                "--checkpoint-dir",
                str(training_dir / "checkpoint"),
                "--output",
                selection["selection_path"],
                "--device",
                selector_device,
                "--batch-size",
                str(selector.get("candidate_batch_size", 8)),
                "--top-m",
                str(params["top_m"]),
                "--min-docs",
                str(params["min_docs"]),
                *(
                    ["--score-threshold", str(params["score_threshold"])]
                    if params["score_threshold"] is not None
                    else []
                ),
                *common_resume,
            ]
    if selection:
        predictions, metrics, _ = _evaluation_paths(selection)
        commands["evaluation"] = [
            py,
            str(project_root / "scripts/07_eval_rag_classification.py"),
            "--config",
            inputs["config"]["path"],
            "--split",
            "validation",
            "--selection",
            selection["selection_path"],
            "--output",
            str(predictions),
            "--metrics-output",
            str(metrics),
            "--method-name",
            node["method"],
            *(
                ["--model-name", generator_model]
                if generator_model
                else []
            ),
            *common_resume,
        ]
    return commands


def _publish_candidates(
    plan: dict[str, Any],
    records: list[dict[str, Any]],
    output_root: Path,
    *,
    execution_contract: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    records.sort(key=lambda item: item["candidate_fingerprint"])
    json_path = output_root / "calibration_candidates.json"
    csv_path = output_root / "calibration_candidates.csv"
    report_path = output_root / "calibration_grid_report.md"
    manifest_path = output_root / "calibration_grid_manifest.json"
    atomic_write_json(
        json_path,
        {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "status": "completed",
            "split": "validation",
            "candidates": records,
        },
    )
    fields = [
        "method",
        "candidate_fingerprint",
        "training_fingerprint",
        "selection_fingerprint",
        "status",
        "accuracy",
        "macro_f1",
        "avg_num_docs",
        "avg_evidence_chars",
        "parameters",
        "reason",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for record in records:
        metrics = record["metrics"]
        writer.writerow(
            {
                "method": record["method"],
                "candidate_fingerprint": record["candidate_fingerprint"],
                "training_fingerprint": record["training_fingerprint"],
                "selection_fingerprint": record["selection_fingerprint"],
                "status": record["status"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "avg_num_docs": metrics["avg_num_docs"],
                "avg_evidence_chars": metrics["avg_evidence_chars"],
                "parameters": record["canonical_parameter_json"],
                "reason": record["reason"],
            }
        )
    atomic_write_text(csv_path, stream.getvalue())
    completed = sum(record["status"] == "completed" for record in records)
    failed = len(records) - completed
    atomic_write_text(
        report_path,
        "\n".join(
            [
                "# FEVER Calibration Grid Execution",
                "",
                f"- Plan fingerprint: `{plan['fingerprint']}`",
                f"- Training candidates selected: {execution_contract['training_candidate_count']}",
                f"- Selection candidates selected: {len(records)}",
                f"- Completed: {completed}",
                f"- Failed: {failed}",
                "- Held-out test used: `false`",
                "",
            ]
        ),
    )
    output_sha = {
        path.name: sha256_file(path) for path in (json_path, csv_path, report_path)
    }
    manifest = {
        "schema_version": GRID_SCHEMA_VERSION,
        "status": "completed" if failed == 0 else "completed_with_failures",
        "fingerprint": stable_hash(execution_contract),
        "plan_fingerprint": plan["fingerprint"],
        "execution_contract": execution_contract,
        "candidate_count": len(records),
        "completed_candidate_count": completed,
        "failed_candidate_count": failed,
        "held_out_test_used": False,
        "output_sha256": output_sha,
        "git": git_state(project_root),
        "completed_at": utc_now(),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def execute_grid(
    plan: dict[str, Any],
    *,
    config_path: str | Path,
    project_root: str | Path,
    methods: set[str] | None = None,
    candidate_limit: int | None = None,
    candidate_fingerprint: str | None = None,
    max_training_candidates: int | None = None,
    skip_completed: bool = False,
    fail_fast: bool = False,
    continue_on_error: bool = True,
    generator_model: str | None = None,
    selector_model: str | None = None,
    infogain_model: str | None = None,
    generator_device: str = "auto",
    selector_device: str = "auto",
    infogain_device: str = "auto",
    seed: int = 13,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute one GPU-heavy stage at a time; never recompute retrieval/posteriors."""
    if fail_fast and continue_on_error:
        continue_on_error = False
    nodes = selected_plan_nodes(
        plan,
        methods=methods,
        candidate_limit=candidate_limit,
        candidate_fingerprint=candidate_fingerprint,
        max_training_candidates=max_training_candidates,
    )
    if dry_run:
        print(dry_run_text(plan, nodes), end="")
        return {
            "status": "dry_run",
            "training_candidate_count": len(nodes),
            "selection_candidate_count": sum(
                len(node["selections"]) for node in nodes
            ),
        }
    root = Path(plan["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    project = Path(project_root).resolve()
    config = load_yaml(config_path)
    env = os.environ.copy()
    records: list[dict[str, Any]] = []
    teacher_completed: set[str] = set()
    for node in nodes:
        training_dir = Path(node["directory"])
        training_dir.mkdir(parents=True, exist_ok=True)
        teacher_path, teacher_manifest = _teacher_paths(
            root, node["method"], node["teacher_fingerprint"]
        )
        teacher_path.parent.mkdir(parents=True, exist_ok=True)
        training_manifest = training_dir / "training.grid.manifest.json"
        base_commands = _commands_for_node(
            plan,
            node,
            None,
            config=config,
            project_root=project,
            generator_model=generator_model,
            selector_model=selector_model,
            infogain_model=infogain_model,
            generator_device=generator_device,
            selector_device=selector_device,
            infogain_device=infogain_device,
            seed=seed,
            resume=resume,
        )
        active_stage = "teacher"
        active_manifest = teacher_manifest
        active_contract: dict[str, Any] = {}
        active_fingerprint = ""
        try:
            teacher_contract = {
                "plan_fingerprint": plan["fingerprint"],
                "teacher_fingerprint": node["teacher_fingerprint"],
                "parameters": node["teacher_parameters"],
                "posteriors_sha256": plan["inputs"]["train_posteriors"]["sha256"],
            }
            teacher_fp = stable_hash(teacher_contract)
            active_contract = teacher_contract
            active_fingerprint = teacher_fp
            teacher_outputs = [teacher_path]
            if not (
                (skip_completed or resume)
                and _stage_manifest_valid(
                    teacher_manifest, teacher_fp, teacher_outputs
                )
            ):
                started_at = utc_now()
                elapsed = _run_command(
                    base_commands["teacher"],
                    log_path=teacher_path.parent
                    / f"{node['teacher_fingerprint']}.log",
                    env=env,
                    project_root=project,
                )
                diagnostics = _teacher_diagnostics(
                    teacher_path, node["method"]
                )
                _write_stage_manifest(
                    teacher_manifest,
                    stage="teacher",
                    fingerprint=teacher_fp,
                    contract=teacher_contract,
                    outputs=teacher_outputs,
                    project_root=project,
                    started_at=started_at,
                    elapsed_seconds=elapsed,
                    diagnostics=diagnostics,
                )
            else:
                diagnostics = _teacher_diagnostics(
                    teacher_path, node["method"]
                )
            if node["method"] == "infogain_fever":
                distribution = diagnostics.get("label_distribution", {})
                if int(distribution.get("positive", 0)) == 0 or int(
                    distribution.get("negative", 0)
                ) == 0:
                    raise ValueError(
                        "InfoGain teacher has an empty positive or negative class: "
                        f"{distribution}"
                    )
            teacher_completed.add(node["teacher_fingerprint"])

            active_stage = "training"
            active_manifest = training_manifest
            training_contract = {
                "plan_fingerprint": plan["fingerprint"],
                "training_fingerprint": node["training_fingerprint"],
                "teacher_sha256": sha256_file(teacher_path),
                "parameters": node["training_parameters"],
                "model": (
                    infogain_model
                    or config.get("baselines", {})
                    .get("infogain_fever", {})
                    .get("model_name")
                    if node["method"] == "infogain_fever"
                    else selector_model
                    or config.get("selector", {}).get("model_name")
                ),
                "model_revision": (
                    config.get("baselines", {})
                    .get("infogain_fever", {})
                    .get("revision")
                    if node["method"] == "infogain_fever"
                    else config.get("selector", {}).get("revision")
                ),
                "git_head": plan["git"].get("commit"),
            }
            training_fp = stable_hash(training_contract)
            active_contract = training_contract
            active_fingerprint = training_fp
            if node["method"] == "infogain_fever":
                training_outputs = [
                    training_dir / "checkpoint" / "heads.pt",
                    training_dir / "checkpoint" / "infogain_config.json",
                ]
            else:
                checkpoint = training_dir / "checkpoint"
                weights = next(
                    (
                        path
                        for name in ("model.safetensors", "pytorch_model.bin")
                        if (path := checkpoint / name).is_file()
                    ),
                    checkpoint / "model.safetensors",
                )
                training_outputs = [checkpoint / "config.json", weights]
            if not (
                (skip_completed or resume)
                and _stage_manifest_valid(
                    training_manifest, training_fp, training_outputs
                )
            ):
                started_at = utc_now()
                elapsed = _run_command(
                    base_commands["training"],
                    log_path=training_dir / "training.log",
                    env=env,
                    project_root=project,
                )
                if node["method"] == "rag_cbwdm":
                    checkpoint = training_dir / "checkpoint"
                    weights = next(
                        (
                            checkpoint / name
                            for name in ("model.safetensors", "pytorch_model.bin")
                            if (checkpoint / name).is_file()
                        ),
                        None,
                    )
                    if weights is None:
                        raise FileNotFoundError(
                            f"RAG-CBWDM checkpoint weights missing: {checkpoint}"
                        )
                    training_outputs = [checkpoint / "config.json", weights]
                _write_stage_manifest(
                    training_manifest,
                    stage="training",
                    fingerprint=training_fp,
                    contract=training_contract,
                    outputs=training_outputs,
                    project_root=project,
                    started_at=started_at,
                    elapsed_seconds=elapsed,
                )
            _empty_cuda_cache()
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            _write_failed_stage_manifest(
                active_manifest,
                stage=active_stage,
                fingerprint=active_fingerprint,
                contract=active_contract,
                reason=reason,
                project_root=project,
            )
            for selection in node["selections"]:
                records.append(
                    {
                        **_common_record(plan, node, selection),
                        "metrics": {
                            "accuracy": None,
                            "macro_f1": None,
                            "avg_num_docs": None,
                            "avg_evidence_chars": None,
                        },
                        "status": "failed",
                        "reason": reason,
                    }
                )
            if fail_fast or not continue_on_error:
                break
            continue

        for selection in node["selections"]:
            record = _common_record(plan, node, selection)
            selection_path = Path(selection["selection_path"])
            selection_path.parent.mkdir(parents=True, exist_ok=True)
            predictions, metrics_path, evaluation_manifest = _evaluation_paths(
                selection
            )
            predictions.parent.mkdir(parents=True, exist_ok=True)
            commands = _commands_for_node(
                plan,
                node,
                selection,
                config=config,
                project_root=project,
                generator_model=generator_model,
                selector_model=selector_model,
                infogain_model=infogain_model,
                generator_device=generator_device,
                selector_device=selector_device,
                infogain_device=infogain_device,
                seed=seed,
                resume=resume,
            )
            active_selection_stage = "selection"
            active_selection_manifest = selection_path.with_suffix(
                ".grid.manifest.json"
            )
            active_selection_contract: dict[str, Any] = {}
            active_selection_fingerprint = ""
            try:
                selection_contract = {
                    "plan_fingerprint": plan["fingerprint"],
                    "training_fingerprint": node["training_fingerprint"],
                    "selection_fingerprint": selection["selection_fingerprint"],
                    "parameters": selection["selection_parameters"],
                    "checkpoint_output_sha256": json.loads(
                        training_manifest.read_text(encoding="utf-8")
                    ).get("output_sha256"),
                    "validation_retrieval_sha256": plan["inputs"][
                        "validation_retrieval"
                    ]["sha256"],
                    "validation_posteriors_sha256": plan["inputs"][
                        "validation_posteriors"
                    ]["sha256"],
                }
                selection_fp = stable_hash(selection_contract)
                active_selection_contract = selection_contract
                active_selection_fingerprint = selection_fp
                selection_grid_manifest = selection_path.with_suffix(
                    ".grid.manifest.json"
                )
                active_selection_manifest = selection_grid_manifest
                if not (
                    (skip_completed or resume)
                    and _stage_manifest_valid(
                        selection_grid_manifest, selection_fp, [selection_path]
                    )
                ):
                    started_at = utc_now()
                    elapsed = _run_command(
                        commands["selection"],
                        log_path=selection_path.with_suffix(".log"),
                        env=env,
                        project_root=project,
                    )
                    _write_stage_manifest(
                        selection_grid_manifest,
                        stage="selection",
                        fingerprint=selection_fp,
                        contract=selection_contract,
                        outputs=[selection_path],
                        project_root=project,
                        started_at=started_at,
                        elapsed_seconds=elapsed,
                    )
                evaluation_contract = {
                    "plan_fingerprint": plan["fingerprint"],
                    "selection_sha256": sha256_file(selection_path),
                    "generator_contract": plan["generator_contract"],
                    "validation_sha256": plan["validation_sha256"],
                }
                evaluation_fp = stable_hash(evaluation_contract)
                active_selection_stage = "evaluation"
                active_selection_manifest = evaluation_manifest
                active_selection_contract = evaluation_contract
                active_selection_fingerprint = evaluation_fp
                if not (
                    (skip_completed or resume)
                    and _stage_manifest_valid(
                        evaluation_manifest,
                        evaluation_fp,
                        [predictions, metrics_path],
                    )
                ):
                    started_at = utc_now()
                    elapsed = _run_command(
                        commands["evaluation"],
                        log_path=predictions.parent / "evaluation.log",
                        env=env,
                        project_root=project,
                    )
                    _write_stage_manifest(
                        evaluation_manifest,
                        stage="evaluation",
                        fingerprint=evaluation_fp,
                        contract=evaluation_contract,
                        outputs=[predictions, metrics_path],
                        project_root=project,
                        started_at=started_at,
                        elapsed_seconds=elapsed,
                    )
                metrics, reason = _read_metrics(metrics_path)
                if reason:
                    raise ValueError(reason)
                record.update(
                    {
                        "metrics": metrics,
                        "status": "completed",
                        "reason": None,
                    }
                )
            except Exception as exc:
                failure_reason = f"{type(exc).__name__}: {exc}"
                _write_failed_stage_manifest(
                    active_selection_manifest,
                    stage=active_selection_stage,
                    fingerprint=active_selection_fingerprint,
                    contract=active_selection_contract,
                    reason=failure_reason,
                    project_root=project,
                )
                record.update(
                    {
                        "metrics": {
                            "accuracy": None,
                            "macro_f1": None,
                            "avg_num_docs": None,
                            "avg_evidence_chars": None,
                        },
                        "status": "failed",
                        "reason": failure_reason,
                    }
                )
                if fail_fast or not continue_on_error:
                    records.append(record)
                    break
            records.append(record)
            _empty_cuda_cache()
        if (fail_fast or not continue_on_error) and records and records[-1][
            "status"
        ] == "failed":
            break
    output_root = Path(plan["output_dir"]).parent
    execution_contract = {
        "plan_fingerprint": plan["fingerprint"],
        "methods": sorted(methods or set(CALIBRATED_METHODS)),
        "candidate_limit": candidate_limit,
        "candidate_fingerprint": candidate_fingerprint,
        "max_training_candidates": max_training_candidates,
        "training_candidate_count": len(nodes),
        "selection_candidate_count": sum(
            len(node["selections"]) for node in nodes
        ),
        "skip_completed": skip_completed,
        "seed": seed,
        "git_head": plan["git"].get("commit"),
    }
    return _publish_candidates(
        plan,
        records,
        output_root,
        execution_contract=execution_contract,
        project_root=project,
    )

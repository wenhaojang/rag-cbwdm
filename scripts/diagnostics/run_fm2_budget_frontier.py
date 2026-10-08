"""Plan and run the FM2 development-only evidence-budget frontier."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import load_yaml
from src.artifact_binding import SELECTION_MANIFEST_SCHEMA_VERSION
from src.run_manifest import atomic_write_json, sha256_file, stable_hash


CONFIG_SCHEMA_VERSION = "rag_cbwdm_fm2_budget_frontier_config.v1"
PLAN_SCHEMA_VERSION = "rag_cbwdm_fm2_budget_frontier_plan.v1"
DATASET_ID = "fm2_official_closed_page_v1"
EXPECTED_GENERATORS = (
    "qwen2.5-0.5b-instruct",
    "qwen2.5-1.5b-instruct",
    "qwen2.5-7b-instruct",
    "mistral-7b-instruct-v0.3",
)
EXPECTED_GENERATOR_CONFIGS = {
    "qwen2.5-0.5b-instruct": "configs/fm2_qwen05_full_development.server.yaml",
    "qwen2.5-1.5b-instruct": "configs/fm2_qwen15_full_development.server.yaml",
    "qwen2.5-7b-instruct": "configs/fm2_qwen7_full_development.server.yaml",
    "mistral-7b-instruct-v0.3": "configs/fm2_mistral7_full_development.server.yaml",
}
EXPECTED_METHODS = (
    "retrieval",
    "bge",
    "infogain",
    "linear_cbwdm",
    "raw_kcbwdm_v2a",
    "normalized_kcbwdm_rho1",
)
EXPECTED_UNDERLYING_METHODS = {
    "retrieval": "retrieval_topk",
    "bge": "bge",
    "infogain": "infogain_fever",
    "linear_cbwdm": "rag_cbwdm_signed_v1",
    "raw_kcbwdm_v2a": "kcbwdm_linear_gate_v2",
    "normalized_kcbwdm_rho1": "kcbwdm_normalized_rho_v1",
}
FORBIDDEN_COMMAND_TOKENS = (
    "posterior",
    "teacher",
    "train",
    "selector",
    "03_compute_label_posteriors.py",
    "25_materialize_signed_v1_teacher.py",
    "26_train_signed_v1.py",
    "27_select_signed_v1.py",
    "28_materialize_kcbwdm_signed_v1_teacher.py",
    "held_out_test",
    "held-out",
    "pyserini",
    "lucene",
    "--limit",
)


class FrontierPlanError(ValueError):
    """Raised when the frozen frontier inputs fail closed validation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or execute the FM2 development evidence-budget frontier."
    )
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--plan-output", required=True)
    parser.add_argument("--commands-output", required=True)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--artifact-root")
    parser.add_argument("--server-python")
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FrontierPlanError(message)


def load_frontier_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    _require(isinstance(config, dict), "Frontier config must be a YAML object")
    _require(config.get("schema_version") == CONFIG_SCHEMA_VERSION, "Config schema mismatch")
    _require(config.get("dataset_id") == DATASET_ID, "Config dataset_id mismatch")
    _require(config.get("split") == "validation", "Frontier is validation-only")
    _require(config.get("held_out") is False, "Frontier must set held_out=false")
    _require(config.get("seed") == 13, "Frontier seed must be 13")
    _require(config.get("expected_rows") == 1169, "FM2 validation must contain 1169 rows")
    _require(config.get("requested_caps") == [1, 2, 3], "Requested caps must be [1, 2, 3]")
    generators = config.get("generators")
    methods = config.get("methods")
    _require(isinstance(generators, list), "Config generators must be a list")
    _require(isinstance(methods, list), "Config methods must be a list")
    generator_ids = tuple(item.get("generator_id") for item in generators)
    method_ids = tuple(item.get("frontier_method_id") for item in methods)
    _require(generator_ids == EXPECTED_GENERATORS, "Config generator set/order mismatch")
    _require(method_ids == EXPECTED_METHODS, "Config method set/order mismatch")
    for generator in generators:
        generator_id = generator["generator_id"]
        _require(
            generator.get("config") == EXPECTED_GENERATOR_CONFIGS[generator_id],
            f"Frozen generator config mismatch for {generator_id}",
        )
    for method in methods:
        frontier_id = method["frontier_method_id"]
        _require(
            method.get("underlying_method") == EXPECTED_UNDERLYING_METHODS[frontier_id],
            f"Underlying method mismatch for {frontier_id}",
        )
        for field in (
            "variant",
            "prefix_status",
            "prefix_contract_preserving",
            "selection_template",
            "selection_manifest_template",
            "native_metrics_template",
        ):
            _require(field in method, f"Method {frontier_id} lacks {field}")
    for field in ("artifact_root", "server_python", "generator_manifest_template", "no_evidence_metrics_template"):
        _require(bool(config.get(field)), f"Config lacks {field}")
    return config


def _render(template: str, generator_id: str) -> str:
    try:
        return template.format(generator_id=generator_id)
    except (KeyError, ValueError) as exc:
        raise FrontierPlanError(f"Invalid path template {template!r}: {exc}") from exc


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"Missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrontierPlanError(f"Invalid {label}: {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} must be a JSON object: {path}")
    return payload


def validate_selection_artifact(
    selection_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_method: str,
    expected_rows: int,
    expected_split: str,
) -> tuple[tuple[str, ...], int]:
    selection = Path(selection_path)
    manifest_file = Path(manifest_path)
    _require(selection.is_file(), f"Missing selection artifact: {selection}")
    manifest = _load_json_object(manifest_file, "selection manifest")
    _require(
        manifest.get("schema_version") == SELECTION_MANIFEST_SCHEMA_VERSION,
        f"Selection manifest schema mismatch: {manifest_file}",
    )
    _require(
        manifest.get("status") == "completed" and manifest.get("completed") is True,
        f"Selection manifest is not completed: {manifest_file}",
    )
    _require(manifest.get("method") == expected_method, f"Selection method mismatch: {selection}")
    _require(
        Path(str(manifest.get("output_path"))).resolve() == selection.resolve(),
        f"Selection manifest output path mismatch: {selection}",
    )
    _require(
        manifest.get("output_sha256") == sha256_file(selection),
        f"Selection checksum mismatch: {selection}",
    )
    _require(
        manifest.get("num_rows") == expected_rows,
        f"Selection manifest row-count mismatch: {selection}",
    )
    contract = manifest.get("contract")
    _require(isinstance(contract, dict), f"Selection manifest lacks contract: {manifest_file}")
    _require(contract.get("method") == expected_method, f"Selection contract method mismatch: {selection}")
    _require(
        manifest.get("fingerprint") == stable_hash(contract),
        f"Selection manifest fingerprint mismatch: {manifest_file}",
    )
    binding = manifest.get("artifact_binding")
    _require(isinstance(binding, dict), f"Selection artifact_binding is invalid: {manifest_file}")
    _require(binding.get("dataset_id") == DATASET_ID, f"Selection dataset identity mismatch: {selection}")
    _require(binding.get("method") == expected_method, f"Selection binding method mismatch: {selection}")
    _require(contract.get("artifact_binding") == binding, f"Selection binding/contract mismatch: {selection}")
    _require(
        manifest.get("method_contract_version") == contract.get("method_contract_version"),
        f"Selection method contract version mismatch: {selection}",
    )

    ids: list[str] = []
    seen: set[str] = set()
    native_max_docs = 0
    try:
        with selection.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                _require(isinstance(row, dict), f"Selection row {line_number} is not an object: {selection}")
                row_id = row.get("id")
                _require(isinstance(row_id, str) and row_id, f"Selection row lacks string id: {selection}:{line_number}")
                _require(row_id not in seen, f"Duplicate selection id {row_id!r}: {selection}")
                seen.add(row_id)
                ids.append(row_id)
                _require(row.get("split") == expected_split, f"Selection split mismatch: {selection}:{line_number}")
                selected_ids = row.get("selected_doc_ids")
                selected_docs = row.get("selected_docs")
                _require(isinstance(selected_ids, list), f"selected_doc_ids must be a list: {selection}:{line_number}")
                _require(isinstance(selected_docs, list), f"selected_docs must be a list: {selection}:{line_number}")
                doc_ids: list[Any] = []
                for doc in selected_docs:
                    _require(isinstance(doc, dict) and "doc_id" in doc, f"selected_docs entry lacks doc_id: {selection}:{line_number}")
                    doc_ids.append(doc["doc_id"])
                _require(selected_ids == doc_ids, f"selected_doc_ids mismatch: {selection}:{line_number}")
                _require(row.get("num_docs") == len(selected_docs), f"num_docs mismatch: {selection}:{line_number}")
                native_max_docs = max(native_max_docs, len(selected_docs))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrontierPlanError(f"Invalid selection JSONL {selection}: {exc}") from exc
    _require(len(ids) == expected_rows, f"Selection row count mismatch: expected={expected_rows} actual={len(ids)} path={selection}")
    return tuple(ids), native_max_docs


def validate_metrics_reference(path: str | Path, *, expected_rows: int, split: str) -> dict[str, Any]:
    metrics = _load_json_object(Path(path), "metrics reference")
    _require(metrics.get("num_examples") == expected_rows, f"Metrics num_examples mismatch: {path}")
    _require(metrics.get("split") == split, f"Metrics split mismatch: {path}")
    for field in (
        "accuracy",
        "macro_f1",
        "num_correct",
        "avg_num_docs",
        "avg_evidence_chars",
    ):
        _require(isinstance(metrics.get(field), (int, float)), f"Metrics field {field} is missing/non-numeric: {path}")
    _require(
        metrics.get("avg_original_retrieval_rank") is None
        or isinstance(metrics.get("avg_original_retrieval_rank"), (int, float)),
        f"Metrics field avg_original_retrieval_rank is invalid: {path}",
    )
    return metrics


def validate_generator_manifest(path: str | Path, generator_id: str) -> None:
    manifest = _load_json_object(Path(path), "generator manifest")
    _require(manifest.get("status") == "completed", f"Generator manifest is not completed: {path}")
    dataset_identity = manifest.get("dataset_identity")
    generator_identity = manifest.get("generator_identity")
    _require(isinstance(dataset_identity, dict), f"Generator manifest lacks dataset identity: {path}")
    _require(isinstance(generator_identity, dict), f"Generator manifest lacks generator identity: {path}")
    _require(dataset_identity.get("dataset_id") == DATASET_ID, f"Generator manifest dataset mismatch: {path}")
    _require(generator_identity.get("generator_id") == generator_id, f"Generator ID mismatch: {path}")
    _require(
        manifest.get("generator_identity_fingerprint") == stable_hash(generator_identity),
        f"Generator identity fingerprint mismatch: {path}",
    )


def _join_artifact(root: str, *parts: str) -> str:
    if root.startswith("/"):
        return str(PurePosixPath(root).joinpath(*parts))
    return str(Path(root).joinpath(*parts))


def _resolve_project_path(project_root: Path, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else project_root / path)


def _prefix_semantics(method: dict[str, Any], cap: int) -> tuple[str, bool]:
    if cap in method.get("diagnostic_caps", []):
        return "posthoc_diagnostic", False
    return str(method["prefix_status"]), bool(method["prefix_contract_preserving"])


def _build_command(
    *,
    server_python: str,
    project_root: Path,
    generator_config: str,
    split: str,
    selection_path: str,
    selection_manifest_path: str,
    cap: int,
    predictions_path: str,
    metrics_path: str,
    generator_manifest_path: str,
    underlying_method: str,
) -> list[str]:
    command = [
        server_python,
        str(project_root / "scripts" / "07_eval_rag_classification.py"),
        "--config",
        _resolve_project_path(project_root, generator_config),
        "--split",
        split,
        "--selection",
        selection_path,
        "--selection-manifest",
        selection_manifest_path,
        "--max-docs",
        str(cap),
        "--output",
        predictions_path,
        "--metrics-output",
        metrics_path,
        "--generator-manifest",
        generator_manifest_path,
        "--experiment-type",
        "budget_frontier",
        "--formal-v2-identity",
        "--method-name",
        underlying_method,
        "--resume",
    ]
    rendered = shlex.join(command).casefold()
    for token in FORBIDDEN_COMMAND_TOKENS:
        _require(token.casefold() not in rendered, f"Forbidden command token generated: {token}")
    # ``infogain_fever`` is the frozen scientific method identity, not a FEVER
    # dataset reference.  It is the sole permitted occurrence of that token.
    _require(
        "fever" not in rendered.replace("infogain_fever", ""),
        "Generated command contains a FEVER dataset reference",
    )
    _require(command.count("--max-docs") == 1, "Evaluation command must contain exactly one --max-docs")
    return command


def build_execution_plan(
    config: dict[str, Any],
    *,
    project_root: str | Path,
    artifact_root: str | None = None,
    server_python: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root)
    output_root = artifact_root or str(config["artifact_root"])
    python = server_python or str(config["server_python"])
    expected_rows = int(config["expected_rows"])
    split = str(config["split"])
    caps = list(config["requested_caps"])
    points: list[dict[str, Any]] = []
    canonical_ids: tuple[str, ...] | None = None
    selection_cache: dict[tuple[str, str, str], tuple[tuple[str, ...], int]] = {}
    metrics_seen: set[str] = set()

    for generator in config["generators"]:
        generator_id = generator["generator_id"]
        generator_manifest = _render(config["generator_manifest_template"], generator_id)
        validate_generator_manifest(generator_manifest, generator_id)
        no_evidence_metrics = _render(config["no_evidence_metrics_template"], generator_id)
        validate_metrics_reference(no_evidence_metrics, expected_rows=expected_rows, split=split)
        metrics_seen.add(no_evidence_metrics)
        points.append(
            {
                "generator_id": generator_id,
                "frontier_method_id": "no_evidence",
                "underlying_method": "no_evidence",
                "variant": "deterministic",
                "frontier_role": "no_evidence_anchor",
                "frontier_status": "native",
                "nominal_cap": 0,
                "source": "native_reuse",
                "effective_equivalence": "native",
                "selection_path": None,
                "selection_manifest_path": None,
                "native_metrics_path": no_evidence_metrics,
                "output_metrics_path": None,
                "output_predictions_path": None,
                "native_max_docs": 0,
                "requires_evaluation": False,
                "contract_preserving": True,
                "command": [],
            }
        )

    for method in config["methods"]:
        frontier_id = method["frontier_method_id"]
        underlying_method = method["underlying_method"]
        for generator in config["generators"]:
            generator_id = generator["generator_id"]
            selection_path = _render(method["selection_template"], generator_id)
            selection_manifest = _render(method["selection_manifest_template"], generator_id)
            cache_key = (selection_path, selection_manifest, underlying_method)
            if cache_key not in selection_cache:
                selection_cache[cache_key] = validate_selection_artifact(
                    selection_path,
                    selection_manifest,
                    expected_method=underlying_method,
                    expected_rows=expected_rows,
                    expected_split=split,
                )
            ids, native_max_docs = selection_cache[cache_key]
            if canonical_ids is None:
                canonical_ids = ids
            else:
                _require(ids == canonical_ids, f"Selection IDs/order mismatch: {selection_path}")
            native_metrics = _render(method["native_metrics_template"], generator_id)
            if native_metrics not in metrics_seen:
                validate_metrics_reference(native_metrics, expected_rows=expected_rows, split=split)
                metrics_seen.add(native_metrics)
            common = {
                "generator_id": generator_id,
                "frontier_method_id": frontier_id,
                "underlying_method": underlying_method,
                "variant": method["variant"],
                "selection_path": selection_path,
                "selection_manifest_path": selection_manifest,
                "native_metrics_path": native_metrics,
                "native_max_docs": native_max_docs,
            }
            points.append(
                {
                    **common,
                    "frontier_role": "native_method_point",
                    "frontier_status": "native",
                    "nominal_cap": None,
                    "source": "native_reuse",
                    "effective_equivalence": "native",
                    "output_metrics_path": None,
                    "output_predictions_path": None,
                    "requires_evaluation": False,
                    "contract_preserving": True,
                    "command": [],
                }
            )
            generator_manifest = _render(config["generator_manifest_template"], generator_id)
            for cap in caps:
                status, contract_preserving = _prefix_semantics(method, cap)
                reuse_native = cap >= native_max_docs
                output_directory = _join_artifact(output_root, generator_id, frontier_id, f"cap_{cap}")
                predictions_path = _join_artifact(output_directory, "predictions.jsonl")
                metrics_path = _join_artifact(output_directory, "metrics.json")
                command: list[str] = []
                if not reuse_native:
                    command = _build_command(
                        server_python=python,
                        project_root=root,
                        generator_config=generator["config"],
                        split=split,
                        selection_path=selection_path,
                        selection_manifest_path=selection_manifest,
                        cap=cap,
                        predictions_path=predictions_path,
                        metrics_path=metrics_path,
                        generator_manifest_path=generator_manifest,
                        underlying_method=underlying_method,
                    )
                points.append(
                    {
                        **common,
                        "frontier_role": "requested_cap",
                        "frontier_status": status,
                        "nominal_cap": cap,
                        "source": "native_reuse" if reuse_native else "evaluation",
                        "effective_equivalence": "native" if reuse_native else None,
                        "output_metrics_path": None if reuse_native else metrics_path,
                        "output_predictions_path": None if reuse_native else predictions_path,
                        "requires_evaluation": not reuse_native,
                        "contract_preserving": contract_preserving,
                        "command": command,
                    }
                )

    evaluation_jobs = sum(point["requires_evaluation"] for point in points)
    native_cap_reuse = sum(
        point["frontier_role"] == "requested_cap" and point["source"] == "native_reuse"
        for point in points
    )
    payload: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "dataset_id": config["dataset_id"],
        "split": split,
        "held_out": config["held_out"],
        "seed": config["seed"],
        "expected_rows": expected_rows,
        "methods": [method["frontier_method_id"] for method in config["methods"]],
        "generators": [generator["generator_id"] for generator in config["generators"]],
        "requested_caps": caps,
        "artifact_root": output_root,
        "server_python": python,
        "total_logical_points": len(points),
        "evaluation_job_count": evaluation_jobs,
        "native_reuse_count": native_cap_reuse,
        "no_evidence_anchor_count": sum(point["frontier_role"] == "no_evidence_anchor" for point in points),
        "config_fingerprint": stable_hash(config),
        "points": points,
    }
    payload["plan_fingerprint"] = stable_hash(payload)
    return payload


def write_commands(path: str | Path, plan: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    lines = [shlex.join(point["command"]) for point in plan["points"] if point["requires_evaluation"]]
    temporary.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8", newline="\n")
    os.replace(temporary, target)


def execute_plan(plan: dict[str, Any]) -> None:
    for point in plan["points"]:
        if not point["requires_evaluation"]:
            continue
        result = subprocess.run(point["command"], check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"Budget-frontier evaluator failed with exit {result.returncode}: "
                f"{point['generator_id']} {point['frontier_method_id']} cap={point['nominal_cap']}"
            )


def main() -> None:
    args = parse_args()
    config = load_frontier_config(args.config)
    plan = build_execution_plan(
        config,
        project_root=args.project_root,
        artifact_root=args.artifact_root,
        server_python=args.server_python,
    )
    atomic_write_json(args.plan_output, plan)
    write_commands(args.commands_output, plan)
    if args.execute:
        execute_plan(plan)
    print(
        f"[fm2_budget_frontier] logical_points={plan['total_logical_points']} "
        f"evaluation_jobs={plan['evaluation_job_count']} "
        f"native_cap_reuse={plan['native_reuse_count']}"
    )


if __name__ == "__main__":
    main()

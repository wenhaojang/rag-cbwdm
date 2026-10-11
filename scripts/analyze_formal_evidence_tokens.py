"""Offline evidence-token audit for the 48 formal-v2 development results.

This command reads completed selection/evaluation artifacts and loads only the
generator tokenizer. It never runs a selector, generator, or training stage.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.artifact_binding import (
    EVALUATION_MANIFEST_SCHEMA_VERSION,
    MATCHED_MAIN,
    build_evaluation_binding,
    validate_generator_identity_against_config,
    validate_generator_manifest,
    validate_selection_provenance,
)
from src.io_utils import load_yaml, read_jsonl
from src.run_manifest import (
    atomic_write_json,
    git_state,
    sha256_file,
    stable_hash,
    utc_now,
)


PLAN_SCHEMA = "rag_cbwdm_evidence_token_analysis_plan.v1"
AUDIT_SCHEMA = "rag_cbwdm_evidence_token_audit.v1"
ANALYSIS_SCHEMA = "rag_cbwdm_evidence_token_analysis.v1"
TOKEN_BASIS = (
    "generator_tokenizer_rendered_evidence_block_before_prompt_truncation"
)
EXPECTED_DATASETS = {
    "fever_binary_v2": 5000,
    "fm2_official_closed_page_v1": 1169,
}
EXPECTED_GENERATORS = (
    "qwen2.5-0.5b-instruct",
    "qwen2.5-1.5b-instruct",
    "qwen2.5-7b-instruct",
    "mistral-7b-instruct-v0.3",
)
EXPECTED_TOKENIZERS = {
    "qwen2.5-0.5b-instruct": "/root/models/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b-instruct": "/root/models/Qwen2.5-1.5B-Instruct",
    "qwen2.5-7b-instruct": "/root/models/Qwen2.5-7B-Instruct",
    "mistral-7b-instruct-v0.3": "/root/models/Mistral-7B-Instruct-v0.3",
}
EXPECTED_METHODS: dict[str, int | None] = {
    "no_evidence": None,
    "retrieval_topk": None,
    "bge": None,
    "infogain_fever": 13,
    "rag_cbwdm_signed_v1": 13,
    "kcbwdm_linear_gate_v2": 13,
}
EXPECTED_ROOTS = {
    "main": "/root/experiments/rag_cbwdm/formal_v2_full_development",
    "corrected_infogain": (
        "/root/experiments/rag_cbwdm/formal_v2_infogain_top4_filter02_final"
    ),
    "fever_kcbwdm_v2a": (
        "/root/experiments/rag_cbwdm/"
        "formal_v2_fever_kcbwdm_linear_gate_v2_full_development"
    ),
    "fm2_kcbwdm_v2a": (
        "/root/experiments/rag_cbwdm/"
        "formal_v2_kcbwdm_linear_gate_v2_full_development"
    ),
}
MANAGED_OUTPUTS = (
    "main_results_48.csv",
    "main_results_48.md",
    "main_results_48.tex",
    "accuracy_token_plot_data.csv",
    "token_audit.json",
    "token_audit.md",
    "analysis_manifest.json",
)


def _load_evaluator_helpers() -> tuple[Callable[..., Any], Callable[..., str]]:
    evaluator_path = PROJECT_ROOT / "scripts" / "07_eval_rag_classification.py"
    spec = importlib.util.spec_from_file_location(
        "rag_cbwdm_eval_rag_classification", evaluator_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import evaluator helpers from {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.recover_selected_docs, module.build_evidence_context


recover_selected_docs, build_evidence_context = _load_evaluator_helpers()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--validate-infogain", action="store_true")
    mode.add_argument("--all", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _forbid_non_validation(value: str, label: str) -> None:
    normalized = value.replace("\\", "/").lower()
    if "held_out" in normalized or "/test" in normalized or "train_core" in normalized:
        raise ValueError(f"{label} touches a forbidden non-validation source: {value}")


def _source_paths(
    *,
    dataset_id: str,
    generator_id: str,
    method_id: str,
    seed: int | None,
    shared_selection: bool,
    root: str,
) -> dict[str, str]:
    base = Path(root) / dataset_id
    artifact_method = "infogain" if method_id == "infogain_fever" else method_id
    generator_manifest = base / generator_id / "generator" / "manifest.json"
    if shared_selection:
        selection = base / "shared" / artifact_method / "validation" / "selection.jsonl"
    else:
        selection = (
            base
            / generator_id
            / artifact_method
            / f"seed{seed}"
            / "selection"
            / "validation.jsonl"
        )
    eval_root = (
        base
        / generator_id
        / "evaluation"
        / artifact_method
        / "matched_main"
        / ("deterministic" if seed is None else f"seed{seed}")
        / "validation"
    )
    return {
        "generator_manifest": str(generator_manifest),
        "selection": str(selection),
        "selection_manifest": str(selection.with_suffix(".manifest.json")),
        "predictions": str(eval_root / "predictions.jsonl"),
        "metrics": str(eval_root / "metrics.json"),
        "evaluation_manifest": str(eval_root / "metrics.manifest.json"),
    }


def load_plan(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan_path = Path(path)
    plan = load_yaml(plan_path)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("Evidence-token analysis plan schema mismatch")
    if plan.get("split") != "validation" or plan.get("experiment_type") != MATCHED_MAIN:
        raise ValueError("Analysis accepts only matched-main validation")
    output_root = str(plan.get("analysis_output_root", ""))
    if output_root != (
        "/root/experiments/rag_cbwdm/"
        "formal_v2_analysis/evidence_token_comparison_v1"
    ):
        raise ValueError("Unexpected analysis_output_root")

    generator_specs = list(plan.get("generators", []))
    generators = tuple(str(item.get("generator_id")) for item in generator_specs)
    if generators != EXPECTED_GENERATORS:
        raise ValueError("Plan must contain the four frozen generators in canonical order")
    generator_by_id = {str(item["generator_id"]): item for item in generator_specs}
    for generator_id, spec in generator_by_id.items():
        if spec.get("tokenizer_name_or_path") != EXPECTED_TOKENIZERS[generator_id]:
            raise ValueError(f"{generator_id}: tokenizer path mismatch")
        if spec.get("tokenizer_revision") is not None:
            raise ValueError(f"{generator_id}: unexpected tokenizer revision")
    methods = list(plan.get("methods", []))
    if {item.get("method_id") for item in methods} != set(EXPECTED_METHODS):
        raise ValueError("Plan must contain exactly the six main-table methods")
    datasets = plan.get("datasets")
    if not isinstance(datasets, Mapping) or set(datasets) != set(EXPECTED_DATASETS):
        raise ValueError("Plan must contain exactly FEVER and FM2")

    groups: list[dict[str, Any]] = []
    for dataset_id, expected_rows in EXPECTED_DATASETS.items():
        dataset = datasets[dataset_id]
        if int(dataset.get("expected_rows", -1)) != expected_rows:
            raise ValueError(f"{dataset_id}: expected_rows mismatch")
        roots = dataset.get("roots")
        if not isinstance(roots, Mapping):
            raise ValueError(f"{dataset_id}: roots must be a mapping")
        if str(roots.get("main")) != EXPECTED_ROOTS["main"]:
            raise ValueError(f"{dataset_id}: main source root mismatch")
        if str(roots.get("corrected_infogain")) != EXPECTED_ROOTS[
            "corrected_infogain"
        ]:
            raise ValueError(f"{dataset_id}: corrected InfoGain root mismatch")
        expected_kcbwdm = EXPECTED_ROOTS[
            "fever_kcbwdm_v2a"
            if dataset_id == "fever_binary_v2"
            else "fm2_kcbwdm_v2a"
        ]
        if str(roots.get("kcbwdm_v2a")) != expected_kcbwdm:
            raise ValueError(f"{dataset_id}: KCBWDM-v2A root mismatch")

        for generator_id in generators:
            main_generator_manifest = str(
                Path(str(roots["main"]))
                / dataset_id
                / generator_id
                / "generator"
                / "manifest.json"
            )
            for method in methods:
                method_id = str(method["method_id"])
                seed = method.get("seed")
                if seed != EXPECTED_METHODS[method_id]:
                    raise ValueError(f"{method_id}: seed policy mismatch")
                source_key = str(method.get("source_root"))
                expected_source_key = (
                    "corrected_infogain"
                    if method_id == "infogain_fever"
                    else (
                        "kcbwdm_v2a"
                        if method_id == "kcbwdm_linear_gate_v2"
                        else "main"
                    )
                )
                if source_key != expected_source_key:
                    raise ValueError(f"{method_id}: source root policy mismatch")
                shared = bool(method.get("shared_selection"))
                if shared != (method_id in {"no_evidence", "retrieval_topk", "bge"}):
                    raise ValueError(f"{method_id}: shared-selection policy mismatch")
                paths = _source_paths(
                    dataset_id=dataset_id,
                    generator_id=generator_id,
                    method_id=method_id,
                    seed=seed,
                    shared_selection=shared,
                    root=str(roots[source_key]),
                )
                # Generator manifests are shared from the original formal-v2 run.
                paths["generator_manifest"] = main_generator_manifest
                group_id = f"{dataset_id}__{generator_id}__{method_id}"
                group = {
                    "group_id": group_id,
                    "dataset_id": dataset_id,
                    "generator_id": generator_id,
                    "method_id": method_id,
                    "experiment_type": MATCHED_MAIN,
                    "seed": seed,
                    "split": "validation",
                    "expected_rows": expected_rows,
                    "no_evidence": method_id == "no_evidence",
                    "expected_tokenizer_name_or_path": generator_by_id[generator_id][
                        "tokenizer_name_or_path"
                    ],
                    "expected_tokenizer_revision": generator_by_id[generator_id][
                        "tokenizer_revision"
                    ],
                    **paths,
                }
                for field, value in paths.items():
                    _forbid_non_validation(value, f"{group_id}.{field}")
                groups.append(group)

    keys = {
        (group["dataset_id"], group["generator_id"], group["method_id"])
        for group in groups
    }
    expected_keys = {
        (dataset, generator, method)
        for dataset in EXPECTED_DATASETS
        for generator in EXPECTED_GENERATORS
        for method in EXPECTED_METHODS
    }
    if len(groups) != 48 or keys != expected_keys:
        raise ValueError("Analysis plan must expand to 48 unique groups")
    return plan, groups


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _recorded_path(value: Any, manifest_path: Path, label: str) -> Path:
    if not value:
        raise ValueError(f"{label} is missing")
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def require_recorded_sha(path: Path, recorded_sha: Any, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    actual = sha256_file(path)
    if recorded_sha != actual:
        raise ValueError(
            f"{label} SHA mismatch: recorded={recorded_sha!r} actual={actual!r}"
        )
    return actual


def _close(actual: Any, expected: Any, label: str, *, tolerance: float = 1e-9) -> None:
    try:
        left = float(actual)
        right = float(expected)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: actual={actual!r} expected={expected!r}") from exc
    if not math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"{label} mismatch: actual={left!r} expected={right!r}")


def validate_row_identities(group: Mapping[str, Any]) -> None:
    selections = list(read_jsonl(group["selection"]))
    predictions = list(read_jsonl(group["predictions"]))
    expected_rows = int(group["expected_rows"])
    if len(selections) != expected_rows or len(predictions) != expected_rows:
        raise ValueError(
            f"{group['group_id']}: row count mismatch selection={len(selections)} "
            f"predictions={len(predictions)} expected={expected_rows}"
        )
    selection_ids = [str(row.get("id")) for row in selections]
    prediction_ids = [str(row.get("id")) for row in predictions]
    if len(set(selection_ids)) != expected_rows:
        raise ValueError(f"{group['group_id']}: duplicate selection ID")
    if len(set(prediction_ids)) != expected_rows:
        raise ValueError(f"{group['group_id']}: duplicate prediction ID")
    if selection_ids != prediction_ids:
        raise ValueError(f"{group['group_id']}: selection/prediction IDs or order mismatch")
    for selection, prediction in zip(selections, predictions):
        row_id = selection.get("id")
        if selection.get("split") != "validation":
            raise ValueError(f"{group['group_id']}/{row_id}: split mismatch")
        if selection.get("method") != group["method_id"]:
            raise ValueError(f"{group['group_id']}/{row_id}: selection method mismatch")
        if prediction.get("method") != group["method_id"]:
            raise ValueError(f"{group['group_id']}/{row_id}: prediction method mismatch")
        if prediction.get("query") != selection.get("query"):
            raise ValueError(f"{group['group_id']}/{row_id}: query mismatch")
        if prediction.get("gold") != selection.get("label"):
            raise ValueError(f"{group['group_id']}/{row_id}: gold label mismatch")


def validate_group_sources(
    group: Mapping[str, Any],
    *,
    generator_cache: dict[str, dict[str, Any]] | None = None,
    selection_cache: dict[tuple[str, str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    selection_path = Path(str(group["selection"]))
    selection_manifest_path = Path(str(group["selection_manifest"]))
    predictions_path = Path(str(group["predictions"]))
    metrics_path = Path(str(group["metrics"]))
    evaluation_manifest_path = Path(str(group["evaluation_manifest"]))
    generator_manifest_path = Path(str(group["generator_manifest"]))

    generator_key = str(generator_manifest_path.resolve())
    if generator_cache is not None and generator_key in generator_cache:
        generator = generator_cache[generator_key]
    else:
        generator = validate_generator_manifest(
            generator_manifest_path,
            expected_dataset_id=str(group["dataset_id"]),
            expected_generator_id=str(group["generator_id"]),
        )
        if generator_cache is not None:
            generator_cache[generator_key] = generator
    generator_manifest = generator["manifest"]
    identity = generator["generator_identity"]
    if identity.get("tokenizer_name_or_path") != group[
        "expected_tokenizer_name_or_path"
    ]:
        raise ValueError(f"{group['group_id']}: tokenizer path identity mismatch")
    if identity.get("tokenizer_revision") != group["expected_tokenizer_revision"]:
        raise ValueError(f"{group['group_id']}: tokenizer revision identity mismatch")
    config_path = _recorded_path(
        generator_manifest.get("config_path"), generator_manifest_path, "generator config_path"
    )
    config = load_yaml(config_path)
    validate_generator_identity_against_config(
        generator["generator_identity"], config["generator"]
    )

    selection_key = (
        str(selection_path.resolve()),
        str(group["dataset_id"]),
        str(group["method_id"]),
    )
    if selection_cache is not None and selection_key in selection_cache:
        selection = selection_cache[selection_key]
    else:
        selection = validate_selection_provenance(
            selection_path,
            selection_manifest_path,
            formal_v2=True,
            expected_dataset_id=str(group["dataset_id"]),
            expected_method=str(group["method_id"]),
        )
        if selection_cache is not None:
            selection_cache[selection_key] = selection
    binding = selection["artifact_binding"]
    if binding.get("seed") != group["seed"]:
        raise ValueError(f"{group['group_id']}: selection seed mismatch")
    if int(selection["manifest"].get("num_rows", -1)) != int(group["expected_rows"]):
        raise ValueError(f"{group['group_id']}: selection manifest row count mismatch")
    expected_evaluation_binding = build_evaluation_binding(
        selection, generator, experiment_type=MATCHED_MAIN
    )

    evaluation_manifest = _load_json(
        evaluation_manifest_path, "evaluation manifest"
    )
    expected_manifest_fields = {
        "schema_version": EVALUATION_MANIFEST_SCHEMA_VERSION,
        "stage": "evaluation",
        "status": "completed",
        "completed": True,
        "method": group["method_id"],
        "identity_mode": "formal_v2",
        "dataset_id": group["dataset_id"],
        "evaluation_generator_id": group["generator_id"],
        "experiment_type": MATCHED_MAIN,
    }
    for field, expected in expected_manifest_fields.items():
        if evaluation_manifest.get(field) != expected:
            raise ValueError(
                f"{group['group_id']}: evaluation manifest {field} mismatch: "
                f"expected={expected!r} actual={evaluation_manifest.get(field)!r}"
            )
    if evaluation_manifest.get("artifact_binding") != expected_evaluation_binding:
        raise ValueError(f"{group['group_id']}: evaluation artifact binding mismatch")
    contract = evaluation_manifest.get("contract")
    if not isinstance(contract, dict):
        raise ValueError(f"{group['group_id']}: evaluation contract is missing")
    if evaluation_manifest.get("fingerprint") != stable_hash(contract):
        raise ValueError(f"{group['group_id']}: evaluation fingerprint mismatch")
    contract_expected = {
        "split": "validation",
        "method_override": group["method_id"],
        "no_evidence": bool(group["no_evidence"]),
        "limit": None,
    }
    for field, expected in contract_expected.items():
        if contract.get(field) != expected:
            raise ValueError(f"{group['group_id']}: evaluation contract {field} mismatch")
    max_docs = contract.get("max_docs")
    if max_docs is not None and (isinstance(max_docs, bool) or int(max_docs) < 0):
        raise ValueError(f"{group['group_id']}: invalid max_docs in evaluation contract")
    if _recorded_path(contract.get("selection_path"), evaluation_manifest_path, "selection_path") != selection_path.resolve():
        raise ValueError(f"{group['group_id']}: evaluation selection path mismatch")
    if contract.get("selection_sha256") != selection["selection_sha256"]:
        raise ValueError(f"{group['group_id']}: evaluation selection SHA mismatch")
    if contract.get("artifact_binding") != expected_evaluation_binding:
        raise ValueError(f"{group['group_id']}: contract evaluation binding mismatch")
    if evaluation_manifest.get("generator_identity") != generator["generator_identity"]:
        raise ValueError(f"{group['group_id']}: evaluation generator identity mismatch")

    if _recorded_path(evaluation_manifest.get("predictions_path"), evaluation_manifest_path, "predictions_path") != predictions_path.resolve():
        raise ValueError(f"{group['group_id']}: predictions path mismatch")
    if _recorded_path(evaluation_manifest.get("metrics_path"), evaluation_manifest_path, "metrics_path") != metrics_path.resolve():
        raise ValueError(f"{group['group_id']}: metrics path mismatch")
    require_recorded_sha(
        predictions_path,
        evaluation_manifest.get("predictions_sha256"),
        f"{group['group_id']}: predictions",
    )
    require_recorded_sha(
        metrics_path,
        evaluation_manifest.get("metrics_sha256"),
        f"{group['group_id']}: metrics",
    )
    if int(evaluation_manifest.get("num_rows", -1)) != int(group["expected_rows"]):
        raise ValueError(f"{group['group_id']}: evaluation row count mismatch")

    metrics = _load_json(metrics_path, "metrics")
    metric_expected = {
        "dataset_id": group["dataset_id"],
        "split": "validation",
        "method": group["method_id"],
        "evaluation_generator_id": group["generator_id"],
        "experiment_type": MATCHED_MAIN,
    }
    for field, expected in metric_expected.items():
        if metrics.get(field) != expected:
            raise ValueError(f"{group['group_id']}: metrics {field} mismatch")
    if int(metrics.get("num_examples", -1)) != int(group["expected_rows"]):
        raise ValueError(f"{group['group_id']}: metrics num_examples mismatch")
    validate_row_identities(group)

    fingerprint_payload = {
        "group": dict(group),
        "selection_sha256": selection["selection_sha256"],
        "selection_manifest_sha256": selection["manifest_sha256"],
        "predictions_sha256": evaluation_manifest["predictions_sha256"],
        "metrics_sha256": evaluation_manifest["metrics_sha256"],
        "evaluation_manifest_sha256": sha256_file(evaluation_manifest_path),
        "generator_manifest_sha256": generator["manifest_sha256"],
        "generator_identity_fingerprint": generator[
            "generator_identity_fingerprint"
        ],
        "max_docs": max_docs,
        "no_evidence": bool(group["no_evidence"]),
    }
    return {
        "selection": selection,
        "generator": generator,
        "generator_config_path": str(config_path),
        "evaluation_manifest": evaluation_manifest,
        "metrics": metrics,
        "max_docs": max_docs,
        "source_fingerprint": stable_hash(fingerprint_payload),
        "source_sha256": {
            "selection": selection["selection_sha256"],
            "selection_manifest": selection["manifest_sha256"],
            "predictions": evaluation_manifest["predictions_sha256"],
            "metrics": evaluation_manifest["metrics_sha256"],
            "evaluation_manifest": sha256_file(evaluation_manifest_path),
            "generator_manifest": generator["manifest_sha256"],
        },
    }


def load_tokenizer_for_identity(identity: Mapping[str, Any]) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Offline token analysis requires transformers for AutoTokenizer only"
        ) from exc
    return AutoTokenizer.from_pretrained(
        str(identity["tokenizer_name_or_path"]),
        trust_remote_code=bool(identity.get("trust_remote_code", False)),
        revision=identity.get("tokenizer_revision") or identity.get("model_revision"),
    )


def render_evidence(
    row: Mapping[str, Any], *, no_evidence: bool, max_docs: int | None
) -> tuple[list[dict[str, Any]], str]:
    docs = [] if no_evidence else list(recover_selected_docs(dict(row)))
    declared_ids = row.get("selected_doc_ids")
    recovered_ids = [doc.get("doc_id") for doc in docs]
    if not no_evidence and isinstance(declared_ids, list) and declared_ids != recovered_ids:
        raise ValueError(
            f"Selection row {row.get('id')} selected_doc_ids differ from selected_docs"
        )
    if max_docs is not None:
        docs = docs[: int(max_docs)]
    evidence_text = build_evidence_context(docs)
    if no_evidence and evidence_text:
        raise ValueError(f"Selection row {row.get('id')} rendered non-empty no_evidence")
    return docs, evidence_text


def audit_evidence_rows(
    selection_rows: Iterable[Mapping[str, Any]],
    prediction_rows: Iterable[Mapping[str, Any]],
    *,
    tokenizer: Any,
    no_evidence: bool,
    max_docs: int | None,
    expected_rows: int,
    group_id: str,
    expected_method: str | None = None,
) -> dict[str, Any]:
    selections = list(selection_rows)
    predictions = list(prediction_rows)
    if len(selections) != expected_rows or len(predictions) != expected_rows:
        raise ValueError(
            f"{group_id}: row count mismatch selection={len(selections)} "
            f"predictions={len(predictions)} expected={expected_rows}"
        )
    selection_ids = [str(row.get("id")) for row in selections]
    prediction_ids = [str(row.get("id")) for row in predictions]
    if len(set(selection_ids)) != expected_rows:
        raise ValueError(f"{group_id}: duplicate selection ID")
    if len(set(prediction_ids)) != expected_rows:
        raise ValueError(f"{group_id}: duplicate prediction ID")
    if selection_ids != prediction_ids:
        raise ValueError(f"{group_id}: selection/prediction IDs or order mismatch")

    total_docs = 0
    total_chars = 0
    total_tokens = 0
    histogram: Counter[str] = Counter()
    for selection, prediction in zip(selections, predictions):
        row_id = str(selection.get("id"))
        if selection.get("split") != "validation":
            raise ValueError(f"{group_id}/{row_id}: selection split is not validation")
        if expected_method is not None and selection.get("method") != expected_method:
            raise ValueError(f"{group_id}/{row_id}: selection method mismatch")
        if expected_method is not None and prediction.get("method") != expected_method:
            raise ValueError(f"{group_id}/{row_id}: prediction method mismatch")
        if prediction.get("query") != selection.get("query"):
            raise ValueError(f"{group_id}/{row_id}: query mismatch")
        if prediction.get("gold") != selection.get("label"):
            raise ValueError(f"{group_id}/{row_id}: gold label mismatch")
        docs, evidence_text = render_evidence(
            selection, no_evidence=no_evidence, max_docs=max_docs
        )
        if len(docs) > 4:
            raise ValueError(f"{group_id}/{row_id}: formal evidence exceeds four docs")
        doc_ids = [doc.get("doc_id") for doc in docs]
        if prediction.get("selected_doc_ids") != doc_ids:
            raise ValueError(f"{group_id}/{row_id}: selected_doc_ids mismatch")
        if int(prediction.get("num_docs", -1)) != len(docs):
            raise ValueError(f"{group_id}/{row_id}: prediction num_docs mismatch")
        token_count = (
            len(tokenizer.encode(evidence_text, add_special_tokens=False))
            if evidence_text
            else 0
        )
        total_docs += len(docs)
        total_chars += len(evidence_text)
        total_tokens += token_count
        histogram[str(len(docs))] += 1
    for value in range(5):
        histogram.setdefault(str(value), 0)
    return {
        "num_rows": expected_rows,
        "avg_num_docs": total_docs / expected_rows,
        "avg_evidence_chars": total_chars / expected_rows,
        "avg_evidence_tokens": total_tokens / expected_rows,
        "total_evidence_tokens": total_tokens,
        "num_docs_histogram": dict(
            sorted(histogram.items(), key=lambda item: int(item[0]))
        ),
        "evidence_chars_basis": "rendered_evidence_block_before_prompt_template",
        "evidence_tokens_basis": TOKEN_BASIS,
    }


def validate_recomputed_metrics(
    group: Mapping[str, Any], metrics: Mapping[str, Any], recomputed: Mapping[str, Any]
) -> None:
    group_id = str(group["group_id"])
    _close(
        recomputed["avg_num_docs"], metrics.get("avg_num_docs"), f"{group_id}: avg_num_docs"
    )
    _close(
        recomputed["avg_evidence_chars"],
        metrics.get("avg_evidence_chars"),
        f"{group_id}: avg_evidence_chars",
    )
    metric_hist = metrics.get("num_docs", {}).get("histogram")
    if metric_hist != recomputed["num_docs_histogram"]:
        raise ValueError(f"{group_id}: num_docs histogram mismatch")
    existing_tokens = metrics.get("avg_evidence_tokens")
    if group["method_id"] == "infogain_fever" and existing_tokens is None:
        raise ValueError(f"{group_id}: corrected InfoGain metrics lack avg_evidence_tokens")
    if existing_tokens is not None:
        _close(
            recomputed["avg_evidence_tokens"],
            existing_tokens,
            f"{group_id}: avg_evidence_tokens",
            tolerance=1e-9,
        )
        if metrics.get("evidence_tokens_basis") != TOKEN_BASIS:
            raise ValueError(f"{group_id}: evidence_tokens_basis mismatch")


def analyze_group(
    group: Mapping[str, Any], source: Mapping[str, Any], tokenizer: Any
) -> dict[str, Any]:
    recomputed = audit_evidence_rows(
        read_jsonl(group["selection"]),
        read_jsonl(group["predictions"]),
        tokenizer=tokenizer,
        no_evidence=bool(group["no_evidence"]),
        max_docs=source["max_docs"],
        expected_rows=int(group["expected_rows"]),
        group_id=str(group["group_id"]),
        expected_method=str(group["method_id"]),
    )
    metrics = source["metrics"]
    validate_recomputed_metrics(group, metrics, recomputed)
    generator = source["generator"]
    tokenizer_identity = {
        key: generator["generator_identity"].get(key)
        for key in (
            "generator_id",
            "tokenizer_name_or_path",
            "tokenizer_revision",
            "tokenizer_sha256",
            "trust_remote_code",
        )
    }
    return {
        "status": "PASS",
        "group_id": group["group_id"],
        "dataset_id": group["dataset_id"],
        "generator_id": group["generator_id"],
        "method_id": group["method_id"],
        "experiment_type": MATCHED_MAIN,
        "seed": group["seed"],
        "split": "validation",
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "num_examples": metrics["num_examples"],
        **recomputed,
        "selection_path": group["selection"],
        "selection_manifest_path": group["selection_manifest"],
        "predictions_path": group["predictions"],
        "metrics_path": group["metrics"],
        "evaluation_manifest_path": group["evaluation_manifest"],
        "generator_manifest_path": group["generator_manifest"],
        "source_sha256": source["source_sha256"],
        "source_fingerprint": source["source_fingerprint"],
        "tokenizer_identity": tokenizer_identity,
        "tokenizer_identity_fingerprint": stable_hash(tokenizer_identity),
        "generator_identity_fingerprint": generator[
            "generator_identity_fingerprint"
        ],
        "max_docs": source["max_docs"],
        "no_evidence": bool(group["no_evidence"]),
    }


def _cache_path(output_dir: Path, group_id: str) -> Path:
    return output_dir / "group_audits" / f"{group_id}.json"


def analyze_or_resume_group(
    group: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    tokenizer: Any | None,
    output_dir: Path,
    resume: bool,
) -> tuple[dict[str, Any], bool]:
    cache = _cache_path(output_dir, str(group["group_id"]))
    if cache.exists():
        if not resume:
            raise FileExistsError(f"Group audit exists; use --resume: {cache}")
        payload = _load_json(cache, "group audit")
        if (
            payload.get("schema_version") != AUDIT_SCHEMA
            or payload.get("status") != "PASS"
            or payload.get("source_fingerprint") != source["source_fingerprint"]
        ):
            raise ValueError(
                f"Cannot resume identity-mismatched group audit: {group['group_id']}"
            )
        return dict(payload["result"]), True
    if tokenizer is None:
        raise ValueError(f"Tokenizer is required for uncached group {group['group_id']}")
    result = analyze_group(group, source, tokenizer)
    atomic_write_json(
        cache,
        {
            "schema_version": AUDIT_SCHEMA,
            "status": "PASS",
            "source_fingerprint": source["source_fingerprint"],
            "result": result,
        },
    )
    return result, False


def validate_result_rows(
    rows: list[Mapping[str, Any]], *, expected_count: int
) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} result rows, got {len(rows)}")
    keys = [
        (row["dataset_id"], row["generator_id"], row["method_id"])
        for row in rows
    ]
    if len(set(keys)) != expected_count:
        raise ValueError("Result rows are not unique by dataset/generator/method")
    if expected_count == 48:
        expected = {
            (dataset, generator, method)
            for dataset in EXPECTED_DATASETS
            for generator in EXPECTED_GENERATORS
            for method in EXPECTED_METHODS
        }
        if set(keys) != expected:
            raise ValueError("Result rows do not cover the frozen 48-group matrix")


def _flat_result(row: Mapping[str, Any]) -> dict[str, Any]:
    histogram = row["num_docs_histogram"]
    sources = row["source_sha256"]
    tokenizer = row["tokenizer_identity"]
    return {
        "dataset_id": row["dataset_id"],
        "generator_id": row["generator_id"],
        "method_id": row["method_id"],
        "experiment_type": row["experiment_type"],
        "seed": row["seed"],
        "split": row["split"],
        "accuracy": row["accuracy"],
        "macro_f1": row["macro_f1"],
        "num_examples": row["num_examples"],
        "avg_num_docs": row["avg_num_docs"],
        "avg_evidence_chars": row["avg_evidence_chars"],
        "avg_evidence_tokens": row["avg_evidence_tokens"],
        "total_evidence_tokens": row["total_evidence_tokens"],
        "evidence_tokens_basis": row["evidence_tokens_basis"],
        **{f"num_docs_{value}": histogram.get(str(value), 0) for value in range(5)},
        "selection_path": row["selection_path"],
        "selection_sha256": sources["selection"],
        "selection_manifest_path": row["selection_manifest_path"],
        "selection_manifest_sha256": sources["selection_manifest"],
        "predictions_path": row["predictions_path"],
        "predictions_sha256": sources["predictions"],
        "metrics_path": row["metrics_path"],
        "metrics_sha256": sources["metrics"],
        "evaluation_manifest_path": row["evaluation_manifest_path"],
        "evaluation_manifest_sha256": sources["evaluation_manifest"],
        "generator_manifest_path": row["generator_manifest_path"],
        "generator_manifest_sha256": sources["generator_manifest"],
        "tokenizer_name_or_path": tokenizer["tokenizer_name_or_path"],
        "tokenizer_revision": tokenizer["tokenizer_revision"],
        "tokenizer_sha256": tokenizer["tokenizer_sha256"],
        "tokenizer_identity_fingerprint": row["tokenizer_identity_fingerprint"],
        "generator_identity_fingerprint": row["generator_identity_fingerprint"],
    }


def add_reduction_ratios(rows: list[dict[str, Any]]) -> None:
    by_key = {
        (row["dataset_id"], row["generator_id"], row["method_id"]): row
        for row in rows
    }
    for row in rows:
        for baseline, column in (
            ("retrieval_topk", "token_reduction_vs_rag"),
            ("bge", "token_reduction_vs_bge"),
            ("infogain_fever", "token_reduction_vs_infogain"),
        ):
            row[column] = None
            if row["method_id"] != "kcbwdm_linear_gate_v2":
                continue
            baseline_tokens = float(
                by_key[(row["dataset_id"], row["generator_id"], baseline)][
                    "avg_evidence_tokens"
                ]
            )
            if baseline_tokens:
                row[column] = 1.0 - float(row["avg_evidence_tokens"]) / baseline_tokens


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    header = "| " + " | ".join(fields) + " |"
    rule = "|" + "|".join("---" for _ in fields) + "|"
    body = [
        "| " + " | ".join(str(row.get(field, "")) for field in fields) + " |"
        for row in rows
    ]
    return "\n".join((header, rule, *body)) + "\n"


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def write_formal_outputs(output_dir: Path, results: list[dict[str, Any]]) -> None:
    validate_result_rows(results, expected_count=48)
    flat = [_flat_result(row) for row in results]
    add_reduction_ratios(flat)
    _write_csv(output_dir / "main_results_48.csv", flat)
    compact_fields = [
        "dataset_id",
        "generator_id",
        "method_id",
        "accuracy",
        "macro_f1",
        "avg_num_docs",
        "avg_evidence_tokens",
        "total_evidence_tokens",
    ]
    (output_dir / "main_results_48.md").write_text(
        _markdown_table(flat, compact_fields), encoding="utf-8"
    )
    latex_lines = []
    for row in flat:
        latex_lines.append(
            " & ".join(
                (
                    _latex_escape(row["dataset_id"]),
                    _latex_escape(row["generator_id"]),
                    _latex_escape(row["method_id"]),
                    f"{100.0 * float(row['accuracy']):.2f}",
                    f"{100.0 * float(row['macro_f1']):.2f}",
                    f"{float(row['avg_evidence_tokens']):.2f}",
                )
            )
            + r" \\"
        )
    (output_dir / "main_results_48.tex").write_text(
        "\n".join(latex_lines) + "\n", encoding="utf-8"
    )
    plot_fields = [
        "dataset_id",
        "generator_id",
        "method_id",
        "accuracy",
        "macro_f1",
        "avg_evidence_tokens",
        "total_evidence_tokens",
        "token_reduction_vs_rag",
        "token_reduction_vs_bge",
        "token_reduction_vs_infogain",
    ]
    _write_csv(
        output_dir / "accuracy_token_plot_data.csv",
        [{field: row.get(field) for field in plot_fields} for row in flat],
    )


def write_token_audit(
    output_dir: Path,
    *,
    mode: str,
    status: str,
    results: list[dict[str, Any]],
    error: str | None = None,
) -> None:
    payload = {
        "schema_version": AUDIT_SCHEMA,
        "status": status,
        "mode": mode,
        "num_groups": len(results),
        "evidence_tokens_basis": TOKEN_BASIS,
        "groups": results,
        "error": error,
    }
    atomic_write_json(output_dir / "token_audit.json", payload)
    lines = [
        "# Evidence Token Audit",
        "",
        f"- Status: `{status}`",
        f"- Mode: `{mode}`",
        f"- Groups: `{len(results)}`",
        f"- Basis: `{TOKEN_BASIS}`",
    ]
    if error:
        lines.extend(("", "## Error", "", error))
    if results:
        lines.extend(
            (
                "",
                "## Groups",
                "",
                _markdown_table(
                    [_flat_result(row) for row in results],
                    [
                        "dataset_id",
                        "generator_id",
                        "method_id",
                        "avg_num_docs",
                        "avg_evidence_chars",
                        "avg_evidence_tokens",
                    ],
                ).rstrip(),
            )
        )
    (output_dir / "token_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _ensure_output_safety(output_dir: Path, *, resume: bool) -> None:
    normalized = str(output_dir).replace("\\", "/")
    protected = (
        EXPECTED_ROOTS["main"],
        EXPECTED_ROOTS["corrected_infogain"],
        EXPECTED_ROOTS["fever_kcbwdm_v2a"],
        EXPECTED_ROOTS["fm2_kcbwdm_v2a"],
    )
    if any(normalized == root or normalized.startswith(root + "/") for root in protected):
        raise ValueError("Analysis output must not be inside a formal experiment root")
    existing = [output_dir / name for name in MANAGED_OUTPUTS if (output_dir / name).exists()]
    existing.extend((output_dir / "group_audits").glob("*.json") if (output_dir / "group_audits").is_dir() else [])
    if existing and not resume:
        raise FileExistsError(
            "Analysis outputs already exist; use --resume: "
            + ", ".join(str(path) for path in existing[:5])
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _dry_run(groups: list[dict[str, Any]]) -> None:
    generator_cache: dict[str, dict[str, Any]] = {}
    selection_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    for index, group in enumerate(groups, start=1):
        validate_group_sources(
            group,
            generator_cache=generator_cache,
            selection_cache=selection_cache,
        )
        print(f"[evidence_tokens][dry-run] {index}/48 PASS {group['group_id']}")


def run_analysis(
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
    groups: list[dict[str, Any]],
    output_dir: Path,
    mode: str,
    resume: bool,
    tokenizer_loader: Callable[[Mapping[str, Any]], Any] = load_tokenizer_for_identity,
) -> list[dict[str, Any]]:
    _ensure_output_safety(output_dir, resume=resume)
    selected = (
        [group for group in groups if group["method_id"] == "infogain_fever"]
        if mode == "validate-infogain"
        else groups
    )
    expected_count = 8 if mode == "validate-infogain" else 48
    results: list[dict[str, Any]] = []
    tokenizer_cache: dict[str, Any] = {}
    generator_cache: dict[str, dict[str, Any]] = {}
    selection_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    reused = 0
    try:
        for index, group in enumerate(selected, start=1):
            source = validate_group_sources(
                group,
                generator_cache=generator_cache,
                selection_cache=selection_cache,
            )
            cache_path = _cache_path(output_dir, group["group_id"])
            tokenizer = None
            if not (resume and cache_path.exists()):
                identity = source["generator"]["generator_identity"]
                fingerprint = stable_hash(
                    {
                        key: identity.get(key)
                        for key in (
                            "tokenizer_name_or_path",
                            "tokenizer_revision",
                            "tokenizer_sha256",
                            "trust_remote_code",
                        )
                    }
                )
                if fingerprint not in tokenizer_cache:
                    tokenizer_cache[fingerprint] = tokenizer_loader(identity)
                tokenizer = tokenizer_cache[fingerprint]
            result, was_reused = analyze_or_resume_group(
                group,
                source,
                tokenizer=tokenizer,
                output_dir=output_dir,
                resume=resume,
            )
            reused += int(was_reused)
            results.append(result)
            print(
                f"[evidence_tokens][{mode}] {index}/{expected_count} "
                f"{'REUSED' if was_reused else 'PASS'} {group['group_id']}"
            )
        validate_result_rows(results, expected_count=expected_count)
        if mode == "all":
            write_formal_outputs(output_dir, results)
        write_token_audit(
            output_dir, mode=mode, status="PASS", results=results
        )
        outputs = {
            name: sha256_file(output_dir / name)
            for name in MANAGED_OUTPUTS
            if name != "analysis_manifest.json" and (output_dir / name).is_file()
        }
        atomic_write_json(
            output_dir / "analysis_manifest.json",
            {
                "schema_version": ANALYSIS_SCHEMA,
                "status": "completed",
                "completed": True,
                "mode": mode,
                "plan_path": str(plan_path.resolve()),
                "plan_sha256": sha256_file(plan_path),
                "plan_fingerprint": stable_hash(plan),
                "num_groups": len(results),
                "reused_groups": reused,
                "evidence_tokens_basis": TOKEN_BASIS,
                "outputs": outputs,
                "git": git_state(PROJECT_ROOT),
                "completed_at": utc_now(),
            },
        )
        return results
    except BaseException as exc:
        write_token_audit(
            output_dir,
            mode=mode,
            status="FAIL",
            results=results,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plan_path = Path(args.plan)
    plan, groups = load_plan(plan_path)
    if args.dry_run:
        _dry_run(groups)
        print("[evidence_tokens] dry-run=PASS groups=48 tokenizer_loaded=false")
        return
    mode = "validate-infogain" if args.validate_infogain else "all"
    output_dir = Path(args.output_dir or plan["analysis_output_root"])
    run_analysis(
        plan_path=plan_path,
        plan=plan,
        groups=groups,
        output_dir=output_dir,
        mode=mode,
        resume=args.resume,
    )
    print(f"[evidence_tokens] mode={mode} status=PASS output={output_dir}")


if __name__ == "__main__":
    main()

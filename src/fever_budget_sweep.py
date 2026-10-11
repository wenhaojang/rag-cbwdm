"""Strict FEVER document-budget sweep planning and artifact materialization.

The module deliberately reuses formal-v2 manifests and bindings.  It never
trains a model and it never mutates a source formal artifact.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from src.artifact_binding import (
    BUDGET_FRONTIER,
    EVALUATION_MANIFEST_SCHEMA_VERSION,
    MATCHED_MAIN,
    build_evaluation_binding,
    validate_generator_manifest,
    validate_selection_provenance,
)
from src.baselines.bge_reranker import make_bge_selection
from src.baselines.common import (
    build_selection_contract,
    publish_selection,
)
from src.baselines.infogain_protocol import (
    BUDGET_PROTOCOL_FAMILY_ID,
    budget_protocol_id,
    rank_and_select,
    stop_reason_for_selection,
)
from src.io_utils import load_yaml, read_jsonl
from src.run_manifest import atomic_write_json, sha256_file, stable_hash, utc_now
from src.selection_schema import make_selection_row, normalize_selected_doc, validate_selection_row


CONFIG_SCHEMA_VERSION = "rag_cbwdm_fever_budget_sweep_config.v1"
PLAN_SCHEMA_VERSION = "rag_cbwdm_fever_budget_sweep_plan.v1"
PREFLIGHT_SCHEMA_VERSION = "rag_cbwdm_fever_budget_sweep_preflight.v1"
AUDIT_SCHEMA_VERSION = "rag_cbwdm_fever_budget_sweep_audit.v1"
JOB_PLAN_SCHEMA_VERSION = "rag_cbwdm_fever_budget_sweep_jobs.v1"
ALIAS_SCHEMA_VERSION = "rag_cbwdm_fever_budget_sweep_alias.v1"
SUMMARY_SCHEMA_VERSION = "rag_cbwdm_fever_budget_sweep_summary.v1"
TOKEN_BASIS = "generator_tokenizer_rendered_evidence_block_before_prompt_truncation"
NATIVE_TOKEN_AUDIT_SCHEMA_VERSION = "rag_cbwdm_evidence_token_audit.v1"
NATIVE_TOKEN_ANALYSIS_SCHEMA_VERSION = "rag_cbwdm_evidence_token_analysis.v1"
NATIVE_TOKEN_HISTOGRAM_BASIS = "observed_num_docs_values_only"
NATIVE_TOKEN_HISTOGRAM_COMPARISON = "zero_bins_elided_nonzero_counts_exact_v1"
NATIVE_TOKEN_DATASETS = {
    "fever_binary_v2": 5000,
    "fm2_official_closed_page_v1": 1169,
}
NATIVE_TOKEN_GENERATORS = (
    "qwen2.5-0.5b-instruct",
    "qwen2.5-1.5b-instruct",
    "qwen2.5-7b-instruct",
    "mistral-7b-instruct-v0.3",
)
NATIVE_TOKEN_METHODS = (
    "no_evidence",
    "retrieval_topk",
    "bge",
    "infogain_fever",
    "rag_cbwdm_signed_v1",
    "kcbwdm_linear_gate_v2",
)

DATASET_ID = "fever_binary_v2"
SPLIT = "validation"
EXPECTED_ROWS = 5000
SEED = 13
BUDGETS = (2, 3, 4)
GENERATORS = ("qwen2.5-1.5b-instruct", "qwen2.5-7b-instruct")
METHODS = (
    "retrieval_topk",
    "bge",
    "infogain_fever",
    "rag_cbwdm_signed_v1",
    "kcbwdm_linear_gate_v2",
)
STRATEGIES = {
    "retrieval_topk": "retrieval_prefix",
    "bge": "reranker_prefix",
    "infogain_fever": "infogain_budget_protocol",
    "rag_cbwdm_signed_v1": "causal_prefix_replay",
    "kcbwdm_linear_gate_v2": "causal_prefix_replay",
}
CONDITIONED_METHODS = {
    "infogain_fever",
    "rag_cbwdm_signed_v1",
    "kcbwdm_linear_gate_v2",
}


class BudgetSweepError(ValueError):
    """Raised when a budget-sweep identity or artifact fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BudgetSweepError(message)


def _load_json(path: str | Path, label: str) -> dict[str, Any]:
    target = Path(path)
    _require(target.is_file(), f"Missing {label}: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BudgetSweepError(f"Invalid {label} {target}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} must be a JSON object: {target}")
    return payload


def _rows(path: str | Path) -> list[dict[str, Any]]:
    try:
        rows = list(read_jsonl(Path(path)))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise BudgetSweepError(f"Invalid JSONL {path}: {exc}") from exc
    _require(all(isinstance(row, dict) for row in rows), f"JSONL rows must be objects: {path}")
    return rows


def _render(template: str, generator_id: str) -> str:
    try:
        return str(template).format(generator_id=generator_id)
    except (KeyError, ValueError) as exc:
        raise BudgetSweepError(f"Invalid generator path template {template!r}: {exc}") from exc


def _join(root: str, *parts: str) -> str:
    if str(root).startswith("/"):
        return str(PurePosixPath(root).joinpath(*parts))
    return str(Path(root).joinpath(*parts))


def _sidecar(path: str) -> str:
    if str(path).startswith("/"):
        target = PurePosixPath(path)
        return str(target.with_suffix(".manifest.json"))
    return str(Path(path).with_suffix(".manifest.json"))


def _normalized_path(value: str) -> str:
    return str(value).replace("\\", "/").rstrip("/")


def assert_isolated_artifact_root(root: str, protected_roots: Sequence[str]) -> None:
    candidate = _normalized_path(root)
    _require(bool(candidate), "artifact_root must not be empty")
    for protected in protected_roots:
        frozen = _normalized_path(protected)
        _require(
            candidate != frozen and not candidate.startswith(frozen + "/"),
            f"Budget artifact root overlaps protected formal root: {protected}",
        )


def load_budget_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    _require(isinstance(config, dict), "Budget config must be a YAML object")
    _require(config.get("schema_version") == CONFIG_SCHEMA_VERSION, "Config schema mismatch")
    _require(config.get("dataset_id") == DATASET_ID, "Budget sweep is FEVER binary v2 only")
    _require(config.get("split") == SPLIT, "Budget sweep is validation-only")
    _require(config.get("held_out") is False, "Budget sweep must set held_out=false")
    _require(config.get("expected_rows") == EXPECTED_ROWS, "FEVER validation must contain 5000 rows")
    _require(config.get("seed") == SEED, "Budget sweep seed must remain 13")
    _require(tuple(config.get("budgets", [])) == BUDGETS, "Budgets must be [2, 3, 4]")
    generators = config.get("generators")
    methods = config.get("methods")
    _require(isinstance(generators, list), "generators must be a list")
    _require(isinstance(methods, list), "methods must be a list")
    _require(tuple(item.get("generator_id") for item in generators) == GENERATORS, "Generator set/order mismatch")
    _require(tuple(item.get("method_id") for item in methods) == METHODS, "Method set/order mismatch")
    for generator in generators:
        for field in ("generator_id", "config", "generator_manifest"):
            _require(bool(generator.get(field)), f"Generator lacks {field}")
    for method in methods:
        method_id = str(method["method_id"])
        _require(method.get("strategy") == STRATEGIES[method_id], f"Strategy mismatch for {method_id}")
        _require(
            bool(method.get("selection_template")) and bool(method.get("selection_manifest_template")),
            f"Method {method_id} lacks source selection paths",
        )
        _require(bool(method.get("native_metrics_template")), f"Method {method_id} lacks native metrics path")
        if method_id == "bge":
            _require(bool(method.get("score_cache")), "BGE lacks score_cache")
            _require(bool(method.get("score_cache_manifest")), "BGE lacks score_cache_manifest")
    for field in (
        "artifact_root",
        "server_python",
        "server_project_root",
        "retrieval",
        "retrieval_manifest",
        "expected_retrieval_sha256",
        "native_token_results",
        "native_token_audit",
        "protected_roots",
    ):
        _require(bool(config.get(field)), f"Config lacks {field}")
    _require(isinstance(config["protected_roots"], list), "protected_roots must be a list")
    assert_isolated_artifact_root(str(config["artifact_root"]), list(config["protected_roots"]))
    return config


def _method_map(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["method_id"]): dict(item) for item in config["methods"]}


def _generator_map(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["generator_id"]): dict(item) for item in config["generators"]}


def generator_manifest_path(
    config: Mapping[str, Any], method_id: str, generator_id: str
) -> str:
    method = _method_map(config)[method_id]
    template = method.get("generator_manifest_template")
    if template:
        return _render(str(template), generator_id)
    return str(_generator_map(config)[generator_id]["generator_manifest"])


def source_selection_paths(method: Mapping[str, Any], generator_id: str) -> tuple[str, str]:
    return (
        _render(str(method["selection_template"]), generator_id),
        _render(str(method["selection_manifest_template"]), generator_id),
    )


def native_evaluation_paths(method: Mapping[str, Any], generator_id: str) -> dict[str, str]:
    metrics = _render(str(method["native_metrics_template"]), generator_id)
    spec = {
        "metrics": metrics,
        "predictions": _render(
            str(method.get("native_predictions_template") or str(Path(metrics).with_name("predictions.jsonl"))),
            generator_id,
        ),
        "manifest": _render(
            str(method.get("native_evaluation_manifest_template") or _sidecar(metrics)),
            generator_id,
        ),
    }
    return spec


def new_selection_paths(artifact_root: str, generator_id: str, method_id: str, budget: int) -> tuple[str, str]:
    if method_id in CONDITIONED_METHODS:
        output = _join(
            artifact_root,
            DATASET_ID,
            generator_id,
            method_id,
            f"k{budget}",
            "selection",
            "validation.jsonl",
        )
    else:
        output = _join(artifact_root, "shared", method_id, f"k{budget}", "selection.jsonl")
    return output, _sidecar(output)


def point_output_paths(artifact_root: str, generator_id: str, method_id: str, budget: int) -> dict[str, str]:
    base = _join(
        artifact_root,
        DATASET_ID,
        generator_id,
        "evaluation",
        method_id,
        "budget_frontier",
        f"k{budget}",
        SPLIT,
    )
    metrics = _join(base, "metrics.json")
    return {
        "directory": base,
        "predictions": _join(base, "predictions.jsonl"),
        "metrics": metrics,
        "manifest": _sidecar(metrics),
        "alias_manifest": _join(base, "alias.manifest.json"),
    }


def build_plan(config: Mapping[str, Any], *, artifact_root: str | None = None) -> dict[str, Any]:
    root = str(artifact_root or config["artifact_root"])
    assert_isolated_artifact_root(root, list(config["protected_roots"]))
    methods = _method_map(config)
    points: list[dict[str, Any]] = []
    selection_tasks: dict[str, dict[str, Any]] = {}
    for generator_id in GENERATORS:
        for method_id in METHODS:
            method = methods[method_id]
            source_selection, source_manifest = source_selection_paths(method, generator_id)
            native = native_evaluation_paths(method, generator_id)
            for budget in BUDGETS:
                if budget == 4:
                    selection, manifest = source_selection, source_manifest
                    selection_source = "native_k4"
                else:
                    selection, manifest = new_selection_paths(root, generator_id, method_id, budget)
                    selection_source = "materialized"
                    key_generator = generator_id if method_id in CONDITIONED_METHODS else "shared"
                    task_id = f"{key_generator}.{method_id}.k{budget}"
                    selection_tasks.setdefault(
                        task_id,
                        {
                            "task_id": task_id,
                            "generator_id": None if key_generator == "shared" else generator_id,
                            "method_id": method_id,
                            "strategy": method["strategy"],
                            "budget_k": budget,
                            "source_selection": source_selection,
                            "source_selection_manifest": source_manifest,
                            "output_selection": selection,
                            "output_selection_manifest": manifest,
                        },
                    )
                outputs = point_output_paths(root, generator_id, method_id, budget)
                points.append(
                    {
                        "point_id": f"{generator_id}.{method_id}.k{budget}",
                        "dataset_id": DATASET_ID,
                        "generator_id": generator_id,
                        "method_id": method_id,
                        "strategy": method["strategy"],
                        "budget_k": budget,
                        "selection_source": selection_source,
                        "evaluation_disposition": (
                            "native_k4_alias_pending_provenance"
                            if budget == 4
                            else "dynamic_after_evidence_vector_audit"
                        ),
                        "selection_path": selection,
                        "selection_manifest_path": manifest,
                        "native_evaluation": native if budget == 4 else None,
                        "output": outputs,
                    }
                )
    _require(len(points) == 30, "Budget plan must contain exactly 30 points")
    _require(len({point["point_id"] for point in points}) == 30, "Budget point IDs must be unique")
    _require(len(selection_tasks) == 16, "Budget plan must contain exactly 16 new selections")
    payload: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "split": SPLIT,
        "expected_rows": EXPECTED_ROWS,
        "seed": SEED,
        "budgets": list(BUDGETS),
        "generators": list(GENERATORS),
        "methods": list(METHODS),
        "artifact_root": root,
        "config_fingerprint": stable_hash(config),
        "selection_tasks": list(selection_tasks.values()),
        "points": points,
    }
    payload["plan_fingerprint"] = stable_hash(payload)
    return payload


def render_evidence(docs: Sequence[Mapping[str, Any]]) -> str:
    blocks: list[str] = []
    for index, doc in enumerate(docs, start=1):
        title = doc.get("title")
        text = doc.get("text", "")
        blocks.append(f"[{index}] Title: {title}\n{text}" if title else f"[{index}] {text}")
    return "\n\n".join(blocks)


def _validate_retrieval(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    retrieval = Path(str(config["retrieval"])).resolve()
    manifest_path = Path(str(config["retrieval_manifest"])).resolve()
    manifest = _load_json(manifest_path, "retrieval manifest")
    _require(manifest.get("schema_version") == "rag_cbwdm_retrieval.v1", "Retrieval manifest schema mismatch")
    _require(manifest.get("status", "completed") == "completed", "Retrieval manifest is not completed")
    _require(manifest.get("completed") is True, "Retrieval manifest is incomplete")
    _require(manifest.get("split") == SPLIT, "Retrieval manifest split mismatch")
    _require(int(manifest.get("top_n", 0)) >= 4, "Retrieval top_n is smaller than four")
    _require(manifest.get("output_sha256") == sha256_file(retrieval), "Retrieval SHA mismatch")
    _require(
        manifest.get("output_sha256") == config.get("expected_retrieval_sha256"),
        "Retrieval SHA differs from the frozen FEVER candidate pool",
    )
    contract = {
        key: manifest.get(key)
        for key in (
            "config_path",
            "config_sha256",
            "split",
            "query_input_sha256",
            "index_fingerprint",
            "backend",
            "top_n",
            "limit",
            "gold_evidence_key_policy",
        )
    }
    _require(manifest.get("fingerprint") == stable_hash(contract), "Retrieval manifest fingerprint mismatch")
    config_path = Path(str(manifest.get("config_path"))).resolve()
    query_path = Path(str(manifest.get("query_input_path"))).resolve()
    _require(sha256_file(config_path) == manifest.get("config_sha256"), "Retrieval config SHA mismatch")
    _require(sha256_file(query_path) == manifest.get("query_input_sha256"), "Retrieval query SHA mismatch")
    recorded_rows = manifest.get("num_output_rows", manifest.get("num_rows"))
    _require(recorded_rows == EXPECTED_ROWS, "Retrieval manifest row count mismatch")
    rows = _rows(retrieval)
    _validate_ids(rows, label="retrieval")
    for row in rows:
        _require(row.get("split") == SPLIT, f"Retrieval split mismatch for {row.get('id')}")
        candidates = row.get("candidates")
        _require(isinstance(candidates, list), f"Retrieval candidates missing for {row.get('id')}")
        _require(len(candidates) >= 4, f"Retrieval has fewer than four candidates for {row.get('id')}")
        ids = [str(item.get("doc_id")) for item in candidates]
        _require(len(ids) == len(set(ids)), f"Duplicate retrieval doc IDs for {row.get('id')}")
    return rows, manifest


def _validate_ids(rows: Sequence[Mapping[str, Any]], *, label: str) -> tuple[str, ...]:
    _require(len(rows) == EXPECTED_ROWS, f"{label} must contain {EXPECTED_ROWS} rows")
    ids = tuple(str(row.get("id")) for row in rows)
    _require(all(value and value != "None" for value in ids), f"{label} contains an empty ID")
    _require(len(ids) == len(set(ids)), f"{label} contains duplicate IDs")
    return ids


def _validate_selected_content(selection_row: Mapping[str, Any], retrieval_row: Mapping[str, Any]) -> None:
    candidate_map = {
        str(candidate.get("doc_id")): candidate
        for candidate in retrieval_row.get("candidates", [])
    }
    for doc in selection_row.get("selected_docs", []):
        doc_id = str(doc.get("doc_id"))
        _require(doc_id in candidate_map, f"Selected doc {doc_id} is outside retrieval pool")
        candidate = candidate_map[doc_id]
        _require(doc.get("title") == candidate.get("title"), f"Selected title mismatch for {doc_id}")
        _require(doc.get("text") == candidate.get("text"), f"Selected text mismatch for {doc_id}")


def _validate_selection_rows(
    rows: Sequence[dict[str, Any]],
    retrieval_rows: Sequence[dict[str, Any]],
    *,
    method_id: str,
    label: str,
) -> tuple[str, ...]:
    ids = _validate_ids(rows, label=label)
    retrieval_ids = _validate_ids(retrieval_rows, label="retrieval")
    _require(ids == retrieval_ids, f"{label} IDs/order differ from retrieval")
    for row, retrieval_row in zip(rows, retrieval_rows):
        try:
            validate_selection_row(row)
        except ValueError as exc:
            raise BudgetSweepError(f"Invalid {label} row {row.get('id')}: {exc}") from exc
        _require(row.get("method") == method_id, f"{label} method mismatch")
        _require(row.get("split") == SPLIT, f"{label} split mismatch")
        _require(row.get("query") == retrieval_row.get("query"), f"{label} query mismatch")
        _require(
            row.get("label", row.get("gold"))
            == retrieval_row.get("label", retrieval_row.get("gold")),
            f"{label} label alignment mismatch",
        )
        _require(row.get("uses_gold_at_test") is False, f"{label} uses validation gold at test time")
        metadata = row.get("selection_metadata")
        _require(isinstance(metadata, dict), f"{label} lacks selection_metadata")
        _require(
            metadata.get("uses_gold_at_test", metadata.get("uses_gold_at_inference", False)) is False,
            f"{label} metadata indicates gold use",
        )
        _validate_selected_content(row, retrieval_row)
    return ids


def _parameters(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    contract = manifest.get("contract")
    _require(isinstance(contract, dict), "Selection manifest lacks contract")
    parameters = contract.get("parameters")
    _require(isinstance(parameters, dict), "Selection contract lacks parameters")
    return parameters


def _validate_contract_inputs(manifest: Mapping[str, Any], *, label: str) -> None:
    contract = manifest.get("contract")
    _require(isinstance(contract, dict), f"{label} lacks a contract")
    inputs = contract.get("inputs")
    _require(isinstance(inputs, dict) and inputs, f"{label} lacks input provenance")
    for name, record in inputs.items():
        _require(isinstance(record, dict), f"{label} input {name} is invalid")
        path = record.get("path")
        recorded_sha = record.get("sha256")
        _require(bool(path) and bool(recorded_sha), f"{label} input {name} lacks path/SHA")
        _require(sha256_file(path) == recorded_sha, f"{label} input {name} SHA mismatch")


def _validate_causal_trace(
    row: Mapping[str, Any], *, require_candidate_scores: bool = True
) -> None:
    docs = list(row.get("selected_docs", []))
    steps = list(row.get("selection_steps", []))
    selected_steps = [step for step in steps if step.get("selected_doc_id") is not None]
    _require(len(selected_steps) == len(docs), f"Causal trace/doc count mismatch for {row.get('id')}")
    seen: set[str] = set()
    previous_remaining: int | None = None
    for index, (step, doc) in enumerate(zip(selected_steps, docs)):
        _require(step.get("step") == index, f"Non-contiguous causal step for {row.get('id')}")
        selected_id = str(step.get("selected_doc_id"))
        _require(selected_id == str(doc.get("doc_id")), f"Causal selected ID mismatch for {row.get('id')}")
        _require(selected_id not in seen, f"Causal trace repeats a document for {row.get('id')}")
        scores = step.get("all_candidate_scores")
        if require_candidate_scores:
            _require(isinstance(scores, list) and scores, f"Causal trace lacks candidate scores for {row.get('id')}")
            score_ids = [str(item.get("doc_id")) for item in scores]
            _require(score_ids[0] == selected_id, f"Causal trace winner/order mismatch for {row.get('id')}")
            score_values = [float(item["score"]) for item in scores]
            _require(score_values == sorted(score_values, reverse=True), f"Causal scores are not descending for {row.get('id')}")
        remaining = int(step.get("remaining_count"))
        if previous_remaining is not None:
            _require(remaining == previous_remaining - 1, f"Causal remaining_count mismatch for {row.get('id')}")
        previous_remaining = remaining
        seen.add(selected_id)
    for step in steps[len(selected_steps):]:
        _require(step.get("stop") is True, f"Unexpected post-selection causal step for {row.get('id')}")
        _require(step.get("selected_doc_id") is None, f"Stopping step selected a document for {row.get('id')}")


def _load_bge_cache(config: Mapping[str, Any], method: Mapping[str, Any]) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    cache_path = Path(str(method["score_cache"])).resolve()
    manifest_path = Path(str(method["score_cache_manifest"])).resolve()
    manifest = _load_json(manifest_path, "BGE score-cache manifest")
    _require(manifest.get("status") == "completed" and manifest.get("completed") is True, "BGE score cache is incomplete")
    _require(manifest.get("output_sha256") == sha256_file(cache_path), "BGE score-cache SHA mismatch")
    contract = manifest.get("contract")
    _require(isinstance(contract, dict), "BGE score cache lacks contract")
    _require(contract.get("retrieval_sha256") == sha256_file(config["retrieval"]), "BGE cache retrieval SHA mismatch")
    _require(manifest.get("fingerprint") == stable_hash(contract), "BGE cache fingerprint mismatch")
    result: dict[str, dict[str, float]] = {}
    for row in _rows(cache_path):
        row_id = str(row.get("id"))
        _require(row_id not in result, f"Duplicate BGE cache ID {row_id}")
        scores = row.get("scores")
        _require(isinstance(scores, list), f"BGE cache scores missing for {row_id}")
        result[row_id] = {str(item["doc_id"]): float(item["score"]) for item in scores}
    _require(len(result) == EXPECTED_ROWS, "BGE score cache row count mismatch")
    return result, manifest


def _infogain_from_trace(
    retrieval_row: Mapping[str, Any],
    source_row: Mapping[str, Any],
    *,
    budget: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metadata = source_row.get("selection_metadata")
    _require(isinstance(metadata, dict), "InfoGain source lacks selection_metadata")
    trace = metadata.get("candidate_score_audit")
    _require(isinstance(trace, list), f"InfoGain source lacks candidate score audit for {source_row.get('id')}")
    candidates = list(retrieval_row.get("candidates", []))
    _require(len(trace) == len(candidates), f"InfoGain trace length mismatch for {source_row.get('id')}")
    by_id = {str(item.get("doc_id")): item for item in trace}
    _require(len(by_id) == len(trace), f"InfoGain trace contains duplicate docs for {source_row.get('id')}")
    rank_scores: list[float] = []
    probabilities: list[float] = []
    for source_position, candidate in enumerate(candidates):
        doc_id = str(candidate.get("doc_id"))
        _require(doc_id in by_id, f"InfoGain trace misses {doc_id}")
        item = by_id[doc_id]
        _require(int(item.get("source_position")) == source_position, f"InfoGain source position mismatch for {doc_id}")
        rank_scores.append(float(item["rank_score"]))
        probabilities.append(float(item["filter_probability"]))
    ranked, chosen = rank_and_select(
        candidates,
        rank_scores,
        probabilities,
        top_m=budget,
        filter_threshold=0.2,
        min_docs=2,
    )
    if budget == 4:
        rebuilt_ids = [str(item["doc_id"]) for item in chosen]
        _require(rebuilt_ids == list(map(str, source_row.get("selected_doc_ids", []))), f"InfoGain k4 reconstruction mismatch for {source_row.get('id')}")
        source_ranked = sorted(trace, key=lambda item: int(item["rerank_position"]))
        _require(
            [str(item["doc_id"]) for item in ranked] == [str(item["doc_id"]) for item in source_ranked],
            f"InfoGain full ranking mismatch for {source_row.get('id')}",
        )
    return ranked, chosen


def _validate_source_method(
    method_id: str,
    source_rows: Sequence[dict[str, Any]],
    retrieval_rows: Sequence[dict[str, Any]],
    manifest: Mapping[str, Any],
    *,
    bge_scores: Mapping[str, Mapping[str, float]] | None,
    causal_replay_required: bool,
) -> None:
    params = _parameters(manifest)
    _require(params.get("top_m") == 4, f"{method_id} source top_m is not 4")
    _require(all(int(row.get("max_docs", -1)) == 4 for row in source_rows), f"{method_id} source rows are not max_docs=4")
    if method_id == "retrieval_topk":
        _require(params.get("min_docs") == 4, "RAG source min_docs is not 4")
        for source, retrieval in zip(source_rows, retrieval_rows):
            expected = [str(item.get("doc_id")) for item in retrieval["candidates"][:4]]
            _require(list(map(str, source["selected_doc_ids"])) == expected, f"RAG top4 prefix mismatch for {source.get('id')}")
    elif method_id == "bge":
        _require(params.get("score_threshold") is None, "BGE source must have score_threshold=null")
        _require(params.get("min_docs") == 4, "BGE source min_docs is not 4")
        _require(bge_scores is not None, "BGE score cache is required")
        model = manifest.get("contract", {}).get("model", {})
        for source, retrieval in zip(source_rows, retrieval_rows):
            rebuilt = make_bge_selection(
                dict(retrieval),
                dict(bge_scores[str(source["id"])]),
                method="bge",
                top_m=4,
                score_threshold=None,
                min_docs=4,
                model_metadata=dict(model),
            )
            _require(rebuilt["selected_doc_ids"] == source["selected_doc_ids"], f"BGE k4 cache reconstruction mismatch for {source.get('id')}")
    elif method_id == "infogain_fever":
        _require(params.get("filter_threshold") == 0.2, "InfoGain filter threshold is not 0.2")
        _require(params.get("min_docs") == 2, "InfoGain min_docs is not 2")
        _require(
            params.get("selection_protocol_id")
            in {None, budget_protocol_id(top_m=4, filter_threshold=0.2, min_docs=2)},
            "InfoGain source protocol ID mismatch",
        )
        for source, retrieval in zip(source_rows, retrieval_rows):
            _infogain_from_trace(retrieval, source, budget=4)
    else:
        _require(params.get("min_docs") == 0, f"{method_id} source min_docs is not 0")
        _require(params.get("score_threshold") == 0.0, f"{method_id} score threshold is not 0")
        for source in source_rows:
            _validate_causal_trace(
                source, require_candidate_scores=causal_replay_required
            )


def _validate_native_evaluation(
    paths: Mapping[str, str],
    *,
    selection: Mapping[str, Any],
    generator: Mapping[str, Any],
    selection_rows: Sequence[dict[str, Any]],
    expected_method: str,
) -> dict[str, Any]:
    predictions_path = Path(paths["predictions"]).resolve()
    metrics_path = Path(paths["metrics"]).resolve()
    manifest_path = Path(paths["manifest"]).resolve()
    metrics = _load_json(metrics_path, "native metrics")
    manifest = _load_json(manifest_path, "native evaluation manifest")
    _require(manifest.get("schema_version") == EVALUATION_MANIFEST_SCHEMA_VERSION, "Native evaluation manifest schema mismatch")
    _require(manifest.get("status") == "completed" and manifest.get("completed") is True, "Native evaluation is incomplete")
    _require(manifest.get("method") == expected_method, "Native evaluation method mismatch")
    _require(manifest.get("num_rows") == EXPECTED_ROWS, "Native evaluation row count mismatch")
    _require(manifest.get("predictions_sha256") == sha256_file(predictions_path), "Native predictions SHA mismatch")
    _require(manifest.get("metrics_sha256") == sha256_file(metrics_path), "Native metrics SHA mismatch")
    _require(Path(str(manifest.get("predictions_path"))).resolve() == predictions_path, "Native predictions path mismatch")
    _require(Path(str(manifest.get("metrics_path"))).resolve() == metrics_path, "Native metrics path mismatch")
    contract = manifest.get("contract")
    _require(isinstance(contract, dict), "Native evaluation lacks contract")
    _require(manifest.get("fingerprint") == stable_hash(contract), "Native evaluation fingerprint mismatch")
    expected_binding = build_evaluation_binding(selection, generator, experiment_type=MATCHED_MAIN)
    _require(manifest.get("artifact_binding") == expected_binding, "Native evaluation binding mismatch")
    _require(contract.get("artifact_binding") == expected_binding, "Native evaluation contract binding mismatch")
    _require(contract.get("max_docs") is None, "Native k4 evaluation unexpectedly uses evaluation-time truncation")
    _require(Path(str(contract.get("selection_path"))).resolve() == Path(selection["selection_path"]).resolve(), "Native evaluation selection path mismatch")
    _require(contract.get("selection_sha256") == selection["selection_sha256"], "Native evaluation selection SHA mismatch")
    config_path = Path(str(contract.get("config_path"))).resolve()
    generator_config_path = Path(str(generator["manifest"].get("config_path"))).resolve()
    _require(config_path == generator_config_path, "Native evaluation config path mismatch")
    _require(contract.get("config_sha256") == sha256_file(config_path), "Native evaluation config SHA mismatch")
    identity = generator["generator_identity"]
    _require(contract.get("generator_identity") == identity, "Native evaluation generator identity mismatch")
    _require(contract.get("prompt_hash") == identity.get("prompt_template_hash"), "Native prompt hash mismatch")
    _require(contract.get("verbalizer_hash") == identity.get("verbalizer_hash"), "Native verbalizer hash mismatch")
    _require(contract.get("generator_model") == identity.get("model_name_or_path"), "Native generator model mismatch")
    _require(contract.get("generator_sha256") == identity.get("model_sha256"), "Native generator model SHA mismatch")
    _require(contract.get("generator_revision") == identity.get("model_revision"), "Native generator revision mismatch")
    _require(contract.get("tokenizer_revision") == identity.get("tokenizer_revision"), "Native tokenizer revision mismatch")
    _require(contract.get("max_context_tokens") == identity.get("max_context_tokens"), "Native context limit mismatch")
    _require(metrics.get("num_examples") == EXPECTED_ROWS, "Native metrics num_examples mismatch")
    _require(metrics.get("split") == SPLIT, "Native metrics split mismatch")
    _require(isinstance(metrics.get("accuracy"), (int, float)), "Native metrics lack accuracy")
    _require(isinstance(metrics.get("macro_f1"), (int, float)), "Native metrics lack macro_f1")
    _require(isinstance(metrics.get("avg_num_docs"), (int, float)), "Native metrics lack avg_num_docs")
    if metrics.get("avg_evidence_tokens") is not None:
        _require(isinstance(metrics.get("avg_evidence_tokens"), (int, float)), "Native avg_evidence_tokens is invalid")
        _require(metrics.get("evidence_tokens_basis") == TOKEN_BASIS, "Native evidence-token basis mismatch")
    predictions = _rows(predictions_path)
    _require(_validate_ids(predictions, label="native predictions") == _validate_ids(selection_rows, label="selection"), "Native prediction IDs/order mismatch")
    for prediction, selected in zip(predictions, selection_rows):
        _require(prediction.get("query") == selected.get("query"), f"Native prediction query mismatch for {selected.get('id')}")
        _require(prediction.get("gold") == selected.get("label", selected.get("gold")), f"Native prediction label mismatch for {selected.get('id')}")
        _require(prediction.get("method") == expected_method, f"Native prediction method mismatch for {selected.get('id')}")
        _require(
            list(map(str, prediction.get("selected_doc_ids", [])))
            == list(map(str, selected.get("selected_doc_ids", []))),
            f"Native prediction evidence mismatch for {selected.get('id')}",
        )
    return {
        "paths": {key: str(Path(value).resolve()) for key, value in paths.items()},
        "manifest": manifest,
        "metrics": metrics,
        "predictions_sha256": sha256_file(predictions_path),
        "metrics_sha256": sha256_file(metrics_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _load_native_token_records(
    config: Mapping[str, Any],
    *,
    selections: Mapping[tuple[str, str], Mapping[str, Any]],
    generators: Mapping[tuple[str, str], Mapping[str, Any]],
    evaluations: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    results_path = Path(str(config["native_token_results"]))
    audit_path = Path(str(config["native_token_audit"]))
    _require(
        results_path.parent.resolve() == audit_path.parent.resolve(),
        "Native token CSV and audit must share one analysis directory",
    )
    analysis_manifest_path = audit_path.with_name("analysis_manifest.json")
    audit = _load_json(audit_path, "native evidence-token audit")
    analysis_manifest = _load_json(
        analysis_manifest_path, "native evidence-token analysis manifest"
    )
    _require(
        audit.get("schema_version") == NATIVE_TOKEN_AUDIT_SCHEMA_VERSION,
        "Native token-audit schema mismatch",
    )
    _require(audit.get("status") == "PASS", "Native token audit did not pass")
    _require(audit.get("mode") == "all", "Native token audit is not a full analysis")
    _require(audit.get("num_groups") == 48, "Native token audit does not cover 48 groups")
    _require(audit.get("error") is None, "Native token audit records an error")
    _require(audit.get("evidence_tokens_basis") == TOKEN_BASIS, "Native token-audit basis mismatch")
    _require(
        audit.get("num_docs_histogram_basis") == NATIVE_TOKEN_HISTOGRAM_BASIS,
        "Native token-audit histogram basis mismatch",
    )
    _require(
        audit.get("num_docs_histogram_comparison")
        == NATIVE_TOKEN_HISTOGRAM_COMPARISON,
        "Native token-audit histogram comparison mismatch",
    )
    groups = audit.get("groups")
    _require(isinstance(groups, list), "Native token audit groups must be a list")
    _require(len(groups) == 48, "Native token audit must contain 48 groups")

    _require(
        analysis_manifest.get("schema_version")
        == NATIVE_TOKEN_ANALYSIS_SCHEMA_VERSION,
        "Native token analysis-manifest schema mismatch",
    )
    _require(
        analysis_manifest.get("status") == "completed",
        "Native token analysis manifest is not completed",
    )
    _require(
        analysis_manifest.get("completed") is True,
        "Native token analysis manifest is incomplete",
    )
    _require(
        analysis_manifest.get("mode") == "all",
        "Native token analysis manifest is not a full analysis",
    )
    _require(
        analysis_manifest.get("num_groups") == 48,
        "Native token analysis manifest does not cover 48 groups",
    )
    _require(
        analysis_manifest.get("evidence_tokens_basis") == TOKEN_BASIS,
        "Native token analysis-manifest basis mismatch",
    )
    _require(
        analysis_manifest.get("num_docs_histogram_basis")
        == NATIVE_TOKEN_HISTOGRAM_BASIS,
        "Native token analysis-manifest histogram basis mismatch",
    )
    _require(
        analysis_manifest.get("num_docs_histogram_comparison")
        == NATIVE_TOKEN_HISTOGRAM_COMPARISON,
        "Native token analysis-manifest histogram comparison mismatch",
    )
    _require(results_path.is_file(), f"Missing native token results: {results_path}")
    outputs = analysis_manifest.get("outputs")
    _require(isinstance(outputs, dict), "Native token analysis manifest lacks outputs")
    results_sha256 = sha256_file(results_path)
    audit_sha256 = sha256_file(audit_path)
    analysis_manifest_sha256 = sha256_file(analysis_manifest_path)
    _require(
        outputs.get(results_path.name) == results_sha256,
        "Native token CSV SHA differs from analysis manifest",
    )
    _require(
        outputs.get(audit_path.name) == audit_sha256,
        "Native token audit SHA differs from analysis manifest",
    )
    with results_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _require(len(rows) == 48, "Native token CSV must contain 48 rows")

    expected_keys = {
        (dataset_id, generator_id, method_id)
        for dataset_id in NATIVE_TOKEN_DATASETS
        for generator_id in NATIVE_TOKEN_GENERATORS
        for method_id in NATIVE_TOKEN_METHODS
    }
    csv_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    required_csv_fields = {
        "dataset_id",
        "generator_id",
        "method_id",
        "experiment_type",
        "seed",
        "split",
        "accuracy",
        "macro_f1",
        "num_examples",
        "avg_num_docs",
        "avg_evidence_chars",
        "avg_evidence_tokens",
        "total_evidence_tokens",
        "evidence_tokens_basis",
        "num_docs_0",
        "num_docs_1",
        "num_docs_2",
        "num_docs_3",
        "num_docs_4",
        "selection_path",
        "selection_sha256",
        "selection_manifest_path",
        "selection_manifest_sha256",
        "predictions_path",
        "predictions_sha256",
        "metrics_path",
        "metrics_sha256",
        "evaluation_manifest_path",
        "evaluation_manifest_sha256",
        "generator_manifest_path",
        "generator_manifest_sha256",
        "tokenizer_name_or_path",
        "tokenizer_revision",
        "tokenizer_sha256",
        "tokenizer_identity_fingerprint",
        "generator_identity_fingerprint",
    }
    for row in rows:
        _require(
            required_csv_fields.issubset(row),
            "Native token CSV lacks required formal-analysis fields",
        )
        key3 = (
            str(row["dataset_id"]),
            str(row["generator_id"]),
            str(row["method_id"]),
        )
        _require(key3 in expected_keys, f"Unexpected native token CSV group: {key3}")
        _require(key3 not in csv_by_key, f"Duplicate native token CSV group: {key3}")
        csv_by_key[key3] = row
    _require(set(csv_by_key) == expected_keys, "Native token CSV group matrix mismatch")

    audit_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for group in groups:
        _require(isinstance(group, dict), "Native token audit group must be an object")
        key3 = (
            str(group.get("dataset_id")),
            str(group.get("generator_id")),
            str(group.get("method_id")),
        )
        _require(key3 in expected_keys, f"Unexpected native token audit group: {key3}")
        _require(key3 not in audit_by_key, f"Duplicate native token audit group: {key3}")
        _require(group.get("status") == "PASS", f"Native token group did not pass: {key3}")
        _require(
            group.get("group_id") == "__".join(key3),
            f"Native token group ID mismatch: {key3}",
        )
        _require(group.get("experiment_type") == MATCHED_MAIN, f"Native token experiment mismatch: {key3}")
        _require(group.get("split") == SPLIT, f"Native token split mismatch: {key3}")
        _require(
            group.get("num_examples") == NATIVE_TOKEN_DATASETS[key3[0]],
            f"Native token example count mismatch: {key3}",
        )
        _require(group.get("evidence_tokens_basis") == TOKEN_BASIS, f"Native token group basis mismatch: {key3}")
        source_sha256 = group.get("source_sha256")
        tokenizer_identity = group.get("tokenizer_identity")
        histogram = group.get("num_docs_histogram")
        _require(isinstance(source_sha256, dict), f"Native token group lacks source SHA map: {key3}")
        _require(isinstance(tokenizer_identity, dict), f"Native token group lacks tokenizer identity: {key3}")
        _require(isinstance(histogram, dict), f"Native token group lacks document histogram: {key3}")
        _require(
            tokenizer_identity.get("generator_id") == key3[1],
            f"Native token group tokenizer generator mismatch: {key3}",
        )
        _require(
            group.get("num_docs_histogram_basis") == NATIVE_TOKEN_HISTOGRAM_BASIS,
            f"Native token group histogram basis mismatch: {key3}",
        )
        _require(
            all(
                isinstance(name, str)
                and name in {"0", "1", "2", "3", "4"}
                and isinstance(count, int)
                and not isinstance(count, bool)
                and count >= 0
                for name, count in histogram.items()
            ),
            f"Native token group histogram is invalid: {key3}",
        )
        _require(
            sum(histogram.values()) == NATIVE_TOKEN_DATASETS[key3[0]],
            f"Native token group histogram count mismatch: {key3}",
        )
        _require(
            group.get("tokenizer_identity_fingerprint") == stable_hash(tokenizer_identity),
            f"Native token group tokenizer fingerprint mismatch: {key3}",
        )
        row = csv_by_key[key3]
        for field in (
            "experiment_type",
            "split",
            "evidence_tokens_basis",
            "selection_path",
            "selection_manifest_path",
            "predictions_path",
            "metrics_path",
            "evaluation_manifest_path",
            "generator_manifest_path",
            "tokenizer_identity_fingerprint",
            "generator_identity_fingerprint",
        ):
            _require(row[field] == str(group.get(field)), f"Native token CSV/audit {field} mismatch: {key3}")
        expected_seed = "" if group.get("seed") is None else str(group.get("seed"))
        _require(row["seed"] == expected_seed, f"Native token CSV/audit seed mismatch: {key3}")
        for field in (
            "accuracy",
            "macro_f1",
            "avg_num_docs",
            "avg_evidence_chars",
            "avg_evidence_tokens",
            "total_evidence_tokens",
        ):
            try:
                matches = float(row[field]) == float(group[field])
            except (KeyError, TypeError, ValueError):
                matches = False
            _require(matches, f"Native token CSV/audit {field} mismatch: {key3}")
        _require(
            int(row["num_examples"]) == int(group["num_examples"]),
            f"Native token CSV/audit num_examples mismatch: {key3}",
        )
        for value in range(5):
            _require(
                int(row[f"num_docs_{value}"]) == histogram.get(str(value), 0),
                f"Native token CSV/audit num_docs_{value} mismatch: {key3}",
            )
        for name in (
            "selection",
            "selection_manifest",
            "predictions",
            "metrics",
            "evaluation_manifest",
            "generator_manifest",
        ):
            source_sha = source_sha256.get(name)
            _require(
                isinstance(source_sha, str)
                and len(source_sha) == 64
                and all(character in "0123456789abcdef" for character in source_sha)
                and row[f"{name}_sha256"] == source_sha,
                f"Native token CSV/audit {name} SHA mismatch: {key3}",
            )
        for field in (
            "tokenizer_name_or_path",
            "tokenizer_sha256",
        ):
            _require(
                row[field] == tokenizer_identity.get(field),
                f"Native token CSV/audit {field} mismatch: {key3}",
            )
        _require(
            (row.get("tokenizer_revision") or None)
            == tokenizer_identity.get("tokenizer_revision"),
            f"Native token CSV/audit tokenizer revision mismatch: {key3}",
        )
        audit_by_key[key3] = group
    _require(set(audit_by_key) == expected_keys, "Native token audit group matrix mismatch")

    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("dataset_id") != DATASET_ID or row.get("generator_id") not in GENERATORS or row.get("method_id") not in METHODS:
            continue
        key = (str(row["method_id"]), str(row["generator_id"]))
        _require(key not in result, f"Duplicate native token record: {key}")
        _require(key in selections, f"Native token selection context missing: {key}")
        _require(key in evaluations, f"Native token evaluation context missing: {key}")
        _require(key in generators, f"Native token generator context missing: {key}")
        selection = selections[key]
        evaluation = evaluations[key]
        generator = generators[key]
        _require(row.get("selection_sha256") == selection["provenance"]["selection_sha256"], f"Native token selection SHA mismatch: {key}")
        _require(row.get("selection_manifest_sha256") == selection["provenance"]["manifest_sha256"], f"Native token selection manifest SHA mismatch: {key}")
        _require(row.get("predictions_sha256") == evaluation["predictions_sha256"], f"Native token predictions SHA mismatch: {key}")
        _require(row.get("metrics_sha256") == evaluation["metrics_sha256"], f"Native token metrics SHA mismatch: {key}")
        _require(row.get("evaluation_manifest_sha256") == evaluation["manifest_sha256"], f"Native token evaluation manifest SHA mismatch: {key}")
        _require(row.get("generator_manifest_sha256") == generator["manifest_sha256"], f"Native token generator manifest SHA mismatch: {key}")
        _require(row.get("generator_identity_fingerprint") == generator["generator_identity_fingerprint"], f"Native token generator identity mismatch: {key}")
        _require(row.get("evidence_tokens_basis") == TOKEN_BASIS, f"Native token basis mismatch: {key}")
        record = {
            "accuracy": float(row["accuracy"]),
            "macro_f1": float(row["macro_f1"]),
            "avg_num_docs": float(row["avg_num_docs"]),
            "avg_evidence_tokens": float(row["avg_evidence_tokens"]),
            "evidence_tokens_basis": row["evidence_tokens_basis"],
            "tokenizer_name_or_path": row["tokenizer_name_or_path"],
            "tokenizer_revision": row.get("tokenizer_revision") or None,
            "tokenizer_sha256": row["tokenizer_sha256"],
            "tokenizer_identity_fingerprint": row["tokenizer_identity_fingerprint"],
            "source_csv_path": str(results_path.resolve()),
            "source_csv_sha256": results_sha256,
            "source_audit_path": str(audit_path.resolve()),
            "source_audit_sha256": audit_sha256,
            "source_analysis_manifest_path": str(analysis_manifest_path.resolve()),
            "source_analysis_manifest_sha256": analysis_manifest_sha256,
        }
        metrics = evaluation["metrics"]
        for field in ("accuracy", "macro_f1", "avg_num_docs"):
            _require(abs(float(metrics[field]) - float(record[field])) <= 1e-12, f"Native token CSV {field} mismatch: {key}")
        identity = generator["generator_identity"]
        _require(record["tokenizer_name_or_path"] == identity["tokenizer_name_or_path"], f"Native token tokenizer path mismatch: {key}")
        _require(record["tokenizer_revision"] == identity.get("tokenizer_revision"), f"Native token tokenizer revision mismatch: {key}")
        _require(record["tokenizer_sha256"] == identity["tokenizer_sha256"], f"Native token tokenizer SHA mismatch: {key}")
        result[key] = record
    _require(len(result) == 10, "Native token CSV must contain all 10 FEVER budget anchors")
    return result


def preflight_sources(
    config: Mapping[str, Any],
    *,
    include_evaluations: bool = True,
    causal_replay_required: bool = True,
) -> dict[str, Any]:
    assert_isolated_artifact_root(str(config["artifact_root"]), list(config["protected_roots"]))
    retrieval_rows, retrieval_manifest = _validate_retrieval(config)
    methods = _method_map(config)
    generators = _generator_map(config)
    generator_context: dict[tuple[str, str], dict[str, Any]] = {}
    for generator_id, spec in generators.items():
        for method_id in METHODS:
            generator_context[(method_id, generator_id)] = validate_generator_manifest(
                generator_manifest_path(config, method_id, generator_id),
                expected_dataset_id=DATASET_ID,
                expected_generator_id=generator_id,
                expected_config_path=spec["config"],
            )
    bge_scores, bge_cache_manifest = _load_bge_cache(config, methods["bge"])
    selection_context: dict[tuple[str, str], dict[str, Any]] = {}
    canonical_ids = _validate_ids(retrieval_rows, label="retrieval")
    for method_id in METHODS:
        method = methods[method_id]
        generator_scope = GENERATORS if method_id in CONDITIONED_METHODS else (GENERATORS[0],)
        for generator_id in generator_scope:
            selection_path, manifest_path = source_selection_paths(method, generator_id)
            provenance = validate_selection_provenance(
                selection_path,
                manifest_path,
                formal_v2=True,
                expected_dataset_id=DATASET_ID,
                expected_method=method_id,
            )
            rows = _rows(selection_path)
            _validate_contract_inputs(
                provenance["manifest"], label=f"{generator_id}/{method_id} source"
            )
            ids = _validate_selection_rows(
                rows,
                retrieval_rows,
                method_id=method_id,
                label=f"{generator_id}/{method_id} source",
            )
            _require(ids == canonical_ids, "Source selection order mismatch")
            _validate_source_method(
                method_id,
                rows,
                retrieval_rows,
                provenance["manifest"],
                bge_scores=bge_scores if method_id == "bge" else None,
                causal_replay_required=causal_replay_required,
            )
            binding = provenance["artifact_binding"]
            expected_dependency = "conditioned" if method_id in CONDITIONED_METHODS else "independent"
            _require(binding.get("generator_dependency") == expected_dependency, f"{method_id} dependency mismatch")
            if method_id in CONDITIONED_METHODS:
                _require(binding.get("conditioning_generator_id") == generator_id, f"{method_id} conditioning generator mismatch")
                _require(
                    binding.get("generator_identity_fingerprint")
                    == generator_context[(method_id, generator_id)]["generator_identity_fingerprint"],
                    f"{method_id} generator identity mismatch",
                )
            selection_context[(method_id, generator_id)] = {
                "rows": rows,
                "provenance": provenance,
                "path": selection_path,
                "manifest_path": manifest_path,
            }
            if method_id not in CONDITIONED_METHODS:
                selection_context[(method_id, GENERATORS[1])] = selection_context[(method_id, generator_id)]

    evaluation_context: dict[tuple[str, str], dict[str, Any]] = {}
    token_records: dict[tuple[str, str], dict[str, Any]] = {}
    if include_evaluations:
        for generator_id in GENERATORS:
            for method_id in METHODS:
                source = selection_context[(method_id, generator_id)]
                evaluation_context[(method_id, generator_id)] = _validate_native_evaluation(
                    native_evaluation_paths(methods[method_id], generator_id),
                    selection=source["provenance"],
                    generator=generator_context[(method_id, generator_id)],
                    selection_rows=source["rows"],
                    expected_method=method_id,
                )
        token_records = _load_native_token_records(
            config,
            selections=selection_context,
            generators=generator_context,
            evaluations=evaluation_context,
        )

    report: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "passed",
        "dataset_id": DATASET_ID,
        "split": SPLIT,
        "expected_rows": EXPECTED_ROWS,
        "retrieval_path": str(Path(config["retrieval"]).resolve()),
        "retrieval_sha256": sha256_file(config["retrieval"]),
        "retrieval_manifest_sha256": sha256_file(config["retrieval_manifest"]),
        "bge_score_cache_sha256": sha256_file(methods["bge"]["score_cache"]),
        "source_selection_count": len({context["path"] for context in selection_context.values()}),
        "native_evaluation_count": len(evaluation_context),
        "native_token_record_count": len(token_records),
        "native_token_results_sha256": (
            sha256_file(config["native_token_results"]) if include_evaluations else None
        ),
        "native_token_audit_sha256": (
            sha256_file(config["native_token_audit"]) if include_evaluations else None
        ),
        "native_token_analysis_manifest_sha256": (
            sha256_file(
                Path(str(config["native_token_audit"])).with_name(
                    "analysis_manifest.json"
                )
            )
            if include_evaluations
            else None
        ),
        "causal_replay_required": causal_replay_required,
        "generator_identity_fingerprints": {
            f"{method_id}/{generator_id}": item["generator_identity_fingerprint"]
            for (method_id, generator_id), item in generator_context.items()
        },
    }
    report["preflight_fingerprint"] = stable_hash(report)
    return {
        "report": report,
        "retrieval_rows": retrieval_rows,
        "retrieval_manifest": retrieval_manifest,
        "bge_scores": bge_scores,
        "bge_cache_manifest": bge_cache_manifest,
        "generators": generator_context,
        "selections": selection_context,
        "evaluations": evaluation_context,
        "token_records": token_records,
    }


def _rag_rows(retrieval_rows: Sequence[dict[str, Any]], budget: int) -> Iterable[dict[str, Any]]:
    for row in retrieval_rows:
        docs = [
            normalize_selected_doc(candidate, selector_score=None, selection_step=step)
            for step, candidate in enumerate(row["candidates"][:budget])
        ]
        yield make_selection_row(
            row,
            method="retrieval_topk",
            selected_docs=docs,
            selection_steps=[
                {
                    "step": step,
                    "selected_doc_id": str(doc["doc_id"]),
                    "predicted_score": None,
                    "remaining_count": len(row["candidates"]) - step,
                    "stop": False,
                }
                for step, doc in enumerate(docs)
            ],
            stop_reason="inference_budget_reached",
            max_docs=budget,
            uses_gold_at_test=False,
            selection_metadata={
                "state_aware": False,
                "uses_gold_at_test": False,
                "ranking": "source_order",
                "min_docs": budget,
                "budget_k": budget,
                "budget_selection_mode": "fixed_ranking_prefix",
            },
        )


def _bge_rows(
    retrieval_rows: Sequence[dict[str, Any]],
    scores: Mapping[str, Mapping[str, float]],
    source_manifest: Mapping[str, Any],
    budget: int,
) -> Iterable[dict[str, Any]]:
    model = dict(source_manifest.get("contract", {}).get("model", {}))
    for row in retrieval_rows:
        selected = make_bge_selection(
            row,
            dict(scores[str(row["id"])]),
            method="bge",
            top_m=budget,
            score_threshold=None,
            min_docs=budget,
            model_metadata=model,
        )
        selected["selection_metadata"].update(
            {
                "budget_k": budget,
                "budget_selection_mode": "full_reranking_prefix",
                "source_top_m": 4,
            }
        )
        yield selected


def _infogain_rows(
    retrieval_rows: Sequence[dict[str, Any]],
    source_rows: Sequence[dict[str, Any]],
    budget: int,
) -> Iterable[dict[str, Any]]:
    protocol_id = budget_protocol_id(top_m=budget, filter_threshold=0.2, min_docs=2)
    for retrieval, source in zip(retrieval_rows, source_rows):
        ranked, chosen = _infogain_from_trace(retrieval, source, budget=budget)
        source_metadata = dict(source.get("selection_metadata") or {})
        docs: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []
        for step, item in enumerate(chosen):
            doc = normalize_selected_doc(
                item["candidate"],
                selector_score=float(item["rank_score"]),
                selection_step=step,
            )
            doc.update(
                {
                    "rank_score": item["rank_score"],
                    "filter_probability": item["filter_probability"],
                    "min_docs_fallback": item["min_docs_fallback"],
                }
            )
            docs.append(doc)
            steps.append(
                {
                    "step": step,
                    "selected_doc_id": item["doc_id"],
                    "predicted_score": item["rank_score"],
                    "filter_probability": item["filter_probability"],
                    "min_docs_fallback": item["min_docs_fallback"],
                    "stop": False,
                }
            )
        yield make_selection_row(
            retrieval,
            method="infogain_fever",
            selected_docs=docs,
            selection_steps=steps,
            stop_reason=stop_reason_for_selection(
                candidate_count=len(retrieval["candidates"]),
                selected_count=len(docs),
                top_m=budget,
                min_docs=2,
                filter_threshold=0.2,
                used_fallback=any(item["min_docs_fallback"] for item in chosen),
            ),
            max_docs=budget,
            uses_gold_at_test=False,
            selection_metadata={
                "official_fidelity": "adapted",
                "state_aware": False,
                "uses_gold_at_test": False,
                "deployable": True,
                "filter_threshold": 0.2,
                "top_m": budget,
                "min_docs": 2,
                "selection_protocol_family_id": BUDGET_PROTOCOL_FAMILY_ID,
                "selection_protocol_id": protocol_id,
                "budget_k": budget,
                "budget_selection_mode": "rerank_then_window_filter",
                "ranking_score_semantics": "unconstrained_rank_head_score",
                "filter_probability_semantics": "softmax_filter_head_positive_class",
                "candidate_score_audit": [
                    {
                        key: item[key]
                        for key in (
                            "doc_id",
                            "source_position",
                            "rerank_position",
                            "rank_score",
                            "filter_probability",
                            "within_top_m",
                            "passes_filter",
                        )
                    }
                    for item in ranked
                ],
                "checkpoint_fingerprint": source_metadata.get("checkpoint_fingerprint"),
                "source_k4_protocol_id": source_metadata.get("selection_protocol_id"),
            },
        )


def replay_causal_prefix_row(source: Mapping[str, Any], budget: int) -> dict[str, Any]:
    _require(budget in BUDGETS, "Causal replay budget must be 2, 3, or 4")
    _validate_causal_trace(source)
    result = copy.deepcopy(dict(source))
    docs = list(result["selected_docs"])
    steps = list(result["selection_steps"])
    if len(docs) >= budget:
        result["selected_docs"] = docs[:budget]
        result["selected_doc_ids"] = [str(doc["doc_id"]) for doc in docs[:budget]]
        result["num_docs"] = budget
        selected_steps = [step for step in steps if step.get("selected_doc_id") is not None]
        result["selection_steps"] = selected_steps[:budget]
        result["stop_reason"] = "inference_budget_reached"
    else:
        result["selected_docs"] = docs
        result["selected_doc_ids"] = [str(doc["doc_id"]) for doc in docs]
        result["num_docs"] = len(docs)
    result["max_docs"] = budget
    metadata = dict(result.get("selection_metadata") or {})
    metadata.update(
        {
            "trained_max_docs": 4,
            "inference_budget": budget,
            "budget_k": budget,
            "budget_selection_mode": "verified_causal_prefix_replay",
            "uses_gold_at_inference": False,
            "uses_gold_at_test": False,
        }
    )
    result["selection_metadata"] = metadata
    result["uses_gold_at_test"] = False
    validate_selection_row(result)
    return result


def _budget_contract(
    task: Mapping[str, Any],
    config: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    method_id = str(task["method_id"])
    budget = int(task["budget_k"])
    generator_id = str(task.get("generator_id") or GENERATORS[0])
    source = context["selections"][(method_id, generator_id)]
    source_manifest = source["provenance"]["manifest"]
    inputs: dict[str, str | Path] = {
        "source_k4_selection": source["path"],
        "source_k4_selection_manifest": source["manifest_path"],
    }
    parameters: dict[str, Any]
    if method_id == "retrieval_topk":
        inputs.update({"retrieval": config["retrieval"], "retrieval_manifest": config["retrieval_manifest"]})
        parameters = {
            "top_m": budget,
            "min_docs": budget,
            "budget_k": budget,
            "budget_selection_mode": "fixed_ranking_prefix",
            "uses_gold_at_inference": False,
            "split": SPLIT,
        }
    elif method_id == "bge":
        method = _method_map(config)[method_id]
        inputs.update(
            {
                "retrieval": config["retrieval"],
                "retrieval_manifest": config["retrieval_manifest"],
                "score_cache": method["score_cache"],
                "score_cache_manifest": method["score_cache_manifest"],
            }
        )
        parameters = {
            "top_m": budget,
            "min_docs": budget,
            "score_threshold": None,
            "budget_k": budget,
            "budget_selection_mode": "full_reranking_prefix",
            "uses_gold_at_inference": False,
            "split": SPLIT,
        }
    elif method_id == "infogain_fever":
        inputs.update({"retrieval": config["retrieval"], "retrieval_manifest": config["retrieval_manifest"]})
        parameters = {
            "seed": SEED,
            "top_m": budget,
            "min_docs": 2,
            "filter_threshold": 0.2,
            "selection_protocol_family_id": BUDGET_PROTOCOL_FAMILY_ID,
            "selection_protocol_id": budget_protocol_id(
                top_m=budget, filter_threshold=0.2, min_docs=2
            ),
            "budget_k": budget,
            "budget_selection_mode": "rerank_then_window_filter",
            "uses_gold_at_inference": False,
            "split": SPLIT,
        }
    else:
        source_parameters = dict(_parameters(source_manifest))
        _require(source_parameters.get("top_m") == 4, "Causal source trained top_m changed")
        parameters = {
            **source_parameters,
            "top_m": 4,
            "trained_max_docs": 4,
            "inference_budget": budget,
            "budget_k": budget,
            "budget_selection_mode": "verified_causal_prefix_replay",
            "uses_gold_at_inference": False,
            "split": SPLIT,
        }
    return build_selection_contract(
        method=method_id,
        method_contract_version=source_manifest.get("method_contract_version"),
        input_paths=inputs,
        parameters=parameters,
        model=dict(source_manifest.get("contract", {}).get("model", {})),
        artifact_binding=source["provenance"]["artifact_binding"],
    )


def _state_aware_inference_command(
    task: Mapping[str, Any],
    config: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    resume: bool,
) -> list[str]:
    method_id = str(task["method_id"])
    generator_id = str(task["generator_id"])
    budget = int(task["budget_k"])
    source = context["selections"][(method_id, generator_id)]
    contract = source["provenance"]["manifest"].get("contract", {})
    inputs = contract.get("inputs", {})
    model = contract.get("model", {})
    _require(isinstance(inputs, dict), "State-aware source contract lacks inputs")
    _require(isinstance(model, dict), "State-aware source contract lacks model")
    posteriors = inputs.get("posteriors", {}).get("path")
    training_manifest = inputs.get("training_manifest", {}).get("path")
    checkpoint = model.get("checkpoint")
    _require(bool(posteriors), "State-aware source lacks posterior path")
    _require(bool(training_manifest), "State-aware source lacks training manifest path")
    _require(bool(checkpoint), "State-aware source lacks checkpoint path")
    command = [
        str(config["server_python"]),
        _join(
            str(config["server_project_root"]),
            "scripts",
            "preformal",
            "27_select_signed_v1.py",
        ),
        "--posteriors",
        str(posteriors),
        "--checkpoint-dir",
        str(checkpoint),
        "--training-manifest",
        str(training_manifest),
        "--dataset-id",
        DATASET_ID,
        "--generator-id",
        generator_id,
        "--formal-v2-identity",
        "--method-name",
        method_id,
        "--output",
        str(task["output_selection"]),
        "--seed",
        str(SEED),
        "--split",
        SPLIT,
        "--inference-budget",
        str(budget),
    ]
    if resume:
        command.append("--resume")
    return command


def materialize_selections(
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    project_root: str | Path,
    resume: bool,
    state_aware_mode: str = "replay",
) -> dict[str, Any]:
    _require(
        state_aware_mode in {"replay", "inference"},
        "state_aware_mode must be replay or inference",
    )
    context = preflight_sources(
        config,
        include_evaluations=False,
        causal_replay_required=state_aware_mode == "replay",
    )
    outputs: list[dict[str, Any]] = []
    for task in plan["selection_tasks"]:
        method_id = str(task["method_id"])
        generator_id = str(task.get("generator_id") or GENERATORS[0])
        budget = int(task["budget_k"])
        source = context["selections"][(method_id, generator_id)]
        if method_id == "retrieval_topk":
            rows = _rag_rows(context["retrieval_rows"], budget)
        elif method_id == "bge":
            rows = _bge_rows(
                context["retrieval_rows"],
                context["bge_scores"],
                source["provenance"]["manifest"],
                budget,
            )
        elif method_id == "infogain_fever":
            rows = _infogain_rows(context["retrieval_rows"], source["rows"], budget)
        elif state_aware_mode == "replay":
            rows = (replay_causal_prefix_row(row, budget) for row in source["rows"])
        else:
            command = _state_aware_inference_command(
                task, config, context, resume=resume
            )
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                raise RuntimeError(
                    f"State-aware budget selection failed with exit {result.returncode}: "
                    f"{task['task_id']}"
                )
            count = len(_rows(task["output_selection"]))
            reused = resume
            rows = ()
        if method_id not in CONDITIONED_METHODS or method_id == "infogain_fever" or state_aware_mode == "replay":
            contract = _budget_contract(task, config, context)
            count, reused = publish_selection(
                task["output_selection"],
                rows,
                contract=contract,
                project_root=project_root,
                resume=resume,
                overwrite=False,
            )
        _require(count == EXPECTED_ROWS, f"Materialized selection row count mismatch for {task['task_id']}")
        outputs.append(
            {
                "task_id": task["task_id"],
                "selection_path": task["output_selection"],
                "selection_manifest_path": task["output_selection_manifest"],
                "selection_sha256": sha256_file(task["output_selection"]),
                "reused": reused,
            }
        )
    return {
        "schema_version": "rag_cbwdm_fever_budget_materialization.v1",
        "status": "completed",
        "plan_fingerprint": plan["plan_fingerprint"],
        "selection_count": len(outputs),
        "state_aware_mode": state_aware_mode,
        "outputs": outputs,
    }


def _validate_materialized_budget_rows(
    point: Mapping[str, Any],
    rows: Sequence[dict[str, Any]],
    context: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    method_id = str(point["method_id"])
    generator_id = str(point["generator_id"])
    budget = int(point["budget_k"])
    source = context["selections"][(method_id, generator_id)]
    if method_id == "retrieval_topk":
        expected = list(_rag_rows(context["retrieval_rows"], budget))
    elif method_id == "bge":
        expected = list(
            _bge_rows(
                context["retrieval_rows"],
                context["bge_scores"],
                source["provenance"]["manifest"],
                budget,
            )
        )
    elif method_id == "infogain_fever":
        expected = list(
            _infogain_rows(context["retrieval_rows"], source["rows"], budget)
        )
    elif _parameters(provenance["manifest"]).get("budget_selection_mode") == "verified_causal_prefix_replay":
        expected = [replay_causal_prefix_row(row, budget) for row in source["rows"]]
    else:
        parameters = _parameters(provenance["manifest"])
        _require(parameters.get("top_m") == 4, "Direct state-aware selection changed trained top_m")
        _require(parameters.get("inference_budget") == budget, "Direct state-aware selection budget mismatch")
        expected = []
        for actual, native in zip(rows, source["rows"]):
            _validate_causal_trace(actual, require_candidate_scores=True)
            native_docs = list(native.get("selected_docs", []))[:budget]
            _require(
                list(map(str, actual.get("selected_doc_ids", [])))
                == [str(doc.get("doc_id")) for doc in native_docs],
                f"Direct state-aware inference is not the native causal prefix for {actual.get('id')}",
            )
            _require(
                [
                    (doc.get("title"), doc.get("text"))
                    for doc in actual.get("selected_docs", [])
                ]
                == [(doc.get("title"), doc.get("text")) for doc in native_docs],
                f"Direct state-aware evidence content mismatch for {actual.get('id')}",
            )
        return
    _require(len(expected) == len(rows), f"Materialized row count mismatch for {point['point_id']}")
    for actual, reference in zip(rows, expected):
        for field in (
            "id",
            "query",
            "split",
            "method",
            "selected_doc_ids",
            "selected_docs",
            "num_docs",
            "max_docs",
            "selection_steps",
            "stop_reason",
            "uses_gold_at_test",
        ):
            _require(
                actual.get(field) == reference.get(field),
                f"Materialized {field} mismatch for {point['point_id']} row {actual.get('id')}",
            )
        actual_metadata = actual.get("selection_metadata", {})
        reference_metadata = reference.get("selection_metadata", {})
        for field in (
            "budget_k",
            "budget_selection_mode",
            "inference_budget",
            "selection_protocol_id",
            "filter_threshold",
            "min_docs",
        ):
            if field in reference_metadata:
                _require(
                    actual_metadata.get(field) == reference_metadata.get(field),
                    f"Materialized metadata {field} mismatch for {point['point_id']}",
                )


def evidence_vector(
    rows: Sequence[Mapping[str, Any]], *, expected_rows: int = EXPECTED_ROWS
) -> tuple[str, dict[str, Any]]:
    _require(len(rows) == expected_rows, f"budget selection must contain {expected_rows} rows")
    ids = tuple(str(row.get("id")) for row in rows)
    _require(all(value and value != "None" for value in ids), "budget selection contains an empty ID")
    _require(len(ids) == len(set(ids)), "budget selection contains duplicate IDs")
    vector: list[dict[str, Any]] = []
    histogram: dict[str, int] = {}
    for row in rows:
        docs = list(row.get("selected_docs", []))
        rendered = render_evidence(docs)
        count = len(docs)
        histogram[str(count)] = histogram.get(str(count), 0) + 1
        vector.append(
            {
                "id": str(row["id"]),
                "query": row.get("query"),
                "selected_doc_ids": [str(doc.get("doc_id")) for doc in docs],
                "rendered_evidence": rendered,
                "num_docs": count,
            }
        )
    return stable_hash(vector), {
        "num_rows": len(vector),
        "num_docs_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
        "max_effective_docs": max((item["num_docs"] for item in vector), default=0),
    }


def deduplicate_budget_vectors(vectors: Mapping[int, str]) -> dict[int, int]:
    """Map each budget to the first equivalent canonical point, preferring k=4."""
    _require(set(vectors) == set(BUDGETS), "Evidence-vector map must contain k=2,3,4")
    canonical_by_hash: dict[str, int] = {str(vectors[4]): 4}
    result = {4: 4}
    for budget in (2, 3):
        fingerprint = str(vectors[budget])
        canonical = canonical_by_hash.get(fingerprint)
        if canonical is None:
            canonical_by_hash[fingerprint] = budget
            canonical = budget
        result[budget] = canonical
    return result


def _generator_input_fingerprint(
    generator: Mapping[str, Any], native_evaluation: Mapping[str, Any]
) -> str:
    contract = native_evaluation["manifest"]["contract"]
    identity = generator["generator_identity"]
    payload = {
        "generator_identity": identity,
        "generator_manifest_sha256": generator["manifest_sha256"],
        "config_sha256": contract.get("config_sha256"),
        "prompt_hash": contract.get("prompt_hash"),
        "verbalizer_hash": contract.get("verbalizer_hash"),
        "generator_model": contract.get("generator_model"),
        "generator_sha256": contract.get("generator_sha256"),
        "generator_revision": contract.get("generator_revision"),
        "tokenizer_revision": contract.get("tokenizer_revision"),
        "max_context_tokens": contract.get("max_context_tokens"),
    }
    return stable_hash(payload)


def _evaluation_command(
    config: Mapping[str, Any],
    point: Mapping[str, Any],
    generator: Mapping[str, Any],
) -> list[str]:
    generator_id = str(point["generator_id"])
    generator_spec = _generator_map(config)[generator_id]
    command = [
        str(config["server_python"]),
        _join(str(config["server_project_root"]), "scripts", "07_eval_rag_classification.py"),
        "--config",
        str(generator_spec["config"]),
        "--split",
        SPLIT,
        "--selection",
        str(point["selection_path"]),
        "--selection-manifest",
        str(point["selection_manifest_path"]),
        "--output",
        str(point["output"]["predictions"]),
        "--metrics-output",
        str(point["output"]["metrics"]),
        "--generator-manifest",
        generator_manifest_path(config, str(point["method_id"]), generator_id),
        "--experiment-type",
        BUDGET_FRONTIER,
        "--formal-v2-identity",
        "--method-name",
        str(point["method_id"]),
        "--resume",
    ]
    _require("--max-docs" not in command, "Budget evaluation must consume its native budget selection")
    _require(generator["generator_identity"]["generator_id"] == generator_id, "Generator command identity mismatch")
    return command


def _write_text_atomic(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".tmp")
    partial.write_text(text, encoding="utf-8", newline="\n")
    os.replace(partial, target)


def _write_alias(
    point: Mapping[str, Any],
    *,
    source_point: Mapping[str, Any],
    source_evaluation: Mapping[str, Any],
    vector_sha256: str,
    generator_input_fingerprint: str,
    plan_fingerprint: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": ALIAS_SCHEMA_VERSION,
        "status": "completed",
        "completed": True,
        "point_id": point["point_id"],
        "source_point_id": source_point["point_id"],
        "dataset_id": DATASET_ID,
        "generator_id": point["generator_id"],
        "method_id": point["method_id"],
        "budget_k": point["budget_k"],
        "source_budget_k": source_point["budget_k"],
        "source_or_alias": "native_k4_alias" if source_point["budget_k"] == 4 else "evidence_vector_alias",
        "plan_fingerprint": plan_fingerprint,
        "evidence_vector_sha256": vector_sha256,
        "generator_input_fingerprint": generator_input_fingerprint,
        "selection_path": point["selection_path"],
        "selection_sha256": sha256_file(point["selection_path"]),
        "selection_manifest_path": point["selection_manifest_path"],
        "selection_manifest_sha256": sha256_file(point["selection_manifest_path"]),
        "source_evaluation": {
            "predictions_path": source_evaluation["paths"]["predictions"],
            "predictions_sha256": source_evaluation["predictions_sha256"],
            "metrics_path": source_evaluation["paths"]["metrics"],
            "metrics_sha256": source_evaluation["metrics_sha256"],
            "manifest_path": source_evaluation["paths"]["manifest"],
            "manifest_sha256": source_evaluation["manifest_sha256"],
        },
        "checks": {
            "ordered_sample_ids": True,
            "ordered_document_ids": True,
            "document_text_and_rendering": True,
            "generator_identity": True,
            "prompt_and_verbalizer": True,
            "tokenizer_identity": True,
            "evaluation_binding": True,
            "source_checksums": True,
        },
        "created_at": utc_now(),
    }
    payload["fingerprint"] = stable_hash(payload)
    atomic_write_json(point["output"]["alias_manifest"], payload)
    return payload


def audit_selections(
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    context = preflight_sources(
        config, include_evaluations=True, causal_replay_required=False
    )
    point_rows: dict[str, list[dict[str, Any]]] = {}
    point_vectors: dict[str, str] = {}
    point_stats: dict[str, dict[str, Any]] = {}
    selection_provenance: dict[str, dict[str, Any]] = {}
    for point in plan["points"]:
        provenance = validate_selection_provenance(
            point["selection_path"],
            point["selection_manifest_path"],
            formal_v2=True,
            expected_dataset_id=DATASET_ID,
            expected_method=point["method_id"],
        )
        _validate_contract_inputs(provenance["manifest"], label=point["point_id"])
        rows = _rows(point["selection_path"])
        _validate_selection_rows(
            rows,
            context["retrieval_rows"],
            method_id=point["method_id"],
            label=point["point_id"],
        )
        if point["budget_k"] < 4:
            _require(all(int(row["max_docs"]) == point["budget_k"] for row in rows), f"Budget max_docs mismatch for {point['point_id']}")
            _validate_materialized_budget_rows(point, rows, context, provenance)
        vector, stats = evidence_vector(rows)
        point_rows[point["point_id"]] = rows
        point_vectors[point["point_id"]] = vector
        point_stats[point["point_id"]] = stats
        selection_provenance[point["point_id"]] = provenance

    audited_points: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for generator_id in GENERATORS:
        for method_id in METHODS:
            generator = context["generators"][(method_id, generator_id)]
            group = {
                int(point["budget_k"]): point
                for point in plan["points"]
                if point["generator_id"] == generator_id and point["method_id"] == method_id
            }
            native_point = group[4]
            native_eval = context["evaluations"][(method_id, generator_id)]
            native_token_record = context["token_records"][(method_id, generator_id)]
            input_fingerprint = _generator_input_fingerprint(generator, native_eval)
            decisions = deduplicate_budget_vectors(
                {
                    budget: point_vectors[group[budget]["point_id"]]
                    for budget in BUDGETS
                }
            )
            native_alias = _write_alias(
                native_point,
                source_point=native_point,
                source_evaluation=native_eval,
                vector_sha256=point_vectors[native_point["point_id"]],
                generator_input_fingerprint=input_fingerprint,
                plan_fingerprint=plan["plan_fingerprint"],
            )
            audited_points.append(
                {
                    **native_point,
                    "evidence_vector_sha256": point_vectors[native_point["point_id"]],
                    "evidence_stats": point_stats[native_point["point_id"]],
                    "generator_input_fingerprint": input_fingerprint,
                    "requires_evaluation": False,
                    "canonical_point_id": native_point["point_id"],
                    "source_or_alias": "native_k4_alias",
                    "resolved_metrics_path": native_eval["paths"]["metrics"],
                    "resolved_metric_values": native_token_record,
                    "alias_manifest": native_point["output"]["alias_manifest"],
                    "alias_fingerprint": native_alias["fingerprint"],
                }
            )
            for budget in (2, 3):
                point = group[budget]
                vector = point_vectors[point["point_id"]]
                canonical_budget = decisions[budget]
                canonical = group[canonical_budget]
                if canonical_budget == budget:
                    command = _evaluation_command(config, point, generator)
                    job = {
                        "job_id": point["point_id"],
                        "point_id": point["point_id"],
                        "generator_id": generator_id,
                        "method_id": method_id,
                        "budget_k": budget,
                        "selection_path": point["selection_path"],
                        "selection_sha256": sha256_file(point["selection_path"]),
                        "selection_manifest_path": point["selection_manifest_path"],
                        "selection_manifest_sha256": sha256_file(point["selection_manifest_path"]),
                        "generator_manifest_path": generator["manifest_path"],
                        "generator_manifest_sha256": generator["manifest_sha256"],
                        "generator_config_path": _generator_map(config)[generator_id]["config"],
                        "generator_config_sha256": sha256_file(_generator_map(config)[generator_id]["config"]),
                        "evidence_vector_sha256": vector,
                        "generator_input_fingerprint": input_fingerprint,
                        "output": point["output"],
                        "command": command,
                    }
                    jobs.append(job)
                    audited_points.append(
                        {
                            **point,
                            "evidence_vector_sha256": vector,
                            "evidence_stats": point_stats[point["point_id"]],
                            "generator_input_fingerprint": input_fingerprint,
                            "requires_evaluation": True,
                            "canonical_point_id": point["point_id"],
                            "source_or_alias": "new_evaluation",
                            "resolved_metrics_path": point["output"]["metrics"],
                            "resolved_metric_values": None,
                            "alias_manifest": None,
                        }
                    )
                else:
                    alias = None
                    if canonical["budget_k"] == 4:
                        alias = _write_alias(
                            point,
                            source_point=canonical,
                            source_evaluation=native_eval,
                            vector_sha256=vector,
                            generator_input_fingerprint=input_fingerprint,
                            plan_fingerprint=plan["plan_fingerprint"],
                        )
                        resolved_metrics_path = native_eval["paths"]["metrics"]
                        resolved_values = native_token_record
                        source_kind = "evidence_vector_alias"
                        alias_manifest = point["output"]["alias_manifest"]
                    else:
                        resolved_metrics_path = canonical["output"]["metrics"]
                        resolved_values = None
                        source_kind = "planned_evidence_vector_alias"
                        alias_manifest = None
                    audited_points.append(
                        {
                            **point,
                            "evidence_vector_sha256": vector,
                            "evidence_stats": point_stats[point["point_id"]],
                            "generator_input_fingerprint": input_fingerprint,
                            "requires_evaluation": False,
                            "canonical_point_id": canonical["point_id"],
                            "source_or_alias": source_kind,
                            "resolved_metrics_path": resolved_metrics_path,
                            "resolved_metric_values": resolved_values,
                            "alias_manifest": alias_manifest,
                            "alias_fingerprint": alias["fingerprint"] if alias else None,
                        }
                    )

    audited_points.sort(key=lambda item: (GENERATORS.index(item["generator_id"]), METHODS.index(item["method_id"]), item["budget_k"]))
    jobs.sort(key=lambda item: (GENERATORS.index(item["generator_id"]), METHODS.index(item["method_id"]), item["budget_k"]))
    audit: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "completed",
        "completed": True,
        "dataset_id": DATASET_ID,
        "split": SPLIT,
        "plan_fingerprint": plan["plan_fingerprint"],
        "preflight": context["report"],
        "logical_point_count": len(audited_points),
        "new_selection_count": len(plan["selection_tasks"]),
        "native_k4_alias_count": sum(point["budget_k"] == 4 for point in audited_points),
        "evidence_vector_alias_count": sum(point["source_or_alias"] == "evidence_vector_alias" for point in audited_points),
        "evaluation_job_count": len(jobs),
        "points": audited_points,
        "created_at": utc_now(),
    }
    audit["audit_fingerprint"] = stable_hash(audit)
    audit_path = _join(str(plan["artifact_root"]), "audit", "selection_audit.json")
    job_plan: dict[str, Any] = {
        "schema_version": JOB_PLAN_SCHEMA_VERSION,
        "status": "frozen",
        "dataset_id": DATASET_ID,
        "plan_fingerprint": plan["plan_fingerprint"],
        "audit_path": audit_path,
        "audit_fingerprint": audit["audit_fingerprint"],
        "evaluation_job_count": len(jobs),
        "jobs": jobs,
    }
    job_plan["job_plan_fingerprint"] = stable_hash(job_plan)
    job_path = _join(str(plan["artifact_root"]), "audit", "evaluation_jobs.json")
    commands_path = _join(str(plan["artifact_root"]), "audit", "evaluation_commands.sh")
    atomic_write_json(audit_path, audit)
    atomic_write_json(job_path, job_plan)
    _write_text_atomic(commands_path, "".join(shlex.join(job["command"]) + "\n" for job in jobs))
    return {"audit": audit, "job_plan": job_plan, "audit_path": audit_path, "job_plan_path": job_path, "commands_path": commands_path}


def _load_fingerprinted(path: str | Path, schema: str, fingerprint_field: str) -> dict[str, Any]:
    payload = _load_json(path, schema)
    _require(payload.get("schema_version") == schema, f"Schema mismatch for {path}")
    recorded = payload.get(fingerprint_field)
    without = dict(payload)
    without.pop(fingerprint_field, None)
    _require(recorded == stable_hash(without), f"Fingerprint mismatch for {path}")
    return payload


def validate_alias_manifest(
    path: str | Path, *, expected_point_id: str | None = None
) -> dict[str, Any]:
    alias = _load_fingerprinted(path, ALIAS_SCHEMA_VERSION, "fingerprint")
    _require(alias.get("status") == "completed" and alias.get("completed") is True, "Alias is incomplete")
    if expected_point_id is not None:
        _require(alias.get("point_id") == expected_point_id, "Alias point identity mismatch")
    _require(sha256_file(alias["selection_path"]) == alias["selection_sha256"], "Alias selection SHA changed")
    _require(
        sha256_file(alias["selection_manifest_path"])
        == alias["selection_manifest_sha256"],
        "Alias selection manifest SHA changed",
    )
    source = alias.get("source_evaluation")
    _require(isinstance(source, dict), "Alias lacks source evaluation")
    for label in ("predictions", "metrics", "manifest"):
        _require(
            sha256_file(source[f"{label}_path"]) == source[f"{label}_sha256"],
            f"Alias source {label} SHA changed",
        )
    return alias


def validate_job_plan_inputs(job_plan_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = _load_fingerprinted(job_plan_path, JOB_PLAN_SCHEMA_VERSION, "job_plan_fingerprint")
    _require(plan.get("status") == "frozen", "Evaluation job plan is not frozen")
    audit = _load_fingerprinted(plan["audit_path"], AUDIT_SCHEMA_VERSION, "audit_fingerprint")
    _require(audit["audit_fingerprint"] == plan["audit_fingerprint"], "Evaluation audit fingerprint changed")
    _require(len(plan["jobs"]) == plan["evaluation_job_count"], "Evaluation job count mismatch")
    for job in plan["jobs"]:
        for path_field, sha_field in (
            ("selection_path", "selection_sha256"),
            ("selection_manifest_path", "selection_manifest_sha256"),
            ("generator_manifest_path", "generator_manifest_sha256"),
            ("generator_config_path", "generator_config_sha256"),
        ):
            _require(
                sha256_file(job[path_field]) == job[sha_field],
                f"Frozen job input changed: {job['job_id']} {path_field}",
            )
    return plan, audit


def run_evaluations(job_plan_path: str | Path) -> None:
    _require(sys.platform.startswith("linux"), "Formal generator evaluation is Linux-server only")
    plan, _audit = validate_job_plan_inputs(job_plan_path)
    for job in plan["jobs"]:
        result = subprocess.run(job["command"], check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Budget evaluation failed with exit {result.returncode}: {job['job_id']}")


SUMMARY_FIELDS = (
    "dataset_id",
    "generator_id",
    "method_id",
    "budget_k",
    "accuracy",
    "macro_f1",
    "avg_num_docs",
    "avg_evidence_tokens",
    "selection_path",
    "evaluation_path",
    "source_or_alias",
    "analysis_status",
)


def summarize(audit_path: str | Path, *, output_csv: str | Path, output_manifest: str | Path) -> dict[str, Any]:
    audit = _load_fingerprinted(audit_path, AUDIT_SCHEMA_VERSION, "audit_fingerprint")
    _require(len(audit.get("points", [])) == 30, "Audit does not contain 30 points")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for point in audit["points"]:
        if point.get("alias_manifest"):
            validate_alias_manifest(
                point["alias_manifest"], expected_point_id=point["point_id"]
            )
        metrics_path = Path(point["resolved_metrics_path"])
        metrics = _load_json(metrics_path, "budget metrics")
        _require(metrics.get("num_examples") == EXPECTED_ROWS, f"Metrics row count mismatch for {point['point_id']}")
        resolved = point.get("resolved_metric_values") or metrics
        _require(resolved.get("evidence_tokens_basis") == TOKEN_BASIS, f"Evidence-token basis mismatch for {point['point_id']}")
        for field in ("accuracy", "macro_f1", "avg_num_docs", "avg_evidence_tokens"):
            _require(isinstance(resolved.get(field), (int, float)), f"Metrics lack {field} for {point['point_id']}")
        rows.append(
            {
                "dataset_id": DATASET_ID,
                "generator_id": point["generator_id"],
                "method_id": point["method_id"],
                "budget_k": point["budget_k"],
                "accuracy": resolved["accuracy"],
                "macro_f1": resolved["macro_f1"],
                "avg_num_docs": resolved["avg_num_docs"],
                "avg_evidence_tokens": resolved["avg_evidence_tokens"],
                "selection_path": point["selection_path"],
                "evaluation_path": str(metrics_path),
                "source_or_alias": point["source_or_alias"],
                "analysis_status": "completed",
            }
        )
        sources.append(
            {
                "point_id": point["point_id"],
                "metrics_path": str(metrics_path.resolve()),
                "metrics_sha256": sha256_file(metrics_path),
                "selection_sha256": sha256_file(point["selection_path"]),
                "evidence_vector_sha256": point["evidence_vector_sha256"],
                "source_or_alias": point["source_or_alias"],
            }
        )
    target = Path(output_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".tmp")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, target)
    manifest: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "completed",
        "completed": True,
        "dataset_id": DATASET_ID,
        "split": SPLIT,
        "audit_path": str(Path(audit_path).resolve()),
        "audit_fingerprint": audit["audit_fingerprint"],
        "row_count": len(rows),
        "output_csv": str(target.resolve()),
        "output_csv_sha256": sha256_file(target),
        "evidence_tokens_basis": TOKEN_BASIS,
        "sources": sources,
        "created_at": utc_now(),
    }
    manifest["fingerprint"] = stable_hash(manifest)
    atomic_write_json(output_manifest, manifest)
    return manifest

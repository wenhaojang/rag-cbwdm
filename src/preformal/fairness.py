"""Fail-closed artifact comparability checks for the preformal benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.io_utils import read_jsonl
from src.preformal.registry import PREFORMAL_METHODS
from src.preformal.statistics import indexed_rows, require_identical_ids
from src.run_manifest import sha256_file


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit_preformal(*, split_manifest_path: str | Path, retrieval_manifest_path: str | Path,
                    posterior_manifest_path: str | Path, selection_manifests: dict[str, Path],
                    evaluation_manifests: dict[str, Path], required_methods: set[str] | None = None) -> dict[str, Any]:
    split = _load(split_manifest_path); retrieval = _load(retrieval_manifest_path); posterior = _load(posterior_manifest_path)
    reasons: list[str] = []
    required = required_methods if required_methods is not None else {
        method for method, metadata in PREFORMAL_METHODS.items()
        if metadata.get("deployable")
    }
    supplied_methods = {key.split(":", 1)[0] for key in selection_manifests}
    reasons.extend(f"missing deployable method: {method}" for method in sorted(required - supplied_methods))
    supplied_keys = set(selection_manifests)
    for learned_method in ({"rag_cbwdm_signed_v1", "infogain_fever"} & required):
        expected = {f"{learned_method}:{seed}" for seed in (13, 21, 42)}
        reasons.extend(f"missing required learned-method seed: {key}" for key in sorted(expected - supplied_keys))
    split_sha = split.get("preformal_eval_sha256"); retrieval_sha = retrieval.get("output_sha256"); posterior_sha = posterior.get("output_sha256")
    if retrieval.get("query_input_sha256") != split_sha: reasons.append("retrieval query SHA differs from preformal split SHA")
    if posterior.get("provenance", {}).get("input_sha256") != retrieval_sha: reasons.append("posterior input SHA differs from retrieval SHA")
    prompt_hash = posterior.get("provenance", {}).get("prompt_template_hash"); verbalizer_hash = posterior.get("provenance", {}).get("verbalizers_hash")
    generator_sha = posterior.get("provenance", {}).get("generator_sha256")
    split_rows = indexed_rows(read_jsonl(split["preformal_eval_path"]), "preformal split")
    methods: dict[str, Any] = {}
    for key, selection_path in sorted(selection_manifests.items()):
        method = key.split(":", 1)[0]; method_reasons: list[str] = []
        if method not in PREFORMAL_METHODS: method_reasons.append(f"unknown method {method}")
        selection = _load(selection_path); contract = selection.get("contract", {}); inputs = contract.get("inputs", {})
        if selection.get("stage") != "selection" or selection.get("status") != "completed": method_reasons.append("selection manifest is not completed")
        if selection.get("method") != method: method_reasons.append("selection manifest method mismatch")
        input_shas = {value.get("sha256") for value in inputs.values() if isinstance(value, dict)}
        if retrieval_sha not in input_shas and posterior_sha not in input_shas:
            method_reasons.append("selection does not consume canonical retrieval or shared posterior SHA")
        selection_output = Path(selection["output_path"])
        if selection.get("output_sha256") != sha256_file(selection_output): method_reasons.append("selection output checksum mismatch")
        selection_rows = indexed_rows(read_jsonl(selection_output), f"{key} selection")
        if any(row.get("split") != "preformal_eval" for row in selection_rows.values()): method_reasons.append("selection contains a non-preformal row")
        if any(row.get("uses_gold_at_test") is not False for row in selection_rows.values()): method_reasons.append("selection does not explicitly prove uses_gold_at_test=false")
        try: require_identical_ids(split_rows, selection_rows, "preformal split", f"{key} selection")
        except ValueError as exc: method_reasons.append(str(exc))
        evaluation_path = evaluation_manifests.get(key)
        if evaluation_path is None: method_reasons.append("missing evaluation manifest")
        else:
            evaluation = _load(evaluation_path); eval_contract = evaluation.get("contract", {})
            if evaluation.get("stage") != "evaluation" or evaluation.get("status") != "completed": method_reasons.append("evaluation manifest is not completed")
            if evaluation.get("method") != method: method_reasons.append("evaluation manifest method mismatch")
            if eval_contract.get("selection_sha256") != selection.get("output_sha256"): method_reasons.append("evaluation selection SHA mismatch")
            if eval_contract.get("split") != "preformal_eval": method_reasons.append("evaluation split is not preformal_eval")
            if eval_contract.get("prompt_hash") != prompt_hash: method_reasons.append("evaluation prompt SHA differs from shared posterior prompt SHA")
            if eval_contract.get("verbalizer_hash") != verbalizer_hash: method_reasons.append("evaluation verbalizer SHA differs from shared posterior verbalizer SHA")
            if eval_contract.get("generator_sha256") != generator_sha: method_reasons.append("evaluation generator SHA differs from shared posterior generator SHA")
            predictions = indexed_rows(read_jsonl(evaluation["predictions_path"]), f"{key} predictions")
            try: require_identical_ids(split_rows, predictions, "preformal split", f"{key} predictions")
            except ValueError as exc: method_reasons.append(str(exc))
        methods[key] = {"method": method, "status": "comparable" if not method_reasons else "blocked", "reasons": method_reasons,
            "effective_retrieval_sha256": retrieval_sha, "posterior_sha256": posterior_sha, "generator_sha256": generator_sha, "prompt_hash": prompt_hash, "verbalizer_hash": verbalizer_hash}
        reasons.extend(f"{key}: {reason}" for reason in method_reasons)
    missing_evaluations = sorted(set(evaluation_manifests) - set(selection_manifests))
    reasons.extend(f"evaluation without selection: {key}" for key in missing_evaluations)
    return {"schema_version": "rag_cbwdm_preformal_fairness.v1", "status": "comparable" if not reasons else "blocked",
        "split_sha256": split_sha, "retrieval_sha256": retrieval_sha, "posterior_sha256": posterior_sha,
        "generator_sha256": generator_sha, "prompt_hash": prompt_hash, "verbalizer_hash": verbalizer_hash, "methods": methods, "reasons": reasons,
        "held_out_test_consumed": False}

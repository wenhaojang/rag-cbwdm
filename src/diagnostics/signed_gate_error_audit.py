"""Read-only residual-error analysis for the signed-gate validation oracle."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.cbwdm_score import build_local_effects, marginal_gain
from src.diagnostics.method_failure import evidence_coverage, summary
from src.formal_provenance import atomic_write_text
from src.io_utils import read_jsonl
from src.run_manifest import atomic_write_json, git_state, sha256_file, utc_now


def _map(path: str | Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        identifier = str(row["id"])
        if identifier in rows:
            raise ValueError(f"Duplicate id in {path}: {identifier}")
        rows[identifier] = row
    return rows


def _sign(value: float, eps: float) -> str:
    return "positive" if value > eps else "negative" if value < -eps else "zero"


def _histogram(values: Iterable[int]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(Counter(values).items())}


def _candidate_key(candidate: dict[str, Any]) -> str | None:
    meta = candidate.get("meta")
    if not isinstance(meta, dict) or meta.get("page_id") is None or meta.get("sentence_id") is None:
        return None
    return f"{meta['page_id']}\t{meta['sentence_id']}"


def _confusion(rows: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    matrix = {gold: {pred: 0 for pred in labels} for gold in labels}
    distribution: Counter[str] = Counter()
    correct = 0
    for row in rows:
        gold, pred = str(row["gold"]), str(row["pred"])
        if gold in matrix and pred in matrix[gold]:
            matrix[gold][pred] += 1
        distribution[pred] += 1
        correct += int(gold == pred)
    return {"num_examples": len(rows), "accuracy": correct / len(rows) if rows else None,
            "confusion_matrix": matrix, "prediction_distribution": dict(distribution)}


def _stratified_zero_doc(rows: list[dict[str, Any]], predictions: dict[str, dict[str, Any]],
                         total_by_label: Counter[str]) -> dict[str, Any]:
    result = {}
    for group, subset in [("ALL", rows)] + [
        (f"gold={label}", [row for row in rows if row["gold"] == label])
        for label in ("SUPPORTS", "REFUTES")
    ]:
        prediction_rows = [predictions[str(row["id"])] for row in subset]
        denominator = sum(total_by_label.values()) if group == "ALL" else total_by_label[group.split("=", 1)[1]]
        result[group] = {"zero_doc_count": len(subset),
                         "zero_doc_ratio": len(subset) / denominator if denominator else None,
                         "query_only": _confusion(prediction_rows, ["SUPPORTS", "REFUTES"]),
                         "positive_alignment_candidate_count": {
                             **summary([row["positive_count"] for row in subset]),
                             "histogram": _histogram(row["positive_count"] for row in subset),
                         },
                         "zero_doc_reason_counts": dict(Counter(row["reason"] for row in subset))}
    return result


def _gold_alignment_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    signs = Counter(item["alignment_sign"] for item in items)
    count = len(items)
    return {"gold_candidate_count": count,
            **{f"{name}_alignment_count": signs[name] for name in ("positive", "zero", "negative")},
            **{f"{name}_alignment_ratio": signs[name] / count if count else None
               for name in ("positive", "zero", "negative")},
            "signed_alignment": summary([item["alignment"] for item in items]),
            "delta_gold_probability": summary([item["delta_gold_probability"] for item in items])}


def _production_controls(supplement: dict[str, Any]) -> dict[str, Any]:
    audit = supplement.get("candidate_audit", {})
    controls = []
    for candidate in audit.get("candidates", []) if isinstance(audit, dict) else []:
        controls.append({"candidate_id": candidate.get("candidate_id"),
            "record_parameters": candidate.get("record_parameters"),
            "selection_parameters": candidate.get("selection_parameters"),
            "metrics": candidate.get("metrics"),
            "generator_statistics": candidate.get("generator_statistics")})
    return {"status": "AVAILABLE" if controls else "UNAVAILABLE", "rag_cbwdm_candidates": controls,
            "two_candidate_parameter_differences": audit.get("two_candidate_parameter_differences") if isinstance(audit, dict) else None}


def build_audit(*, trajectories_path: Path, selection_path: Path, predictions_path: Path,
                metrics_path: Path, retrieval_path: Path, posteriors_path: Path,
                manifest_path: Path, config: dict[str, Any], supplement_path: Path | None = None) -> dict[str, Any]:
    trajectories, selections = _map(trajectories_path), _map(selection_path)
    predictions, retrieval, posteriors = _map(predictions_path), _map(retrieval_path), _map(posteriors_path)
    collections = [set(value) for value in (trajectories, selections, predictions, retrieval, posteriors)]
    common = set.intersection(*collections)
    if any(ids != common for ids in collections):
        raise ValueError("Signed-gate audit inputs do not have identical id sets")
    signed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_params = signed_manifest.get("contract", {}).get("parameters", {})
    cbwdm = {**config["cbwdm"], **manifest_params}
    eps = float(manifest_params.get("alignment_eps", 0.0))
    total_by_label: Counter[str] = Counter()
    zero_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    all_gold_candidates: list[dict[str, Any]] = []
    selection_quality: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "doc_counts": [], "stops": Counter(), "zero": 0, "coverage": [], "positive_available": []})

    for identifier in sorted(common):
        trajectory, selection = trajectories[identifier], selections[identifier]
        prediction, ret, posterior = predictions[identifier], retrieval[identifier], posteriors[identifier]
        label = str(prediction.get("gold", posterior.get("label")))
        total_by_label[label] += 1
        candidates = list(posterior.get("candidates", []))
        if not candidates:
            X = np.empty((0, len(posterior["labels"])), dtype=float)
            d = np.zeros(len(posterior["labels"]), dtype=float)
        else:
            X, d = build_local_effects(np.asarray(posterior["eta0"]),
                np.asarray([candidate["eta"] for candidate in candidates]), label,
                list(posterior["labels"]), cbwdm.get("L_type", "euclidean_posterior_shift"),
                float(cbwdm.get("eps_smooth", 0.0)), cbwdm.get("target_smoothing", "paper_mixture"))
        index = {str(candidate["doc_id"]): idx for idx, candidate in enumerate(candidates)}
        alignments = {doc_id: float(X[idx] @ d) for doc_id, idx in index.items()}
        positive_count = sum(value > eps for value in alignments.values())
        selected_ids = list(map(str, selection.get("selected_doc_ids", [])))
        coverage = evidence_coverage(ret, selected_ids)
        for group in ("ALL", f"gold={label}"):
            bucket = selection_quality[group]
            bucket["doc_counts"].append(len(selected_ids)); bucket["stops"][str(trajectory.get("stop_reason"))] += 1
            bucket["zero"] += int(not selected_ids); bucket["coverage"].append(coverage)
            bucket["positive_available"].append(int(positive_count > 0))

        if not selected_ids:
            if positive_count == 0:
                reason = "no_positive_alignment_candidate"
            elif trajectory.get("stop_reason") == "gain_below_threshold":
                reason = "positive_alignment_exists_but_gain_below_stop"
            else:
                reason = f"other:{trajectory.get('stop_reason', 'missing')}"
            zero_rows.append({"id": identifier, "gold": label, "positive_count": positive_count,
                              "reason": reason})

        gold_keys = set(map(str, ret.get("gold_evidence_keys", [])))
        retrieval_by_id = {str(candidate["doc_id"]): candidate for candidate in ret.get("candidates", [])}
        gold_ids = [doc_id for doc_id, candidate in retrieval_by_id.items()
                    if _candidate_key(candidate) in gold_keys]
        gold_items = []
        for doc_id in gold_ids:
            idx = index.get(doc_id)
            if idx is None:
                continue
            alignment = alignments[doc_id]
            gold_index = list(posterior["labels"]).index(label)
            eta_gold = float(candidates[idx]["eta"][gold_index])
            delta_gold = eta_gold - float(posterior["eta0"][gold_index])
            singleton_gain, singleton_theta = marginal_gain(X, d, [], idx, float(cbwdm["ridge_lambda"]))
            item = {"id": identifier, "gold": label, "candidate_id": doc_id,
                    "rank": retrieval_by_id[doc_id].get("rank"), "alignment": alignment,
                    "alignment_sign": _sign(alignment, eps), "eta_j_gold": eta_gold,
                    "delta_gold_probability": delta_gold, "theta_singleton": singleton_theta,
                    "step0_marginal": singleton_gain, "selected": doc_id in selected_ids}
            gold_items.append(item); all_gold_candidates.append(item)

        if label == "SUPPORTS" and str(prediction.get("pred")) == "REFUTES":
            eta0 = list(map(float, posterior["eta0"])); labels = list(posterior["labels"])
            query_pred = labels[max(range(len(labels)), key=lambda idx: eta0[idx])]
            ranks = [retrieval_by_id[doc_id].get("rank") for doc_id in selected_ids if doc_id in retrieval_by_id]
            any_gold = bool(gold_ids); positive_gold = [item for item in gold_items if item["alignment_sign"] == "positive"]
            selected_gold = [item for item in gold_items if item["selected"]]
            if not selected_ids:
                taxonomy = "F5_zero_doc_query_only_failure"
            elif not any_gold:
                taxonomy = "F1_no_gold_evidence_in_bm25_top20"
            elif not positive_gold:
                taxonomy = "F2_all_gold_evidence_alignment_nonpositive"
            elif selected_gold:
                taxonomy = "F4_gold_evidence_selected_generator_still_wrong"
            else:
                taxonomy = "F3_positive_aligned_gold_evidence_not_selected"
            residual_rows.append({"id": identifier, "selected_doc_count": len(selected_ids),
                "query_only_prediction": query_pred,
                "query_only_probabilities": {name: eta0[idx] for idx, name in enumerate(labels)},
                "bm25_any_gold_evidence_hit": any_gold, "gold_evidence_count": len(gold_items),
                "best_gold_evidence_bm25_rank": min((item["rank"] for item in gold_items if item["rank"] is not None), default=None),
                "selection_any_gold_evidence": bool(selected_gold),
                "complete_flattened_gold_union_coverage": coverage.get("complete_flattened_gold_key_union_covered"),
                "selected_original_bm25_ranks": ranks, "positive_alignment_candidate_count": positive_count,
                "gold_candidates": gold_items, "taxonomy": taxonomy})

    zero_audit = _stratified_zero_doc(zero_rows, predictions, total_by_label)
    gold_alignment = {}
    for group, items in [("ALL", all_gold_candidates)] + [
        (f"gold={label}", [item for item in all_gold_candidates if item["gold"] == label])
        for label in ("SUPPORTS", "REFUTES")]:
        gold_alignment[group] = _gold_alignment_summary(items)

    residual_doc_counts = Counter(row["selected_doc_count"] for row in residual_rows)
    residual_gold_items = [item for row in residual_rows for item in row["gold_candidates"]]
    residual_selected_ranks = [float(rank) for row in residual_rows for rank in row["selected_original_bm25_ranks"] if rank is not None]
    supports_probs = [row["query_only_probabilities"].get("SUPPORTS") for row in residual_rows]
    refutes_probs = [row["query_only_probabilities"].get("REFUTES") for row in residual_rows]
    residual_aggregate = {"selected_doc_count_distribution": {str(k): v for k, v in sorted(residual_doc_counts.items())},
        "query_only_prediction_distribution": dict(Counter(row["query_only_prediction"] for row in residual_rows)),
        "query_only_p_supports": summary([value for value in supports_probs if value is not None]),
        "query_only_p_refutes": summary([value for value in refutes_probs if value is not None]),
        "bm25_any_gold_evidence_hit_count": sum(row["bm25_any_gold_evidence_hit"] for row in residual_rows),
        "bm25_any_gold_evidence_hit_ratio": sum(row["bm25_any_gold_evidence_hit"] for row in residual_rows)/len(residual_rows) if residual_rows else None,
        "gold_evidence_candidate_count_per_query": summary([row["gold_evidence_count"] for row in residual_rows]),
        "best_gold_evidence_bm25_rank": summary([row["best_gold_evidence_bm25_rank"] for row in residual_rows if row["best_gold_evidence_bm25_rank"] is not None]),
        "selection_any_gold_evidence_count": sum(row["selection_any_gold_evidence"] for row in residual_rows),
        "selection_any_gold_evidence_ratio": sum(row["selection_any_gold_evidence"] for row in residual_rows)/len(residual_rows) if residual_rows else None,
        "complete_flattened_gold_union_coverage_ratio": sum(bool(row["complete_flattened_gold_union_coverage"]) for row in residual_rows)/len(residual_rows) if residual_rows else None,
        "selected_original_bm25_rank": summary(residual_selected_ranks),
        "positive_alignment_candidate_count": summary([row["positive_alignment_candidate_count"] for row in residual_rows]),
        "retrieved_gold_candidate_diagnostics": {"count":len(residual_gold_items),
            "alignment":summary([item["alignment"] for item in residual_gold_items]),
            "alignment_sign_counts":dict(Counter(item["alignment_sign"] for item in residual_gold_items)),
            "eta_j_gold":summary([item["eta_j_gold"] for item in residual_gold_items]),
            "delta_gold_probability":summary([item["delta_gold_probability"] for item in residual_gold_items]),
            "theta_singleton":summary([item["theta_singleton"] for item in residual_gold_items]),
            "step0_marginal":summary([item["step0_marginal"] for item in residual_gold_items])}}
    nonexclusive = {
        "zero_doc": sum(row["selected_doc_count"] == 0 for row in residual_rows),
        "no_gold_evidence_in_bm25_top20": sum(not row["bm25_any_gold_evidence_hit"] for row in residual_rows),
        "all_gold_evidence_alignment_nonpositive": sum(row["bm25_any_gold_evidence_hit"] and not any(
            item["alignment_sign"] == "positive" for item in row["gold_candidates"]) for row in residual_rows),
        "positive_aligned_gold_evidence_not_selected": sum(any(item["alignment_sign"] == "positive" for item in row["gold_candidates"])
            and not row["selection_any_gold_evidence"] for row in residual_rows),
        "gold_evidence_selected_generator_still_wrong": sum(row["selection_any_gold_evidence"] for row in residual_rows),
    }
    quality = {}
    for group, bucket in sorted(selection_quality.items()):
        available = [item for item in bucket["coverage"] if item.get("status") == "AVAILABLE"]
        n = len(bucket["doc_counts"])
        quality[group] = {"query_count": n, "selected_docs_per_query": {**summary(bucket["doc_counts"]),
            "histogram": _histogram(bucket["doc_counts"])}, "stop_reason_counts": dict(bucket["stops"]),
            "zero_doc_ratio": bucket["zero"] / n if n else None,
            "gold_evidence_any_hit_ratio": sum(item["any_hit"] for item in available) / len(available) if available else None,
            "positive_aligned_candidate_availability_ratio": sum(bucket["positive_available"]) / n if n else None}

    supplement = None
    if supplement_path and supplement_path.is_file():
        supplement = json.loads(supplement_path.read_text(encoding="utf-8"))
    return {"schema_version": "signed_gate_full500_error_audit.v1", "created_at": utc_now(),
        "read_only": True, "alignment_eps": eps, "num_aligned_examples": len(common),
        "source_metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
        "signed_gate_contract": signed_manifest.get("contract"),
        "zero_doc_audit": zero_audit,
        "supports_to_refutes_residual": {"count": len(residual_rows),
            "aggregate": residual_aggregate,
            "taxonomy_priority": ["F5", "F1", "F2", "F4", "F3"],
            "taxonomy_counts": dict(Counter(row["taxonomy"] for row in residual_rows)),
            "nonexclusive_counts": nonexclusive, "examples": residual_rows},
        "gold_evidence_alignment_audit": gold_alignment,
        "signed_gate_selection_quality": quality,
        "production_controls_from_existing_supplement": _production_controls(supplement) if supplement is not None else {
            "status": "UNAVAILABLE", "missing": str(supplement_path) if supplement_path else "supplement path not provided",
            "generation": "reuse the existing method-failure server supplement; do not rerun production"}}


def render_markdown(payload: dict[str, Any]) -> str:
    residual = payload["supports_to_refutes_residual"]
    lines = ["# Signed-Gate Full500 Residual Error Audit", "",
        f"- Read only: `{payload['read_only']}`", f"- Aligned validation examples: {payload['num_aligned_examples']}",
        f"- SUPPORTS→REFUTES residuals: {residual['count']}", "", "## Zero-document audit", "", "```json",
        json.dumps(payload["zero_doc_audit"], ensure_ascii=False, indent=2, sort_keys=True), "```", "",
        "## SUPPORTS→REFUTES taxonomy", "",
        "Priority is F5 zero-doc, then F1 no retrieved gold, F2 all retrieved gold non-positive, "
        "F4 selected gold but still wrong, and F3 positive gold not selected. Non-exclusive counts are also retained.", "", "```json",
        json.dumps({key: residual[key] for key in ("aggregate", "taxonomy_counts", "nonexclusive_counts")},
                   ensure_ascii=False, indent=2, sort_keys=True), "```", "",
        "## Gold-evidence alignment audit", "", "```json",
        json.dumps(payload["gold_evidence_alignment_audit"], ensure_ascii=False, indent=2, sort_keys=True), "```", "",
        "## Signed-gate selection quality", "", "```json",
        json.dumps(payload["signed_gate_selection_quality"], ensure_ascii=False, indent=2, sort_keys=True), "```", "",
        "## Existing production controls", "", "```json",
        json.dumps(payload["production_controls_from_existing_supplement"], ensure_ascii=False, indent=2, sort_keys=True), "```", "",
        "Per-example residual details are retained in the JSON report.", ""]
    return "\n".join(lines)


def publish(payload: dict[str, Any], output_dir: Path, project_root: Path,
            inputs: dict[str, Path]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "SIGNED_GATE_FULL500_ERROR_AUDIT.json"
    md_path = output_dir / "SIGNED_GATE_FULL500_ERROR_AUDIT.md"
    if json_path.exists() or md_path.exists():
        raise FileExistsError("Residual-audit output exists; use a new isolated output directory")
    payload["inputs"] = {name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                         for name, path in sorted(inputs.items()) if path.is_file()}
    payload["git"] = git_state(project_root)
    atomic_write_json(json_path, payload)
    atomic_write_text(md_path, render_markdown(payload))

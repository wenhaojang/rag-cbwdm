"""Diagnostic-only CBWDM variants and artifact helpers.

Nothing in this module is imported by the production teacher, selector, or
evaluator.  The helpers deliberately require output paths below the dedicated
method-failure audit directory.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from src.cbwdm_score import build_local_effects, marginal_gain, theta_for_indices
from src.formal_provenance import atomic_write_text
from src.metrics import ClassificationMetrics
from src.prompts import build_fever_prompt, fever_prompt_hash
from src.run_manifest import atomic_write_json, sha256_file, stable_hash
from src.selection_schema import make_selection_row, normalize_selected_doc, validate_selection_row

DIAGNOSTIC_RELATIVE_ROOT = Path("artifacts/diagnostics/method_failure_audit")
SCHEMA_VERSION = "rag_cbwdm_method_failure_diagnostic.v1"


def require_diagnostic_output(run_dir: str | Path, output: str | Path) -> Path:
    """Return a resolved output path, rejecting production-artifact locations."""
    run = Path(run_dir).resolve()
    root = (run / DIAGNOSTIC_RELATIVE_ROOT).resolve()
    target = Path(output).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Diagnostic output must be below {root}, got {target}") from exc
    if target == (run / "artifacts" / "formal").resolve() or "formal" in target.parts[len(run.parts):]:
        raise ValueError("Diagnostic output cannot be placed in artifacts/formal")
    return target


def atomic_write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    count = 0
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, target)
    return count


def load_evaluator_module(project_root: str | Path) -> Any:
    """Load the production evaluator so diagnostics cannot drift in serialization."""
    path = Path(project_root) / "scripts" / "07_eval_rag_classification.py"
    spec = importlib.util.spec_from_file_location("rag_cbwdm_production_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load production evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summary(values: Sequence[float]) -> dict[str, Any]:
    finite = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not finite:
        return {"count": 0, "mean": None, "std": None, "min": None, "p10": None,
                "p25": None, "median": None, "p75": None, "p90": None, "max": None}
    def q(frac: float) -> float:
        if len(finite) == 1:
            return finite[0]
        pos = frac * (len(finite) - 1)
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        return finite[lo] + (finite[hi] - finite[lo]) * (pos - lo)
    return {
        "count": len(finite), "mean": statistics.fmean(finite),
        "std": statistics.pstdev(finite), "min": finite[0], "p10": q(.10),
        "p25": q(.25), "median": statistics.median(finite), "p75": q(.75),
        "p90": q(.90), "max": finite[-1],
    }


def _rankdata(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        for idx, _ in ordered[start:end]:
            result[idx] = rank
        start = end
    return result


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    x, y = np.asarray(_rankdata(left)), np.asarray(_rankdata(right))
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def signed_admissible(alignments: np.ndarray, alignment_eps: float = 0.0) -> np.ndarray:
    if alignment_eps < 0:
        raise ValueError("alignment_eps must be non-negative")
    return np.asarray(alignments, dtype=float) > float(alignment_eps)


def signed_gated_greedy(
    X_all: np.ndarray,
    d: np.ndarray,
    *,
    top_m: int,
    ridge_lambda: float,
    stop_threshold: float = 0.0,
    alignment_eps: float = 0.0,
    gain_tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Run the production Theta greedy rule only over x dot d > eps candidates."""
    X_all, d = np.asarray(X_all, dtype=float), np.asarray(d, dtype=float)
    alignments = X_all @ d
    admissible = signed_admissible(alignments, alignment_eps)
    selected: list[int] = []
    steps: list[dict[str, Any]] = []
    stop_reason = "top_m_reached" if top_m == 0 else "no_admissible_candidates"
    for step_index in range(top_m):
        remaining = [i for i in range(len(X_all)) if admissible[i] and i not in selected]
        if not remaining:
            stop_reason = "no_admissible_candidates"
            break
        before = theta_for_indices(X_all, d, selected, ridge_lambda)
        gains = []
        for idx in remaining:
            raw_gain, after = marginal_gain(X_all, d, selected, idx, ridge_lambda)
            if raw_gain < -gain_tolerance:
                raise FloatingPointError(f"Negative Theta marginal gain: {raw_gain}")
            gain = 0.0 if abs(raw_gain) <= gain_tolerance else float(raw_gain)
            gains.append({"index": idx, "gain": gain, "raw_gain": float(raw_gain),
                          "theta_after_add": float(after), "alignment": float(alignments[idx])})
        best = max(gains, key=lambda item: (item["gain"], -item["index"]))
        if best["gain"] < stop_threshold:
            stop_reason = "gain_below_threshold"
            break
        steps.append({"step": step_index, "current_indices": list(selected),
                      "theta_before": float(before), "candidate_gains": gains,
                      "best_index": best["index"], "best_gain": best["gain"],
                      "theta_after": best["theta_after_add"]})
        selected.append(int(best["index"]))
        stop_reason = "top_m_reached" if len(selected) >= top_m else "no_admissible_candidates"
    return {"selected_indices": selected, "steps": steps, "stop_reason": stop_reason,
            "alignments": alignments.tolist(), "admissible": admissible.tolist(),
            "theta_final": theta_for_indices(X_all, d, selected, ridge_lambda)}


def build_signed_trajectory(row: dict[str, Any], params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = list(row.get("candidates", []))
    X_all, d = build_local_effects(
        np.asarray(row["eta0"], dtype=float),
        np.asarray([c["eta"] for c in candidates], dtype=float),
        str(row["label"]), list(row["labels"]), params["l_type"],
        params["eps_smooth"], params["target_smoothing"],
    )
    result = signed_gated_greedy(X_all, d, top_m=params["top_m"],
        ridge_lambda=params["ridge_lambda"], stop_threshold=params["stop_threshold"],
        alignment_eps=params["alignment_eps"], gain_tolerance=params["gain_tolerance"])
    candidate_rows = []
    for idx, candidate in enumerate(candidates):
        candidate_rows.append({"candidate_id": candidate["doc_id"], "index": idx,
            "original_bm25_rank": candidate.get("rank"), "eta_j": candidate["eta"],
            "x_j": X_all[idx].tolist(), "alignment": result["alignments"][idx],
            "alignment_sign": "positive" if result["alignments"][idx] > params["alignment_eps"]
            else ("negative" if result["alignments"][idx] < -params["alignment_eps"] else "zero"),
            "admissible": result["admissible"][idx]})
    steps = []
    for step in result["steps"]:
        best = int(step["best_index"])
        steps.append({**step, "current_doc_ids": [candidates[i]["doc_id"] for i in step["current_indices"]],
            "best_candidate": candidates[best]["doc_id"], "selected_candidate": candidates[best]["doc_id"],
            "rejected_candidate_ids": [candidates[i]["doc_id"] for i,ok in enumerate(result["admissible"])
                                       if not ok and i not in step["current_indices"]]})
    trajectory = {"schema_version": SCHEMA_VERSION, "variant": "signed_gate",
        "id": row["id"], "query": row["query"], "gold": row["label"],
        "label": row["label"], "split": row["split"], "eta0": row["eta0"],
        "d_i": d.tolist(), "candidates": candidate_rows, "steps": steps,
        "selected_doc_ids": [candidates[i]["doc_id"] for i in result["selected_indices"]],
        "stop_reason": result["stop_reason"], "theta_final": result["theta_final"]}
    selected_docs = [normalize_selected_doc(candidates[i], selector_score=None, selection_step=s)
                     for s, i in enumerate(result["selected_indices"])]
    selection = make_selection_row(row, method="cbwdm_signed_gate_oracle",
        selected_docs=selected_docs,
        selection_steps=[{"step": s, "selected_doc_id": doc["doc_id"],
                          "predicted_score": None, "stop": False}
                         for s, doc in enumerate(selected_docs)],
        stop_reason=result["stop_reason"], diagnostic_only=True,
        max_docs=params["top_m"], uses_gold_at_test=True,
        selection_metadata={"variant": "signed_gate", "alignment_eps": params["alignment_eps"],
                            "uses_gold_at_test": True, "state_aware": True})
    validate_selection_row(selection)
    return trajectory, selection


class PosteriorCache:
    """Append-only, fingerprint-bound state-posterior cache."""
    def __init__(self, path: str | Path, fingerprint: str, resume: bool = False) -> None:
        self.path, self.fingerprint = Path(path), fingerprint
        self.rows: dict[str, list[float]] = {}
        if self.path.exists():
            if not resume:
                raise FileExistsError(f"Cache exists; pass --resume: {self.path}")
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("fingerprint") != fingerprint:
                        raise ValueError("Posterior cache fingerprint mismatch")
                    self.rows[str(row["key"])] = list(row["posterior"])
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> list[float] | None:
        return self.rows.get(key)

    def put(self, key: str, posterior: Sequence[float], metadata: dict[str, Any] | None = None) -> None:
        values = [float(v) for v in posterior]
        if key in self.rows:
            if not np.allclose(self.rows[key], values):
                raise ValueError(f"Conflicting cached posterior for key {key}")
            return
        row = {"fingerprint": self.fingerprint, "key": key, "posterior": values,
               "metadata": metadata or {}}
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        self.rows[key] = values


def joint_posterior_greedy(
    *,
    gold_index: int,
    eta_empty: Sequence[float],
    num_candidates: int,
    score_states: Callable[[list[list[int]]], Sequence[Sequence[float]]],
    top_m: int,
    min_gain: float = 0.0,
) -> dict[str, Any]:
    """Greedy oracle using exact state posterior probability differences."""
    selected: list[int] = []
    eta_current = np.asarray(eta_empty, dtype=float)
    steps = []
    stop_reason = "top_m_reached" if top_m == 0 else "no_candidates"
    for step_idx in range(top_m):
        remaining = [i for i in range(num_candidates) if i not in selected]
        if not remaining:
            stop_reason = "no_candidates"; break
        states = [selected + [i] for i in remaining]
        posteriors = np.asarray(score_states(states), dtype=float)
        if posteriors.shape[0] != len(remaining):
            raise ValueError("score_states returned the wrong number of posteriors")
        gains = posteriors[:, gold_index] - eta_current[gold_index]
        best_pos = max(range(len(remaining)), key=lambda i: (float(gains[i]), -remaining[i]))
        best_gain = float(gains[best_pos])
        candidates = [{"index": idx, "eta_S_plus_j": posteriors[pos].tolist(),
                       "gold_marginal_utility": float(gains[pos])}
                      for pos, idx in enumerate(remaining)]
        if best_gain <= min_gain:
            steps.append({"step": step_idx, "ordered_S": list(selected),
                          "eta_S": eta_current.tolist(), "candidates": candidates,
                          "best_index": remaining[best_pos], "best_gain": best_gain,
                          "selected_index": None, "stop_reason": "no_positive_gold_marginal"})
            stop_reason = "no_positive_gold_marginal"; break
        chosen = remaining[best_pos]
        steps.append({"step": step_idx, "ordered_S": list(selected),
                      "eta_S": eta_current.tolist(), "candidates": candidates,
                      "best_index": chosen, "best_gain": best_gain,
                      "selected_index": chosen, "stop_reason": None})
        selected.append(chosen); eta_current = posteriors[best_pos]
        stop_reason = "top_m_reached" if len(selected) >= top_m else "no_candidates"
    return {"selected_indices": selected, "steps": steps, "final_posterior": eta_current.tolist(),
            "stop_reason": stop_reason}


def evidence_coverage(retrieval_row: dict[str, Any], selected_ids: Sequence[str]) -> dict[str, Any]:
    """Use the repository's existing retrieval gold key and candidate metadata schema."""
    gold = retrieval_row.get("gold_evidence_keys")
    if not isinstance(gold, list) or not gold:
        return {"status": "UNAVAILABLE", "reason": "gold_evidence_keys missing"}
    chosen = set(map(str, selected_ids)); hit = set()
    for candidate in retrieval_row.get("candidates", []):
        meta = candidate.get("meta")
        if str(candidate.get("doc_id")) in chosen and isinstance(meta, dict):
            if meta.get("page_id") is not None and meta.get("sentence_id") is not None:
                hit.add(f"{meta['page_id']}\t{meta['sentence_id']}")
    gold_set = set(map(str, gold))
    return {"status": "AVAILABLE", "any_hit": bool(hit & gold_set),
            "complete_flattened_gold_key_union_covered": gold_set <= hit,
            "complete_gold_evidence_group_covered": {"status": "UNAVAILABLE",
                "reason": "retrieval artifact stores a flattened gold_evidence_keys union, not alternative FEVER evidence groups"},
            "sentence_recall": len(hit & gold_set) / len(gold_set),
            "gold_count": len(gold_set), "hit_count": len(hit & gold_set)}


def evaluate_selection_rows(rows: list[dict[str, Any]], scorer: Any, config: dict[str, Any],
                            project_root: str | Path, batch_size: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluator = load_evaluator_module(project_root)
    labels, verbalizers = list(config["task"]["labels"]), dict(config["task"]["verbalizers"])
    prompts, metadata = [], []
    for row in rows:
        docs = evaluator.recover_selected_docs(row)
        evidence = evaluator.build_evidence_context(docs) if docs else None
        prompts.append(build_fever_prompt(row["query"], labels, verbalizers, evidence))
        metadata.append((row, docs, evidence))
    posterior = scorer.score_prompts(prompts, batch_size=batch_size, labels=labels, verbalizers=verbalizers)
    metrics = ClassificationMetrics(labels=labels); predictions = []
    for (row, docs, evidence), probs in zip(metadata, posterior):
        values = [float(v) for v in probs]; pred = evaluator.argmax_label(labels, values)
        ranks = [float(d.get("source_rank", d.get("rank"))) for d in docs
                 if d.get("source_rank", d.get("rank")) is not None]
        metrics.update(row["label"], pred, len(docs), len(evidence or ""), values, ranks)
        predictions.append({"id": row["id"], "query": row["query"], "gold": row["label"],
                            "pred": pred, "correct": pred == row["label"], "labels": labels,
                            "probs": values, "selected_doc_ids": row["selected_doc_ids"],
                            "num_docs": len(docs), "source_ranks": ranks, "method": row["method"]})
    return predictions, metrics.compute()


def prompt_contract(config: dict[str, Any]) -> dict[str, Any]:
    return {"prompt_hash": fever_prompt_hash(config["task"]["labels"], config["task"]["verbalizers"]),
            "labels": config["task"]["labels"], "verbalizer_hash": stable_hash(config["task"]["verbalizers"])}

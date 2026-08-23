"""Experimental signed_teacher_v1 teacher and supervision contracts."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.cbwdm_score import build_local_effects, marginal_gain, theta_for_indices
from src.diagnostics.method_failure import signed_gated_greedy, summary
from src.io_utils import read_jsonl

SIGNED_TEACHER_SCHEMA = "rag_cbwdm_signed_teacher_v1.v1"


def supervision_for_candidate(*, alignment: float, gain: float, alignment_eps: float,
                              b_plus: float, b_minus: float,
                              neutral_sample_policy: str = "negative") -> dict[str, Any]:
    """Map signed alignment and unsigned Theta gain to the fixed training contract."""
    if not b_plus > b_minus > 0:
        raise ValueError("Expected b_plus > b_minus > 0")
    if neutral_sample_policy not in {"negative", "ignore"}:
        raise ValueError("neutral_sample_policy must be negative or ignore")
    if alignment <= alignment_eps:
        classification = "explicit_harmful_negative"
        effective_gain = 0.0
        bce_target, bce_included, ranking_role = 0, True, "negative"
    elif gain > b_plus:
        classification = "positive"
        effective_gain = float(gain)
        bce_target, bce_included, ranking_role = 1, True, "positive"
    elif gain < b_minus:
        classification = "negative"
        effective_gain = float(gain)
        bce_target, bce_included, ranking_role = 0, True, "negative"
    else:
        classification = "neutral"
        effective_gain = float(gain)
        bce_target = 0
        bce_included = neutral_sample_policy == "negative"
        ranking_role = "excluded"
    return {"signed_supervision_class": classification,
            "effective_training_gain": effective_gain,
            "effective_bce_target": bce_target if bce_included else None,
            "bce_included": bce_included, "ranking_role": ranking_role}


def build_signed_teacher_row(row: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Build all visited action and semantic-terminal states for one query."""
    candidates = list(row.get("candidates", [])); labels = list(row["labels"])
    if candidates:
        X, d = build_local_effects(np.asarray(row["eta0"]),
            np.asarray([candidate["eta"] for candidate in candidates]), str(row["label"]), labels,
            params["l_type"], params["eps_smooth"], params["target_smoothing"])
    else:
        X = np.empty((0, len(labels)), dtype=float)
        _, d = build_local_effects(np.asarray(row["eta0"]), X, str(row["label"]), labels,
            params["l_type"], params["eps_smooth"], params["target_smoothing"])
    alignments = X @ d
    admissible = alignments > float(params["alignment_eps"])
    selected: list[int] = []; states: list[dict[str, Any]] = []
    stop_reason = "top_m_reached" if int(params["top_m"]) == 0 else "no_admissible_candidates"
    for step in range(int(params["top_m"])):
        remaining = [idx for idx in range(len(candidates)) if idx not in selected]
        if not remaining:
            stop_reason = "no_admissible_candidates"
            break
        theta_before = theta_for_indices(X, d, selected, float(params["ridge_lambda"]))
        candidate_rows = []
        for idx in remaining:
            raw_gain, theta_after = marginal_gain(X, d, selected, idx, float(params["ridge_lambda"]))
            if raw_gain < -float(params["gain_tolerance"]):
                raise FloatingPointError(f"Negative Theta marginal gain: {raw_gain}")
            gain = 0.0 if abs(raw_gain) <= float(params["gain_tolerance"]) else float(raw_gain)
            supervision = supervision_for_candidate(alignment=float(alignments[idx]), gain=gain,
                alignment_eps=float(params["alignment_eps"]), b_plus=float(params["b_plus"]),
                b_minus=float(params["b_minus"]), neutral_sample_policy=params["neutral_sample_policy"])
            candidate = candidates[idx]
            candidate_rows.append({"candidate_id": str(candidate["doc_id"]), "index": idx,
                "original_bm25_rank": candidate.get("rank"), "eta_j": candidate["eta"],
                "x_j": X[idx].tolist(), "alignment": float(alignments[idx]),
                "alignment_sign": "positive" if alignments[idx] > params["alignment_eps"]
                    else "negative" if alignments[idx] < -params["alignment_eps"] else "zero",
                "admissible": bool(admissible[idx]), "theta_marginal_gain": gain,
                "raw_theta_marginal_gain": float(raw_gain), "theta_after_add": float(theta_after),
                **supervision})
        admissible_rows = [item for item in candidate_rows if item["admissible"]]
        if not admissible_rows:
            states.append({"step": step, "ordered_selected_ids_before_state": [candidates[i]["doc_id"] for i in selected],
                "selected_indices_before_state": list(selected), "teacher_action": "STOP", "selected_candidate_id": None,
                "stop_reason": "no_admissible_candidates", "theta_before": theta_before, "theta_after": theta_before,
                "best_gain": None, "is_terminal_state": True, "remaining_candidates": candidate_rows})
            stop_reason = "no_admissible_candidates"
            break
        best = max(admissible_rows, key=lambda item: (item["theta_marginal_gain"], -item["index"]))
        if float(best["theta_marginal_gain"]) < float(params["stop_threshold"]):
            states.append({"step": step, "ordered_selected_ids_before_state": [candidates[i]["doc_id"] for i in selected],
                "selected_indices_before_state": list(selected), "teacher_action": "STOP", "selected_candidate_id": None,
                "stop_reason": "gain_below_threshold", "theta_before": theta_before, "theta_after": theta_before,
                "best_gain": float(best["theta_marginal_gain"]), "best_candidate_id": best["candidate_id"],
                "is_terminal_state": True, "remaining_candidates": candidate_rows})
            stop_reason = "gain_below_threshold"
            break
        states.append({"step": step, "ordered_selected_ids_before_state": [candidates[i]["doc_id"] for i in selected],
            "selected_indices_before_state": list(selected), "teacher_action": "SELECT",
            "selected_candidate_id": best["candidate_id"], "stop_reason": None, "theta_before": theta_before,
            "theta_after": best["theta_after_add"], "best_gain": best["theta_marginal_gain"],
            "is_terminal_state": False, "remaining_candidates": candidate_rows})
        selected.append(int(best["index"]))
        stop_reason = "top_m_reached" if len(selected) >= int(params["top_m"]) else "no_admissible_candidates"
    result = {"schema_version": SIGNED_TEACHER_SCHEMA, "variant": "signed_teacher_v1",
        "diagnostic_only": True, "uses_gold_for_teacher": True, "deployable_teacher": False,
        "id": row["id"], "query": row["query"], "label": row["label"], "gold": row["label"],
        "split": row["split"], "labels": labels, "eta0": row["eta0"], "d_i": d.tolist(),
        "teacher_selected_doc_ids": [candidates[idx]["doc_id"] for idx in selected],
        "teacher_selected_indices": selected, "states": states, "stop_reason": stop_reason,
        "theta_final": theta_for_indices(X, d, selected, float(params["ridge_lambda"])), "parameters": dict(params)}
    oracle = signed_gated_greedy(X, d, top_m=int(params["top_m"]), ridge_lambda=float(params["ridge_lambda"]),
        stop_threshold=float(params["stop_threshold"]), alignment_eps=float(params["alignment_eps"]),
        gain_tolerance=float(params["gain_tolerance"]))
    if selected != oracle["selected_indices"] or stop_reason != oracle["stop_reason"]:
        raise AssertionError("signed_teacher_v1 action trajectory drifted from signed-gate oracle")
    return result


@dataclass
class SignedTrainingGroup:
    example_id: str
    gold: str
    step: int
    query: str
    current_doc_ids: list[str]
    selected_docs: list[dict[str, Any]]
    candidate_doc_ids: list[str]
    candidate_docs: list[dict[str, Any]]
    effective_gains: list[float]
    supervision_classes: list[str]
    is_terminal_state: bool
    teacher_action: str


def build_signed_training_groups(*, teacher_path: str | Path, posteriors_path: str | Path,
                                 retrieval_path: str | Path | None = None,
                                 max_train_groups: int | None = None) -> list[SignedTrainingGroup]:
    posterior = {str(row["id"]): row for row in read_jsonl(posteriors_path)}
    retrieval = {str(row["id"]): {str(c["doc_id"]): c for c in row.get("candidates", [])}
                 for row in read_jsonl(retrieval_path)} if retrieval_path else {}
    groups = []
    for teacher in read_jsonl(teacher_path):
        source = posterior.get(str(teacher["id"]))
        if source is None: raise KeyError(f"Teacher id missing from posterior: {teacher['id']}")
        docs = {str(c["doc_id"]): dict(c) for c in source.get("candidates", [])}
        for doc_id, item in retrieval.get(str(teacher["id"]), {}).items():
            docs.setdefault(doc_id, dict(item))
            for key in ("title", "text", "rank", "retrieval_score", "score"):
                if docs[doc_id].get(key) in (None, "") and item.get(key) not in (None, ""):
                    docs[doc_id][key] = item[key]
        for state in teacher.get("states", []):
            candidates = []
            for item in state.get("remaining_candidates", []):
                doc = docs.get(str(item["candidate_id"]))
                if doc is not None and str(doc.get("title") or doc.get("text") or "").strip():
                    candidates.append((item, doc))
            if not candidates:
                continue
            current_ids = list(map(str, state.get("ordered_selected_ids_before_state", [])))
            groups.append(SignedTrainingGroup(example_id=str(teacher["id"]), gold=str(teacher["label"]),
                step=int(state["step"]), query=str(teacher["query"]), current_doc_ids=current_ids,
                selected_docs=[docs[item] for item in current_ids if item in docs],
                candidate_doc_ids=[str(item["candidate_id"]) for item, _ in candidates],
                candidate_docs=[doc for _, doc in candidates],
                effective_gains=[float(item["effective_training_gain"]) for item, _ in candidates],
                supervision_classes=[str(item["signed_supervision_class"]) for item, _ in candidates],
                is_terminal_state=bool(state["is_terminal_state"]), teacher_action=str(state["teacher_action"])))
            if max_train_groups is not None and len(groups) >= max_train_groups:
                return groups
    if not groups: raise ValueError("No signed_teacher_v1 training groups were built")
    return groups


def teacher_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = defaultdict(lambda: {"groups": 0, "terminal": 0,
        "alignments": Counter(), "classes": Counter(), "zero_positive": 0, "zero_negative": 0,
        "rank_valid": 0, "rank_skipped": 0})
    lengths=[];stops=Counter();zero_docs=semantic=budget=0
    for row in rows:
        label=str(row["label"]);length=len(row.get("teacher_selected_doc_ids",[]));lengths.append(length)
        stops[str(row["stop_reason"])]+=1;zero_docs+=int(length==0)
        semantic+=int(row["stop_reason"] in {"no_admissible_candidates","gain_below_threshold"})
        budget+=int(row["stop_reason"]=="top_m_reached")
        for state in row.get("states",[]):
            step=int(state["step"]);items=state.get("remaining_candidates",[])
            for key in ("ALL",f"gold={label}",f"step={step}"):
                b=buckets[key];b["groups"]+=1;b["terminal"]+=int(state["is_terminal_state"])
                signs=Counter(item["alignment_sign"] for item in items);classes=Counter(item["signed_supervision_class"] for item in items)
                b["alignments"].update(signs);b["classes"].update(classes)
                positive=classes["positive"];negative=classes["negative"]+classes["explicit_harmful_negative"]
                b["zero_positive"]+=int(positive==0);b["zero_negative"]+=int(negative==0)
                valid=positive>0 and negative>0;b["rank_valid"]+=int(valid);b["rank_skipped"]+=int(not valid)
    grouped={}
    for key,b in sorted(buckets.items()):
        grouped[key]={"total_groups":b["groups"],"terminal_groups":b["terminal"],
            "candidate_alignment_counts":{name:b["alignments"][name] for name in ("positive","zero","negative")},
            "supervision_counts":{name:b["classes"][name] for name in ("positive","negative","neutral","explicit_harmful_negative")},
            "zero_positive_groups":b["zero_positive"],"zero_negative_groups":b["zero_negative"],
            "ranking_valid_groups":b["rank_valid"],"ranking_skipped_groups":b["rank_skipped"],
            "ranking_skipped_ratio":b["rank_skipped"]/b["groups"] if b["groups"] else None}
    return {"query_count":len(rows),"trajectory_length":{**summary(lengths),"distribution":dict(sorted(Counter(lengths).items()))},
        "zero_doc_teacher_count":zero_docs,"semantic_stop_count":semantic,"top_m_stop_count":budget,
        "stop_reason_counts":dict(stops),"groups":grouped,
        "old_teacher_ranking_skipped_ratio_reference":0.7748}

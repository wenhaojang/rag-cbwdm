"""Pure inference and post-hoc helpers for experimental signed_selector_v1."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable, Sequence

import numpy as np

from src.cbwdm_score import build_local_effects
from src.diagnostics.method_failure import evidence_coverage, summary


def greedy_select_without_gold(*, query: str, candidates: list[dict[str, Any]],
                               score_remaining: Callable[[str, list[dict[str, Any]], list[dict[str, Any]]], Sequence[float]],
                               top_m: int = 4, min_docs: int = 0,
                               score_threshold: float | None = 0.0) -> dict[str, Any]:
    """State-aware greedy selection whose API has no gold/direction/alignment input."""
    if top_m < 0 or min_docs < 0: raise ValueError("top_m and min_docs must be non-negative")
    selected: list[dict[str, Any]]=[];selected_ids:set[str]=set();steps=[]
    stop_reason="top_m_reached" if top_m==0 else "no_candidates"
    for step in range(min(top_m,len(candidates))):
        remaining=[candidate for candidate in candidates if str(candidate.get("doc_id")) not in selected_ids]
        if not remaining:stop_reason="no_candidates";break
        values=[float(value) for value in score_remaining(query,selected,remaining)]
        if len(values)!=len(remaining):raise ValueError("score_remaining returned wrong number of scores")
        scored=list(zip(values,remaining));best_score,best=max(scored,key=lambda pair:(pair[0],-int(pair[1].get("rank",0) or 0)))
        ranking=sorted(scored,key=lambda pair:(pair[0],-int(pair[1].get("rank",0) or 0)),reverse=True)
        score_rows=[{"doc_id":str(candidate.get("doc_id")),"score":score,"rank":candidate.get("rank")} for score,candidate in ranking]
        if score_threshold is not None and len(selected)>=min_docs and best_score<float(score_threshold):
            stop_reason="score_below_threshold";steps.append({"step":step,"selected_doc_id":None,"predicted_score":best_score,
                "remaining_count":len(remaining),"stop":True,"stop_reason":stop_reason,"all_candidate_scores":score_rows});break
        selected.append(best);selected_ids.add(str(best["doc_id"]));steps.append({"step":step,"selected_doc_id":str(best["doc_id"]),
            "predicted_score":best_score,"remaining_count":len(remaining),"stop":False,"all_candidate_scores":score_rows})
        stop_reason="top_m_reached" if len(selected)>=top_m else "no_candidates"
    return {"selected_docs":selected,"selection_steps":steps,"stop_reason":stop_reason}


def posthoc_alignment(*, selection_rows: dict[str, dict[str, Any]],
                      posterior_rows: dict[str, dict[str, Any]], cbwdm: dict[str, Any]) -> dict[str, Any]:
    buckets:dict[str,list[float]]=defaultdict(list)
    for identifier,selection in selection_rows.items():
        source=posterior_rows[identifier];candidates=list(source.get("candidates",[]));label=str(source["label"])
        if not candidates:continue
        X,d=build_local_effects(np.asarray(source["eta0"]),np.asarray([c["eta"] for c in candidates]),label,list(source["labels"]),
            cbwdm.get("L_type","euclidean_posterior_shift"),float(cbwdm.get("eps_smooth",0)),cbwdm.get("target_smoothing","paper_mixture"))
        index={str(c["doc_id"]):idx for idx,c in enumerate(candidates)}
        for doc_id in map(str,selection.get("selected_doc_ids",[])):
            if doc_id in index:
                value=float(X[index[doc_id]]@d);buckets["ALL"].append(value);buckets[f"gold={label}"].append(value)
    result={}
    for key,values in sorted(buckets.items()):
        signs=Counter("positive" if value>0 else "negative" if value<0 else "zero" for value in values);n=len(values)
        result[key]={"selected_doc_count":n,"alignment":summary(values),
            **{f"selected_{name}_alignment_count":signs[name] for name in ("positive","zero","negative")},
            **{f"selected_{name}_alignment_ratio":signs[name]/n if n else None for name in ("positive","zero","negative")}}
    return {"posthoc_only":True,"uses_gold_for_selection":False,"groups":result}


def oracle_imitation(*, selection_rows: dict[str,dict[str,Any]], oracle_rows: dict[str,dict[str,Any]],
                     posterior_rows: dict[str,dict[str,Any]]) -> dict[str,Any]:
    buckets:dict[str,dict[str,list[Any]]]=defaultdict(lambda:{"top1":[],"overlap":[],"teacher_rank":[],"zero":[],"count":[]})
    for identifier,selection in selection_rows.items():
        oracle=oracle_rows.get(identifier);source=posterior_rows.get(identifier)
        if oracle is None or source is None:continue
        label=str(source["label"]);selected=list(map(str,selection.get("selected_doc_ids",[])));teacher=list(map(str,oracle.get("selected_doc_ids",[])))
        first_scores={str(item["doc_id"]):rank+1 for rank,item in enumerate(selection.get("selection_steps",[{}])[0].get("all_candidate_scores",[]))} if selection.get("selection_steps") else {}
        for key in ("ALL",f"gold={label}"):
            b=buckets[key]
            if teacher:b["top1"].append(bool(selected and selected[0]==teacher[0]))
            union=set(selected)|set(teacher);b["overlap"].append(len(set(selected)&set(teacher))/len(union) if union else 1.0)
            if teacher and teacher[0] in first_scores:b["teacher_rank"].append(first_scores[teacher[0]])
            b["zero"].append((not selected)==(not teacher));b["count"].append(len(selected)==len(teacher))
    return {key:{"query_count":len(b["overlap"]),"step0_comparable_teacher_action_count":len(b["top1"]),"step0_top1_agreement":sum(b["top1"])/len(b["top1"]) if b["top1"] else None,
        "final_selected_set_jaccard":summary(b["overlap"]),"teacher_best_selector_rank":summary(b["teacher_rank"]),
        "zero_doc_agreement":sum(b["zero"])/len(b["zero"]) if b["zero"] else None,
        "selected_doc_count_agreement":sum(b["count"])/len(b["count"]) if b["count"] else None} for key,b in sorted(buckets.items())}


def selection_evidence_coverage(selection_rows:dict[str,dict[str,Any]],retrieval_rows:dict[str,dict[str,Any]])->dict[str,Any]:
    values=[evidence_coverage(retrieval_rows[identifier],row.get("selected_doc_ids",[])) for identifier,row in selection_rows.items() if identifier in retrieval_rows]
    available=[value for value in values if value["status"]=="AVAILABLE"]
    return {"num_examples":len(available),"any_hit_ratio":sum(v["any_hit"] for v in available)/len(available) if available else None,
        "complete_flattened_gold_key_union_covered_ratio":sum(v["complete_flattened_gold_key_union_covered"] for v in available)/len(available) if available else None,
        "complete_gold_evidence_group_coverage":"UNAVAILABLE from flattened retrieval gold_evidence_keys"}

"""Read-only server-artifact supplement for the method failure audit."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.cbwdm_score import build_local_effects
from src.diagnostics.method_failure import evidence_coverage, spearman, summary
from src.formal_provenance import atomic_write_text
from src.io_utils import read_jsonl
from src.run_manifest import atomic_write_json, git_state, sha256_file, utc_now


def unavailable(missing: Iterable[str], generation: str) -> dict[str, Any]:
    return {"status": "UNAVAILABLE", "missing": list(missing), "generation": generation}


def _json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value


def _map(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in read_jsonl(path)}


def _resolve_record_path(run: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [run / path, run / "artifacts" / "formal" / path]
    return next((p for p in candidates if p.exists()), candidates[0])


def _find_path(record: dict[str, Any], run: Path, names: tuple[str, ...]) -> Path | None:
    containers = [record]
    for key in ("paths", "artifacts", "outputs", "result"):
        if isinstance(record.get(key), dict):
            containers.append(record[key])
    for container in containers:
        for name in names:
            value = container.get(name)
            path = _resolve_record_path(run, value)
            if path is not None:
                return path
    return None


def _manifest_for(path: Path | None, explicit: Path | None = None) -> Path | None:
    if explicit and explicit.is_file():
        return explicit
    if path is None:
        return None
    options = [path.with_suffix(".manifest.json"), path.with_suffix(".grid.manifest.json"),
               path.parent / "evaluation.grid.manifest.json", path.parent / "training.grid.manifest.json"]
    return next((item for item in options if item.is_file()), None)


def _fingerprint(manifest: Path | None) -> Any:
    payload = _json(manifest) if manifest else None
    if isinstance(payload, dict):
        return payload.get("fingerprint", payload.get("source_fingerprint"))
    return None


def _candidate_records(run: Path, methods: set[str] | None = None) -> list[dict[str, Any]]:
    path = run / "artifacts/formal/calibration_candidates.json"
    payload = _json(path)
    if payload is None:
        return []
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = next((payload[key] for key in ("candidates", "records", "results")
                        if isinstance(payload.get(key), list)), [])
    else:
        records = []
    result = [r for r in records if isinstance(r, dict)]
    if methods is not None:
        result = [r for r in result if str(r.get("method", r.get("method_name", ""))).lower() in methods]
    return result


def _contract(record: dict[str, Any], run: Path) -> dict[str, Any]:
    teacher = _find_path(record, run, ("teacher_path", "teacher"))
    training = _find_path(record, run, ("training_dir", "checkpoint", "checkpoint_dir"))
    selection = _find_path(record, run, ("selection_path", "selection"))
    predictions = _find_path(record, run, ("prediction_path", "predictions_path", "predictions"))
    metrics = _find_path(record, run, ("metrics_path", "metrics"))
    explicit = lambda names: _find_path(record, run, names)
    teacher_manifest = _manifest_for(teacher, explicit(("teacher_manifest", "teacher_manifest_path")))
    training_manifest = _manifest_for(training, explicit(("checkpoint_manifest", "training_manifest", "training_manifest_path")))
    selection_manifest = _manifest_for(selection, explicit(("selection_manifest", "selection_manifest_path")))
    evaluation_manifest = _manifest_for(metrics, explicit(("evaluation_manifest", "evaluation_manifest_path")))
    if training is None and training_manifest is not None:
        training = training_manifest.parent
    training_config = None
    if training:
        base = training if training.is_dir() else training.parent
        for item in (base / "training_config.json", base.parent / "training_config.json"):
            if item.is_file(): training_config = item; break
    manifests = {"teacher": teacher_manifest, "training": training_manifest,
                 "selection": selection_manifest, "evaluation": evaluation_manifest}
    manifest_payloads = {name: (_json(path) if path else None) for name, path in manifests.items()}
    params = dict(record.get("parameters", {})) if isinstance(record.get("parameters"), dict) else {}
    train_cfg = _json(training_config) if training_config else None
    teacher_params = {}
    if isinstance(manifest_payloads["teacher"], dict):
        teacher_params = manifest_payloads["teacher"].get("contract", {}).get("parameters", {})
    selection_params = {}
    if isinstance(manifest_payloads["selection"], dict):
        selection_params = manifest_payloads["selection"].get("contract", {}).get("parameters", {})
    return {
        "candidate_id": record.get("candidate_id", record.get("candidate_fingerprint", record.get("name"))),
        "record_git_head": record.get("git_head"),
        "grid_fingerprints": {"candidate": record.get("candidate_fingerprint"),
                              "training": record.get("training_fingerprint"),
                              "selection": record.get("selection_fingerprint")},
        "record_parameters": params,
        "paths": {"teacher": str(teacher) if teacher else None, "training": str(training) if training else None,
                  "training_config": str(training_config) if training_config else None,
                  "selection": str(selection) if selection else None,
                  "predictions": str(predictions) if predictions else None, "metrics": str(metrics) if metrics else None,
                  **{f"{k}_manifest": str(v) if v else None for k, v in manifests.items()}},
        "fingerprints": {name: _fingerprint(path) for name, path in manifests.items()},
        "teacher_parameters": teacher_params,
        "training_parameters": train_cfg,
        "selection_parameters": selection_params,
        "metrics": _json(metrics) if metrics else None,
        "_resolved": {"teacher": teacher, "selection": selection, "predictions": predictions,
                       "metrics": metrics, "training_config": training_config},
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    result={}
    for key,item in value.items():
        name=f"{prefix}.{key}" if prefix else str(key)
        result.update(_flatten(item,name) if isinstance(item,dict) else {name:item})
    return result


def _contract_differences(contracts: list[dict[str, Any]]) -> dict[str, Any]:
    if len(contracts)!=2:
        return unavailable([f"exactly two RAG-CBWDM candidates (found {len(contracts)})"],
            "review all candidate contracts; pairwise comparison is emitted only when the aggregate contains exactly two")
    left,right=contracts
    compared_keys=("record_parameters","teacher_parameters","training_parameters","selection_parameters")
    lf=_flatten({k:left.get(k) for k in compared_keys});rf=_flatten({k:right.get(k) for k in compared_keys})
    differences={key:{"candidate_1":lf.get(key),"candidate_2":rf.get(key)} for key in sorted(set(lf)|set(rf)) if lf.get(key)!=rf.get(key)}
    return {"status":"AVAILABLE","candidate_1":left["candidate_id"],"candidate_2":right["candidate_id"],"differences":differences}


def _group_keys(label: str, step: int) -> list[str]:
    return ["ALL", f"gold={label}", f"step={step}", f"gold={label}|step={step}"]


def teacher_statistics(teacher_path: Path, posterior_path: Path) -> dict[str, Any]:
    posterior = _map(posterior_path)
    accum: dict[str, dict[str, Any]] = defaultdict(lambda: {"alignments": [], "gains": [], "selected": [], "ranks": []})
    lengths: list[int] = []; stops: Counter[str] = Counter(); step0_gain=[]; step0_norm=[]; step0_alignment=[]
    for row in read_jsonl(teacher_path):
        source = posterior.get(str(row["id"])); label = str(row.get("label"))
        if source is None: continue
        params = {"l_type": row.get("l_type", "euclidean_posterior_shift"),
                  "eps_smooth": float(row.get("eps_smooth", 0.0)),
                  "target_smoothing": row.get("target_smoothing", "paper_mixture")}
        candidates = source.get("candidates", [])
        X, d = build_local_effects(np.asarray(source["eta0"]), np.asarray([c["eta"] for c in candidates]),
                                   label, list(source["labels"]), **params)
        by_id = {str(c["doc_id"]): i for i,c in enumerate(candidates)}
        lengths.append(len(row.get("teacher_selected_doc_ids", []))); stops[str(row.get("stop_reason", "missing"))] += 1
        for step in row.get("steps", []):
            step_no = int(step.get("step", 0)); best = str(step.get("best_doc_id"))
            for gain_row in step.get("candidate_gains", []):
                doc_id = str(gain_row.get("doc_id")); idx = by_id.get(doc_id)
                if idx is None: continue
                alignment, gain = float(X[idx] @ d), float(gain_row.get("gain", 0.0)); selected = doc_id == best
                rank = candidates[idx].get("rank")
                for key in _group_keys(label, step_no):
                    bucket=accum[key]; bucket["alignments"].append(alignment); bucket["gains"].append(gain)
                    bucket["selected"].append((selected, alignment));
                    if selected and rank is not None: bucket["ranks"].append(float(rank))
                if step_no == 0:
                    step0_gain.append(gain); step0_norm.append(float(np.linalg.norm(X[idx]))); step0_alignment.append(alignment)
    groups = {}
    for key, bucket in sorted(accum.items()):
        signs = Counter("positive" if a > 0 else "negative" if a < 0 else "zero" for a in bucket["alignments"])
        chosen = [a for selected,a in bucket["selected"] if selected]
        chosen_signs = Counter("positive" if a > 0 else "negative" if a < 0 else "zero" for a in chosen)
        n=len(bucket["alignments"]); sn=len(chosen)
        groups[key]={"candidate_count":n,
            **{f"alignment_{s}_count":signs[s] for s in ("positive","zero","negative")},
            **{f"alignment_{s}_ratio":signs[s]/n if n else None for s in ("positive","zero","negative")},
            "gain":summary(bucket["gains"]), "selected_count":sn,
            **{f"selected_alignment_{s}":chosen_signs[s] for s in ("positive","zero","negative")},
            "selected_negative_alignment_ratio":chosen_signs["negative"]/sn if sn else None,
            "selected_original_bm25_rank":summary(bucket["ranks"])}
    return {"status":"AVAILABLE", "groups":groups,
            "trajectory_length":{**summary(lengths),"distribution":dict(sorted(Counter(lengths).items()))},
            "stop_reasons":{"counts":dict(stops),"ratios":{k:v/sum(stops.values()) for k,v in stops.items()}},
            "step0_correlations":{"gain_vs_abs_posterior_shift":spearman(step0_gain,step0_norm),
                                  "gain_vs_signed_alignment":spearman(step0_gain,step0_alignment),
                                  "candidate_count":len(step0_gain)}}


def supervision_statistics(teacher_path: Path, training: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(training, dict):
        return unavailable(["training_config.json"], "run the existing selector training stage")
    b_plus=float(training.get("b_plus", .01)); b_minus=float(training.get("b_minus", .001))
    neutral_policy=str(training.get("neutral_sample_policy", "negative"))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(teacher_path):
        label=str(row.get("label"))
        for step in row.get("steps",[]):
            step_no=int(step.get("step",0)); gains=[float(c.get("gain",0)) for c in step.get("candidate_gains",[])]
            cats=["positive" if g>b_plus else "negative" if g<b_minus else "neutral" for g in gains]
            item={"gains":gains,"cats":cats,"ranking_valid":("positive" in cats and "negative" in cats)}
            for key in _group_keys(label,step_no): groups[key].append(item)
    result={}
    for key,items in sorted(groups.items()):
        cats=[c for item in items for c in item["cats"]]; count=Counter(cats); n=len(cats); g=len(items)
        zero_pos=sum("positive" not in i["cats"] for i in items); zero_neg=sum("negative" not in i["cats"] for i in items)
        valid=sum(i["ranking_valid"] for i in items)
        result[key]={"group_count":g,"candidate_count":n,
            **{f"{c}_count":count[c] for c in ("positive","negative","neutral")},
            **{f"{c}_ratio":count[c]/n if n else None for c in ("positive","negative","neutral")},
            "zero_positive_group_count":zero_pos,"zero_positive_group_ratio":zero_pos/g if g else None,
            "zero_negative_group_count":zero_neg,"zero_negative_group_ratio":zero_neg/g if g else None,
            "ranking_valid_groups":valid,"ranking_skipped_groups":g-valid,
            "ranking_skipped_ratio":(g-valid)/g if g else None}
    return {"status":"AVAILABLE","threshold_contract":{"positive":f"gain > {b_plus}",
        "negative":f"gain < {b_minus}","neutral":f"{b_minus} <= gain <= {b_plus}",
        "neutral_sample_policy":neutral_policy,
        "bce_effect":"neutral enters BCE as negative" if neutral_policy=="negative" else neutral_policy,
        "ranking_effect":"ranking requires at least one positive and one negative"},"groups":result}


def prediction_statistics(prediction_path: Path, selection_path: Path | None,
                          posterior_path: Path, retrieval_path: Path,
                          cbwdm_config: dict[str, Any] | None = None) -> dict[str, Any]:
    predictions=_map(prediction_path); posterior=_map(posterior_path); retrieval=_map(retrieval_path)
    selections=_map(selection_path) if selection_path and selection_path.is_file() else {}
    labels=["SUPPORTS","REFUTES"]; confusion={g:{p:0 for p in labels} for g in labels}; pred_counts=Counter(); gold_counts=Counter()
    errors=[]
    for identifier,row in predictions.items():
        gold=str(row.get("gold",row.get("label"))); pred=str(row.get("pred",row.get("prediction")))
        if gold in confusion and pred in confusion[gold]: confusion[gold][pred]+=1
        pred_counts[pred]+=1; gold_counts[gold]+=1
        if gold=="SUPPORTS" and pred=="REFUTES": errors.append(identifier)
    per_class=[]
    for label in labels:
        tp=confusion[label][label]; total=sum(confusion[label].values()); predicted=sum(confusion[g][label] for g in labels)
        prec=tp/predicted if predicted else 0; rec=tp/total if total else 0
        per_class.append(2*prec*rec/(prec+rec) if prec+rec else 0)
    n=sum(gold_counts.values()); correct=sum(confusion[l][l] for l in labels)
    neg=0; selected_n=0; ranks=[]; any_hits=[]; complete=[]
    for identifier in errors:
        selection=selections.get(identifier); source=posterior.get(identifier); ret=retrieval.get(identifier)
        if not selection or not source: continue
        ids=list(map(str,selection.get("selected_doc_ids",[]))); by_id={str(c["doc_id"]):c for c in source.get("candidates",[])}
        cfg=cbwdm_config or {}
        X,d=build_local_effects(np.asarray(source["eta0"]),np.asarray([c["eta"] for c in source.get("candidates",[])]),
                                str(source["label"]),list(source["labels"]),cfg.get("L_type","euclidean_posterior_shift"),
                                float(cfg.get("eps_smooth",0.0)),cfg.get("target_smoothing","paper_mixture"))
        index={str(c["doc_id"]):i for i,c in enumerate(source.get("candidates",[]))}
        for doc_id in ids:
            if doc_id in index:
                selected_n+=1; neg+=int(float(X[index[doc_id]]@d)<0)
                if by_id[doc_id].get("rank") is not None:ranks.append(float(by_id[doc_id]["rank"]))
        if ret:
            cov=evidence_coverage(ret,ids)
            if cov["status"]=="AVAILABLE": any_hits.append(int(cov["any_hit"]));complete.append(int(cov["complete_flattened_gold_key_union_covered"]))
    return {"status":"AVAILABLE","num_examples":n,"accuracy":correct/n if n else None,
            "macro_f1":sum(per_class)/len(per_class),"confusion_matrix":confusion,
            "prediction_distribution":dict(pred_counts),"gold_distribution":dict(gold_counts),
            "supports_to_refutes":{"count":len(errors),"avg_selected_docs":selected_n/len(errors) if errors else None,
                "selected_original_bm25_rank":summary(ranks),"selected_negative_alignment_ratio":neg/selected_n if selected_n else None,
                "any_gold_evidence_hit_ratio":sum(any_hits)/len(any_hits) if any_hits else None,
                "complete_gold_evidence_ratio":sum(complete)/len(complete) if complete else None,
                "matched_selection_examples":len(any_hits)}}


def coverage_statistics(retrieval_path: Path, selection_path: Path | None = None) -> dict[str, Any]:
    selections=_map(selection_path) if selection_path and selection_path.is_file() else {}
    buckets:dict[str,list[dict[str,Any]]]=defaultdict(list)
    for row in read_jsonl(retrieval_path):
        identifier=str(row["id"]);label=str(row.get("label"))
        if selections:
            ids=list(map(str,selections.get(identifier,{}).get("selected_doc_ids",[])))
        else:
            ids=[str(c["doc_id"]) for c in row.get("candidates",[])]
        value=evidence_coverage(row,ids)
        if value["status"]=="AVAILABLE":
            buckets["ALL"].append(value);buckets[f"gold={label}"].append(value)
    return {key:{"num_examples":len(values),
        "any_gold_evidence_sentence_hit_ratio":sum(v["any_hit"] for v in values)/len(values),
        "complete_flattened_gold_key_union_covered_ratio":sum(v["complete_flattened_gold_key_union_covered"] for v in values)/len(values),
        "complete_gold_evidence_group_coverage":"UNAVAILABLE: retrieval stores flattened gold_evidence_keys, not evidence groups",
        "mean_sentence_recall":sum(v["sentence_recall"] for v in values)/len(values)}
        for key,values in sorted(buckets.items())}


def teacher_coverage_statistics(retrieval_path: Path, teacher_path: Path) -> dict[str, Any]:
    teacher={str(row["id"]):row for row in read_jsonl(teacher_path)}
    buckets:dict[str,list[dict[str,Any]]]=defaultdict(list)
    for row in read_jsonl(retrieval_path):
        source=teacher.get(str(row["id"]));label=str(row.get("label"))
        if source is None:continue
        value=evidence_coverage(row,list(map(str,source.get("teacher_selected_doc_ids",[]))))
        if value["status"]=="AVAILABLE":
            buckets["ALL"].append(value);buckets[f"gold={label}"].append(value)
    return {key:{"num_examples":len(values),
        "any_gold_evidence_sentence_hit_ratio":sum(v["any_hit"] for v in values)/len(values),
        "complete_flattened_gold_key_union_covered_ratio":sum(v["complete_flattened_gold_key_union_covered"] for v in values)/len(values),
        "complete_gold_evidence_group_coverage":"UNAVAILABLE: retrieval stores flattened gold_evidence_keys, not evidence groups",
        "mean_sentence_recall":sum(v["sentence_recall"] for v in values)/len(values)}
        for key,values in sorted(buckets.items())}


def selector_selection_statistics(selection_path: Path, posterior_path: Path,
                                  cbwdm_config: dict[str, Any]) -> dict[str, Any]:
    posterior=_map(posterior_path);buckets:dict[str,dict[str,Any]]=defaultdict(lambda:{"alignments":[],"counts":[],"stops":Counter()})
    for selection in read_jsonl(selection_path):
        source=posterior.get(str(selection["id"]));label=str(selection.get("label"))
        if source is None:continue
        candidates=list(source.get("candidates",[]));index={str(c["doc_id"]):i for i,c in enumerate(candidates)}
        X,d=build_local_effects(np.asarray(source["eta0"]),np.asarray([c["eta"] for c in candidates]),str(source["label"]),list(source["labels"]),
            cbwdm_config.get("L_type","euclidean_posterior_shift"),float(cbwdm_config.get("eps_smooth",0)),cbwdm_config.get("target_smoothing","paper_mixture"))
        alignments=[float(X[index[doc]]@d) for doc in map(str,selection.get("selected_doc_ids",[])) if doc in index]
        for key in ("ALL",f"gold={label}"):
            buckets[key]["alignments"].extend(alignments);buckets[key]["counts"].append(len(alignments));buckets[key]["stops"][str(selection.get("stop_reason","missing"))]+=1
    result={}
    for key,b in sorted(buckets.items()):
        signs=Counter("positive" if a>0 else "negative" if a<0 else "zero" for a in b["alignments"]);n=len(b["alignments"])
        result[key]={"selected_doc_count":n,"selected_docs_per_query":summary(b["counts"]),
            **{f"alignment_{s}_count":signs[s] for s in ("positive","zero","negative")},
            **{f"alignment_{s}_ratio":signs[s]/n if n else None for s in ("positive","zero","negative")},
            "stop_reason_counts":dict(b["stops"])}
    return result


def build_server_supplement(*, config_path: Path, run_dir: Path, project_root: Path) -> dict[str, Any]:
    from src.io_utils import load_yaml
    config=load_yaml(config_path)
    run=run_dir.resolve(); formal=run/"artifacts/formal"
    dataset=str(config["dataset"]);top_n=int(config["retrieval"]["top_n"])
    train_post=formal/f"{dataset}_train_core_posteriors.jsonl"; val_post=formal/f"{dataset}_validation_posteriors.jsonl"
    train_ret=formal/f"{dataset}_train_core_bm25_top{top_n}.jsonl";val_ret=formal/f"{dataset}_validation_bm25_top{top_n}.jsonl"
    records=_candidate_records(run,{"rag_cbwdm","cbwdm"}); contracts=[]
    for record in records:
        contract=_contract(record,run); resolved=contract.pop("_resolved")
        teacher=resolved["teacher"]
        if teacher and teacher.is_file() and train_post.is_file():
            contract["teacher_statistics"]=teacher_statistics(teacher,train_post)
            contract["supervision_statistics"]=supervision_statistics(teacher,contract["training_parameters"])
            contract["teacher_evidence_coverage"]=teacher_coverage_statistics(train_ret,teacher) if train_ret.is_file() else unavailable([str(train_ret)],"reuse existing train_core retrieval")
        else:
            missing=[str(p) for p in (teacher,train_post) if p is None or not p.is_file()]
            contract["teacher_statistics"]=unavailable(missing,"run existing train_core CBWDM teacher stage")
            contract["supervision_statistics"]=unavailable(missing,"run existing selector training stage")
            contract["teacher_evidence_coverage"]=unavailable(missing+[str(train_ret)],"run existing train_core teacher stage")
        pred=resolved["predictions"]
        if pred and pred.is_file() and val_post.is_file() and val_ret.is_file():
            contract["generator_statistics"]=prediction_statistics(pred,resolved["selection"],val_post,val_ret,config.get("cbwdm"))
            contract["evidence_coverage"]={"bm25_candidate_pool":coverage_statistics(val_ret),
                "selector_selection":coverage_statistics(val_ret,resolved["selection"]),
                "validation_teacher_selection":unavailable(["validation gold teacher trajectory"],"run an isolated validation oracle diagnostic")}
            selector_summary=selector_selection_statistics(resolved["selection"],val_post,config.get("cbwdm",{})) if resolved["selection"] and resolved["selection"].is_file() else unavailable([str(resolved["selection"])],"run existing selector selection")
        else:
            contract["generator_statistics"]=unavailable([str(p) for p in (pred,val_post,val_ret) if p is None or not p.is_file()],"run existing validation selection/evaluation stage")
            contract["evidence_coverage"]=unavailable([str(val_ret),str(resolved["selection"])],"run existing validation retrieval/selection stages")
            selector_summary=unavailable([str(resolved["selection"])],"run existing selector selection")
        contract["selector_diagnostics"]={"selection":selector_summary,
            "teacher_top1_and_ranking_agreement":unavailable(
                ["validation gold teacher trajectory and/or full candidate selector scores"],
                "run a diagnostic validation teacher and persist full selector scores; production artifacts are not regenerated by this supplement")}
        contracts.append(contract)
    if not records:
        candidate_section=unavailable([str(formal/"calibration_candidates.json")],"run the existing calibration-grid aggregation stage")
    else:candidate_section={"status":"AVAILABLE","count":len(contracts),"candidates":contracts,
                            "two_candidate_parameter_differences":_contract_differences(contracts)}
    inputs={}
    for path in (config_path,formal/"calibration_candidates.json",train_post,val_post,train_ret,val_ret):
        inputs[str(path)]={"exists":path.is_file(),"sha256":sha256_file(path) if path.is_file() else None}
    controls=[]
    for record in _candidate_records(run,{"infogain_fever","infogain"}):
        control=_contract(record,run);resolved=control.pop("_resolved");pred=resolved["predictions"]
        if pred and pred.is_file() and val_post.is_file() and val_ret.is_file():
            control["generator_statistics"]=prediction_statistics(pred,resolved["selection"],val_post,val_ret,config.get("cbwdm"))
            control["evidence_coverage"]={"bm25_candidate_pool":coverage_statistics(val_ret),
                "selector_selection":coverage_statistics(val_ret,resolved["selection"])}
        else:control["generator_statistics"]=unavailable([str(pred)],"run existing InfoGain validation evaluation")
        controls.append(control)
    fixed=formal/"fixed_baselines";fixed_results={}
    for name,selection_name in (("no_evidence",None),("naive_topm","naive_topm_selection.jsonl"),("cbwdm_oracle","cbwdm_oracle_selection.jsonl")):
        pred=fixed/f"{name}_predictions.jsonl";selection=fixed/selection_name if selection_name else None
        if pred.is_file() and val_post.is_file() and val_ret.is_file():
            fixed_results[name]=prediction_statistics(pred,selection,val_post,val_ret,config.get("cbwdm"))
            if selection and selection.is_file():fixed_results[name]["evidence_coverage"]=coverage_statistics(val_ret,selection)
        else:fixed_results[name]=unavailable([str(pred)],f"run existing {name} validation evaluation")
    return {"schema_version":"rag_cbwdm_method_failure_server_supplement.v1","created_at":utc_now(),
            "read_only":True,"run_dir":str(run),"git":git_state(project_root),"inputs":inputs,
            "candidate_audit":candidate_section,
            "fixed_baseline_and_teacher_oracle_control":fixed_results,
            "evidence_coverage_note":"Uses existing retrieval gold_evidence_keys and candidate meta.page_id/meta.sentence_id semantics.",
            "infogain_control":{"status":"AVAILABLE","candidates":controls} if controls else unavailable(
                ["InfoGain candidate records in calibration_candidates.json"],"run existing InfoGain calibration-grid aggregation")}


def render_markdown(payload: dict[str, Any]) -> str:
    lines=["# RAG-CBWDM Method Failure Audit — Server Supplement","",
           f"- Read only: `{payload['read_only']}`",f"- Run: `{payload['run_dir']}`",
           f"- Git head: `{payload['git'].get('commit')}`","","## Candidate / manifest audit",""]
    audit=payload["candidate_audit"]
    if audit.get("status")=="UNAVAILABLE": lines += ["UNAVAILABLE", "", f"Missing: `{audit['missing']}`", f"Generation: {audit['generation']}"]
    else:
        lines += ["### Two-candidate parameter differences","","```json",
                  json.dumps(audit["two_candidate_parameter_differences"],ensure_ascii=False,indent=2,sort_keys=True),"```",""]
        for c in audit["candidates"]:
            lines += [f"### Candidate `{c['candidate_id']}`","","```json",
                      json.dumps({k:v for k,v in c.items() if k not in {"teacher_statistics","supervision_statistics","generator_statistics"}},ensure_ascii=False,indent=2,sort_keys=True),"```","",
                      "#### Teacher statistics","","```json",json.dumps(c["teacher_statistics"],ensure_ascii=False,indent=2,sort_keys=True),"```","",
                      "#### Supervision statistics","","```json",json.dumps(c["supervision_statistics"],ensure_ascii=False,indent=2,sort_keys=True),"```","",
                      "#### Selector / teacher agreement","","```json",json.dumps(c["selector_diagnostics"],ensure_ascii=False,indent=2,sort_keys=True),"```","",
                      "#### Generator / REFUTES bias","","```json",json.dumps(c["generator_statistics"],ensure_ascii=False,indent=2,sort_keys=True),"```",""]
            lines += ["#### Retrieval / evidence coverage","","```json",json.dumps(c["evidence_coverage"],ensure_ascii=False,indent=2,sort_keys=True),"```",""]
            lines += ["#### Train-core teacher evidence coverage","","```json",json.dumps(c["teacher_evidence_coverage"],ensure_ascii=False,indent=2,sort_keys=True),"```",""]
    lines += ["## InfoGain control","","```json",json.dumps(payload["infogain_control"],ensure_ascii=False,indent=2,sort_keys=True),"```",""]
    lines += ["## Fixed baselines and production CBWDM teacher oracle","","```json",json.dumps(payload["fixed_baseline_and_teacher_oracle_control"],ensure_ascii=False,indent=2,sort_keys=True),"```",""]
    return "\n".join(lines)


def publish(*, config_path: Path, run_dir: Path, output_dir: Path, project_root: Path) -> dict[str, Any]:
    payload=build_server_supplement(config_path=config_path,run_dir=run_dir,project_root=project_root)
    output_dir.mkdir(parents=True,exist_ok=True)
    atomic_write_json(output_dir/"RAG_CBWDM_METHOD_FAILURE_AUDIT_SERVER_SUPPLEMENT.json",payload)
    atomic_write_text(output_dir/"RAG_CBWDM_METHOD_FAILURE_AUDIT_SERVER_SUPPLEMENT.md",render_markdown(payload))
    return payload

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))

from src.cbwdm_score import build_local_effects, marginal_gain
from src.diagnostics.method_failure import (PosteriorCache, atomic_write_jsonl, evidence_coverage,
    joint_posterior_greedy, load_evaluator_module, prompt_contract, require_diagnostic_output,
    spearman)
from src.formal_provenance import atomic_write_text
from src.io_utils import load_yaml, read_jsonl
from src.label_logits import LabelLogitScorer
from src.metrics import ClassificationMetrics
from src.prompts import build_fever_prompt
from src.run_manifest import atomic_write_json, git_state, sha256_file, stable_hash, utc_now
from src.selection_schema import make_selection_row, normalize_selected_doc, validate_selection_row


def _append(path:Path,row:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8",newline="\n") as handle:
        handle.write(json.dumps(row,ensure_ascii=False)+"\n");handle.flush();os.fsync(handle.fileno())


def main()->None:
    p=argparse.ArgumentParser(description="Diagnostic true joint-posterior gold oracle.")
    p.add_argument("--config",required=True);p.add_argument("--run-dir",required=True);p.add_argument("--output-dir",default=None)
    p.add_argument("--validation-limit",type=int,default=None);p.add_argument("--candidate-limit",type=int,default=None)
    p.add_argument("--top-m",type=int,default=None);p.add_argument("--generator-model",default=None);p.add_argument("--device",default="auto")
    p.add_argument("--batch-size",type=int,default=None);p.add_argument("--resume",action="store_true");args=p.parse_args()
    if args.validation_limit is not None and args.validation_limit < 1:raise ValueError("--validation-limit must be >= 1")
    if args.candidate_limit is not None and args.candidate_limit < 1:raise ValueError("--candidate-limit must be >= 1")
    if args.top_m is not None and args.top_m < 0:raise ValueError("--top-m must be non-negative")
    config_path=Path(args.config).resolve();config=load_yaml(config_path);run=Path(args.run_dir).resolve()
    output=Path(args.output_dir).resolve() if args.output_dir else run/"artifacts/diagnostics/method_failure_audit/joint_posterior_oracle"
    output=require_diagnostic_output(run,output);formal=run/"artifacts/formal"
    posterior=formal/f"{config['dataset']}_validation_posteriors.jsonl";retrieval=formal/f"{config['dataset']}_validation_bm25_top{config['retrieval']['top_n']}.jsonl"
    model=args.generator_model or config["generator"]["model_name"];top_m=args.top_m if args.top_m is not None else int(config["cbwdm"]["top_m"])
    contract={"variant":"joint_posterior_oracle","config_sha256":sha256_file(config_path),"posterior_sha256":sha256_file(posterior),
        "retrieval_sha256":sha256_file(retrieval),"validation_limit":args.validation_limit,"candidate_limit":args.candidate_limit,"top_m":top_m,
        "selection_utility":"probability_difference","acceptance":"strictly_positive","generator_model":model,
        "generator_revision":config["generator"].get("revision"),"device":args.device,**prompt_contract(config)}
    fingerprint=stable_hash(contract);manifest_path=output/"manifest.json";cache_path=output/"state_posterior_cache.jsonl"
    trajectories=output/"trajectories.jsonl";partial=output/"trajectories.jsonl.partial"
    final_outputs=[trajectories,output/"selection.jsonl",output/"predictions.jsonl",output/"metrics.json",output/"comparison.json",output/"report.md"]
    existing=json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    if existing:
        if not args.resume:raise FileExistsError("Joint diagnostic exists; pass --resume or use a new output directory")
        if existing.get("fingerprint")!=fingerprint:raise ValueError("Joint diagnostic manifest fingerprint mismatch")
        if existing.get("status")=="completed":
            if all(path.is_file() and existing["outputs"][path.name]["sha256"]==sha256_file(path) for path in final_outputs+[cache_path]):
                print(f"[joint_posterior] reused=true output={output}");return
            raise ValueError("Completed joint diagnostic checksum mismatch")
    elif any(path.exists() for path in final_outputs+[partial,cache_path]):
        raise FileExistsError("Unowned joint diagnostic files exist; use a new output directory")
    output.mkdir(parents=True,exist_ok=True)
    atomic_write_json(manifest_path,{"schema_version":"rag_cbwdm_method_failure_manifest.v1","status":"running",
        "completed":False,"fingerprint":fingerprint,"contract":contract,"git":git_state(PROJECT_ROOT),"started_at":utc_now()})
    cache=PosteriorCache(cache_path,fingerprint,resume=args.resume);evaluator=load_evaluator_module(PROJECT_ROOT)
    labels=list(config["task"]["labels"]);verbalizers=dict(config["task"]["verbalizers"]);batch=args.batch_size or int(config["generator"].get("posterior_batch_size",4))
    scorer=LabelLogitScorer(model_name=model,dtype=config["generator"].get("dtype","auto"),device_map=args.device,
        trust_remote_code=bool(config["generator"].get("trust_remote_code",False)),revision=config["generator"].get("revision"),
        tokenizer_revision=config["generator"].get("tokenizer_revision"),max_length=config["generator"].get("max_context_tokens"))
    completed={str(row["id"]) for row in read_jsonl(partial)} if partial.is_file() else set()
    for row in read_jsonl(posterior,limit=args.validation_limit):
        identifier=str(row["id"])
        if identifier in completed:continue
        candidates=list(row.get("candidates",[]))
        if args.candidate_limit is not None:candidates=candidates[:args.candidate_limit]
        gold_index=labels.index(str(row["label"]));eta_empty=list(map(float,row["eta0"]))
        # Stage 03 and production evaluation share build_fever_prompt with evidence=None;
        # the prompt/verbalizer hash is part of this diagnostic fingerprint.
        def score_states(states:list[list[int]])->list[list[float]]:
            results:[list[float]|None]=[None]*len(states);missing=[];prompts=[];keys=[]
            for pos,state in enumerate(states):
                doc_ids=[str(candidates[i]["doc_id"]) for i in state]
                context=evaluator.build_evidence_context([candidates[i] for i in state])
                prompt=build_fever_prompt(row["query"],labels,verbalizers,context)
                key=stable_hash({"fingerprint":fingerprint,"query_id":identifier,"ordered_doc_ids":doc_ids,
                                 "prompt_sha256":stable_hash(prompt)})
                value=cache.get(key)
                if value is None:missing.append(pos);prompts.append(prompt);keys.append((key,doc_ids))
                else:results[pos]=value
            if prompts:
                scored=scorer.score_prompts(prompts,batch_size=batch,labels=labels,verbalizers=verbalizers)
                for pos,values,(key,doc_ids) in zip(missing,scored,keys):
                    result=[float(v) for v in values];cache.put(key,result,{"query_id":identifier,"ordered_doc_ids":doc_ids});results[pos]=result
            return [value for value in results if value is not None]
        result=joint_posterior_greedy(gold_index=gold_index,eta_empty=eta_empty,num_candidates=len(candidates),
                                      score_states=score_states,top_m=top_m,min_gain=0.0)
        X,d=build_local_effects(np.asarray(row["eta0"]),np.asarray([c["eta"] for c in candidates]),str(row["label"]),labels,
            config["cbwdm"].get("L_type","euclidean_posterior_shift"),float(config["cbwdm"].get("eps_smooth",0)),config["cbwdm"].get("target_smoothing","paper_mixture"))
        selected_before=[]
        for step in result["steps"]:
            surrogate=[]
            joint_by={int(item["index"]):float(item["gold_marginal_utility"]) for item in step["candidates"]}
            for idx in joint_by:
                gain,_=marginal_gain(X,d,selected_before,idx,float(config["cbwdm"]["ridge_lambda"]));surrogate.append({"index":idx,"surrogate_gain":gain})
            step["surrogate_comparison"]=surrogate
            step["candidate_doc_ids"]={str(item["index"]):str(candidates[int(item["index"])]["doc_id"]) for item in step["candidates"]}
            if step["selected_index"] is not None:selected_before.append(int(step["selected_index"]))
        chosen=[candidates[i] for i in result["selected_indices"]];pred=evaluator.argmax_label(labels,result["final_posterior"])
        trajectory={"schema_version":"rag_cbwdm_joint_posterior_oracle.v1","id":identifier,"query":row["query"],"gold":row["label"],
            "eta_empty":eta_empty,"eta_empty_source":"stage03_same_query_only_prompt_contract","steps":result["steps"],
            "final_selected_indices":result["selected_indices"],"final_selected_doc_ids":[c["doc_id"] for c in chosen],
            "final_posterior":result["final_posterior"],"final_prediction":pred,"stop_reason":result["stop_reason"]}
        _append(partial,trajectory);completed.add(identifier)
    os.replace(partial,trajectories)
    trajectory_rows=list(read_jsonl(trajectories));posterior_map={str(r["id"]):r for r in read_jsonl(posterior,limit=args.validation_limit)}
    retrieval_map={str(r["id"]):r for r in read_jsonl(retrieval,limit=args.validation_limit)};selections=[];predictions=[];metrics=ClassificationMetrics(labels=labels)
    comparisons=defaultdict(lambda:{"surrogate":[],"joint":[],"top1":[],"sign":[]});coverage=[]
    for trajectory in trajectory_rows:
        source=posterior_map[str(trajectory["id"])];candidates=list(source["candidates"])
        if args.candidate_limit is not None:candidates=candidates[:args.candidate_limit]
        chosen=[candidates[i] for i in trajectory["final_selected_indices"]]
        docs=[normalize_selected_doc(c,selector_score=None,selection_step=i) for i,c in enumerate(chosen)]
        selection=make_selection_row(source,method="joint_posterior_gold_oracle",selected_docs=docs,
            selection_steps=[{"step":i,"selected_doc_id":d["doc_id"],"predicted_score":None,"stop":False} for i,d in enumerate(docs)],
            stop_reason=trajectory["stop_reason"],diagnostic_only=True,max_docs=top_m,uses_gold_at_test=True,
            selection_metadata={"variant":"joint_posterior_oracle","utility":"gold_probability_difference","uses_gold_at_test":True})
        validate_selection_row(selection);selections.append(selection)
        probs=trajectory["final_posterior"];pred=trajectory["final_prediction"]
        ranks=[float(d["rank"]) for d in docs if d.get("rank") is not None]
        evidence_text=evaluator.build_evidence_context(docs) if docs else ""
        metrics.update(source["label"],pred,len(docs),len(evidence_text),probs,ranks)
        predictions.append({"id":source["id"],"query":source["query"],"gold":source["label"],"pred":pred,"correct":pred==source["label"],
            "labels":labels,"probs":probs,"selected_doc_ids":[d["doc_id"] for d in docs],"num_docs":len(docs),"source_ranks":ranks,"method":"joint_posterior_gold_oracle"})
        coverage.append(evidence_coverage(retrieval_map[str(source["id"])],[d["doc_id"] for d in docs]))
        for step in trajectory["steps"]:
            s={int(x["index"]):float(x["surrogate_gain"]) for x in step["surrogate_comparison"]};j={int(x["index"]):float(x["gold_marginal_utility"]) for x in step["candidates"]};keys=sorted(set(s)&set(j))
            group_keys=["ALL",f"gold={source['label']}",f"step={step['step']}",f"gold={source['label']}|step={step['step']}"]
            for group in group_keys:
                comparisons[group]["surrogate"].extend(s[k] for k in keys);comparisons[group]["joint"].extend(j[k] for k in keys)
                if keys:
                    comparisons[group]["top1"].append(max(keys,key=lambda k:(s[k],-k))==max(keys,key=lambda k:(j[k],-k)))
                    comparisons[group]["sign"].extend((s[k]>0)==(j[k]>0) for k in keys)
    comparison_out={k:{"matched_candidate_states":len(v["joint"]),"spearman":spearman(v["surrogate"],v["joint"]),
        "top1_agreement":sum(v["top1"])/len(v["top1"]) if v["top1"] else None,"gain_sign_agreement":sum(v["sign"])/len(v["sign"]) if v["sign"] else None}
        for k,v in sorted(comparisons.items())}
    metric_out=metrics.compute();available=[c for c in coverage if c["status"]=="AVAILABLE"]
    metric_out.update({"method":"joint_posterior_gold_oracle","diagnostic_only":True,"deployable":False,
        "gold_evidence_coverage":{"num_examples":len(available),"any_hit_ratio":sum(c["any_hit"] for c in available)/len(available) if available else None,
            "complete_flattened_gold_key_union_covered_ratio":sum(c["complete_flattened_gold_key_union_covered"] for c in available)/len(available) if available else None,
            "complete_gold_evidence_group_coverage":"UNAVAILABLE from flattened retrieval gold_evidence_keys"}})
    atomic_write_jsonl(final_outputs[1],selections);atomic_write_jsonl(final_outputs[2],predictions);atomic_write_json(final_outputs[3],metric_out);atomic_write_json(final_outputs[4],comparison_out)
    atomic_write_text(final_outputs[5],"# Joint-Posterior Gold Oracle\n\n"
        "Diagnostic-only validation oracle. Utility is the strict-positive probability difference `eta_(S+j)[gold] - eta_S[gold]`.\n\n"
        f"## Metrics\n\n```json\n{json.dumps(metric_out,ensure_ascii=False,indent=2,sort_keys=True)}\n```\n\n"
        f"## Original surrogate comparison\n\n```json\n{json.dumps(comparison_out,ensure_ascii=False,indent=2,sort_keys=True)}\n```\n")
    atomic_write_json(manifest_path,{"schema_version":"rag_cbwdm_method_failure_manifest.v1","status":"completed","completed":True,
        "fingerprint":fingerprint,"contract":contract,"num_rows":len(trajectory_rows),"git":git_state(PROJECT_ROOT),
        "outputs":{path.name:{"path":str(path),"sha256":sha256_file(path)} for path in final_outputs+[cache_path]},"completed_at":utc_now()})
    print(f"[joint_posterior] rows={len(trajectory_rows)} accuracy={metric_out['accuracy']:.6f} output={output}")


if __name__=="__main__":main()

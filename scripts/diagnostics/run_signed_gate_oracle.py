from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.method_failure import (atomic_write_jsonl, build_signed_trajectory,
    evaluate_selection_rows, evidence_coverage, prompt_contract, require_diagnostic_output)
from src.formal_provenance import atomic_write_text
from src.io_utils import load_yaml, read_jsonl
from src.label_logits import LabelLogitScorer
from src.run_manifest import atomic_write_json, git_state, sha256_file, stable_hash, utc_now


def main() -> None:
    p=argparse.ArgumentParser(description="Diagnostic-only signed-gated CBWDM gold oracle.")
    p.add_argument("--config",required=True);p.add_argument("--run-dir",required=True)
    p.add_argument("--output-dir",default=None);p.add_argument("--validation-limit",type=int,default=None)
    p.add_argument("--top-m",type=int,default=None);p.add_argument("--teacher-stop-threshold",type=float,default=None)
    p.add_argument("--alignment-eps",type=float,default=0.0);p.add_argument("--generator-model",default=None)
    p.add_argument("--device",default="auto");p.add_argument("--batch-size",type=int,default=None)
    p.add_argument("--resume",action="store_true");args=p.parse_args()
    if args.validation_limit is not None and args.validation_limit < 1:raise ValueError("--validation-limit must be >= 1")
    if args.top_m is not None and args.top_m < 0:raise ValueError("--top-m must be non-negative")
    if args.alignment_eps < 0:raise ValueError("--alignment-eps must be non-negative")
    config_path=Path(args.config).resolve();config=load_yaml(config_path);run=Path(args.run_dir).resolve()
    output=Path(args.output_dir).resolve() if args.output_dir else run/"artifacts/diagnostics/method_failure_audit/signed_gate"
    output=require_diagnostic_output(run,output);formal=run/"artifacts/formal"
    posterior=formal/f"{config['dataset']}_validation_posteriors.jsonl"
    retrieval=formal/f"{config['dataset']}_validation_bm25_top{config['retrieval']['top_n']}.jsonl"
    params={"top_m":args.top_m if args.top_m is not None else int(config["cbwdm"]["top_m"]),
        "ridge_lambda":float(config["cbwdm"]["ridge_lambda"]),
        "stop_threshold":args.teacher_stop_threshold if args.teacher_stop_threshold is not None else float(config["cbwdm"]["stop_threshold"]),
        "alignment_eps":args.alignment_eps,"gain_tolerance":float(config["cbwdm"].get("gain_tolerance",1e-10)),
        "eps_smooth":float(config["cbwdm"].get("eps_smooth",0)),"l_type":config["cbwdm"].get("L_type","euclidean_posterior_shift"),
        "target_smoothing":config["cbwdm"].get("target_smoothing","paper_mixture")}
    model=args.generator_model or config["generator"]["model_name"]
    contract={"variant":"signed_gate","config_sha256":sha256_file(config_path),"posterior_sha256":sha256_file(posterior),
        "retrieval_sha256":sha256_file(retrieval),
        "parameters":params,"validation_limit":args.validation_limit,"generator_model":model,
        "generator_revision":config["generator"].get("revision"),"device":args.device,**prompt_contract(config)}
    fingerprint=stable_hash(contract);manifest_path=output/"manifest.json"
    outputs=[output/"trajectories.jsonl",output/"selection.jsonl",output/"predictions.jsonl",output/"metrics.json",output/"report.md"]
    if args.resume and manifest_path.is_file():
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status")=="completed" and manifest.get("fingerprint")==fingerprint and all(path.is_file() for path in outputs):
            if all(manifest.get("outputs",{}).get(path.name,{}).get("sha256")==sha256_file(path) for path in outputs):
                print(f"[signed_gate] reused=true output={output}");return
        raise ValueError("Cannot resume signed gate: fingerprint/checksum mismatch")
    if manifest_path.exists() or any(path.exists() for path in outputs):
        raise FileExistsError("Signed-gate output exists; use --resume with the identical contract or a new output directory")
    trajectories=[];selections=[]
    for row in read_jsonl(posterior,limit=args.validation_limit):
        trajectory,selection=build_signed_trajectory(row,params);trajectories.append(trajectory);selections.append(selection)
    output.mkdir(parents=True,exist_ok=True);atomic_write_jsonl(outputs[0],trajectories);atomic_write_jsonl(outputs[1],selections)
    scorer=LabelLogitScorer(model_name=model,dtype=config["generator"].get("dtype","auto"),device_map=args.device,
        trust_remote_code=bool(config["generator"].get("trust_remote_code",False)),revision=config["generator"].get("revision"),
        tokenizer_revision=config["generator"].get("tokenizer_revision"),max_length=config["generator"].get("max_context_tokens"))
    predictions,metrics=evaluate_selection_rows(selections,scorer,config,PROJECT_ROOT,
        args.batch_size or int(config["generator"].get("posterior_batch_size",4)))
    selected_alignments=[c["alignment"] for row in trajectories for c in row["candidates"] if c["candidate_id"] in row["selected_doc_ids"]]
    retrieval_map={str(row["id"]):row for row in read_jsonl(retrieval,limit=args.validation_limit)}
    coverage=[evidence_coverage(retrieval_map[str(row["id"])],row["selected_doc_ids"]) for row in selections]
    available=[item for item in coverage if item["status"]=="AVAILABLE"]
    metrics.update({"method":"cbwdm_signed_gate_oracle","diagnostic_only":True,"deployable":False,
        "selected_negative_alignment_ratio":sum(v<0 for v in selected_alignments)/len(selected_alignments) if selected_alignments else 0.0,
        "gold_evidence_coverage":{"num_examples":len(available),
            "any_hit_ratio":sum(v["any_hit"] for v in available)/len(available) if available else None,
            "complete_flattened_gold_key_union_covered_ratio":sum(v["complete_flattened_gold_key_union_covered"] for v in available)/len(available) if available else None,
            "complete_gold_evidence_group_coverage":"UNAVAILABLE from flattened retrieval gold_evidence_keys"}})
    atomic_write_jsonl(outputs[2],predictions);atomic_write_json(outputs[3],metrics)
    atomic_write_text(outputs[4],"# Signed-Gated CBWDM Oracle\n\n"
        "Diagnostic-only gold oracle. Admissibility is `x_ij^T d_i > alignment_eps`; accepted candidates retain the production Theta marginal rule.\n\n"
        f"```json\n{json.dumps(metrics,ensure_ascii=False,indent=2,sort_keys=True)}\n```\n")
    atomic_write_json(manifest_path,{"schema_version":"rag_cbwdm_method_failure_manifest.v1","status":"completed",
        "completed":True,"fingerprint":fingerprint,"contract":contract,"num_rows":len(selections),"git":git_state(PROJECT_ROOT),
        "outputs":{path.name:{"path":str(path),"sha256":sha256_file(path)} for path in outputs},"completed_at":utc_now()})
    print(f"[signed_gate] rows={len(selections)} accuracy={metrics['accuracy']:.6f} output={output}")


if __name__=="__main__":main()

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))

from src.diagnostics.signed_selector_v1 import selection_evidence_coverage
from src.io_utils import read_jsonl
from src.preformal.registry import PREFORMAL_METHODS
from src.preformal.statistics import indexed_rows, mean_sd, paired_comparison
from src.run_manifest import atomic_write_json, git_state, sha256_file, stable_hash, utc_now


def assignments(values:list[str])->dict[str,Path]:
    result={}
    for value in values:
        if "=" not in value:raise ValueError(f"Expected METHOD:SEED=PATH, got {value!r}")
        key,path=value.split("=",1)
        if key in result:raise ValueError(f"Duplicate assignment {key}")
        result[key]=Path(path).resolve()
    return result


def key_parts(key:str)->tuple[str,int]:
    method,separator,seed=key.partition(":")
    if method not in PREFORMAL_METHODS:raise ValueError(f"Unknown preformal method {method!r}")
    return method,int(seed) if separator else 13


def write_csv(path:Path,rows:list[dict[str,Any]])->None:
    fields=sorted({field for row in rows for field in row});partial=path.with_name(path.name+".partial")
    with partial.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        writer.writerows({field:json.dumps(value,sort_keys=True) if isinstance(value,(dict,list)) else value for field,value in row.items()} for row in rows)
    partial.replace(path)


def main()->None:
    p=argparse.ArgumentParser(description="Build formal-grade signed-v1 preformal results and paired statistics")
    p.add_argument("--fairness-audit",required=True);p.add_argument("--retrieval",required=True)
    p.add_argument("--evaluation-manifest",action="append",default=[]);p.add_argument("--selection",action="append",default=[])
    p.add_argument("--output-dir",required=True);p.add_argument("--bootstrap-seed",type=int,default=130421);p.add_argument("--bootstrap-samples",type=int,default=10000);a=p.parse_args()
    fairness=json.loads(Path(a.fairness_audit).read_text(encoding="utf-8"))
    if fairness.get("status")!="comparable":raise ValueError("Refusing summary: preformal fairness audit is not comparable")
    evaluations=assignments(a.evaluation_manifest);selections=assignments(a.selection);retrieval_path=Path(a.retrieval).resolve()
    retrieval=indexed_rows(read_jsonl(retrieval_path),"retrieval");rows=[];predictions={}
    for key,manifest_path in sorted(evaluations.items(),key=lambda item:(key_parts(item[0])[0],key_parts(item[0])[1])):
        method,seed=key_parts(key);manifest=json.loads(manifest_path.read_text(encoding="utf-8"));metrics_path=Path(manifest["metrics_path"])
        metrics=json.loads(metrics_path.read_text(encoding="utf-8"));prediction_path=Path(manifest["predictions_path"]);predictions[key]=list(read_jsonl(prediction_path))
        selection_path=selections.get(key)
        coverage={"any_hit_ratio":None,"complete_flattened_gold_key_union_covered_ratio":None}
        if selection_path is not None and method != "no_evidence":
            selected=indexed_rows(read_jsonl(selection_path),f"{key} selection");coverage=selection_evidence_coverage(selected,retrieval)
        per_class=metrics.get("per_class",{})
        summary_row={"method":method,"seed":seed,"accuracy":metrics.get("accuracy"),"macro_f1":metrics.get("macro_f1"),
            "SUPPORTS_recall":per_class.get("SUPPORTS",{}).get("recall"),"SUPPORTS_f1":per_class.get("SUPPORTS",{}).get("f1"),
            "REFUTES_recall":per_class.get("REFUTES",{}).get("recall"),"REFUTES_f1":per_class.get("REFUTES",{}).get("f1"),
            "avg_num_docs":metrics.get("avg_num_docs"),"avg_evidence_chars":metrics.get("avg_evidence_chars"),
            "avg_original_retrieval_rank":metrics.get("avg_original_retrieval_rank",metrics.get("avg_original_bm25_rank")),"gold_evidence_any_hit":coverage.get("any_hit_ratio"),
            "complete_flattened_union_coverage":coverage.get("complete_flattened_gold_key_union_covered_ratio"),
            "num_examples":metrics.get("num_examples"),"evaluation_manifest":str(manifest_path)}
        if "avg_original_bm25_rank" in metrics:summary_row["avg_original_bm25_rank"]=metrics["avg_original_bm25_rank"]
        rows.append(summary_row)
    aggregates={}
    for method in ("infogain_fever","rag_cbwdm","rag_cbwdm_signed_v1"):
        method_rows=[row for row in rows if row["method"]==method]
        if method_rows:
            aggregates[method]={"num_seeds":len(method_rows),"accuracy":mean_sd([float(row["accuracy"]) for row in method_rows]),
                "macro_f1":mean_sd([float(row["macro_f1"]) for row in method_rows]),
                "avg_num_docs":mean_sd([float(row["avg_num_docs"]) for row in method_rows])}
    comparisons={}
    for signed_key,signed_predictions in sorted(predictions.items()):
        signed_method,signed_seed=key_parts(signed_key)
        if signed_method!="rag_cbwdm_signed_v1":continue
        for baseline in ("infogain_fever","rag_cbwdm","bge","naive_topm"):
            baseline_key=f"{baseline}:{signed_seed}" if f"{baseline}:{signed_seed}" in predictions else f"{baseline}:13"
            if baseline_key not in predictions:continue
            name=f"signed_seed{signed_seed}_vs_{baseline}_seed{key_parts(baseline_key)[1]}"
            comparisons[name]={"signed_key":signed_key,"baseline_key":baseline_key,
                **paired_comparison(signed_predictions,predictions[baseline_key],bootstrap_seed=a.bootstrap_seed,samples=a.bootstrap_samples)}
    output=Path(a.output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    payload={"schema_version":"rag_cbwdm_preformal_results.v1","status":"completed","benchmark_role":"formal-grade internal preformal benchmark",
        "not_final_held_out_result":True,"rows":rows,"learned_method_aggregates":aggregates,"fairness_audit":str(Path(a.fairness_audit).resolve()),
        "go_no_go_rules":"report_only_no_parameter_updates","created_at":utc_now()}
    atomic_write_json(output/"PREFORMAL_SIGNED_V1_RESULTS.json",payload);write_csv(output/"PREFORMAL_SIGNED_V1_RESULTS.csv",rows)
    lines=["# Preformal signed-v1 results","","This is a formal-grade internal preformal benchmark, not a final paper held-out result.","",
        "| Method | Seed | Accuracy | Macro-F1 | SUPPORTS R/F1 | REFUTES R/F1 | Avg docs | Avg chars | Avg retrieval rank | Any-hit | Flat union |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    show=lambda value:"NA" if value is None else f"{float(value):.6f}"
    for row in rows:lines.append(f"| {row['method']} | {row['seed']} | {show(row['accuracy'])} | {show(row['macro_f1'])} | {show(row['SUPPORTS_recall'])}/{show(row['SUPPORTS_f1'])} | {show(row['REFUTES_recall'])}/{show(row['REFUTES_f1'])} | {show(row['avg_num_docs'])} | {show(row['avg_evidence_chars'])} | {show(row['avg_original_retrieval_rank'])} | {show(row['gold_evidence_any_hit'])} | {show(row['complete_flattened_union_coverage'])} |")
    lines.extend(("","## Learned-method stability","",f"```json\n{json.dumps(aggregates,indent=2,sort_keys=True)}\n```","",
        "## Interpretation boundary","","The current pilot validation 500 was used for method development; preformal_eval was clean at run start; official-dev held_out_test remains untouched. If the algorithm changes after this benchmark, preformal_eval becomes development data and the final conclusion must wait for the untouched held_out_test.",""))
    (output/"PREFORMAL_SIGNED_V1_RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    paired={"schema_version":"rag_cbwdm_preformal_paired.v1","status":"completed","bootstrap_seed":a.bootstrap_seed,"comparisons":comparisons}
    atomic_write_json(output/"PREFORMAL_PAIRED_COMPARISONS.json",paired)
    paired_lines=["# Preformal paired comparisons","","Differences are signed-v1 minus baseline. McNemar uses the exact two-sided binomial equivalent.",""]
    for name,value in comparisons.items():paired_lines.extend((f"## {name}","",f"Correctness table: `{json.dumps(value['correctness_table'],sort_keys=True)}`",f"McNemar exact p: {value['mcnemar_exact_two_sided_p']:.6g}",f"Accuracy difference 95% CI: `{value['bootstrap']['accuracy_difference_signed_minus_baseline']['ci95']}`",f"Macro-F1 difference 95% CI: `{value['bootstrap']['macro_f1_difference_signed_minus_baseline']['ci95']}`",""))
    (output/"PREFORMAL_PAIRED_COMPARISONS.md").write_text("\n".join(paired_lines),encoding="utf-8")
    atomic_write_json(output/"summary.manifest.json",{"schema_version":"rag_cbwdm_preformal_summary_manifest.v1","status":"completed",
        "config_sha256":None,"split_sha256":fairness.get("split_sha256"),"retrieval_sha256":fairness.get("retrieval_sha256"),
        "posterior_sha256":fairness.get("posterior_sha256"),"generator_sha256":fairness.get("generator_sha256"),"prompt_hash":fairness.get("prompt_hash"),
        "verbalizer_hash":fairness.get("verbalizer_hash"),"fairness_sha256":sha256_file(a.fairness_audit),
        "results_sha256":sha256_file(output/"PREFORMAL_SIGNED_V1_RESULTS.json"),"paired_sha256":sha256_file(output/"PREFORMAL_PAIRED_COMPARISONS.json"),
        "fingerprint":stable_hash({"rows":rows,"comparisons":comparisons}),"git":git_state(PROJECT_ROOT),"created_at":utc_now()})
    print(f"[preformal_summary] rows={len(rows)} comparisons={len(comparisons)} output={output}")
if __name__=="__main__":main()

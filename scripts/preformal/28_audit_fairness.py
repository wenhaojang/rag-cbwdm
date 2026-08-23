from __future__ import annotations
import argparse, sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from src.preformal.fairness import audit_preformal
from src.run_manifest import atomic_write_json

def assignments(values:list[str])->dict[str,Path]:
    result={}
    for value in values:
        if "=" not in value:raise ValueError(f"Expected METHOD:SEED=PATH, got {value!r}")
        key,path=value.split("=",1)
        if key in result:raise ValueError(f"Duplicate assignment {key}")
        result[key]=Path(path).resolve()
    return result

def main()->None:
    p=argparse.ArgumentParser(description="Audit preformal retrieval/generator fairness")
    p.add_argument("--split-manifest",required=True);p.add_argument("--retrieval-manifest",required=True);p.add_argument("--posterior-manifest",required=True)
    p.add_argument("--selection-manifest",action="append",default=[]);p.add_argument("--evaluation-manifest",action="append",default=[]);p.add_argument("--output",required=True);a=p.parse_args()
    payload=audit_preformal(split_manifest_path=a.split_manifest,retrieval_manifest_path=a.retrieval_manifest,
        posterior_manifest_path=a.posterior_manifest,selection_manifests=assignments(a.selection_manifest),evaluation_manifests=assignments(a.evaluation_manifest))
    atomic_write_json(a.output,payload);print(f"[preformal_fairness] status={payload['status']} methods={len(payload['methods'])} output={a.output}")
    if payload["status"]!="comparable":raise SystemExit(2)
if __name__=="__main__":main()

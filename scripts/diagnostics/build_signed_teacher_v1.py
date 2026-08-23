from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.method_failure import atomic_write_jsonl, require_diagnostic_output
from src.diagnostics.signed_teacher_v1 import build_signed_teacher_row, teacher_statistics
from src.formal_provenance import atomic_write_text
from src.io_utils import load_yaml, read_jsonl
from src.run_manifest import atomic_write_json, git_state, sha256_file, stable_hash, utc_now


def main() -> None:
    p=argparse.ArgumentParser(description="Build experimental signed_teacher_v1 supervision")
    p.add_argument("--config",required=True);p.add_argument("--run-dir",required=True)
    p.add_argument("--split",choices=["train_core","validation"],default="train_core")
    p.add_argument("--output-dir",required=True);p.add_argument("--limit",type=int,default=None)
    p.add_argument("--top-m",type=int,default=4);p.add_argument("--teacher-stop-threshold",type=float,default=.001)
    p.add_argument("--alignment-eps",type=float,default=0.0);p.add_argument("--b-plus",type=float,default=.01)
    p.add_argument("--b-minus",type=float,default=.001);p.add_argument("--neutral-sample-policy",choices=["negative","ignore"],default="negative")
    p.add_argument("--resume",action="store_true");args=p.parse_args()
    if args.limit is not None and args.limit<1:raise ValueError("--limit must be >= 1")
    if args.top_m<0 or args.alignment_eps<0:raise ValueError("top_m/alignment_eps must be non-negative")
    config_path=Path(args.config).resolve();config=load_yaml(config_path);run=Path(args.run_dir).resolve()
    output=require_diagnostic_output(run,Path(args.output_dir).resolve())
    required_root=(run/"artifacts/diagnostics/method_failure_audit/signed_teacher_v1").resolve()
    try:output.relative_to(required_root/"teacher")
    except ValueError as exc:raise ValueError(f"signed teacher output must be below {required_root/'teacher'}") from exc
    formal=run/"artifacts/formal";dataset=config["dataset"];top_n=config["retrieval"]["top_n"]
    posterior=formal/f"{dataset}_{args.split}_posteriors.jsonl";retrieval=formal/f"{dataset}_{args.split}_bm25_top{top_n}.jsonl"
    params={"top_m":args.top_m,"stop_threshold":args.teacher_stop_threshold,"alignment_eps":args.alignment_eps,
        "b_plus":args.b_plus,"b_minus":args.b_minus,"neutral_sample_policy":args.neutral_sample_policy,
        "ridge_lambda":float(config["cbwdm"]["ridge_lambda"]),"eps_smooth":float(config["cbwdm"].get("eps_smooth",0)),
        "l_type":config["cbwdm"].get("L_type","euclidean_posterior_shift"),
        "target_smoothing":config["cbwdm"].get("target_smoothing","paper_mixture"),
        "gain_tolerance":float(config["cbwdm"].get("gain_tolerance",1e-10))}
    contract={"variant":"signed_teacher_v1","diagnostic_only":True,"uses_gold_for_teacher":True,"deployable_teacher":False,
        "config_path":str(config_path),"config_sha256":sha256_file(config_path),"split":args.split,"limit":args.limit,
        "posteriors":{"path":str(posterior),"sha256":sha256_file(posterior)},
        "retrieval":{"path":str(retrieval),"sha256":sha256_file(retrieval)},"parameters":params}
    fingerprint=stable_hash(contract);manifest_path=output/"manifest.json"
    files=[output/"teacher.jsonl",output/"statistics.json",output/"report.md"]
    if args.resume and manifest_path.is_file():
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status")=="completed" and manifest.get("fingerprint")==fingerprint and all(path.is_file() for path in files):
            if all(manifest["outputs"][path.name]["sha256"]==sha256_file(path) for path in files):
                print(f"[signed_teacher_v1] reused=true output={output}");return
        raise ValueError("Cannot resume signed_teacher_v1: fingerprint/checksum mismatch")
    if manifest_path.exists() or any(path.exists() for path in files):raise FileExistsError("signed_teacher_v1 artifacts exist; use matching --resume or a new directory")
    rows=[build_signed_teacher_row(row,params) for row in read_jsonl(posterior,limit=args.limit)]
    stats=teacher_statistics(rows);output.mkdir(parents=True,exist_ok=True)
    atomic_write_jsonl(files[0],rows);atomic_write_json(files[1],stats)
    atomic_write_text(files[2],"# signed_teacher_v1 Statistics\n\n```json\n"+json.dumps(stats,ensure_ascii=False,indent=2,sort_keys=True)+"\n```\n")
    atomic_write_json(manifest_path,{"schema_version":"rag_cbwdm_signed_teacher_manifest.v1","status":"completed","completed":True,
        "fingerprint":fingerprint,"contract":contract,"num_rows":len(rows),"diagnostic_only":True,"uses_gold_for_teacher":True,
        "deployable_teacher":False,"git":git_state(PROJECT_ROOT),"outputs":{path.name:{"path":str(path),"sha256":sha256_file(path)} for path in files},"completed_at":utc_now()})
    print(f"[signed_teacher_v1] rows={len(rows)} groups={stats['groups']['ALL']['total_groups']} output={output}")


if __name__=="__main__":main()

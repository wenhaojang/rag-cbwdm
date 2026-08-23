from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))

from src.diagnostics.method_failure import require_diagnostic_output
from src.diagnostics.signed_selector_v1 import oracle_imitation,posthoc_alignment,selection_evidence_coverage
from src.formal_provenance import atomic_write_text
from src.io_utils import load_yaml,read_jsonl
from src.run_manifest import atomic_write_json,git_state,sha256_file,utc_now


def _map(path:Path)->dict[str,dict]:return {str(row["id"]):row for row in read_jsonl(path)}


def main()->None:
    p=argparse.ArgumentParser(description="Post-hoc signed_selector_v1 validation diagnostics")
    p.add_argument("--config",required=True);p.add_argument("--run-dir",required=True);p.add_argument("--selection",required=True)
    p.add_argument("--predictions",required=True);p.add_argument("--metrics",required=True);p.add_argument("--oracle-selection",required=True)
    p.add_argument("--output-dir",required=True);args=p.parse_args()
    config_path=Path(args.config).resolve();config=load_yaml(config_path);run=Path(args.run_dir).resolve();output=require_diagnostic_output(run,Path(args.output_dir).resolve())
    root=(run/"artifacts/diagnostics/method_failure_audit/signed_teacher_v1/reports").resolve()
    try:output.relative_to(root)
    except ValueError as exc:raise ValueError(f"post-hoc output must be below {root}") from exc
    selection_path=Path(args.selection).resolve();predictions_path=Path(args.predictions).resolve();metrics_path=Path(args.metrics).resolve();oracle_path=Path(args.oracle_selection).resolve()
    formal=run/"artifacts/formal";dataset=config["dataset"];top_n=config["retrieval"]["top_n"]
    posterior_path=formal/f"{dataset}_validation_posteriors.jsonl";retrieval_path=formal/f"{dataset}_validation_bm25_top{top_n}.jsonl"
    inputs={"config":config_path,"selection":selection_path,"predictions":predictions_path,"metrics":metrics_path,
        "oracle_selection":oracle_path,"posteriors":posterior_path,"retrieval":retrieval_path}
    missing=[str(path) for path in inputs.values() if not path.is_file()]
    if missing:raise FileNotFoundError("Missing post-hoc inputs:\n- "+"\n- ".join(missing))
    selection=_map(selection_path);posterior=_map(posterior_path);retrieval=_map(retrieval_path);oracle=_map(oracle_path)
    for identifier,row in selection.items():
        meta=row.get("selection_metadata",{})
        if row.get("uses_gold_at_test") or meta.get("uses_gold_at_inference") is not False:
            raise ValueError(f"Selection does not prove no-gold inference for id={identifier}")
    payload={"schema_version":"rag_cbwdm_signed_selector_posthoc.v1","created_at":utc_now(),"posthoc_only":True,
        "uses_gold_for_selection":False,"inference_gold_leakage":"NO GOLD LEAKAGE",
        "production_evaluation_metrics":json.loads(metrics_path.read_text(encoding="utf-8")),
        "evidence_coverage":selection_evidence_coverage(selection,retrieval),
        "selected_alignment":posthoc_alignment(selection_rows=selection,posterior_rows=posterior,cbwdm=config["cbwdm"]),
        "signed_oracle_imitation":oracle_imitation(selection_rows=selection,oracle_rows=oracle,posterior_rows=posterior),
        "inputs":{name:{"path":str(path),"sha256":sha256_file(path)} for name,path in inputs.items()},"git":git_state(PROJECT_ROOT)}
    output.mkdir(parents=True,exist_ok=True);json_path=output/"SIGNED_SELECTOR_V1_VALIDATION_DIAGNOSTICS.json";md_path=output/"SIGNED_SELECTOR_V1_VALIDATION_DIAGNOSTICS.md"
    if json_path.exists() or md_path.exists():raise FileExistsError("Post-hoc diagnostics exist; use a new output directory")
    atomic_write_json(json_path,payload);atomic_write_text(md_path,"# signed_selector_v1 Validation Diagnostics\n\n"
        "- Selection-time gold leakage: **NO GOLD LEAKAGE**\n- Gold alignment below is post-hoc only.\n\n"
        f"```json\n{json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)}\n```\n")
    print(f"[signed_selector_v1_posthoc] rows={len(selection)} output={output}")


if __name__=="__main__":main()

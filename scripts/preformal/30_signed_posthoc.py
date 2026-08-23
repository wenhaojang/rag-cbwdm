from __future__ import annotations
import argparse,json,sys
from collections import Counter
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from src.diagnostics.signed_selector_v1 import oracle_imitation,posthoc_alignment,selection_evidence_coverage
from src.io_utils import load_yaml,read_jsonl
from src.preformal.statistics import indexed_rows,require_identical_ids
from src.run_manifest import atomic_write_json,git_state,sha256_file,utc_now

def main()->None:
    p=argparse.ArgumentParser(description="Post-hoc-only signed-v1 diagnostics")
    p.add_argument("--config",required=True);p.add_argument("--selection",required=True);p.add_argument("--posteriors",required=True)
    p.add_argument("--retrieval",required=True);p.add_argument("--oracle-selection");p.add_argument("--output",required=True);a=p.parse_args()
    selection=indexed_rows(read_jsonl(a.selection),"selection");posterior=indexed_rows(read_jsonl(a.posteriors),"posteriors");retrieval=indexed_rows(read_jsonl(a.retrieval),"retrieval")
    require_identical_ids(selection,posterior,"selection","posteriors");require_identical_ids(selection,retrieval,"selection","retrieval")
    config=load_yaml(a.config);lengths=Counter(len(row.get("selected_doc_ids",[])) for row in selection.values());stops=Counter(str(row.get("stop_reason")) for row in selection.values())
    payload={"schema_version":"rag_cbwdm_preformal_signed_posthoc.v1","posthoc_only":True,"uses_gold_for_selection":False,
        "alignment":posthoc_alignment(selection_rows=selection,posterior_rows=posterior,cbwdm=config["cbwdm"]),
        "zero_doc_ratio":lengths[0]/len(selection) if selection else None,"doc_count_distribution":dict(sorted(lengths.items())),
        "stop_reasons":dict(sorted(stops.items())),"evidence_coverage":selection_evidence_coverage(selection,retrieval)}
    if a.oracle_selection:
        oracle=indexed_rows(read_jsonl(a.oracle_selection),"oracle");require_identical_ids(selection,oracle,"selection","oracle")
        payload["oracle_imitation"]=oracle_imitation(selection_rows=selection,oracle_rows=oracle,posterior_rows=posterior)
    payload.update({"inputs":{"selection_sha256":sha256_file(a.selection),"posterior_sha256":sha256_file(a.posteriors),"retrieval_sha256":sha256_file(a.retrieval)},"git":git_state(PROJECT_ROOT),"created_at":utc_now()})
    atomic_write_json(a.output,payload);print(f"[preformal_posthoc] rows={len(selection)} output={a.output}")
if __name__=="__main__":main()

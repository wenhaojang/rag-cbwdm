from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))

from src.baselines.common import build_selection_contract,publish_selection
from src.io_utils import read_jsonl
from src.selection_schema import make_selection_row


def main()->None:
    p=argparse.ArgumentParser(description="Create canonical no-evidence selections for preformal_eval")
    p.add_argument("--retrieval",required=True);p.add_argument("--output",required=True);p.add_argument("--resume",action="store_true");a=p.parse_args()
    retrieval=Path(a.retrieval).resolve();output=Path(a.output).resolve()
    contract=build_selection_contract(method="no_evidence",input_paths={"retrieval":retrieval},
        parameters={"split":"preformal_eval","top_m":0,"uses_gold_at_inference":False},model={})
    if a.resume and output.is_file():
        written,reused=publish_selection(output,[],contract=contract,project_root=PROJECT_ROOT,resume=True)
        print(f"[preformal_no_evidence] rows={written} reused={reused} output={output}");return
    source=list(read_jsonl(retrieval))
    if not source or {row.get("split") for row in source}!={"preformal_eval"}:raise ValueError("Input must contain only preformal_eval retrieval rows")
    rows=(make_selection_row(row,method="no_evidence",selected_docs=[],selection_steps=[],stop_reason="no_evidence",
        diagnostic_only=False,max_docs=0,uses_gold_at_test=False,selection_metadata={"uses_gold_at_inference":False}) for row in source)
    written,reused=publish_selection(output,rows,contract=contract,project_root=PROJECT_ROOT,resume=a.resume)
    print(f"[preformal_no_evidence] rows={written} reused={reused} output={output}")
if __name__=="__main__":main()

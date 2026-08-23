from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.common import build_selection_contract, publish_selection
from src.diagnostics.signed_selector_v1 import select_row_without_gold
from src.formal_provenance import sha256_path
from src.io_utils import read_jsonl
from src.preformal.registry import SIGNED_V1_CONTRACT, assert_frozen_signed_contract
from src.run_manifest import sha256_file
from src.selector_cross_encoder import CrossEncoderSelector


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal deployable rag_cbwdm_signed_v1 selection")
    parser.add_argument("--posteriors", required=True); parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--seed", type=int, required=True, choices=[13,21,42])
    parser.add_argument("--device", default="auto"); parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true"); args = parser.parse_args()
    frozen = SIGNED_V1_CONTRACT["selector"]
    assert_frozen_signed_contract({"top_m": frozen["top_m"], "min_docs": frozen["min_docs"], "score_threshold": frozen["score_threshold"]})
    posterior = Path(args.posteriors).resolve(); checkpoint = Path(args.checkpoint_dir).resolve(); output = Path(args.output).resolve()
    training_manifest = checkpoint.parent / "training_manifest.json"
    payload = json.loads(training_manifest.read_text(encoding="utf-8"))
    if payload.get("method") != "rag_cbwdm_signed_v1" or payload.get("seed") != args.seed or payload.get("status") != "completed":
        raise ValueError("Checkpoint is not the requested completed formal signed-v1 seed")
    if payload.get("checkpoint_sha256") != sha256_path(checkpoint): raise ValueError("Checkpoint SHA mismatch")
    if payload.get("train_core_posterior_sha256") == sha256_file(posterior):
        raise ValueError("Refusing selection on the training posterior artifact")
    contract = build_selection_contract(method="rag_cbwdm_signed_v1", input_paths={"posteriors": posterior,
        "training_manifest": training_manifest}, parameters={"seed": args.seed, "top_m": 4, "min_docs": 0,
        "score_threshold": 0.0, "uses_gold_at_inference": False, "split": "preformal_eval"},
        model={"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_path(checkpoint)})
    if args.resume and output.is_file():
        written, reused = publish_selection(output, [], contract=contract, project_root=PROJECT_ROOT, resume=True)
        print(f"[preformal_signed_select] rows={written} reused={reused} output={output}"); return
    selector = CrossEncoderSelector.load_checkpoint(checkpoint, device=args.device); selector.model.eval()
    rows = list(read_jsonl(posterior))
    if not rows or {row.get("split") for row in rows} != {"preformal_eval"}: raise ValueError("Selection input must be preformal_eval only")
    selected = (select_row_without_gold(row, selector, method="rag_cbwdm_signed_v1", top_m=4, min_docs=0,
        score_threshold=0.0, batch_size=args.batch_size, max_candidates=None) for row in rows)
    written, reused = publish_selection(output, selected, contract=contract, project_root=PROJECT_ROOT, resume=args.resume)
    print(f"[preformal_signed_select] rows={written} reused={reused} uses_gold_at_inference=false output={output}")


if __name__ == "__main__": main()

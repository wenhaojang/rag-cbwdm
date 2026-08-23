from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.method_failure import require_diagnostic_output
from src.diagnostics.signed_gate_error_audit import build_audit, publish
from src.io_utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only signed-gate full500 residual audit")
    parser.add_argument("--config", required=True); parser.add_argument("--run-dir", required=True)
    parser.add_argument("--signed-gate-dir", default=None); parser.add_argument("--output-dir", default=None)
    parser.add_argument("--supplement-json", default=None); args = parser.parse_args()
    config_path = Path(args.config).resolve(); config = load_yaml(config_path); run = Path(args.run_dir).resolve()
    signed = Path(args.signed_gate_dir).resolve() if args.signed_gate_dir else run / "artifacts/diagnostics/method_failure_audit/signed_gate/full500"
    output = Path(args.output_dir).resolve() if args.output_dir else run / "artifacts/diagnostics/method_failure_audit/signed_gate/full500_error_audit"
    output = require_diagnostic_output(run, output)
    formal = run / "artifacts/formal"; dataset = config["dataset"]; top_n = config["retrieval"]["top_n"]
    paths = {"trajectories": signed / "trajectories.jsonl", "selection": signed / "selection.jsonl",
        "predictions": signed / "predictions.jsonl", "metrics": signed / "metrics.json", "manifest": signed / "manifest.json",
        "retrieval": formal / f"{dataset}_validation_bm25_top{top_n}.jsonl",
        "posteriors": formal / f"{dataset}_validation_posteriors.jsonl", "config": config_path}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing: raise FileNotFoundError("Missing residual-audit inputs:\n- " + "\n- ".join(missing))
    supplement = Path(args.supplement_json).resolve() if args.supplement_json else run / "artifacts/diagnostics/method_failure_audit/RAG_CBWDM_METHOD_FAILURE_AUDIT_SERVER_SUPPLEMENT.json"
    payload = build_audit(trajectories_path=paths["trajectories"], selection_path=paths["selection"],
        predictions_path=paths["predictions"], metrics_path=paths["metrics"], retrieval_path=paths["retrieval"],
        posteriors_path=paths["posteriors"], manifest_path=paths["manifest"], config=config, supplement_path=supplement)
    publish(payload, output, PROJECT_ROOT, {**paths, "supplement": supplement})
    print(f"[signed_gate_error_audit] examples={payload['num_aligned_examples']} output={output}")


if __name__ == "__main__": main()

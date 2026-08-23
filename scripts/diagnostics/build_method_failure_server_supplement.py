from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.method_failure import require_diagnostic_output
from src.diagnostics.server_supplement import publish


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a read-only method-failure server supplement.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else run / "artifacts/diagnostics/method_failure_audit"
    output = require_diagnostic_output(run, output)
    expected = [output / "RAG_CBWDM_METHOD_FAILURE_AUDIT_SERVER_SUPPLEMENT.md",
                output / "RAG_CBWDM_METHOD_FAILURE_AUDIT_SERVER_SUPPLEMENT.json"]
    if any(path.exists() for path in expected):
        raise FileExistsError("Supplement output already exists; use a new output directory (existing diagnostics are not overwritten)")
    payload = publish(config_path=Path(args.config).resolve(), run_dir=run,
                      output_dir=output, project_root=PROJECT_ROOT)
    print(f"[method_failure_supplement] status={payload['candidate_audit']['status']} output={output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.formal_readiness import publish_readiness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check every FEVER formal P0 gate.")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--frozen-manifest", required=True)
    parser.add_argument("--fairness-audit", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--cbwdm-diagnostics", required=True)
    parser.add_argument("--tests-status", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--skip-artifact-rehash",
        action="store_true",
        help="Testing/debug only; formal server checks must not use this flag.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = publish_readiness(
        args.output_dir,
        project_root=PROJECT_ROOT,
        split_manifest_path=args.split_manifest,
        calibration_manifest_path=args.calibration_manifest,
        frozen_manifest_path=args.frozen_manifest,
        fairness_audit_path=args.fairness_audit,
        baseline_summary_path=args.baseline_summary,
        diagnostics_path=args.cbwdm_diagnostics,
        tests_status_path=args.tests_status,
        verify_model_artifacts=not args.skip_artifact_rehash,
    )
    print(
        f"[formal_readiness] status={payload['status']} "
        f"blockers={len(payload['blockers'])} output={Path(args.output_dir).resolve()}"
    )
    if payload["status"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

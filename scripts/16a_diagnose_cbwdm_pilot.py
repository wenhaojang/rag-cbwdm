from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cbwdm_diagnostics import publish_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build validation pilot CBWDM diagnostics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-evidence", required=True)
    parser.add_argument("--naive", required=True)
    parser.add_argument("--cbwdm")
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--naive-selection", required=True)
    parser.add_argument("--cbwdm-selection")
    parser.add_argument("--oracle-selection", required=True)
    parser.add_argument("--teacher")
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--posteriors", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration-manifest")
    parser.add_argument("--calibration-candidates")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration_selection = None
    if args.calibration_manifest or args.calibration_candidates:
        if not args.calibration_manifest or not args.calibration_candidates:
            raise ValueError(
                "--calibration-manifest and --calibration-candidates must be provided together"
            )
        calibration = json.loads(
            Path(args.calibration_manifest).read_text(encoding="utf-8")
        )
        winner = calibration.get("selected", {}).get("rag_cbwdm", {})
        fingerprint = winner.get("candidate_fingerprint")
        if winner.get("status") != "selected" or not fingerprint:
            raise ValueError("Calibration manifest lacks a selected RAG-CBWDM candidate")
        candidate_payload = json.loads(
            Path(args.calibration_candidates).read_text(encoding="utf-8")
        )
        candidates = (
            candidate_payload.get("candidates")
            if isinstance(candidate_payload, dict)
            else candidate_payload
        )
        if not isinstance(candidates, list):
            raise ValueError("Calibration candidates file has no candidate list")
        matches = [
            item
            for item in candidates
            if isinstance(item, dict)
            and item.get("candidate_fingerprint") == fingerprint
            and item.get("method") == "rag_cbwdm"
        ]
        if len(matches) != 1 or matches[0].get("status") != "completed":
            raise ValueError(
                f"Selected RAG-CBWDM candidate is missing or incomplete: {fingerprint}"
            )
        selected = matches[0]
        args.cbwdm = selected.get("prediction_path")
        args.cbwdm_selection = selected.get("selection_path")
        args.teacher = selected.get("teacher_path")
        calibration_selection = {
            "candidate_fingerprint": fingerprint,
            "training_fingerprint": selected.get("training_fingerprint"),
            "selection_fingerprint": selected.get("selection_fingerprint"),
            "calibration_manifest": str(Path(args.calibration_manifest).resolve()),
            "calibration_candidates": str(
                Path(args.calibration_candidates).resolve()
            ),
        }
    missing_winner_inputs = [
        name
        for name in ("cbwdm", "cbwdm_selection", "teacher")
        if not getattr(args, name)
    ]
    if missing_winner_inputs:
        raise ValueError(
            "CBWDM diagnostics require explicit winner artifacts or calibration "
            f"resolution; missing={missing_winner_inputs}"
        )
    keys = (
        "no_evidence",
        "naive",
        "cbwdm",
        "oracle",
        "naive_selection",
        "cbwdm_selection",
        "oracle_selection",
        "teacher",
        "retrieval",
        "posteriors",
    )
    payload = publish_diagnostics(
        config_path=args.config,
        paths={key: getattr(args, key) for key in keys},
        output_dir=args.output_dir,
        project_root=PROJECT_ROOT,
        calibration_selection=calibration_selection,
    )
    print(
        f"[cbwdm_diagnostics] status={payload['status']} "
        f"output={Path(args.output_dir).resolve()}"
    )
    if payload["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

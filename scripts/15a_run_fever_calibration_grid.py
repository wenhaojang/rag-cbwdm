from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.grid import build_grid_plan, execute_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute the YAML-defined FEVER validation calibration grid sequentially."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--train-retrieval", required=True)
    parser.add_argument("--validation-retrieval", required=True)
    parser.add_argument("--train-posteriors", required=True)
    parser.add_argument("--validation-posteriors", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--generator-model")
    parser.add_argument("--selector-model")
    parser.add_argument("--infogain-model")
    parser.add_argument("--generator-device", default="auto")
    parser.add_argument("--selector-device", default="auto")
    parser.add_argument("--infogain-device", default="auto")
    parser.add_argument("--methods", default="infogain_fever,rag_cbwdm")
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--candidate-fingerprint")
    parser.add_argument("--max-training-candidates", type=int)
    parser.add_argument("--skip-completed", action="store_true")
    failure = parser.add_mutually_exclusive_group()
    failure.add_argument("--fail-fast", action="store_true")
    failure.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = {value.strip() for value in args.methods.split(",") if value.strip()}
    unknown = methods - {"infogain_fever", "rag_cbwdm"}
    if unknown:
        raise ValueError(f"Unsupported grid methods: {sorted(unknown)}")
    plan = build_grid_plan(
        config_path=args.config,
        split_manifest_path=args.split_manifest,
        train_retrieval_path=args.train_retrieval,
        validation_retrieval_path=args.validation_retrieval,
        train_posteriors_path=args.train_posteriors,
        validation_posteriors_path=args.validation_posteriors,
        output_dir=args.output_dir,
        project_root=PROJECT_ROOT,
        generator_model=args.generator_model,
        selector_model=args.selector_model,
        infogain_model=args.infogain_model,
    )
    result = execute_grid(
        plan,
        config_path=args.config,
        project_root=PROJECT_ROOT,
        methods=methods,
        candidate_limit=args.candidate_limit,
        candidate_fingerprint=args.candidate_fingerprint,
        max_training_candidates=args.max_training_candidates,
        skip_completed=args.skip_completed,
        fail_fast=args.fail_fast,
        continue_on_error=args.continue_on_error or not args.fail_fast,
        generator_model=args.generator_model,
        selector_model=args.selector_model,
        infogain_model=args.infogain_model,
        generator_device=args.generator_device,
        selector_device=args.selector_device,
        infogain_device=args.infogain_device,
        seed=args.seed,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(
            f"[calibration_grid] status={result['status']} "
            f"candidates={result['candidate_count']}"
        )
        if result["status"] not in {"completed", "completed_with_failures"}:
            raise SystemExit(2)


if __name__ == "__main__":
    main()

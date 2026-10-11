"""Stage-gated FEVER document-budget sweep runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fever_budget_sweep import (
    audit_selections,
    build_plan,
    load_budget_config,
    materialize_selections,
    preflight_sources,
    run_evaluations,
    summarize,
)
from src.run_manifest import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan, audit, materialize, evaluate, or summarize the FEVER budget sweep."
    )
    parser.add_argument("--config", required=True)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--preflight", action="store_true")
    stage.add_argument("--dry-run", action="store_true")
    stage.add_argument("--materialize-selections", action="store_true")
    stage.add_argument("--audit-selections", action="store_true")
    stage.add_argument("--run-evaluations", action="store_true")
    stage.add_argument("--summarize", action="store_true")
    parser.add_argument("--artifact-root")
    parser.add_argument("--plan-output")
    parser.add_argument("--job-plan")
    parser.add_argument("--audit-manifest")
    parser.add_argument("--output-csv")
    parser.add_argument("--summary-manifest")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--state-aware-inference",
        action="store_true",
        help=(
            "Re-run the frozen signed/KCBWDM checkpoint with --inference-budget "
            "instead of replaying a verified k4 trajectory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.state_aware_inference and not (
        args.preflight or args.dry_run or args.materialize_selections
    ):
        raise ValueError(
            "--state-aware-inference is valid only for preflight, dry-run, or materialization"
        )
    config = load_budget_config(args.config)
    if args.artifact_root:
        config = {**config, "artifact_root": args.artifact_root}
    plan = build_plan(config, artifact_root=args.artifact_root)
    root = str(plan["artifact_root"])
    plan_path = args.plan_output or str(Path(root) / "plan" / "plan.json")

    if args.preflight:
        result = preflight_sources(
            config,
            include_evaluations=True,
            causal_replay_required=not args.state_aware_inference,
        )["report"]
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.dry_run:
        if args.plan_output:
            atomic_write_json(args.plan_output, plan)
        print(
            f"[fever_budget][dry-run] points={len(plan['points'])} "
            f"new_selections={len(plan['selection_tasks'])} "
            f"state_aware_mode={'inference' if args.state_aware_inference else 'replay'} "
            f"artifact_root={root}"
        )
        for point in plan["points"]:
            print(
                f"{point['point_id']} selection={point['selection_source']} "
                f"strategy={point['strategy']} "
                f"evaluation={point['evaluation_disposition']}"
            )
        return

    if args.materialize_selections:
        atomic_write_json(plan_path, plan)
        result = materialize_selections(
            config,
            plan,
            project_root=PROJECT_ROOT,
            resume=args.resume,
            state_aware_mode=(
                "inference" if args.state_aware_inference else "replay"
            ),
        )
        manifest_path = str(Path(root) / "plan" / "materialization.manifest.json")
        atomic_write_json(manifest_path, result)
        print(
            f"[fever_budget][materialize] selections={result['selection_count']} "
            f"manifest={manifest_path}"
        )
        return

    if args.audit_selections:
        atomic_write_json(plan_path, plan)
        result = audit_selections(config, plan)
        print(
            f"[fever_budget][audit] points={result['audit']['logical_point_count']} "
            f"evaluation_jobs={result['audit']['evaluation_job_count']} "
            f"job_plan={result['job_plan_path']}"
        )
        return

    if args.run_evaluations:
        job_plan = args.job_plan or str(Path(root) / "audit" / "evaluation_jobs.json")
        run_evaluations(job_plan)
        print(f"[fever_budget][evaluate] completed job_plan={job_plan}")
        return

    audit_manifest = args.audit_manifest or str(Path(root) / "audit" / "selection_audit.json")
    output_csv = args.output_csv or str(Path(root) / "summaries" / "fever_budget_accuracy_tokens.csv")
    summary_manifest = args.summary_manifest or str(Path(root) / "summaries" / "fever_budget_accuracy_tokens.manifest.json")
    result = summarize(
        audit_manifest,
        output_csv=output_csv,
        output_manifest=summary_manifest,
    )
    print(
        f"[fever_budget][summarize] rows={result['row_count']} "
        f"csv={result['output_csv']}"
    )


if __name__ == "__main__":
    main()

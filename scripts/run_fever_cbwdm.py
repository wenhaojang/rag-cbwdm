from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import load_yaml, read_jsonl
from src.formal_config import validate_frozen_manifest
from src.formal_splits import validate_split_manifest
from src.prompts import FEVER_PROMPT_VERSION, fever_prompt_hash
from src.retrieval.pyserini_bm25 import (
    index_contract,
    pyserini_version,
    validate_index,
)
from src.run_manifest import (
    atomic_write_json,
    environment_info,
    git_state,
    sha256_file,
    stable_hash,
    utc_now,
)

ALL_STAGES = [
    "prepare_formal_splits",
    "retrieve_train_core",
    "retrieve_validation",
    "retrieve_test",
    "posterior_train_core",
    "posterior_validation",
    "posterior_test",
    "run_calibration_grid",
    "calibrate_methods",
    "cbwdm_diagnostics",
    "prepare",
    "corpus",
    "index",
    "retrieve",
    "posterior",
    "teacher",
    "train_cross_encoder",
    "select_cross_encoder",
    "eval",
    "no_evidence",
    "naive_topm",
    "oracle_diagnostic",
    "score_bge",
    "select_bge",
    "build_infogain_teacher",
    "train_infogain",
    "select_infogain",
    "eval_naive_topm",
    "eval_bge",
    "eval_infogain",
    "eval_cbwdm",
    "eval_oracle",
    "fairness_audit",
    "summarize_baselines",
]
ALIASES = {
    "train": "train_cross_encoder",
    "select": "select_cross_encoder",
    "build_bm25_index": "index",
    "select_naive_topm": "naive_topm",
    "select_oracle": "oracle_diagnostic",
    "eval_no_evidence": "no_evidence",
    "bge": "select_bge",
    "infogain_fever": "select_infogain",
    "calibrate_infogain": "calibrate_methods",
    "calibrate_cbwdm": "calibrate_methods",
}
BASELINE_SUITE = [
    "naive_topm",
    "score_bge",
    "select_bge",
    "build_infogain_teacher",
    "train_infogain",
    "select_infogain",
    "oracle_diagnostic",
    "no_evidence",
    "eval_naive_topm",
    "eval_bge",
    "eval_infogain",
    "eval_cbwdm",
    "eval_oracle",
    "fairness_audit",
    "summarize_baselines",
]
FORMAL_PILOT_STAGES = [
    "prepare_formal_splits",
    "corpus",
    "index",
    "retrieve_train_core",
    "retrieve_validation",
    "posterior_train_core",
    "posterior_validation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Linux/server RAG-CBWDM pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--frozen-manifest",
        help="Required sidecar manifest when profile=server_formal_frozen.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stages", default="prepare,corpus,index,retrieve,posterior,teacher,train_cross_encoder,select_cross_encoder,eval")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-stage", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--dev-limit", type=int)
    parser.add_argument("--raw-limit", type=int)
    parser.add_argument("--corpus-limit", type=int)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-root", default="outputs/runs")
    parser.add_argument("--cache-root", default=".cache/huggingface")
    parser.add_argument("--generator-model")
    parser.add_argument("--selector-model")
    parser.add_argument("--bge-model", help="Local BGE reranker path or frozen model id.")
    parser.add_argument("--infogain-model", help="Local InfoGain backbone path or frozen model id.")
    parser.add_argument("--bge-device", default="auto")
    parser.add_argument("--infogain-device", default="auto")
    parser.add_argument("--selector-device", default="auto")
    parser.add_argument("--posterior-batch-size", type=int)
    parser.add_argument("--selector-batch-size", type=int)
    parser.add_argument(
        "--methods",
        default="infogain_fever,rag_cbwdm",
        help="Methods for run_calibration_grid.",
    )
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--candidate-fingerprint")
    parser.add_argument("--max-training-candidates", type=int)
    parser.add_argument("--skip-completed", action="store_true")
    grid_failure = parser.add_mutually_exclusive_group()
    grid_failure.add_argument("--fail-fast", action="store_true")
    grid_failure.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def absolute(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def resolve_limits(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, int | None]:
    profile = config.get("profile_limits", {})
    return {
        "train": args.train_limit
        if args.train_limit is not None
        else profile.get("train", args.limit),
        "dev": args.dev_limit
        if args.dev_limit is not None
        else profile.get("dev", args.limit),
        "corpus": args.corpus_limit
        if args.corpus_limit is not None
        else profile.get("corpus"),
        "raw": args.raw_limit,
    }


def load_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, f"cannot read JSON: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON at line {exc.lineno}: {exc.msg}"
    if not isinstance(payload, dict):
        return None, f"expected JSON object, got {type(payload).__name__}"
    return payload, None


def jsonl_row_count(path: Path) -> tuple[int | None, str | None]:
    try:
        return sum(1 for _ in read_jsonl(path)), None
    except (OSError, ValueError) as exc:
        return None, f"invalid JSONL: {exc}"


def posterior_provenance(
    *,
    config: dict[str, Any],
    config_path: Path,
    retrieval_path: Path,
    split: str,
    model_name: str,
    batch_size: int,
) -> dict[str, Any]:
    generator = config["generator"]
    revision = generator.get("revision")
    tokenizer_revision = generator.get("tokenizer_revision") or revision
    labels = list(config["task"]["labels"])
    verbalizers = dict(config["task"]["verbalizers"])
    return {
        "dataset": config["dataset"],
        "split": split,
        "generator_model": model_name,
        "generator_revision": revision,
        "tokenizer_name": model_name,
        "tokenizer_revision": tokenizer_revision,
        "dtype": generator.get("dtype", "auto"),
        "device_map": generator.get("device_map", "auto"),
        "trust_remote_code": bool(generator.get("trust_remote_code", False)),
        "prompt_template_version": FEVER_PROMPT_VERSION,
        "prompt_template_hash": fever_prompt_hash(labels, verbalizers),
        "labels": labels,
        "verbalizers": verbalizers,
        "verbalizers_hash": stable_hash(verbalizers),
        "max_context_tokens": generator.get("max_context_tokens"),
        "batch_size": batch_size,
        "max_candidates": None,
        "input_path": str(retrieval_path.resolve()),
        "input_sha256": sha256_file(retrieval_path),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
    }


def validate_completed_boolean_manifest(
    manifest_path: Path, output_path: Path | None = None
) -> list[str]:
    payload, error = load_json_object(manifest_path)
    if error:
        return [error]
    assert payload is not None
    reasons = []
    if payload.get("completed") is not True:
        reasons.append("completed: expected true")
    expected_sha = payload.get("output_sha256")
    if expected_sha and output_path:
        actual_sha = sha256_file(output_path)
        if actual_sha != expected_sha:
            reasons.append(
                f"output_sha256: expected={expected_sha} actual={actual_sha}"
            )
    return reasons


def validate_posterior_artifact(
    *,
    split: str,
    output_path: Path,
    manifest_path: Path,
    retrieval_path: Path,
    query_path: Path,
    expected_provenance: dict[str, Any],
) -> list[str]:
    payload, error = load_json_object(manifest_path)
    if error:
        return [error]
    assert payload is not None
    reasons: list[str] = []
    expected_fingerprint = stable_hash(expected_provenance)
    expected_fields = {
        "schema_version": "rag_cbwdm_posterior_manifest.v1",
        "stage": "posterior",
        "status": "completed",
        "fingerprint": expected_fingerprint,
    }
    for field, expected in expected_fields.items():
        actual = payload.get(field)
        if actual != expected:
            reasons.append(f"{field}: expected={expected!r} actual={actual!r}")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        reasons.append("provenance: expected object")
    else:
        for field, expected in expected_provenance.items():
            actual = provenance.get(field)
            if actual != expected:
                reasons.append(
                    f"provenance.{field}: expected={expected!r} actual={actual!r}"
                )
    expected_sha = payload.get("output_sha256")
    actual_sha = sha256_file(output_path)
    if expected_sha != actual_sha:
        reasons.append(
            f"output_sha256: expected={expected_sha!r} actual={actual_sha!r}"
        )
    output_rows, output_error = jsonl_row_count(output_path)
    retrieval_rows, retrieval_error = jsonl_row_count(retrieval_path)
    query_rows, query_error = jsonl_row_count(query_path)
    for label, count_error in (
        ("posterior", output_error),
        ("retrieval", retrieval_error),
        ("prepared query", query_error),
    ):
        if count_error:
            reasons.append(f"{label}: {count_error}")
    if not any((output_error, retrieval_error, query_error)):
        assert output_rows is not None and retrieval_rows is not None and query_rows is not None
        if len({output_rows, retrieval_rows, query_rows}) != 1:
            reasons.append(
                "row_count: "
                f"posterior={output_rows} retrieval={retrieval_rows} prepared={query_rows}"
            )
        for field in ("expected_rows", "completed_rows"):
            actual = payload.get(field)
            if actual != retrieval_rows:
                reasons.append(
                    f"{field}: expected={retrieval_rows} actual={actual!r}"
                )
    if payload.get("split") not in (None, split):
        reasons.append(f"split: expected={split!r} actual={payload.get('split')!r}")
    return reasons


def validate_stage_outputs(
    stage: str,
    stage_outputs: list[Path],
    context: dict[str, Any],
) -> tuple[list[str], list[str]]:
    missing = [str(path) for path in stage_outputs if not path.exists()]
    if missing:
        return missing, []
    invalid: list[str] = []
    if stage == "prepare_formal_splits":
        try:
            validate_split_manifest(stage_outputs[-1])
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            invalid.append(f"{stage_outputs[-1]}: {exc}")
    elif stage in {
        "prepare",
        "corpus",
        "retrieve",
        "retrieve_train_core",
        "retrieve_validation",
        "retrieve_test",
    }:
        for manifest_path in (
            path for path in stage_outputs if path.name.endswith(".manifest.json")
        ):
            output_path = manifest_path.with_name(
                manifest_path.name.removesuffix(".manifest.json") + ".jsonl"
            )
            for reason in validate_completed_boolean_manifest(
                manifest_path, output_path
            ):
                invalid.append(f"{manifest_path}: {reason}")
    elif stage in {
        "posterior_train_core",
        "posterior_validation",
        "posterior_test",
    }:
        role = {
            "posterior_train_core": "train_core",
            "posterior_validation": "validation",
            "posterior_test": "held_out_test",
        }[stage]
        output_path = context["formal_posterior"][role]
        manifest_path = output_path.with_suffix(".manifest.json")
        for reason in validate_posterior_artifact(
            split=role,
            output_path=output_path,
            manifest_path=manifest_path,
            retrieval_path=context["formal_retrieval"][role],
            query_path=context["formal_query"][role],
            expected_provenance=context["formal_posterior_provenance"](role),
        ):
            invalid.append(f"{manifest_path}: {reason}")
    elif stage == "posterior":
        for split in ("train", "dev"):
            output_path = context["posterior"][split]
            manifest_path = output_path.with_suffix(".manifest.json")
            for reason in validate_posterior_artifact(
                split=split,
                output_path=output_path,
                manifest_path=manifest_path,
                retrieval_path=context["retrieval"][split],
                query_path=context["query"][split],
                expected_provenance=context["posterior_provenance"](split),
            ):
                invalid.append(f"{manifest_path}: {reason}")
    elif stage == "teacher":
        for split in ("train", "dev"):
            output_rows, error = jsonl_row_count(context["teacher"][split])
            input_rows, input_error = jsonl_row_count(context["posterior"][split])
            if error:
                invalid.append(f"{context['teacher'][split]}: {error}")
            if input_error:
                invalid.append(f"{context['posterior'][split]}: {input_error}")
            if not error and not input_error and output_rows != input_rows:
                invalid.append(
                    f"{context['teacher'][split]}: row_count expected={input_rows} "
                    f"actual={output_rows}"
                )
    elif stage == "train_cross_encoder":
        for path in stage_outputs:
            _, error = load_json_object(path)
            if error:
                invalid.append(f"{path}: {error}")
    elif stage == "select_cross_encoder":
        output_rows, error = jsonl_row_count(context["selection"])
        input_rows, input_error = jsonl_row_count(context["posterior"]["dev"])
        if error:
            invalid.append(f"{context['selection']}: {error}")
        if input_error:
            invalid.append(f"{context['posterior']['dev']}: {input_error}")
        if not error and not input_error and output_rows != input_rows:
            invalid.append(
                f"{context['selection']}: row_count expected={input_rows} actual={output_rows}"
            )
        for reason in validate_completed_boolean_manifest(
            context["selection"].with_suffix(".manifest.json"), context["selection"]
        ):
            invalid.append(f"{context['selection'].with_suffix('.manifest.json')}: {reason}")
    elif stage in {
        "naive_topm",
        "oracle_diagnostic",
        "select_bge",
        "select_infogain",
        "score_bge",
        "build_infogain_teacher",
    }:
        output_path, manifest_path = stage_outputs
        for reason in validate_completed_boolean_manifest(manifest_path, output_path):
            invalid.append(f"{manifest_path}: {reason}")
    elif stage in {
        "eval",
        "no_evidence",
        "eval_naive_topm",
        "eval_bge",
        "eval_infogain",
        "eval_cbwdm",
        "eval_oracle",
    }:
        prediction_path, metrics_path = stage_outputs[:2]
        prediction_rows, error = jsonl_row_count(prediction_path)
        metrics, metrics_error = load_json_object(metrics_path)
        if error:
            invalid.append(f"{prediction_path}: {error}")
        if metrics_error:
            invalid.append(f"{metrics_path}: {metrics_error}")
        if not error and not metrics_error:
            assert metrics is not None
            if metrics.get("num_examples") != prediction_rows:
                invalid.append(
                    f"{metrics_path}: num_examples expected={prediction_rows} "
                    f"actual={metrics.get('num_examples')!r}"
                )
        for reason in validate_completed_boolean_manifest(stage_outputs[2]):
            invalid.append(f"{stage_outputs[2]}: {reason}")
    elif stage == "run_calibration_grid":
        payload, error = load_json_object(stage_outputs[2])
        if error:
            invalid.append(f"{stage_outputs[2]}: {error}")
        elif payload and payload.get("status") not in {
            "completed",
            "completed_with_failures",
        }:
            invalid.append(
                f"{stage_outputs[2]}: grid status={payload.get('status')!r}"
            )
        elif payload:
            for path in (stage_outputs[0], stage_outputs[1], stage_outputs[3]):
                expected = payload.get("output_sha256", {}).get(path.name)
                if expected != sha256_file(path):
                    invalid.append(f"{path}: output SHA differs from grid manifest")
    elif stage == "cbwdm_diagnostics":
        payload, error = load_json_object(stage_outputs[0])
        if error:
            invalid.append(f"{stage_outputs[0]}: {error}")
        elif payload and (
            payload.get("status") != "passed"
            or payload.get("gate", {}).get("status") != "passed"
            or not payload.get("calibration_selection", {}).get(
                "candidate_fingerprint"
            )
        ):
            invalid.append(
                f"{stage_outputs[0]}: diagnostic gate failed or winner provenance missing"
            )
    elif stage == "calibrate_methods":
        payload, error = load_json_object(stage_outputs[-1])
        if error:
            invalid.append(f"{stage_outputs[-1]}: {error}")
        elif payload and payload.get("status") != "completed":
            invalid.append(
                f"{stage_outputs[-1]}: calibration status={payload.get('status')!r}"
            )
    elif stage in {"train_infogain", "fairness_audit"}:
        for path in stage_outputs:
            if path.suffix == ".json":
                payload, error = load_json_object(path)
                if error:
                    invalid.append(f"{path}: {error}")
                elif stage == "fairness_audit" and payload and payload.get("status") != "comparable":
                    invalid.append(f"{path}: status={payload.get('status')!r}")
    elif stage == "summarize_baselines":
        summary_dir = context.get("baseline_summary_dir")
        if not isinstance(summary_dir, Path):
            invalid.append("validation context lacks baseline_summary_dir")
        else:
            expected_paths = [
                summary_dir / "baseline_summary.json",
                summary_dir / "baseline_summary.csv",
                summary_dir / "baseline_summary.md",
            ]
            if stage_outputs != expected_paths:
                invalid.append(
                    f"summary outputs must be read from {summary_dir}: "
                    f"actual={[str(path) for path in stage_outputs]}"
                )
            payload, error = load_json_object(expected_paths[0])
            if error:
                invalid.append(f"{expected_paths[0]}: {error}")
            elif payload is not None:
                if payload.get("status") != "completed" or payload.get("comparable") is not True:
                    invalid.append(
                        f"{expected_paths[0]}: expected completed comparable summary"
                    )
                methods = payload.get("methods")
                actual_methods = (
                    [row.get("method") for row in methods if isinstance(row, dict)]
                    if isinstance(methods, list)
                    else None
                )
                expected_methods = [
                    "no_evidence",
                    "naive_topm",
                    "bge",
                    "infogain_fever",
                    "rag_cbwdm",
                    "cbwdm_oracle",
                ]
                if actual_methods != expected_methods:
                    invalid.append(
                        f"{expected_paths[0]}: canonical methods expected={expected_methods!r} "
                        f"actual={actual_methods!r}"
                    )
                if isinstance(methods, list):
                    oracle = next(
                        (
                            row
                            for row in methods
                            if isinstance(row, dict)
                            and row.get("method") == "cbwdm_oracle"
                        ),
                        None,
                    )
                    if not oracle or oracle.get("deployable") is not False or oracle.get(
                        "diagnostic_only"
                    ) is not True:
                        invalid.append(
                            f"{expected_paths[0]}: cbwdm_oracle must be non-deployable diagnostic"
                        )
    return missing, invalid


def require_valid_stage_outputs(
    stage: str, stage_outputs: list[Path], context: dict[str, Any]
) -> None:
    missing, invalid = validate_stage_outputs(stage, stage_outputs, context)
    if missing:
        raise RuntimeError(f"Stage {stage!r} has missing outputs: {missing}")
    if invalid:
        details = "\n  - ".join(invalid)
        raise RuntimeError(f"Stage {stage!r} has invalid outputs:\n  - {details}")


def main() -> None:
    args = parse_args()
    config_path = absolute(args.config)
    config = load_yaml(config_path)
    if config.get("profile") == "server_formal_frozen":
        if not args.frozen_manifest:
            raise ValueError("Frozen formal runs require --frozen-manifest")
        frozen_manifest = validate_frozen_manifest(absolute(args.frozen_manifest))
        if Path(frozen_manifest["frozen_config_path"]).resolve() != config_path.resolve():
            raise ValueError("--config does not match the frozen manifest config path")
        forbidden_values = {
            "--limit": args.limit,
            "--train-limit": args.train_limit,
            "--dev-limit": args.dev_limit,
            "--raw-limit": args.raw_limit,
            "--corpus-limit": args.corpus_limit,
            "--generator-model": args.generator_model,
            "--selector-model": args.selector_model,
            "--bge-model": args.bge_model,
            "--infogain-model": args.infogain_model,
            "--posterior-batch-size": args.posterior_batch_size,
            "--selector-batch-size": args.selector_batch_size,
        }
        explicit_flags = {
            token.split("=", 1)[0]
            for token in sys.argv[1:]
            if token.startswith("--")
        }
        used = sorted(
            flag
            for flag, value in forbidden_values.items()
            if value is not None or flag in explicit_flags
        )
        frozen_seed = int(
            config.get("formal_protocol", {}).get("runtime", {}).get("seed", 13)
        )
        if args.seed != frozen_seed or "--seed" in explicit_flags:
            used.append("--seed")
        if used:
            raise ValueError(
                "Frozen formal runs forbid critical CLI overrides: " + ", ".join(used)
            )
    limits = resolve_limits(args, config)
    if config.get("profile") == "server_formal" and limits["corpus"] is not None:
        raise ValueError("Formal runs must use the full corpus (resolved corpus limit must be null)")
    run_contract = {
        "config": config,
        "seed": args.seed,
        "generator_model_override": args.generator_model,
        "selector_model_override": args.selector_model,
        "resolved_limits": limits,
    }
    if args.frozen_manifest:
        run_contract["frozen_manifest"] = {
            "path": str(absolute(args.frozen_manifest)),
            "sha256": sha256_file(absolute(args.frozen_manifest)),
        }
    if config.get("baselines") or args.bge_model or args.infogain_model:
        run_contract["baseline_overrides"] = {
            "bge_model": args.bge_model,
            "infogain_model": args.infogain_model,
            "bge_device": args.bge_device,
            "infogain_device": args.infogain_device,
        }
    run_fingerprint = stable_hash(run_contract)
    requested = []
    for raw_value in args.stages.split(","):
        value = raw_value.strip()
        if not value:
            continue
        if value == "baselines":
            requested.extend(BASELINE_SUITE)
        elif value == "pilot":
            requested.extend(FORMAL_PILOT_STAGES)
        else:
            requested.append(ALIASES.get(value, value))
    unknown = sorted(set(requested) - set(ALL_STAGES))
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}; choices={ALL_STAGES}")
    overwritten = {
        ALIASES.get(value, value)
        for item in args.overwrite_stage
        for value in item.split(",")
        if value
    }
    unknown_overwrite = sorted(overwritten - set(ALL_STAGES))
    if unknown_overwrite:
        raise ValueError(f"Unknown --overwrite-stage values: {unknown_overwrite}")
    run_dir = absolute(args.output_root) / args.run_name
    logs_dir, commands_dir, artifacts = run_dir / "logs", run_dir / "commands", run_dir / "artifacts"
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != run_fingerprint:
            existing_config = dict(manifest.get("config_snapshot") or {})
            requested_config = dict(config)
            existing_config.pop("baselines", None)
            requested_config.pop("baselines", None)
            baseline_extension_compatible = (
                existing_config == requested_config
                and manifest.get("seed") == args.seed
                and manifest.get("resolved_limits") == limits
            )
            if not baseline_extension_compatible:
                raise ValueError(
                    "Run manifest fingerprint differs from the requested config/seed/model/limit. "
                    "Use a new --run-name instead of resuming incompatible artifacts."
                )
        if not args.resume and not overwritten and not args.dry_run:
            raise FileExistsError(f"Run exists: {run_dir}. Use --resume or --overwrite-stage.")
    else:
        manifest = {
            "schema_version": "rag_cbwdm_run_manifest.v1",
            "run_name": args.run_name,
            "fingerprint": run_fingerprint,
            "git": git_state(PROJECT_ROOT),
            "config_path": str(config_path),
            "config_snapshot": config,
            "seed": args.seed,
            "dataset": config.get("dataset"),
            "splits": (
                ["train_core", "validation", "held_out_test"]
                if config.get("formal_splits", {}).get("enabled")
                else ["train", "dev"]
            ),
            "resolved_limits": limits,
            "models": {
                "generator": args.generator_model or config.get("generator", {}).get("model_name"),
                "generator_revision": config.get("generator", {}).get("revision"),
                "selector": args.selector_model or config.get("selector", {}).get("model_name"),
                "selector_revision": config.get("selector", {}).get("revision"),
                "bge": args.bge_model or config.get("baselines", {}).get("bge", {}).get("model_name"),
                "infogain": args.infogain_model or config.get("baselines", {}).get("infogain_fever", {}).get("model_name"),
            },
            "prompt_hash": fever_prompt_hash(
                list(config["task"]["labels"]), dict(config["task"]["verbalizers"])
            ),
            "environment": environment_info(),
            "start_time": utc_now(),
            "end_time": None,
            "paths": {},
            "stages": {stage: {"status": "pending"} for stage in ALL_STAGES},
        }
    manifest.setdefault("stages", {})
    for stage in ALL_STAGES:
        manifest["stages"].setdefault(stage, {"status": "pending"})
    py = sys.executable
    dataset = str(config["dataset"])
    top_n = int(config.get("retrieval", {}).get("top_n", 20))
    cbwdm_top_m = int(config.get("cbwdm", {}).get("top_m", 4))
    baseline_config = config.get("baselines", {})
    common_baseline = baseline_config.get("common", {})
    top_m = int(common_baseline.get("top_m", cbwdm_top_m))
    if top_m != cbwdm_top_m and any(stage in BASELINE_SUITE for stage in requested):
        raise ValueError(
            f"Fair baseline run requires baselines.common.top_m ({top_m}) to equal "
            f"cbwdm.top_m ({cbwdm_top_m})"
        )
    common_min_docs = int(common_baseline.get("min_docs", top_m))
    processed = absolute(config["paths"]["processed_dir"])
    shared = absolute(args.output_root) / "_shared"
    corpus_key = stable_hash(
        {
            "wiki": config["paths"].get("raw_wiki_pages_dir"),
            "corpus": config.get("corpus", {}),
            "limit": limits["corpus"],
        }
    )[:16]
    corpus = shared / "corpora" / corpus_key / "fever_corpus_sentence.jsonl"
    retrieval_config = config.get("retrieval", {})
    index_key = stable_hash(
        {"corpus_key": corpus_key, "retrieval": retrieval_config}
    )[:16]
    configured_index_path = retrieval_config.get("index", {}).get("path")
    index_path = (
        absolute(configured_index_path)
        if configured_index_path
        else shared / "indexes" / index_key
    )
    query = {split: artifacts / f"{dataset}_{split}.jsonl" for split in ("train", "dev")}
    retrieval = {split: artifacts / f"{dataset}_{split}_bm25_top{top_n}.jsonl" for split in ("train", "dev")}
    posterior = {split: artifacts / f"{dataset}_{split}_posteriors.jsonl" for split in ("train", "dev")}
    teacher = {split: artifacts / f"{dataset}_{split}_teacher.jsonl" for split in ("train", "dev")}
    checkpoint = artifacts / "cross_encoder"
    selection = artifacts / f"{dataset}_dev_cross_encoder_selection.jsonl"
    predictions = artifacts / f"{dataset}_dev_predictions.jsonl"
    metrics = artifacts / f"{dataset}_dev_metrics.json"
    selections_dir = artifacts / "selections"
    eval_dir = artifacts / "eval"
    baseline_dir = artifacts / "baselines"
    naive = selections_dir / f"naive_top{top_m}.jsonl"
    oracle = selections_dir / f"cbwdm_oracle_top{top_m}.jsonl"
    bge_selection = selections_dir / f"bge_top{top_m}.jsonl"
    infogain_selection = selections_dir / f"infogain_top{top_m}.jsonl"
    bge_score_cache = baseline_dir / "bge_candidate_scores.jsonl"
    infogain_teacher = baseline_dir / "infogain_train_teacher.jsonl"
    infogain_train_dir = baseline_dir / "infogain_reranker"
    no_evidence_predictions = eval_dir / "no_evidence_predictions.jsonl"
    no_evidence_metrics = eval_dir / "no_evidence_metrics.json"
    naive_predictions = eval_dir / f"naive_top{top_m}_predictions.jsonl"
    naive_metrics = eval_dir / f"naive_top{top_m}_metrics.json"
    bge_predictions = eval_dir / f"bge_top{top_m}_predictions.jsonl"
    bge_metrics = eval_dir / f"bge_top{top_m}_metrics.json"
    infogain_predictions = eval_dir / f"infogain_top{top_m}_predictions.jsonl"
    infogain_metrics = eval_dir / f"infogain_top{top_m}_metrics.json"
    cbwdm_predictions = eval_dir / "rag_cbwdm_predictions.jsonl"
    cbwdm_metrics = eval_dir / "rag_cbwdm_metrics.json"
    oracle_predictions = eval_dir / f"cbwdm_oracle_top{top_m}_predictions.jsonl"
    oracle_metrics = eval_dir / f"cbwdm_oracle_top{top_m}_metrics.json"
    fairness_audit = baseline_dir / "baseline_fairness_audit.json"
    summary_dir = baseline_dir / "summary"
    formal_config = config.get("formal_splits", {})
    formal_split_dir = absolute(
        formal_config.get(
            "output_dir", str(run_dir / "artifacts" / "formal_splits")
        )
    )
    formal_query = {
        role: formal_split_dir / f"{role}.jsonl"
        for role in ("train_core", "validation", "held_out_test")
    }
    formal_retrieval = {
        role: artifacts / "formal" / f"{dataset}_{role}_bm25_top{top_n}.jsonl"
        for role in ("train_core", "validation", "held_out_test")
    }
    formal_posterior = {
        role: artifacts / "formal" / f"{dataset}_{role}_posteriors.jsonl"
        for role in ("train_core", "validation", "held_out_test")
    }
    formal_calibration_candidates = (
        artifacts / "formal" / "calibration_candidates.json"
    )
    formal_calibration_grid_dir = artifacts / "formal" / "calibration_grid"
    formal_calibration_dir = artifacts / "formal" / "calibration"
    formal_fixed_dir = artifacts / "formal" / "fixed_baselines"
    formal_diagnostics_dir = artifacts / "formal" / "diagnostics"
    formal_limits = {
        role: config.get("profile_limits", {}).get(role)
        for role in ("train_core", "validation", "held_out_test")
    }
    split_limit_args = {
        split: (["--limit", str(limits[split])] if limits[split] is not None else [])
        for split in ("train", "dev")
    }
    downstream_limit_args = split_limit_args
    raw_limit_args = ["--raw-limit", str(limits["raw"])] if limits["raw"] is not None else []
    corpus_limit_args = (
        ["--limit", str(limits["corpus"])] if limits["corpus"] is not None else []
    )
    generator_args = ["--model-name", args.generator_model] if args.generator_model else []
    posterior_batch = args.posterior_batch_size or int(config.get("generator", {}).get("posterior_batch_size", 4))
    selector_batch = args.selector_batch_size or int(config.get("selector", {}).get("candidate_batch_size", 8))
    selector_model = args.selector_model or config.get("selector", {}).get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    bge_config = baseline_config.get("bge", {})
    infogain_config = baseline_config.get("infogain_fever", {})
    bge_model = args.bge_model or bge_config.get("model_name")
    infogain_model = args.infogain_model or infogain_config.get("model_name")
    manifest["baseline_config_snapshot"] = baseline_config
    manifest.setdefault("models", {}).update(
        {
            "bge": bge_model,
            "bge_revision": bge_config.get("revision"),
            "infogain": infogain_model,
            "infogain_revision": infogain_config.get("revision"),
        }
    )
    baseline_flags = lambda stage: (
        ["--overwrite"] if stage in overwritten else (["--resume"] if args.resume else [])
    )
    eval_flags = lambda stage: baseline_flags(stage)
    stage_commands: dict[str, list[list[str]]] = {
        "prepare_formal_splits": [[
            py,
            str(PROJECT_ROOT / "scripts/01a_build_fever_formal_splits.py"),
            "--official-train",
            str(absolute(formal_config.get("official_train", config["paths"]["raw_fever_train"]))),
            "--official-dev",
            str(absolute(formal_config.get("official_dev", config["paths"]["raw_fever_dev"]))),
            "--output-dir",
            str(formal_split_dir),
            "--seed",
            str(formal_config.get("seed", args.seed)),
            "--validation-size",
            str(formal_config.get("validation_size", 5000)),
            *(
                ["--train-limit", str(formal_limits["train_core"])]
                if formal_limits["train_core"] is not None
                else []
            ),
            *(
                ["--validation-limit", str(formal_limits["validation"])]
                if formal_limits["validation"] is not None
                else []
            ),
            *(
                ["--test-limit", str(formal_limits["held_out_test"])]
                if formal_limits["held_out_test"] is not None
                else []
            ),
            *(
                ["--overwrite"]
                if "prepare_formal_splits" in overwritten
                else (["--resume"] if args.resume else [])
            ),
        ]],
        **{
            f"retrieve_{'test' if role == 'held_out_test' else role}": [[
                py,
                str(PROJECT_ROOT / "scripts/02_retrieve_bm25.py"),
                "--config",
                str(config_path),
                "--split",
                role,
                "--queries",
                str(formal_query[role]),
                "--index",
                str(index_path),
                "--output",
                str(formal_retrieval[role]),
                *(
                    ["--overwrite"]
                    if f"retrieve_{'test' if role == 'held_out_test' else role}" in overwritten
                    else []
                ),
            ]]
            for role in ("train_core", "validation", "held_out_test")
        },
        **{
            f"posterior_{'test' if role == 'held_out_test' else role}": [[
                py,
                str(PROJECT_ROOT / "scripts/03_compute_label_posteriors.py"),
                "--config",
                str(config_path),
                "--split",
                role,
                "--retrieval",
                str(formal_retrieval[role]),
                "--output",
                str(formal_posterior[role]),
                "--batch-size",
                str(posterior_batch),
                *generator_args,
                *(
                    ["--overwrite"]
                    if f"posterior_{'test' if role == 'held_out_test' else role}" in overwritten
                    else (["--resume"] if args.resume else [])
                ),
            ]]
            for role in ("train_core", "validation", "held_out_test")
        },
        "calibrate_methods": [[
            py,
            str(PROJECT_ROOT / "scripts/15_calibrate_fever_methods.py"),
            "--config",
            str(config_path),
            "--split-manifest",
            str(formal_split_dir / "fever2_formal_splits.manifest.json"),
            "--validation-metrics",
            str(formal_calibration_candidates),
            "--output-dir",
            str(formal_calibration_dir),
            "--objective",
            str(config.get("calibration", {}).get("objective", "macro_f1")),
            "--artifact",
            f"validation_retrieval={formal_retrieval['validation']}",
            "--artifact",
            f"validation_posteriors={formal_posterior['validation']}",
            *(
                ["--overwrite"]
                if "calibrate_methods" in overwritten
                else (["--resume"] if args.resume else [])
            ),
        ]],
        "run_calibration_grid": [[
            py,
            str(PROJECT_ROOT / "scripts/15a_run_fever_calibration_grid.py"),
            "--config",
            str(config_path),
            "--split-manifest",
            str(formal_split_dir / "fever2_formal_splits.manifest.json"),
            "--train-retrieval",
            str(formal_retrieval["train_core"]),
            "--validation-retrieval",
            str(formal_retrieval["validation"]),
            "--train-posteriors",
            str(formal_posterior["train_core"]),
            "--validation-posteriors",
            str(formal_posterior["validation"]),
            "--output-dir",
            str(formal_calibration_grid_dir),
            "--methods",
            args.methods,
            "--seed",
            str(args.seed),
            "--selector-device",
            args.selector_device,
            "--infogain-device",
            args.infogain_device,
            *(
                ["--generator-model", args.generator_model]
                if args.generator_model
                else []
            ),
            *(
                ["--selector-model", args.selector_model]
                if args.selector_model
                else []
            ),
            *(
                ["--infogain-model", args.infogain_model]
                if args.infogain_model
                else []
            ),
            *(
                ["--candidate-limit", str(args.candidate_limit)]
                if args.candidate_limit is not None
                else []
            ),
            *(
                ["--candidate-fingerprint", args.candidate_fingerprint]
                if args.candidate_fingerprint
                else []
            ),
            *(
                [
                    "--max-training-candidates",
                    str(args.max_training_candidates),
                ]
                if args.max_training_candidates is not None
                else []
            ),
            *(["--skip-completed"] if args.skip_completed else []),
            *(["--fail-fast"] if args.fail_fast else []),
            *(["--continue-on-error"] if args.continue_on_error else []),
            *(["--resume"] if args.resume else []),
            *(["--dry-run"] if args.dry_run else []),
        ]],
        "cbwdm_diagnostics": [[
            py,
            str(PROJECT_ROOT / "scripts/16a_diagnose_cbwdm_pilot.py"),
            "--config",
            str(config_path),
            "--no-evidence",
            str(formal_fixed_dir / "no_evidence_predictions.jsonl"),
            "--naive",
            str(formal_fixed_dir / "naive_topm_predictions.jsonl"),
            "--oracle",
            str(formal_fixed_dir / "cbwdm_oracle_predictions.jsonl"),
            "--naive-selection",
            str(formal_fixed_dir / "naive_topm_selection.jsonl"),
            "--oracle-selection",
            str(formal_fixed_dir / "cbwdm_oracle_selection.jsonl"),
            "--retrieval",
            str(formal_retrieval["validation"]),
            "--posteriors",
            str(formal_posterior["validation"]),
            "--calibration-manifest",
            str(formal_calibration_dir / "calibration.manifest.json"),
            "--calibration-candidates",
            str(formal_calibration_candidates),
            "--output-dir",
            str(formal_diagnostics_dir),
        ]],
        "prepare": [
            [
                py,
                str(PROJECT_ROOT / "scripts/00_prepare_fever.py"),
                "--config",
                str(config_path),
                "--splits",
                split,
                "--output-dir",
                str(artifacts),
                *split_limit_args[split],
                *raw_limit_args,
            ]
            for split in ("train", "dev")
        ],
        "corpus": [[py, str(PROJECT_ROOT / "scripts/01_prepare_fever_corpus.py"), "--config", str(config_path), "--output", str(corpus), *(["--overwrite"] if "corpus" in overwritten else ["--resume"]), *corpus_limit_args]],
        "index": [[
            py,
            str(PROJECT_ROOT / "scripts/02a_build_bm25_index.py"),
            "--config",
            str(config_path),
            "--corpus",
            str(corpus),
            "--index",
            str(index_path),
            *(["--overwrite"] if "index" in overwritten else ["--resume"]),
        ]],
        "retrieve": [
            [py, str(PROJECT_ROOT / "scripts/02_retrieve_bm25.py"), "--config", str(config_path), "--split", split, "--queries", str(query[split]), "--index", str(index_path), "--output", str(retrieval[split]), "--overwrite", *split_limit_args[split]]
            for split in ("train", "dev")
        ],
        "posterior": [
            [py, str(PROJECT_ROOT / "scripts/03_compute_label_posteriors.py"), "--config", str(config_path), "--split", split, "--retrieval", str(retrieval[split]), "--output", str(posterior[split]), "--batch-size", str(posterior_batch), *(["--resume"] if args.resume and "posterior" not in overwritten and posterior[split].with_suffix(".manifest.json").exists() else []), *(["--overwrite"] if "posterior" in overwritten else []), *generator_args, *downstream_limit_args[split]]
            for split in ("train", "dev")
        ],
        "teacher": [
            [py, str(PROJECT_ROOT / "scripts/04_build_cbwdm_teacher.py"), "--config", str(config_path), "--split", split, "--posteriors", str(posterior[split]), "--output", str(teacher[split]), *baseline_flags("teacher"), *downstream_limit_args[split]]
            for split in ("train", "dev")
        ],
        "train_cross_encoder": [[py, str(PROJECT_ROOT / "scripts/10_train_cross_encoder_selector.py"), "--config", str(config_path), "--posteriors", str(posterior["train"]), "--teacher", str(teacher["train"]), "--retrieval", str(retrieval["train"]), "--output-dir", str(checkpoint), "--model-name", str(selector_model), "--batch-size", str(config.get("selector", {}).get("batch_size", 1)), "--seed", str(args.seed)]],
        "select_cross_encoder": [[py, str(PROJECT_ROOT / "scripts/11_select_with_cross_encoder.py"), "--config", str(config_path), "--posteriors", str(posterior["dev"]), "--checkpoint-dir", str(checkpoint / "checkpoint"), "--output", str(selection), "--batch-size", str(selector_batch), "--top-m", str(top_m), *baseline_flags("select_cross_encoder"), *downstream_limit_args["dev"]]],
        "eval": [[py, str(PROJECT_ROOT / "scripts/07_eval_rag_classification.py"), "--config", str(config_path), "--split", "dev", "--selection", str(selection), "--output", str(predictions), "--metrics-output", str(metrics), *eval_flags("eval"), *generator_args, *downstream_limit_args["dev"]]],
        "no_evidence": [[py, str(PROJECT_ROOT / "scripts/07_eval_rag_classification.py"), "--config", str(config_path), "--split", "dev", "--selection", str(query["dev"]), "--output", str(no_evidence_predictions), "--metrics-output", str(no_evidence_metrics), "--no-evidence", "--method-name", "no_evidence", *eval_flags("no_evidence"), *generator_args, *downstream_limit_args["dev"]]],
        "naive_topm": [[py, str(PROJECT_ROOT / "scripts/08_select_naive_topm.py"), "--config", str(config_path), "--retrieval", str(retrieval["dev"]), "--output", str(naive), "--top-m", str(top_m), "--min-docs", str(int(baseline_config.get("naive", {}).get("min_docs", common_min_docs))), "--method-name", "naive_topm", *baseline_flags("naive_topm"), *downstream_limit_args["dev"]]],
        "oracle_diagnostic": [[py, str(PROJECT_ROOT / "scripts/09_select_cbwdm_oracle_from_teacher.py"), "--config", str(config_path), "--teacher", str(teacher["dev"]), "--posteriors", str(posterior["dev"]), "--output", str(oracle), "--top-m", str(top_m), *baseline_flags("oracle_diagnostic"), *downstream_limit_args["dev"]]],
        "score_bge": [[py, str(PROJECT_ROOT / "scripts/12_select_bge_reranker.py"), "--retrieval", str(retrieval["dev"]), "--output", str(bge_selection), "--score-cache", str(bge_score_cache), "--model-name-or-path", str(bge_model), "--device", args.bge_device, "--dtype", str(bge_config.get("dtype", "auto")), "--batch-size", str(bge_config.get("batch_size", 8)), "--max-length", str(bge_config.get("max_length", 512)), "--top-m", str(top_m), "--min-docs", str(bge_config.get("min_docs", top_m)), "--score-only", *(["--revision", str(bge_config["revision"])] if bge_config.get("revision") else []), *(["--local-files-only"] if bge_config.get("local_files_only") else []), *baseline_flags("score_bge"), *downstream_limit_args["dev"]]],
        "select_bge": [[py, str(PROJECT_ROOT / "scripts/12_select_bge_reranker.py"), "--retrieval", str(retrieval["dev"]), "--output", str(bge_selection), "--score-cache", str(bge_score_cache), "--model-name-or-path", str(bge_model), "--device", args.bge_device, "--dtype", str(bge_config.get("dtype", "auto")), "--batch-size", str(bge_config.get("batch_size", 8)), "--max-length", str(bge_config.get("max_length", 512)), "--top-m", str(top_m), "--min-docs", str(bge_config.get("min_docs", top_m)), *(["--revision", str(bge_config["revision"])] if bge_config.get("revision") else []), *(["--local-files-only"] if bge_config.get("local_files_only") else []), *(["--normalize-score"] if bge_config.get("normalize_score") else []), *(["--score-threshold", str(bge_config["threshold"])] if bge_config.get("threshold") is not None else []), *baseline_flags("select_bge"), *downstream_limit_args["dev"]]],
        "build_infogain_teacher": [[py, str(PROJECT_ROOT / "scripts/12a_build_infogain_teacher.py"), "--posteriors", str(posterior["train"]), "--output", str(infogain_teacher), "--threshold-mode", str(infogain_config.get("threshold_mode", "train_quantile")), "--positive-quantile", str(infogain_config.get("positive_quantile", 0.75)), "--negative-quantile", str(infogain_config.get("negative_quantile", 0.25)), "--generator-model", str(args.generator_model or config["generator"]["model_name"]), "--prompt-hash", fever_prompt_hash(list(config["task"]["labels"]), dict(config["task"]["verbalizers"])), "--verbalizer-hash", stable_hash(config["task"]["verbalizers"]), *baseline_flags("build_infogain_teacher"), *downstream_limit_args["train"]]],
        "train_infogain": [[py, str(PROJECT_ROOT / "scripts/12b_train_infogain_reranker.py"), "--teacher", str(infogain_teacher), "--output-dir", str(infogain_train_dir), "--model-name-or-path", str(infogain_model), "--device", args.infogain_device, "--max-length", str(infogain_config.get("max_length", 512)), "--epochs", str(infogain_config.get("epochs", 1)), "--lr", str(infogain_config.get("lr", 2e-5)), "--beta", str(infogain_config.get("beta", 0.75)), "--seed", str(args.seed), *(["--revision", str(infogain_config["revision"])] if infogain_config.get("revision") else []), *baseline_flags("train_infogain")]],
        "select_infogain": [[py, str(PROJECT_ROOT / "scripts/12c_select_infogain_reranker.py"), "--retrieval", str(retrieval["dev"]), "--checkpoint-dir", str(infogain_train_dir / "checkpoint"), "--output", str(infogain_selection), "--device", args.infogain_device, "--batch-size", str(infogain_config.get("candidate_batch_size", 16)), "--top-m", str(top_m), "--min-docs", str(infogain_config.get("min_docs", common_min_docs)), *(["--filter-threshold", str(infogain_config["inference_threshold"])] if infogain_config.get("inference_threshold") is not None else []), *baseline_flags("select_infogain"), *downstream_limit_args["dev"]]],
        "eval_naive_topm": [[py, str(PROJECT_ROOT / "scripts/07_eval_rag_classification.py"), "--config", str(config_path), "--split", "dev", "--selection", str(naive), "--output", str(naive_predictions), "--metrics-output", str(naive_metrics), "--method-name", "naive_topm", *eval_flags("eval_naive_topm"), *generator_args, *downstream_limit_args["dev"]]],
        "eval_bge": [[py, str(PROJECT_ROOT / "scripts/07_eval_rag_classification.py"), "--config", str(config_path), "--split", "dev", "--selection", str(bge_selection), "--output", str(bge_predictions), "--metrics-output", str(bge_metrics), "--method-name", "bge", *eval_flags("eval_bge"), *generator_args, *downstream_limit_args["dev"]]],
        "eval_infogain": [[py, str(PROJECT_ROOT / "scripts/07_eval_rag_classification.py"), "--config", str(config_path), "--split", "dev", "--selection", str(infogain_selection), "--output", str(infogain_predictions), "--metrics-output", str(infogain_metrics), "--method-name", "infogain_fever", *eval_flags("eval_infogain"), *generator_args, *downstream_limit_args["dev"]]],
        "eval_cbwdm": [[py, str(PROJECT_ROOT / "scripts/07_eval_rag_classification.py"), "--config", str(config_path), "--split", "dev", "--selection", str(selection), "--output", str(cbwdm_predictions), "--metrics-output", str(cbwdm_metrics), "--method-name", "rag_cbwdm", *eval_flags("eval_cbwdm"), *generator_args, *downstream_limit_args["dev"]]],
        "eval_oracle": [[py, str(PROJECT_ROOT / "scripts/07_eval_rag_classification.py"), "--config", str(config_path), "--split", "dev", "--selection", str(oracle), "--output", str(oracle_predictions), "--metrics-output", str(oracle_metrics), "--method-name", "cbwdm_oracle", *eval_flags("eval_oracle"), *generator_args, *downstream_limit_args["dev"]]],
        "fairness_audit": [[py, str(PROJECT_ROOT / "scripts/14_audit_fever_baselines.py"), "--retrieval", str(retrieval["dev"]), "--selection", f"naive_topm={naive}", "--selection", f"bge={bge_selection}", "--selection", f"infogain_fever={infogain_selection}", "--selection", f"rag_cbwdm={selection}", "--selection", f"cbwdm_oracle={oracle}", "--evaluation-manifest", f"no_evidence={no_evidence_metrics.with_suffix('.manifest.json')}", "--evaluation-manifest", f"naive_topm={naive_metrics.with_suffix('.manifest.json')}", "--evaluation-manifest", f"bge={bge_metrics.with_suffix('.manifest.json')}", "--evaluation-manifest", f"infogain_fever={infogain_metrics.with_suffix('.manifest.json')}", "--evaluation-manifest", f"rag_cbwdm={cbwdm_metrics.with_suffix('.manifest.json')}", "--evaluation-manifest", f"cbwdm_oracle={oracle_metrics.with_suffix('.manifest.json')}", "--expected-top-m", str(top_m), "--output", str(fairness_audit)]],
        "summarize_baselines": [[
            py,
            str(PROJECT_ROOT / "scripts/13_summarize_fever_baselines.py"),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(summary_dir),
            "--fairness-audit",
            str(fairness_audit),
            "--evaluation-manifest",
            f"no_evidence={no_evidence_metrics.with_suffix('.manifest.json')}",
            "--evaluation-manifest",
            f"naive_topm={naive_metrics.with_suffix('.manifest.json')}",
            "--evaluation-manifest",
            f"bge={bge_metrics.with_suffix('.manifest.json')}",
            "--evaluation-manifest",
            f"infogain_fever={infogain_metrics.with_suffix('.manifest.json')}",
            "--evaluation-manifest",
            f"rag_cbwdm={cbwdm_metrics.with_suffix('.manifest.json')}",
            "--evaluation-manifest",
            f"cbwdm_oracle={oracle_metrics.with_suffix('.manifest.json')}",
        ]],
    }
    manifest["paths"] = {
        "run_dir": str(run_dir), "artifacts": str(artifacts), "logs": str(logs_dir),
        "commands": str(commands_dir), "cache_root": str(absolute(args.cache_root)),
        "processed_dir": str(processed),
        "corpus": str(corpus),
        "corpus_manifest": str(corpus.with_suffix(".manifest.json")),
        "index_path": str(index_path),
        "index_manifest": str(index_path / "index_manifest.json"),
        "index_fingerprint": index_key,
        "queries": {key: str(value) for key, value in query.items()},
        "retrieval": {key: str(value) for key, value in retrieval.items()},
        "posteriors": {key: str(value) for key, value in posterior.items()},
        "teachers": {key: str(value) for key, value in teacher.items()},
        "selector_checkpoint": str(checkpoint / "checkpoint"),
        "selection": str(selection),
        "predictions": str(predictions),
        "metrics": str(metrics),
        "naive_selection": str(naive),
        "oracle_selection": str(oracle),
        "bge_selection": str(bge_selection),
        "bge_score_cache": str(bge_score_cache),
        "infogain_teacher": str(infogain_teacher),
        "infogain_checkpoint": str(infogain_train_dir / "checkpoint"),
        "infogain_selection": str(infogain_selection),
        "baseline_eval_dir": str(eval_dir),
        "fairness_audit": str(fairness_audit),
        "baseline_summary": str(summary_dir / "baseline_summary.json"),
        "formal_split_manifest": str(
            formal_split_dir / "fever2_formal_splits.manifest.json"
        ),
        "formal_queries": {key: str(value) for key, value in formal_query.items()},
        "formal_retrieval": {
            key: str(value) for key, value in formal_retrieval.items()
        },
        "formal_posteriors": {
            key: str(value) for key, value in formal_posterior.items()
        },
        "formal_calibration_candidates": str(formal_calibration_candidates),
        "formal_calibration_grid_dir": str(formal_calibration_grid_dir),
        "formal_calibration_manifest": str(
            formal_calibration_dir / "calibration.manifest.json"
        ),
        "formal_fixed_baselines_dir": str(formal_fixed_dir),
        "formal_cbwdm_diagnostics": str(
            formal_diagnostics_dir / "cbwdm_pilot_diagnostics.json"
        ),
    }
    stage_outputs: dict[str, list[Path]] = {
        "prepare_formal_splits": [
            formal_query["train_core"],
            formal_query["validation"],
            formal_query["held_out_test"],
            formal_split_dir / "fever2_formal_splits.manifest.json",
        ],
        **{
            f"retrieve_{'test' if role == 'held_out_test' else role}": [
                formal_retrieval[role],
                formal_retrieval[role].with_suffix(".manifest.json"),
            ]
            for role in ("train_core", "validation", "held_out_test")
        },
        **{
            f"posterior_{'test' if role == 'held_out_test' else role}": [
                formal_posterior[role],
                formal_posterior[role].with_suffix(".manifest.json"),
            ]
            for role in ("train_core", "validation", "held_out_test")
        },
        "calibrate_methods": [
            formal_calibration_dir / "calibration_results.json",
            formal_calibration_dir / "calibration_results.csv",
            formal_calibration_dir / "calibration_report.md",
            formal_calibration_dir / "frozen_parameters.yaml",
            formal_calibration_dir / "calibration.manifest.json",
        ],
        "run_calibration_grid": [
            artifacts / "formal" / "calibration_candidates.json",
            artifacts / "formal" / "calibration_candidates.csv",
            artifacts / "formal" / "calibration_grid_manifest.json",
            artifacts / "formal" / "calibration_grid_report.md",
        ],
        "cbwdm_diagnostics": [
            formal_diagnostics_dir / "cbwdm_pilot_diagnostics.json",
            formal_diagnostics_dir / "cbwdm_pilot_diagnostics.md",
        ],
        "prepare": [
            query["train"], query["train"].with_suffix(".manifest.json"),
            query["dev"], query["dev"].with_suffix(".manifest.json"),
        ],
        "corpus": [corpus, corpus.with_suffix(".manifest.json")],
        "index": [index_path / "index_manifest.json"],
        "retrieve": [
            retrieval["train"], retrieval["train"].with_suffix(".manifest.json"),
            retrieval["dev"], retrieval["dev"].with_suffix(".manifest.json"),
        ],
        "posterior": [
            posterior["train"],
            posterior["train"].with_suffix(".manifest.json"),
            posterior["dev"],
            posterior["dev"].with_suffix(".manifest.json"),
        ],
        "teacher": [teacher["train"], teacher["dev"]],
        "train_cross_encoder": [
            checkpoint / "checkpoint" / "config.json",
            checkpoint / "training_config.json",
        ],
        "select_cross_encoder": [selection, selection.with_suffix(".manifest.json")],
        "eval": [predictions, metrics, metrics.with_suffix(".manifest.json")],
        "no_evidence": [no_evidence_predictions, no_evidence_metrics, no_evidence_metrics.with_suffix(".manifest.json")],
        "naive_topm": [naive, naive.with_suffix(".manifest.json")],
        "oracle_diagnostic": [oracle, oracle.with_suffix(".manifest.json")],
        "score_bge": [bge_score_cache, bge_score_cache.with_suffix(".manifest.json")],
        "select_bge": [bge_selection, bge_selection.with_suffix(".manifest.json")],
        "build_infogain_teacher": [infogain_teacher, infogain_teacher.with_suffix(".manifest.json")],
        "train_infogain": [
            infogain_train_dir / "checkpoint" / "infogain_config.json",
            infogain_train_dir / "checkpoint" / "heads.pt",
            infogain_train_dir / "training_manifest.json",
        ],
        "select_infogain": [infogain_selection, infogain_selection.with_suffix(".manifest.json")],
        "eval_naive_topm": [naive_predictions, naive_metrics, naive_metrics.with_suffix(".manifest.json")],
        "eval_bge": [bge_predictions, bge_metrics, bge_metrics.with_suffix(".manifest.json")],
        "eval_infogain": [infogain_predictions, infogain_metrics, infogain_metrics.with_suffix(".manifest.json")],
        "eval_cbwdm": [cbwdm_predictions, cbwdm_metrics, cbwdm_metrics.with_suffix(".manifest.json")],
        "eval_oracle": [oracle_predictions, oracle_metrics, oracle_metrics.with_suffix(".manifest.json")],
        "fairness_audit": [fairness_audit],
        "summarize_baselines": [
            summary_dir / "baseline_summary.json",
            summary_dir / "baseline_summary.csv",
            summary_dir / "baseline_summary.md",
        ],
    }
    validation_context: dict[str, Any] = {
        "query": query,
        "retrieval": retrieval,
        "posterior": posterior,
        "teacher": teacher,
        "selection": selection,
        "baseline_summary_dir": summary_dir,
        "formal_query": formal_query,
        "formal_retrieval": formal_retrieval,
        "formal_posterior": formal_posterior,
        "posterior_provenance": lambda split: posterior_provenance(
            config=config,
            config_path=config_path,
            retrieval_path=retrieval[split],
            split=split,
            model_name=args.generator_model
            or str(config.get("generator", {}).get("model_name")),
            batch_size=posterior_batch,
        ),
        "formal_posterior_provenance": lambda role: posterior_provenance(
            config=config,
            config_path=config_path,
            retrieval_path=formal_retrieval[role],
            split=role,
            model_name=args.generator_model
            or str(config.get("generator", {}).get("model_name")),
            batch_size=posterior_batch,
        ),
    }
    if args.dry_run:
        for stage in requested:
            for command in stage_commands[stage]:
                print(f"[dry-run][{stage}] {command_text(command)}")
                if stage == "run_calibration_grid":
                    completed = subprocess.run(
                        command,
                        cwd=PROJECT_ROOT,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    if completed.stdout:
                        print(completed.stdout, end="")
                    if completed.returncode:
                        if completed.stderr:
                            print(completed.stderr, file=sys.stderr, end="")
                        raise subprocess.CalledProcessError(
                            completed.returncode, command
                        )
        return
    for directory in (logs_dir, commands_dir, artifacts, absolute(args.cache_root)):
        directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest_path, manifest)
    env = os.environ.copy()
    env["HF_HOME"] = str(absolute(args.cache_root))
    for stage in requested:
        state = manifest["stages"][stage]
        reuse_candidate = state.get("status") in {"completed", "skipped"}
        script_validated_resume_stages = {
            "no_evidence",
            "naive_topm",
            "oracle_diagnostic",
            "score_bge",
            "select_bge",
            "build_infogain_teacher",
            "train_infogain",
            "select_infogain",
            "eval_naive_topm",
            "eval_bge",
            "eval_infogain",
            "eval_cbwdm",
            "eval_oracle",
            "fairness_audit",
            "summarize_baselines",
        }
        if args.resume and stage in script_validated_resume_stages and stage not in overwritten:
            # These entry points validate their own method-specific fingerprint
            # before loading a model. Invoke the cheap validator instead of
            # trusting only the runner's prior stage status.
            reuse_candidate = False
        if stage == "posterior" and args.resume and stage not in overwritten:
            posterior_statuses = []
            for split in ("train", "dev"):
                payload, _ = load_json_object(
                    posterior[split].with_suffix(".manifest.json")
                )
                posterior_statuses.append((payload or {}).get("status"))
            if all(status == "completed" for status in posterior_statuses):
                reuse_candidate = True
            elif all(path.exists() for path in stage_outputs["posterior"]):
                # A complete-looking set with non-completed manifests is invalid,
                # not a reason to silently rerun the generator.
                reuse_candidate = True
        if reuse_candidate and args.resume and stage not in overwritten:
            if stage == "index":
                bm25_config = retrieval_config.get("bm25", {})
                requested_contract = index_contract(
                    corpus,
                    backend_version=pyserini_version(),
                    k1=float(bm25_config.get("k1", 0.9)),
                    b=float(bm25_config.get("b", 0.4)),
                    analyzer=str(
                        retrieval_config.get("index", {}).get("analyzer", "english")
                    ),
                )
                validate_index(index_path, requested_contract)
            try:
                require_valid_stage_outputs(
                    stage, stage_outputs[stage], validation_context
                )
            except RuntimeError as exc:
                state.update(
                    {
                        "status": "failed",
                        "end_time": utc_now(),
                        "exit_code": 1,
                        "error": str(exc),
                    }
                )
                atomic_write_json(manifest_path, manifest)
                raise
            state.update(
                {
                    "status": "skipped",
                    "reason": "validated_completed_outputs",
                    "updated_at": utc_now(),
                }
            )
            atomic_write_json(manifest_path, manifest)
            continue
        commands = stage_commands[stage]
        command_path = commands_dir / f"{stage}.txt"
        command_path.write_text("\n".join(command_text(command) for command in commands) + "\n", encoding="utf-8")
        log_path = logs_dir / f"{stage}.log"
        state.update({"status": "running", "start_time": utc_now(), "log_path": str(log_path), "command_path": str(command_path)})
        atomic_write_json(manifest_path, manifest)
        try:
            with log_path.open("a", encoding="utf-8", newline="\n") as log:
                for command in commands:
                    log.write(f"$ {command_text(command)}\n")
                    log.flush()
                    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    if completed.returncode:
                        raise subprocess.CalledProcessError(completed.returncode, command)
            require_valid_stage_outputs(
                stage, stage_outputs[stage], validation_context
            )
            state.update({"status": "completed", "end_time": utc_now(), "exit_code": 0})
            if stage == "index":
                index_manifest = json.loads(
                    (index_path / "index_manifest.json").read_text(encoding="utf-8")
                )
                manifest["paths"]["index_fingerprint"] = index_manifest["fingerprint"]
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            exit_code = exc.returncode if isinstance(exc, subprocess.CalledProcessError) else 1
            state.update({"status": "failed", "end_time": utc_now(), "exit_code": exit_code})
            manifest["end_time"] = utc_now()
            atomic_write_json(manifest_path, manifest)
            raise
        atomic_write_json(manifest_path, manifest)
    manifest["end_time"] = utc_now()
    atomic_write_json(manifest_path, manifest)
    print(f"[runner] completed run={args.run_name} manifest={manifest_path}")


if __name__ == "__main__":
    main()

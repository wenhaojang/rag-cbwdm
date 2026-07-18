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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Linux/server RAG-CBWDM pipeline.")
    parser.add_argument("--config", required=True)
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
    parser.add_argument("--posterior-batch-size", type=int)
    parser.add_argument("--selector-batch-size", type=int)
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
    if stage in {"prepare", "corpus", "retrieve"}:
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
    elif stage in {"train_infogain", "fairness_audit", "summarize_baselines"}:
        for path in stage_outputs:
            if path.suffix == ".json":
                payload, error = load_json_object(path)
                if error:
                    invalid.append(f"{path}: {error}")
                elif stage == "fairness_audit" and payload and payload.get("status") != "comparable":
                    invalid.append(f"{path}: status={payload.get('status')!r}")
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
        requested.extend(BASELINE_SUITE if value == "baselines" else [ALIASES.get(value, value)])
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
            "splits": ["train", "dev"],
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
            [py, str(PROJECT_ROOT / "scripts/04_build_cbwdm_teacher.py"), "--config", str(config_path), "--split", split, "--posteriors", str(posterior[split]), "--output", str(teacher[split]), *downstream_limit_args[split]]
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
        "summarize_baselines": [[py, str(PROJECT_ROOT / "scripts/13_summarize_fever_baselines.py"), "--run-dir", str(run_dir), "--output-dir", str(summary_dir), "--fairness-audit", str(fairness_audit)]],
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
    }
    stage_outputs: dict[str, list[Path]] = {
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
        "posterior_provenance": lambda split: posterior_provenance(
            config=config,
            config_path=config_path,
            retrieval_path=retrieval[split],
            split=split,
            model_name=args.generator_model
            or str(config.get("generator", {}).get("model_name")),
            batch_size=posterior_batch,
        ),
    }
    if args.dry_run:
        for stage in requested:
            for command in stage_commands[stage]:
                print(f"[dry-run][{stage}] {command_text(command)}")
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

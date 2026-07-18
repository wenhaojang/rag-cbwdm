from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import load_yaml, read_jsonl, require_keys
from src.label_logits import LabelLogitScorer
from src.prompts import FEVER_PROMPT_VERSION, build_fever_prompt, fever_prompt_hash
from src.run_manifest import (
    atomic_write_json,
    environment_info,
    git_state,
    sha256_file,
    stable_hash,
    utc_now,
    validate_resume_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute batched, resumable fixed-generator label posteriors."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--split",
        required=True,
        choices=[
            "train",
            "dev",
            "test",
            "train_core",
            "validation",
            "held_out_test",
        ],
    )
    parser.add_argument("--retrieval")
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model-name")
    parser.add_argument("--model-revision")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_retrieval_path(config: dict[str, Any], split: str) -> Path:
    processed = resolve_project_path(config["paths"]["processed_dir"])
    top_n = int(config.get("retrieval", {}).get("top_n", 20))
    return processed / f"{config['dataset']}_{split}_bm25_top{top_n}.jsonl"


def default_output_path(config: dict[str, Any], split: str) -> Path:
    return PROJECT_ROOT / "outputs" / "posteriors" / f"{config['dataset']}_{split}_posteriors.jsonl"


def validate_posterior(posterior: np.ndarray, where: str) -> list[float]:
    vector = np.asarray(posterior, dtype=np.float32)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise FloatingPointError(f"{where} is not a finite 1D posterior")
    if np.any(vector < 0) or not np.isclose(vector.sum(), 1.0, atol=1e-5):
        raise FloatingPointError(f"{where} is not normalized: {vector.tolist()}")
    return [float(value) for value in vector]


def score_retrieval_row(
    row: dict[str, Any],
    scorer: LabelLogitScorer,
    labels: list[str],
    verbalizers: dict[str, list[str]],
    batch_size: int,
    max_candidates: int | None,
) -> tuple[dict[str, Any], int]:
    """Score query-only once and all candidate prompts in stable input order."""
    require_keys(row, ["id", "query", "label", "split", "candidates"], "retrieval row")
    candidates = list(row["candidates"])
    if max_candidates is not None:
        if max_candidates < 0:
            raise ValueError("--max-candidates must be non-negative")
        candidates = candidates[:max_candidates]
    prompts = [
        build_fever_prompt(row["query"], labels, verbalizers, evidence=None),
        *[
            build_fever_prompt(
                row["query"], labels, verbalizers, evidence=candidate.get("text", "")
            )
            for candidate in candidates
        ],
    ]
    posteriors = scorer.score_prompts(
        prompts, batch_size=batch_size, labels=labels, verbalizers=verbalizers
    )
    if posteriors.shape != (len(prompts), len(labels)):
        raise ValueError(f"Unexpected posterior shape {posteriors.shape}")
    scored_candidates = []
    for idx, candidate in enumerate(candidates):
        require_keys(candidate, ["doc_id", "rank", "title", "text"], "candidate")
        scored_candidates.append(
            {
                "doc_id": candidate["doc_id"],
                "rank": candidate["rank"],
                "retrieval_score": candidate.get("score", candidate.get("retrieval_score")),
                "title": candidate["title"],
                "text": candidate["text"],
                "eta": validate_posterior(posteriors[idx + 1], f"candidate[{idx}].eta"),
            }
        )
    return (
        {
            "schema_version": "rag_cbwdm_posteriors.v2",
            "id": row["id"],
            "query": row["query"],
            "label": row["label"],
            "split": row["split"],
            "labels": labels,
            "eta0": validate_posterior(posteriors[0], "eta0"),
            "candidates": scored_candidates,
        },
        len(candidates),
    )


def completed_ids(partial_path: Path) -> set[str]:
    ids: set[str] = set()
    if not partial_path.exists():
        return ids
    for row in read_jsonl(partial_path):
        sample_id = str(row.get("id"))
        if sample_id in ids:
            raise ValueError(f"Duplicate id {sample_id!r} in partial posterior output")
        ids.add(sample_id)
    return ids


def partial_candidate_count(partial_path: Path) -> int:
    if not partial_path.exists():
        return 0
    return sum(len(row.get("candidates", [])) for row in read_jsonl(partial_path))


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    config_path = resolve_project_path(args.config)
    config = load_yaml(config_path)
    require_keys(config, ["dataset", "paths", "task", "generator"], "config")
    require_keys(config["task"], ["labels", "verbalizers"], "config.task")
    generator = config["generator"]
    model_name = args.model_name or generator["model_name"]
    revision = args.model_revision or generator.get("revision")
    tokenizer_revision = args.tokenizer_revision or generator.get("tokenizer_revision") or revision
    batch_size = args.batch_size or int(generator.get("posterior_batch_size", 4))
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    labels = list(config["task"]["labels"])
    verbalizers = dict(config["task"]["verbalizers"])
    retrieval_path = (
        resolve_project_path(args.retrieval)
        if args.retrieval
        else default_retrieval_path(config, args.split)
    )
    output_path = (
        resolve_project_path(args.output)
        if args.output
        else default_output_path(config, args.split)
    )
    partial_path = output_path.with_name(output_path.name + ".partial")
    manifest_path = output_path.with_suffix(".manifest.json")
    retrieval_rows = list(read_jsonl(retrieval_path, limit=args.limit))
    expected_ids = [str(row.get("id")) for row in retrieval_rows]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Retrieval input contains duplicate query ids")

    provenance = {
        "dataset": config["dataset"],
        "split": args.split,
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
        "max_candidates": args.max_candidates,
        "input_path": str(retrieval_path.resolve()),
        "input_sha256": sha256_file(retrieval_path),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
    }
    fingerprint = stable_hash(provenance)
    existing_manifest: dict[str, Any] | None = None
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if output_path.exists() and not args.overwrite:
        if (
            args.resume
            and existing_manifest
            and existing_manifest.get("status") == "completed"
        ):
            validate_resume_manifest(existing_manifest, fingerprint, stage="posterior")
            expected_output_hash = existing_manifest.get("output_sha256")
            if expected_output_hash and sha256_file(output_path) != expected_output_hash:
                raise ValueError(
                    "Completed posterior output hash does not match its manifest"
                )
            print(f"[posteriors] already completed: {output_path}")
            return
        raise FileExistsError(f"Output exists: {output_path}. Use --overwrite explicitly.")
    if args.overwrite:
        for path in (output_path, partial_path, manifest_path):
            if path.exists():
                path.unlink()
        existing_manifest = None
    if partial_path.exists() and not args.resume:
        raise FileExistsError(f"Partial output exists: {partial_path}. Use --resume or --overwrite.")
    if args.resume:
        if not existing_manifest:
            raise FileNotFoundError(f"Resume requires manifest: {manifest_path}")
        validate_resume_manifest(existing_manifest, fingerprint, stage="posterior")

    done = completed_ids(partial_path)
    unknown = done - set(expected_ids)
    if unknown:
        raise ValueError(f"Partial output contains ids absent from current input: {sorted(unknown)[:3]}")
    manifest = {
        "schema_version": "rag_cbwdm_posterior_manifest.v1",
        "stage": "posterior",
        "status": "running",
        "fingerprint": fingerprint,
        "provenance": provenance,
        "git": git_state(PROJECT_ROOT),
        "environment": environment_info(),
        "created_at": (existing_manifest or {}).get("created_at", utc_now()),
        "updated_at": utc_now(),
        "expected_rows": len(retrieval_rows),
        "completed_rows": len(done),
        "num_candidate_prompts": partial_candidate_count(partial_path),
        "partial_path": str(partial_path),
        "output_path": str(output_path),
    }
    atomic_write_json(manifest_path, manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_count = int(manifest["num_candidate_prompts"])
    try:
        scorer = LabelLogitScorer(
            model_name=model_name,
            dtype=generator.get("dtype", "auto"),
            device_map=generator.get("device_map", "auto"),
            trust_remote_code=bool(generator.get("trust_remote_code", False)),
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            max_length=generator.get("max_context_tokens"),
        )
        manifest["provenance"].update(
            {
                "resolved_model_commit": getattr(
                    getattr(scorer.model, "config", None), "_commit_hash", None
                ),
                "resolved_tokenizer_commit": (
                    getattr(scorer.tokenizer, "init_kwargs", {}) or {}
                ).get("_commit_hash"),
            }
        )
        manifest["updated_at"] = utc_now()
        atomic_write_json(manifest_path, manifest)
        with partial_path.open("a", encoding="utf-8", newline="\n") as handle:
            for index, row in enumerate(retrieval_rows, start=1):
                if str(row.get("id")) in done:
                    continue
                posterior_row, row_candidate_count = score_retrieval_row(
                    row,
                    scorer=scorer,
                    labels=labels,
                    verbalizers=verbalizers,
                    batch_size=batch_size,
                    max_candidates=args.max_candidates,
                )
                handle.write(json.dumps(posterior_row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                done.add(str(row["id"]))
                candidate_count += row_candidate_count
                manifest.update(
                    {
                        "completed_rows": len(done),
                        "num_candidate_prompts": candidate_count,
                        "updated_at": utc_now(),
                    }
                )
                atomic_write_json(manifest_path, manifest)
                if index % 10 == 0:
                    print(f"[posteriors] completed={len(done)}/{len(retrieval_rows)}", file=sys.stderr)
        if len(done) != len(retrieval_rows):
            raise RuntimeError("Posterior output row-count validation failed")
        os.replace(partial_path, output_path)
        manifest.update(
            {
                "status": "completed",
                "completed_at": utc_now(),
                "updated_at": utc_now(),
                "output_sha256": sha256_file(output_path),
            }
        )
        atomic_write_json(manifest_path, manifest)
    except BaseException as exc:
        manifest.update(
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "updated_at": utc_now()}
        )
        atomic_write_json(manifest_path, manifest)
        raise
    print(
        f"[posteriors][{config['dataset']}][{args.split}] rows={len(done)} "
        f"candidate_prompts={candidate_count} output={output_path}"
    )


if __name__ == "__main__":
    main()

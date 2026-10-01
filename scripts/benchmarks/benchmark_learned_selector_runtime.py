from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_manifest import atomic_write_json, git_state, utc_now
from src.baselines.infogain import group_teacher_rows
from src.diagnostics.signed_teacher_v1 import build_signed_training_groups
from src.io_utils import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark learned-selector workers outside formal artifact roots."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--infogain-teacher", required=True)
    parser.add_argument("--infogain-teacher-manifest", required=True)
    parser.add_argument("--infogain-model", required=True)
    parser.add_argument("--signed-teacher", required=True)
    parser.add_argument("--signed-teacher-manifest", required=True)
    parser.add_argument("--posteriors", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--signed-model", required=True)
    parser.add_argument("--dataset-id")
    parser.add_argument("--generator-id")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--group-counts", type=int, nargs="+", default=[1000, 5000])
    parser.add_argument(
        "--forward-batch-sizes", type=int, nargs="+", default=[16, 32, 64, 128]
    )
    parser.add_argument("--seed", type=int, default=13, choices=[13])
    parser.add_argument("--formal-v2-identity", action="store_true")
    return parser.parse_args()


def add_identity_flags(command: list[str], args: argparse.Namespace) -> None:
    if args.formal_v2_identity:
        command.append("--formal-v2-identity")
        if args.dataset_id:
            command.extend(("--dataset-id", args.dataset_id))
        if args.generator_id:
            command.extend(("--generator-id", args.generator_id))


def execute_case(name: str, command: list[str], output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True)
    wall = time.perf_counter() - started
    combined = f"{completed.stdout}\n{completed.stderr}"
    status = "completed" if completed.returncode == 0 else "failed"
    if completed.returncode != 0 and "out of memory" in combined.casefold():
        status = "oom"
    manifest_path = output_dir / "training_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else None
    )
    return {
        "name": name,
        "status": status,
        "return_code": completed.returncode,
        "wall_clock_seconds_external": wall,
        "command": command,
        "runtime_metrics": (manifest or {}).get("runtime_metrics"),
        "fingerprint": (manifest or {}).get("fingerprint"),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def input_statistics(args: argparse.Namespace) -> dict[str, Any]:
    info_groups = group_teacher_rows(list(read_jsonl(args.infogain_teacher)))
    info_lengths = Counter(len(group) for group in info_groups)
    pair_counts: Counter[int] = Counter()
    for group in info_groups:
        digs = [float(row["dig"]) for row in group]
        valid = sum(
            digs[left] != digs[right]
            for left in range(len(digs))
            for right in range(left + 1, len(digs))
        )
        pair_counts[valid] += 1
    signed_groups = build_signed_training_groups(
        teacher_path=args.signed_teacher,
        posteriors_path=args.posteriors,
        retrieval_path=args.retrieval,
    )
    signed_lengths = Counter(len(group.candidate_docs) for group in signed_groups)
    return {
        "infogain": {
            "groups": len(info_groups),
            "candidate_count_distribution": dict(sorted(info_lengths.items())),
            "valid_pair_count_distribution": dict(sorted(pair_counts.items())),
            "forward_calls_per_epoch_legacy": len(info_groups),
            "tokenizer_calls_per_epoch_legacy": len(info_groups),
            "optimizer_steps_per_epoch": len(info_groups),
        },
        "signed_v1": {
            "groups": len(signed_groups),
            "candidate_count_distribution": dict(sorted(signed_lengths.items())),
            "forward_calls_per_epoch_legacy": len(signed_groups),
            "tokenizer_calls_per_epoch_legacy": len(signed_groups),
            "optimizer_steps_per_epoch": (
                len(signed_groups) + 7
            ) // 8,
            "optimizer_group_batch_size": 8,
            "incomplete_block_divisor": 8,
        },
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []
    for count in args.group_counts:
        if count < 1:
            raise ValueError("group counts must be positive")
        for implementation in ("legacy", "vectorized"):
            output = output_root / "infogain" / f"groups{count}" / implementation
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts/12b_train_infogain_reranker.py"),
                "--teacher", args.infogain_teacher,
                "--teacher-manifest", args.infogain_teacher_manifest,
                "--config", args.config,
                "--output-dir", str(output),
                "--model-name-or-path", args.infogain_model,
                "--device", args.device,
                "--seed", str(args.seed),
                "--max-groups", str(count),
                "--rank-loss-implementation", implementation,
            ]
            add_identity_flags(command, args)
            results.append(
                execute_case(f"infogain.{count}.{implementation}", command, output)
            )
        for implementation, sizes in (
            ("legacy", [None]),
            ("block_v1", list(args.forward_batch_sizes)),
        ):
            for size in sizes:
                suffix = implementation if size is None else f"{implementation}.fb{size}"
                output = output_root / "signed_v1" / f"groups{count}" / suffix
                command = [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts/preformal/26_train_signed_v1.py"),
                    "--config", args.config,
                    "--teacher", args.signed_teacher,
                    "--teacher-manifest", args.signed_teacher_manifest,
                    "--posteriors", args.posteriors,
                    "--retrieval", args.retrieval,
                    "--output-dir", str(output),
                    "--model-name", args.signed_model,
                    "--device", args.device,
                    "--seed", str(args.seed),
                    "--max-groups", str(count),
                    "--runtime-implementation", implementation,
                ]
                if size is not None:
                    command.extend(("--forward-batch-size", str(size)))
                add_identity_flags(command, args)
                results.append(
                    execute_case(f"signed_v1.{count}.{suffix}", command, output)
                )
    payload = {
        "schema_version": "rag_cbwdm_runtime_benchmark.v1",
        "created_at": utc_now(),
        "seed": args.seed,
        "group_counts": args.group_counts,
        "forward_batch_sizes": args.forward_batch_sizes,
        "device": args.device,
        "git": git_state(PROJECT_ROOT),
        "formal_artifacts_modified": False,
        "input_statistics": input_statistics(args),
        "results": results,
    }
    atomic_write_json(output_root / "benchmark_results.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

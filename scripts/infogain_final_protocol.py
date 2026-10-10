"""Run and audit the eight adapted-InfoGain development groups.

This command intentionally separates CPU selection, protocol audit, GPU
evaluation, and result summarization.  It never trains a model and accepts only
the development ``validation`` split declared by the plan.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.artifact_binding import (
    MATCHED_MAIN,
    build_evaluation_binding,
    validate_generator_identity_against_config,
    validate_generator_manifest,
    validate_formal_training_binding,
    validate_selection_provenance,
)
from src.baselines.infogain_protocol import (
    FINAL_PROTOCOL_ID,
    rank_and_select,
    stop_reason_for_selection,
)
from src.io_utils import load_yaml, read_jsonl
from src.fm2_formal import validate_fm2_retrieval_manifest
from src.run_manifest import atomic_write_json, sha256_file


EXPECTED_GENERATORS = {
    "qwen2.5-0.5b-instruct",
    "qwen2.5-1.5b-instruct",
    "qwen2.5-7b-instruct",
    "mistral-7b-instruct-v0.3",
}
EXPECTED_DATASETS = {
    "fever_binary_v2": 5000,
    "fm2_official_closed_page_v1": 1169,
}
EXPECTED_RETRIEVAL_PROTOCOLS = {
    "fever_binary_v2": "fever_bm25_v1",
    "fm2_official_closed_page_v1": "fm2_official_closed_page_v1",
}
EXPECTED_ARTIFACT_ROOT = Path(
    "/root/experiments/rag_cbwdm/formal_v2_infogain_top4_filter02_final"
)
PROTECTED_ARTIFACT_ROOTS = (
    Path("/root/experiments/rag_cbwdm/formal_v2_full_development"),
    Path("/root/experiments/rag_cbwdm/formal_v2_infogain_threshold02"),
    Path(
        "/root/experiments/rag_cbwdm/"
        "formal_v2_fever_kcbwdm_linear_gate_v2_full_development"
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "source-preflight",
            "selection",
            "audit",
            "preflight",
            "smoke",
            "evaluation",
            "summary",
            "cpu-pipeline",
            "gpu-pipeline",
        ),
    )
    parser.add_argument("--group", action="append", dest="groups")
    parser.add_argument("--audit-output")
    parser.add_argument("--summary-dir")
    parser.add_argument("--smoke-limit", type=int, default=32)
    parser.add_argument("--attempt-dir")
    return parser.parse_args(argv)


def load_plan(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan = load_yaml(path)
    if plan.get("schema_version") != "rag_cbwdm_infogain_final_plan.v1":
        raise ValueError("InfoGain final plan schema mismatch")
    if plan.get("split") != "validation":
        raise ValueError("This development-only runner accepts only split=validation")
    if plan.get("method_id") != "infogain_fever" or int(plan.get("seed", -1)) != 13:
        raise ValueError("Final plan requires method_id=infogain_fever and seed=13")
    if Path(str(plan.get("artifact_root", ""))) != EXPECTED_ARTIFACT_ROOT:
        raise ValueError(f"Final artifact_root must equal {EXPECTED_ARTIFACT_ROOT}")
    protocol = plan.get("protocol", {})
    expected_protocol = {
        "protocol_id": FINAL_PROTOCOL_ID,
        "top_m": 4,
        "filter_threshold": 0.2,
        "min_docs": 2,
    }
    if protocol != expected_protocol:
        raise ValueError(f"Final protocol must equal {expected_protocol!r}")
    groups = list(plan.get("groups", []))
    keys = [str(group.get("group_id")) for group in groups]
    if len(groups) != 8 or len(set(keys)) != 8:
        raise ValueError("Final plan must contain exactly eight unique groups")
    pairs = {(group.get("dataset_id"), group.get("generator_id")) for group in groups}
    expected_pairs = {
        (dataset, generator)
        for dataset in EXPECTED_DATASETS
        for generator in EXPECTED_GENERATORS
    }
    if pairs != expected_pairs:
        raise ValueError("Final plan must cover the 2 datasets x 4 generators exactly once")
    for group in groups:
        dataset_id = str(group["dataset_id"])
        if int(group.get("expected_rows", -1)) != EXPECTED_DATASETS[dataset_id]:
            raise ValueError(f"Unexpected row count in plan group {group['group_id']}")
        for field in (
            "retrieval",
            "retrieval_manifest",
            "checkpoint",
            "training_manifest",
            "selection",
            "old_selection",
            "generator_manifest",
            "evaluation_root",
        ):
            value = str(group.get(field, ""))
            if not value.startswith("/"):
                raise ValueError(f"{group['group_id']} field {field} must be absolute")
            if "held_out" in value.lower() or "/test" in value.lower():
                raise ValueError(f"{group['group_id']} field {field} touches a forbidden split")
        if group.get("retrieval_protocol_id") != EXPECTED_RETRIEVAL_PROTOCOLS[dataset_id]:
            raise ValueError(f"Unexpected retrieval protocol in {group['group_id']}")
        if dataset_id == "fever_binary_v2" and group.get(
            "expected_retrieval_sha256"
        ) != "a71fffda269f9112c6c243cee3384df072455a604eb801c838b69e32993786eb":
            raise ValueError(f"Unexpected FEVER retrieval SHA in {group['group_id']}")
        if dataset_id == "fm2_official_closed_page_v1" and group.get(
            "expected_retrieval_sha256_prefix"
        ) != "f371600059769a24":
            raise ValueError(f"Unexpected FM2 retrieval SHA prefix in {group['group_id']}")
    return plan, groups


def choose_groups(groups: list[dict[str, Any]], selected: list[str] | None) -> list[dict[str, Any]]:
    if not selected:
        return groups
    by_id = {str(group["group_id"]): group for group in groups}
    unknown = set(selected) - set(by_id)
    if unknown:
        raise ValueError(f"Unknown group ids: {sorted(unknown)}")
    return [by_id[group_id] for group_id in selected]


def _score_trace(row: dict[str, Any], *, row_id: str) -> list[dict[str, Any]]:
    metadata = row.get("selection_metadata")
    trace = metadata.get("candidate_score_audit") if isinstance(metadata, dict) else None
    if not isinstance(trace, list):
        raise ValueError(f"{row_id}: missing candidate_score_audit")
    return trace


def validate_retrieval_source(group: dict[str, Any]) -> dict[str, Any]:
    retrieval = Path(group["retrieval"])
    manifest_path = Path(group["retrieval_manifest"])
    if not retrieval.is_file():
        raise FileNotFoundError(
            f"{group['group_id']}: retrieval file does not exist: {retrieval}"
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{group['group_id']}: retrieval manifest does not exist: {manifest_path}"
        )
    if group["dataset_id"] == "fm2_official_closed_page_v1":
        result = validate_fm2_retrieval_manifest(
            manifest_path,
            retrieval,
            expected_formal_role="validation",
            expected_dataset_id=group["dataset_id"],
            expected_retrieval_protocol_id=group["retrieval_protocol_id"],
        )
        prefix = str(group.get("expected_retrieval_sha256_prefix") or "")
        if prefix and not str(result["pool_sha256"]).startswith(prefix):
            raise ValueError(
                f"{group['group_id']}: FM2 retrieval SHA does not start with {prefix}"
            )
        if int(result["num_rows"]) != int(group["expected_rows"]):
            raise ValueError(f"{group['group_id']}: FM2 retrieval row count mismatch")
        return result
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "rag_cbwdm_retrieval.v1":
        raise ValueError(f"{group['group_id']}: FEVER retrieval manifest schema mismatch")
    if manifest.get("completed") is not True or manifest.get("split") != "validation":
        raise ValueError(f"{group['group_id']}: FEVER retrieval manifest is not completed validation")
    if manifest.get("output_sha256") != sha256_file(retrieval):
        raise ValueError(f"{group['group_id']}: FEVER retrieval output SHA mismatch")
    expected_sha = group.get("expected_retrieval_sha256")
    if expected_sha and manifest.get("output_sha256") != expected_sha:
        raise ValueError(
            f"{group['group_id']}: FEVER retrieval SHA differs from frozen plan"
        )
    if int(manifest.get("num_output_rows", -1)) != int(group["expected_rows"]):
        raise ValueError(f"{group['group_id']}: FEVER retrieval row count mismatch")
    return {
        "dataset_id": group["dataset_id"],
        "retrieval_protocol_id": group["retrieval_protocol_id"],
        "formal_role": "validation",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "pool_path": str(retrieval.resolve()),
        "pool_sha256": sha256_file(retrieval),
        "num_rows": int(manifest["num_output_rows"]),
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def source_preflight_group(
    group: dict[str, Any], *, artifact_root: Path
) -> dict[str, Any]:
    """Validate one CPU-selection source without loading a model."""
    retrieval = validate_retrieval_source(group)
    checkpoint = Path(group["checkpoint"])
    training_manifest = Path(group["training_manifest"])
    if not checkpoint.is_dir():
        raise FileNotFoundError(
            f"{group['group_id']}: checkpoint directory does not exist: {checkpoint}"
        )
    if not training_manifest.is_file():
        raise FileNotFoundError(
            f"{group['group_id']}: training manifest does not exist: {training_manifest}"
        )
    for required in (
        checkpoint / "infogain_config.json",
        checkpoint / "heads.pt",
        checkpoint / "encoder",
    ):
        if not required.exists():
            raise FileNotFoundError(
                f"{group['group_id']}: checkpoint component does not exist: {required}"
            )
    if not any(
        (checkpoint / "encoder" / name).is_file()
        for name in ("model.safetensors", "pytorch_model.bin")
    ):
        raise FileNotFoundError(
            f"{group['group_id']}: checkpoint encoder weights do not exist"
        )
    training_binding = validate_formal_training_binding(
        training_manifest,
        checkpoint,
        method="infogain_fever",
        expected_dataset_id=group["dataset_id"],
        expected_conditioning_generator_id=group["generator_id"],
        expected_seed=13,
    )
    if training_binding.get("retrieval_protocol_id") != group["retrieval_protocol_id"]:
        raise ValueError(f"{group['group_id']}: checkpoint retrieval protocol mismatch")
    if training_binding.get("retrieval_source_sha256") != retrieval["pool_sha256"]:
        raise ValueError(f"{group['group_id']}: checkpoint retrieval source SHA mismatch")

    selection = Path(group["selection"])
    selection_manifest = selection.with_suffix(".manifest.json")
    if not _is_within(selection, artifact_root):
        raise ValueError(f"{group['group_id']}: selection is outside final artifact root")
    if any(_is_within(selection, protected) for protected in PROTECTED_ARTIFACT_ROOTS):
        raise ValueError(f"{group['group_id']}: selection overlaps a protected experiment root")
    existing = (selection.is_file(), selection_manifest.is_file())
    if existing[0] != existing[1]:
        raise ValueError(f"{group['group_id']}: partial selection artifact already exists")
    output_state = "absent_ready"
    if all(existing):
        audit_group(group)
        output_state = "completed_compatible_resume"
    return {
        "status": "PASS",
        "group_id": group["group_id"],
        "dataset_id": group["dataset_id"],
        "generator_id": group["generator_id"],
        "retrieval_sha256": retrieval["pool_sha256"],
        "retrieval_manifest_sha256": retrieval["manifest_sha256"],
        "checkpoint_fingerprint": training_binding["checkpoint_fingerprint"],
        "selection_output_state": output_state,
    }


def source_preflight_groups(
    groups: list[dict[str, Any]],
    *,
    artifact_root: Path,
    validator: Any = source_preflight_group,
) -> list[dict[str, Any]]:
    """Fail on the first invalid group; callers must not launch tmux on failure."""
    return [validator(group, artifact_root=artifact_root) for group in groups]


def _append_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()


def run_logged_stage_command(
    command: list[str], attempt_dir: Path, stage_name: str
) -> int:
    """Run one pipeline stage while preserving merged output and its exit code."""
    attempt_dir.mkdir(parents=True, exist_ok=True)
    launcher_log = attempt_dir / "launcher.log"
    stage_log = attempt_dir / f"{stage_name}.log"
    exit_path = attempt_dir / f"{stage_name}.exitcode"
    status_path = attempt_dir / f"{stage_name}.status.json"
    heading = f"\n=== stage={stage_name} ===\nCOMMAND {json.dumps(command)}\n"
    _append_text(launcher_log, heading)
    _append_text(stage_log, heading)
    return_code = 1
    try:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            _append_text(stage_log, line)
            _append_text(launcher_log, line)
            print(line, end="", flush=True)
        return_code = process.wait()
    except BaseException:
        details = traceback.format_exc()
        _append_text(stage_log, details)
        _append_text(launcher_log, details)
        print(details, file=sys.stderr, end="", flush=True)
    finally:
        exit_path.write_text(f"{return_code}\n", encoding="utf-8")
        atomic_write_json(
            status_path,
            {
                "stage": stage_name,
                "status": "PASS" if return_code == 0 else "FAIL",
                "exit_code": return_code,
                "command": command,
                "log": str(stage_log),
            },
        )
        trailer = f"EXIT stage={stage_name} code={return_code}\n"
        _append_text(stage_log, trailer)
        _append_text(launcher_log, trailer)
    return return_code


def run_pipeline(
    plan_path: str | Path,
    pipeline: str,
    attempt_dir: Path,
    *,
    smoke_limit: int = 32,
) -> int:
    """Run a fail-fast CPU or GPU pipeline with durable per-stage status."""
    if pipeline not in {"cpu", "gpu"}:
        raise ValueError(f"Unknown pipeline: {pipeline}")
    script = str(Path(__file__).resolve())
    base = [sys.executable, script, "--plan", str(plan_path)]
    if pipeline == "cpu":
        stages = [
            (
                "01_source_preflight",
                base
                + [
                    "--stage",
                    "source-preflight",
                    "--audit-output",
                    str(attempt_dir / "source_preflight.json"),
                ],
            ),
            ("02_selection", base + ["--stage", "selection"]),
            (
                "03_protocol_audit",
                base
                + [
                    "--stage",
                    "audit",
                    "--audit-output",
                    str(attempt_dir / "protocol_audit.json"),
                ],
            ),
        ]
    else:
        stages = [
            (
                "01_gpu_preflight",
                base
                + [
                    "--stage",
                    "preflight",
                    "--audit-output",
                    str(attempt_dir / "evaluation_preflight.json"),
                ],
            ),
            (
                "02_smoke",
                base + ["--stage", "smoke", "--smoke-limit", str(smoke_limit)],
            ),
            ("03_evaluation", base + ["--stage", "evaluation"]),
            ("04_summary", base + ["--stage", "summary"]),
        ]

    attempt_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        attempt_dir / "pipeline_status.json",
        {"pipeline": pipeline, "status": "RUNNING", "stages": [name for name, _ in stages]},
    )
    final_code = 0
    failed_index: int | None = None
    for index, (stage_name, command) in enumerate(stages):
        final_code = run_logged_stage_command(command, attempt_dir, stage_name)
        if final_code:
            failed_index = index
            break
    if failed_index is not None:
        for stage_name, _ in stages[failed_index + 1 :]:
            (attempt_dir / f"{stage_name}.skipped").write_text(
                f"Skipped after {stages[failed_index][0]} failed with exit code {final_code}.\n",
                encoding="utf-8",
            )
    (attempt_dir / "overall.exitcode").write_text(f"{final_code}\n", encoding="utf-8")
    atomic_write_json(
        attempt_dir / "pipeline_status.json",
        {
            "pipeline": pipeline,
            "status": "PASS" if final_code == 0 else "FAIL",
            "exit_code": final_code,
            "failed_stage": stages[failed_index][0] if failed_index is not None else None,
            "stages": [name for name, _ in stages],
        },
    )
    _append_text(
        attempt_dir / "launcher.log",
        f"PIPELINE_EXIT pipeline={pipeline} code={final_code}\n",
    )
    return final_code


def audit_rows(
    retrieval_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    *,
    expected_rows: int,
) -> dict[str, Any]:
    if len(retrieval_rows) != expected_rows or len(selection_rows) != expected_rows:
        raise ValueError(
            f"row count mismatch retrieval={len(retrieval_rows)} "
            f"selection={len(selection_rows)} expected={expected_rows}"
        )
    retrieval_ids = [str(row.get("id")) for row in retrieval_rows]
    selection_ids = [str(row.get("id")) for row in selection_rows]
    if len(set(retrieval_ids)) != expected_rows or len(set(selection_ids)) != expected_rows:
        raise ValueError("retrieval/selection ids must each be unique")
    if selection_ids != retrieval_ids:
        raise ValueError("selection ids/order must exactly match retrieval identity")

    hist: Counter[int] = Counter()
    fallback_examples = 0
    evidence_chars: list[int] = []
    for source, selected in zip(retrieval_rows, selection_rows):
        row_id = str(source.get("id"))
        if selected.get("split") != "validation":
            raise ValueError(f"{row_id}: selection split is not validation")
        if selected.get("method") != "infogain_fever":
            raise ValueError(f"{row_id}: method identity mismatch")
        if selected.get("uses_gold_at_test") is not False:
            raise ValueError(f"{row_id}: uses_gold_at_test must be false")
        candidates = list(source.get("candidates", []))
        candidate_ids = [str(candidate.get("doc_id")) for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"{row_id}: duplicate candidate doc_id")
        trace = _score_trace(selected, row_id=row_id)
        if len(trace) != len(candidates):
            raise ValueError(f"{row_id}: audit trace length mismatch")
        by_source_position = sorted(trace, key=lambda item: int(item["source_position"]))
        if [str(item["doc_id"]) for item in by_source_position] != candidate_ids:
            raise ValueError(f"{row_id}: trace does not match retrieval candidate identity/order")
        rank_scores = [float(item["rank_score"]) for item in by_source_position]
        probabilities = [float(item["filter_probability"]) for item in by_source_position]
        ranked, expected = rank_and_select(
            candidates,
            rank_scores,
            probabilities,
            top_m=4,
            filter_threshold=0.2,
            min_docs=2,
        )
        expected_trace = [
            {
                key: item[key]
                for key in (
                    "doc_id", "source_position", "rerank_position", "rank_score",
                    "filter_probability", "within_top_m", "passes_filter",
                )
            }
            for item in ranked
        ]
        if trace != expected_trace:
            raise ValueError(f"{row_id}: candidate score audit trace is inconsistent")
        expected_ids = [item["doc_id"] for item in expected]
        if selected.get("selected_doc_ids") != expected_ids:
            raise ValueError(f"{row_id}: selected ids violate final protocol")
        if int(selected.get("num_docs", -1)) != len(expected_ids):
            raise ValueError(f"{row_id}: num_docs mismatch")
        if int(selected.get("max_docs", -1)) != 4:
            raise ValueError(f"{row_id}: max_docs mismatch")
        if not 0 <= len(expected_ids) <= min(4, len(candidates)):
            raise ValueError(f"{row_id}: selected count outside protocol bounds")
        if len(candidates) >= 2 and not 2 <= len(expected_ids) <= 4:
            raise ValueError(f"{row_id}: min_docs was not met")
        top_ids = {item["doc_id"] for item in ranked[:4]}
        if not set(expected_ids) <= top_ids:
            raise ValueError(f"{row_id}: selected document came from below rank Top-4")

        docs = list(selected.get("selected_docs", []))
        steps = list(selected.get("selection_steps", []))
        if [str(doc.get("doc_id")) for doc in docs] != expected_ids:
            raise ValueError(f"{row_id}: selected_docs order mismatch")
        if [str(step.get("selected_doc_id")) for step in steps] != expected_ids:
            raise ValueError(f"{row_id}: selection_steps order mismatch")
        expected_fallback = [bool(item["min_docs_fallback"]) for item in expected]
        if [bool(doc.get("min_docs_fallback")) for doc in docs] != expected_fallback:
            raise ValueError(f"{row_id}: selected_docs fallback flags mismatch")
        if [bool(step.get("min_docs_fallback")) for step in steps] != expected_fallback:
            raise ValueError(f"{row_id}: selection_steps fallback flags mismatch")
        for index, (doc, step, item) in enumerate(zip(docs, steps, expected)):
            if int(doc.get("selection_step", -1)) != index or int(step.get("step", -1)) != index:
                raise ValueError(f"{row_id}: selection step indices mismatch")
            if step.get("stop") is not False:
                raise ValueError(f"{row_id}: selected step must have stop=false")
            if not math.isclose(float(doc["rank_score"]), item["rank_score"]):
                raise ValueError(f"{row_id}: selected doc rank score mismatch")
            if not math.isclose(float(doc["filter_probability"]), item["filter_probability"]):
                raise ValueError(f"{row_id}: selected doc filter probability mismatch")
            if not math.isclose(float(step["predicted_score"]), item["rank_score"]):
                raise ValueError(f"{row_id}: selection step rank score mismatch")
            if not math.isclose(float(step["filter_probability"]), item["filter_probability"]):
                raise ValueError(f"{row_id}: selection step filter probability mismatch")
        expected_stop = stop_reason_for_selection(
            candidate_count=len(candidates),
            selected_count=len(expected),
            top_m=4,
            min_docs=2,
            filter_threshold=0.2,
            used_fallback=any(expected_fallback),
        )
        if selected.get("stop_reason") != expected_stop:
            raise ValueError(f"{row_id}: stop_reason mismatch")
        metadata = selected.get("selection_metadata", {})
        for field, value in (
            ("selection_protocol_id", FINAL_PROTOCOL_ID),
            ("top_m", 4),
            ("filter_threshold", 0.2),
            ("min_docs", 2),
            ("uses_gold_at_test", False),
        ):
            if metadata.get(field) != value:
                raise ValueError(f"{row_id}: selection metadata {field} mismatch")
        hist[len(expected_ids)] += 1
        fallback_examples += int(any(expected_fallback))
        evidence_chars.append(
            sum(len(str(doc.get("text") or "")) for doc in docs)
        )
    return {
        "status": "PASS",
        "num_rows": expected_rows,
        "num_docs_histogram": {str(value): hist.get(value, 0) for value in range(5)},
        "avg_num_docs": sum(value * count for value, count in hist.items()) / expected_rows,
        "fallback_examples": fallback_examples,
        "avg_selected_docs_text_chars": sum(evidence_chars) / expected_rows,
        "evidence_chars_basis": "sum_selected_docs.text_without_prompt_formatting",
    }


def audit_group(group: dict[str, Any]) -> dict[str, Any]:
    retrieval_binding = validate_retrieval_source(group)
    selection_path = Path(group["selection"])
    binding = validate_selection_provenance(
        selection_path,
        selection_path.with_suffix(".manifest.json"),
        formal_v2=True,
        expected_dataset_id=str(group["dataset_id"]),
        expected_method="infogain_fever",
    )
    artifact_binding = binding["artifact_binding"]
    if artifact_binding.get("conditioning_generator_id") != group["generator_id"]:
        raise ValueError(f"{group['group_id']}: conditioning generator mismatch")
    if artifact_binding.get("retrieval_protocol_id") != group["retrieval_protocol_id"]:
        raise ValueError(f"{group['group_id']}: retrieval protocol mismatch")
    if artifact_binding.get("seed") != 13:
        raise ValueError(f"{group['group_id']}: training seed mismatch")
    manifest = binding["manifest"]
    contract = manifest.get("contract", {})
    parameters = contract.get("parameters", {})
    expected_parameters = {
        "top_m": 4,
        "min_docs": 2,
        "filter_threshold": 0.2,
        "limit": None,
        "selection_protocol_id": FINAL_PROTOCOL_ID,
    }
    if parameters != expected_parameters:
        raise ValueError(f"{group['group_id']}: selection contract parameters mismatch")
    retrieval = Path(group["retrieval"])
    retrieval_input = contract.get("inputs", {}).get("retrieval", {})
    if Path(retrieval_input.get("path", "")).resolve() != retrieval.resolve():
        raise ValueError(f"{group['group_id']}: retrieval path differs from selection contract")
    if retrieval_input.get("sha256") != sha256_file(retrieval):
        raise ValueError(f"{group['group_id']}: retrieval SHA differs from selection contract")
    retrieval_manifest = Path(group["retrieval_manifest"])
    manifest_input = contract.get("inputs", {}).get("retrieval_manifest", {})
    if Path(manifest_input.get("path", "")).resolve() != retrieval_manifest.resolve():
        raise ValueError(f"{group['group_id']}: retrieval manifest path differs from contract")
    if manifest_input.get("sha256") != sha256_file(retrieval_manifest):
        raise ValueError(f"{group['group_id']}: retrieval manifest SHA differs from contract")
    row_report = audit_rows(
        list(read_jsonl(retrieval)),
        list(read_jsonl(selection_path)),
        expected_rows=int(group["expected_rows"]),
    )
    row_report.update(
        {
            "group_id": group["group_id"],
            "dataset_id": group["dataset_id"],
            "generator_id": group["generator_id"],
            "selection": str(selection_path),
            "selection_sha256": binding["selection_sha256"],
            "manifest_fingerprint": binding["manifest_fingerprint"],
            "checkpoint_fingerprint": artifact_binding["checkpoint_fingerprint"],
            "retrieval_manifest_sha256": retrieval_binding["manifest_sha256"],
            "generator_identity_fingerprint": artifact_binding[
                "generator_identity_fingerprint"
            ],
        }
    )
    old_path = Path(group["old_selection"])
    if old_path.is_file():
        old_rows = list(read_jsonl(old_path))
        new_rows = list(read_jsonl(selection_path))
        old_by_id = {str(row.get("id")): row for row in old_rows}
        changed = sum(
            old_by_id.get(str(row.get("id")), {}).get("selected_doc_ids")
            != row.get("selected_doc_ids")
            for row in new_rows
        )
        old_counts = [int(row.get("num_docs", 0)) for row in old_rows]
        new_counts = [int(row.get("num_docs", 0)) for row in new_rows]
        row_report["historical_threshold02_comparison"] = {
            "historical_path": str(old_path),
            "historical_rows": len(old_rows),
            "changed_selected_doc_ids": changed,
            "historical_avg_num_docs": sum(old_counts) / len(old_counts) if old_counts else None,
            "final_avg_num_docs": sum(new_counts) / len(new_counts) if new_counts else None,
        }
    else:
        row_report["historical_threshold02_comparison"] = {
            "historical_path": str(old_path),
            "status": "not_present_on_this_host",
        }
    return row_report


def preflight_group(group: dict[str, Any]) -> dict[str, Any]:
    audit = audit_group(group)
    generator = validate_generator_manifest(
        group["generator_manifest"],
        expected_dataset_id=group["dataset_id"],
        expected_generator_id=group["generator_id"],
    )
    generator_manifest = generator["manifest"]
    config_path = Path(generator_manifest["config_path"])
    config = load_yaml(config_path)
    validate_generator_identity_against_config(
        generator["generator_identity"], config["generator"]
    )
    model_path = Path(generator["generator_identity"]["model_name_or_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Generator model is missing: {model_path}")
    selection = validate_selection_provenance(
        group["selection"],
        Path(group["selection"]).with_suffix(".manifest.json"),
        expected_dataset_id=group["dataset_id"],
        expected_method="infogain_fever",
    )
    evaluation_binding = build_evaluation_binding(
        selection, generator, experiment_type=MATCHED_MAIN
    )
    eval_root = Path(group["evaluation_root"])
    outputs = [
        eval_root / "predictions.jsonl",
        eval_root / "metrics.json",
        eval_root / "metrics.manifest.json",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and len(existing) != len(outputs):
        raise ValueError(
            f"{group['group_id']}: partial evaluation artifacts exist: {existing}"
        )
    if len(existing) == len(outputs):
        manifest = json.loads(outputs[2].read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or manifest.get("completed") is not True:
            raise ValueError(f"{group['group_id']}: existing evaluation is incomplete")
        if manifest.get("artifact_binding") != evaluation_binding:
            raise ValueError(f"{group['group_id']}: existing evaluation binding mismatch")
        if manifest.get("predictions_sha256") != sha256_file(outputs[0]):
            raise ValueError(f"{group['group_id']}: prediction checksum mismatch")
        if manifest.get("metrics_sha256") != sha256_file(outputs[1]):
            raise ValueError(f"{group['group_id']}: metrics checksum mismatch")
        output_state = "completed_compatible_resume"
    else:
        output_state = "absent_ready"
    return {
        "status": "PASS",
        "group_id": group["group_id"],
        "protocol_audit": audit,
        "config_path": str(config_path),
        "model_path": str(model_path),
        "generator_manifest_sha256": generator["manifest_sha256"],
        "evaluation_output_state": output_state,
    }


def _run_logged(command: list[str], *, log_path: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    exit_path = log_path.with_suffix(log_path.suffix + ".exitcode")
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("COMMAND " + json.dumps(command) + "\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    exit_path.write_text(str(completed.returncode) + "\n", encoding="utf-8")
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def run_selection(group: dict[str, Any], run_root: Path) -> None:
    validate_retrieval_source(group)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "12c_select_infogain_reranker.py"),
        "--retrieval", str(group["retrieval"]),
        "--retrieval-manifest", str(group["retrieval_manifest"]),
        "--checkpoint-dir", str(group["checkpoint"]),
        "--training-manifest", str(group["training_manifest"]),
        "--dataset-id", str(group["dataset_id"]),
        "--generator-id", str(group["generator_id"]),
        "--expected-seed", "13",
        "--formal-v2-identity",
        "--output", str(group["selection"]),
        "--device", "cpu",
        "--batch-size", "32",
        "--top-m", "4",
        "--filter-threshold", "0.2",
        "--min-docs", "2",
        "--method-name", "infogain_fever",
        "--resume",
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    _run_logged(
        command,
        log_path=run_root / "selection_logs" / f"{group['group_id']}.log",
        env=env,
    )


def _evaluation_command(group: dict[str, Any], *, output_root: Path, limit: int | None) -> list[str]:
    manifest = json.loads(Path(group["generator_manifest"]).read_text(encoding="utf-8"))
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "07_eval_rag_classification.py"),
        "--config", str(manifest["config_path"]),
        "--split", "validation",
        "--selection", str(group["selection"]),
        "--selection-manifest", str(Path(group["selection"]).with_suffix(".manifest.json")),
        "--output", str(output_root / "predictions.jsonl"),
        "--metrics-output", str(output_root / "metrics.json"),
        "--generator-manifest", str(group["generator_manifest"]),
        "--experiment-type", "matched_main",
        "--formal-v2-identity",
        "--method-name", "infogain_fever",
        "--resume",
    ]
    if limit is not None:
        command.extend(("--limit", str(limit)))
    return command


def run_evaluation(group: dict[str, Any], run_root: Path, *, smoke_limit: int | None) -> None:
    if smoke_limit is None:
        output_root = Path(group["evaluation_root"])
        log_name = f"{group['group_id']}.log"
    else:
        output_root = run_root / "smoke" / str(group["group_id"])
        log_name = f"smoke_{group['group_id']}.log"
    command = _evaluation_command(group, output_root=output_root, limit=smoke_limit)
    _run_logged(command, log_path=run_root / "evaluation_logs" / log_name)


def write_audit_report(reports: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output,
        {
            "schema_version": "rag_cbwdm_infogain_final_audit.v1",
            "status": "PASS",
            "protocol_id": FINAL_PROTOCOL_ID,
            "groups": reports,
        },
    )
    comparison = output.with_name(
        "historical_threshold02_comparison.csv"
        if output.name == "protocol_audit.json"
        else f"{output.stem}.historical_threshold02_comparison.csv"
    )
    fields = [
        "group_id", "dataset_id", "generator_id", "status", "historical_rows",
        "changed_selected_doc_ids", "historical_avg_num_docs", "final_avg_num_docs",
        "historical_path",
    ]
    with comparison.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            old = report["historical_threshold02_comparison"]
            writer.writerow(
                {
                    "group_id": report["group_id"],
                    "dataset_id": report["dataset_id"],
                    "generator_id": report["generator_id"],
                    **{field: old.get(field) for field in fields[3:]},
                }
            )


def summarize(groups: list[dict[str, Any]], output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for group in groups:
        root = Path(group["evaluation_root"])
        metrics_path = root / "metrics.json"
        manifest_path = root / "metrics.manifest.json"
        if not metrics_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Missing completed evaluation for {group['group_id']}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or manifest.get("metrics_sha256") != sha256_file(metrics_path):
            raise ValueError(f"Invalid evaluation manifest for {group['group_id']}")
        binding = manifest.get("artifact_binding", {})
        if (
            binding.get("dataset_id") != group["dataset_id"]
            or binding.get("evaluation_generator_id") != group["generator_id"]
            or binding.get("experiment_type") != "matched_main"
            or binding.get("seed") != 13
        ):
            raise ValueError(f"Evaluation identity mismatch for {group['group_id']}")
        if int(metrics.get("num_examples", -1)) != int(group["expected_rows"]):
            raise ValueError(f"Evaluation row count mismatch for {group['group_id']}")
        hist = metrics.get("num_docs", {}).get("histogram", {})
        rows.append(
            {
                "dataset_id": group["dataset_id"],
                "generator_id": group["generator_id"],
                "method_id": "infogain_fever",
                "experiment_type": "matched_main",
                "seed": 13,
                "split": "validation",
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "num_examples": metrics["num_examples"],
                "missing_predictions": metrics["missing_predictions"],
                "nan_inf": metrics["nan_inf"],
                "avg_num_docs": metrics["avg_num_docs"],
                **{f"num_docs_{value}": hist.get(str(value), 0) for value in range(5)},
                "avg_evidence_chars": metrics["avg_evidence_chars"],
                "evidence_chars_basis": metrics["evidence_chars_basis"],
                "avg_evidence_tokens": metrics.get("avg_evidence_tokens"),
                "evidence_tokens_basis": metrics.get("evidence_tokens_basis"),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output_dir / "infogain_final_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    md_header = "| " + " | ".join(fields) + " |"
    md_rule = "|" + "|".join("---" for _ in fields) + "|"
    md_rows = ["| " + " | ".join(str(row[field]) for field in fields) + " |" for row in rows]
    (output_dir / "infogain_final_results.md").write_text(
        "\n".join((md_header, md_rule, *md_rows)) + "\n", encoding="utf-8"
    )
    latex_rows = [
        " & ".join(
            (
                str(row["dataset_id"]), str(row["generator_id"]), "InfoGain (adapted)",
                f"{float(row['accuracy']):.4f}", f"{float(row['macro_f1']):.4f}",
                f"{float(row['avg_num_docs']):.3f}",
            )
        ) + r" \\"
        for row in rows
    ]
    (output_dir / "infogain_final_results.tex").write_text(
        "\n".join(latex_rows) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plan, all_groups = load_plan(args.plan)
    groups = choose_groups(all_groups, args.groups)
    run_root = Path(plan["run_root"])
    if args.stage in {"cpu-pipeline", "gpu-pipeline"}:
        if not args.attempt_dir:
            raise ValueError(f"--attempt-dir is required for {args.stage}")
        return_code = run_pipeline(
            args.plan,
            args.stage.removesuffix("-pipeline"),
            Path(args.attempt_dir),
            smoke_limit=args.smoke_limit,
        )
        raise SystemExit(return_code)
    if args.stage == "source-preflight":
        reports = source_preflight_groups(
            all_groups, artifact_root=Path(plan["artifact_root"])
        )
        output = Path(args.audit_output or run_root / "audit" / "source_preflight.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            output,
            {
                "schema_version": "rag_cbwdm_infogain_source_preflight.v1",
                "status": "PASS",
                "groups": reports,
            },
        )
        print(f"[infogain_final] source-preflight=PASS groups={len(reports)} output={output}")
    elif args.stage == "selection":
        for group in groups:
            run_selection(group, run_root)
    elif args.stage == "audit":
        reports = [audit_group(group) for group in groups]
        default_name = (
            "protocol_audit.json"
            if not args.groups
            else "protocol_audit." + ".".join(args.groups) + ".json"
        )
        output = Path(args.audit_output or run_root / "audit" / default_name)
        write_audit_report(reports, output)
        print(f"[infogain_final] audit=PASS groups={len(reports)} output={output}")
    elif args.stage == "preflight":
        reports = [preflight_group(group) for group in all_groups]
        output = Path(args.audit_output or run_root / "audit" / "evaluation_preflight.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output, {"status": "PASS", "groups": reports})
        print(f"[infogain_final] preflight=PASS groups={len(reports)} output={output}")
    elif args.stage == "smoke":
        if args.smoke_limit < 1:
            raise ValueError("smoke-limit must be positive")
        preflight_group(next(group for group in all_groups if group["group_id"] == "fever_qwen05"))
        run_evaluation(
            next(group for group in all_groups if group["group_id"] == "fever_qwen05"),
            run_root,
            smoke_limit=args.smoke_limit,
        )
    elif args.stage == "evaluation":
        # A formal GPU invocation is gated on all eight groups, even when only
        # one --group is requested for failure recovery.
        for group in all_groups:
            preflight_group(group)
        for group in groups:
            run_evaluation(group, run_root, smoke_limit=None)
    else:
        summarize(
            all_groups,
            Path(args.summary_dir or Path(plan["artifact_root"]) / "summary"),
        )


def _attempt_dir_from_argv(argv: list[str]) -> Path | None:
    try:
        index = argv.index("--attempt-dir")
        return Path(argv[index + 1])
    except (ValueError, IndexError):
        return None


def _main_entrypoint() -> None:
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        details = traceback.format_exc()
        attempt_dir = _attempt_dir_from_argv(sys.argv[1:])
        if attempt_dir is not None:
            attempt_dir.mkdir(parents=True, exist_ok=True)
            _append_text(attempt_dir / "launcher.log", details)
            (attempt_dir / "overall.exitcode").write_text("1\n", encoding="utf-8")
            atomic_write_json(
                attempt_dir / "pipeline_status.json",
                {"status": "FAIL", "exit_code": 1, "failure": "launcher_exception"},
            )
        print(details, file=sys.stderr, end="", flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    _main_entrypoint()

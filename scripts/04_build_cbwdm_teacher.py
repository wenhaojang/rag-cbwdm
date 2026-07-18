from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterator

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cbwdm_score import (
    TEACHER_SCHEMA_VERSION,
    build_local_effects,
    canonical_l_type,
    greedy_teacher,
)
from src.io_utils import load_yaml, read_jsonl, require_keys, write_jsonl


def parse_bool(value: str | bool) -> bool:
    """Parse common true/false strings for CLI flags."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for CBWDM teacher construction."""
    parser = argparse.ArgumentParser(description="Build greedy RAG-CBWDM teacher trajectories.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
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
        help="Data role. held_out_test is allowed only for explicitly diagnostic Oracle.",
    )
    parser.add_argument(
        "--diagnostic-oracle",
        action="store_true",
        help="Allow gold-dependent teacher construction on held_out_test for non-deployable Oracle only.",
    )
    parser.add_argument("--posteriors", default=None, help="Override posterior JSONL path.")
    parser.add_argument("--output", default=None, help="Override teacher JSONL output path.")
    parser.add_argument("--limit", type=int, default=None, help="Max posterior rows to process.")
    parser.add_argument("--top-m", type=int, default=None, help="Override config.cbwdm.top_m.")
    parser.add_argument("--ridge-lambda", type=float, default=None, help="Override config.cbwdm.ridge_lambda.")
    parser.add_argument(
        "--stop-threshold",
        type=float,
        default=None,
        help="Override config.cbwdm.stop_threshold.",
    )
    parser.add_argument("--eps-smooth", type=float, default=None, help="Override config.cbwdm.eps_smooth.")
    parser.add_argument("--l-type", default=None, help="Override config.cbwdm.L_type.")
    parser.add_argument("--target-smoothing", default=None)
    parser.add_argument("--gain-tolerance", type=float, default=None)
    parser.add_argument(
        "--store-all-gains",
        type=parse_bool,
        default=None,
        help="Whether to store all per-step candidate gains. Default comes from config.",
    )
    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve relative paths against the project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def validate_config(config: Dict[str, Any]) -> None:
    """Validate config fields needed for CBWDM teacher construction."""
    require_keys(config, ["dataset", "paths", "cbwdm"], "config")
    require_keys(config["paths"], ["processed_dir"], "config.paths")


def default_posteriors_path(config: Dict[str, Any], split: str) -> Path:
    """Infer default posterior input path."""
    return PROJECT_ROOT / "outputs" / "posteriors" / f"{config['dataset']}_{split}_posteriors.jsonl"


def default_output_path(config: Dict[str, Any], split: str) -> Path:
    """Infer default teacher output path."""
    return PROJECT_ROOT / "outputs" / "teacher" / f"{config['dataset']}_{split}_cbwdm_teacher.jsonl"


def get_cbwdm_params(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Merge CBWDM config values with CLI overrides."""
    cbwdm = config.get("cbwdm", {})
    return {
        "top_m": args.top_m if args.top_m is not None else int(cbwdm.get("top_m", 4)),
        "ridge_lambda": (
            args.ridge_lambda if args.ridge_lambda is not None else float(cbwdm.get("ridge_lambda", 0.01))
        ),
        "stop_threshold": (
            args.stop_threshold
            if args.stop_threshold is not None
            else float(cbwdm.get("stop_threshold", 0.0))
        ),
        "eps_smooth": args.eps_smooth if args.eps_smooth is not None else float(cbwdm.get("eps_smooth", 0.0)),
        "l_type": canonical_l_type(
            args.l_type
            if args.l_type is not None
            else cbwdm.get("L_type", "euclidean_posterior_shift")
        ),
        "target_smoothing": (
            args.target_smoothing
            if args.target_smoothing is not None
            else cbwdm.get("target_smoothing", "paper_mixture")
        ),
        "gain_tolerance": (
            args.gain_tolerance
            if args.gain_tolerance is not None
            else float(cbwdm.get("gain_tolerance", 1e-10))
        ),
        "store_all_gains": (
            args.store_all_gains
            if args.store_all_gains is not None
            else bool(cbwdm.get("store_all_gains", True))
        ),
    }


def candidate_retrieval_score(candidate: Dict[str, Any]) -> Any:
    """Return retrieval score from either Stage 2 or Stage 3 field names."""
    return candidate.get("retrieval_score", candidate.get("score"))


def enrich_candidate(
    candidate: Dict[str, Any],
    x: np.ndarray,
    d: np.ndarray,
) -> Dict[str, Any]:
    """Attach local CBWDM feature diagnostics to one candidate."""
    require_keys(candidate, ["doc_id", "rank", "title", "text", "eta"], "candidate")
    return {
        "doc_id": candidate["doc_id"],
        "rank": candidate["rank"],
        "retrieval_score": candidate_retrieval_score(candidate),
        "title": candidate["title"],
        "text": candidate["text"],
        "eta": candidate["eta"],
        "x": [float(v) for v in x.tolist()],
        "alignment": float(x @ d),
        "norm_x": float(np.linalg.norm(x)),
    }


def enrich_step(step: Dict[str, Any], candidates: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Attach document metadata to a numeric greedy step."""
    current_indices = step["current_indices"]
    best_index = step["best_index"]
    enriched_gains = []
    for gain in step["candidate_gains"]:
        candidate = candidates[gain["index"]]
        enriched_gains.append(
            {
                "doc_id": candidate["doc_id"],
                "index": gain["index"],
                "rank": candidate.get("rank"),
                "retrieval_score": candidate_retrieval_score(candidate),
                "gain": gain["gain"],
                "raw_gain": gain.get("raw_gain", gain["gain"]),
                "theta_after_add": gain["theta_after_add"],
            }
        )

    return {
        "step": step["step"],
        "current_doc_ids": [candidates[idx]["doc_id"] for idx in current_indices],
        "current_indices": current_indices,
        "theta_before": step["theta_before"],
        "best_doc_id": candidates[best_index]["doc_id"],
        "best_index": best_index,
        "best_gain": step["best_gain"],
        "raw_best_gain": step.get("raw_best_gain", step["best_gain"]),
        "theta_after": step["theta_after"],
        "stop_decision": step.get("stop_decision", {"selected": True, "reason": "continue"}),
        "candidate_gains": enriched_gains,
    }


def build_teacher_row(row: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Build one teacher JSON object from one posterior JSON object."""
    require_keys(row, ["id", "query", "label", "split", "labels", "eta0", "candidates"], "posterior row")
    candidates = row["candidates"]
    if not isinstance(candidates, list):
        raise ValueError(f"posterior row {row['id']} has non-list candidates")

    base = {
        "schema_version": TEACHER_SCHEMA_VERSION,
        "teacher_type": "cbwdm_greedy_gold_label",
        "id": row["id"],
        "query": row["query"],
        "label": row["label"],
        "split": row["split"],
        "labels": row["labels"],
        "eta0": row["eta0"],
        "theta_empty": 0.0,
        "theta_final": 0.0,
        "teacher_selected_doc_ids": [],
        "teacher_selected_indices": [],
        "top_m": params["top_m"],
        "l_type": params["l_type"],
        "lambda": params["ridge_lambda"],
        "ridge_lambda": params["ridge_lambda"],
        "eps_smooth": params["eps_smooth"],
        "target_smoothing": params["target_smoothing"],
        "stop_threshold": params["stop_threshold"],
        "gain_tolerance": params["gain_tolerance"],
        "stop_reason": "no_candidates",
        "terminal_stop_decision": None,
        "steps": [],
        "candidates": [],
    }
    if not candidates:
        return base

    candidate_etas = np.asarray([candidate["eta"] for candidate in candidates], dtype=float)
    X_all, d = build_local_effects(
        eta0=np.asarray(row["eta0"], dtype=float),
        candidate_etas=candidate_etas,
        label=row["label"],
        labels=list(row["labels"]),
        l_type=params["l_type"],
        eps_smooth=params["eps_smooth"],
        target_smoothing=params["target_smoothing"],
    )
    teacher = greedy_teacher(
        X_all=X_all,
        d=d,
        top_m=params["top_m"],
        ridge_lambda=params["ridge_lambda"],
        stop_threshold=params["stop_threshold"],
        store_all_gains=params["store_all_gains"],
        gain_tolerance=params["gain_tolerance"],
    )
    selected_indices = teacher["selected_indices"]
    enriched_candidates = [
        enrich_candidate(candidate, X_all[idx], d) for idx, candidate in enumerate(candidates)
    ]
    enriched_steps = [enrich_step(step, candidates) for step in teacher["steps"]]

    base.update(
        {
            "teacher_selected_doc_ids": [candidates[idx]["doc_id"] for idx in selected_indices],
            "teacher_selected_indices": selected_indices,
            "steps": enriched_steps,
            "candidates": enriched_candidates,
            "theta_final": teacher["theta_final"],
            "stop_reason": teacher["stop_reason"],
            "terminal_stop_decision": teacher["terminal_stop_decision"],
        }
    )
    return base


def iter_teacher_rows(
    posteriors_path: Path,
    params: Dict[str, Any],
    limit: int | None = None,
    log_every: int = 100,
) -> Iterator[Dict[str, Any]]:
    """Stream teacher rows from posterior JSONL input."""
    for row_index, row in enumerate(read_jsonl(posteriors_path, limit=limit), start=1):
        teacher_row = build_teacher_row(row, params)
        if row_index % log_every == 0:
            print(f"[cbwdm_teacher] processed={row_index}", file=sys.stderr)
        yield teacher_row


def main() -> None:
    args = parse_args()
    if args.split in {"test", "held_out_test"} and not args.diagnostic_oracle:
        raise ValueError(
            "Gold-dependent CBWDM teacher construction is forbidden on held_out_test "
            "unless --diagnostic-oracle is explicit"
        )
    config = load_yaml(args.config)
    validate_config(config)
    params = get_cbwdm_params(config, args)

    posteriors_path = (
        resolve_project_path(args.posteriors) if args.posteriors else default_posteriors_path(config, args.split)
    )
    output_path = resolve_project_path(args.output) if args.output else default_output_path(config, args.split)
    written = write_jsonl(output_path, iter_teacher_rows(posteriors_path, params, limit=args.limit))
    print(
        f"[cbwdm_teacher][{config['dataset']}][{args.split}] rows={written} "
        f"top_m={params['top_m']} ridge_lambda={params['ridge_lambda']} "
        f"posteriors={posteriors_path} output={output_path}"
    )


if __name__ == "__main__":
    main()

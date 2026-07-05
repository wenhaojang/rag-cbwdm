from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import load_yaml, require_keys, write_jsonl
from src.retrieval_bm25 import BM25Retriever, iter_retrieval_results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for BM25 retrieval."""
    parser = argparse.ArgumentParser(description="Run BM25 retrieval for prepared FEVER data.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--split", required=True, choices=["train", "dev"], help="Query split.")
    parser.add_argument("--queries", default=None, help="Override prepared query JSONL path.")
    parser.add_argument("--corpus", default=None, help="Override corpus JSONL path.")
    parser.add_argument("--output", default=None, help="Override output JSONL path.")
    parser.add_argument("--top-n", type=int, default=None, help="Number of candidates per query.")
    parser.add_argument("--limit", type=int, default=None, help="Max query rows to process.")
    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve relative paths against the project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def validate_config(config: Dict[str, Any]) -> None:
    """Validate config fields needed by BM25 retrieval."""
    require_keys(config, ["dataset", "paths", "retrieval"], "config")
    require_keys(config["paths"], ["processed_dir"], "config.paths")
    require_keys(config["retrieval"], ["top_n"], "config.retrieval")


def get_top_n(config: Dict[str, Any], override: int | None) -> int:
    """Choose top_n from CLI override or config."""
    top_n = override if override is not None else int(config["retrieval"]["top_n"])
    if top_n <= 0:
        raise ValueError(f"--top-n must be positive, got {top_n}")
    return top_n


def default_queries_path(config: Dict[str, Any], split: str) -> Path:
    """Return default prepared query path for a split."""
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    return processed_dir / f"{config['dataset']}_{split}.jsonl"


def default_corpus_path(config: Dict[str, Any]) -> Path:
    """Return default prepared FEVER corpus path."""
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    return processed_dir / "fever_corpus_sentence.jsonl"


def default_output_path(config: Dict[str, Any], split: str, top_n: int) -> Path:
    """Return default BM25 retrieval output path."""
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    return processed_dir / f"{config['dataset']}_{split}_bm25_top{top_n}.jsonl"


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    validate_config(config)

    top_n = get_top_n(config, args.top_n)
    queries_path = resolve_project_path(args.queries) if args.queries else default_queries_path(config, args.split)
    corpus_path = resolve_project_path(args.corpus) if args.corpus else default_corpus_path(config)
    output_path = resolve_project_path(args.output) if args.output else default_output_path(config, args.split, top_n)

    retriever = BM25Retriever.from_jsonl(corpus_path)
    written = write_jsonl(
        output_path,
        iter_retrieval_results(
            retriever=retriever,
            queries_path=queries_path,
            top_n=top_n,
            limit=args.limit,
        ),
    )
    print(
        f"[bm25][{config['dataset']}][{args.split}] queries={written} top_n={top_n} "
        f"corpus={corpus_path} output={output_path}"
    )


if __name__ == "__main__":
    main()

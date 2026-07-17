"""Compatibility exports for the tiny in-memory BM25 backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator

from src.io_utils import read_jsonl, require_keys
from src.retrieval.memory_bm25 import MemoryBM25Retriever, tokenize

BM25Retriever = MemoryBM25Retriever


def iter_retrieval_results(
    retriever: Any,
    queries_path: str | Path,
    top_n: int,
    limit: int | None = None,
) -> Iterator[Dict[str, Any]]:
    """Stream BM25 retrieval results for prepared query samples."""
    for row_index, row in enumerate(read_jsonl(queries_path, limit=limit), start=1):
        require_keys(row, ["id", "query", "label", "split"], f"query row {row_index}")
        if not isinstance(row["query"], str):
            raise ValueError(f"query row {row_index} has non-string query")

        yield {
            "id": row["id"],
            "query": row["query"],
            "label": row["label"],
            "split": row["split"],
            "candidates": retriever.search(row["query"], top_n=top_n),
        }

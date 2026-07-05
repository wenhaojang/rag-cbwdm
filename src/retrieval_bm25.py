"""Lightweight BM25 retrieval for prepared RAG-CBWDM data."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

from src.io_utils import read_jsonl, require_keys

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Tokenize English text with lowercase alphanumeric regex matching."""
    return TOKEN_RE.findall(text.lower())


def import_bm25_okapi() -> Any:
    """Import rank_bm25 with a clear installation hint when it is absent."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise ImportError(
            "rank_bm25 is required for BM25 retrieval. Install it with: "
            "pip install rank_bm25"
        ) from exc
    return BM25Okapi


class BM25Retriever:
    """In-memory BM25 index over a JSONL passage corpus."""

    def __init__(self, docs: List[Dict[str, Any]]) -> None:
        if not docs:
            raise ValueError("Cannot build BM25 index from an empty corpus.")

        self.docs = docs
        tokenized_docs = [tokenize(f"{doc.get('title', '')} {doc['text']}") for doc in docs]
        BM25Okapi = import_bm25_okapi()
        self.index = BM25Okapi(tokenized_docs)

    @classmethod
    def from_jsonl(cls, corpus_path: str | Path) -> "BM25Retriever":
        """Load corpus JSONL into memory and build a BM25 index."""
        docs: List[Dict[str, Any]] = []
        for row_index, row in enumerate(read_jsonl(corpus_path), start=1):
            require_keys(row, ["doc_id", "title", "text"], f"corpus row {row_index}")
            if not isinstance(row["text"], str):
                raise ValueError(f"corpus row {row_index} has non-string text")
            docs.append(row)

        print(f"[bm25] loaded corpus docs={len(docs)} path={corpus_path}", file=sys.stderr)
        return cls(docs)

    def search(self, query: str, top_n: int) -> List[Dict[str, Any]]:
        """Return top_n candidate documents for a query."""
        if top_n <= 0:
            raise ValueError(f"top_n must be positive, got {top_n}")

        query_tokens = tokenize(query)
        scores = self.index.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)

        candidates: List[Dict[str, Any]] = []
        for rank, doc_index in enumerate(ranked_indices[:top_n], start=1):
            doc = self.docs[doc_index]
            candidates.append(
                {
                    "doc_id": doc["doc_id"],
                    "rank": rank,
                    "score": float(scores[doc_index]),
                    "title": doc["title"],
                    "text": doc["text"],
                }
            )
        return candidates


def iter_retrieval_results(
    retriever: BM25Retriever,
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

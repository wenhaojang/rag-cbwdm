"""Small-fixture-only in-memory rank_bm25 backend."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.io_utils import read_jsonl, require_keys
from src.retrieval.base import RetrievalBackend

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
DEFAULT_MAX_DOCUMENTS = 50_000


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class MemoryBM25Retriever(RetrievalBackend):
    def __init__(
        self, docs: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75
    ) -> None:
        if not docs:
            raise ValueError("Cannot build BM25 index from an empty corpus.")
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError("Install the debug backend with: pip install rank_bm25") from exc
        self.docs = docs
        self.index = BM25Okapi(
            [tokenize(f"{doc.get('title', '')} {doc['text']}") for doc in docs],
            k1=float(k1),
            b=float(b),
        )

    @classmethod
    def from_jsonl(
        cls,
        corpus_path: str | Path,
        max_documents: int = DEFAULT_MAX_DOCUMENTS,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "MemoryBM25Retriever":
        docs: list[dict[str, Any]] = []
        for row_index, row in enumerate(read_jsonl(corpus_path), start=1):
            if row_index > max_documents:
                raise RuntimeError(
                    f"memory_rank_bm25 safety limit exceeded ({max_documents} documents). "
                    "Use backend=pyserini_lucene for a persistent on-disk index."
                )
            require_keys(row, ["doc_id", "title", "text"], f"corpus row {row_index}")
            docs.append(row)
        return cls(docs, k1=k1, b=b)

    def search(self, query: str, top_n: int) -> list[dict[str, Any]]:
        if top_n <= 0:
            raise ValueError(f"top_n must be positive, got {top_n}")
        scores = self.index.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        return [
            {
                "doc_id": self.docs[index]["doc_id"],
                "rank": rank,
                "score": float(scores[index]),
                "title": self.docs[index]["title"],
                "text": self.docs[index]["text"],
                "meta": self.docs[index].get("meta"),
            }
            for rank, index in enumerate(ranked[:top_n], start=1)
        ]

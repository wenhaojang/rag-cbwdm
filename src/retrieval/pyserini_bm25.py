"""Pyserini/Lucene persistent BM25 index build, validation, and search."""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.io_utils import read_jsonl, require_keys
from src.retrieval.base import RetrievalBackend
from src.run_manifest import (
    atomic_write_json,
    git_state,
    sha256_file,
    stable_hash,
    utc_now,
)

MANIFEST_NAME = "index_manifest.json"
SCHEMA_VERSION = "rag_cbwdm_bm25_index.v1"
SEGMENTS_FILE_RE = re.compile(r"^segments_[^/\\]+$")


def pyserini_version() -> str:
    try:
        return importlib.metadata.version("pyserini")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError(
            "pyserini_lucene requires Pyserini and Java 11+. "
            "Install the server retrieval requirements before building the index."
        ) from exc


def corpus_identity(path: str | Path) -> dict[str, Any]:
    corpus = Path(path).resolve()
    stat = corpus.stat()
    return {
        "corpus_path": str(corpus),
        "corpus_sha256": sha256_file(corpus),
        "corpus_size_bytes": stat.st_size,
    }


def index_contract(
    corpus_path: str | Path,
    *,
    backend_version: str,
    k1: float,
    b: float,
    analyzer: str,
) -> dict[str, Any]:
    return {
        **corpus_identity(corpus_path),
        "backend": "pyserini_lucene",
        "backend_version": backend_version,
        "analyzer": analyzer,
        "language": "en",
        "tokenizer": "Lucene analyzer selected by Pyserini JsonCollection",
        "contents_construction_rule": "title + single ASCII space + text",
        "bm25": {"k1": float(k1), "b": float(b)},
        "document_schema": ["doc_id", "title", "text"],
    }


def index_inventory(index_dir: Path) -> list[dict[str, Any]]:
    """Return the canonical inventory used by both build and completion validation."""
    inventory = []
    for path in sorted(index_dir.rglob("*")):
        if path.is_file() and path.name != MANIFEST_NAME:
            inventory.append(
                {"path": str(path.relative_to(index_dir)), "size_bytes": path.stat().st_size}
            )
    return inventory


def validate_lucene_inventory(inventory: list[dict[str, Any]]) -> None:
    """Validate Lucene structure without relying on codec/version extensions."""
    segments = [
        item
        for item in inventory
        if SEGMENTS_FILE_RE.fullmatch(Path(str(item["path"])).name)
        and int(item["size_bytes"]) > 0
    ]
    if not segments:
        raise ValueError("Lucene index has no non-empty segments_N file")
    data_files = [
        item
        for item in inventory
        if Path(str(item["path"])).name not in {MANIFEST_NAME, "write.lock"}
        and not SEGMENTS_FILE_RE.fullmatch(Path(str(item["path"])).name)
        and int(item["size_bytes"]) > 0
    ]
    if not data_files:
        raise ValueError("Lucene index has no non-empty data files")


def _open_lucene_searcher(index_dir: Path) -> Any:
    try:
        from pyserini.search.lucene import LuceneSearcher
    except ImportError as exc:
        raise ImportError("Pyserini search API cannot be imported") from exc
    return LuceneSearcher(str(index_dir))


def validate_index(
    index_dir: str | Path,
    expected_contract: dict[str, Any] | None = None,
    *,
    searcher_factory: Any | None = None,
) -> dict[str, Any]:
    directory = Path(index_dir)
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"Index is incomplete: missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or not manifest.get("completed"):
        raise ValueError(f"Index manifest is not completed: {manifest_path}")
    contract = manifest.get("contract")
    if not isinstance(contract, dict) or manifest.get("fingerprint") != stable_hash(contract):
        raise ValueError("Index manifest fingerprint does not match its contract")
    if not manifest.get("probe_passed"):
        raise ValueError("Index manifest does not record a successful probe")
    inventory = index_inventory(directory)
    validate_lucene_inventory(inventory)
    if inventory != manifest.get("index_file_inventory"):
        raise ValueError("Index file inventory differs from the completed manifest")
    if not isinstance(manifest.get("num_documents"), int) or manifest["num_documents"] <= 0:
        raise ValueError("Index manifest has an invalid num_documents")
    if sum(1 for _ in read_jsonl(Path(manifest["corpus_path"]))) != manifest.get(
        "num_documents"
    ):
        raise ValueError("Index document count no longer matches the corpus")
    if expected_contract:
        if contract != expected_contract or manifest["fingerprint"] != stable_hash(
            expected_contract
        ):
            raise ValueError(
                "Existing index is incompatible with the requested corpus/backend/analyzer/BM25 "
                "parameters. Use --overwrite to rebuild it."
            )
    factory = searcher_factory or _open_lucene_searcher
    searcher = factory(directory)
    close = getattr(searcher, "close", None)
    if callable(close):
        close()
    return manifest


def _write_collection(corpus_path: Path, collection_dir: Path) -> int:
    collection_dir.mkdir(parents=True, exist_ok=True)
    target = collection_dir / "documents.jsonl"
    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for count, row in enumerate(read_jsonl(corpus_path), start=1):
            require_keys(row, ["doc_id", "title", "text"], f"corpus row {count}")
            stored = {
                "id": str(row["doc_id"]),
                "doc_id": str(row["doc_id"]),
                "title": str(row["title"]),
                "text": str(row["text"]),
                "contents": f"{row['title']} {row['text']}",
            }
            handle.write(json.dumps(stored, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if count == 0:
        raise ValueError("Cannot build an index from an empty corpus")
    return count


def build_index(
    corpus_path: str | Path,
    index_dir: str | Path,
    *,
    k1: float = 0.9,
    b: float = 0.4,
    analyzer: str = "english",
    threads: int = 1,
    resume: bool = False,
    overwrite: bool = False,
    run_command: Any = subprocess.run,
) -> dict[str, Any]:
    if analyzer != "english":
        raise ValueError("This FEVER index implementation currently supports analyzer=english only")
    version = pyserini_version()
    corpus = Path(corpus_path).resolve()
    directory = Path(index_dir).resolve()
    contract = index_contract(
        corpus, backend_version=version, k1=k1, b=b, analyzer=analyzer
    )
    if (directory / MANIFEST_NAME).exists() and resume and not overwrite:
        return validate_index(directory, contract)
    if directory.exists() and any(directory.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Index directory is not empty: {directory}. Use --resume or --overwrite."
            )
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    collection = directory.parent / f".{directory.name}.collection.partial"
    if collection.exists():
        shutil.rmtree(collection)
    try:
        num_documents = _write_collection(corpus, collection)
        command = [
            sys.executable,
            "-m",
            "pyserini.index.lucene",
            "--collection",
            "JsonCollection",
            "--input",
            str(collection),
            "--index",
            str(directory),
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--language",
            "en",
            "--threads",
            str(threads),
            "--storeRaw",
            "--storePositions",
            "--storeDocvectors",
        ]
        run_command(command, check=True)
        inventory = index_inventory(directory)
        validate_lucene_inventory(inventory)
        probe = _open_lucene_searcher(directory)
        try:
            probe.set_bm25(float(k1), float(b))
            first = next(read_jsonl(corpus))
            probe_query = str(first["title"] or first["text"]).strip()
            if not probe_query or not probe.search(probe_query, k=1):
                raise RuntimeError("Lucene index probe query returned no documents")
        finally:
            close = getattr(probe, "close", None)
            if callable(close):
                close()
        # Capture the same canonical post-probe inventory that resume validates.
        inventory = index_inventory(directory)
        validate_lucene_inventory(inventory)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "completed": True,
            "fingerprint": stable_hash(contract),
            "contract": contract,
            **contract,
            "num_documents": num_documents,
            "index_start_time": started,
            "index_end_time": utc_now(),
            "git_head": git_state(Path(__file__).resolve().parents[2]).get("commit"),
            "config_hash": stable_hash(
                {"backend": "pyserini_lucene", "k1": k1, "b": b, "analyzer": analyzer}
            ),
            "index_file_inventory": inventory,
            "probe_query": probe_query,
            "probe_passed": True,
        }
        # completed=true is published only after inventory, searcher-open, and probe checks pass.
        atomic_write_json(directory / MANIFEST_NAME, manifest)
        return manifest
    finally:
        if collection.exists():
            shutil.rmtree(collection)


class PyseriniBM25Retriever(RetrievalBackend):
    def __init__(self, index_dir: str | Path, k1: float = 0.9, b: float = 0.4) -> None:
        self.manifest = validate_index(index_dir)
        try:
            from pyserini.search.lucene import LuceneSearcher
        except ImportError as exc:
            raise ImportError(
                "Cannot open the Lucene index because Pyserini is unavailable."
            ) from exc
        self.searcher = LuceneSearcher(str(Path(index_dir).resolve()))
        self.searcher.set_bm25(float(k1), float(b))

    def search(self, query: str, top_n: int) -> list[dict[str, Any]]:
        if top_n <= 0:
            raise ValueError(f"top_n must be positive, got {top_n}")
        candidates = []
        for rank, hit in enumerate(self.searcher.search(query, k=top_n), start=1):
            raw = self.searcher.doc(hit.docid).raw()
            stored = json.loads(raw)
            candidates.append(
                {
                    "doc_id": stored.get("doc_id", hit.docid),
                    "rank": rank,
                    "score": float(hit.score),
                    "title": stored["title"],
                    "text": stored["text"],
                }
            )
        return candidates

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
SCHEMA_VERSION = "rag_cbwdm_bm25_index.v2"
INDEX_CONTRACT_VERSION = "rag_cbwdm_lucene_index_contract.v2"
DOCUMENT_ID_CONTRACT_VERSION = "fever_sentence_doc_id.v1"
RAW_METADATA_CONTRACT_VERSION = "fever_page_sentence_metadata.v1"
PAGE_SENTENCE_METADATA_CONTRACT_VERSION = "fever_page_sentence_metadata.v1"
COLLECTION_CLASS = "JsonCollection"
GENERATOR_CLASS = "DefaultLuceneDocumentGenerator"
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
    manifest_path = corpus.with_suffix(".manifest.json")
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            manifest = payload
    return {
        "corpus_path": str(corpus),
        "corpus_sha256": sha256_file(corpus),
        "corpus_size_bytes": stat.st_size,
        "corpus_fingerprint": manifest.get(
            "fingerprint", manifest.get("source_fingerprint")
        ),
        "corpus_contract_version": manifest.get(
            "corpus_contract_version", manifest.get("schema_version")
        ),
    }


def index_contract(
    corpus_path: str | Path,
    *,
    backend_version: str,
    k1: float,
    b: float,
    analyzer: str,
    store_raw: bool = True,
    store_positions: bool = True,
    store_docvectors: bool = True,
    lucene_version: str | None = None,
    document_id_contract_version: str = DOCUMENT_ID_CONTRACT_VERSION,
    raw_metadata_contract_version: str = RAW_METADATA_CONTRACT_VERSION,
    page_sentence_metadata_contract_version: str = (
        PAGE_SENTENCE_METADATA_CONTRACT_VERSION
    ),
) -> dict[str, Any]:
    return {
        "schema_version": INDEX_CONTRACT_VERSION,
        **corpus_identity(corpus_path),
        "document_id_contract_version": str(document_id_contract_version),
        "raw_metadata_contract_version": str(raw_metadata_contract_version),
        "page_sentence_metadata_contract_version": (
            str(page_sentence_metadata_contract_version)
        ),
        "backend": "pyserini_lucene",
        "backend_version": str(backend_version),
        "pyserini_version": str(backend_version),
        "lucene_version": str(
            lucene_version or f"bundled-with-pyserini-{backend_version}"
        ),
        "analyzer": str(analyzer),
        "language": "en",
        "tokenizer": "Lucene analyzer selected by Pyserini JsonCollection",
        "contents_construction_rule": "title + single ASCII space + text",
        "bm25": {"k1": float(k1), "b": float(b)},
        "store_raw": bool(store_raw),
        "store_positions": bool(store_positions),
        "store_docvectors": bool(store_docvectors),
        "collection": COLLECTION_CLASS,
        "generator": GENERATOR_CLASS,
        "document_schema": ["doc_id", "title", "text", "meta.page_id", "meta.sentence_id"],
        "stored_metadata": "FEVER page_id and sentence_id for validation diagnostics",
    }


def index_fingerprint(contract: dict[str, Any]) -> str:
    """Hash the complete canonical index contract."""
    if contract.get("schema_version") != INDEX_CONTRACT_VERSION:
        raise ValueError("Cannot fingerprint a legacy or unversioned index contract")
    return stable_hash(contract)


def contract_diff(
    existing: dict[str, Any], requested: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return a deterministic leaf-level contract diff."""
    differences: list[dict[str, Any]] = []

    def walk(left: Any, right: Any, field: str) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                walk(left.get(key), right.get(key), f"{field}.{key}" if field else key)
        elif left != right:
            differences.append(
                {"field": field, "existing": left, "requested": right}
            )

    walk(existing, requested, "")
    return differences


def index_inventory(index_dir: Path) -> list[dict[str, Any]]:
    """Return the canonical inventory used by both build and completion validation."""
    inventory = []
    for path in sorted(index_dir.rglob("*")):
        if path.is_file() and path.name != MANIFEST_NAME:
            inventory.append(
                {
                    "path": str(path.relative_to(index_dir)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
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


def probe_index_metadata(searcher: Any, corpus_path: str | Path) -> dict[str, Any]:
    """Verify search plus exact stored-raw FEVER metadata recovery."""
    first = next(read_jsonl(corpus_path))
    require_keys(first, ["doc_id", "title", "text", "meta"], "corpus probe row")
    meta = first["meta"]
    if not isinstance(meta, dict):
        raise ValueError("Corpus probe row meta must be an object")
    require_keys(meta, ["page_id", "sentence_id"], "corpus probe row meta")
    query = str(first["title"] or first["text"]).strip()
    if not query:
        raise ValueError("Corpus probe row has no searchable title or text")
    hits = searcher.search(query, k=10)
    if not hits:
        raise RuntimeError("Lucene index probe query returned no documents")
    expected_id = str(first["doc_id"])
    recovered: dict[str, Any] | None = None
    for hit in hits:
        docid = str(getattr(hit, "docid", ""))
        document = searcher.doc(docid)
        raw = document.raw()
        candidate = json.loads(raw)
        if str(candidate.get("doc_id", candidate.get("id"))) == expected_id:
            recovered = candidate
            break
    if recovered is None:
        document = searcher.doc(expected_id)
        if document is not None:
            candidate = json.loads(document.raw())
            if str(candidate.get("doc_id", candidate.get("id"))) == expected_id:
                recovered = candidate
    if recovered is None:
        raise ValueError("Lucene probe could not recover the sampled corpus document")
    recovered_meta = recovered.get("meta")
    if not isinstance(recovered_meta, dict):
        raise ValueError("Stored raw JSON lacks meta")
    require_keys(
        recovered,
        ["doc_id", "title", "text"],
        "stored raw probe document",
    )
    require_keys(
        recovered_meta,
        ["page_id", "sentence_id"],
        "stored raw probe metadata",
    )
    page_id = str(recovered_meta["page_id"]).strip()
    if not page_id:
        raise ValueError("Stored raw page_id is empty")
    try:
        sentence_id = int(recovered_meta["sentence_id"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Stored raw sentence_id is not an integer") from exc
    expected = {
        "doc_id": expected_id,
        "title": str(first["title"]),
        "text": str(first["text"]),
        "page_id": str(meta["page_id"]),
        "sentence_id": int(meta["sentence_id"]),
    }
    actual = {
        "doc_id": str(recovered["doc_id"]),
        "title": str(recovered["title"]),
        "text": str(recovered["text"]),
        "page_id": page_id,
        "sentence_id": sentence_id,
    }
    if actual != expected:
        raise ValueError(
            f"Stored raw metadata differs from sampled corpus row: "
            f"{contract_diff(actual, expected)}"
        )
    return {
        "passed": True,
        "query": query,
        "sample_doc_id": expected_id,
        "raw_json_parsed": True,
        "required_fields_present": True,
        "page_sentence_metadata_recovered": True,
        "matches_corpus_sample": True,
        "recovered": actual,
    }


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
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or not manifest.get("completed")
    ):
        raise ValueError(f"Index manifest is not completed: {manifest_path}")
    contract = manifest.get("contract")
    if (
        not isinstance(contract, dict)
        or contract.get("schema_version") != INDEX_CONTRACT_VERSION
        or manifest.get("index_fingerprint") != index_fingerprint(contract)
        or manifest.get("fingerprint") != index_fingerprint(contract)
    ):
        raise ValueError("Index manifest fingerprint does not match its contract")
    if (
        not manifest.get("probe_passed")
        or not manifest.get("raw_metadata_recovery", {}).get("passed")
    ):
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
        if (
            expected_contract.get("schema_version") != INDEX_CONTRACT_VERSION
            or contract != expected_contract
            or manifest["fingerprint"] != index_fingerprint(expected_contract)
        ):
            differences = contract_diff(contract, expected_contract)
            raise ValueError(
                "Existing index is incompatible with the requested corpus/backend/analyzer/BM25 "
                f"parameters; contract_diff={json.dumps(differences, sort_keys=True)}. "
                "Build the requested fingerprint in a new directory."
            )
    factory = searcher_factory or _open_lucene_searcher
    searcher = factory(directory)
    try:
        current_probe = probe_index_metadata(searcher, manifest["corpus_path"])
        if current_probe.get("recovered") != manifest.get(
            "raw_metadata_recovery", {}
        ).get("recovered"):
            raise ValueError("Current raw metadata probe differs from manifest")
    finally:
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
                "meta": row.get("meta"),
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
    store_raw: bool = True,
    store_positions: bool = True,
    store_docvectors: bool = True,
    lucene_version: str | None = None,
    document_id_contract_version: str = DOCUMENT_ID_CONTRACT_VERSION,
    raw_metadata_contract_version: str = RAW_METADATA_CONTRACT_VERSION,
    page_sentence_metadata_contract_version: str = (
        PAGE_SENTENCE_METADATA_CONTRACT_VERSION
    ),
    threads: int = 1,
    resume: bool = False,
    overwrite: bool = False,
    run_command: Any = subprocess.run,
) -> dict[str, Any]:
    if analyzer != "english":
        raise ValueError("This FEVER index implementation currently supports analyzer=english only")
    if not (store_raw and store_positions and store_docvectors):
        raise ValueError(
            "The formal FEVER Lucene index requires storeRaw, storePositions, "
            "and storeDocvectors"
        )
    version = pyserini_version()
    corpus = Path(corpus_path).resolve()
    directory = Path(index_dir).resolve()
    contract = index_contract(
        corpus,
        backend_version=version,
        k1=k1,
        b=b,
        analyzer=analyzer,
        store_raw=store_raw,
        store_positions=store_positions,
        store_docvectors=store_docvectors,
        lucene_version=lucene_version,
        document_id_contract_version=document_id_contract_version,
        raw_metadata_contract_version=raw_metadata_contract_version,
        page_sentence_metadata_contract_version=(
            page_sentence_metadata_contract_version
        ),
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
            COLLECTION_CLASS,
            "--input",
            str(collection),
            "--index",
            str(directory),
            "--generator",
            GENERATOR_CLASS,
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
            probe_result = probe_index_metadata(probe, corpus)
        finally:
            close = getattr(probe, "close", None)
            if callable(close):
                close()
        # Capture the same canonical post-probe inventory that resume validates.
        inventory = index_inventory(directory)
        validate_lucene_inventory(inventory)
        manifest = {
            **contract,
            "schema_version": SCHEMA_VERSION,
            "index_contract_schema_version": INDEX_CONTRACT_VERSION,
            "status": "completed",
            "completed": True,
            "index_fingerprint": index_fingerprint(contract),
            "fingerprint": index_fingerprint(contract),
            "contract": contract,
            "num_documents": num_documents,
            "index_start_time": started,
            "index_end_time": utc_now(),
            "git_head": git_state(Path(__file__).resolve().parents[2]).get("commit"),
            "config_hash": stable_hash(
                {"backend": "pyserini_lucene", "k1": k1, "b": b, "analyzer": analyzer}
            ),
            "index_file_inventory": inventory,
            "probe_query": probe_result["query"],
            "probe_passed": True,
            "probe_result": probe_result,
            "raw_metadata_recovery": probe_result,
            "git": git_state(Path(__file__).resolve().parents[2]),
            "created_at": utc_now(),
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
                    "meta": stored.get("meta"),
                }
            )
        return candidates

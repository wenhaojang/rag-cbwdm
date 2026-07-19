from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import load_yaml
from src.retrieval.pyserini_bm25 import build_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reusable Pyserini/Lucene BM25 index.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    retrieval = config.get("retrieval", {})
    backend = retrieval.get("backend", "pyserini_lucene")
    if backend != "pyserini_lucene":
        raise ValueError("The persistent index stage requires retrieval.backend=pyserini_lucene")
    bm25 = retrieval.get("bm25", {})
    index_config = retrieval.get("index", {})
    manifest = build_index(
        args.corpus,
        args.index,
        k1=float(bm25.get("k1", 0.9)),
        b=float(bm25.get("b", 0.4)),
        analyzer=str(index_config.get("analyzer", "english")),
        store_raw=bool(index_config.get("store_raw", True)),
        store_positions=bool(index_config.get("store_positions", True)),
        store_docvectors=bool(index_config.get("store_docvectors", True)),
        lucene_version=index_config.get("lucene_version"),
        document_id_contract_version=str(
            index_config.get(
                "document_id_contract_version", "fever_sentence_doc_id.v1"
            )
        ),
        raw_metadata_contract_version=str(
            index_config.get(
                "raw_metadata_contract_version",
                "fever_page_sentence_metadata.v1",
            )
        ),
        page_sentence_metadata_contract_version=str(
            index_config.get(
                "page_sentence_metadata_contract_version",
                "fever_page_sentence_metadata.v1",
            )
        ),
        threads=args.threads,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(
        f"[index] completed documents={manifest['num_documents']} "
        f"fingerprint={manifest['fingerprint']} path={Path(args.index).resolve()}"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import load_yaml, read_jsonl, require_keys
from src.run_manifest import atomic_write_json, sha256_file, stable_hash, utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FEVER wiki sentence corpus.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--wiki-pages-dir", default=None, help="Override FEVER wiki-pages directory.")
    parser.add_argument(
        "--limit",
        dest="limit_documents",
        type=int,
        default=None,
        help="Max sentence documents to emit (micro/debug only).",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-pages", type=int, default=None, help="Optional raw page scan cap.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Defaults to processed_dir/fever_corpus_sentence.jsonl.",
    )
    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def validate_config(config: Dict[str, Any]) -> None:
    require_keys(config, ["paths", "corpus"], "config")
    require_keys(config["paths"], ["raw_wiki_pages_dir", "processed_dir"], "config.paths")
    require_keys(config["corpus"], ["mode", "min_text_chars"], "config.corpus")
    if config["corpus"]["mode"] != "sentence":
        raise ValueError("Only corpus.mode='sentence' is supported in the minimal stage.")


def iter_wiki_files(wiki_dir: Path) -> Iterator[Path]:
    if not wiki_dir.exists():
        raise FileNotFoundError(f"FEVER wiki-pages directory does not exist: {wiki_dir}")
    if not wiki_dir.is_dir():
        raise NotADirectoryError(f"Expected FEVER wiki-pages directory: {wiki_dir}")
    yield from sorted(wiki_dir.rglob("*.jsonl"))


def parse_sentence_line(line: str, fallback_id: int) -> tuple[int, str]:
    parts = line.split("\t")
    sentence_text = parts[1] if len(parts) >= 2 else line
    try:
        sentence_id = int(parts[0])
    except (ValueError, IndexError):
        sentence_id = fallback_id
    return sentence_id, sentence_text.strip()


def sentences_from_page(page: Dict[str, Any]) -> Iterator[tuple[int, str]]:
    lines = page.get("lines")
    if isinstance(lines, str) and lines.strip():
        for fallback_id, line in enumerate(lines.splitlines()):
            yield parse_sentence_line(line, fallback_id)
        return

    text = page.get("text")
    if isinstance(text, str) and text.strip():
        yield 0, text.strip()


def make_doc_id(global_index: int, sentence_id: int) -> str:
    return f"feverwiki_{global_index:08d}_sent_{sentence_id:04d}"


def iter_passages(
    wiki_dir: Path,
    min_text_chars: int,
    limit_pages: int | None,
    limit_documents: int | None,
    stats: Dict[str, int],
) -> Iterator[Dict[str, Any]]:
    for wiki_file in iter_wiki_files(wiki_dir):
        if limit_pages is not None and stats["pages"] >= limit_pages:
            break
        stats["files"] += 1

        for page in read_jsonl(wiki_file):
            if limit_pages is not None and stats["pages"] >= limit_pages:
                break
            stats["pages"] += 1

            title = page.get("id")
            if not isinstance(title, str) or not title:
                stats["skipped"] += 1
                print(
                    f"[warning][corpus] {wiki_file} page {stats['pages']} skipped: missing id",
                    file=sys.stderr,
                )
                continue

            page_has_passage = False
            for sentence_id, text in sentences_from_page(page):
                if limit_documents is not None and stats["passages"] >= limit_documents:
                    return
                if len(text) < min_text_chars:
                    stats["skipped"] += 1
                    continue
                doc_global_index = stats["passages"] + 1
                yield {
                    "doc_id": make_doc_id(doc_global_index, sentence_id),
                    "title": title,
                    "text": text,
                    "meta": {
                        "source": "fever_wiki_pages",
                        "page_id": title,
                        "sentence_id": sentence_id,
                    },
                }
                stats["passages"] += 1
                page_has_passage = True
            if not page_has_passage:
                stats["skipped"] += 1


def default_output_path(config: Dict[str, Any]) -> Path:
    return resolve_project_path(config["paths"]["processed_dir"]) / "fever_corpus_sentence.jsonl"


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    validate_config(config)

    wiki_dir = resolve_project_path(args.wiki_pages_dir) if args.wiki_pages_dir else resolve_project_path(config["paths"]["raw_wiki_pages_dir"])
    output_path = resolve_project_path(args.output) if args.output else default_output_path(config)
    manifest_path = output_path.with_suffix(".manifest.json")
    source_inventory = [
        {
            "path": str(path.relative_to(wiki_dir)),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in iter_wiki_files(wiki_dir)
    ]
    source_fingerprint = stable_hash(source_inventory)
    expected = {
        "source_fingerprint": source_fingerprint,
        "corpus_limit": args.limit_documents,
        "page_limit": args.limit_pages,
        "min_text_chars": int(config["corpus"]["min_text_chars"]),
    }
    if output_path.exists() or manifest_path.exists():
        if args.resume and output_path.is_file() and manifest_path.is_file() and not args.overwrite:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            actual = {key: existing.get(key) for key in expected}
            if existing.get("completed") and actual == expected:
                if existing.get("output_sha256") != sha256_file(output_path):
                    raise ValueError("Existing corpus output checksum differs from its manifest")
                print(f"[corpus] reused documents={existing['num_documents']} output={output_path}")
                return
            raise ValueError(
                "Existing corpus is incompatible or incomplete; use --overwrite to rebuild."
            )
        if not args.overwrite:
            raise FileExistsError(
                f"Corpus output already exists: {output_path}. Use --resume or --overwrite."
            )
    stats = {"files": 0, "pages": 0, "passages": 0, "skipped": 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for row in iter_passages(
            wiki_dir=wiki_dir,
            min_text_chars=int(config["corpus"]["min_text_chars"]),
            limit_pages=args.limit_pages,
            limit_documents=args.limit_documents,
            stats=stats,
        ):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, output_path)
    manifest = {
        "schema_version": "rag_cbwdm_corpus.v1",
        "completed": True,
        "wiki_pages_dir": str(wiki_dir.resolve()),
        "source_fingerprint": source_fingerprint,
        "files_scanned": stats["files"],
        "pages_scanned": stats["pages"],
        "num_documents": stats["passages"],
        "rows_skipped": stats["skipped"],
        "corpus_limit": args.limit_documents,
        "page_limit": args.limit_pages,
        "min_text_chars": int(config["corpus"]["min_text_chars"]),
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "output_size_bytes": output_path.stat().st_size,
        "completed_at": utc_now(),
    }
    atomic_write_json(output_path.with_suffix(".manifest.json"), manifest)
    print(
        f"[corpus] files={stats['files']} pages={stats['pages']} passages={stats['passages']} "
        f"skipped={stats['skipped']} output={output_path}"
    )


if __name__ == "__main__":
    main()

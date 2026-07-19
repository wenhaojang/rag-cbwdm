"""Download and materialize FEVER data for the rag_cbwdm project.

This script uses the Hugging Face `datasets` loader and writes files to the
raw layout expected by the current rag_cbwdm pipeline:

    data/raw/fever/train.jsonl
    data/raw/fever/dev.jsonl
    data/raw/fever/wiki-pages/wiki-000.jsonl ...

It writes the FEVER v1.0 train and labelled_dev splits in the original-style
fields used by our preprocessing script: id, label, claim, evidence.
The HF v1.0 builder is sentence/evidence flattened, so this script groups rows
by claim id and reconstructs a minimal evidence list.

It writes wiki_pages as JSONL shards with fields: id, text, lines.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def write_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def materialize_claim_split(ds, out_path: Path, limit: int | None = None) -> None:
    """Group HF FEVER v1.0 rows by claim id and write FEVER-style JSONL."""
    grouped: dict[int, dict[str, Any]] = {}
    evidences: dict[int, list[list[Any]]] = defaultdict(list)

    n_read = 0
    for row in ds:
        if limit is not None and len(grouped) >= limit:
            # Because rows are flattened, this may cut off some evidence for the last claim.
            # That is acceptable for small debug downloads; use full download for final runs.
            break
        cid = _safe_int(row.get("id"))
        label = row.get("label")
        claim = row.get("claim")
        if cid not in grouped:
            grouped[cid] = {
                "id": cid,
                "label": label,
                "claim": claim,
            }
        wiki_url = row.get("evidence_wiki_url", "")
        sent_id = _safe_int(row.get("evidence_sentence_id"))
        ann_id = _safe_int(row.get("evidence_annotation_id"))
        ev_id = _safe_int(row.get("evidence_id"))
        if label == "NOT ENOUGH INFO" or not wiki_url or sent_id < 0:
            ev_tuple = [ann_id, ev_id, None, None]
        else:
            ev_tuple = [ann_id, ev_id, wiki_url, sent_id]
        if ev_tuple not in evidences[cid]:
            evidences[cid].append(ev_tuple)
        n_read += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for cid, base in grouped.items():
            evs = evidences.get(cid) or [[-1, -1, None, None]]
            # FEVER official shape is list of evidence sets. The HF flattened rows do not
            # preserve set grouping perfectly, so we store a single evidence set containing
            # unique tuples. Our current preprocessing only needs id/claim/label.
            base["evidence"] = [evs]
            f.write(json.dumps(base, ensure_ascii=False) + "\n")
    print(f"wrote {len(grouped)} claims to {out_path} from {n_read} HF rows")


def materialize_wiki_pages(ds, out_dir: Path, shard_size: int = 100_000, limit: int | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = 0
    in_shard = 0
    total = 0
    f = None
    try:
        for row in ds:
            if limit is not None and total >= limit:
                break
            if f is None or in_shard >= shard_size:
                if f is not None:
                    f.close()
                shard_path = out_dir / f"wiki-{shard_idx:03d}.jsonl"
                f = shard_path.open("w", encoding="utf-8")
                print(f"writing {shard_path}")
                shard_idx += 1
                in_shard = 0
            out = {
                "id": row.get("id"),
                "text": row.get("text", ""),
                "lines": row.get("lines", ""),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            in_shard += 1
            total += 1
    finally:
        if f is not None:
            f.close()
    print(f"wrote {total} wiki pages to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/materialize FEVER v1.0 and wiki_pages for rag_cbwdm.")
    parser.add_argument("--output-root", default="data/raw/fever", help="Output root, default: data/raw/fever")
    parser.add_argument("--skip-wiki", action="store_true", help="Only download train/dev claims, skip wiki_pages.")
    parser.add_argument("--claim-limit", type=int, default=None, help="Optional number of claim ids per split for debugging.")
    parser.add_argument("--wiki-limit", type=int, default=None, help="Optional number of wiki pages for debugging.")
    parser.add_argument("--wiki-shard-size", type=int, default=100000, help="Rows per wiki shard JSONL.")
    parser.add_argument("--cache-dir", default=None, help="Optional HF datasets cache directory, e.g. D:/hf_cache/datasets")
    parser.add_argument(
        "--revision",
        required=True,
        help="Immutable Hugging Face dataset revision/commit for auditable recovery.",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: datasets. Install with `python -m pip install datasets`." ) from exc

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("Loading FEVER v1.0 from Hugging Face datasets...")
    train = load_dataset(
        "fever",
        "v1.0",
        split="train",
        cache_dir=args.cache_dir,
        revision=args.revision,
        trust_remote_code=True,
    )
    dev = load_dataset(
        "fever",
        "v1.0",
        split="labelled_dev",
        cache_dir=args.cache_dir,
        revision=args.revision,
        trust_remote_code=True,
    )
    materialize_claim_split(train, output_root / "train.jsonl", limit=args.claim_limit)
    materialize_claim_split(dev, output_root / "dev.jsonl", limit=args.claim_limit)

    if not args.skip_wiki:
        print("Loading FEVER wiki_pages from Hugging Face datasets. This can take a while and several GB of disk.")
        wiki = load_dataset(
            "fever",
            "wiki_pages",
            split="wikipedia_pages",
            cache_dir=args.cache_dir,
            revision=args.revision,
            trust_remote_code=True,
        )
        materialize_wiki_pages(wiki, output_root / "wiki-pages", shard_size=args.wiki_shard_size, limit=args.wiki_limit)

    print("Done.")
    print(f"Expected raw layout under: {output_root.resolve()}")
    print("  train.jsonl")
    print("  dev.jsonl")
    print("  wiki-pages/wiki-*.jsonl")


if __name__ == "__main__":
    main()

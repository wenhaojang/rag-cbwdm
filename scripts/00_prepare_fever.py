from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import load_yaml, read_jsonl, require_keys, write_jsonl


DEFAULT_SPLITS = ["train", "dev", "test"]
SPLIT_PATH_KEYS = {
    "train": "raw_fever_train",
    "dev": "raw_fever_dev",
    "test": "raw_fever_test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FEVER claims for RAG-CBWDM.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Max raw rows to read per split.")
    parser.add_argument("--raw-train", default=None, help="Override raw train JSONL path.")
    parser.add_argument("--raw-dev", default=None, help="Override raw dev JSONL path.")
    parser.add_argument("--raw-test", default=None, help="Override raw test JSONL path.")
    parser.add_argument("--output-dir", default=None, help="Override processed output directory.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        choices=DEFAULT_SPLITS,
        help="Splits to process.",
    )
    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def validate_config(config: Dict[str, Any]) -> None:
    require_keys(config, ["dataset", "paths", "task"], "config")
    require_keys(config["paths"], list(SPLIT_PATH_KEYS.values()) + ["processed_dir"], "config.paths")
    require_keys(config["task"], ["label_map", "drop_labels"], "config.task")


def convert_label(raw_label: str | None, label_map: Dict[str, str], drop_labels: set[str]) -> str | None:
    if raw_label is None:
        return None
    if raw_label in drop_labels:
        return None
    return label_map.get(raw_label)


def make_output_id(dataset: str, split: str, index: int) -> str:
    return f"{dataset}_{split}_{index:08d}"


def convert_row(
    row: Dict[str, Any],
    dataset: str,
    split: str,
    output_index: int,
    label_map: Dict[str, str],
    drop_labels: set[str],
) -> tuple[Dict[str, Any] | None, str | None]:
    if "claim" not in row:
        return None, "missing required field: claim"
    if split != "test" and "label" not in row:
        return None, "missing required field: label"

    original_label = row.get("label")
    mapped_label = convert_label(original_label, label_map, drop_labels)
    if original_label is not None and mapped_label is None:
        if original_label in drop_labels:
            return None, None
        return None, f"unknown label: {original_label}"

    return {
        "id": make_output_id(dataset, split, output_index),
        "dataset": dataset,
        "split": split,
        "query": row["claim"],
        "label": mapped_label,
        "meta": {
            "source": "fever",
            "original_id": row.get("id"),
            "original_label": original_label,
        },
    }, None


def iter_converted_rows(
    raw_path: Path,
    dataset: str,
    split: str,
    label_map: Dict[str, str],
    drop_labels: set[str],
    limit: int | None,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw FEVER file does not exist for split '{split}': {raw_path}")

    rows: List[Dict[str, Any]] = []
    stats = {"read": 0, "kept": 0, "dropped": 0, "skipped": 0}

    for raw_row in read_jsonl(raw_path, limit=limit):
        stats["read"] += 1
        converted, warning = convert_row(
            raw_row,
            dataset=dataset,
            split=split,
            output_index=stats["kept"] + 1,
            label_map=label_map,
            drop_labels=drop_labels,
        )
        if warning:
            stats["skipped"] += 1
            print(f"[warning][{split}] row {stats['read']} skipped: {warning}", file=sys.stderr)
            continue
        if converted is None:
            stats["dropped"] += 1
            continue
        rows.append(converted)
        stats["kept"] += 1

    return rows, stats


def process_split(
    config: Dict[str, Any],
    split: str,
    limit: int | None,
    raw_override: str | None = None,
    output_dir_override: str | None = None,
) -> None:
    dataset = config["dataset"]
    paths = config["paths"]
    task = config["task"]

    raw_path = resolve_project_path(raw_override) if raw_override else resolve_project_path(paths[SPLIT_PATH_KEYS[split]])
    processed_dir = resolve_project_path(output_dir_override) if output_dir_override else resolve_project_path(paths["processed_dir"])
    output_path = processed_dir / f"{dataset}_{split}.jsonl"

    rows, stats = iter_converted_rows(
        raw_path=raw_path,
        dataset=dataset,
        split=split,
        label_map=task["label_map"],
        drop_labels=set(task.get("drop_labels", [])),
        limit=limit,
    )
    write_jsonl(output_path, rows)
    print(
        f"[{dataset}][{split}] read={stats['read']} kept={stats['kept']} "
        f"dropped={stats['dropped']} skipped={stats['skipped']} output={output_path}"
    )


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    validate_config(config)

    raw_overrides = {
        "train": args.raw_train,
        "dev": args.raw_dev,
        "test": args.raw_test,
    }
    for split in args.splits:
        if split == "test" and raw_overrides[split] is None and args.raw_train and args.raw_dev:
            continue
        process_split(
            config,
            split,
            args.limit,
            raw_override=raw_overrides[split],
            output_dir_override=args.output_dir,
        )


if __name__ == "__main__":
    main()

"""Small IO helpers shared by the data preparation scripts."""

from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List


def ensure_dir(path: str | Path) -> Path:
    """Create directory if needed and return Path."""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def read_jsonl(path: str | Path, limit: int | None = None) -> Iterator[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file. Skip empty lines."""
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file does not exist: {jsonl_path}")
    if not jsonl_path.is_file():
        raise IsADirectoryError(f"Expected a JSONL file but got a directory: {jsonl_path}")

    yielded = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if limit is not None and yielded >= limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {jsonl_path} at line {line_no}: {exc.msg}"
                ) from exc
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Expected JSON object in {jsonl_path} at line {line_no}, "
                    f"got {type(obj).__name__}"
                )
            yielded += 1
            yield obj


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    """Write rows to JSONL. Return number of written rows."""
    jsonl_path = Path(path)
    ensure_dir(jsonl_path.parent)

    count = 0
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load YAML config."""
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML config does not exist: {yaml_path}")
    if not yaml_path.is_file():
        raise IsADirectoryError(f"Expected a YAML config file but got a directory: {yaml_path}")

    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load YAML configs. Install it with: pip install pyyaml"
        ) from exc

    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at top level in {yaml_path}")
    return data


def require_keys(obj: Dict[str, Any], keys: List[str], where: str = "object") -> None:
    """Raise KeyError with clear message if required keys are absent."""
    missing = [key for key in keys if key not in obj]
    if missing:
        raise KeyError(f"Missing required key(s) in {where}: {', '.join(missing)}")

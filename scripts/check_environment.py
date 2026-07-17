from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check server prerequisites without downloading models.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default="outputs/runs")
    parser.add_argument("--cache-root", default=".cache/huggingface")
    parser.add_argument("--json-output")
    return parser.parse_args()


def module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
        return str(getattr(module, "__version__", "installed"))
    except ImportError:
        return None


def writable_check(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return os.access(path, os.W_OK)
    except OSError:
        return False


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_yaml(config_path)
    modules = {
        name: module_version(name)
        for name in ("torch", "transformers", "accelerate", "numpy", "yaml", "datasets", "pyserini")
    }
    torch_info: dict[str, Any] = {}
    try:
        import torch

        torch_info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpus": [
                {
                    "name": torch.cuda.get_device_name(index),
                    "memory_gib": round(
                        torch.cuda.get_device_properties(index).total_memory / 2**30, 2
                    ),
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        torch_info = {"cuda_available": False, "error": "torch is not installed"}
    output_root = Path(args.output_root)
    cache_root = Path(args.cache_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    if not cache_root.is_absolute():
        cache_root = PROJECT_ROOT / cache_root
    required_scripts = [
        "02a_build_bm25_index.py",
        "02_retrieve_bm25.py",
        "03_compute_label_posteriors.py",
        "04_build_cbwdm_teacher.py",
        "10_train_cross_encoder_selector.py",
        "11_select_with_cross_encoder.py",
    ]
    result = {
        "python": sys.version,
        "executable": sys.executable,
        "os": platform.platform(),
        "packages": modules,
        "torch": torch_info,
        "models": {
            "generator": config.get("generator", {}).get("model_name"),
            "generator_revision": config.get("generator", {}).get("revision"),
            "selector": config.get("selector", {}).get("model_name"),
            "selector_revision": config.get("selector", {}).get("revision"),
        },
        "trust_remote_code": bool(
            config.get("generator", {}).get("trust_remote_code", False)
        ),
        "paths": {
            "output_root": str(output_root),
            "output_writable": writable_check(output_root),
            "cache_root": str(cache_root),
            "cache_writable": writable_check(cache_root),
            "disk_free_gib": round(shutil.disk_usage(output_root).free / 2**30, 2),
        },
        "scripts": {
            name: (PROJECT_ROOT / "scripts" / name).exists() for name in required_scripts
        },
    }
    java = shutil.which("java")
    java_version = None
    if java:
        completed = subprocess.run(
            [java, "-version"], text=True, capture_output=True, check=False
        )
        java_version = (completed.stderr or completed.stdout).splitlines()[0]
    java_21 = bool(java_version and ('"21.' in java_version or '"21"' in java_version))
    result["retrieval"] = {
        "backend": config.get("retrieval", {}).get("backend"),
        "pyserini": modules.get("pyserini"),
        "java": java,
        "java_version": java_version,
        "java_21": java_21,
        "ready": bool(modules.get("pyserini") and java_21),
    }
    result["ready_for_cpu_checks"] = (
        all(modules[name] for name in ("torch", "transformers", "accelerate", "numpy", "yaml"))
        and result["paths"]["output_writable"]
        and result["paths"]["cache_writable"]
        and all(result["scripts"].values())
        and (
            config.get("retrieval", {}).get("backend") != "pyserini_lucene"
            or result["retrieval"]["ready"]
        )
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.json_output:
        target = Path(args.json_output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")
    if not result["ready_for_cpu_checks"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.formal_matrix import (
    build_execution_plan,
    execute_plan,
    load_matrix_config,
    render_commands,
)
from src.io_utils import load_yaml
from src.run_manifest import atomic_write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or sequentially execute a formal-v2 experiment matrix."
    )
    parser.add_argument("--config", required=True, help="Formal matrix YAML config.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Plan only; run no workers.")
    mode.add_argument("--execute", action="store_true", help="Run workers in DAG order.")
    parser.add_argument("--plan-output", required=True)
    parser.add_argument("--commands-output")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--server-project-root", default="/root/rag-cbwdm")
    parser.add_argument("--artifact-root")
    parser.add_argument("--server-python", default="python")
    parser.add_argument("--generator-id", action="append", dest="generator_ids")
    parser.add_argument("--method", action="append", dest="methods")
    parser.add_argument("--learned-seed", action="append", type=int, dest="learned_seeds")
    parser.add_argument("--held-out", action="store_true")
    parser.add_argument("--held-out-freeze")
    return parser.parse_args(argv)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _apply_subsets(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    selected = copy.deepcopy(config)
    if args.generator_ids:
        requested = set(args.generator_ids)
        available = {
            str(spec.get("generator_id")): spec
            for spec in selected.get("generators", [])
        }
        unknown = requested - set(available)
        if unknown:
            raise ValueError(f"Unknown generator IDs: {sorted(unknown)}")
        selected["generators"] = [available[item] for item in args.generator_ids]
    if args.methods:
        selected["methods"] = list(args.methods)
    if args.learned_seeds:
        selected["learned_seeds"] = list(args.learned_seeds)
    return selected


def run_cli(
    argv: list[str] | None = None,
    *,
    execute: Any = execute_plan,
) -> dict[str, Any]:
    args = parse_args(argv)
    project_root = Path(args.project_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = _apply_subsets(load_matrix_config(config_path), args)
    freeze = None
    if args.held_out_freeze:
        freeze_path = Path(args.held_out_freeze)
        if not freeze_path.is_absolute():
            freeze_path = project_root / freeze_path
        freeze = load_yaml(freeze_path)
    plan = build_execution_plan(
        config,
        project_root=project_root,
        server_project_root=args.server_project_root,
        artifact_root=args.artifact_root,
        server_python=args.server_python,
        held_out=args.held_out,
        held_out_freeze=freeze,
    )
    plan_output = Path(args.plan_output)
    if not plan_output.is_absolute():
        plan_output = project_root / plan_output
    atomic_write_json(plan_output, plan)
    rendered = render_commands(plan)
    if args.commands_output:
        commands_output = Path(args.commands_output)
        if not commands_output.is_absolute():
            commands_output = project_root / commands_output
        _atomic_write_text(commands_output, rendered)
    if args.dry_run:
        print(rendered, end="")
    else:
        execute(plan)
    return plan


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.artifact_binding import GENERATOR_MANIFEST_SCHEMA_VERSION
from src.experiment_identity import (
    build_generator_identity,
    generator_artifact_sha256,
    resolve_dataset_identity,
)
from src.io_utils import load_yaml, require_keys
from src.prompts import classification_prompt_hash, classification_prompt_version
from src.run_manifest import (
    atomic_write_json,
    environment_info,
    git_state,
    sha256_file,
    stable_hash,
    utc_now,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish an atomic formal-v2 generator identity manifest."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--generator-id", required=True)
    parser.add_argument("--dataset-id")
    parser.add_argument("--model-family")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_manifest(
    *,
    config_path: Path,
    generator_id: str,
    dataset_id: str | None,
    model_family: str | None,
) -> dict:
    config = load_yaml(config_path)
    require_keys(config, ["dataset", "task", "generator"], "config")
    require_keys(config["task"], ["labels", "verbalizers"], "config.task")
    require_keys(config["generator"], ["model_name"], "config.generator")
    dataset = resolve_dataset_identity(
        config["dataset"], explicit_dataset_id=dataset_id
    )
    generator = config["generator"]
    model_name = str(generator["model_name"])
    revision = generator.get("revision")
    tokenizer_name = str(generator.get("tokenizer_name") or model_name)
    tokenizer_revision = generator.get("tokenizer_revision") or revision
    labels = list(config["task"]["labels"])
    verbalizers = dict(config["task"]["verbalizers"])
    identity = build_generator_identity(
        model_name_or_path=model_name,
        generator_id=generator_id,
        model_family=model_family,
        formal_v2=True,
        model_revision=revision,
        tokenizer_name_or_path=tokenizer_name,
        tokenizer_revision=tokenizer_revision,
        model_sha256=generator_artifact_sha256(model_name, revision),
        tokenizer_sha256=generator_artifact_sha256(
            tokenizer_name, tokenizer_revision
        ),
        prompt_template_version=classification_prompt_version(config["dataset"]),
        prompt_template_hash=classification_prompt_hash(
            config["dataset"], labels, verbalizers
        ),
        verbalizer_hash=stable_hash(verbalizers),
        dtype=generator.get("dtype", "auto"),
        device_map=generator.get("device_map", "auto"),
        trust_remote_code=bool(generator.get("trust_remote_code", False)),
        max_context_tokens=generator.get("max_context_tokens"),
    ).to_dict()
    return {
        "schema_version": GENERATOR_MANIFEST_SCHEMA_VERSION,
        "status": "completed",
        "dataset_identity": dataset.to_dict(),
        "generator_identity": identity,
        "generator_identity_fingerprint": stable_hash(identity),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "git": git_state(PROJECT_ROOT),
        "environment": environment_info(),
        "created_at": utc_now(),
    }


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config).resolve()
    output = resolve(args.output).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Generator manifest exists: {output}. Use --overwrite explicitly."
        )
    payload = build_manifest(
        config_path=config_path,
        generator_id=args.generator_id,
        dataset_id=args.dataset_id,
        model_family=args.model_family,
    )
    atomic_write_json(output, payload)
    print(
        f"[generator_manifest] generator_id={args.generator_id} output={output}"
    )


if __name__ == "__main__":
    main()

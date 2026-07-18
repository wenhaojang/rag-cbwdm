"""Freeze and validate immutable FEVER formal experiment configurations."""

from __future__ import annotations

import copy
import difflib
import json
from pathlib import Path
from typing import Any, Iterable

from src.formal_provenance import artifact_identity, atomic_write_text, sha256_path
from src.formal_splits import SPLIT_NAMES, validate_split_manifest
from src.io_utils import load_yaml
from src.run_manifest import (
    atomic_write_json,
    environment_info,
    git_state,
    sha256_file,
    stable_hash,
    utc_now,
)

FROZEN_SCHEMA_VERSION = "rag_cbwdm_fever_formal_config.v1"
REQUIRED_MODELS = ("generator", "tokenizer", "bge", "infogain", "rag_cbwdm")
CRITICAL_OVERRIDE_FLAGS = {
    "--config",
    "--seed",
    "--train-limit",
    "--dev-limit",
    "--limit",
    "--generator-model",
    "--selector-model",
    "--bge-model",
    "--infogain-model",
    "--split-manifest",
    "--stop-threshold",
    "--score-threshold",
    "--top-m",
    "--min-docs",
    "--beta",
    "--gamma",
    "--model-revision",
}


def reject_critical_cli_overrides(argv: Iterable[str]) -> None:
    """Reject mutable scientific settings on a frozen formal invocation."""
    forbidden = []
    for token in argv:
        flag = token.split("=", 1)[0]
        if flag in CRITICAL_OVERRIDE_FLAGS:
            forbidden.append(flag)
    if forbidden:
        raise ValueError(
            "Frozen formal runs forbid critical CLI overrides: "
            + ", ".join(sorted(set(forbidden)))
        )


def _require_mapping(root: dict[str, Any], dotted: str) -> Any:
    value: Any = root
    for component in dotted.split("."):
        if not isinstance(value, dict) or component not in value:
            raise ValueError(f"Cannot freeze formal config: missing critical field {dotted}")
        value = value[component]
    if value is None:
        raise ValueError(f"Cannot freeze formal config: critical field {dotted} is null")
    return value


def _validate_base_config(config: dict[str, Any]) -> None:
    for key in (
        "dataset",
        "task.labels",
        "task.verbalizers",
        "retrieval.top_n",
        "retrieval.bm25.k1",
        "retrieval.bm25.b",
        "generator.max_context_tokens",
        "generator.posterior_batch_size",
        "selector.max_length",
        "baselines.infogain_fever.max_length",
    ):
        _require_mapping(config, key)
    if config["dataset"] != "fever2":
        raise ValueError("Formal config freezing currently supports dataset=fever2 only")
    if config["task"]["labels"] != ["SUPPORTS", "REFUTES"]:
        raise ValueError("Frozen FEVER-2 label order must be [SUPPORTS, REFUTES]")


def _load_completed_calibration(path: Path, split_manifest_sha: str) -> dict[str, Any]:
    calibration = json.loads(path.read_text(encoding="utf-8"))
    if calibration.get("status") != "completed":
        raise ValueError("Cannot freeze formal config: calibration is not completed")
    if calibration.get("held_out_test_used") is not False:
        raise ValueError("Cannot freeze formal config: calibration used held_out_test")
    if calibration.get("split_manifest_sha256") != split_manifest_sha:
        raise ValueError("Calibration references a different split manifest SHA")
    for method in ("infogain_fever", "rag_cbwdm"):
        selected = calibration.get("selected", {}).get(method, {})
        if selected.get("status") != "selected" or not isinstance(
            selected.get("parameters"), dict
        ):
            raise ValueError(f"Cannot freeze formal config: missing {method} calibration")
    return calibration


def build_frozen_config(
    base_config: dict[str, Any],
    *,
    split_manifest: dict[str, Any],
    split_manifest_path: Path,
    calibration_manifest: dict[str, Any],
    calibration_manifest_path: Path,
    model_artifacts: dict[str, dict[str, Any]],
    corpus_artifact: dict[str, Any],
    index_artifact: dict[str, Any],
    project_root: str | Path,
) -> dict[str, Any]:
    _validate_base_config(base_config)
    missing = [name for name in REQUIRED_MODELS if name not in model_artifacts]
    if missing:
        raise ValueError(
            "Cannot freeze formal config: missing model artifacts " + ", ".join(missing)
        )
    for name, identity in model_artifacts.items():
        if not identity.get("revision"):
            raise ValueError(f"Cannot freeze formal config: {name} revision is missing")
        if not identity.get("sha256"):
            raise ValueError(f"Cannot freeze formal config: {name} SHA is missing")
    for name, identity in (
        ("corpus", corpus_artifact),
        ("lucene_index", index_artifact),
    ):
        if identity.get("manifest_status") not in {None, "completed"}:
            raise ValueError(f"Cannot freeze formal config: {name} manifest is incomplete")
        if identity.get("manifest_status") == "completed" and not identity.get(
            "manifest_fingerprint"
        ):
            raise ValueError(
                f"Cannot freeze formal config: {name} manifest lacks a fingerprint"
            )

    frozen = copy.deepcopy(base_config)
    frozen["profile"] = "server_formal_frozen"
    frozen["profile_limits"] = {
        "train_core": None,
        "validation": None,
        "held_out_test": None,
        "seeds": list(base_config.get("profile_limits", {}).get("seeds", [13, 21, 42])),
    }
    frozen["formal_protocol"] = {
        "schema_version": FROZEN_SCHEMA_VERSION,
        "automatic_calibration": False,
        "critical_cli_overrides_allowed": False,
        "split_manifest": {
            "path": str(split_manifest_path.resolve()),
            "sha256": sha256_file(split_manifest_path),
            "fingerprint": split_manifest["fingerprint"],
        },
        "splits": {
            role: {
                key: split_manifest["splits"][role][key]
                for key in ("path", "sha256", "num_rows", "id_sha256")
            }
            for role in SPLIT_NAMES
        },
        "calibration_manifest": {
            "path": str(calibration_manifest_path.resolve()),
            "sha256": sha256_file(calibration_manifest_path),
            "fingerprint": calibration_manifest["fingerprint"],
            "validation_sha256": calibration_manifest["validation_sha256"],
        },
        "frozen_parameters": {
            method: calibration_manifest["selected"][method]["parameters"]
            for method in ("infogain_fever", "rag_cbwdm")
        },
        "corpus": corpus_artifact,
        "lucene_index": index_artifact,
        "models": model_artifacts,
        "prompt_hash": stable_hash(
            {
                "prompt_contract": "src.prompts.classification_prompt.v1",
                "labels": base_config["task"]["labels"],
            }
        ),
        "verbalizer_hash": stable_hash(base_config["task"]["verbalizers"]),
        "retrieval_contract": {
            "top_n": base_config["retrieval"]["top_n"],
            "bm25": base_config["retrieval"]["bm25"],
            "backend": base_config["retrieval"].get("backend"),
        },
        "context_and_truncation": {
            "generator_max_context_tokens": base_config["generator"]["max_context_tokens"],
            "selector_max_length": base_config["selector"]["max_length"],
            "infogain_max_length": base_config["baselines"]["infogain_fever"][
                "max_length"
            ],
        },
        "runtime": {
            "generator_batch_size": base_config["generator"]["posterior_batch_size"],
            "dtype": base_config["generator"].get("dtype"),
            "seed": int(base_config.get("seed", 13)),
        },
        "git": git_state(project_root),
        "environment": environment_info(),
        "created_at": utc_now(),
    }
    info_params = frozen["formal_protocol"]["frozen_parameters"]["infogain_fever"]
    cbwdm_params = frozen["formal_protocol"]["frozen_parameters"]["rag_cbwdm"]
    info_runtime_params = dict(info_params)
    info_runtime_params["inference_threshold"] = info_runtime_params.pop(
        "filter_threshold"
    )
    frozen["baselines"]["infogain_fever"].update(info_runtime_params)
    frozen["cbwdm"]["stop_threshold"] = cbwdm_params["stop_threshold"]
    frozen["cbwdm"]["top_m"] = cbwdm_params["top_m"]
    frozen["selector"].update(
        {
            key: cbwdm_params[key]
            for key in (
                "score_threshold",
                "min_docs",
                "top_m",
                "beta",
                "gamma",
                "b_plus",
                "b_minus",
            )
        }
    )
    frozen["generator"].update(
        {
            "model_name": model_artifacts["generator"]["path"],
            "revision": model_artifacts["generator"]["revision"],
            "weight_sha256": model_artifacts["generator"]["sha256"],
            "tokenizer_revision": model_artifacts["tokenizer"]["revision"],
            "tokenizer_sha256": model_artifacts["tokenizer"]["sha256"],
        }
    )
    frozen["baselines"]["bge"].update(
        {
            "model_name": model_artifacts["bge"]["path"],
            "revision": model_artifacts["bge"]["revision"],
            "weight_sha256": model_artifacts["bge"]["sha256"],
        }
    )
    frozen["baselines"]["infogain_fever"].update(
        {
            "model_name": model_artifacts["infogain"]["path"],
            "revision": model_artifacts["infogain"]["revision"],
            "weight_sha256": model_artifacts["infogain"]["sha256"],
        }
    )
    frozen["selector"].update(
        {
            "model_name": model_artifacts["rag_cbwdm"]["path"],
            "revision": model_artifacts["rag_cbwdm"]["revision"],
            "weight_sha256": model_artifacts["rag_cbwdm"]["sha256"],
            "parameter_status": "frozen_from_validation_calibration",
        }
    )
    return frozen


def publish_frozen_config(
    base_config_path: str | Path,
    split_manifest_path: str | Path,
    calibration_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    models: dict[str, str | Path],
    revisions: dict[str, str],
    corpus: str | Path,
    index: str | Path,
    overwrite: bool = False,
    project_root: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    base_path = Path(base_config_path).resolve()
    split_path = Path(split_manifest_path).resolve()
    calibration_path = Path(calibration_manifest_path).resolve()
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    base = load_yaml(base_path)
    split = validate_split_manifest(split_path)
    calibration = _load_completed_calibration(calibration_path, sha256_file(split_path))
    model_artifacts = {
        name: artifact_identity(path, revision=revisions.get(name))
        for name, path in models.items()
    }
    frozen = build_frozen_config(
        base,
        split_manifest=split,
        split_manifest_path=split_path,
        calibration_manifest=calibration,
        calibration_manifest_path=calibration_path,
        model_artifacts=model_artifacts,
        corpus_artifact=artifact_identity(corpus),
        index_artifact=artifact_identity(index),
        project_root=root,
    )
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to freeze formal config") from exc
    fingerprint_payload = copy.deepcopy(frozen)
    fingerprint_payload["formal_protocol"].pop("created_at", None)
    fingerprint_payload["formal_protocol"].pop("environment", None)
    fingerprint = stable_hash(fingerprint_payload)
    frozen["formal_protocol"]["fingerprint"] = fingerprint
    rendered = yaml.safe_dump(frozen, sort_keys=False, allow_unicode=True)
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"fever2_formal_frozen_{fingerprint[:16]}"
    config_path = directory / f"{stem}.yaml"
    manifest_path = directory / f"{stem}.manifest.json"
    diff_path = directory / f"{stem}.diff.md"
    existing = [path for path in (config_path, manifest_path, diff_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Frozen output exists: {existing[0]}")
    atomic_write_text(config_path, rendered)
    base_text = base_path.read_text(encoding="utf-8").splitlines()
    frozen_text = rendered.splitlines()
    diff = "\n".join(
        difflib.unified_diff(
            base_text,
            frozen_text,
            fromfile=str(base_path),
            tofile=str(config_path),
            lineterm="",
        )
    )
    atomic_write_text(
        diff_path,
        "# Frozen FEVER Formal Config Diff\n\n```diff\n" + diff + "\n```\n",
    )
    manifest = {
        "schema_version": FROZEN_SCHEMA_VERSION,
        "status": "completed",
        "fingerprint": fingerprint,
        "frozen_config_path": str(config_path),
        "frozen_config_sha256": sha256_file(config_path),
        "base_config_path": str(base_path),
        "base_config_sha256": sha256_file(base_path),
        "split_manifest_path": str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "calibration_manifest_path": str(calibration_path),
        "calibration_manifest_sha256": sha256_file(calibration_path),
        "models": model_artifacts,
        "corpus": frozen["formal_protocol"]["corpus"],
        "lucene_index": frozen["formal_protocol"]["lucene_index"],
        "prompt_hash": frozen["formal_protocol"]["prompt_hash"],
        "verbalizer_hash": frozen["formal_protocol"]["verbalizer_hash"],
        "git": frozen["formal_protocol"]["git"],
        "environment": frozen["formal_protocol"]["environment"],
        "diff_path": str(diff_path),
        "diff_sha256": sha256_file(diff_path),
        "created_at": frozen["formal_protocol"]["created_at"],
    }
    atomic_write_json(manifest_path, manifest)
    return config_path, manifest_path, diff_path


def validate_frozen_manifest(path: str | Path, *, verify_artifacts: bool = True) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != FROZEN_SCHEMA_VERSION or manifest.get(
        "status"
    ) != "completed":
        raise ValueError("Frozen formal manifest is not completed")
    config_path = Path(str(manifest.get("frozen_config_path", "")))
    if not config_path.is_file() or sha256_file(config_path) != manifest.get(
        "frozen_config_sha256"
    ):
        raise ValueError("Frozen config SHA changed")
    if verify_artifacts:
        for name, identity in manifest.get("models", {}).items():
            if sha256_path(identity["path"]) != identity.get("sha256"):
                raise ValueError(f"Frozen model SHA changed: {name}")
        for name in ("corpus", "lucene_index"):
            identity = manifest.get(name, {})
            if sha256_path(identity["path"]) != identity.get("sha256"):
                raise ValueError(f"Frozen {name} SHA changed")
    return manifest

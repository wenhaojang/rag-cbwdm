"""Formal-v2 provenance bindings for learned and evaluated artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from src.experiment_identity import (
    FORMAL_V2_MODE,
    GENERATOR_IDENTITY_SCHEMA_VERSION,
    generator_artifact_sha256,
)
from src.formal_provenance import sha256_path
from src.run_manifest import sha256_file, stable_hash


BINDING_SCHEMA_VERSION = "rag_cbwdm_artifact_binding.v1"
GENERATOR_MANIFEST_SCHEMA_VERSION = "rag_cbwdm_generator_manifest.v1"
SELECTION_MANIFEST_SCHEMA_VERSION = "rag_cbwdm_selection_manifest.v2"
EVALUATION_MANIFEST_SCHEMA_VERSION = "rag_cbwdm_evaluation_manifest.v2"
EVALUATION_BINDING_SCHEMA_VERSION = "rag_cbwdm_evaluation_binding.v1"

GENERATOR_DEPENDENCY_CONDITIONED = "conditioned"
GENERATOR_DEPENDENCY_INDEPENDENT = "independent"
MATCHED_MAIN = "matched_main"
CROSS_GENERATOR_TRANSFER = "cross_generator_transfer"


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _resolved_recorded_path(value: Any, manifest_path: Path, field: str) -> Path:
    if not value:
        raise ValueError(f"Manifest is missing {field}")
    path = Path(str(value))
    return (
        path.resolve()
        if path.is_absolute()
        else (manifest_path.parent / path).resolve()
    )


def _expect(value: Any, expected: Any, label: str) -> None:
    if expected is not None and value != expected:
        raise ValueError(f"{label} mismatch: expected={expected!r} actual={value!r}")


def validate_formal_teacher_binding(
    teacher_path: str | Path,
    teacher_manifest_path: str | Path,
    *,
    method: str,
    expected_dataset_id: str | None = None,
    expected_conditioning_generator_id: str | None = None,
) -> dict[str, Any]:
    """Validate a Phase-A formal-v2 teacher and return its compact binding."""
    teacher = Path(teacher_path).resolve()
    manifest_path = Path(teacher_manifest_path).resolve()
    manifest = _load_object(manifest_path, "teacher manifest")
    if manifest.get("status") != "completed" or manifest.get("completed") is not True:
        raise ValueError("Formal-v2 training requires a completed teacher manifest")
    if manifest.get("identity_mode") != FORMAL_V2_MODE:
        raise ValueError("Formal-v2 training requires a formal-v2 teacher")
    manifest_method = manifest.get("method")
    if manifest_method is None and manifest.get("stage") == "build_infogain_teacher":
        manifest_method = "infogain_fever"
    _expect(manifest_method, method, "teacher method")
    teacher_sha = sha256_file(teacher)
    recorded_sha = manifest.get("teacher_sha256", manifest.get("output_sha256"))
    if recorded_sha != teacher_sha:
        raise ValueError("Teacher SHA does not match its manifest")
    posterior = manifest.get("posterior_binding")
    if not isinstance(posterior, dict):
        raise ValueError("Formal-v2 teacher lacks posterior_binding")
    dataset = posterior.get("dataset_identity")
    generator = posterior.get("generator_identity")
    retrieval = posterior.get("retrieval_protocol_identity")
    if not all(isinstance(item, dict) for item in (dataset, generator, retrieval)):
        raise ValueError("Formal-v2 teacher has incomplete identity objects")
    dataset_id = dataset.get("dataset_id")
    generator_id = generator.get("generator_id")
    retrieval_protocol_id = retrieval.get("retrieval_protocol_id")
    if not dataset_id or not generator_id or not retrieval_protocol_id:
        raise ValueError("Formal-v2 teacher has ambiguous dataset/generator/retrieval identity")
    _expect(dataset_id, expected_dataset_id, "teacher dataset_id")
    _expect(
        generator_id,
        expected_conditioning_generator_id,
        "teacher conditioning_generator_id",
    )
    if manifest.get("generator_id") not in {None, generator_id}:
        raise ValueError("Teacher generator_id conflicts with posterior binding")
    posterior_required = (
        "posterior_sha256",
        "posterior_manifest_path",
        "posterior_manifest_sha256",
        "posterior_manifest_fingerprint",
        "posterior_identity_fingerprint",
    )
    missing = [field for field in posterior_required if not posterior.get(field)]
    if missing:
        raise ValueError(
            "Formal-v2 teacher posterior binding is missing: " + ", ".join(missing)
        )
    if not retrieval.get("source_artifact_sha256"):
        raise ValueError("Formal-v2 teacher lacks retrieval source SHA")
    if not manifest.get("fingerprint"):
        raise ValueError("Formal-v2 teacher lacks manifest fingerprint")
    teacher_contract = manifest.get("contract", manifest.get("provenance"))
    if manifest.get("fingerprint") != stable_hash(teacher_contract):
        raise ValueError("Teacher manifest fingerprint mismatch")
    posterior_manifest_path = _resolved_recorded_path(
        posterior["posterior_manifest_path"], manifest_path, "posterior_manifest_path"
    )
    if sha256_file(posterior_manifest_path) != posterior["posterior_manifest_sha256"]:
        raise ValueError("Upstream posterior manifest SHA mismatch")
    posterior_manifest = _load_object(posterior_manifest_path, "posterior manifest")
    if posterior_manifest.get("fingerprint") != posterior[
        "posterior_manifest_fingerprint"
    ]:
        raise ValueError("Upstream posterior manifest fingerprint mismatch")
    posterior_path = _resolved_recorded_path(
        posterior_manifest.get("output_path"), posterior_manifest_path, "output_path"
    )
    if sha256_file(posterior_path) != posterior["posterior_sha256"]:
        raise ValueError("Upstream posterior JSONL SHA mismatch")
    return {
        "schema_version": BINDING_SCHEMA_VERSION,
        "method": method,
        "generator_dependency": GENERATOR_DEPENDENCY_CONDITIONED,
        "dataset_id": dataset_id,
        "conditioning_generator_id": generator_id,
        "generator_identity_fingerprint": stable_hash(generator),
        "retrieval_protocol_id": retrieval_protocol_id,
        "retrieval_source_sha256": retrieval.get("source_artifact_sha256"),
        "posterior": {
            key: posterior[key]
            for key in posterior_required
        },
        "teacher": {
            "path": str(teacher),
            "sha256": teacher_sha,
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_fingerprint": manifest.get("fingerprint"),
        },
    }


def complete_training_binding(
    teacher_binding: Mapping[str, Any],
    *,
    seed: int,
    config_path: str | Path,
) -> dict[str, Any]:
    binding = dict(teacher_binding)
    config = Path(config_path).resolve()
    binding.update(
        {
            "seed": int(seed),
            "config_path": str(config),
            "config_sha256": sha256_file(config),
        }
    )
    return binding


def validate_formal_training_binding(
    training_manifest_path: str | Path,
    checkpoint_path: str | Path,
    *,
    method: str,
    expected_dataset_id: str | None = None,
    expected_conditioning_generator_id: str | None = None,
    expected_seed: int | None = None,
) -> dict[str, Any]:
    """Validate a learned formal-v2 checkpoint and return selection provenance."""
    manifest_path = Path(training_manifest_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    manifest = _load_object(manifest_path, "training manifest")
    if manifest.get("status") != "completed" or manifest.get("completed") is not True:
        raise ValueError("Formal-v2 selection requires completed training")
    if manifest.get("identity_mode") != FORMAL_V2_MODE:
        raise ValueError("Formal-v2 selection requires a formal-v2 training manifest")
    _expect(manifest.get("method"), method, "training method")
    binding = manifest.get("artifact_binding")
    if not isinstance(binding, dict):
        raise ValueError("Formal-v2 training manifest lacks artifact_binding")
    validate_binding_shape(binding, conditioned=True)
    if binding.get("method") != method:
        raise ValueError("Training binding method mismatch")
    if manifest.get("fingerprint") != stable_hash(manifest.get("contract")):
        raise ValueError("Training manifest fingerprint mismatch")
    if manifest.get("contract", {}).get("artifact_binding") != binding:
        raise ValueError("Training binding differs between manifest and contract")
    for field in (
        "dataset_id",
        "conditioning_generator_id",
        "generator_identity_fingerprint",
        "retrieval_protocol_id",
    ):
        if manifest.get(field) != binding.get(field):
            raise ValueError(f"Training manifest {field} conflicts with binding")
    config_path = _resolved_recorded_path(
        binding.get("config_path"), manifest_path, "artifact_binding.config_path"
    )
    if sha256_file(config_path) != binding.get("config_sha256"):
        raise ValueError("Training config SHA mismatch")
    git = manifest.get("git")
    if not isinstance(git, dict) or not git.get("commit"):
        raise ValueError("Formal-v2 training manifest lacks Git provenance")
    if not binding.get("retrieval_source_sha256"):
        raise ValueError("Training binding lacks retrieval source SHA")
    if not isinstance(binding.get("posterior"), dict) or not isinstance(
        binding.get("teacher"), dict
    ):
        raise ValueError("Training binding lacks teacher/posterior provenance")
    _expect(binding.get("dataset_id"), expected_dataset_id, "training dataset_id")
    _expect(
        binding.get("conditioning_generator_id"),
        expected_conditioning_generator_id,
        "training conditioning_generator_id",
    )
    _expect(binding.get("seed"), expected_seed, "training seed")
    recorded_checkpoint = _resolved_recorded_path(
        manifest.get("checkpoint_path", manifest.get("checkpoint")),
        manifest_path,
        "checkpoint_path",
    )
    if recorded_checkpoint != checkpoint:
        raise ValueError("Training manifest checkpoint path mismatch")
    actual_checkpoint_sha = sha256_path(checkpoint)
    if manifest.get("checkpoint_sha256") != actual_checkpoint_sha:
        raise ValueError("Checkpoint SHA does not match training manifest")
    checkpoint_fingerprint = manifest.get("checkpoint_fingerprint")
    expected_fingerprint = stable_hash(
        {
            "contract": manifest.get("fingerprint"),
            "checkpoint_sha256": actual_checkpoint_sha,
        }
    )
    if checkpoint_fingerprint != expected_fingerprint:
        raise ValueError("Checkpoint fingerprint mismatch")
    return {
        "schema_version": BINDING_SCHEMA_VERSION,
        "method": method,
        "generator_dependency": GENERATOR_DEPENDENCY_CONDITIONED,
        "dataset_id": binding["dataset_id"],
        "conditioning_generator_id": binding["conditioning_generator_id"],
        "generator_identity_fingerprint": binding[
            "generator_identity_fingerprint"
        ],
        "retrieval_protocol_id": binding["retrieval_protocol_id"],
        "retrieval_source_sha256": binding["retrieval_source_sha256"],
        "seed": binding["seed"],
        "training_manifest_path": str(manifest_path),
        "training_manifest_sha256": sha256_file(manifest_path),
        "training_manifest_fingerprint": manifest.get("fingerprint"),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": actual_checkpoint_sha,
        "checkpoint_fingerprint": checkpoint_fingerprint,
    }


def conditioned_selection_binding(
    training_binding: Mapping[str, Any],
) -> dict[str, Any]:
    binding = dict(training_binding)
    validate_binding_shape(binding, conditioned=True)
    return binding


def independent_selection_binding(
    *,
    dataset_id: str,
    retrieval_protocol_id: str,
    method: str,
    source_artifact_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": BINDING_SCHEMA_VERSION,
        "method": method,
        "generator_dependency": GENERATOR_DEPENDENCY_INDEPENDENT,
        "dataset_id": dataset_id,
        "conditioning_generator_id": None,
        "generator_identity_fingerprint": None,
        "retrieval_protocol_id": retrieval_protocol_id,
        "source_artifact_sha256": source_artifact_sha256,
    }


def validate_binding_shape(binding: Mapping[str, Any], *, conditioned: bool) -> None:
    if binding.get("schema_version") != BINDING_SCHEMA_VERSION:
        raise ValueError("Artifact binding schema version mismatch")
    if not binding.get("dataset_id") or not binding.get("method"):
        raise ValueError("Artifact binding lacks dataset or method identity")
    if not binding.get("retrieval_protocol_id"):
        raise ValueError("Artifact binding lacks retrieval_protocol_id")
    expected_dependency = (
        GENERATOR_DEPENDENCY_CONDITIONED
        if conditioned
        else GENERATOR_DEPENDENCY_INDEPENDENT
    )
    if binding.get("generator_dependency") != expected_dependency:
        raise ValueError("Artifact generator_dependency mismatch")
    if conditioned:
        required = (
            "conditioning_generator_id",
            "generator_identity_fingerprint",
            "seed",
        )
        missing = [field for field in required if binding.get(field) is None]
        if missing:
            raise ValueError(
                "Conditioned artifact binding is missing: " + ", ".join(missing)
            )
    elif binding.get("conditioning_generator_id") is not None:
        raise ValueError("Independent selection cannot have conditioning_generator_id")
    if not conditioned and not binding.get("source_artifact_sha256"):
        raise ValueError("Independent selection lacks retrieval source SHA")


def validate_selection_provenance(
    selection_path: str | Path,
    selection_manifest_path: str | Path | None = None,
    *,
    formal_v2: bool = True,
    expected_dataset_id: str | None = None,
    expected_method: str | None = None,
) -> dict[str, Any]:
    selection = Path(selection_path).resolve()
    manifest_path = (
        Path(selection_manifest_path).resolve()
        if selection_manifest_path
        else selection.with_suffix(".manifest.json")
    )
    manifest = _load_object(manifest_path, "selection manifest")
    if manifest.get("status") != "completed" or manifest.get("completed") is not True:
        raise ValueError("Selection manifest is not completed")
    recorded = _resolved_recorded_path(
        manifest.get("output_path"), manifest_path, "output_path"
    )
    if recorded != selection:
        raise ValueError("Selection manifest output path mismatch")
    selection_sha = sha256_file(selection)
    if manifest.get("output_sha256") != selection_sha:
        raise ValueError("Selection SHA does not match its manifest")
    _expect(manifest.get("method"), expected_method, "selection method")
    if not formal_v2:
        return {
            "manifest": manifest,
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "selection_sha256": selection_sha,
            "artifact_binding": manifest.get("artifact_binding"),
        }
    if manifest.get("schema_version") != SELECTION_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Formal-v2 requires selection manifest schema v2")
    binding = manifest.get("artifact_binding")
    if not isinstance(binding, dict):
        raise ValueError("Formal-v2 selection lacks artifact_binding")
    dependency = binding.get("generator_dependency")
    if dependency == GENERATOR_DEPENDENCY_CONDITIONED:
        validate_binding_shape(binding, conditioned=True)
        checkpoint_required = (
            "training_manifest_path",
            "training_manifest_sha256",
            "training_manifest_fingerprint",
            "checkpoint_path",
            "checkpoint_sha256",
            "checkpoint_fingerprint",
        )
        missing = [field for field in checkpoint_required if not binding.get(field)]
        if missing:
            raise ValueError(
                "Conditioned selection lacks checkpoint provenance: "
                + ", ".join(missing)
            )
        current_binding = validate_formal_training_binding(
            binding["training_manifest_path"],
            binding["checkpoint_path"],
            method=binding["method"],
            expected_dataset_id=binding["dataset_id"],
            expected_conditioning_generator_id=binding[
                "conditioning_generator_id"
            ],
            expected_seed=binding["seed"],
        )
        if current_binding != binding:
            raise ValueError(
                "Selection checkpoint provenance differs from current training artifacts"
            )
    elif dependency == GENERATOR_DEPENDENCY_INDEPENDENT:
        validate_binding_shape(binding, conditioned=False)
    else:
        raise ValueError("Formal-v2 selection has unknown generator_dependency")
    if manifest.get("contract", {}).get("artifact_binding") != binding:
        raise ValueError("Selection binding differs between manifest and contract")
    if manifest.get("fingerprint") != stable_hash(manifest.get("contract")):
        raise ValueError("Selection contract fingerprint mismatch")
    _expect(binding.get("dataset_id"), expected_dataset_id, "selection dataset_id")
    _expect(binding.get("method"), expected_method, "selection binding method")
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_fingerprint": manifest.get("fingerprint"),
        "selection_path": str(selection),
        "selection_sha256": selection_sha,
        "artifact_binding": binding,
    }


def validate_generator_manifest(
    generator_manifest_path: str | Path,
    *,
    expected_dataset_id: str | None = None,
    expected_generator_id: str | None = None,
    expected_config_path: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(generator_manifest_path).resolve()
    manifest = _load_object(path, "generator manifest")
    if manifest.get("schema_version") != GENERATOR_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Generator manifest schema version mismatch")
    if manifest.get("status") != "completed":
        raise ValueError("Generator manifest is not completed")
    git = manifest.get("git")
    if not isinstance(git, dict) or not git.get("commit"):
        raise ValueError("Formal generator manifest lacks Git provenance")
    dataset = manifest.get("dataset_identity")
    generator = manifest.get("generator_identity")
    if not isinstance(dataset, dict) or not isinstance(generator, dict):
        raise ValueError("Generator manifest lacks structured identities")
    if generator.get("schema_version") != GENERATOR_IDENTITY_SCHEMA_VERSION:
        raise ValueError("Generator identity schema version mismatch")
    if generator.get("identity_source") != "explicit" or not generator.get(
        "generator_id"
    ):
        raise ValueError("Formal generator manifest requires explicit generator_id")
    required_generator_fields = (
        "model_name_or_path",
        "model_sha256",
        "tokenizer_name_or_path",
        "tokenizer_sha256",
        "prompt_template_version",
        "prompt_template_hash",
        "verbalizer_hash",
    )
    missing = [field for field in required_generator_fields if generator.get(field) is None]
    if missing:
        raise ValueError("Generator identity is incomplete: " + ", ".join(missing))
    if manifest.get("generator_identity_fingerprint") != stable_hash(generator):
        raise ValueError("Generator identity fingerprint mismatch")
    _expect(dataset.get("dataset_id"), expected_dataset_id, "generator dataset_id")
    _expect(
        generator.get("generator_id"),
        expected_generator_id,
        "evaluation generator_id",
    )
    config_path = _resolved_recorded_path(
        manifest.get("config_path"), path, "config_path"
    )
    if expected_config_path is not None and config_path != Path(
        expected_config_path
    ).resolve():
        raise ValueError("Generator manifest config path mismatch")
    if manifest.get("config_sha256") != sha256_file(config_path):
        raise ValueError("Generator manifest config SHA mismatch")
    actual_model_sha = generator_artifact_sha256(
        generator["model_name_or_path"], generator.get("model_revision")
    )
    if generator.get("model_sha256") != actual_model_sha:
        raise ValueError("Generator model SHA/revision identity mismatch")
    actual_tokenizer_sha = generator_artifact_sha256(
        generator["tokenizer_name_or_path"], generator.get("tokenizer_revision")
    )
    if generator.get("tokenizer_sha256") != actual_tokenizer_sha:
        raise ValueError("Generator tokenizer SHA/revision identity mismatch")
    return {
        "manifest": manifest,
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
        "dataset_identity": dataset,
        "generator_identity": generator,
        "generator_identity_fingerprint": manifest[
            "generator_identity_fingerprint"
        ],
    }


def validate_generator_identity_against_config(
    generator_identity: Mapping[str, Any], generator_config: Mapping[str, Any]
) -> None:
    """Reject generator manifests assembled from incompatible config fragments."""
    expected = {
        "model_name_or_path": str(generator_config["model_name"]),
        "model_revision": generator_config.get("revision"),
        "tokenizer_name_or_path": str(
            generator_config.get("tokenizer_name") or generator_config["model_name"]
        ),
        "tokenizer_revision": generator_config.get("tokenizer_revision")
        or generator_config.get("revision"),
        "dtype": generator_config.get("dtype", "auto"),
        "device_map": generator_config.get("device_map", "auto"),
        "trust_remote_code": bool(generator_config.get("trust_remote_code", False)),
        "max_context_tokens": generator_config.get("max_context_tokens"),
    }
    for field, value in expected.items():
        if generator_identity.get(field) != value:
            raise ValueError(
                f"Generator manifest {field} differs from its frozen config"
            )


def build_evaluation_binding(
    selection: Mapping[str, Any],
    generator: Mapping[str, Any],
    *,
    experiment_type: str = MATCHED_MAIN,
) -> dict[str, Any]:
    selection_binding = selection["artifact_binding"]
    generator_identity = generator["generator_identity"]
    if selection_binding["dataset_id"] != generator["dataset_identity"]["dataset_id"]:
        raise ValueError("Selection and evaluation generator dataset_id mismatch")
    dependency = selection_binding["generator_dependency"]
    conditioning_id = selection_binding.get("conditioning_generator_id")
    evaluation_id = generator_identity["generator_id"]
    if experiment_type not in {MATCHED_MAIN, CROSS_GENERATOR_TRANSFER}:
        raise ValueError(f"Unknown evaluation experiment_type: {experiment_type!r}")
    if dependency == GENERATOR_DEPENDENCY_CONDITIONED:
        if experiment_type == MATCHED_MAIN and conditioning_id != evaluation_id:
            raise ValueError(
                "matched_main requires conditioning_generator_id == "
                "evaluation_generator_id"
            )
        if experiment_type == CROSS_GENERATOR_TRANSFER and conditioning_id == evaluation_id:
            raise ValueError(
                "cross_generator_transfer requires distinct conditioning and "
                "evaluation generators"
            )
    elif dependency == GENERATOR_DEPENDENCY_INDEPENDENT:
        if conditioning_id is not None:
            raise ValueError("Independent selection cannot carry a conditioning generator")
        if experiment_type != MATCHED_MAIN:
            raise ValueError("Generator-independent selection is not a transfer experiment")
    else:
        raise ValueError("Unknown selection generator_dependency")
    return {
        "schema_version": EVALUATION_BINDING_SCHEMA_VERSION,
        "dataset_id": selection_binding["dataset_id"],
        "method": selection_binding["method"],
        "generator_dependency": dependency,
        "conditioning_generator_id": conditioning_id,
        "evaluation_generator_id": evaluation_id,
        "experiment_type": experiment_type,
        "evaluation_generator_identity_fingerprint": generator[
            "generator_identity_fingerprint"
        ],
        "retrieval_protocol_id": selection_binding["retrieval_protocol_id"],
        "selection_path": selection["selection_path"],
        "selection_sha256": selection["selection_sha256"],
        "selection_manifest_path": selection["manifest_path"],
        "selection_manifest_sha256": selection["manifest_sha256"],
        "selection_manifest_fingerprint": selection["manifest_fingerprint"],
        "generator_manifest_path": generator["manifest_path"],
        "generator_manifest_sha256": generator["manifest_sha256"],
    }

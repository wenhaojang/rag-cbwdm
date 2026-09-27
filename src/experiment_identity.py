"""Versioned experiment identities and posterior provenance validation.

The identity objects in this module separate scientific experiment identity
from deployment-specific paths.  Formal-v2 validation is deliberately strict;
legacy validation preserves access to historical posterior manifests without
pretending that they have stable dataset or generator identities.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from src.run_manifest import sha256_file, stable_hash


EXPERIMENT_IDENTITY_SCHEMA_VERSION = "rag_cbwdm_experiment_identity.v1"
DATASET_IDENTITY_SCHEMA_VERSION = "rag_cbwdm_dataset_identity.v1"
GENERATOR_IDENTITY_SCHEMA_VERSION = "rag_cbwdm_generator_identity.v1"
RETRIEVAL_IDENTITY_SCHEMA_VERSION = "rag_cbwdm_retrieval_protocol_identity.v1"
POSTERIOR_MANIFEST_SCHEMA_VERSION = "rag_cbwdm_posterior_manifest.v2"

FORMAL_V2_MODE = "formal_v2"
LEGACY_MODE = "legacy"
LEGACY_COMPATIBLE_MODE = "legacy_compatible"

DATASET_ALIASES = {
    "fever2": "fever_binary_v2",
    "fever_binary_v2": "fever_binary_v2",
    "fever3": "fever3",
    "fm2": "fm2_official_closed_page_v1",
    "fm2_official_closed_page_v1": "fm2_official_closed_page_v1",
}

_DATASET_FAMILIES = {
    "fever_binary_v2": "fever",
    "fever3": "fever",
    "fm2_official_closed_page_v1": "fm2",
}

_KNOWN_GENERATORS = {
    "qwen2.5-1.5b-instruct": ("qwen2.5-1.5b-instruct", "qwen2.5"),
    "qwen2.5-7b-instruct": ("qwen2.5-7b-instruct", "qwen2.5"),
}

_STABLE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _validate_stable_id(value: str, field: str) -> str:
    normalized = str(value).strip()
    if not _STABLE_ID.fullmatch(normalized):
        raise ValueError(
            f"{field} must match {_STABLE_ID.pattern!r}; got {value!r}"
        )
    return normalized


@dataclass(frozen=True)
class DatasetIdentity:
    dataset_id: str
    dataset_family: str
    dataset_alias: str | None = None
    protocol_metadata: dict[str, Any] | None = None
    schema_version: str = DATASET_IDENTITY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GeneratorIdentity:
    generator_id: str | None
    model_family: str | None
    model_name_or_path: str
    model_revision: str | None = None
    resolved_model_revision: str | None = None
    tokenizer_name_or_path: str | None = None
    tokenizer_revision: str | None = None
    resolved_tokenizer_revision: str | None = None
    model_sha256: str | None = None
    tokenizer_sha256: str | None = None
    prompt_template_version: str | None = None
    prompt_template_hash: str | None = None
    verbalizer_hash: str | None = None
    dtype: str | None = None
    device_map: str | None = None
    trust_remote_code: bool = False
    identity_source: str = "unresolved_legacy"
    schema_version: str = GENERATOR_IDENTITY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalProtocolIdentity:
    retrieval_protocol_id: str | None
    dataset_id: str
    source_artifact_sha256: str
    retrieval_model_or_index_identity: dict[str, Any] | None = None
    schema_version: str = RETRIEVAL_IDENTITY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_dataset_identity(
    dataset_name: str,
    *,
    explicit_dataset_id: str | None = None,
    protocol_metadata: dict[str, Any] | None = None,
) -> DatasetIdentity:
    """Resolve a legacy config name to a stable dataset identity."""
    legacy_name = str(dataset_name).strip()
    resolved = DATASET_ALIASES.get(legacy_name.casefold())
    if explicit_dataset_id is not None:
        explicit = _validate_stable_id(explicit_dataset_id, "dataset_id")
        if resolved is not None and explicit != resolved:
            raise ValueError(
                f"dataset_id {explicit!r} conflicts with legacy dataset "
                f"{legacy_name!r} -> {resolved!r}"
            )
        resolved = explicit
    if resolved is None:
        raise ValueError(
            f"No stable dataset identity mapping for {legacy_name!r}; "
            "provide --dataset-id explicitly"
        )
    return DatasetIdentity(
        dataset_id=resolved,
        dataset_family=_DATASET_FAMILIES.get(resolved, legacy_name.casefold()),
        dataset_alias=legacy_name if legacy_name != resolved else None,
        protocol_metadata=protocol_metadata,
    )


def infer_known_generator_id(model_name_or_path: str) -> tuple[str, str] | None:
    """Infer only explicitly supported generator IDs, independent of path root."""
    leaf = str(model_name_or_path).replace("\\", "/").rstrip("/").split("/")[-1]
    return _KNOWN_GENERATORS.get(leaf.casefold())


def build_generator_identity(
    *,
    model_name_or_path: str,
    generator_id: str | None = None,
    model_family: str | None = None,
    formal_v2: bool = False,
    **fields: Any,
) -> GeneratorIdentity:
    """Build a generator identity while keeping logical ID separate from path."""
    inferred = infer_known_generator_id(model_name_or_path)
    if generator_id is not None:
        resolved_id = _validate_stable_id(generator_id, "generator_id")
        if inferred is not None and resolved_id != inferred[0]:
            raise ValueError(
                f"generator_id {resolved_id!r} conflicts with known model "
                f"{model_name_or_path!r} -> {inferred[0]!r}"
            )
        identity_source = "explicit"
    elif formal_v2:
        raise ValueError(
            "formal-v2 identity requires an explicit generator_id; "
            "model paths are not experiment identities"
        )
    elif inferred is not None:
        resolved_id = inferred[0]
        identity_source = "inferred_known_model"
    else:
        resolved_id = None
        identity_source = "unresolved_legacy"
    resolved_family = model_family or (inferred[1] if inferred is not None else None)
    return GeneratorIdentity(
        generator_id=resolved_id,
        model_family=resolved_family,
        model_name_or_path=str(model_name_or_path),
        identity_source=identity_source,
        **fields,
    )


def resolve_retrieval_protocol_identity(
    *,
    dataset_identity: DatasetIdentity,
    source_artifact_sha256: str,
    retrieval_method: str | None = None,
    retrieval_protocol_id: str | None = None,
    formal_v2: bool = False,
    retrieval_model_or_index_identity: dict[str, Any] | None = None,
) -> RetrievalProtocolIdentity:
    """Resolve the current FEVER BM25 and FM2 official-pool protocols."""
    resolved = retrieval_protocol_id
    if resolved is None:
        method = str(retrieval_method or "").casefold()
        if (
            dataset_identity.dataset_id == "fm2_official_closed_page_v1"
            and method in {"", "fm2_official_closed_page_v1"}
        ):
            resolved = "fm2_official_closed_page_v1"
        elif dataset_identity.dataset_id == "fever_binary_v2" and method == "bm25":
            resolved = "fever_bm25_v1"
    if resolved is not None:
        resolved = _validate_stable_id(resolved, "retrieval_protocol_id")
    if formal_v2 and resolved is None:
        raise ValueError(
            "formal-v2 identity requires a resolvable retrieval_protocol_id"
        )
    return RetrievalProtocolIdentity(
        retrieval_protocol_id=resolved,
        dataset_id=dataset_identity.dataset_id,
        source_artifact_sha256=source_artifact_sha256,
        retrieval_model_or_index_identity=retrieval_model_or_index_identity,
    )


def experiment_identity_payload(
    dataset: DatasetIdentity,
    generator: GeneratorIdentity,
    retrieval: RetrievalProtocolIdentity,
) -> dict[str, Any]:
    return {
        "schema_version": EXPERIMENT_IDENTITY_SCHEMA_VERSION,
        "dataset_identity": dataset.to_dict(),
        "generator_identity": generator.to_dict(),
        "retrieval_protocol_identity": retrieval.to_dict(),
    }


def default_posterior_manifest_path(posterior_path: str | Path) -> Path:
    return Path(posterior_path).with_suffix(".manifest.json")


def _recorded_path(value: Any, manifest_path: Path) -> Path:
    if not value:
        raise ValueError("Posterior manifest is missing output_path")
    path = Path(str(value))
    return (
        path.resolve()
        if path.is_absolute()
        else (manifest_path.parent / path).resolve()
    )


def validate_posterior_provenance(
    posterior_path: str | Path,
    manifest_path: str | Path | None = None,
    *,
    mode: str = FORMAL_V2_MODE,
    expected_dataset_id: str | None = None,
    expected_split: str | None = None,
    expected_generator_id: str | None = None,
    expected_retrieval_protocol_id: str | None = None,
) -> dict[str, Any]:
    """Validate a posterior JSONL/sidecar pair and return its identity binding."""
    if mode not in {FORMAL_V2_MODE, LEGACY_MODE}:
        raise ValueError(f"Unsupported posterior validation mode: {mode!r}")
    posterior = Path(posterior_path).resolve()
    sidecar = (
        Path(manifest_path).resolve()
        if manifest_path
        else default_posterior_manifest_path(posterior).resolve()
    )
    if not posterior.is_file():
        raise FileNotFoundError(f"Posterior JSONL does not exist: {posterior}")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Posterior sidecar manifest does not exist: {sidecar}")
    try:
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid posterior sidecar manifest {sidecar}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Posterior sidecar manifest must be a JSON object")
    if manifest.get("status") != "completed":
        raise ValueError(
            f"Posterior manifest status must be 'completed'; got {manifest.get('status')!r}"
        )
    recorded = _recorded_path(manifest.get("output_path"), sidecar)
    if recorded != posterior:
        raise ValueError(
            f"Posterior manifest output_path mismatch: recorded={recorded} intended={posterior}"
        )
    actual_sha = sha256_file(posterior)
    if manifest.get("output_sha256") != actual_sha:
        raise ValueError("Posterior JSONL SHA does not match its sidecar manifest")

    provenance = manifest.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    dataset_identity = manifest.get("dataset_identity")
    generator_identity = manifest.get("generator_identity")
    retrieval_identity = manifest.get("retrieval_protocol_identity")
    identity_objects = (dataset_identity, generator_identity, retrieval_identity)
    has_structured_identity = all(isinstance(item, dict) for item in identity_objects)

    if mode == FORMAL_V2_MODE:
        if manifest.get("schema_version") != POSTERIOR_MANIFEST_SCHEMA_VERSION:
            raise ValueError("Formal-v2 requires posterior manifest schema v2")
        if manifest.get("identity_mode") != FORMAL_V2_MODE:
            raise ValueError("Formal-v2 requires identity_mode='formal_v2'")
        if not has_structured_identity:
            raise ValueError("Formal-v2 posterior manifest lacks required identity objects")
        assert isinstance(dataset_identity, dict)
        assert isinstance(generator_identity, dict)
        assert isinstance(retrieval_identity, dict)
        for obj, key in (
            (dataset_identity, "dataset_id"),
            (generator_identity, "generator_id"),
            (retrieval_identity, "retrieval_protocol_id"),
        ):
            if not obj.get(key):
                raise ValueError(f"Formal-v2 posterior identity is missing {key}")
        expected_schemas = (
            (dataset_identity, DATASET_IDENTITY_SCHEMA_VERSION),
            (generator_identity, GENERATOR_IDENTITY_SCHEMA_VERSION),
            (retrieval_identity, RETRIEVAL_IDENTITY_SCHEMA_VERSION),
        )
        if any(
            obj.get("schema_version") != expected
            for obj, expected in expected_schemas
        ):
            raise ValueError("Formal-v2 posterior identity schema version mismatch")
        if generator_identity.get("identity_source") != "explicit":
            raise ValueError(
                "Formal-v2 posterior generator identity must be explicitly assigned"
            )
        required_provenance = (
            "dataset",
            "split",
            "generator_model",
            "generator_sha256",
            "input_sha256",
            "config_sha256",
            "prompt_template_hash",
            "verbalizers_hash",
        )
        missing = [key for key in required_provenance if not provenance.get(key)]
        if missing:
            raise ValueError(
                "Formal-v2 posterior provenance is missing: " + ", ".join(missing)
            )
        git = manifest.get("git")
        if not isinstance(git, dict) or not git.get("commit"):
            raise ValueError("Formal-v2 posterior manifest lacks Git commit provenance")
        try:
            provenance_dataset = resolve_dataset_identity(
                str(provenance["dataset"]),
                explicit_dataset_id=dataset_identity["dataset_id"],
            )
        except ValueError as exc:
            raise ValueError(
                "Posterior dataset identity conflicts with provenance dataset"
            ) from exc
        if provenance_dataset.dataset_id != dataset_identity["dataset_id"]:
            raise ValueError(
                "Posterior dataset identity conflicts with provenance dataset"
            )
        generator_consistency = (
            ("model path", "model_name_or_path", "generator_model"),
            ("model SHA", "model_sha256", "generator_sha256"),
            ("prompt hash", "prompt_template_hash", "prompt_template_hash"),
            ("verbalizer hash", "verbalizer_hash", "verbalizers_hash"),
        )
        for label, identity_key, provenance_key in generator_consistency:
            if generator_identity.get(identity_key) != provenance.get(provenance_key):
                raise ValueError(
                    f"Posterior generator {label} differs between identity and provenance"
                )
        if retrieval_identity.get("dataset_id") != dataset_identity.get("dataset_id"):
            raise ValueError("Retrieval identity dataset differs from dataset identity")
        if retrieval_identity.get("source_artifact_sha256") != provenance.get(
            "input_sha256"
        ):
            raise ValueError("Retrieval identity SHA differs from posterior input SHA")
        identity = experiment_identity_payload(
            DatasetIdentity(**dataset_identity),
            GeneratorIdentity(**generator_identity),
            RetrievalProtocolIdentity(**retrieval_identity),
        )
        if manifest.get("identity_fingerprint") != stable_hash(identity):
            raise ValueError("Posterior identity fingerprint mismatch")
    else:
        dataset_identity = dataset_identity if isinstance(dataset_identity, dict) else {}
        generator_identity = generator_identity if isinstance(generator_identity, dict) else {}
        retrieval_identity = retrieval_identity if isinstance(retrieval_identity, dict) else {}

    dataset_id = dataset_identity.get("dataset_id")
    generator_id = generator_identity.get("generator_id")
    retrieval_protocol_id = retrieval_identity.get("retrieval_protocol_id")
    split = provenance.get("split")
    comparisons = (
        ("dataset_id", expected_dataset_id, dataset_id),
        ("split", expected_split, split),
        ("generator_id", expected_generator_id, generator_id),
        (
            "retrieval_protocol_id",
            expected_retrieval_protocol_id,
            retrieval_protocol_id,
        ),
    )
    for field, expected, actual in comparisons:
        if expected is not None and expected != actual:
            raise ValueError(
                f"Posterior {field} mismatch: expected={expected!r} actual={actual!r}"
            )

    return {
        "mode": mode,
        "manifest_path": str(sidecar),
        "manifest_sha256": sha256_file(sidecar),
        "manifest_fingerprint": manifest.get("fingerprint"),
        "identity_fingerprint": manifest.get("identity_fingerprint"),
        "posterior_path": str(posterior),
        "posterior_sha256": actual_sha,
        "dataset_identity": dataset_identity,
        "generator_identity": generator_identity,
        "retrieval_protocol_identity": retrieval_identity,
        "split": split,
        "manifest": manifest,
    }


def posterior_binding_contract(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable subset embedded by generator-conditioned teachers."""
    return {
        "posterior_manifest_path": binding["manifest_path"],
        "posterior_manifest_sha256": binding["manifest_sha256"],
        "posterior_manifest_fingerprint": binding.get("manifest_fingerprint"),
        "posterior_identity_fingerprint": binding.get("identity_fingerprint"),
        "posterior_sha256": binding["posterior_sha256"],
        "dataset_identity": binding.get("dataset_identity"),
        "generator_identity": binding.get("generator_identity"),
        "retrieval_protocol_identity": binding.get("retrieval_protocol_identity"),
        "split": binding.get("split"),
    }


def load_optional_posterior_binding(
    posterior_path: str | Path,
    manifest_path: str | Path | None = None,
    *,
    formal_v2: bool = False,
    expected_dataset_id: str | None = None,
    expected_split: str | None = None,
    expected_generator_id: str | None = None,
    expected_retrieval_protocol_id: str | None = None,
) -> dict[str, Any] | None:
    """Auto-locate a sidecar, preserving legacy commands that never had one."""
    posterior = Path(posterior_path).resolve()
    sidecar = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else default_posterior_manifest_path(posterior).resolve()
    )
    identity_expected = any(
        value is not None
        for value in (
            expected_dataset_id,
            expected_generator_id,
            expected_retrieval_protocol_id,
        )
    )
    if not sidecar.is_file() and not (formal_v2 or manifest_path or identity_expected):
        return None
    return validate_posterior_provenance(
        posterior,
        sidecar,
        mode=FORMAL_V2_MODE if formal_v2 else LEGACY_MODE,
        expected_dataset_id=expected_dataset_id,
        expected_split=expected_split,
        expected_generator_id=expected_generator_id,
        expected_retrieval_protocol_id=expected_retrieval_protocol_id,
    )


def formal_v2_dataset_root(root: str | Path, dataset_id: str) -> Path:
    return Path(root) / _validate_stable_id(dataset_id, "dataset_id")


def formal_v2_generator_root(
    root: str | Path, dataset_id: str, generator_id: str
) -> Path:
    return formal_v2_dataset_root(root, dataset_id) / _validate_stable_id(
        generator_id, "generator_id"
    )


def formal_v2_posterior_split_root(
    root: str | Path, dataset_id: str, generator_id: str, split: str
) -> Path:
    return (
        formal_v2_generator_root(root, dataset_id, generator_id)
        / "posteriors"
        / _validate_stable_id(split, "split")
    )

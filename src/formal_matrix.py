"""Thin formal-v2 matrix planner over existing worker scripts."""

from __future__ import annotations

import copy
import json
import os
import re
import shlex
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping

from scripts.create_generator_manifest import build_manifest as build_generator_manifest
from src.artifact_binding import validate_generator_manifest
from src.formal_registry import (
    CANONICAL_OURS,
    FORMAL_REGISTRY_FINGERPRINT,
    FORMAL_REGISTRY_VERSION,
    MAIN_TABLE_METHODS,
    KCBWDM_LINEAR_GATE_V2,
    KCBWDM_NORMALIZED_RHO_V1,
    KCBWDM_SIGNED_V1,
    dataset_protocol,
    held_out_freeze_status,
    method_spec,
    retrieval_protocol_fingerprint,
    validate_dataset_method_compatibility,
)
from src.experiment_identity import FORMAL_V2_MODE, validate_posterior_provenance
from src.fm2_formal import validate_fm2_retrieval_manifest
from src.io_utils import load_yaml
from src.experiment_identity import (
    formal_v2_dataset_root,
    formal_v2_evaluation_root,
    formal_v2_generator_root,
    formal_v2_method_seed_root,
    formal_v2_posterior_split_root,
)
from src.run_manifest import git_state, sha256_file, stable_hash, utc_now
from src.preformal.registry import (
    KCBWDM_LINEAR_GATE_V2_CONTRACT,
    KCBWDM_NORMALIZED_RHO_V1_CONTRACT,
    KCBWDM_SIGNED_V1_CONTRACT,
)


ORCHESTRATOR_SCHEMA_VERSION = "rag_cbwdm_formal_matrix_orchestrator.v1"
PLAN_MANIFEST_SCHEMA_VERSION = "rag_cbwdm_formal_matrix_plan.v1"
MATRIX_CONFIG_SCHEMA_VERSION = "rag_cbwdm_formal_matrix_config.v1"

TRAINING_RUNTIME = {
    "infogain": {
        "implementation_version": "infogain_vectorized_rank_v1",
        "optimizer_group_batch_size": 1,
        "rank_loss_implementation": "vectorized",
    },
    CANONICAL_OURS: {
        "implementation_version": "signed_optimizer_block_v1",
        "runtime_implementation": "block_v1",
        "optimizer_group_batch_size": 8,
        "forward_batch_size": 32,
    },
    KCBWDM_SIGNED_V1: {
        "implementation_version": "signed_optimizer_block_v1",
        "runtime_implementation": "block_v1",
        "optimizer_group_batch_size": 8,
        "forward_batch_size": 32,
    },
    KCBWDM_LINEAR_GATE_V2: {
        "implementation_version": "signed_optimizer_block_v1",
        "runtime_implementation": "block_v1",
        "optimizer_group_batch_size": 8,
        "forward_batch_size": 32,
    },
    KCBWDM_NORMALIZED_RHO_V1: {
        "implementation_version": "signed_optimizer_block_v1",
        "runtime_implementation": "block_v1",
        "optimizer_group_batch_size": 8,
        "forward_batch_size": 32,
    },
}

DEVELOPMENT_SMOKE = "development_smoke"
FULL_DEVELOPMENT = "full_development"
HELD_OUT = "held_out"

_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


class MatrixPlanError(ValueError):
    """Raised when a requested matrix is not scientifically or structurally valid."""


def _server_path(root: str, value: str) -> str:
    if "\\" in value or _WINDOWS_DRIVE.match(value):
        raise MatrixPlanError(f"Server paths must be POSIX paths, got {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute():
        return str(path)
    return str(PurePosixPath(root) / path)


def _local_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _repo_server_path(
    project_root: Path, server_project_root: str, local_path: Path, explicit: str | None
) -> str:
    if explicit:
        return _server_path(server_project_root, explicit)
    try:
        relative = local_path.resolve().relative_to(project_root.resolve())
    except ValueError as exc:
        raise MatrixPlanError(
            f"Path outside the repository requires an explicit server path: {local_path}"
        ) from exc
    return str(PurePosixPath(server_project_root) / PurePosixPath(relative.as_posix()))


def _sidecar(path: str) -> str:
    return str(PurePosixPath(path).with_suffix(".manifest.json"))


def _phase_a_path(root: str, helper: Callable[..., Path], *parts: object) -> str:
    """Render a Phase A formal-v2 path helper result as a server POSIX path."""
    relative = helper(Path("."), *parts)
    return str(PurePosixPath(root).joinpath(*(str(part) for part in relative.parts)))


def _command(python: str, project_root: str, script: str, *args: object) -> list[str]:
    return [
        python,
        str(PurePosixPath(project_root) / "scripts" / script),
        *(str(value) for value in args),
    ]


def _append_option(command: list[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend((flag, str(value)))


def _split_limit(config: Mapping[str, Any], split: str) -> int | None:
    limits = config.get("profile_limits")
    if not isinstance(limits, Mapping):
        return None
    aliases = {
        "train_core": ("train_core", "train"),
        "train": ("train", "train_core"),
        "validation": ("validation", "dev"),
        "dev": ("dev", "validation"),
    }
    for key in aliases.get(split, (split,)):
        value = limits.get(key)
        if isinstance(value, int):
            return value
    return None


def _kcbwdm_matrix_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    teacher = contract["teacher"]
    kernel = contract["kernel"]
    selector = contract["selector"]
    return {
        "contract_version": contract["contract_version"],
        "method": contract["method"],
        "teacher": {
            key: teacher[key]
            for key in (
                "top_m",
                "stop_threshold",
                "alignment_eps",
                "b_plus",
                "b_minus",
                "neutral_sample_policy",
                "gain_tolerance",
            )
        },
        "kernel": dict(kernel),
        "sign_policy": contract["sign_policy"],
        "selector": {
            key: selector[key]
            for key in ("top_m", "min_docs", "score_threshold")
        },
        "seed": contract["seed"],
    }


def _resolve_method_variants(
    config: Mapping[str, Any], canonical_methods: list[str]
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Resolve explicit per-method variants without changing legacy matrices."""
    raw_variants = config.get("method_variants")
    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in canonical_methods}
    flattened: list[dict[str, Any]] = []
    if raw_variants is not None:
        if not isinstance(raw_variants, list) or not raw_variants:
            raise MatrixPlanError("method_variants must be a non-empty list")
        for raw in raw_variants:
            if not isinstance(raw, Mapping):
                raise MatrixPlanError("Every method variant must be a mapping")
            method = method_spec(str(raw.get("method") or ""))["method_id"]
            if method not in by_method:
                raise MatrixPlanError(
                    f"Method variant references an unrequested method: {method!r}"
                )
            variant_id = str(raw.get("variant_id") or "")
            if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", variant_id):
                raise MatrixPlanError(f"Invalid method variant_id: {variant_id!r}")
            parameters = raw.get("parameters")
            if not isinstance(parameters, Mapping):
                raise MatrixPlanError(
                    f"Method variant {variant_id!r} requires parameters"
                )
            parameters = dict(parameters)
            if method != KCBWDM_NORMALIZED_RHO_V1:
                raise MatrixPlanError(
                    f"Explicit method variants are not supported for {method!r}"
                )
            if set(parameters) != {"rho"}:
                raise MatrixPlanError(
                    "kcbwdm_normalized_rho_v1 variants require only explicit rho"
                )
            rho = float(parameters["rho"])
            if not 0.0 <= rho <= 1.0:
                raise MatrixPlanError("Normalized-rho method variant rho must be in [0, 1]")
            entry = {
                "method_id": method,
                "variant_id": variant_id,
                "parameters": {"rho": rho},
                "explicit": True,
            }
            if any(item["variant_id"] == variant_id for item in by_method[method]):
                raise MatrixPlanError(
                    f"Duplicate method variant_id for {method}: {variant_id!r}"
                )
            by_method[method].append(entry)
            flattened.append(entry)
    for method in canonical_methods:
        if by_method[method]:
            continue
        if method == KCBWDM_NORMALIZED_RHO_V1:
            raise MatrixPlanError(
                "kcbwdm_normalized_rho_v1 requires explicit method_variants with rho"
            )
        by_method[method].append(
            {
                "method_id": method,
                "variant_id": method,
                "parameters": {},
                "explicit": False,
            }
        )
    return by_method, flattened


def _resolve_bge_spec(
    value: Any, *, project_root: Path
) -> dict[str, Any]:
    spec = dict(value) if isinstance(value, Mapping) else {}
    manifest_value = spec.get("manifest")
    if manifest_value:
        manifest_path = _local_path(project_root, str(manifest_value))
        payload = load_yaml(manifest_path)
        if not isinstance(payload, Mapping):
            raise MatrixPlanError("BGE model manifest must be a mapping")
        identity = payload.get("bge_model_freeze", payload)
        if not isinstance(identity, Mapping):
            raise MatrixPlanError("BGE model manifest identity must be a mapping")
        for key in (
            "model_id",
            "revision",
            "sha256",
            "model_name_or_path",
            "development_only",
            "local_files_only",
        ):
            if spec.get(key) is None and identity.get(key) is not None:
                spec[key] = identity[key]
        if spec.get("model_name_or_path") is None and identity.get("path") is not None:
            spec["model_name_or_path"] = identity["path"]
        spec["manifest_sha256"] = sha256_file(manifest_path)
        spec["manifest_fingerprint"] = stable_hash(identity)
    if spec.get("model_id") is None and spec.get("bge_model_id") is not None:
        spec["model_id"] = spec["bge_model_id"]
    if spec.get("sha256") is None and spec.get("model_sha256") is not None:
        spec["sha256"] = spec["model_sha256"]
    return spec


def _resolve_generator(
    spec: Mapping[str, Any],
    *,
    dataset_id: str,
    project_root: Path,
    server_project_root: str,
) -> dict[str, Any]:
    generator_id = str(spec.get("generator_id") or "")
    if not generator_id:
        raise MatrixPlanError("Every generator requires a stable generator_id")
    config_value = spec.get("config")
    manifest_value = spec.get("generator_manifest")
    if not config_value and not manifest_value:
        raise MatrixPlanError(
            f"Generator {generator_id!r} requires config or generator_manifest"
        )
    if manifest_value:
        manifest_path = _local_path(project_root, str(manifest_value))
        validated = validate_generator_manifest(
            manifest_path,
            expected_dataset_id=dataset_id,
            expected_generator_id=generator_id,
        )
        config_path = Path(validated["manifest"]["config_path"]).resolve()
        server_manifest = _repo_server_path(
            project_root,
            server_project_root,
            manifest_path,
            spec.get("server_generator_manifest"),
        )
        generator_command: list[str] = []
        generator_external = True
        manifest_preview = validated["manifest"]
    else:
        config_path = _local_path(project_root, str(config_value))
        manifest_preview = build_generator_manifest(
            config_path=config_path,
            generator_id=generator_id,
            dataset_id=dataset_id,
            model_family=spec.get("model_family"),
        )
        server_manifest = ""
        generator_command = []
        generator_external = False
    server_config = _repo_server_path(
        project_root,
        server_project_root,
        config_path,
        spec.get("server_config"),
    )
    return {
        "generator_id": generator_id,
        "generator_identity": manifest_preview["generator_identity"],
        "generator_identity_fingerprint": manifest_preview[
            "generator_identity_fingerprint"
        ],
        "dataset_identity": manifest_preview["dataset_identity"],
        "local_config": config_path,
        "server_config": server_config,
        "server_manifest": server_manifest,
        "external_manifest": generator_external,
        "generator_command": generator_command,
        "model_family": spec.get("model_family"),
        "config_sha256": sha256_file(config_path),
    }


def _node(
    *,
    node_id: str,
    stage: str,
    dataset_id: str,
    split: str | None,
    inputs: Mapping[str, str],
    outputs: Mapping[str, str],
    dependencies: Iterable[str],
    command: Iterable[str],
    generator_id: str | None = None,
    method_id: str | None = None,
    seed: int | None = None,
    reusable: bool = False,
    status_expectation: str = "missing_or_reusable_if_valid",
    execution_policy: str = "worker_validated_resume",
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "stage": stage,
        "dataset_id": dataset_id,
        "generator_id": generator_id,
        "method_id": method_id,
        "seed": seed,
        "split": split,
        "inputs": dict(inputs),
        "outputs": dict(outputs),
        "dependencies": list(dependencies),
        "command": list(command),
        "formal_v2": True,
        "reusable": reusable,
        "status_expectation": status_expectation,
        "execution_policy": execution_policy,
    }


def validate_dag(nodes: Iterable[Mapping[str, Any]]) -> list[str]:
    """Validate uniqueness, output ownership, dependencies, and acyclicity."""
    node_list = [dict(node) for node in nodes]
    ids = [str(node.get("node_id")) for node in node_list]
    if len(ids) != len(set(ids)):
        raise MatrixPlanError("Duplicate semantic node_id in formal matrix DAG")
    by_id = {str(node["node_id"]): node for node in node_list}
    output_owner: dict[str, str] = {}
    for node in node_list:
        for path in node.get("outputs", {}).values():
            path = str(path)
            owner = output_owner.get(path)
            if owner is not None and owner != node["node_id"]:
                raise MatrixPlanError(
                    f"Output collision: {path!r} owned by {owner!r} and "
                    f"{node['node_id']!r}"
                )
            output_owner[path] = str(node["node_id"])
        for dependency in node.get("dependencies", []):
            if dependency not in by_id:
                raise MatrixPlanError(
                    f"Node {node['node_id']!r} has unknown dependency {dependency!r}"
                )
    incoming = {
        node_id: set(map(str, node.get("dependencies", [])))
        for node_id, node in by_id.items()
    }
    ready = sorted(node_id for node_id, deps in incoming.items() if not deps)
    ordered: list[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for node_id in sorted(incoming):
            if current in incoming[node_id]:
                incoming[node_id].remove(current)
                if not incoming[node_id] and node_id not in ordered and node_id not in ready:
                    ready.append(node_id)
                    ready.sort()
    if len(ordered) != len(node_list):
        raise MatrixPlanError("Formal matrix DAG contains a cycle")
    return ordered


def _semantic_plan_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": plan["schema_version"],
        "orchestrator_schema_version": plan["orchestrator_schema_version"],
        "registry_version": plan["registry_version"],
        "registry_fingerprint": plan["registry_fingerprint"],
        "dataset_identity": plan["dataset_identity"],
        "retrieval_protocol": plan["retrieval_protocol"],
        "generator_identities": [
            {
                "generator_id": item["generator_id"],
                "generator_identity": item["generator_identity"],
                "generator_identity_fingerprint": item[
                    "generator_identity_fingerprint"
                ],
                "config_sha256": item["config_sha256"],
            }
            for item in plan["generator_identities"]
        ],
        "generator_registry_fingerprint": plan["generator_registry_fingerprint"],
        "methods": plan["methods"],
        "method_contracts": plan["method_contracts"],
        "method_variants": plan.get("method_variants", []),
        "seed_policy": plan["seed_policy"],
        "profile": plan["profile"],
        "split_role": plan["split_role"],
        "training_split": plan["training_split"],
        "evaluation_split": plan["evaluation_split"],
        "held_out": plan["held_out"],
        "limits": plan["limits"],
        "retrieval_inputs": plan["retrieval_inputs"],
        "posterior_reuse_bindings": plan["posterior_reuse_bindings"],
        "dataset_config_sha256": plan["dataset_config_sha256"],
        "bge_contract": {
            key: plan["bge_contract"].get(key)
            for key in (
                "model_id",
                "revision",
                "sha256",
                "development_only",
                "manifest_fingerprint",
            )
        },
        "training_runtime": plan["training_runtime"],
        "git_commit": plan["git"].get("commit"),
        "nodes": [
            {
                key: node.get(key)
                for key in (
                    "node_id",
                    "stage",
                    "dataset_id",
                    "generator_id",
                    "method_id",
                    "method_variant_id",
                    "method_parameters",
                    "seed",
                    "split",
                    "dependencies",
                    "formal_v2",
                    "reusable",
                    "status_expectation",
                    "execution_policy",
                )
            }
            for node in plan["nodes"]
        ],
        "result_index": [
            {
                key: row.get(key)
                for key in (
                    "dataset_id",
                    "generator_id",
                    "method_id",
                    "method",
                    "method_variant_id",
                    "method_parameters",
                    "seed",
                    "split",
                    "retrieval_protocol_id",
                    "retrieval_protocol_fingerprint",
                    "experiment_type",
                    "node_id",
                )
            }
            for row in plan["result_index"]
        ],
        "unresolved_freeze_decisions": plan["unresolved_freeze_decisions"],
    }


def build_execution_plan(
    matrix_config: Mapping[str, Any],
    *,
    project_root: str | Path,
    server_project_root: str = "/root/rag-cbwdm",
    artifact_root: str | None = None,
    server_python: str = "python",
    held_out: bool = False,
    held_out_freeze: Mapping[str, Any] | None = None,
    git: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate a formal-v2 plan without touching matrix artifacts."""
    config = copy.deepcopy(dict(matrix_config))
    if config.get("schema_version") != MATRIX_CONFIG_SCHEMA_VERSION:
        raise MatrixPlanError("Formal matrix config schema version mismatch")
    project = Path(project_root).resolve()
    dataset_id = str(config.get("dataset_id") or "")
    protocol = dataset_protocol(dataset_id)
    protocol_fingerprint = retrieval_protocol_fingerprint(dataset_id)
    retrieval_protocol_id = str(
        config.get("retrieval_protocol_id") or protocol["retrieval_protocol_id"]
    )
    profile = str(config.get("profile") or FULL_DEVELOPMENT)
    if profile not in {DEVELOPMENT_SMOKE, FULL_DEVELOPMENT, HELD_OUT}:
        raise MatrixPlanError(f"Unknown formal matrix profile: {profile!r}")
    raw_methods = config.get("methods", list(MAIN_TABLE_METHODS))
    methods = [str(method) for method in raw_methods]
    if len(methods) != len(set(methods)):
        raise MatrixPlanError("Duplicate method in formal matrix config")
    for method in methods:
        validate_dataset_method_compatibility(
            dataset_id,
            method,
            retrieval_protocol_id,
            development=not held_out and profile != HELD_OUT,
        )
    canonical_methods = [method_spec(method)["method_id"] for method in methods]
    method_variants_by_method, explicit_method_variants = _resolve_method_variants(
        config, canonical_methods
    )
    method_contracts: dict[str, Any] = {}
    kernel_method_configs = {
        KCBWDM_SIGNED_V1: ("kcbwdm", KCBWDM_SIGNED_V1_CONTRACT),
        KCBWDM_LINEAR_GATE_V2: (
            "kcbwdm_linear_gate_v2",
            KCBWDM_LINEAR_GATE_V2_CONTRACT,
        ),
        KCBWDM_NORMALIZED_RHO_V1: (
            "kcbwdm_normalized_rho_v1",
            KCBWDM_NORMALIZED_RHO_V1_CONTRACT,
        ),
    }
    for kernel_method, (config_key, contract) in kernel_method_configs.items():
        if kernel_method not in canonical_methods:
            continue
        expected_kcbwdm = _kcbwdm_matrix_contract(contract)
        configured_kcbwdm = config.get(config_key)
        if configured_kcbwdm != expected_kcbwdm:
            raise MatrixPlanError(
                f"{kernel_method} matrix policy must exactly match its frozen contract"
            )
        method_contracts[kernel_method] = expected_kcbwdm
    if held_out:
        if profile != HELD_OUT:
            profile = HELD_OUT
        freeze_status = held_out_freeze_status(
            held_out_freeze, expected_dataset_id=dataset_id
        )
        if not freeze_status["ready"]:
            raise MatrixPlanError(
                "Held-out planning is blocked: " + "; ".join(freeze_status["blockers"])
            )
    elif profile == HELD_OUT:
        raise MatrixPlanError("Held-out profile requires explicit held_out=True opt-in")

    training_split = str(config.get("training_split") or "")
    evaluation_split = str(config.get("evaluation_split") or "")
    if held_out:
        if evaluation_split != protocol["final_split"]:
            raise MatrixPlanError("Held-out plan does not target the registry final split")
    else:
        if evaluation_split == protocol["final_split"]:
            raise MatrixPlanError("Development plan cannot target or claim the final split")
        role = protocol["split_policy"].get(evaluation_split)
        if role not in {"development_calibration", "development_evaluation"}:
            raise MatrixPlanError("Development plan requires a registered development split")
    if protocol["split_policy"].get(training_split) != "learning":
        raise MatrixPlanError("training_split is not registered as a learning split")

    dataset_config_value = str(config.get("dataset_config") or "")
    if not dataset_config_value:
        raise MatrixPlanError("Formal matrix config requires dataset_config")
    dataset_config_path = _local_path(project, dataset_config_value)
    dataset_config = load_yaml(dataset_config_path)
    dataset_config_sha256 = sha256_file(dataset_config_path)
    baseline_config = dataset_config.get("baselines", {})
    common_config = baseline_config.get("common") or baseline_config.get(
        "naive_topm", {}
    )
    top_m = int(common_config.get("top_m", 4))
    server_dataset_config = _repo_server_path(
        project,
        server_project_root,
        dataset_config_path,
        config.get("server_dataset_config"),
    )
    limits = {
        "training": _split_limit(dataset_config, training_split),
        "evaluation": _split_limit(dataset_config, evaluation_split),
    }
    requested_seeds = config.get("learned_seeds")
    if requested_seeds is None:
        if profile == DEVELOPMENT_SMOKE:
            requested_seeds = dataset_config.get("profile_limits", {}).get("seeds", [13])
        else:
            requested_seeds = method_spec("infogain")["seed_policy"]["seeds"]
    learned_seeds = [int(seed) for seed in requested_seeds]
    if len(learned_seeds) != len(set(learned_seeds)):
        raise MatrixPlanError("Duplicate learned seed")
    for method in canonical_methods:
        spec = method_spec(method)
        if not spec["learned_selector"]:
            continue
        allowed_seeds = set(spec["seed_policy"]["seeds"])
        if any(seed not in allowed_seeds for seed in learned_seeds):
            raise MatrixPlanError(
                f"Illegal learned seed for {method}; allowed seeds are {sorted(allowed_seeds)}"
            )

    generator_specs = config.get("generators")
    if not isinstance(generator_specs, list) or not generator_specs:
        raise MatrixPlanError("Formal matrix requires at least one generator")
    generators = [
        _resolve_generator(
            spec,
            dataset_id=dataset_id,
            project_root=project,
            server_project_root=server_project_root,
        )
        for spec in generator_specs
    ]
    generator_ids = [item["generator_id"] for item in generators]
    if len(generator_ids) != len(set(generator_ids)):
        raise MatrixPlanError("Duplicate generator_id in formal matrix")
    generator_registry_fingerprint = stable_hash(
        [
            {
                "generator_id": item["generator_id"],
                "generator_identity_fingerprint": item[
                    "generator_identity_fingerprint"
                ],
            }
            for item in sorted(generators, key=lambda value: value["generator_id"])
        ]
    )
    current_git = dict(git or git_state(project))
    if held_out:
        assert held_out_freeze is not None
        if held_out_freeze.get(
            "generator_registry_fingerprint"
        ) != generator_registry_fingerprint:
            raise MatrixPlanError("Held-out generator registry fingerprint mismatch")
        if held_out_freeze.get("git_commit") != current_git.get("commit"):
            raise MatrixPlanError("Held-out freeze Git commit mismatch")
        if held_out_freeze.get("config_sha256") != dataset_config_sha256:
            raise MatrixPlanError("Held-out freeze config SHA mismatch")
        if held_out_freeze.get("learned_method_seeds") != {
            "infogain": learned_seeds,
            CANONICAL_OURS: learned_seeds,
        }:
            raise MatrixPlanError("Held-out plan seed set does not match the freeze")
        prompt_hashes = {
            item["generator_identity"]["prompt_template_hash"] for item in generators
        }
        verbalizer_hashes = {
            item["generator_identity"]["verbalizer_hash"] for item in generators
        }
        if prompt_hashes != {held_out_freeze.get("prompt_hash")}:
            raise MatrixPlanError("Held-out generator prompt hash mismatch")
        if verbalizer_hashes != {held_out_freeze.get("verbalizer_hash")}:
            raise MatrixPlanError("Held-out generator verbalizer hash mismatch")
        if held_out_freeze.get("top_k") != top_m:
            raise MatrixPlanError("Held-out Top-k does not match the matrix config")
        if held_out_freeze.get("evidence_budget", {}).get("max_docs") != top_m:
            raise MatrixPlanError(
                "Held-out evidence budget does not match the matrix config"
            )

    configured_root = artifact_root or config.get("artifact_root") or "artifacts/formal_v2"
    server_artifact_root = _server_path(server_project_root, str(configured_root))
    dataset_root = _phase_a_path(
        server_artifact_root, formal_v2_dataset_root, dataset_id
    )
    shared_root = str(PurePosixPath(dataset_root) / "shared")
    retrieval_inputs = config.get("retrieval_inputs")
    if not isinstance(retrieval_inputs, Mapping):
        raise MatrixPlanError("Formal matrix requires retrieval_inputs")
    retrieval_bindings: dict[str, dict[str, Any]] = {}

    def resolve_retrieval_input(split: str) -> str:
        try:
            value = retrieval_inputs[split]
        except KeyError as exc:
            raise MatrixPlanError(f"Missing retrieval input for split {split!r}") from exc
        if retrieval_protocol_id != "fm2_official_closed_page_v1":
            if isinstance(value, Mapping):
                value = value.get("pool") or value.get("path")
            if not value:
                raise MatrixPlanError(f"Missing retrieval path for split {split!r}")
            return _server_path(server_project_root, str(value))
        if not isinstance(value, Mapping):
            raise MatrixPlanError(
                "FM2 retrieval_inputs entries require pool and manifest paths"
            )
        pool_value = value.get("pool")
        manifest_value = value.get("manifest")
        if not pool_value or not manifest_value:
            raise MatrixPlanError(
                f"FM2 retrieval input {split!r} requires pool and manifest"
            )
        local_pool = _local_path(project, str(pool_value))
        local_manifest = _local_path(project, str(manifest_value))
        try:
            validated = validate_fm2_retrieval_manifest(
                local_manifest,
                local_pool,
                expected_formal_role=split,
                expected_dataset_id=dataset_id,
                expected_retrieval_protocol_id=retrieval_protocol_id,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise MatrixPlanError(
                f"Invalid FM2 retrieval input for split {split!r}: {exc}"
            ) from exc
        server_pool = _server_path(
            server_project_root, str(value.get("server_pool") or pool_value)
        )
        server_manifest = _server_path(
            server_project_root, str(value.get("server_manifest") or manifest_value)
        )
        retrieval_bindings[split] = {
            **{
                key: value
                for key, value in validated.items()
                if key not in {"pool_path", "manifest_path"}
            },
            "pool": server_pool,
            "manifest": server_manifest,
        }
        return server_pool

    retrieval_train = resolve_retrieval_input(training_split)
    retrieval_eval = resolve_retrieval_input(evaluation_split)

    bge_spec = _resolve_bge_spec(config.get("bge"), project_root=project)
    if "bge" in canonical_methods and not bge_spec.get("model_name_or_path"):
        raise MatrixPlanError("BGE method requires an explicitly supplied BGE model")
    if held_out and "bge" in canonical_methods:
        assert held_out_freeze is not None
        frozen_bge = held_out_freeze["bge_model_freeze"]
        comparisons = {
            "model_id": bge_spec.get("model_id"),
            "revision": bge_spec.get("revision"),
            "sha256": bge_spec.get("sha256"),
        }
        if comparisons != dict(frozen_bge):
            raise MatrixPlanError("Held-out BGE model does not match the signed freeze")

    nodes: list[dict[str, Any]] = []

    def add(node: dict[str, Any]) -> None:
        nodes.append(node)

    dataset_node = f"{dataset_id}.shared.dataset"
    add(
        _node(
            node_id=dataset_node,
            stage="dataset_input",
            dataset_id=dataset_id,
            split=None,
            inputs={},
            outputs={"config": server_dataset_config},
            dependencies=[],
            command=[],
            reusable=True,
            status_expectation="external_input_required",
            execution_policy="validate_external_input",
        )
    )
    retrieval_nodes: dict[str, str] = {}
    for split, path in ((training_split, retrieval_train), (evaluation_split, retrieval_eval)):
        if split in retrieval_nodes:
            continue
        node_id = f"{dataset_id}.shared.retrieval.{split}"
        retrieval_nodes[split] = node_id
        retrieval_outputs = {"retrieval": path}
        retrieval_node_inputs = {"dataset_config": server_dataset_config}
        execution_policy = "validate_external_input"
        if split in retrieval_bindings:
            retrieval_node_inputs["manifest"] = retrieval_bindings[split]["manifest"]
            execution_policy = "validate_external_fm2_retrieval"
        retrieval_node = _node(
            node_id=node_id,
            stage="retrieval_input",
            dataset_id=dataset_id,
            split=split,
            inputs=retrieval_node_inputs,
            outputs=retrieval_outputs,
            dependencies=[dataset_node],
            command=[],
            reusable=True,
            status_expectation="external_input_required",
            execution_policy=execution_policy,
        )
        if split in retrieval_bindings:
            retrieval_node["retrieval_binding"] = retrieval_bindings[split]
        add(retrieval_node)

    shared_selection: dict[str, dict[str, str]] = {}
    if "no_evidence" in canonical_methods:
        output = str(
            PurePosixPath(shared_root) / "no_evidence" / evaluation_split / "selection.jsonl"
        )
        node_id = f"{dataset_id}.shared.no_evidence.{evaluation_split}"
        command = _command(
            server_python,
            server_project_root,
            "preformal/27a_select_no_evidence.py",
            "--retrieval",
            retrieval_eval,
            "--output",
            output,
            "--dataset-id",
            dataset_id,
            "--retrieval-protocol-id",
            retrieval_protocol_id,
            "--formal-v2-identity",
            "--split",
            evaluation_split,
            "--resume",
        )
        add(
            _node(
                node_id=node_id,
                stage="selection",
                dataset_id=dataset_id,
                split=evaluation_split,
                inputs={"retrieval": retrieval_eval},
                outputs={"selection": output, "manifest": _sidecar(output)},
                dependencies=[retrieval_nodes[evaluation_split]],
                command=command,
                method_id="no_evidence",
                reusable=True,
            )
        )
        shared_selection["no_evidence"] = {
            "node": node_id,
            "selection": output,
            "manifest": _sidecar(output),
        }
    if "retrieval_topk" in canonical_methods:
        output = str(
            PurePosixPath(shared_root)
            / "retrieval_topk"
            / evaluation_split
            / "selection.jsonl"
        )
        node_id = f"{dataset_id}.shared.retrieval_topk.{evaluation_split}"
        naive = baseline_config.get("naive") or baseline_config.get("naive_topm", {})
        command = _command(
            server_python,
            server_project_root,
            "08_select_naive_topm.py",
            "--config",
            server_dataset_config,
            "--retrieval",
            retrieval_eval,
            "--output",
            output,
            "--top-m",
            top_m,
            "--min-docs",
            int(naive.get("min_docs", top_m)),
            "--method-name",
            "retrieval_topk",
            "--dataset-id",
            dataset_id,
            "--retrieval-protocol-id",
            retrieval_protocol_id,
            "--formal-v2-identity",
            "--resume",
        )
        _append_option(command, "--limit", limits["evaluation"])
        add(
            _node(
                node_id=node_id,
                stage="selection",
                dataset_id=dataset_id,
                split=evaluation_split,
                inputs={"retrieval": retrieval_eval},
                outputs={"selection": output, "manifest": _sidecar(output)},
                dependencies=[retrieval_nodes[evaluation_split]],
                command=command,
                method_id="retrieval_topk",
                reusable=True,
            )
        )
        shared_selection["retrieval_topk"] = {
            "node": node_id,
            "selection": output,
            "manifest": _sidecar(output),
        }
    if "bge" in canonical_methods:
        output = str(
            PurePosixPath(shared_root) / "bge" / evaluation_split / "selection.jsonl"
        )
        cache = str(
            PurePosixPath(shared_root) / "bge" / evaluation_split / "scores.jsonl"
        )
        node_id = f"{dataset_id}.shared.bge.{evaluation_split}"
        bge_config = baseline_config.get("bge", {})
        command = _command(
            server_python,
            server_project_root,
            "12_select_bge_reranker.py",
            "--retrieval",
            retrieval_eval,
            "--output",
            output,
            "--score-cache",
            cache,
            "--model-name-or-path",
            bge_spec["model_name_or_path"],
            "--dataset-id",
            dataset_id,
            "--retrieval-protocol-id",
            retrieval_protocol_id,
            "--formal-v2-identity",
            "--dtype",
            bge_config.get("dtype", "auto"),
            "--batch-size",
            int(bge_config.get("batch_size", 8)),
            "--max-length",
            int(bge_config.get("max_length", 512)),
            "--top-m",
            top_m,
            "--min-docs",
            int(bge_config.get("min_docs", top_m)),
            "--resume",
        )
        _append_option(command, "--revision", bge_spec.get("revision"))
        _append_option(command, "--model-sha256", bge_spec.get("sha256"))
        _append_option(command, "--score-threshold", bge_config.get("threshold"))
        _append_option(command, "--limit", limits["evaluation"])
        if bge_config.get("normalize_score"):
            command.append("--normalize-score")
        if bge_spec.get("local_files_only", bge_config.get("local_files_only")):
            command.append("--local-files-only")
        add(
            _node(
                node_id=node_id,
                stage="selection",
                dataset_id=dataset_id,
                split=evaluation_split,
                inputs={"retrieval": retrieval_eval},
                outputs={
                    "score_cache": cache,
                    "score_manifest": _sidecar(cache),
                    "selection": output,
                    "manifest": _sidecar(output),
                },
                dependencies=[retrieval_nodes[evaluation_split]],
                command=command,
                method_id="bge",
                reusable=True,
            )
        )
        shared_selection["bge"] = {
            "node": node_id,
            "selection": output,
            "manifest": _sidecar(output),
        }

    result_index: list[dict[str, Any]] = []
    generator_identities: list[dict[str, Any]] = []
    learned_methods = {
        "infogain",
        CANONICAL_OURS,
        KCBWDM_SIGNED_V1,
        KCBWDM_LINEAR_GATE_V2,
        KCBWDM_NORMALIZED_RHO_V1,
    } & set(canonical_methods)
    posterior_inputs = config.get("posterior_inputs")
    if posterior_inputs is not None and not isinstance(posterior_inputs, Mapping):
        raise MatrixPlanError("posterior_inputs must be keyed by generator_id")
    posterior_reuse_bindings: list[dict[str, Any]] = []
    for generator in generators:
        generator_id = generator["generator_id"]
        generator_root = _phase_a_path(
            server_artifact_root,
            formal_v2_generator_root,
            dataset_id,
            generator_id,
        )
        manifest_output = generator["server_manifest"] or str(
            PurePosixPath(generator_root) / "generator" / "manifest.json"
        )
        generator_identities.append(
            {
                "generator_id": generator_id,
                "generator_identity": generator["generator_identity"],
                "generator_identity_fingerprint": generator[
                    "generator_identity_fingerprint"
                ],
                "generator_manifest": manifest_output,
                "config_sha256": generator["config_sha256"],
            }
        )
        generator_node = f"{dataset_id}.{generator_id}.generator_manifest"
        if generator["external_manifest"]:
            generator_command = []
            generator_policy = "validate_external_generator_manifest"
            generator_status = "external_input_required"
        else:
            generator_command = _command(
                server_python,
                server_project_root,
                "create_generator_manifest.py",
                "--config",
                generator["server_config"],
                "--generator-id",
                generator_id,
                "--dataset-id",
                dataset_id,
                "--output",
                manifest_output,
            )
            _append_option(generator_command, "--model-family", generator["model_family"])
            generator_policy = "validate_existing_generator_manifest_or_create"
            generator_status = "missing_or_reusable_if_valid"
        add(
            _node(
                node_id=generator_node,
                stage="generator_manifest",
                dataset_id=dataset_id,
                generator_id=generator_id,
                split=None,
                inputs={"config": generator["server_config"]},
                outputs={"manifest": manifest_output},
                dependencies=[dataset_node],
                command=generator_command,
                reusable=True,
                status_expectation=generator_status,
                execution_policy=generator_policy,
            )
        )

        posterior_nodes: dict[str, dict[str, str]] = {}
        if learned_methods:
            for split, retrieval_path in (
                (training_split, retrieval_train),
                (evaluation_split, retrieval_eval),
            ):
                if split in posterior_nodes:
                    continue
                node_id = f"{dataset_id}.{generator_id}.posteriors.{split}"
                generator_posterior_inputs = (
                    posterior_inputs.get(generator_id)
                    if isinstance(posterior_inputs, Mapping)
                    else None
                )
                external_spec = (
                    generator_posterior_inputs.get(split)
                    if isinstance(generator_posterior_inputs, Mapping)
                    else None
                )
                if external_spec is not None:
                    if not isinstance(external_spec, Mapping):
                        raise MatrixPlanError(
                            f"posterior_inputs[{generator_id!r}][{split!r}] must be a mapping"
                        )
                    posterior_value = external_spec.get("posteriors")
                    manifest_value = external_spec.get("manifest")
                    if not posterior_value or not manifest_value:
                        raise MatrixPlanError(
                            f"External posterior {generator_id}/{split} requires posteriors and manifest"
                        )
                    local_posterior = _local_path(project, str(posterior_value))
                    local_manifest = _local_path(project, str(manifest_value))
                    try:
                        validated = validate_posterior_provenance(
                            local_posterior,
                            local_manifest,
                            mode=FORMAL_V2_MODE,
                            expected_dataset_id=dataset_id,
                            expected_split=split,
                            expected_generator_id=generator_id,
                            expected_retrieval_protocol_id=retrieval_protocol_id,
                        )
                    except (FileNotFoundError, ValueError) as exc:
                        raise MatrixPlanError(
                            f"Invalid external posterior for {generator_id}/{split}: {exc}"
                        ) from exc
                    expected_rows = retrieval_bindings.get(split, {}).get("num_rows")
                    posterior_manifest = validated["manifest"]
                    if expected_rows is not None and (
                        posterior_manifest.get("expected_rows") != expected_rows
                        or posterior_manifest.get("completed_rows") != expected_rows
                    ):
                        raise MatrixPlanError(
                            "External posterior row coverage does not match the "
                            f"authoritative retrieval input for {generator_id}/{split}: "
                            f"expected={expected_rows!r} "
                            f"manifest_expected={posterior_manifest.get('expected_rows')!r} "
                            f"manifest_completed={posterior_manifest.get('completed_rows')!r}"
                        )
                    output = _server_path(
                        server_project_root,
                        str(external_spec.get("server_posteriors") or posterior_value),
                    )
                    output_manifest = _server_path(
                        server_project_root,
                        str(external_spec.get("server_manifest") or manifest_value),
                    )
                    binding = {
                        key: validated[key]
                        for key in (
                            "manifest_sha256",
                            "manifest_fingerprint",
                            "identity_fingerprint",
                            "posterior_sha256",
                            "dataset_identity",
                            "generator_identity",
                            "retrieval_protocol_identity",
                            "split",
                        )
                    }
                    binding.update(
                        {
                            "expected_rows": posterior_manifest.get("expected_rows"),
                            "completed_rows": posterior_manifest.get("completed_rows"),
                        }
                    )
                    binding.update(
                        {
                            "posteriors": output,
                            "manifest": output_manifest,
                            "generator_id": generator_id,
                        }
                    )
                    posterior_node = _node(
                        node_id=node_id,
                        stage="posteriors",
                        dataset_id=dataset_id,
                        generator_id=generator_id,
                        split=split,
                        inputs={
                            "retrieval": retrieval_path,
                            "generator_manifest": manifest_output,
                            "manifest": output_manifest,
                        },
                        outputs={
                            "posteriors": output,
                            "manifest": output_manifest,
                        },
                        dependencies=[generator_node, retrieval_nodes[split]],
                        command=[],
                        reusable=True,
                        status_expectation="external_input_required",
                        execution_policy="validate_external_posterior",
                    )
                    posterior_node["posterior_binding"] = binding
                    add(posterior_node)
                    posterior_reuse_bindings.append(binding)
                else:
                    posterior_root = _phase_a_path(
                        server_artifact_root,
                        formal_v2_posterior_split_root,
                        dataset_id,
                        generator_id,
                        split,
                    )
                    output = str(PurePosixPath(posterior_root) / "posteriors.jsonl")
                    output_manifest = _sidecar(output)
                    command = _command(
                        server_python,
                        server_project_root,
                        "03_compute_label_posteriors.py",
                        "--config",
                        generator["server_config"],
                        "--split",
                        split,
                        "--retrieval",
                        retrieval_path,
                        "--output",
                        output,
                        "--dataset-id",
                        dataset_id,
                        "--generator-id",
                        generator_id,
                        "--retrieval-protocol-id",
                        retrieval_protocol_id,
                        "--formal-v2-identity",
                        "--resume",
                    )
                    limit = limits["training"] if split == training_split else limits["evaluation"]
                    _append_option(command, "--limit", limit)
                    add(
                        _node(
                            node_id=node_id,
                            stage="posteriors",
                            dataset_id=dataset_id,
                            generator_id=generator_id,
                            split=split,
                            inputs={
                                "retrieval": retrieval_path,
                                "generator_manifest": manifest_output,
                            },
                            outputs={"posteriors": output, "manifest": output_manifest},
                            dependencies=[generator_node, retrieval_nodes[split]],
                            command=command,
                            reusable=False,
                        )
                    )
                posterior_nodes[split] = {
                    "node": node_id,
                    "posteriors": output,
                    "manifest": output_manifest,
                }

        learned_selections: dict[tuple[Any, ...], dict[str, str]] = {}
        if "infogain" in learned_methods:
            info_root = str(PurePosixPath(generator_root) / "infogain")
            teacher = str(PurePosixPath(info_root) / "teacher" / "teacher.jsonl")
            teacher_node = f"{dataset_id}.{generator_id}.infogain.teacher"
            info_config = baseline_config.get("infogain_fever") or baseline_config.get(
                "infogain_adapter", {}
            )
            if not info_config.get("model_name"):
                raise MatrixPlanError(
                    "InfoGain requires baselines.infogain_fever.model_name"
                )
            command = _command(
                server_python,
                server_project_root,
                "12a_build_infogain_teacher.py",
                "--posteriors",
                posterior_nodes[training_split]["posteriors"],
                "--posterior-manifest",
                posterior_nodes[training_split]["manifest"],
                "--output",
                teacher,
                "--purpose",
                "training",
                "--threshold-mode",
                info_config.get("threshold_mode", "train_quantile"),
                "--positive-quantile",
                info_config.get("positive_quantile", 0.75),
                "--negative-quantile",
                info_config.get("negative_quantile", 0.25),
                "--dataset-id",
                dataset_id,
                "--generator-id",
                generator_id,
                "--retrieval-protocol-id",
                retrieval_protocol_id,
                "--formal-v2-identity",
                "--resume",
            )
            _append_option(command, "--limit", limits["training"])
            add(
                _node(
                    node_id=teacher_node,
                    stage="teacher",
                    dataset_id=dataset_id,
                    generator_id=generator_id,
                    method_id="infogain",
                    split=training_split,
                    inputs={
                        "posteriors": posterior_nodes[training_split]["posteriors"],
                        "posterior_manifest": posterior_nodes[training_split]["manifest"],
                    },
                    outputs={"teacher": teacher, "manifest": _sidecar(teacher)},
                    dependencies=[posterior_nodes[training_split]["node"]],
                    command=command,
                )
            )
            for seed in learned_seeds:
                seed_root = _phase_a_path(
                    server_artifact_root,
                    formal_v2_method_seed_root,
                    dataset_id,
                    generator_id,
                    "infogain",
                    seed,
                )
                training_manifest = str(PurePosixPath(seed_root) / "training_manifest.json")
                checkpoint = str(PurePosixPath(seed_root) / "checkpoint")
                train_node = f"{dataset_id}.{generator_id}.infogain.seed{seed}.train"
                command = _command(
                    server_python,
                    server_project_root,
                    "12b_train_infogain_reranker.py",
                    "--teacher",
                    teacher,
                    "--teacher-manifest",
                    _sidecar(teacher),
                    "--config",
                    generator["server_config"],
                    "--dataset-id",
                    dataset_id,
                    "--generator-id",
                    generator_id,
                    "--formal-v2-identity",
                    "--output-dir",
                    seed_root,
                    "--model-name-or-path",
                    info_config.get("model_name"),
                    "--max-length",
                    info_config.get("max_length", 512),
                    "--epochs",
                    info_config.get("epochs", 1),
                    "--lr",
                    info_config.get("lr", 2e-5),
                    "--beta",
                    info_config.get("beta", 0.75),
                    "--rank-loss-implementation",
                    TRAINING_RUNTIME["infogain"]["rank_loss_implementation"],
                    "--seed",
                    seed,
                    "--resume",
                )
                _append_option(command, "--revision", info_config.get("revision"))
                add(
                    _node(
                        node_id=train_node,
                        stage="training",
                        dataset_id=dataset_id,
                        generator_id=generator_id,
                        method_id="infogain",
                        seed=seed,
                        split=training_split,
                        inputs={"teacher": teacher, "teacher_manifest": _sidecar(teacher)},
                        outputs={
                            "checkpoint": checkpoint,
                            "training_manifest": training_manifest,
                        },
                        dependencies=[teacher_node],
                        command=command,
                    )
                )
                selection = str(PurePosixPath(seed_root) / "selection" / f"{evaluation_split}.jsonl")
                select_node = f"{dataset_id}.{generator_id}.infogain.seed{seed}.select"
                command = _command(
                    server_python,
                    server_project_root,
                    "12c_select_infogain_reranker.py",
                    "--retrieval",
                    retrieval_eval,
                    "--checkpoint-dir",
                    checkpoint,
                    "--training-manifest",
                    training_manifest,
                    "--dataset-id",
                    dataset_id,
                    "--generator-id",
                    generator_id,
                    "--formal-v2-identity",
                    "--output",
                    selection,
                    "--top-m",
                    info_config.get("top_m", top_m),
                    "--min-docs",
                    info_config.get("min_docs", top_m),
                    "--method-name",
                    "infogain_fever",
                    "--resume",
                )
                _append_option(command, "--filter-threshold", info_config.get("inference_threshold"))
                _append_option(command, "--limit", limits["evaluation"])
                add(
                    _node(
                        node_id=select_node,
                        stage="selection",
                        dataset_id=dataset_id,
                        generator_id=generator_id,
                        method_id="infogain",
                        seed=seed,
                        split=evaluation_split,
                        inputs={
                            "retrieval": retrieval_eval,
                            "checkpoint": checkpoint,
                            "training_manifest": training_manifest,
                        },
                        outputs={"selection": selection, "manifest": _sidecar(selection)},
                        dependencies=[train_node, retrieval_nodes[evaluation_split]],
                        command=command,
                    )
                )
                learned_selections[("infogain", seed)] = {
                    "node": select_node,
                    "selection": selection,
                    "manifest": _sidecar(selection),
                }

        if CANONICAL_OURS in learned_methods:
            ours_root = str(PurePosixPath(generator_root) / "ours_signed_v1")
            teacher_dir = str(PurePosixPath(ours_root) / "teacher")
            teacher = str(PurePosixPath(teacher_dir) / "teacher.jsonl")
            teacher_manifest = str(PurePosixPath(teacher_dir) / "manifest.json")
            teacher_node = f"{dataset_id}.{generator_id}.ours_signed_v1.teacher"
            command = _command(
                server_python,
                server_project_root,
                "preformal/25_materialize_signed_v1_teacher.py",
                "--config",
                generator["server_config"],
                "--posteriors",
                posterior_nodes[training_split]["posteriors"],
                "--posterior-manifest",
                posterior_nodes[training_split]["manifest"],
                "--retrieval",
                retrieval_train,
                "--output-dir",
                teacher_dir,
                "--training-split",
                training_split,
                "--dataset-id",
                dataset_id,
                "--generator-id",
                generator_id,
                "--retrieval-protocol-id",
                retrieval_protocol_id,
                "--formal-v2-identity",
                "--resume",
            )
            add(
                _node(
                    node_id=teacher_node,
                    stage="teacher",
                    dataset_id=dataset_id,
                    generator_id=generator_id,
                    method_id=CANONICAL_OURS,
                    split=training_split,
                    inputs={
                        "posteriors": posterior_nodes[training_split]["posteriors"],
                        "posterior_manifest": posterior_nodes[training_split]["manifest"],
                        "retrieval": retrieval_train,
                    },
                    outputs={"teacher": teacher, "manifest": teacher_manifest},
                    dependencies=[
                        posterior_nodes[training_split]["node"],
                        retrieval_nodes[training_split],
                    ],
                    command=command,
                )
            )
            selector_config = dataset_config.get("selector") or dataset_config.get(
                "signed_v1", {}
            )
            if not selector_config.get("model_name"):
                raise MatrixPlanError("Signed-v1 requires selector.model_name")
            for seed in learned_seeds:
                seed_root = _phase_a_path(
                    server_artifact_root,
                    formal_v2_method_seed_root,
                    dataset_id,
                    generator_id,
                    CANONICAL_OURS,
                    seed,
                )
                checkpoint = str(PurePosixPath(seed_root) / "checkpoint")
                training_manifest = str(PurePosixPath(seed_root) / "training_manifest.json")
                train_node = f"{dataset_id}.{generator_id}.ours_signed_v1.seed{seed}.train"
                command = _command(
                    server_python,
                    server_project_root,
                    "preformal/26_train_signed_v1.py",
                    "--config",
                    generator["server_config"],
                    "--teacher",
                    teacher,
                    "--teacher-manifest",
                    teacher_manifest,
                    "--posteriors",
                    posterior_nodes[training_split]["posteriors"],
                    "--retrieval",
                    retrieval_train,
                    "--output-dir",
                    seed_root,
                    "--model-name",
                    selector_config.get("model_name"),
                    "--training-split",
                    training_split,
                    "--dataset-id",
                    dataset_id,
                    "--generator-id",
                    generator_id,
                    "--formal-v2-identity",
                    "--runtime-implementation",
                    TRAINING_RUNTIME[CANONICAL_OURS]["runtime_implementation"],
                    "--forward-batch-size",
                    TRAINING_RUNTIME[CANONICAL_OURS]["forward_batch_size"],
                    "--seed",
                    seed,
                    "--resume",
                )
                add(
                    _node(
                        node_id=train_node,
                        stage="training",
                        dataset_id=dataset_id,
                        generator_id=generator_id,
                        method_id=CANONICAL_OURS,
                        seed=seed,
                        split=training_split,
                        inputs={
                            "teacher": teacher,
                            "teacher_manifest": teacher_manifest,
                            "posteriors": posterior_nodes[training_split]["posteriors"],
                            "retrieval": retrieval_train,
                        },
                        outputs={
                            "checkpoint": checkpoint,
                            "training_manifest": training_manifest,
                        },
                        dependencies=[teacher_node],
                        command=command,
                    )
                )
                selection = str(PurePosixPath(seed_root) / "selection" / f"{evaluation_split}.jsonl")
                select_node = f"{dataset_id}.{generator_id}.ours_signed_v1.seed{seed}.select"
                command = _command(
                    server_python,
                    server_project_root,
                    "preformal/27_select_signed_v1.py",
                    "--posteriors",
                    posterior_nodes[evaluation_split]["posteriors"],
                    "--checkpoint-dir",
                    checkpoint,
                    "--training-manifest",
                    training_manifest,
                    "--dataset-id",
                    dataset_id,
                    "--generator-id",
                    generator_id,
                    "--formal-v2-identity",
                    "--output",
                    selection,
                    "--seed",
                    seed,
                    "--split",
                    evaluation_split,
                    "--resume",
                )
                add(
                    _node(
                        node_id=select_node,
                        stage="selection",
                        dataset_id=dataset_id,
                        generator_id=generator_id,
                        method_id=CANONICAL_OURS,
                        seed=seed,
                        split=evaluation_split,
                        inputs={
                            "posteriors": posterior_nodes[evaluation_split]["posteriors"],
                            "checkpoint": checkpoint,
                            "training_manifest": training_manifest,
                        },
                        outputs={"selection": selection, "manifest": _sidecar(selection)},
                        dependencies=[train_node, posterior_nodes[evaluation_split]["node"]],
                        command=command,
                    )
                )
                learned_selections[(CANONICAL_OURS, seed)] = {
                    "node": select_node,
                    "selection": selection,
                    "manifest": _sidecar(selection),
                }

        if KCBWDM_SIGNED_V1 in learned_methods:
            kcbwdm_root = str(
                PurePosixPath(generator_root) / KCBWDM_SIGNED_V1
            )
            teacher_dir = str(PurePosixPath(kcbwdm_root) / "teacher")
            teacher = str(PurePosixPath(teacher_dir) / "teacher.jsonl")
            teacher_manifest = str(PurePosixPath(teacher_dir) / "manifest.json")
            teacher_node = f"{dataset_id}.{generator_id}.kcbwdm_signed_v1.teacher"
            command = _command(
                server_python,
                server_project_root,
                "preformal/28_materialize_kcbwdm_signed_v1_teacher.py",
                "--config",
                generator["server_config"],
                "--posteriors",
                posterior_nodes[training_split]["posteriors"],
                "--posterior-manifest",
                posterior_nodes[training_split]["manifest"],
                "--retrieval",
                retrieval_train,
                "--output-dir",
                teacher_dir,
                "--training-split",
                training_split,
                "--kernel",
                KCBWDM_SIGNED_V1_CONTRACT["kernel"]["base_kernel"],
                "--dataset-id",
                dataset_id,
                "--generator-id",
                generator_id,
                "--retrieval-protocol-id",
                retrieval_protocol_id,
                "--formal-v2-identity",
                "--resume",
            )
            _append_option(command, "--max-rows", limits["training"])
            add(
                _node(
                    node_id=teacher_node,
                    stage="teacher",
                    dataset_id=dataset_id,
                    generator_id=generator_id,
                    method_id=KCBWDM_SIGNED_V1,
                    split=training_split,
                    inputs={
                        "posteriors": posterior_nodes[training_split]["posteriors"],
                        "posterior_manifest": posterior_nodes[training_split]["manifest"],
                        "retrieval": retrieval_train,
                    },
                    outputs={"teacher": teacher, "manifest": teacher_manifest},
                    dependencies=[
                        posterior_nodes[training_split]["node"],
                        retrieval_nodes[training_split],
                    ],
                    command=command,
                )
            )
            selector_config = dataset_config.get("selector") or dataset_config.get(
                "signed_v1", {}
            )
            if not selector_config.get("model_name"):
                raise MatrixPlanError("KCBWDM requires selector.model_name")
            for seed in learned_seeds:
                seed_root = _phase_a_path(
                    server_artifact_root,
                    formal_v2_method_seed_root,
                    dataset_id,
                    generator_id,
                    KCBWDM_SIGNED_V1,
                    seed,
                )
                checkpoint = str(PurePosixPath(seed_root) / "checkpoint")
                training_manifest = str(
                    PurePosixPath(seed_root) / "training_manifest.json"
                )
                train_node = (
                    f"{dataset_id}.{generator_id}.kcbwdm_signed_v1.seed{seed}.train"
                )
                command = _command(
                    server_python,
                    server_project_root,
                    "preformal/26_train_signed_v1.py",
                    "--config",
                    generator["server_config"],
                    "--teacher",
                    teacher,
                    "--teacher-manifest",
                    teacher_manifest,
                    "--posteriors",
                    posterior_nodes[training_split]["posteriors"],
                    "--retrieval",
                    retrieval_train,
                    "--output-dir",
                    seed_root,
                    "--model-name",
                    selector_config.get("model_name"),
                    "--training-split",
                    training_split,
                    "--dataset-id",
                    dataset_id,
                    "--generator-id",
                    generator_id,
                    "--formal-v2-identity",
                    "--method-name",
                    KCBWDM_SIGNED_V1,
                    "--runtime-implementation",
                    TRAINING_RUNTIME[KCBWDM_SIGNED_V1]["runtime_implementation"],
                    "--forward-batch-size",
                    TRAINING_RUNTIME[KCBWDM_SIGNED_V1]["forward_batch_size"],
                    "--seed",
                    seed,
                    "--resume",
                )
                add(
                    _node(
                        node_id=train_node,
                        stage="training",
                        dataset_id=dataset_id,
                        generator_id=generator_id,
                        method_id=KCBWDM_SIGNED_V1,
                        seed=seed,
                        split=training_split,
                        inputs={
                            "teacher": teacher,
                            "teacher_manifest": teacher_manifest,
                            "posteriors": posterior_nodes[training_split]["posteriors"],
                            "retrieval": retrieval_train,
                        },
                        outputs={
                            "checkpoint": checkpoint,
                            "training_manifest": training_manifest,
                        },
                        dependencies=[teacher_node],
                        command=command,
                    )
                )
                selection = str(
                    PurePosixPath(seed_root)
                    / "selection"
                    / f"{evaluation_split}.jsonl"
                )
                select_node = (
                    f"{dataset_id}.{generator_id}.kcbwdm_signed_v1.seed{seed}.select"
                )
                command = _command(
                    server_python,
                    server_project_root,
                    "preformal/27_select_signed_v1.py",
                    "--posteriors",
                    posterior_nodes[evaluation_split]["posteriors"],
                    "--checkpoint-dir",
                    checkpoint,
                    "--training-manifest",
                    training_manifest,
                    "--dataset-id",
                    dataset_id,
                    "--generator-id",
                    generator_id,
                    "--formal-v2-identity",
                    "--method-name",
                    KCBWDM_SIGNED_V1,
                    "--output",
                    selection,
                    "--seed",
                    seed,
                    "--split",
                    evaluation_split,
                    "--resume",
                )
                _append_option(command, "--limit", limits["evaluation"])
                add(
                    _node(
                        node_id=select_node,
                        stage="selection",
                        dataset_id=dataset_id,
                        generator_id=generator_id,
                        method_id=KCBWDM_SIGNED_V1,
                        seed=seed,
                        split=evaluation_split,
                        inputs={
                            "posteriors": posterior_nodes[evaluation_split]["posteriors"],
                            "checkpoint": checkpoint,
                            "training_manifest": training_manifest,
                        },
                        outputs={"selection": selection, "manifest": _sidecar(selection)},
                        dependencies=[
                            train_node,
                            posterior_nodes[evaluation_split]["node"],
                        ],
                        command=command,
                    )
                )
                learned_selections[(KCBWDM_SIGNED_V1, seed)] = {
                    "node": select_node,
                    "selection": selection,
                    "manifest": _sidecar(selection),
                }

        if KCBWDM_LINEAR_GATE_V2 in learned_methods:
            method_root = str(
                PurePosixPath(generator_root) / KCBWDM_LINEAR_GATE_V2
            )
            teacher_dir = str(PurePosixPath(method_root) / "teacher")
            teacher = str(PurePosixPath(teacher_dir) / "teacher.jsonl")
            teacher_manifest = str(PurePosixPath(teacher_dir) / "manifest.json")
            teacher_node = (
                f"{dataset_id}.{generator_id}.{KCBWDM_LINEAR_GATE_V2}.teacher"
            )
            command = _command(
                server_python,
                server_project_root,
                "preformal/28_materialize_kcbwdm_signed_v1_teacher.py",
                "--config",
                generator["server_config"],
                "--posteriors",
                posterior_nodes[training_split]["posteriors"],
                "--posterior-manifest",
                posterior_nodes[training_split]["manifest"],
                "--retrieval",
                retrieval_train,
                "--output-dir",
                teacher_dir,
                "--training-split",
                training_split,
                "--kernel",
                KCBWDM_LINEAR_GATE_V2_CONTRACT["kernel"]["base_kernel"],
                "--dataset-id",
                dataset_id,
                "--generator-id",
                generator_id,
                "--retrieval-protocol-id",
                retrieval_protocol_id,
                "--method-name",
                KCBWDM_LINEAR_GATE_V2,
                "--formal-v2-identity",
                "--resume",
            )
            _append_option(command, "--max-rows", limits["training"])
            add(
                _node(
                    node_id=teacher_node,
                    stage="teacher",
                    dataset_id=dataset_id,
                    generator_id=generator_id,
                    method_id=KCBWDM_LINEAR_GATE_V2,
                    split=training_split,
                    inputs={
                        "posteriors": posterior_nodes[training_split]["posteriors"],
                        "posterior_manifest": posterior_nodes[training_split]["manifest"],
                        "retrieval": retrieval_train,
                    },
                    outputs={"teacher": teacher, "manifest": teacher_manifest},
                    dependencies=[
                        posterior_nodes[training_split]["node"],
                        retrieval_nodes[training_split],
                    ],
                    command=command,
                )
            )
            selector_config = dataset_config.get("selector") or dataset_config.get(
                "signed_v1", {}
            )
            if not selector_config.get("model_name"):
                raise MatrixPlanError(
                    "KCBWDM-v2A requires selector.model_name"
                )
            for seed in learned_seeds:
                seed_root = _phase_a_path(
                    server_artifact_root,
                    formal_v2_method_seed_root,
                    dataset_id,
                    generator_id,
                    KCBWDM_LINEAR_GATE_V2,
                    seed,
                )
                checkpoint = str(PurePosixPath(seed_root) / "checkpoint")
                training_manifest = str(
                    PurePosixPath(seed_root) / "training_manifest.json"
                )
                train_node = (
                    f"{dataset_id}.{generator_id}.{KCBWDM_LINEAR_GATE_V2}."
                    f"seed{seed}.train"
                )
                command = _command(
                    server_python,
                    server_project_root,
                    "preformal/26_train_signed_v1.py",
                    "--config",
                    generator["server_config"],
                    "--teacher",
                    teacher,
                    "--teacher-manifest",
                    teacher_manifest,
                    "--posteriors",
                    posterior_nodes[training_split]["posteriors"],
                    "--retrieval",
                    retrieval_train,
                    "--output-dir",
                    seed_root,
                    "--model-name",
                    selector_config.get("model_name"),
                    "--training-split",
                    training_split,
                    "--dataset-id",
                    dataset_id,
                    "--generator-id",
                    generator_id,
                    "--formal-v2-identity",
                    "--method-name",
                    KCBWDM_LINEAR_GATE_V2,
                    "--runtime-implementation",
                    TRAINING_RUNTIME[KCBWDM_LINEAR_GATE_V2][
                        "runtime_implementation"
                    ],
                    "--forward-batch-size",
                    TRAINING_RUNTIME[KCBWDM_LINEAR_GATE_V2]["forward_batch_size"],
                    "--seed",
                    seed,
                    "--resume",
                )
                add(
                    _node(
                        node_id=train_node,
                        stage="training",
                        dataset_id=dataset_id,
                        generator_id=generator_id,
                        method_id=KCBWDM_LINEAR_GATE_V2,
                        seed=seed,
                        split=training_split,
                        inputs={
                            "teacher": teacher,
                            "teacher_manifest": teacher_manifest,
                            "posteriors": posterior_nodes[training_split][
                                "posteriors"
                            ],
                            "retrieval": retrieval_train,
                        },
                        outputs={
                            "checkpoint": checkpoint,
                            "training_manifest": training_manifest,
                        },
                        dependencies=[teacher_node],
                        command=command,
                    )
                )
                selection = str(
                    PurePosixPath(seed_root)
                    / "selection"
                    / f"{evaluation_split}.jsonl"
                )
                select_node = (
                    f"{dataset_id}.{generator_id}.{KCBWDM_LINEAR_GATE_V2}."
                    f"seed{seed}.select"
                )
                command = _command(
                    server_python,
                    server_project_root,
                    "preformal/27_select_signed_v1.py",
                    "--posteriors",
                    posterior_nodes[evaluation_split]["posteriors"],
                    "--checkpoint-dir",
                    checkpoint,
                    "--training-manifest",
                    training_manifest,
                    "--dataset-id",
                    dataset_id,
                    "--generator-id",
                    generator_id,
                    "--formal-v2-identity",
                    "--method-name",
                    KCBWDM_LINEAR_GATE_V2,
                    "--output",
                    selection,
                    "--seed",
                    seed,
                    "--split",
                    evaluation_split,
                    "--resume",
                )
                _append_option(command, "--limit", limits["evaluation"])
                add(
                    _node(
                        node_id=select_node,
                        stage="selection",
                        dataset_id=dataset_id,
                        generator_id=generator_id,
                        method_id=KCBWDM_LINEAR_GATE_V2,
                        seed=seed,
                        split=evaluation_split,
                        inputs={
                            "posteriors": posterior_nodes[evaluation_split][
                                "posteriors"
                            ],
                            "checkpoint": checkpoint,
                            "training_manifest": training_manifest,
                        },
                        outputs={
                            "selection": selection,
                            "manifest": _sidecar(selection),
                        },
                        dependencies=[
                            train_node,
                            posterior_nodes[evaluation_split]["node"],
                        ],
                        command=command,
                    )
                )
                learned_selections[(KCBWDM_LINEAR_GATE_V2, seed)] = {
                    "node": select_node,
                    "selection": selection,
                    "manifest": _sidecar(selection),
                }

        if KCBWDM_NORMALIZED_RHO_V1 in learned_methods:
            selector_config = dataset_config.get("selector") or dataset_config.get(
                "signed_v1", {}
            )
            if not selector_config.get("model_name"):
                raise MatrixPlanError(
                    "Normalized-rho KCBWDM requires selector.model_name"
                )
            for variant in method_variants_by_method[KCBWDM_NORMALIZED_RHO_V1]:
                variant_id = str(variant["variant_id"])
                rho = float(variant["parameters"]["rho"])
                variant_root = str(
                    PurePosixPath(generator_root)
                    / KCBWDM_NORMALIZED_RHO_V1
                    / variant_id
                )
                teacher_dir = str(PurePosixPath(variant_root) / "teacher")
                teacher = str(PurePosixPath(teacher_dir) / "teacher.jsonl")
                teacher_manifest = str(PurePosixPath(teacher_dir) / "manifest.json")
                teacher_node = (
                    f"{dataset_id}.{generator_id}.{KCBWDM_NORMALIZED_RHO_V1}."
                    f"{variant_id}.teacher"
                )
                command = _command(
                    server_python,
                    server_project_root,
                    "preformal/28_materialize_kcbwdm_signed_v1_teacher.py",
                    "--config",
                    generator["server_config"],
                    "--posteriors",
                    posterior_nodes[training_split]["posteriors"],
                    "--posterior-manifest",
                    posterior_nodes[training_split]["manifest"],
                    "--retrieval",
                    retrieval_train,
                    "--output-dir",
                    teacher_dir,
                    "--training-split",
                    training_split,
                    "--kernel",
                    "rbf",
                    "--rho",
                    rho,
                    "--dataset-id",
                    dataset_id,
                    "--generator-id",
                    generator_id,
                    "--retrieval-protocol-id",
                    retrieval_protocol_id,
                    "--method-name",
                    KCBWDM_NORMALIZED_RHO_V1,
                    "--formal-v2-identity",
                    "--resume",
                )
                teacher_plan_node = _node(
                    node_id=teacher_node,
                    stage="teacher",
                    dataset_id=dataset_id,
                    generator_id=generator_id,
                    method_id=KCBWDM_NORMALIZED_RHO_V1,
                    split=training_split,
                    inputs={
                        "posteriors": posterior_nodes[training_split]["posteriors"],
                        "posterior_manifest": posterior_nodes[training_split]["manifest"],
                        "retrieval": retrieval_train,
                    },
                    outputs={"teacher": teacher, "manifest": teacher_manifest},
                    dependencies=[
                        posterior_nodes[training_split]["node"],
                        retrieval_nodes[training_split],
                    ],
                    command=command,
                )
                teacher_plan_node["method_variant_id"] = variant_id
                teacher_plan_node["method_parameters"] = {"rho": rho}
                add(teacher_plan_node)

                for seed in learned_seeds:
                    seed_root = str(PurePosixPath(variant_root) / f"seed{seed}")
                    checkpoint = str(PurePosixPath(seed_root) / "checkpoint")
                    training_manifest = str(
                        PurePosixPath(seed_root) / "training_manifest.json"
                    )
                    train_node = (
                        f"{dataset_id}.{generator_id}.{KCBWDM_NORMALIZED_RHO_V1}."
                        f"{variant_id}.seed{seed}.train"
                    )
                    command = _command(
                        server_python,
                        server_project_root,
                        "preformal/26_train_signed_v1.py",
                        "--config",
                        generator["server_config"],
                        "--teacher",
                        teacher,
                        "--teacher-manifest",
                        teacher_manifest,
                        "--posteriors",
                        posterior_nodes[training_split]["posteriors"],
                        "--retrieval",
                        retrieval_train,
                        "--output-dir",
                        seed_root,
                        "--model-name",
                        selector_config.get("model_name"),
                        "--training-split",
                        training_split,
                        "--dataset-id",
                        dataset_id,
                        "--generator-id",
                        generator_id,
                        "--formal-v2-identity",
                        "--method-name",
                        KCBWDM_NORMALIZED_RHO_V1,
                        "--rho",
                        rho,
                        "--runtime-implementation",
                        TRAINING_RUNTIME[KCBWDM_NORMALIZED_RHO_V1][
                            "runtime_implementation"
                        ],
                        "--forward-batch-size",
                        TRAINING_RUNTIME[KCBWDM_NORMALIZED_RHO_V1][
                            "forward_batch_size"
                        ],
                        "--seed",
                        seed,
                        "--resume",
                    )
                    train_plan_node = _node(
                        node_id=train_node,
                        stage="training",
                        dataset_id=dataset_id,
                        generator_id=generator_id,
                        method_id=KCBWDM_NORMALIZED_RHO_V1,
                        seed=seed,
                        split=training_split,
                        inputs={
                            "teacher": teacher,
                            "teacher_manifest": teacher_manifest,
                            "posteriors": posterior_nodes[training_split][
                                "posteriors"
                            ],
                            "retrieval": retrieval_train,
                        },
                        outputs={
                            "checkpoint": checkpoint,
                            "training_manifest": training_manifest,
                        },
                        dependencies=[teacher_node],
                        command=command,
                    )
                    train_plan_node["method_variant_id"] = variant_id
                    train_plan_node["method_parameters"] = {"rho": rho}
                    add(train_plan_node)

                    selection = str(
                        PurePosixPath(seed_root)
                        / "selection"
                        / f"{evaluation_split}.jsonl"
                    )
                    select_node = (
                        f"{dataset_id}.{generator_id}.{KCBWDM_NORMALIZED_RHO_V1}."
                        f"{variant_id}.seed{seed}.select"
                    )
                    command = _command(
                        server_python,
                        server_project_root,
                        "preformal/27_select_signed_v1.py",
                        "--posteriors",
                        posterior_nodes[evaluation_split]["posteriors"],
                        "--checkpoint-dir",
                        checkpoint,
                        "--training-manifest",
                        training_manifest,
                        "--dataset-id",
                        dataset_id,
                        "--generator-id",
                        generator_id,
                        "--formal-v2-identity",
                        "--method-name",
                        KCBWDM_NORMALIZED_RHO_V1,
                        "--rho",
                        rho,
                        "--output",
                        selection,
                        "--seed",
                        seed,
                        "--split",
                        evaluation_split,
                        "--resume",
                    )
                    select_plan_node = _node(
                        node_id=select_node,
                        stage="selection",
                        dataset_id=dataset_id,
                        generator_id=generator_id,
                        method_id=KCBWDM_NORMALIZED_RHO_V1,
                        seed=seed,
                        split=evaluation_split,
                        inputs={
                            "posteriors": posterior_nodes[evaluation_split][
                                "posteriors"
                            ],
                            "checkpoint": checkpoint,
                            "training_manifest": training_manifest,
                        },
                        outputs={
                            "selection": selection,
                            "manifest": _sidecar(selection),
                        },
                        dependencies=[
                            train_node,
                            posterior_nodes[evaluation_split]["node"],
                        ],
                        command=command,
                    )
                    select_plan_node["method_variant_id"] = variant_id
                    select_plan_node["method_parameters"] = {"rho": rho}
                    add(select_plan_node)
                    learned_selections[
                        (KCBWDM_NORMALIZED_RHO_V1, variant_id, seed)
                    ] = {
                        "node": select_node,
                        "selection": selection,
                        "manifest": _sidecar(selection),
                    }

        for method in canonical_methods:
            seeds: list[int | None] = (
                learned_seeds if method_spec(method)["learned_selector"] else [None]
            )
            for variant in method_variants_by_method[method]:
                variant_id = str(variant["variant_id"])
                variant_parameters = dict(variant["parameters"])
                explicit_variant = bool(variant["explicit"])
                for seed in seeds:
                    if seed is None:
                        selection_ref = shared_selection[method]
                    elif explicit_variant:
                        selection_ref = learned_selections[(method, variant_id, seed)]
                    else:
                        selection_ref = learned_selections[(method, seed)]
                    evaluation_base = _phase_a_path(
                        server_artifact_root,
                        formal_v2_evaluation_root,
                        dataset_id,
                        generator_id,
                        method,
                        "matched_main",
                    )
                    evaluation_path = PurePosixPath(evaluation_base)
                    if explicit_variant:
                        evaluation_path /= variant_id
                    eval_root = str(
                        evaluation_path
                        / ("deterministic" if seed is None else f"seed{seed}")
                        / evaluation_split
                    )
                    predictions = str(PurePosixPath(eval_root) / "predictions.jsonl")
                    metrics = str(PurePosixPath(eval_root) / "metrics.json")
                    eval_manifest = _sidecar(metrics)
                    seed_part = "" if seed is None else f".seed{seed}"
                    variant_part = f".{variant_id}" if explicit_variant else ""
                    node_id = (
                        f"{dataset_id}.{generator_id}.{method}{variant_part}"
                        f"{seed_part}.evaluate"
                    )
                    worker_method = (
                        "infogain_fever" if method == "infogain" else method
                    )
                    command = _command(
                        server_python,
                        server_project_root,
                        "07_eval_rag_classification.py",
                        "--config",
                        generator["server_config"],
                        "--split",
                        evaluation_split,
                        "--selection",
                        selection_ref["selection"],
                        "--selection-manifest",
                        selection_ref["manifest"],
                        "--output",
                        predictions,
                        "--metrics-output",
                        metrics,
                        "--generator-manifest",
                        manifest_output,
                        "--experiment-type",
                        "matched_main",
                        "--formal-v2-identity",
                        "--method-name",
                        worker_method,
                        "--resume",
                    )
                    if method == "no_evidence":
                        command.append("--no-evidence")
                    _append_option(command, "--limit", limits["evaluation"])
                    eval_plan_node = _node(
                        node_id=node_id,
                        stage="evaluation",
                        dataset_id=dataset_id,
                        generator_id=generator_id,
                        method_id=method,
                        seed=seed,
                        split=evaluation_split,
                        inputs={
                            "selection": selection_ref["selection"],
                            "selection_manifest": selection_ref["manifest"],
                            "generator_manifest": manifest_output,
                        },
                        outputs={
                            "predictions": predictions,
                            "metrics": metrics,
                            "evaluation_manifest": eval_manifest,
                        },
                        dependencies=[selection_ref["node"], generator_node],
                        command=command,
                        reusable=False,
                    )
                    if explicit_variant:
                        eval_plan_node["method_variant_id"] = variant_id
                        eval_plan_node["method_parameters"] = variant_parameters
                    add(eval_plan_node)
                    result_row = {
                        "dataset_id": dataset_id,
                        "generator_id": generator_id,
                        "method_id": method,
                        "method": method,
                        "seed": seed,
                        "split": evaluation_split,
                        "retrieval_protocol_id": retrieval_protocol_id,
                        "retrieval_protocol_fingerprint": protocol_fingerprint,
                        "experiment_type": "matched_main",
                        "evaluation_manifest": eval_manifest,
                        "node_id": node_id,
                    }
                    if explicit_variant:
                        result_row["method_variant_id"] = variant_id
                        result_row["method_parameters"] = variant_parameters
                        result_row.update(variant_parameters)
                    result_index.append(result_row)

    topological_order = validate_dag(nodes)
    node_by_id = {node["node_id"]: node for node in nodes}
    ordered_nodes = [node_by_id[node_id] for node_id in topological_order]
    unresolved = []
    if "bge" in canonical_methods and (
        not bge_spec.get("revision") or not bge_spec.get("sha256")
    ):
        unresolved.append("bge_model_id_revision_sha")
    if held_out and unresolved:
        raise MatrixPlanError("Held-out plan has unresolved BGE freeze")
    edges = [
        {"from": dependency, "to": node["node_id"]}
        for node in ordered_nodes
        for dependency in node["dependencies"]
    ]
    plan = {
        "schema_version": PLAN_MANIFEST_SCHEMA_VERSION,
        "orchestrator_schema_version": ORCHESTRATOR_SCHEMA_VERSION,
        "created_at": utc_now(),
        "git": current_git,
        "registry_version": FORMAL_REGISTRY_VERSION,
        "registry_fingerprint": FORMAL_REGISTRY_FINGERPRINT,
        "dataset_identity": generators[0]["dataset_identity"],
        "retrieval_protocol": protocol,
        "generator_identities": sorted(
            generator_identities, key=lambda value: value["generator_id"]
        ),
        "generator_registry_fingerprint": generator_registry_fingerprint,
        "methods": canonical_methods,
        "method_contracts": method_contracts,
        "method_variants": explicit_method_variants,
        "seed_policy": {
            method: (
                learned_seeds
                if method_spec(method)["learned_selector"]
                else []
            )
            for method in canonical_methods
        },
        "profile": profile,
        "split_role": protocol["split_policy"][evaluation_split],
        "training_split": training_split,
        "evaluation_split": evaluation_split,
        "held_out": held_out,
        "formal_result_claim": held_out,
        "limits": limits,
        "retrieval_inputs": (
            {
                training_split: retrieval_bindings[training_split],
                evaluation_split: retrieval_bindings[evaluation_split],
            }
            if retrieval_protocol_id == "fm2_official_closed_page_v1"
            else {
                training_split: str(retrieval_inputs[training_split]),
                evaluation_split: str(retrieval_inputs[evaluation_split]),
            }
        ),
        "posterior_reuse_bindings": sorted(
            posterior_reuse_bindings,
            key=lambda value: (value["generator_id"], value["split"]),
        ),
        "dataset_config_sha256": dataset_config_sha256,
        "bge_contract": bge_spec,
        "training_runtime": TRAINING_RUNTIME,
        "nodes": ordered_nodes,
        "dependency_edges": edges,
        "artifact_roots": {
            "formal_v2": server_artifact_root,
            "dataset": dataset_root,
            "shared": shared_root,
        },
        "result_index": sorted(
            result_index,
            key=lambda value: (
                value["dataset_id"],
                value["generator_id"],
                canonical_methods.index(value["method_id"]),
                str(value.get("method_variant_id") or ""),
                value["seed"] if value["seed"] is not None else -1,
            ),
        ),
        "unresolved_freeze_decisions": unresolved,
        "dry_run_executes_workers": False,
    }
    plan["plan_fingerprint"] = stable_hash(_semantic_plan_payload(plan))
    return plan


def load_matrix_config(path: str | Path) -> dict[str, Any]:
    payload = load_yaml(path)
    if not isinstance(payload, dict):
        raise MatrixPlanError("Formal matrix config must be a mapping")
    return payload


def _validate_external_outputs(node: Mapping[str, Any]) -> None:
    missing = [path for path in node["outputs"].values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            f"External dependency for {node['node_id']} is missing: {missing}"
        )


def _validate_external_fm2_retrieval(node: Mapping[str, Any]) -> None:
    binding = node.get("retrieval_binding")
    if not isinstance(binding, Mapping):
        raise MatrixPlanError("FM2 retrieval node lacks its authoritative binding")
    manifest = node.get("inputs", {}).get("manifest")
    if not manifest or manifest != binding.get("manifest"):
        raise MatrixPlanError("FM2 retrieval node manifest input/binding mismatch")
    actual = validate_fm2_retrieval_manifest(
        manifest,
        node["outputs"]["retrieval"],
        expected_formal_role=str(node["split"]),
        expected_dataset_id=str(node["dataset_id"]),
        expected_retrieval_protocol_id=str(binding["retrieval_protocol_id"]),
    )
    for field in (
        "dataset_id",
        "retrieval_protocol_id",
        "official_source_split",
        "formal_role",
        "manifest_sha256",
        "pool_sha256",
        "num_rows",
    ):
        if actual.get(field) != binding.get(field):
            raise MatrixPlanError(
                f"FM2 retrieval binding changed after planning: {field}"
            )


def _validate_external_posterior(node: Mapping[str, Any]) -> None:
    binding = node.get("posterior_binding")
    if not isinstance(binding, Mapping):
        raise MatrixPlanError("External posterior node lacks its immutable binding")
    actual = validate_posterior_provenance(
        node["outputs"]["posteriors"],
        node["outputs"]["manifest"],
        mode=FORMAL_V2_MODE,
        expected_dataset_id=str(node["dataset_id"]),
        expected_split=str(node["split"]),
        expected_generator_id=str(node["generator_id"]),
        expected_retrieval_protocol_id=str(
            binding["retrieval_protocol_identity"]["retrieval_protocol_id"]
        ),
    )
    for field in (
        "manifest_sha256",
        "manifest_fingerprint",
        "identity_fingerprint",
        "posterior_sha256",
        "dataset_identity",
        "generator_identity",
        "retrieval_protocol_identity",
        "split",
    ):
        if actual.get(field) != binding.get(field):
            raise MatrixPlanError(
                f"External posterior binding changed after planning: {field}"
            )


def _reuse_existing_generator_manifest(node: Mapping[str, Any]) -> bool:
    manifest_path = Path(node["outputs"]["manifest"])
    if not manifest_path.exists():
        return False
    validate_generator_manifest(
        manifest_path,
        expected_dataset_id=str(node["dataset_id"]),
        expected_generator_id=str(node["generator_id"]),
        expected_config_path=node["inputs"]["config"],
    )
    return True


def execute_plan(
    plan: Mapping[str, Any],
    *,
    run: Callable[..., Any] = subprocess.run,
) -> None:
    """Execute sequentially; dry-run callers never call this function."""
    if plan.get("held_out") and plan.get("unresolved_freeze_decisions"):
        raise MatrixPlanError("Held-out plan has unresolved freeze decisions")
    if os.name == "nt":
        raise MatrixPlanError("Formal matrix execution is supported only on the Linux server")
    order = validate_dag(plan["nodes"])
    by_id = {node["node_id"]: node for node in plan["nodes"]}
    for node_id in order:
        node = by_id[node_id]
        command = node["command"]
        policy = node["execution_policy"]
        if not command:
            if policy == "validate_external_posterior":
                _validate_external_posterior(node)
                continue
            if policy == "validate_external_fm2_retrieval":
                _validate_external_fm2_retrieval(node)
                continue
            if policy == "validate_external_generator_manifest":
                if not _reuse_existing_generator_manifest(node):
                    raise FileNotFoundError(
                        f"External generator manifest for {node_id} is missing"
                    )
                continue
            _validate_external_outputs(node)
            continue
        if policy == "validate_existing_generator_manifest_or_create" and _reuse_existing_generator_manifest(node):
            continue
        run(command, check=True)


def render_commands(plan: Mapping[str, Any]) -> str:
    lines = []
    for node in plan["nodes"]:
        command = node["command"]
        if command:
            lines.append(f"# {node['node_id']}")
            lines.append(shlex.join(command))
        else:
            lines.append(f"# {node['node_id']} (external input; no command)")
    return "\n".join(lines) + "\n"

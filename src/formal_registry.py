"""Versioned formal-v2 method, dataset, split, and eligibility registry."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from src.artifact_binding import (
    CROSS_GENERATOR_TRANSFER,
    EVALUATION_BINDING_SCHEMA_VERSION,
    EVALUATION_MANIFEST_SCHEMA_VERSION,
    GENERATOR_DEPENDENCY_CONDITIONED,
    GENERATOR_DEPENDENCY_INDEPENDENT,
    MATCHED_MAIN,
)
from src.run_manifest import stable_hash


FORMAL_REGISTRY_SCHEMA_VERSION = "rag_cbwdm_formal_registry.v2"
FORMAL_REGISTRY_VERSION = "formal_v2.1"
HISTORICAL_FORMAL_REGISTRY_IDENTITIES = {
    # Frozen compatibility identity for artifacts created before the
    # development-only KCBWDM-v2A registry entry was added.
    "formal_v2.0": "0aa02250daa72d3eddc6bd60b34a6cb3f38a2a9773f4fad3dd9b751a19eca739",
}
HELD_OUT_FREEZE_SCHEMA_VERSION = "rag_cbwdm_held_out_freeze.v1"
FORMAL_READINESS_SCHEMA_VERSION = "rag_cbwdm_formal_readiness.v2"

CANONICAL_OURS = "rag_cbwdm_signed_v1"
MAIN_TABLE_METHODS = (
    "no_evidence",
    "retrieval_topk",
    "bge",
    "infogain",
    CANONICAL_OURS,
)
LEARNED_METHOD_SEEDS = (13, 21, 42)
KCBWDM_SIGNED_V1 = "kcbwdm_signed_v1"
KCBWDM_LINEAR_GATE_V2 = "kcbwdm_linear_gate_v2"

_NO_TRAINING_SEED = {
    "kind": "deterministic_no_training_seed",
    "seeds": [],
    "replicate_across_learned_seeds": False,
}
_LEARNED_SEEDS = {
    "kind": "fixed_training_seeds",
    "seeds": list(LEARNED_METHOD_SEEDS),
    "replicate_across_learned_seeds": True,
}


def _method(
    method_id: str,
    display_name: str,
    *,
    main_table_eligible: bool,
    generator_dependency: str,
    learned_selector: bool,
    selection_kind: str,
    diagnostic_only: bool = False,
    transfer_eligible: bool = False,
    artifact_method_ids: tuple[str, ...] | None = None,
    legacy: bool = False,
    development_matrix_eligible: bool | None = None,
    held_out_eligible: bool | None = None,
    state_aware: bool = False,
    learned_seeds: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    seed_policy = copy.deepcopy(
        _LEARNED_SEEDS if learned_selector else _NO_TRAINING_SEED
    )
    if learned_selector and learned_seeds is not None:
        seed_policy["seeds"] = list(learned_seeds)
    return {
        "method_id": method_id,
        "display_name": display_name,
        "main_table_eligible": main_table_eligible,
        "generator_dependency": generator_dependency,
        "learned_selector": learned_selector,
        "seed_policy": seed_policy,
        "selection_kind": selection_kind,
        "diagnostic_only": diagnostic_only,
        "transfer_eligible": transfer_eligible,
        "required_provenance_level": "formal_v2",
        "artifact_method_ids": list(artifact_method_ids or (method_id,)),
        "legacy": legacy,
        "development_matrix_eligible": (
            main_table_eligible
            if development_matrix_eligible is None
            else development_matrix_eligible
        ),
        "held_out_eligible": (
            main_table_eligible if held_out_eligible is None else held_out_eligible
        ),
        "state_aware": state_aware,
        "development_only": bool(
            (main_table_eligible if development_matrix_eligible is None else development_matrix_eligible)
            and not main_table_eligible
        ),
    }


_METHODS: dict[str, dict[str, Any]] = {
    "no_evidence": _method(
        "no_evidence",
        "No Evidence",
        main_table_eligible=True,
        generator_dependency=GENERATOR_DEPENDENCY_INDEPENDENT,
        learned_selector=False,
        selection_kind="empty_evidence",
    ),
    "retrieval_topk": _method(
        "retrieval_topk",
        "Retrieval Top-k",
        main_table_eligible=True,
        generator_dependency=GENERATOR_DEPENDENCY_INDEPENDENT,
        learned_selector=False,
        selection_kind="dataset_protocol_source_order_topk",
        artifact_method_ids=("retrieval_topk", "naive_topm"),
    ),
    "bge": _method(
        "bge",
        "BGE",
        main_table_eligible=True,
        generator_dependency=GENERATOR_DEPENDENCY_INDEPENDENT,
        learned_selector=False,
        selection_kind="frozen_bge_reranker",
    ),
    "infogain": _method(
        "infogain",
        "InfoGain",
        main_table_eligible=True,
        generator_dependency=GENERATOR_DEPENDENCY_CONDITIONED,
        learned_selector=True,
        selection_kind="learned_pointwise_reranker",
        transfer_eligible=True,
        artifact_method_ids=("infogain", "infogain_fever"),
    ),
    CANONICAL_OURS: _method(
        CANONICAL_OURS,
        "Ours",
        main_table_eligible=True,
        generator_dependency=GENERATOR_DEPENDENCY_CONDITIONED,
        learned_selector=True,
        selection_kind="learned_state_aware_selector",
        transfer_eligible=True,
        state_aware=True,
    ),
    KCBWDM_SIGNED_V1: _method(
        KCBWDM_SIGNED_V1,
        "KCBWDM signed-v1 (development)",
        main_table_eligible=False,
        generator_dependency=GENERATOR_DEPENDENCY_CONDITIONED,
        learned_selector=True,
        selection_kind="learned_state_aware_selector",
        development_matrix_eligible=True,
        held_out_eligible=False,
        state_aware=True,
        learned_seeds=(13,),
    ),
    KCBWDM_LINEAR_GATE_V2: _method(
        KCBWDM_LINEAR_GATE_V2,
        "KCBWDM linear-gate v2 (development)",
        main_table_eligible=False,
        generator_dependency=GENERATOR_DEPENDENCY_CONDITIONED,
        learned_selector=True,
        selection_kind="learned_state_aware_selector",
        development_matrix_eligible=True,
        held_out_eligible=False,
        state_aware=True,
        learned_seeds=(13,),
    ),
    "rag_cbwdm": _method(
        "rag_cbwdm",
        "Legacy RAG-CBWDM",
        main_table_eligible=False,
        generator_dependency=GENERATOR_DEPENDENCY_CONDITIONED,
        learned_selector=True,
        selection_kind="legacy_learned_selector",
        legacy=True,
    ),
    "rag_cbwdm_signed_v2": _method(
        "rag_cbwdm_signed_v2",
        "signed-v2 diagnostic",
        main_table_eligible=False,
        generator_dependency=GENERATOR_DEPENDENCY_CONDITIONED,
        learned_selector=True,
        selection_kind="diagnostic_ablation",
        diagnostic_only=True,
    ),
    "rag_cbwdm_signed_v21": _method(
        "rag_cbwdm_signed_v21",
        "signed-v2.1 diagnostic",
        main_table_eligible=False,
        generator_dependency=GENERATOR_DEPENDENCY_CONDITIONED,
        learned_selector=True,
        selection_kind="diagnostic_ablation",
        diagnostic_only=True,
    ),
    "rag_cbwdm_signed_v22": _method(
        "rag_cbwdm_signed_v22",
        "signed-v2.2 diagnostic",
        main_table_eligible=False,
        generator_dependency=GENERATOR_DEPENDENCY_CONDITIONED,
        learned_selector=True,
        selection_kind="diagnostic_ablation",
        diagnostic_only=True,
    ),
    "signed_gate_oracle": _method(
        "signed_gate_oracle",
        "Signed-gate Oracle",
        main_table_eligible=False,
        generator_dependency=GENERATOR_DEPENDENCY_CONDITIONED,
        learned_selector=False,
        selection_kind="gold_oracle",
        diagnostic_only=True,
    ),
    "cbwdm_oracle": _method(
        "cbwdm_oracle",
        "CBWDM Oracle",
        main_table_eligible=False,
        generator_dependency=GENERATOR_DEPENDENCY_CONDITIONED,
        learned_selector=False,
        selection_kind="gold_oracle",
        diagnostic_only=True,
    ),
    "gold_oracle": _method(
        "gold_oracle",
        "Gold Oracle",
        main_table_eligible=False,
        generator_dependency=GENERATOR_DEPENDENCY_CONDITIONED,
        learned_selector=False,
        selection_kind="gold_oracle",
        diagnostic_only=True,
    ),
}

_DATASETS: dict[str, dict[str, Any]] = {
    "fever_binary_v2": {
        "dataset_id": "fever_binary_v2",
        "dataset_family": "fever",
        "retrieval_protocol_id": "fever_bm25_v1",
        "retrieval_topk_display": "BM25 Top-k",
        "main_table_column": "Retrieval Top-k",
        "candidate_pool": "frozen_open_domain_bm25",
        "allowed_main_table_methods": list(MAIN_TABLE_METHODS),
        "split_policy": {
            "train_core": "learning",
            "validation": "development_calibration",
            "preformal_eval": "development_evaluation",
            "held_out_test": "held_out_final_evaluation",
        },
        "final_split": "held_out_test",
        "development_500_is_final_test": False,
    },
    "fm2_official_closed_page_v1": {
        "dataset_id": "fm2_official_closed_page_v1",
        "dataset_family": "fm2",
        "retrieval_protocol_id": "fm2_official_closed_page_v1",
        "retrieval_topk_display": "Official-pool Top-k",
        "main_table_column": "Retrieval Top-k",
        "candidate_pool": "official_closed_page_source_order",
        "allowed_main_table_methods": list(MAIN_TABLE_METHODS),
        "split_policy": {
            "train_core": "learning",
            "validation": "development_calibration",
            "held_out_test": "held_out_final_evaluation",
        },
        "official_partition_mapping": {
            "train": "train_core",
            "dev": "validation",
            "test": "held_out_test",
        },
        "final_split": "held_out_test",
        "fm2_bm25_implemented": False,
    },
}

_ELIGIBILITY_RULES = {
    "required_status": "completed",
    "required_identity_mode": "formal_v2",
    "required_experiment_type": MATCHED_MAIN,
    "conditioned_generators_must_match": True,
    "diagnostic_or_oracle_forbidden": True,
    "held_out_freeze_and_sign_off_required": True,
    "dataset_protocol_compatibility_required": True,
}

_REGISTRY_CONTRACT: dict[str, Any] = {
    "schema_version": FORMAL_REGISTRY_SCHEMA_VERSION,
    "registry_version": FORMAL_REGISTRY_VERSION,
    "canonical_ours": CANONICAL_OURS,
    "main_table_method_ids": list(MAIN_TABLE_METHODS),
    "methods": _METHODS,
    "datasets": _DATASETS,
    "eligibility_rules": _ELIGIBILITY_RULES,
    "held_out_freeze_required_fields": [
        "dataset_id",
        "formal_registry_version",
        "method_registry_fingerprint",
        "generator_registry_fingerprint",
        "retrieval_protocol_fingerprint",
        "prompt_hash",
        "verbalizer_hash",
        "top_k",
        "evidence_budget",
        "learned_method_seeds",
        "bge_model_freeze",
        "git_commit",
        "config_sha256",
        "freeze_fingerprint",
        "runtime_sign_off",
    ],
    "unresolved_freeze_decisions": ["bge_model_id_revision_sha"],
}


def build_formal_registry() -> dict[str, Any]:
    """Return a detached copy of the path-independent formal-v2 registry."""
    return copy.deepcopy(_REGISTRY_CONTRACT)


def formal_registry_fingerprint(registry: Mapping[str, Any] | None = None) -> str:
    """Hash all registry fields that define formal scientific semantics."""
    return stable_hash(dict(registry or _REGISTRY_CONTRACT))


FORMAL_REGISTRY_FINGERPRINT = formal_registry_fingerprint()


def formal_registry_identities() -> dict[str, str]:
    """Return the exact version/fingerprint pairs accepted by validators."""
    return {
        **HISTORICAL_FORMAL_REGISTRY_IDENTITIES,
        FORMAL_REGISTRY_VERSION: FORMAL_REGISTRY_FINGERPRINT,
    }


def validate_formal_registry_identity(version: Any, fingerprint: Any) -> None:
    """Fail closed unless ``version`` and ``fingerprint`` are an allowed pair."""
    identities = formal_registry_identities()
    if version not in identities:
        raise ValueError(f"Unknown formal registry version: {version!r}")
    expected = identities[str(version)]
    if fingerprint != expected:
        raise ValueError(
            "Formal registry version/fingerprint mismatch: "
            f"version={version!r} expected={expected!r} actual={fingerprint!r}"
        )


def method_spec(method_id: str) -> dict[str, Any]:
    try:
        return copy.deepcopy(_METHODS[method_id])
    except KeyError as exc:
        raise ValueError(f"Method is not registered for formal-v2: {method_id!r}") from exc


def canonical_method_id(artifact_method_id: str) -> str:
    matches = [
        method_id
        for method_id, spec in _METHODS.items()
        if artifact_method_id in spec["artifact_method_ids"]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Artifact method ID is not uniquely registered: {artifact_method_id!r}"
        )
    return matches[0]


def dataset_protocol(dataset_id: str) -> dict[str, Any]:
    try:
        return copy.deepcopy(_DATASETS[dataset_id])
    except KeyError as exc:
        raise ValueError(f"Dataset is not registered for formal-v2: {dataset_id!r}") from exc


def retrieval_protocol_fingerprint(dataset_id: str) -> str:
    protocol = dataset_protocol(dataset_id)
    return stable_hash(
        {
            "dataset_id": protocol["dataset_id"],
            "retrieval_protocol_id": protocol["retrieval_protocol_id"],
            "retrieval_topk_display": protocol["retrieval_topk_display"],
            "main_table_column": protocol["main_table_column"],
            "candidate_pool": protocol["candidate_pool"],
        }
    )


def validate_dataset_method_compatibility(
    dataset_id: str,
    method_id: str,
    retrieval_protocol_id: str,
    *,
    development: bool = False,
) -> dict[str, Any]:
    canonical = canonical_method_id(method_id)
    spec = method_spec(canonical)
    protocol = dataset_protocol(dataset_id)
    if development:
        if not spec["development_matrix_eligible"]:
            raise ValueError(
                f"Method is not formal-v2 development-matrix eligible: {canonical}"
            )
    else:
        if not spec["main_table_eligible"] or not spec["held_out_eligible"]:
            raise ValueError(f"Method is not formal-v2 main-table eligible: {canonical}")
        if canonical not in protocol["allowed_main_table_methods"]:
            raise ValueError(
                f"Method {canonical!r} is not compatible with dataset {dataset_id!r}"
            )
    if retrieval_protocol_id != protocol["retrieval_protocol_id"]:
        raise ValueError(
            "Dataset/retrieval protocol mismatch: "
            f"dataset={dataset_id!r} expected={protocol['retrieval_protocol_id']!r} "
            f"actual={retrieval_protocol_id!r}"
        )
    return {"method": spec, "dataset_protocol": protocol}


def split_role(dataset_id: str, split: str) -> str:
    protocol = dataset_protocol(dataset_id)
    try:
        return str(protocol["split_policy"][split])
    except KeyError as exc:
        raise ValueError(
            f"Split {split!r} is not registered for dataset {dataset_id!r}"
        ) from exc


def is_final_split(dataset_id: str, split: str) -> bool:
    return dataset_protocol(dataset_id)["final_split"] == split


def held_out_freeze_template(dataset_id: str) -> dict[str, Any]:
    """Return an incomplete source-level template; it never authorizes execution."""
    protocol = dataset_protocol(dataset_id)
    return {
        "schema_version": HELD_OUT_FREEZE_SCHEMA_VERSION,
        "status": "draft",
        "dataset_id": dataset_id,
        "formal_registry_version": FORMAL_REGISTRY_VERSION,
        "method_registry_fingerprint": FORMAL_REGISTRY_FINGERPRINT,
        "generator_registry_fingerprint": None,
        "retrieval_protocol_fingerprint": retrieval_protocol_fingerprint(dataset_id),
        "retrieval_protocol_id": protocol["retrieval_protocol_id"],
        "prompt_hash": None,
        "verbalizer_hash": None,
        "top_k": None,
        "evidence_budget": None,
        "learned_method_seeds": {
            "infogain": list(LEARNED_METHOD_SEEDS),
            CANONICAL_OURS: list(LEARNED_METHOD_SEEDS),
        },
        "bge_model_freeze": {
            "model_id": None,
            "revision": None,
            "sha256": None,
        },
        "git_commit": None,
        "config_sha256": None,
        "freeze_fingerprint": None,
        "runtime_sign_off": {
            "approved": False,
            "approved_by": None,
            "approved_at": None,
            "evidence_reference": None,
            "freeze_fingerprint": None,
        },
    }


def held_out_freeze_fingerprint(freeze: Mapping[str, Any]) -> str:
    """Fingerprint the scientific freeze fields, excluding the later sign-off."""
    payload = copy.deepcopy(dict(freeze))
    payload.pop("runtime_sign_off", None)
    payload.pop("freeze_fingerprint", None)
    return stable_hash(payload)


def held_out_freeze_status(
    freeze: Mapping[str, Any] | None, *, expected_dataset_id: str | None = None
) -> dict[str, Any]:
    """Check source-level freeze/sign-off metadata without opening held-out data."""
    if freeze is None:
        return {
            "ready": False,
            "status": "blocked",
            "blockers": ["held-out freeze/sign-off manifest is missing"],
        }
    payload = dict(freeze)
    blockers: list[str] = []
    if payload.get("schema_version") != HELD_OUT_FREEZE_SCHEMA_VERSION:
        blockers.append("held-out freeze schema mismatch")
    if payload.get("status") != "frozen":
        blockers.append("held-out freeze status is not frozen")
    dataset_id = payload.get("dataset_id")
    if expected_dataset_id is not None and dataset_id != expected_dataset_id:
        blockers.append("held-out freeze dataset_id mismatch")
    try:
        protocol = dataset_protocol(str(dataset_id))
    except ValueError as exc:
        blockers.append(str(exc))
        protocol = None
    required = _REGISTRY_CONTRACT["held_out_freeze_required_fields"]
    blockers.extend(
        f"held-out freeze field is missing: {field}"
        for field in required
        if payload.get(field) is None
    )
    for field in (
        "generator_registry_fingerprint",
        "prompt_hash",
        "verbalizer_hash",
        "git_commit",
        "config_sha256",
    ):
        if not payload.get(field):
            blockers.append(f"held-out freeze field is empty: {field}")
    try:
        validate_formal_registry_identity(
            payload.get("formal_registry_version"),
            payload.get("method_registry_fingerprint"),
        )
    except ValueError as exc:
        blockers.append(str(exc))
    if protocol is not None:
        if payload.get("retrieval_protocol_id") != protocol["retrieval_protocol_id"]:
            blockers.append("retrieval protocol ID mismatch")
        if payload.get("retrieval_protocol_fingerprint") != retrieval_protocol_fingerprint(
            str(dataset_id)
        ):
            blockers.append("retrieval protocol fingerprint mismatch")
    expected_seeds = {
        "infogain": list(LEARNED_METHOD_SEEDS),
        CANONICAL_OURS: list(LEARNED_METHOD_SEEDS),
    }
    if payload.get("learned_method_seeds") != expected_seeds:
        blockers.append("learned method seed set mismatch")
    bge = payload.get("bge_model_freeze")
    if not isinstance(bge, Mapping) or any(
        not bge.get(field) for field in ("model_id", "revision", "sha256")
    ):
        blockers.append("BGE model ID/revision/SHA is not frozen")
    if not isinstance(payload.get("evidence_budget"), Mapping) or not payload[
        "evidence_budget"
    ].get("max_docs"):
        blockers.append("evidence budget is not frozen")
    if not isinstance(payload.get("top_k"), int) or payload["top_k"] <= 0:
        blockers.append("Top-k is not frozen")
    sign_off = payload.get("runtime_sign_off")
    if not isinstance(sign_off, Mapping) or sign_off.get("approved") is not True:
        blockers.append("human/runtime held-out sign-off is absent")
    elif any(
        not sign_off.get(field)
        for field in (
            "approved_by",
            "approved_at",
            "evidence_reference",
            "freeze_fingerprint",
        )
    ):
        blockers.append("human/runtime held-out sign-off is incomplete")
    expected_freeze_fingerprint = held_out_freeze_fingerprint(payload)
    if payload.get("freeze_fingerprint") != expected_freeze_fingerprint:
        blockers.append("held-out freeze fingerprint mismatch")
    if isinstance(sign_off, Mapping) and sign_off.get(
        "freeze_fingerprint"
    ) != payload.get("freeze_fingerprint"):
        blockers.append("human/runtime sign-off references a different freeze")
    return {
        "ready": not blockers,
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "dataset_id": dataset_id,
    }


def validate_main_table_result(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one explicit candidate; never discover artifacts by fuzzy names."""
    binding = candidate.get("artifact_binding")
    binding = binding if isinstance(binding, Mapping) else {}
    contract = candidate.get("contract")
    contract = contract if isinstance(contract, Mapping) else {}
    artifact_method = candidate.get("method") or binding.get("method")
    canonical = canonical_method_id(str(artifact_method))
    binding_method = binding.get("method")
    if binding_method is None or canonical_method_id(str(binding_method)) != canonical:
        raise ValueError("Evaluation manifest and binding method identity differ")
    spec = method_spec(canonical)
    if not spec["main_table_eligible"] or spec["diagnostic_only"]:
        raise ValueError(f"Method is not formal-v2 main-table eligible: {canonical}")
    if candidate.get("status") != "completed" or candidate.get("completed") is not True:
        raise ValueError("Formal main-table result is unfinished")
    if candidate.get("schema_version") != EVALUATION_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Formal main-table result requires evaluation manifest v2")
    if candidate.get("identity_mode") != "formal_v2":
        raise ValueError("Formal main-table result requires formal-v2 provenance")
    if binding.get("schema_version") != EVALUATION_BINDING_SCHEMA_VERSION:
        raise ValueError("Formal main-table result lacks evaluation binding v1")
    if candidate.get("diagnostic_only") is True or candidate.get("deployable") is False:
        raise ValueError("Diagnostic/oracle result is not main-table eligible")
    experiment_type = candidate.get("experiment_type") or binding.get(
        "experiment_type"
    )
    if candidate.get("experiment_type") != binding.get("experiment_type"):
        raise ValueError("Evaluation manifest and binding experiment type differ")
    if experiment_type != MATCHED_MAIN:
        if experiment_type == CROSS_GENERATOR_TRANSFER:
            raise ValueError("Cross-generator transfer is never main-table eligible")
        raise ValueError("Formal main-table result must be matched_main")
    dataset_id = candidate.get("dataset_id") or binding.get("dataset_id")
    if candidate.get("dataset_id") != binding.get("dataset_id"):
        raise ValueError("Evaluation manifest and binding dataset identity differ")
    retrieval_protocol_id = binding.get("retrieval_protocol_id") or candidate.get(
        "retrieval_protocol_id"
    )
    validate_dataset_method_compatibility(
        str(dataset_id), canonical, str(retrieval_protocol_id)
    )
    dependency = binding.get("generator_dependency") or candidate.get(
        "generator_dependency"
    )
    if dependency != spec["generator_dependency"]:
        raise ValueError("Result generator dependency conflicts with formal registry")
    conditioning_id = binding.get("conditioning_generator_id") or candidate.get(
        "conditioning_generator_id"
    )
    evaluation_id = binding.get("evaluation_generator_id") or candidate.get(
        "evaluation_generator_id"
    )
    if dependency == GENERATOR_DEPENDENCY_CONDITIONED:
        if not conditioning_id or conditioning_id != evaluation_id:
            raise ValueError("Conditioned matched-main generator IDs must match")
        seed = binding.get("seed")
        if seed not in spec["seed_policy"]["seeds"]:
            raise ValueError("Learned result seed is outside the formal seed policy")
    elif binding.get("seed") is not None:
        raise ValueError("Deterministic method must not be replicated by training seed")
    split = candidate.get("split") or contract.get("split")
    if not is_final_split(str(dataset_id), str(split)):
        raise ValueError("Result split is not the dataset's held-out final split")
    freeze = candidate.get("formal_freeze")
    freeze_status = held_out_freeze_status(
        freeze if isinstance(freeze, Mapping) else None,
        expected_dataset_id=str(dataset_id),
    )
    if not freeze_status["ready"]:
        raise ValueError(
            "Held-out freeze/sign-off is not ready: "
            + "; ".join(freeze_status["blockers"])
        )
    result_registry_version = candidate.get("formal_registry_version")
    result_registry_fingerprint = candidate.get("formal_registry_fingerprint")
    validate_formal_registry_identity(
        result_registry_version, result_registry_fingerprint
    )
    if result_registry_version != freeze.get("formal_registry_version") or (
        result_registry_fingerprint != freeze.get("method_registry_fingerprint")
    ):
        raise ValueError("Result and held-out freeze registry identities differ")
    if candidate.get("generator_registry_fingerprint") != freeze.get(
        "generator_registry_fingerprint"
    ):
        raise ValueError("Result generator registry fingerprint mismatch")
    if candidate.get("retrieval_protocol_fingerprint") != freeze.get(
        "retrieval_protocol_fingerprint"
    ):
        raise ValueError("Result retrieval protocol fingerprint mismatch")
    required_binding_fields = (
        "evaluation_generator_identity_fingerprint",
        "selection_sha256",
        "selection_manifest_sha256",
        "selection_manifest_fingerprint",
        "generator_manifest_sha256",
    )
    missing_binding = [field for field in required_binding_fields if not binding.get(field)]
    if missing_binding:
        raise ValueError(
            "Formal evaluation binding is incomplete: " + ", ".join(missing_binding)
        )
    required_result_fields = ("predictions_sha256", "metrics_sha256")
    missing_result = [field for field in required_result_fields if not candidate.get(field)]
    if missing_result:
        raise ValueError(
            "Formal evaluation result is incomplete: " + ", ".join(missing_result)
        )
    git = candidate.get("git")
    if not isinstance(git, Mapping) or git.get("commit") != freeze.get("git_commit"):
        raise ValueError("Result Git provenance differs from held-out freeze")
    for field, freeze_field in (
        ("config_sha256", "config_sha256"),
        ("prompt_hash", "prompt_hash"),
        ("verbalizer_hash", "verbalizer_hash"),
    ):
        if contract.get(field) != freeze.get(freeze_field):
            raise ValueError(f"Result {field} differs from held-out freeze")
    return {
        "eligible": True,
        "canonical_method_id": canonical,
        "dataset_id": dataset_id,
        "retrieval_protocol_id": retrieval_protocol_id,
        "experiment_type": MATCHED_MAIN,
        "registry_version": result_registry_version,
        "registry_fingerprint": result_registry_fingerprint,
    }

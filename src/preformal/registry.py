"""Frozen method registry for the signed-v1 preformal benchmark."""

from __future__ import annotations

from typing import Any, Mapping

from src.run_manifest import stable_hash

SIGNED_V1_METHOD = "rag_cbwdm_signed_v1"
KCBWDM_SIGNED_V1_METHOD = "kcbwdm_signed_v1"
KCBWDM_LINEAR_GATE_V2_METHOD = "kcbwdm_linear_gate_v2"
KCBWDM_NORMALIZED_RHO_V1_METHOD = "kcbwdm_normalized_rho_v1"
PREFORMAL_SPLIT_ROLE = "preformal_eval"
LEARNED_SEEDS = (13, 21, 42)
SIGNED_V1_CONTRACT_VERSION = "rag_cbwdm_signed_v1.v1"

SIGNED_V1_CONTRACT: dict[str, Any] = {
    "method": SIGNED_V1_METHOD,
    "status": "experimental_formal_method",
    "production_method": False,
    "teacher": {
        "alignment": "x_ij^T d_i",
        "admissible": "alignment_ij > 0",
        "action": "maximize_current_production_theta_marginal_within_admissible",
        "top_m": 4,
        "stop_threshold": 0.001,
        "alignment_eps": 0.0,
        "b_plus": 0.01,
        "b_minus": 0.001,
        "neutral_sample_policy": "negative",
        "nonpositive_supervision": "explicit_harmful_negative",
    },
    "selector": {
        "model_name": "/root/models/ms-marco-MiniLM-L-6-v2",
        "epochs": 3,
        "lr": 2e-5,
        "batch_size": 8,
        "beta": 0.25,
        "gamma": 1.0,
        "loss_type": "cbwdm_multitask",
        "top_m": 4,
        "min_docs": 0,
        "score_threshold": 0.0,
    },
    "seeds": list(LEARNED_SEEDS),
    "calibration_eligible": False,
    "preformal_can_update_parameters": False,
}

KCBWDM_SIGNED_V1_CONTRACT: dict[str, Any] = {
    "contract_version": "kcbwdm_signed_v1.v1",
    "method": KCBWDM_SIGNED_V1_METHOD,
    "status": "development_only",
    "production_method": False,
    "teacher": {
        "top_m": 4,
        "stop_threshold": 0.001,
        "alignment_eps": 0.0,
        "b_plus": 0.01,
        "b_minus": 0.001,
        "neutral_sample_policy": "negative",
        "gain_tolerance": 1e-10,
    },
    "kernel": {
        "base_kernel": "rbf",
        "anchor": "zero_effect",
        "bandwidth_policy": "train_core_within_query_positive_distance_median",
        "ridge_lambda": 0.01,
        "lambda_policy": "absolute",
        "target_normalization": False,
        "set_dependent_centering": False,
    },
    "sign_policy": "static_anchored_target_alignment_gt_0",
    "selector": {
        "model_name": "/root/models/ms-marco-MiniLM-L-6-v2",
        "epochs": 3,
        "lr": 2e-5,
        "batch_size": 8,
        "beta": 0.25,
        "gamma": 1.0,
        "loss_type": "cbwdm_multitask",
        "top_m": 4,
        "min_docs": 0,
        "score_threshold": 0.0,
        "runtime_implementation": "block_v1",
        "forward_batch_size": 32,
    },
    "seed": 13,
    "calibration_eligible": False,
    "held_out_eligible": False,
}

KCBWDM_LINEAR_GATE_V2_CONTRACT: dict[str, Any] = {
    "contract_version": "kcbwdm_linear_gate_v2.v1",
    "method": KCBWDM_LINEAR_GATE_V2_METHOD,
    "status": "development_only",
    "production_method": False,
    "teacher": {
        "top_m": 4,
        "stop_threshold": 0.001,
        "alignment_eps": 0.0,
        "b_plus": 0.01,
        "b_minus": 0.001,
        "neutral_sample_policy": "negative",
        "gain_tolerance": 1e-10,
    },
    "kernel": {
        "base_kernel": "rbf",
        "anchor": "zero_effect",
        "bandwidth_policy": "train_core_within_query_positive_distance_median",
        "ridge_lambda": 0.01,
        "lambda_policy": "absolute",
        "target_normalization": False,
        "set_dependent_centering": False,
    },
    "sign_policy": "static_linear_target_alignment_gt_0",
    "selector": {
        "model_name": "/root/models/ms-marco-MiniLM-L-6-v2",
        "epochs": 3,
        "lr": 2e-5,
        "batch_size": 8,
        "beta": 0.25,
        "gamma": 1.0,
        "loss_type": "cbwdm_multitask",
        "top_m": 4,
        "min_docs": 0,
        "score_threshold": 0.0,
        "runtime_implementation": "block_v1",
        "forward_batch_size": 32,
    },
    "seed": 13,
    "calibration_eligible": False,
    "held_out_eligible": False,
}

KCBWDM_NORMALIZED_RHO_V1_CONTRACT: dict[str, Any] = {
    "contract_version": "kcbwdm_normalized_rho_v1.v1",
    "method": KCBWDM_NORMALIZED_RHO_V1_METHOD,
    "status": "development_only",
    "production_method": False,
    "teacher": {
        "top_m": 4,
        "stop_threshold": 0.001,
        "alignment_eps": 0.0,
        "b_plus": 0.01,
        "b_minus": 0.001,
        "neutral_sample_policy": "negative",
        "gain_tolerance": 1e-10,
    },
    "kernel": {
        "family": "median_diagonal_normalized_rho",
        "formula_version": "median_diagonal_normalized_rho_v1",
        "linear_component": "linear",
        "nonlinear_component": "zero_anchored_rbf",
        "bandwidth_policy": "train_core_within_query_positive_distance_median",
        "normalization_policy": "median_positive_diag_ratio_v1",
        "fit_scope": "full_train_core",
        "rho_min": 0.0,
        "rho_max": 1.0,
        "ridge_lambda": 0.01,
        "lambda_policy": "absolute_in_normalized_geometry",
        "target_normalization": False,
        "set_dependent_centering": False,
    },
    "sign_policy": "static_linear_target_alignment_gt_0",
    "selector": {
        "model_name": "/root/models/ms-marco-MiniLM-L-6-v2",
        "epochs": 3,
        "lr": 2e-5,
        "batch_size": 8,
        "beta": 0.25,
        "gamma": 1.0,
        "loss_type": "cbwdm_multitask",
        "top_m": 4,
        "min_docs": 0,
        "score_threshold": 0.0,
        "runtime_implementation": "block_v1",
        "forward_batch_size": 32,
    },
    "seed": 13,
    "calibration_eligible": False,
    "held_out_eligible": False,
}

METHOD_CONTRACT_VERSIONS = {
    SIGNED_V1_METHOD: SIGNED_V1_CONTRACT_VERSION,
    KCBWDM_SIGNED_V1_METHOD: KCBWDM_SIGNED_V1_CONTRACT["contract_version"],
    KCBWDM_LINEAR_GATE_V2_METHOD: KCBWDM_LINEAR_GATE_V2_CONTRACT[
        "contract_version"
    ],
    KCBWDM_NORMALIZED_RHO_V1_METHOD: KCBWDM_NORMALIZED_RHO_V1_CONTRACT[
        "contract_version"
    ],
}

# Historical schemas are deliberately enumerated. Missing contract-version
# fields are accepted only for these frozen pre-K5.2 artifact identities after
# their method and self-fingerprint have also been validated.
HISTORICAL_TEACHER_SCHEMAS_WITHOUT_CONTRACT_VERSION = {
    SIGNED_V1_METHOD: {
        "rag_cbwdm_preformal_signed_teacher.v1",
        "rag_cbwdm_preformal_signed_teacher.v2",
    },
}
HISTORICAL_TRAINING_SCHEMAS_WITHOUT_CONTRACT_VERSION = {
    SIGNED_V1_METHOD: {
        "rag_cbwdm_preformal_signed_training.v1",
        "rag_cbwdm_preformal_signed_training.v2",
    },
    KCBWDM_SIGNED_V1_METHOD: {
        "rag_kcbwdm_preformal_signed_training.v1",
        "rag_kcbwdm_preformal_signed_training.v2",
    },
}
HISTORICAL_SELECTION_SCHEMAS_WITHOUT_CONTRACT_VERSION = {
    SIGNED_V1_METHOD: {
        "rag_cbwdm_selection_manifest.v1",
        "rag_cbwdm_selection_manifest.v2",
    },
    KCBWDM_SIGNED_V1_METHOD: {
        "rag_cbwdm_selection_manifest.v1",
        "rag_cbwdm_selection_manifest.v2",
    },
}


def method_contract_version(method_name: str) -> str:
    try:
        return str(METHOD_CONTRACT_VERSIONS[method_name])
    except KeyError as exc:
        raise ValueError(
            f"Method has no registered contract version: {method_name!r}"
        ) from exc


def _frozen_contract(method_name: str) -> Mapping[str, Any]:
    if method_name == SIGNED_V1_METHOD:
        return SIGNED_V1_CONTRACT
    if method_name == KCBWDM_SIGNED_V1_METHOD:
        return KCBWDM_SIGNED_V1_CONTRACT
    if method_name == KCBWDM_LINEAR_GATE_V2_METHOD:
        return KCBWDM_LINEAR_GATE_V2_CONTRACT
    if method_name == KCBWDM_NORMALIZED_RHO_V1_METHOD:
        return KCBWDM_NORMALIZED_RHO_V1_CONTRACT
    raise ValueError(f"Method has no frozen contract: {method_name!r}")


def _validate_historical_teacher_semantics(
    contract: Mapping[str, Any], method_name: str
) -> None:
    frozen = _frozen_contract(method_name)
    parameters = contract.get("parameters")
    if contract.get("stage") != "teacher_training_only" or not isinstance(
        parameters, Mapping
    ):
        raise ValueError("Historical teacher contract lacks frozen semantics")
    expected = frozen["teacher"]
    for key in (
        "top_m",
        "stop_threshold",
        "alignment_eps",
        "b_plus",
        "b_minus",
        "neutral_sample_policy",
    ):
        if parameters.get(key) != expected[key]:
            raise ValueError(f"Historical teacher contract changed frozen {key}")


def _validate_historical_training_semantics(
    contract: Mapping[str, Any], method_name: str
) -> None:
    frozen = _frozen_contract(method_name)
    selector = frozen["selector"]
    teacher = frozen["teacher"]
    if contract.get("stage") != "training":
        raise ValueError("Historical training contract stage mismatch")
    expected = {
        "epochs": selector["epochs"],
        "lr": selector["lr"],
        "batch_size": selector["batch_size"],
        "beta": selector["beta"],
        "gamma": selector["gamma"],
        "loss_type": selector["loss_type"],
        "b_plus": teacher["b_plus"],
        "b_minus": teacher["b_minus"],
        "neutral_sample_policy": teacher["neutral_sample_policy"],
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"Historical training contract changed frozen {key}")
    allowed_seeds = (
        set(frozen.get("seeds", []))
        if method_name == SIGNED_V1_METHOD
        else {int(frozen["seed"])}
    )
    if contract.get("seed") not in allowed_seeds:
        raise ValueError("Historical training contract seed mismatch")


def _validate_historical_selection_semantics(
    contract: Mapping[str, Any], method_name: str
) -> None:
    frozen = _frozen_contract(method_name)
    selector = frozen["selector"]
    parameters = contract.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("Historical selection contract lacks parameters")
    expected = {
        "top_m": selector["top_m"],
        "min_docs": selector["min_docs"],
        "score_threshold": selector["score_threshold"],
        "uses_gold_at_inference": False,
    }
    for key, value in expected.items():
        if parameters.get(key) != value:
            raise ValueError(f"Historical selection contract changed frozen {key}")
    allowed_seeds = (
        set(frozen.get("seeds", []))
        if method_name == SIGNED_V1_METHOD
        else {int(frozen["seed"])}
    )
    if parameters.get("seed") not in allowed_seeds:
        raise ValueError("Historical selection contract seed mismatch")


def validate_teacher_method_contract(
    payload: Mapping[str, Any], *, method_name: str
) -> str:
    """Validate explicit teacher identity or a frozen historical equivalent."""
    expected = method_contract_version(method_name)
    if payload.get("method") != method_name:
        raise ValueError("Teacher method mismatch")
    contract = payload.get("contract")
    if not isinstance(contract, Mapping) or contract.get("method") != method_name:
        raise ValueError("Teacher contract method mismatch")
    actual = contract.get("method_contract_version")
    if actual == expected:
        return expected
    if actual is not None:
        raise ValueError(
            "Teacher method contract version mismatch: "
            f"expected={expected!r} actual={actual!r}"
        )
    allowed_schemas = HISTORICAL_TEACHER_SCHEMAS_WITHOUT_CONTRACT_VERSION.get(
        method_name, set()
    )
    if payload.get("schema_version") not in allowed_schemas:
        raise ValueError("Teacher manifest lacks a recognized contract version")
    if payload.get("fingerprint") != stable_hash(dict(contract)):
        raise ValueError("Historical teacher manifest fingerprint mismatch")
    _validate_historical_teacher_semantics(contract, method_name)
    return expected


def validate_training_method_contract(
    payload: Mapping[str, Any], *, method_name: str
) -> str:
    """Validate training identity with an explicit, fail-closed legacy path."""
    expected = method_contract_version(method_name)
    if payload.get("method") != method_name:
        raise ValueError("Checkpoint method mismatch")
    contract = payload.get("contract")
    if not isinstance(contract, Mapping) or contract.get("method") != method_name:
        raise ValueError("Training contract method mismatch")
    top_level = payload.get("method_contract_version")
    nested = contract.get("method_contract_version")
    if top_level == expected and nested == expected:
        return expected
    if top_level is not None or nested is not None:
        raise ValueError(
            "Training method contract version mismatch: "
            f"expected={expected!r} top_level={top_level!r} contract={nested!r}"
        )
    allowed_schemas = HISTORICAL_TRAINING_SCHEMAS_WITHOUT_CONTRACT_VERSION.get(
        method_name, set()
    )
    if payload.get("schema_version") not in allowed_schemas:
        raise ValueError("Training manifest lacks a recognized contract version")
    if payload.get("fingerprint") != stable_hash(dict(contract)):
        raise ValueError("Historical training manifest fingerprint mismatch")
    _validate_historical_training_semantics(contract, method_name)
    return expected


def validate_selection_method_contract(
    payload: Mapping[str, Any], *, method_name: str
) -> str:
    """Validate selection contract version, including enumerated old schemas."""
    expected = method_contract_version(method_name)
    if payload.get("method") != method_name:
        raise ValueError("Selection method mismatch")
    contract = payload.get("contract")
    if not isinstance(contract, Mapping) or contract.get("method") != method_name:
        raise ValueError("Selection contract method mismatch")
    top_level = payload.get("method_contract_version")
    nested = contract.get("method_contract_version")
    if top_level == expected and nested == expected:
        return expected
    if top_level is not None or nested is not None:
        raise ValueError(
            "Selection method contract version mismatch: "
            f"expected={expected!r} top_level={top_level!r} contract={nested!r}"
        )
    allowed_schemas = HISTORICAL_SELECTION_SCHEMAS_WITHOUT_CONTRACT_VERSION.get(
        method_name, set()
    )
    if payload.get("schema_version") not in allowed_schemas:
        raise ValueError("Selection manifest lacks a recognized contract version")
    if payload.get("fingerprint") != stable_hash(dict(contract)):
        raise ValueError("Historical selection manifest fingerprint mismatch")
    _validate_historical_selection_semantics(contract, method_name)
    return expected

PREFORMAL_METHODS: dict[str, dict[str, Any]] = {
    "no_evidence": {"deployable": True, "learned": False, "seeds": [13]},
    "naive_topm": {"deployable": True, "learned": False, "seeds": [13]},
    "bge": {"deployable": True, "learned": False, "seeds": [13]},
    "infogain_fever": {"deployable": True, "learned": True, "seeds": list(LEARNED_SEEDS)},
    "rag_cbwdm": {"deployable": True, "learned": True, "seeds": list(LEARNED_SEEDS)},
    SIGNED_V1_METHOD: {"deployable": True, "learned": True, "seeds": list(LEARNED_SEEDS)},
    KCBWDM_SIGNED_V1_METHOD: {"deployable": True, "learned": True, "seeds": [13]},
    KCBWDM_LINEAR_GATE_V2_METHOD: {"deployable": True, "learned": True, "seeds": [13]},
    KCBWDM_NORMALIZED_RHO_V1_METHOD: {"deployable": True, "learned": True, "seeds": [13]},
    "signed_gate_oracle": {"deployable": False, "learned": False, "seeds": [13], "diagnostic_only": True},
}


def assert_frozen_signed_contract(parameters: dict[str, Any]) -> None:
    """Reject any scientific override of the signed-v1 first-round contract."""
    expected = {**SIGNED_V1_CONTRACT["teacher"], **SIGNED_V1_CONTRACT["selector"]}
    comparable = {
        "top_m": expected["top_m"],
        "teacher_stop_threshold": expected["stop_threshold"],
        "alignment_eps": expected["alignment_eps"],
        "b_plus": expected["b_plus"],
        "b_minus": expected["b_minus"],
        "neutral_sample_policy": expected["neutral_sample_policy"],
        "model_name": expected["model_name"],
        "epochs": expected["epochs"],
        "lr": expected["lr"],
        "batch_size": expected["batch_size"],
        "beta": expected["beta"],
        "gamma": expected["gamma"],
        "loss_type": expected["loss_type"],
        "min_docs": expected["min_docs"],
        "score_threshold": expected["score_threshold"],
    }
    changed = {
        key: {"actual": parameters.get(key), "expected": value}
        for key, value in comparable.items()
        if key in parameters and parameters.get(key) != value
    }
    if changed:
        raise ValueError(f"rag_cbwdm_signed_v1 parameters are frozen: {changed}")


def assert_frozen_kcbwdm_contract(parameters: dict[str, Any]) -> None:
    """Reject scientific overrides of the independent KCBWDM-v1 contract."""
    teacher = KCBWDM_SIGNED_V1_CONTRACT["teacher"]
    kernel = KCBWDM_SIGNED_V1_CONTRACT["kernel"]
    selector = KCBWDM_SIGNED_V1_CONTRACT["selector"]
    comparable = {
        "method": KCBWDM_SIGNED_V1_CONTRACT["method"],
        "top_m": teacher["top_m"],
        "teacher_stop_threshold": teacher["stop_threshold"],
        "alignment_eps": teacher["alignment_eps"],
        "b_plus": teacher["b_plus"],
        "b_minus": teacher["b_minus"],
        "neutral_sample_policy": teacher["neutral_sample_policy"],
        "gain_tolerance": teacher["gain_tolerance"],
        "base_kernel": kernel["base_kernel"],
        "anchor": kernel["anchor"],
        "bandwidth_policy": kernel["bandwidth_policy"],
        "ridge_lambda": kernel["ridge_lambda"],
        "lambda_policy": kernel["lambda_policy"],
        "target_normalization": kernel["target_normalization"],
        "set_dependent_centering": kernel["set_dependent_centering"],
        "sign_policy": KCBWDM_SIGNED_V1_CONTRACT["sign_policy"],
        "model_name": selector["model_name"],
        "epochs": selector["epochs"],
        "lr": selector["lr"],
        "batch_size": selector["batch_size"],
        "beta": selector["beta"],
        "gamma": selector["gamma"],
        "loss_type": selector["loss_type"],
        "min_docs": selector["min_docs"],
        "score_threshold": selector["score_threshold"],
        "runtime_implementation": selector["runtime_implementation"],
        "forward_batch_size": selector["forward_batch_size"],
        "seed": KCBWDM_SIGNED_V1_CONTRACT["seed"],
    }
    changed = {
        key: {"actual": parameters.get(key), "expected": value}
        for key, value in comparable.items()
        if key in parameters and parameters.get(key) != value
    }
    if changed:
        raise ValueError(f"kcbwdm_signed_v1 parameters are frozen: {changed}")


def assert_frozen_kcbwdm_linear_gate_v2_contract(
    parameters: dict[str, Any],
) -> None:
    """Reject scientific overrides of the controlled KCBWDM-v2A contract."""
    teacher = KCBWDM_LINEAR_GATE_V2_CONTRACT["teacher"]
    kernel = KCBWDM_LINEAR_GATE_V2_CONTRACT["kernel"]
    selector = KCBWDM_LINEAR_GATE_V2_CONTRACT["selector"]
    comparable = {
        "method": KCBWDM_LINEAR_GATE_V2_CONTRACT["method"],
        "top_m": teacher["top_m"],
        "teacher_stop_threshold": teacher["stop_threshold"],
        "alignment_eps": teacher["alignment_eps"],
        "b_plus": teacher["b_plus"],
        "b_minus": teacher["b_minus"],
        "neutral_sample_policy": teacher["neutral_sample_policy"],
        "gain_tolerance": teacher["gain_tolerance"],
        "base_kernel": kernel["base_kernel"],
        "anchor": kernel["anchor"],
        "bandwidth_policy": kernel["bandwidth_policy"],
        "ridge_lambda": kernel["ridge_lambda"],
        "lambda_policy": kernel["lambda_policy"],
        "target_normalization": kernel["target_normalization"],
        "set_dependent_centering": kernel["set_dependent_centering"],
        "sign_policy": KCBWDM_LINEAR_GATE_V2_CONTRACT["sign_policy"],
        "model_name": selector["model_name"],
        "epochs": selector["epochs"],
        "lr": selector["lr"],
        "batch_size": selector["batch_size"],
        "beta": selector["beta"],
        "gamma": selector["gamma"],
        "loss_type": selector["loss_type"],
        "min_docs": selector["min_docs"],
        "score_threshold": selector["score_threshold"],
        "runtime_implementation": selector["runtime_implementation"],
        "forward_batch_size": selector["forward_batch_size"],
        "seed": KCBWDM_LINEAR_GATE_V2_CONTRACT["seed"],
    }
    changed = {
        key: {"actual": parameters.get(key), "expected": value}
        for key, value in comparable.items()
        if key in parameters and parameters.get(key) != value
    }
    if changed:
        raise ValueError(
            f"kcbwdm_linear_gate_v2 parameters are frozen: {changed}"
        )


def assert_frozen_kcbwdm_normalized_rho_v1_contract(
    parameters: dict[str, Any],
) -> None:
    """Reject changes outside the explicit normalized-rho family parameter."""
    teacher = KCBWDM_NORMALIZED_RHO_V1_CONTRACT["teacher"]
    kernel = KCBWDM_NORMALIZED_RHO_V1_CONTRACT["kernel"]
    selector = KCBWDM_NORMALIZED_RHO_V1_CONTRACT["selector"]
    comparable = {
        "method": KCBWDM_NORMALIZED_RHO_V1_CONTRACT["method"],
        "top_m": teacher["top_m"],
        "teacher_stop_threshold": teacher["stop_threshold"],
        "alignment_eps": teacher["alignment_eps"],
        "b_plus": teacher["b_plus"],
        "b_minus": teacher["b_minus"],
        "neutral_sample_policy": teacher["neutral_sample_policy"],
        "gain_tolerance": teacher["gain_tolerance"],
        "kernel_family": kernel["family"],
        "kernel_formula_version": kernel["formula_version"],
        "bandwidth_policy": kernel["bandwidth_policy"],
        "normalization_policy": kernel["normalization_policy"],
        "fit_scope": kernel["fit_scope"],
        "ridge_lambda": kernel["ridge_lambda"],
        "lambda_policy": kernel["lambda_policy"],
        "target_normalization": kernel["target_normalization"],
        "set_dependent_centering": kernel["set_dependent_centering"],
        "sign_policy": KCBWDM_NORMALIZED_RHO_V1_CONTRACT["sign_policy"],
        "model_name": selector["model_name"],
        "epochs": selector["epochs"],
        "lr": selector["lr"],
        "batch_size": selector["batch_size"],
        "beta": selector["beta"],
        "gamma": selector["gamma"],
        "loss_type": selector["loss_type"],
        "min_docs": selector["min_docs"],
        "score_threshold": selector["score_threshold"],
        "runtime_implementation": selector["runtime_implementation"],
        "forward_batch_size": selector["forward_batch_size"],
        "seed": KCBWDM_NORMALIZED_RHO_V1_CONTRACT["seed"],
    }
    changed = {
        key: {"actual": parameters.get(key), "expected": value}
        for key, value in comparable.items()
        if key in parameters and parameters.get(key) != value
    }
    if changed:
        raise ValueError(
            f"kcbwdm_normalized_rho_v1 parameters are frozen: {changed}"
        )
    rho = parameters.get("rho")
    if rho is None:
        raise ValueError("kcbwdm_normalized_rho_v1 requires explicit rho")
    rho_value = float(rho)
    if not kernel["rho_min"] <= rho_value <= kernel["rho_max"]:
        raise ValueError("kcbwdm_normalized_rho_v1 rho must be in [0, 1]")


def assert_not_preformal_parameter_source(payload: Any) -> None:
    """Fail closed if preformal artifacts are offered to a parameter selector."""
    text = str(payload).casefold()
    if "preformal_eval" in text or "preformal_signed_v1" in text:
        raise ValueError("preformal_eval cannot be used for calibration or frozen parameter selection")


def assert_no_held_out_reference(payload: Any) -> None:
    """Fail closed on held-out references in preformal training/calibration inputs."""
    text = str(payload).casefold()
    forbidden = ("held_out_test", "retrieve_test", "posterior_test")
    matches = [value for value in forbidden if value in text]
    if matches:
        raise ValueError(f"held_out_test reference is forbidden in preformal training/calibration: {matches}")

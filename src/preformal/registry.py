"""Frozen method registry for the signed-v1 preformal benchmark."""

from __future__ import annotations

from typing import Any

SIGNED_V1_METHOD = "rag_cbwdm_signed_v1"
KCBWDM_SIGNED_V1_METHOD = "kcbwdm_signed_v1"
PREFORMAL_SPLIT_ROLE = "preformal_eval"
LEARNED_SEEDS = (13, 21, 42)

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

PREFORMAL_METHODS: dict[str, dict[str, Any]] = {
    "no_evidence": {"deployable": True, "learned": False, "seeds": [13]},
    "naive_topm": {"deployable": True, "learned": False, "seeds": [13]},
    "bge": {"deployable": True, "learned": False, "seeds": [13]},
    "infogain_fever": {"deployable": True, "learned": True, "seeds": list(LEARNED_SEEDS)},
    "rag_cbwdm": {"deployable": True, "learned": True, "seeds": list(LEARNED_SEEDS)},
    SIGNED_V1_METHOD: {"deployable": True, "learned": True, "seeds": list(LEARNED_SEEDS)},
    KCBWDM_SIGNED_V1_METHOD: {"deployable": True, "learned": True, "seeds": [13]},
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

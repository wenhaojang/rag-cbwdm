"""Frozen method registry for the signed-v1 preformal benchmark."""

from __future__ import annotations

from typing import Any

SIGNED_V1_METHOD = "rag_cbwdm_signed_v1"
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

PREFORMAL_METHODS: dict[str, dict[str, Any]] = {
    "no_evidence": {"deployable": True, "learned": False, "seeds": [13]},
    "naive_topm": {"deployable": True, "learned": False, "seeds": [13]},
    "bge": {"deployable": True, "learned": False, "seeds": [13]},
    "infogain_fever": {"deployable": True, "learned": True, "seeds": list(LEARNED_SEEDS)},
    "rag_cbwdm": {"deployable": True, "learned": True, "seeds": list(LEARNED_SEEDS)},
    SIGNED_V1_METHOD: {"deployable": True, "learned": True, "seeds": list(LEARNED_SEEDS)},
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

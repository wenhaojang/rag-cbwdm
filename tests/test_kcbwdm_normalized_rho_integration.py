from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest

from src.artifact_binding import (
    BINDING_SCHEMA_VERSION,
    GENERATOR_DEPENDENCY_CONDITIONED,
    build_evaluation_binding,
    conditioned_selection_binding,
)
from src.cbwdm_score import build_local_effects
from src.diagnostics.kcbwdm_normalized_rho_v1 import (
    KCBWDM_NORMALIZED_RHO_V1_METHOD,
    build_kcbwdm_normalized_rho_v1_teacher_row,
)
from src.diagnostics.signed_selector_v1 import select_row_without_gold
from src.kcbwdm_score import fit_kernel_scale_c, fit_train_core_bandwidth
from src.preformal.registry import (
    HISTORICAL_SELECTION_SCHEMAS_WITHOUT_CONTRACT_VERSION,
    HISTORICAL_TEACHER_SCHEMAS_WITHOUT_CONTRACT_VERSION,
    HISTORICAL_TRAINING_SCHEMAS_WITHOUT_CONTRACT_VERSION,
    KCBWDM_LINEAR_GATE_V2_CONTRACT,
    KCBWDM_NORMALIZED_RHO_V1_CONTRACT,
    KCBWDM_SIGNED_V1_CONTRACT,
    METHOD_CONTRACT_VERSIONS,
    SIGNED_V1_CONTRACT_VERSION,
    assert_frozen_kcbwdm_normalized_rho_v1_contract,
    method_contract_version,
    validate_teacher_method_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = PROJECT_ROOT / "scripts" / "preformal" / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_parameters(rho: float) -> dict:
    contract = KCBWDM_NORMALIZED_RHO_V1_CONTRACT
    return {
        "method": KCBWDM_NORMALIZED_RHO_V1_METHOD,
        "top_m": contract["teacher"]["top_m"],
        "teacher_stop_threshold": contract["teacher"]["stop_threshold"],
        "alignment_eps": contract["teacher"]["alignment_eps"],
        "b_plus": contract["teacher"]["b_plus"],
        "b_minus": contract["teacher"]["b_minus"],
        "neutral_sample_policy": contract["teacher"]["neutral_sample_policy"],
        "gain_tolerance": contract["teacher"]["gain_tolerance"],
        "kernel_family": contract["kernel"]["family"],
        "kernel_formula_version": contract["kernel"]["formula_version"],
        "bandwidth_policy": contract["kernel"]["bandwidth_policy"],
        "normalization_policy": contract["kernel"]["normalization_policy"],
        "fit_scope": contract["kernel"]["fit_scope"],
        "ridge_lambda": contract["kernel"]["ridge_lambda"],
        "lambda_policy": contract["kernel"]["lambda_policy"],
        "target_normalization": contract["kernel"]["target_normalization"],
        "set_dependent_centering": contract["kernel"]["set_dependent_centering"],
        "sign_policy": contract["sign_policy"],
        "model_name": contract["selector"]["model_name"],
        "epochs": contract["selector"]["epochs"],
        "lr": contract["selector"]["lr"],
        "batch_size": contract["selector"]["batch_size"],
        "beta": contract["selector"]["beta"],
        "gamma": contract["selector"]["gamma"],
        "loss_type": contract["selector"]["loss_type"],
        "min_docs": contract["selector"]["min_docs"],
        "score_threshold": contract["selector"]["score_threshold"],
        "runtime_implementation": contract["selector"]["runtime_implementation"],
        "forward_batch_size": contract["selector"]["forward_batch_size"],
        "seed": contract["seed"],
        "rho": rho,
    }


def teacher_fixture(rho: float) -> tuple[dict, dict]:
    row = {
        "id": "q1",
        "query": "claim",
        "label": "SUPPORTS",
        "split": "train_core",
        "labels": ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"],
        "eta0": [0.25, 0.45, 0.30],
        "candidates": [
            {"doc_id": "help-a", "rank": 1, "eta": [0.65, 0.20, 0.15]},
            {"doc_id": "help-b", "rank": 2, "eta": [0.48, 0.31, 0.21]},
            {"doc_id": "harm", "rank": 3, "eta": [0.12, 0.55, 0.33]},
        ],
    }
    X, _ = build_local_effects(
        np.asarray(row["eta0"]),
        np.asarray([candidate["eta"] for candidate in row["candidates"]]),
        row["label"],
        row["labels"],
        "euclidean_posterior_shift",
        0.001,
        "paper_mixture",
    )
    bandwidth = fit_train_core_bandwidth([X], split="train_core")
    scale = fit_kernel_scale_c(X, sigma=bandwidth.sigma, split="train_core")
    contract = KCBWDM_NORMALIZED_RHO_V1_CONTRACT
    params = {
        "top_m": 4,
        "stop_threshold": 0.001,
        "alignment_eps": 0.0,
        "b_plus": 0.01,
        "b_minus": 0.001,
        "neutral_sample_policy": "negative",
        "gain_tolerance": 1e-10,
        "ridge_lambda": 0.01,
        "eps_smooth": 0.001,
        "l_type": "euclidean_posterior_shift",
        "target_smoothing": "paper_mixture",
        "sigma": bandwidth.sigma,
        "rho": rho,
        "linear_diag_median_positive": scale.linear_diag_median_positive,
        "rbf_diag_median_positive": scale.rbf_diag_median_positive,
        "kernel_scale_c": scale.kernel_scale_c,
        "kernel_family": contract["kernel"]["family"],
        "kernel_formula_version": contract["kernel"]["formula_version"],
        "normalization_policy": contract["kernel"]["normalization_policy"],
        "sign_policy": contract["sign_policy"],
    }
    return row, params


def test_new_method_identity_is_explicit_and_old_contracts_are_unchanged() -> None:
    assert KCBWDM_NORMALIZED_RHO_V1_METHOD == "kcbwdm_normalized_rho_v1"
    assert method_contract_version(KCBWDM_NORMALIZED_RHO_V1_METHOD) == (
        "kcbwdm_normalized_rho_v1.v1"
    )
    assert METHOD_CONTRACT_VERSIONS["rag_cbwdm_signed_v1"] == (
        SIGNED_V1_CONTRACT_VERSION
    )
    assert KCBWDM_SIGNED_V1_CONTRACT["contract_version"] == "kcbwdm_signed_v1.v1"
    assert KCBWDM_LINEAR_GATE_V2_CONTRACT["contract_version"] == (
        "kcbwdm_linear_gate_v2.v1"
    )
    assert KCBWDM_SIGNED_V1_CONTRACT["sign_policy"] == (
        "static_anchored_target_alignment_gt_0"
    )
    assert KCBWDM_LINEAR_GATE_V2_CONTRACT["sign_policy"] == (
        "static_linear_target_alignment_gt_0"
    )
    for historical in (
        HISTORICAL_TEACHER_SCHEMAS_WITHOUT_CONTRACT_VERSION,
        HISTORICAL_TRAINING_SCHEMAS_WITHOUT_CONTRACT_VERSION,
        HISTORICAL_SELECTION_SCHEMAS_WITHOUT_CONTRACT_VERSION,
    ):
        assert KCBWDM_NORMALIZED_RHO_V1_METHOD not in historical


@pytest.mark.parametrize("rho", [0.0, 1.0])
def test_normalized_rho_contract_accepts_endpoints(rho: float) -> None:
    assert_frozen_kcbwdm_normalized_rho_v1_contract(frozen_parameters(rho))


@pytest.mark.parametrize("rho", [-1e-12, 1.000000000001])
def test_normalized_rho_contract_rejects_out_of_range(rho: float) -> None:
    with pytest.raises(ValueError, match=r"rho must be in \[0, 1\]"):
        assert_frozen_kcbwdm_normalized_rho_v1_contract(frozen_parameters(rho))


def test_new_method_has_no_versionless_teacher_compatibility() -> None:
    contract = {
        "method": KCBWDM_NORMALIZED_RHO_V1_METHOD,
        "stage": "teacher_training_only",
        "parameters": {},
    }
    with pytest.raises(ValueError, match="recognized contract version"):
        validate_teacher_method_contract(
            {
                "schema_version": "rag_kcbwdm_normalized_rho_v1_teacher_manifest.v1",
                "method": KCBWDM_NORMALIZED_RHO_V1_METHOD,
                "contract": contract,
            },
            method_name=KCBWDM_NORMALIZED_RHO_V1_METHOD,
        )


def test_teacher_uses_linear_gate_and_rho_specific_normalized_utility() -> None:
    row, params0 = teacher_fixture(0.0)
    _, params1 = teacher_fixture(1.0)
    teacher0 = build_kcbwdm_normalized_rho_v1_teacher_row(row, params0)
    teacher1 = build_kcbwdm_normalized_rho_v1_teacher_row(row, params1)
    assert teacher0["rho"] == 0.0
    assert teacher1["rho"] == 1.0
    for teacher in (teacher0, teacher1):
        candidates = teacher["states"][0]["remaining_candidates"]
        assert all(item["admissible"] == (item["linear_alignment"] > 0.0) for item in candidates)
        harmful = next(item for item in candidates if item["candidate_id"] == "harm")
        assert harmful["admissible"] is False
    gains0 = [
        item["theta_marginal_gain"]
        for item in teacher0["states"][0]["remaining_candidates"]
    ]
    gains1 = [
        item["theta_marginal_gain"]
        for item in teacher1["states"][0]["remaining_candidates"]
    ]
    assert not np.allclose(gains0, gains1, rtol=1e-12, atol=1e-12)


def test_scale_fit_is_full_train_core_deterministic_and_label_free() -> None:
    materializer = load_script("28_materialize_kcbwdm_signed_v1_teacher.py")
    groups = [
        np.asarray([[0.2, -0.1, -0.1], [0.4, -0.3, -0.1]], dtype=np.float64),
        np.asarray([[0.1, 0.1, -0.2]], dtype=np.float64),
    ]
    sigma = fit_train_core_bandwidth(groups, split="train_core").sigma
    first = materializer._fit_and_verify_kernel_scale(
        groups, sigma=sigma, split="train_core"
    )
    second = materializer._fit_and_verify_kernel_scale(
        groups, sigma=sigma, split="train_core"
    )
    assert first == second
    assert first.source_split == "train_core"
    assert first.subsampling == "none"
    assert first.linear_diag_median_positive > 0
    assert first.rbf_diag_median_positive > 0
    assert first.kernel_scale_c == pytest.approx(
        first.linear_diag_median_positive / first.rbf_diag_median_positive
    )
    with pytest.raises(ValueError, match="only from split='train_core'"):
        materializer._fit_and_verify_kernel_scale(
            groups, sigma=sigma, split="validation"
        )


def test_teacher_training_and_selection_rho_mismatches_fail_closed(
    tmp_path: Path,
) -> None:
    trainer = load_script("26_train_signed_v1.py")
    selector = load_script("27_select_signed_v1.py")
    teacher = tmp_path / "teacher.jsonl"
    teacher.write_text("{}\n", encoding="utf-8")
    version = method_contract_version(KCBWDM_NORMALIZED_RHO_V1_METHOD)
    teacher_contract = {
        "method": KCBWDM_NORMALIZED_RHO_V1_METHOD,
        "method_contract_version": version,
        "stage": "teacher_training_only",
        "split": "train_core",
        "rho": 0.0,
    }
    teacher_payload = {
        "method": KCBWDM_NORMALIZED_RHO_V1_METHOD,
        "method_contract_version": version,
        "contract": teacher_contract,
        "teacher_sha256": trainer.sha256_file(teacher),
        "rho": 0.0,
    }
    assert trainer.validate_teacher_method_identity(
        teacher_payload,
        method_name=KCBWDM_NORMALIZED_RHO_V1_METHOD,
        teacher_sha256=trainer.sha256_file(teacher),
        split="train_core",
        rho=0.0,
    ) == version
    with pytest.raises(ValueError, match="Teacher rho"):
        trainer.validate_teacher_method_identity(
            teacher_payload,
            method_name=KCBWDM_NORMALIZED_RHO_V1_METHOD,
            teacher_sha256=trainer.sha256_file(teacher),
            split="train_core",
            rho=1.0,
        )

    training_contract = {
        "method": KCBWDM_NORMALIZED_RHO_V1_METHOD,
        "method_contract_version": version,
        "rho": 0.0,
    }
    training_payload = {
        "method": KCBWDM_NORMALIZED_RHO_V1_METHOD,
        "method_contract_version": version,
        "contract": training_contract,
        "seed": 13,
        "status": "completed",
        "rho": 0.0,
    }
    assert selector.validate_checkpoint_method_identity(
        training_payload,
        method_name=KCBWDM_NORMALIZED_RHO_V1_METHOD,
        seed=13,
        rho=0.0,
    ) == version
    with pytest.raises(ValueError, match="Training rho"):
        selector.validate_checkpoint_method_identity(
            training_payload,
            method_name=KCBWDM_NORMALIZED_RHO_V1_METHOD,
            seed=13,
            rho=1.0,
        )


def test_deployable_selector_signature_has_no_gold_or_kernel_inputs() -> None:
    parameters = inspect.signature(select_row_without_gold).parameters
    for forbidden in ("gold", "d", "posterior_effects", "kernel", "rho", "labels"):
        assert forbidden not in parameters


def test_rho_is_explicit_in_selection_and_evaluation_bindings() -> None:
    training_binding = {
        "schema_version": BINDING_SCHEMA_VERSION,
        "method": KCBWDM_NORMALIZED_RHO_V1_METHOD,
        "generator_dependency": GENERATOR_DEPENDENCY_CONDITIONED,
        "dataset_id": "fm2_official_closed_page_v1",
        "conditioning_generator_id": "qwen2.5-1.5b-instruct",
        "generator_identity_fingerprint": "generator-fingerprint",
        "retrieval_protocol_id": "fm2_official_closed_page_v1",
        "seed": 13,
        "method_parameters": {"rho": 1.0},
    }
    selection_artifact_binding = conditioned_selection_binding(training_binding)
    assert selection_artifact_binding["method_parameters"] == {"rho": 1.0}
    selection = {
        "artifact_binding": selection_artifact_binding,
        "selection_path": "/tmp/selection.jsonl",
        "selection_sha256": "selection-sha",
        "manifest_path": "/tmp/selection.manifest.json",
        "manifest_sha256": "selection-manifest-sha",
        "manifest_fingerprint": "selection-manifest-fingerprint",
    }
    generator = {
        "dataset_identity": {"dataset_id": "fm2_official_closed_page_v1"},
        "generator_identity": {"generator_id": "qwen2.5-1.5b-instruct"},
        "generator_identity_fingerprint": "generator-fingerprint",
        "manifest_path": "/tmp/generator.json",
        "manifest_sha256": "generator-sha",
    }
    evaluation_binding = build_evaluation_binding(selection, generator)
    assert evaluation_binding["method_parameters"] == {"rho": 1.0}

from __future__ import annotations

import copy

import pytest

from src.artifact_binding import (
    EVALUATION_BINDING_SCHEMA_VERSION,
    EVALUATION_MANIFEST_SCHEMA_VERSION,
    GENERATOR_DEPENDENCY_CONDITIONED,
    MATCHED_MAIN,
)
from src.formal_config import formal_v2_held_out_freeze_requirements
from src.formal_readiness import (
    CANONICAL_METHODS as LEGACY_CANONICAL_METHODS,
    check_formal_v2_registry_readiness,
)
from src.formal_registry import (
    CANONICAL_OURS,
    FORMAL_REGISTRY_FINGERPRINT,
    FORMAL_REGISTRY_SCHEMA_VERSION,
    FORMAL_REGISTRY_VERSION,
    HISTORICAL_FORMAL_REGISTRY_IDENTITIES,
    LEARNED_METHOD_SEEDS,
    KCBWDM_LINEAR_GATE_V2,
    KCBWDM_SIGNED_V1,
    MAIN_TABLE_METHODS,
    build_formal_registry,
    dataset_protocol,
    formal_registry_fingerprint,
    held_out_freeze_status,
    held_out_freeze_fingerprint,
    held_out_freeze_template,
    is_final_split,
    method_spec,
    split_role,
    validate_dataset_method_compatibility,
    validate_main_table_result,
    validate_formal_registry_identity,
)
from src.preformal.registry import PREFORMAL_METHODS


def ready_freeze(dataset_id: str) -> dict:
    payload = held_out_freeze_template(dataset_id)
    payload.update(
        {
            "status": "frozen",
            "generator_registry_fingerprint": "generator-registry-sha",
            "prompt_hash": "prompt-sha",
            "verbalizer_hash": "verbalizer-sha",
            "top_k": 4,
            "evidence_budget": {"max_docs": 4},
            "bge_model_freeze": {
                "model_id": "human-choice/bge",
                "revision": "immutable-revision",
                "sha256": "bge-sha",
            },
            "git_commit": "git-sha",
            "config_sha256": "config-sha",
            "runtime_sign_off": {
                "approved": True,
                "approved_by": "future-operator",
                "approved_at": "future-time",
                "evidence_reference": "future-runtime-audit",
                "freeze_fingerprint": None,
            },
        }
    )
    payload["freeze_fingerprint"] = held_out_freeze_fingerprint(payload)
    payload["runtime_sign_off"]["freeze_fingerprint"] = payload[
        "freeze_fingerprint"
    ]
    return payload


def signed_result(*, experiment_type: str = MATCHED_MAIN) -> dict:
    freeze = ready_freeze("fever_binary_v2")
    return {
        "schema_version": EVALUATION_MANIFEST_SCHEMA_VERSION,
        "status": "completed",
        "completed": True,
        "identity_mode": "formal_v2",
        "method": "rag_cbwdm_signed_v1",
        "dataset_id": "fever_binary_v2",
        "experiment_type": experiment_type,
        "split": "held_out_test",
        "diagnostic_only": False,
        "deployable": True,
        "artifact_binding": {
            "schema_version": EVALUATION_BINDING_SCHEMA_VERSION,
            "method": "rag_cbwdm_signed_v1",
            "dataset_id": "fever_binary_v2",
            "retrieval_protocol_id": "fever_bm25_v1",
            "generator_dependency": GENERATOR_DEPENDENCY_CONDITIONED,
            "conditioning_generator_id": "gen-a",
            "evaluation_generator_id": "gen-a",
            "experiment_type": experiment_type,
            "seed": 13,
            "evaluation_generator_identity_fingerprint": "generator-identity-sha",
            "selection_sha256": "selection-sha",
            "selection_manifest_sha256": "selection-manifest-sha",
            "selection_manifest_fingerprint": "selection-manifest-fingerprint",
            "generator_manifest_sha256": "generator-manifest-sha",
        },
        "formal_registry_version": FORMAL_REGISTRY_VERSION,
        "formal_registry_fingerprint": FORMAL_REGISTRY_FINGERPRINT,
        "generator_registry_fingerprint": freeze[
            "generator_registry_fingerprint"
        ],
        "retrieval_protocol_fingerprint": freeze[
            "retrieval_protocol_fingerprint"
        ],
        "contract": {
            "split": "held_out_test",
            "config_sha256": freeze["config_sha256"],
            "prompt_hash": freeze["prompt_hash"],
            "verbalizer_hash": freeze["verbalizer_hash"],
        },
        "predictions_sha256": "predictions-sha",
        "metrics_sha256": "metrics-sha",
        "git": {"commit": freeze["git_commit"]},
        "formal_freeze": freeze,
    }


def test_canonical_main_methods_and_ours_are_exact() -> None:
    assert MAIN_TABLE_METHODS == (
        "no_evidence",
        "retrieval_topk",
        "bge",
        "infogain",
        "rag_cbwdm_signed_v1",
    )
    assert CANONICAL_OURS == "rag_cbwdm_signed_v1"
    assert method_spec(CANONICAL_OURS)["display_name"] == "Ours"


@pytest.mark.parametrize(
    "method_id", [KCBWDM_SIGNED_V1, KCBWDM_LINEAR_GATE_V2]
)
def test_kcbwdm_is_development_only_and_never_main_table_or_held_out(
    method_id: str,
) -> None:
    spec = method_spec(method_id)
    assert spec["generator_dependency"] == GENERATOR_DEPENDENCY_CONDITIONED
    assert spec["learned_selector"] is True
    assert spec["state_aware"] is True
    assert spec["development_matrix_eligible"] is True
    assert spec["development_only"] is True
    assert spec["held_out_eligible"] is False
    assert spec["main_table_eligible"] is False
    assert method_id not in MAIN_TABLE_METHODS
    assert CANONICAL_OURS == "rag_cbwdm_signed_v1"
    assert spec["seed_policy"]["seeds"] == [13]

    validate_dataset_method_compatibility(
        "fm2_official_closed_page_v1",
        method_id,
        "fm2_official_closed_page_v1",
        development=True,
    )
    with pytest.raises(ValueError, match="not formal-v2 main-table eligible"):
        validate_dataset_method_compatibility(
            "fm2_official_closed_page_v1",
            method_id,
            "fm2_official_closed_page_v1",
        )


@pytest.mark.parametrize(
    "method_id",
    [
        "rag_cbwdm",
        "rag_cbwdm_signed_v2",
        "rag_cbwdm_signed_v21",
        "rag_cbwdm_signed_v22",
        "signed_gate_oracle",
        "cbwdm_oracle",
        "gold_oracle",
    ],
)
def test_legacy_diagnostics_and_oracles_are_not_main_table(method_id: str) -> None:
    assert method_spec(method_id)["main_table_eligible"] is False


def test_cross_generator_transfer_is_not_main_table_eligible() -> None:
    candidate = signed_result(experiment_type="cross_generator_transfer")
    candidate["artifact_binding"]["evaluation_generator_id"] = "gen-b"
    with pytest.raises(ValueError, match="never main-table eligible"):
        validate_main_table_result(candidate)


def test_dataset_retrieval_protocols_and_publication_labels() -> None:
    fever = dataset_protocol("fever_binary_v2")
    assert fever["retrieval_protocol_id"] == "fever_bm25_v1"
    assert fever["retrieval_topk_display"] == "BM25 Top-k"
    assert fever["main_table_column"] == "Retrieval Top-k"

    fm2 = dataset_protocol("fm2_official_closed_page_v1")
    assert fm2["retrieval_protocol_id"] == "fm2_official_closed_page_v1"
    assert fm2["retrieval_topk_display"] == "Official-pool Top-k"
    assert "bm25" not in fm2["retrieval_topk_display"].casefold()
    assert fm2["fm2_bm25_implemented"] is False


@pytest.mark.parametrize(
    ("dataset_id", "wrong_protocol"),
    [
        ("fever_binary_v2", "fm2_official_closed_page_v1"),
        ("fm2_official_closed_page_v1", "fever_bm25_v1"),
    ],
)
def test_cross_dataset_retrieval_protocol_mismatch_fails(
    dataset_id: str, wrong_protocol: str
) -> None:
    with pytest.raises(ValueError, match="Dataset/retrieval protocol mismatch"):
        validate_dataset_method_compatibility(
            dataset_id, "retrieval_topk", wrong_protocol
        )


def test_seed_policy_distinguishes_learned_and_deterministic_methods() -> None:
    expected = {13, 21, 42}
    assert set(method_spec("infogain")["seed_policy"]["seeds"]) == expected
    assert set(method_spec(CANONICAL_OURS)["seed_policy"]["seeds"]) == expected
    assert set(LEARNED_METHOD_SEEDS) == expected
    for method_id in ("no_evidence", "retrieval_topk", "bge"):
        policy = method_spec(method_id)["seed_policy"]
        assert policy["seeds"] == []
        assert policy["replicate_across_learned_seeds"] is False


def test_development_splits_are_not_final_test_roles() -> None:
    assert split_role("fever_binary_v2", "validation") == "development_calibration"
    assert not is_final_split("fever_binary_v2", "validation")
    assert is_final_split("fever_binary_v2", "held_out_test")
    assert split_role("fm2_official_closed_page_v1", "validation") == "development_calibration"
    assert not is_final_split("fm2_official_closed_page_v1", "validation")
    assert is_final_split("fm2_official_closed_page_v1", "held_out_test")


def test_held_out_readiness_refuses_missing_freeze_and_signoff() -> None:
    assert held_out_freeze_status(None)["ready"] is False
    draft = held_out_freeze_template("fever_binary_v2")
    blocked = held_out_freeze_status(draft)
    assert blocked["ready"] is False
    assert any("sign-off" in reason for reason in blocked["blockers"])
    readiness = check_formal_v2_registry_readiness(
        dataset_id="fever_binary_v2", held_out_freeze=draft
    )
    assert readiness["status"] == "blocked"
    assert readiness["canonical_ours"] == CANONICAL_OURS
    assert "bge_model_id_revision_sha" in readiness["unresolved_freeze_decisions"]


def test_complete_freeze_can_authorize_source_level_readiness() -> None:
    freeze = ready_freeze("fm2_official_closed_page_v1")
    assert held_out_freeze_status(freeze)["ready"] is True
    readiness = check_formal_v2_registry_readiness(
        dataset_id="fm2_official_closed_page_v1", held_out_freeze=freeze
    )
    assert readiness["status"] == "ready"
    assert readiness["historical_untouched_state_proven_by_source"] is False
    assert readiness["unresolved_freeze_decisions"] == []


def test_registry_fingerprint_is_deterministic_and_semantic() -> None:
    first = build_formal_registry()
    second = build_formal_registry()
    assert first["schema_version"] == FORMAL_REGISTRY_SCHEMA_VERSION
    assert formal_registry_fingerprint(first) == formal_registry_fingerprint(second)
    assert formal_registry_fingerprint(first) == FORMAL_REGISTRY_FINGERPRINT
    changed = copy.deepcopy(first)
    changed["methods"]["bge"]["seed_policy"]["kind"] = "changed"
    assert formal_registry_fingerprint(changed) != FORMAL_REGISTRY_FINGERPRINT


def test_registry_identity_pairs_are_versioned_and_fail_closed() -> None:
    historical = HISTORICAL_FORMAL_REGISTRY_IDENTITIES["formal_v2.0"]
    assert FORMAL_REGISTRY_VERSION == "formal_v2.1"
    assert FORMAL_REGISTRY_FINGERPRINT == (
        "d84198410a3f39b265082a6085f73cbc959f4fac7bd7db1c3b8b5310b105e9a5"
    )
    validate_formal_registry_identity("formal_v2.0", historical)
    validate_formal_registry_identity(
        FORMAL_REGISTRY_VERSION, FORMAL_REGISTRY_FINGERPRINT
    )
    with pytest.raises(ValueError, match="version/fingerprint mismatch"):
        validate_formal_registry_identity("formal_v2.0", FORMAL_REGISTRY_FINGERPRINT)
    with pytest.raises(ValueError, match="version/fingerprint mismatch"):
        validate_formal_registry_identity(FORMAL_REGISTRY_VERSION, historical)
    with pytest.raises(ValueError, match="Unknown formal registry version"):
        validate_formal_registry_identity("formal_v999.0", historical)


def test_historical_formal_v20_result_remains_valid() -> None:
    historical = HISTORICAL_FORMAL_REGISTRY_IDENTITIES["formal_v2.0"]
    candidate = signed_result()
    freeze = candidate["formal_freeze"]
    freeze["formal_registry_version"] = "formal_v2.0"
    freeze["method_registry_fingerprint"] = historical
    freeze["freeze_fingerprint"] = held_out_freeze_fingerprint(freeze)
    freeze["runtime_sign_off"]["freeze_fingerprint"] = freeze[
        "freeze_fingerprint"
    ]
    candidate["formal_registry_version"] = "formal_v2.0"
    candidate["formal_registry_fingerprint"] = historical
    validated = validate_main_table_result(candidate)
    assert validated["registry_version"] == "formal_v2.0"
    assert validated["registry_fingerprint"] == historical


def test_historical_registries_remain_readable_and_unchanged() -> None:
    assert "rag_cbwdm" in LEGACY_CANONICAL_METHODS
    assert "rag_cbwdm" in PREFORMAL_METHODS
    assert method_spec("rag_cbwdm")["main_table_eligible"] is False


def test_formal_config_exposes_non_authorizing_freeze_template() -> None:
    requirements = formal_v2_held_out_freeze_requirements("fever_binary_v2")
    assert requirements["formal_registry_version"] == FORMAL_REGISTRY_VERSION
    assert requirements["formal_registry_fingerprint"] == FORMAL_REGISTRY_FINGERPRINT
    assert requirements["authorizes_held_out_execution"] is False


def test_matched_signed_result_is_eligible_only_with_complete_freeze() -> None:
    eligible = validate_main_table_result(signed_result())
    assert eligible["eligible"] is True
    assert eligible["canonical_method_id"] == CANONICAL_OURS
    blocked = signed_result()
    blocked["formal_freeze"] = held_out_freeze_template("fever_binary_v2")
    with pytest.raises(ValueError, match="freeze/sign-off"):
        validate_main_table_result(blocked)

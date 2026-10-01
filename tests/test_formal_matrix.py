from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.run_formal_matrix import run_cli
from src.formal_matrix import (
    FULL_DEVELOPMENT,
    HELD_OUT,
    MATRIX_CONFIG_SCHEMA_VERSION,
    MatrixPlanError,
    build_execution_plan,
    load_matrix_config,
    validate_dag,
)
from src.formal_registry import CANONICAL_OURS, MAIN_TABLE_METHODS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = PROJECT_ROOT / "configs/formal/fever_qwen15_development_smoke.yaml"
TEST_GIT = {"commit": "test-commit", "dirty": False}


def plan_for(config: dict, **overrides):
    arguments = {
        "project_root": PROJECT_ROOT,
        "server_project_root": "/srv/rag-cbwdm",
        "git": TEST_GIT,
    }
    arguments.update(overrides)
    return build_execution_plan(config, **arguments)


@pytest.fixture(scope="module")
def smoke_config() -> dict:
    return load_matrix_config(SMOKE_CONFIG)


@pytest.fixture(scope="module")
def smoke_plan(smoke_config: dict) -> dict:
    return plan_for(smoke_config)


def nodes(plan: dict, *, stage: str | None = None, method: str | None = None):
    result = plan["nodes"]
    if stage is not None:
        result = [node for node in result if node["stage"] == stage]
    if method is not None:
        result = [node for node in result if node["method_id"] == method]
    return result


def test_fever_qwen15_all_main_methods_dry_run_plan(smoke_plan: dict) -> None:
    assert tuple(smoke_plan["methods"]) == MAIN_TABLE_METHODS
    assert smoke_plan["dataset_identity"]["dataset_id"] == "fever_binary_v2"
    assert {row["method_id"] for row in smoke_plan["result_index"]} == set(
        MAIN_TABLE_METHODS
    )


def test_deterministic_methods_have_no_seed_dimension(smoke_plan: dict) -> None:
    deterministic = {"no_evidence", "retrieval_topk", "bge"}
    relevant = [
        node for node in smoke_plan["nodes"] if node["method_id"] in deterministic
    ]
    assert relevant
    assert all(node["seed"] is None for node in relevant)
    assert all(smoke_plan["seed_policy"][method] == [] for method in deterministic)


def test_requested_seed_expands_both_learned_methods(smoke_plan: dict) -> None:
    for method in ("infogain", CANONICAL_OURS):
        assert smoke_plan["seed_policy"][method] == [13]
        assert {node["seed"] for node in nodes(smoke_plan, stage="training", method=method)} == {
            13
        }
        assert {row["seed"] for row in smoke_plan["result_index"] if row["method_id"] == method} == {
            13
        }


def test_training_runtime_is_explicit_and_plan_fingerprinted(smoke_plan: dict) -> None:
    runtime = smoke_plan["training_runtime"]
    assert runtime["infogain"]["rank_loss_implementation"] == "vectorized"
    assert runtime[CANONICAL_OURS]["optimizer_group_batch_size"] == 8
    assert runtime[CANONICAL_OURS]["forward_batch_size"] == 32
    info_command = nodes(smoke_plan, stage="training", method="infogain")[0]["command"]
    signed_command = nodes(smoke_plan, stage="training", method=CANONICAL_OURS)[0]["command"]
    assert info_command[info_command.index("--rank-loss-implementation") + 1] == "vectorized"
    assert signed_command[signed_command.index("--runtime-implementation") + 1] == "block_v1"
    assert signed_command[signed_command.index("--forward-batch-size") + 1] == "32"


def test_full_development_defaults_to_all_formal_seeds(smoke_config: dict) -> None:
    config = copy.deepcopy(smoke_config)
    config["profile"] = FULL_DEVELOPMENT
    config.pop("learned_seeds")
    plan = plan_for(config)
    assert plan["seed_policy"]["infogain"] == [13, 21, 42]
    assert plan["seed_policy"][CANONICAL_OURS] == [13, 21, 42]


@pytest.mark.parametrize("method", ["no_evidence", "retrieval_topk", "bge"])
def test_generator_independent_selection_is_shared(
    smoke_plan: dict, method: str
) -> None:
    selection = nodes(smoke_plan, stage="selection", method=method)
    assert len(selection) == 1
    assert selection[0]["generator_id"] is None
    assert selection[0]["reusable"] is True


def test_all_evaluations_are_generator_specific(smoke_plan: dict) -> None:
    evaluations = nodes(smoke_plan, stage="evaluation")
    assert evaluations
    assert all(node["generator_id"] == "qwen2.5-1.5b-instruct" for node in evaluations)


def test_learned_posteriors_and_teachers_are_generator_specific(
    smoke_plan: dict,
) -> None:
    posteriors = nodes(smoke_plan, stage="posteriors")
    teachers = nodes(smoke_plan, stage="teacher")
    assert posteriors and teachers
    assert all(node["generator_id"] is not None for node in posteriors + teachers)
    assert {node["method_id"] for node in teachers} == {"infogain", CANONICAL_OURS}


def test_two_generators_branch_learned_work_and_reuse_shared_selections(
    smoke_config: dict, tmp_path: Path,
) -> None:
    config = copy.deepcopy(smoke_config)
    qwen7_config = load_matrix_config(PROJECT_ROOT / "configs/fever2_server_smoke.yaml")
    qwen7_config["generator"]["model_name"] = "Qwen/Qwen2.5-7B-Instruct"
    qwen7_path = tmp_path / "qwen7.json"
    qwen7_path.write_text(json.dumps(qwen7_config), encoding="utf-8")
    config["generators"].append(
        {
            "generator_id": "qwen2.5-7b-instruct",
            "model_family": "qwen2.5",
            "config": str(qwen7_path),
            "server_config": "configs/formal/qwen7-server.yaml",
        }
    )
    plan = plan_for(config)
    for method in ("infogain", CANONICAL_OURS):
        assert {node["generator_id"] for node in nodes(plan, stage="training", method=method)} == {
            "qwen2.5-1.5b-instruct",
            "qwen2.5-7b-instruct",
        }
    assert len(nodes(plan, stage="selection", method="bge")) == 1
    assert len(nodes(plan, stage="selection", method="retrieval_topk")) == 1
    assert {node["generator_id"] for node in nodes(plan, stage="evaluation")} == {
        "qwen2.5-1.5b-instruct",
        "qwen2.5-7b-instruct",
    }


def test_fm2_uses_official_pool_not_bm25() -> None:
    config = {
        "schema_version": MATRIX_CONFIG_SCHEMA_VERSION,
        "profile": "development_smoke",
        "dataset_id": "fm2_official_closed_page_v1",
        "retrieval_protocol_id": "fm2_official_closed_page_v1",
        "dataset_config": "configs/fm2_server_smoke.yaml",
        "training_split": "train",
        "evaluation_split": "dev",
        "artifact_root": "artifacts/formal_v2",
        "retrieval_inputs": {
            "train": "artifacts/formal_v2/fm2_official_closed_page_v1/shared/retrieval/train.jsonl",
            "dev": "artifacts/formal_v2/fm2_official_closed_page_v1/shared/retrieval/dev.jsonl",
        },
        "generators": [
            {
                "generator_id": "qwen2.5-1.5b-instruct",
                "config": "configs/fm2_server_smoke.yaml",
            }
        ],
        "methods": list(MAIN_TABLE_METHODS),
        "learned_seeds": [13],
        "bge": {
            "model_id": "development-only",
            "model_name_or_path": "/srv/models/bge-reranker-large",
            "development_only": True,
        },
    }
    plan = plan_for(config)
    assert plan["retrieval_protocol"]["retrieval_protocol_id"] == (
        "fm2_official_closed_page_v1"
    )
    assert plan["retrieval_protocol"]["retrieval_topk_display"] == (
        "Official-pool Top-k"
    )
    assert "bm25" not in plan["retrieval_protocol"]["retrieval_topk_display"].lower()


def test_cross_dataset_protocol_mismatch_fails(smoke_config: dict) -> None:
    config = copy.deepcopy(smoke_config)
    config["retrieval_protocol_id"] = "fm2_official_closed_page_v1"
    with pytest.raises(ValueError, match="protocol mismatch"):
        plan_for(config)


def test_smoke_is_development_and_never_claims_final(smoke_plan: dict) -> None:
    assert smoke_plan["profile"] == "development_smoke"
    assert smoke_plan["evaluation_split"] == "validation"
    assert smoke_plan["split_role"] == "development_calibration"
    assert smoke_plan["held_out"] is False
    assert smoke_plan["formal_result_claim"] is False


def test_held_out_requires_explicit_valid_freeze_and_signoff(smoke_config: dict) -> None:
    with pytest.raises(MatrixPlanError, match="Held-out planning is blocked"):
        plan_for(smoke_config, held_out=True, held_out_freeze=None)


def test_unresolved_bge_identity_blocks_held_out(
    smoke_config: dict, smoke_plan: dict
) -> None:
    config = copy.deepcopy(smoke_config)
    config["profile"] = HELD_OUT
    config["evaluation_split"] = "held_out_test"
    config["retrieval_inputs"]["held_out_test"] = (
        "artifacts/formal_v2/fever_binary_v2/shared/retrieval/held_out_test.jsonl"
    )
    freeze = {
        "generator_registry_fingerprint": smoke_plan[
            "generator_registry_fingerprint"
        ],
        "git_commit": TEST_GIT["commit"],
        "config_sha256": smoke_plan["dataset_config_sha256"],
        "learned_method_seeds": {
            "infogain": [13],
            CANONICAL_OURS: [13],
        },
        "prompt_hash": smoke_plan["generator_identities"][0][
            "generator_identity"
        ]["prompt_template_hash"],
        "verbalizer_hash": smoke_plan["generator_identities"][0][
            "generator_identity"
        ]["verbalizer_hash"],
        "top_k": 2,
        "evidence_budget": {"max_docs": 2},
        "bge_model_freeze": {
            "model_id": "frozen-bge",
            "revision": "immutable-revision",
            "sha256": "frozen-sha",
        },
    }
    with patch(
        "src.formal_matrix.held_out_freeze_status",
        return_value={"ready": True, "blockers": []},
    ):
        with pytest.raises(MatrixPlanError, match="BGE model"):
            plan_for(config, held_out=True, held_out_freeze=freeze)


def test_bge_identity_can_be_supplied_by_manifest(
    smoke_config: dict, tmp_path: Path
) -> None:
    manifest = tmp_path / "bge-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "model_id": "development/bge",
                "revision": "immutable-revision",
                "sha256": "a" * 64,
                "model_name_or_path": "/srv/models/bge",
                "development_only": True,
            }
        ),
        encoding="utf-8",
    )
    config = copy.deepcopy(smoke_config)
    config["bge"] = {"manifest": str(manifest)}
    plan = plan_for(config)
    assert plan["bge_contract"]["model_id"] == "development/bge"
    assert plan["bge_contract"]["manifest_sha256"]
    command = nodes(plan, stage="selection", method="bge")[0]["command"]
    assert command[command.index("--model-sha256") + 1] == "a" * 64


def test_illegal_formal_seed_fails(smoke_config: dict) -> None:
    config = copy.deepcopy(smoke_config)
    config["profile"] = FULL_DEVELOPMENT
    config["learned_seeds"] = [99]
    with pytest.raises(MatrixPlanError, match="Illegal learned seed"):
        plan_for(config)


def test_output_collision_and_cycle_fail_closed(smoke_plan: dict) -> None:
    colliding = copy.deepcopy(smoke_plan["nodes"][:2])
    colliding[1]["outputs"] = copy.deepcopy(colliding[0]["outputs"])
    with pytest.raises(MatrixPlanError, match="Output collision"):
        validate_dag(colliding)
    cyclic = copy.deepcopy(smoke_plan["nodes"][:2])
    cyclic[0]["dependencies"] = [cyclic[1]["node_id"]]
    cyclic[1]["dependencies"] = [cyclic[0]["node_id"]]
    with pytest.raises(MatrixPlanError, match="cycle"):
        validate_dag(cyclic)


def test_plan_fingerprint_is_deterministic_and_semantic(smoke_config: dict) -> None:
    first = plan_for(smoke_config)
    second = plan_for(smoke_config)
    assert first["plan_fingerprint"] == second["plan_fingerprint"]
    relocated = plan_for(smoke_config, server_project_root="/different/server/root")
    assert relocated["plan_fingerprint"] == first["plan_fingerprint"]
    changed = copy.deepcopy(smoke_config)
    changed["learned_seeds"] = [21]
    assert plan_for(changed)["plan_fingerprint"] != first["plan_fingerprint"]
    changed_retrieval = copy.deepcopy(smoke_config)
    changed_retrieval["retrieval_inputs"]["validation"] = (
        "inputs/another-compatible-validation-pool.jsonl"
    )
    assert plan_for(changed_retrieval)["plan_fingerprint"] != first[
        "plan_fingerprint"
    ]


def test_dry_run_executes_no_subprocess_and_writes_plan(
    tmp_path: Path,
) -> None:
    calls = []
    plan_path = tmp_path / "plan.json"
    commands_path = tmp_path / "commands.sh"
    plan = run_cli(
        [
            "--config",
            str(SMOKE_CONFIG),
            "--dry-run",
            "--plan-output",
            str(plan_path),
            "--commands-output",
            str(commands_path),
            "--project-root",
            str(PROJECT_ROOT),
            "--server-project-root",
            "/srv/rag-cbwdm",
        ],
        execute=lambda candidate: calls.append(candidate),
    )
    assert calls == []
    assert json.loads(plan_path.read_text(encoding="utf-8"))["plan_fingerprint"] == plan[
        "plan_fingerprint"
    ]
    assert "03_compute_label_posteriors.py" in commands_path.read_text(
        encoding="utf-8"
    )


def test_plan_targets_formal_v2_without_mutating_legacy_paths(
    smoke_plan: dict,
) -> None:
    outputs = [
        path for node in smoke_plan["nodes"] for path in node["outputs"].values()
    ]
    generated = [path for path in outputs if "/artifacts/" in path]
    assert generated
    assert all("/artifacts/formal_v2/" in path for path in generated)
    assert all("/artifacts/formal/" not in path for path in generated)

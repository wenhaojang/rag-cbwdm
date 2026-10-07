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
    _semantic_plan_payload,
    _validate_external_fm2_retrieval,
    _validate_external_posterior,
    build_execution_plan,
    load_matrix_config,
    validate_dag,
)
from src.formal_registry import (
    CANONICAL_OURS,
    FORMAL_REGISTRY_FINGERPRINT,
    FORMAL_REGISTRY_VERSION,
    KCBWDM_LINEAR_GATE_V2,
    KCBWDM_NORMALIZED_RHO_V1,
    KCBWDM_SIGNED_V1,
    MAIN_TABLE_METHODS,
)
from src.experiment_identity import (
    POSTERIOR_MANIFEST_SCHEMA_VERSION,
    build_generator_identity,
    experiment_identity_payload,
    resolve_dataset_identity,
    resolve_retrieval_protocol_identity,
)
from src.datasets.fm2 import (
    FM2_DATASET_FAMILY,
    FM2_DATASET_ID,
    FM2_EXPECTED_SHA256,
    FM2_FORMAL_ROLE_BY_OFFICIAL_SPLIT,
    FM2_PREPARE_MANIFEST_SCHEMA_VERSION,
    FM2_RETRIEVAL_PROTOCOL_ID,
    FM2_SOURCE_COMMIT,
    adapt_raw_row,
)
from src.io_utils import write_jsonl
from src.run_manifest import sha256_file, stable_hash


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = PROJECT_ROOT / "configs/formal/fever_qwen15_development_smoke.yaml"
FM2_FULL_CONFIG = (
    PROJECT_ROOT / "configs/formal/fm2_full_development.seed13.matrix.server.yaml"
)
FM2_KCBWDM_FULL_CONFIG = (
    PROJECT_ROOT
    / "configs/formal/fm2_kcbwdm_full_development.seed13.matrix.server.yaml"
)
FM2_KCBWDM_V2A_SMOKE_CONFIG = (
    PROJECT_ROOT
    / "configs/formal/"
    "fm2_qwen15_kcbwdm_linear_gate_v2_development_smoke.seed13.matrix.server.yaml"
)
FM2_KCBWDM_V2A_FULL_CONFIG = (
    PROJECT_ROOT
    / "configs/formal/"
    "fm2_qwen15_kcbwdm_linear_gate_v2_full_development.seed13.matrix.server.yaml"
)
FM2_KCBWDM_V2A_REMAINING3_FULL_CONFIG = (
    PROJECT_ROOT
    / "configs/formal/"
    "fm2_remaining3_kcbwdm_linear_gate_v2_full_development.seed13.matrix.server.yaml"
)
FM2_KCBWDM_NORMALIZED_RHO_ENDPOINTS_CONFIG = (
    PROJECT_ROOT
    / "configs/formal/"
    "fm2_qwen15_qwen7_kcbwdm_normalized_rho_endpoints_development.seed13.matrix.server.yaml"
)
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


def fm2_raw_row(identifier: str) -> dict:
    return {
        "category": "Science",
        "correct_votes": 2,
        "gold_evidence": [{"section_header": "Gold", "text": "Gold sentence."}],
        "id": identifier,
        "label": "SUPPORTS",
        "retrieved_evidence": [
            {"section_header": "Lead", "text": "Candidate A."},
            {"section_header": "History", "text": "Candidate B."},
        ],
        "text": "A claim.",
        "total_likes": 1,
        "total_votes": 3,
        "wikipedia_page": f"Page {identifier}",
    }


def fm2_shared_retrieval_inputs(tmp_path: Path) -> dict[str, dict[str, str]]:
    split_entries = {}
    pool_paths = {}
    for official_split in ("train", "dev"):
        formal_role = FM2_FORMAL_ROLE_BY_OFFICIAL_SPLIT[official_split]
        _, pool, _ = adapt_raw_row(
            fm2_raw_row(official_split), split=official_split, row_number=1
        )
        pool_path = tmp_path / f"{formal_role}.jsonl"
        write_jsonl(pool_path, [pool])
        pool_paths[formal_role] = pool_path
        split_entries[official_split] = {
            "official_source_split": official_split,
            "formal_role": formal_role,
            "raw_sha256": FM2_EXPECTED_SHA256[official_split],
            "candidate_pool_path": str(pool_path.resolve()),
            "candidate_pool_sha256": sha256_file(pool_path),
            "num_rows": 1,
        }
    manifest = {
        "schema_version": FM2_PREPARE_MANIFEST_SCHEMA_VERSION,
        "status": "completed",
        "dataset_id": FM2_DATASET_ID,
        "dataset_family": FM2_DATASET_FAMILY,
        "retrieval_protocol_id": FM2_RETRIEVAL_PROTOCOL_ID,
        "source": {"commit": FM2_SOURCE_COMMIT},
        "formal_role_mapping": dict(FM2_FORMAL_ROLE_BY_OFFICIAL_SPLIT),
        "candidate_pool_contract": {
            "protocol": FM2_RETRIEVAL_PROTOCOL_ID,
            "source_order_preserved": True,
            "construction_gold_free": True,
        },
        "splits": split_entries,
    }
    manifest_path = tmp_path / "fm2_prepare.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        formal_role: {
            "pool": str(pool_path),
            "manifest": str(manifest_path),
            "server_pool": f"artifacts/fm2/{formal_role}.jsonl",
            "server_manifest": "artifacts/fm2/fm2_prepare.manifest.json",
        }
        for formal_role, pool_path in pool_paths.items()
    }


def fm2_matrix_config(tmp_path: Path, generator_ids: list[str]) -> dict:
    model_names = {
        "qwen2.5-0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
        "qwen2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
        "mistral-7b-instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    }
    generators = []
    for generator_id in generator_ids:
        payload = load_matrix_config(PROJECT_ROOT / "configs/fm2_server_smoke.yaml")
        payload["generator"]["model_name"] = model_names[generator_id]
        config_path = tmp_path / f"{generator_id}.json"
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        generators.append(
            {
                "generator_id": generator_id,
                "config": str(config_path),
                "server_config": f"configs/generated/{generator_id}.yaml",
            }
        )
    retrieval_inputs = fm2_shared_retrieval_inputs(tmp_path)
    return {
        "schema_version": MATRIX_CONFIG_SCHEMA_VERSION,
        "profile": FULL_DEVELOPMENT,
        "dataset_id": FM2_DATASET_ID,
        "retrieval_protocol_id": FM2_RETRIEVAL_PROTOCOL_ID,
        "dataset_config": "configs/fm2_server_smoke.yaml",
        "training_split": "train_core",
        "evaluation_split": "validation",
        "artifact_root": "artifacts/formal_v2",
        "retrieval_inputs": {
            "train_core": retrieval_inputs["train_core"],
            "validation": retrieval_inputs["validation"],
        },
        "generators": generators,
        "methods": list(MAIN_TABLE_METHODS),
        "learned_seeds": [13],
        "bge": {
            "model_id": "development-only",
            "model_name_or_path": "/srv/models/bge-reranker-large",
            "development_only": True,
        },
    }


def add_external_fm2_posteriors(
    config: dict, tmp_path: Path, *, generator_id: str
) -> dict[str, tuple[Path, Path]]:
    model_names = {
        "qwen2.5-0.5b-instruct": "/root/models/Qwen2.5-0.5B-Instruct",
        "qwen2.5-1.5b-instruct": "/root/models/Qwen2.5-1.5B-Instruct",
        "qwen2.5-7b-instruct": "/root/models/Qwen2.5-7B-Instruct",
        "mistral-7b-instruct-v0.3": "/root/models/Mistral-7B-Instruct-v0.3",
    }
    model_name = model_names[generator_id]
    generator_offset = list(model_names).index(generator_id) + 1
    configured_entries = config.get("posterior_inputs", {}).get(generator_id, {})
    dataset = resolve_dataset_identity("fm2", explicit_dataset_id=FM2_DATASET_ID)
    generator = build_generator_identity(
        model_name_or_path=model_name,
        generator_id=generator_id,
        formal_v2=True,
        model_sha256="b" * 64,
        tokenizer_name_or_path=model_name,
        prompt_template_version="fm2_classification.v1",
        prompt_template_hash="p" * 64,
        verbalizer_hash="v" * 64,
    )
    paths: dict[str, tuple[Path, Path]] = {}
    entries = {}
    for split in ("train_core", "validation"):
        posterior = tmp_path / generator_id / f"{split}.posteriors.jsonl"
        posterior.parent.mkdir(parents=True, exist_ok=True)
        support_probability = 0.5 + generator_offset * 0.01
        write_jsonl(
            posterior,
            [
                {
                    "schema_version": "rag_cbwdm_posteriors.v2",
                    "id": f"{split}:q1",
                    "query": "claim",
                    "label": "SUPPORTS",
                    "split": split,
                    "labels": ["SUPPORTS", "REFUTES"],
                    "eta0": [support_probability, 1.0 - support_probability],
                    "candidates": [],
                }
            ],
        )
        retrieval_sha = sha256_file(Path(config["retrieval_inputs"][split]["pool"]))
        retrieval = resolve_retrieval_protocol_identity(
            dataset_identity=dataset,
            source_artifact_sha256=retrieval_sha,
            retrieval_method=FM2_RETRIEVAL_PROTOCOL_ID,
            formal_v2=True,
        )
        identity = experiment_identity_payload(dataset, generator, retrieval)
        manifest = {
            "schema_version": POSTERIOR_MANIFEST_SCHEMA_VERSION,
            "stage": "posterior",
            "status": "completed",
            "fingerprint": stable_hash({"generator": generator_id, "split": split}),
            "identity_mode": "formal_v2",
            "dataset_identity": identity["dataset_identity"],
            "generator_identity": identity["generator_identity"],
            "retrieval_protocol_identity": identity["retrieval_protocol_identity"],
            "identity_fingerprint": stable_hash(identity),
            "provenance": {
                "dataset": "fm2",
                "split": split,
                "generator_model": model_name,
                "generator_sha256": "b" * 64,
                "input_sha256": retrieval_sha,
                "config_sha256": "c" * 64,
                "prompt_template_hash": "p" * 64,
                "verbalizers_hash": "v" * 64,
            },
            "git": {"commit": "1" * 40, "branch": "test", "dirty": False},
            "expected_rows": 1,
            "completed_rows": 1,
            "output_path": str(posterior.resolve()),
            "output_sha256": sha256_file(posterior),
        }
        manifest_path = posterior.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        configured_binding = configured_entries.get(split, {})
        entries[split] = {
            "posteriors": str(posterior),
            "manifest": str(manifest_path),
            "server_posteriors": configured_binding.get(
                "server_posteriors",
                f"/srv/existing/{generator_id}/{split}/posteriors.jsonl",
            ),
            "server_manifest": configured_binding.get(
                "server_manifest",
                f"/srv/existing/{generator_id}/{split}/posteriors.manifest.json",
            ),
        }
        paths[split] = (posterior, manifest_path)
    config.setdefault("posterior_inputs", {})[generator_id] = entries
    return paths


def test_fever_qwen15_all_main_methods_dry_run_plan(smoke_plan: dict) -> None:
    assert tuple(smoke_plan["methods"]) == MAIN_TABLE_METHODS
    assert smoke_plan["dataset_identity"]["dataset_id"] == "fever_binary_v2"
    assert {row["method_id"] for row in smoke_plan["result_index"]} == set(
        MAIN_TABLE_METHODS
    )
    assert all(
        row["retrieval_protocol_id"] == "fever_bm25_v1"
        and row["retrieval_protocol_fingerprint"]
        for row in smoke_plan["result_index"]
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
    runtime = "/root/miniconda3/envs/rag-cbwdm-mistral/bin/python"
    plan = plan_for(config, server_python=runtime)
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


def test_fm2_uses_official_pool_not_bm25(tmp_path: Path) -> None:
    config = fm2_matrix_config(tmp_path, ["qwen2.5-1.5b-instruct"])
    plan = plan_for(config)
    assert plan["retrieval_protocol"]["retrieval_protocol_id"] == (
        "fm2_official_closed_page_v1"
    )
    assert plan["retrieval_protocol"]["retrieval_topk_display"] == (
        "Official-pool Top-k"
    )
    assert "bm25" not in plan["retrieval_protocol"]["retrieval_topk_display"].lower()
    commands = "\n".join(" ".join(node["command"]) for node in plan["nodes"])
    assert "02_retrieve_bm25.py" not in commands
    assert "pyserini" not in commands.casefold()
    assert "fever_bm25_v1" not in commands
    assert all(
        row["retrieval_protocol_id"] == FM2_RETRIEVAL_PROTOCOL_ID
        and row["retrieval_protocol_fingerprint"]
        and row["method"] == row["method_id"]
        for row in plan["result_index"]
    )


def test_fm2_shared_prepare_manifest_is_external_input_not_dag_output(
    tmp_path: Path,
) -> None:
    config = fm2_matrix_config(tmp_path, ["qwen2.5-1.5b-instruct"])
    plan = plan_for(config)
    retrieval_nodes = {
        node["split"]: node for node in nodes(plan, stage="retrieval_input")
    }
    train = retrieval_nodes["train_core"]
    validation = retrieval_nodes["validation"]

    assert len(validate_dag(plan["nodes"])) == len(plan["nodes"])
    assert set(train["outputs"]) == {"retrieval"}
    assert set(validation["outputs"]) == {"retrieval"}
    assert train["inputs"]["manifest"] == validation["inputs"]["manifest"]
    assert train["retrieval_binding"]["manifest"] == (
        validation["retrieval_binding"]["manifest"]
    )
    assert train["retrieval_binding"]["manifest_sha256"] == (
        validation["retrieval_binding"]["manifest_sha256"]
    )
    assert train["retrieval_binding"]["pool"] != (
        validation["retrieval_binding"]["pool"]
    )
    assert train["retrieval_binding"]["pool_sha256"] != (
        validation["retrieval_binding"]["pool_sha256"]
    )
    assert train["retrieval_binding"]["formal_role"] == "train_core"
    assert validation["retrieval_binding"]["formal_role"] == "validation"

    manifest_path = Path(config["retrieval_inputs"]["train_core"]["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["audit_note"] = "semantic manifest identity changed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    changed = plan_for(config)
    assert changed["plan_fingerprint"] != plan["plan_fingerprint"]


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("manifest", "status mismatch"),
        ("pool", "pool SHA-256 mismatch"),
        ("split", "formal role mismatch"),
        ("protocol", "retrieval_protocol_id mismatch"),
    ],
)
def test_fm2_shared_retrieval_execute_validation_fails_closed(
    tmp_path: Path, tamper: str, message: str
) -> None:
    config = fm2_matrix_config(tmp_path, ["qwen2.5-1.5b-instruct"])
    plan = plan_for(config)
    train = next(
        node
        for node in nodes(plan, stage="retrieval_input")
        if node["split"] == "train_core"
    )
    execution_node = copy.deepcopy(train)
    manifest_path = Path(config["retrieval_inputs"]["train_core"]["manifest"])
    pool_path = Path(config["retrieval_inputs"]["train_core"]["pool"])
    execution_node["inputs"]["manifest"] = str(manifest_path)
    execution_node["outputs"]["retrieval"] = str(pool_path)
    execution_node["retrieval_binding"]["manifest"] = str(manifest_path)

    _validate_external_fm2_retrieval(execution_node)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tamper == "manifest":
        manifest["status"] = "running"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif tamper == "pool":
        pool_path.write_text(
            pool_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
    elif tamper == "split":
        row = json.loads(pool_path.read_text(encoding="utf-8").strip())
        row["split"] = "validation"
        write_jsonl(pool_path, [row])
        manifest["splits"]["train"]["candidate_pool_sha256"] = sha256_file(
            pool_path
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif tamper == "protocol":
        manifest["retrieval_protocol_id"] = "fever_bm25_v1"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(tamper)

    with pytest.raises((MatrixPlanError, ValueError), match=message):
        _validate_external_fm2_retrieval(execution_node)


def test_fm2_retrieval_input_requires_completed_authoritative_manifest(
    tmp_path: Path,
) -> None:
    config = fm2_matrix_config(tmp_path, ["qwen2.5-1.5b-instruct"])
    Path(config["retrieval_inputs"]["train_core"]["manifest"]).unlink()
    with pytest.raises(MatrixPlanError, match="manifest does not exist"):
        plan_for(config)


def test_fm2_full_development_never_requires_or_mentions_held_out(
    tmp_path: Path,
) -> None:
    config = fm2_matrix_config(tmp_path, ["qwen2.5-1.5b-instruct"])
    plan = plan_for(config)
    assert plan["evaluation_split"] == "validation"
    assert not any(node["split"] == "held_out_test" for node in plan["nodes"])
    assert "held_out_test" not in json.dumps(plan["retrieval_inputs"], sort_keys=True)
    commands = "\n".join(" ".join(node["command"]) for node in plan["nodes"])
    assert "held_out_test" not in commands
    assert "fm2:test:" not in commands


def test_fm2_four_generator_full_development_topology(tmp_path: Path) -> None:
    generator_ids = [
        "qwen2.5-0.5b-instruct",
        "qwen2.5-1.5b-instruct",
        "qwen2.5-7b-instruct",
        "mistral-7b-instruct-v0.3",
    ]
    config = fm2_matrix_config(tmp_path, generator_ids)
    plan = plan_for(config)
    assert len(plan["result_index"]) == 20
    assert len(plan["nodes"]) == 62
    assert len(nodes(plan, stage="selection", method="no_evidence")) == 1
    assert len(nodes(plan, stage="selection", method="retrieval_topk")) == 1
    assert len(nodes(plan, stage="selection", method="bge")) == 1
    for method in ("infogain", CANONICAL_OURS):
        assert {
            node["generator_id"]
            for node in nodes(plan, stage="training", method=method)
        } == set(generator_ids)
        assert {
            node["generator_id"]
            for node in nodes(plan, stage="selection", method=method)
        } == set(generator_ids)
    assert len(validate_dag(plan["nodes"])) == 62
    assert not any(node["split"] == "held_out_test" for node in plan["nodes"])
    commands = "\n".join(" ".join(node["command"]) for node in plan["nodes"])
    assert "02_retrieve_bm25.py" not in commands
    assert "pyserini" not in commands.casefold()
    assert "fever_bm25_v1" not in commands


def test_kcbwdm_and_signed_share_external_posteriors_and_build_expected_dag(
    tmp_path: Path,
) -> None:
    generator_id = "qwen2.5-1.5b-instruct"
    config = fm2_matrix_config(tmp_path, [generator_id])
    config["methods"] = [CANONICAL_OURS, KCBWDM_SIGNED_V1]
    config["kcbwdm"] = load_matrix_config(
        PROJECT_ROOT
        / "configs/formal/fm2_qwen15_kcbwdm_development_smoke.seed13.matrix.server.yaml"
    )["kcbwdm"]
    add_external_fm2_posteriors(config, tmp_path, generator_id=generator_id)

    plan = plan_for(config)
    posterior_nodes = nodes(plan, stage="posteriors")
    assert len(posterior_nodes) == 2
    assert {node["split"] for node in posterior_nodes} == {
        "train_core",
        "validation",
    }
    assert all(node["command"] == [] for node in posterior_nodes)
    assert all(
        node["execution_policy"] == "validate_external_posterior"
        for node in posterior_nodes
    )
    rendered_commands = "\n".join(
        " ".join(node["command"]) for node in plan["nodes"]
    )
    assert "03_compute_label_posteriors.py" not in rendered_commands

    signed_teacher = nodes(plan, stage="teacher", method=CANONICAL_OURS)
    kernel_teacher = nodes(plan, stage="teacher", method=KCBWDM_SIGNED_V1)
    assert len(signed_teacher) == len(kernel_teacher) == 1
    assert signed_teacher[0]["inputs"]["posteriors"] == kernel_teacher[0]["inputs"]["posteriors"]
    assert signed_teacher[0]["inputs"]["posterior_manifest"] == kernel_teacher[0]["inputs"]["posterior_manifest"]
    assert len(nodes(plan, stage="training", method=KCBWDM_SIGNED_V1)) == 1
    assert len(nodes(plan, stage="selection", method=KCBWDM_SIGNED_V1)) == 1
    assert len(nodes(plan, stage="evaluation", method=KCBWDM_SIGNED_V1)) == 1
    assert {row["method_id"] for row in plan["result_index"]} == {
        CANONICAL_OURS,
        KCBWDM_SIGNED_V1,
    }
    assert len(plan["posterior_reuse_bindings"]) == 2
    assert all(
        KCBWDM_SIGNED_V1 in node["outputs"].get("checkpoint", "")
        for node in nodes(plan, stage="training", method=KCBWDM_SIGNED_V1)
    )


def test_kcbwdm_semantic_fingerprint_covers_all_kernel_contract_fields(
    tmp_path: Path,
) -> None:
    generator_id = "qwen2.5-1.5b-instruct"
    config = fm2_matrix_config(tmp_path, [generator_id])
    config["methods"] = [KCBWDM_SIGNED_V1]
    config["kcbwdm"] = load_matrix_config(
        PROJECT_ROOT
        / "configs/formal/fm2_qwen15_kcbwdm_development_smoke.seed13.matrix.server.yaml"
    )["kcbwdm"]
    add_external_fm2_posteriors(config, tmp_path, generator_id=generator_id)
    plan = plan_for(config)
    assert len(plan["nodes"]) == 10
    kernel_nodes = {
        stage: nodes(plan, stage=stage, method=KCBWDM_SIGNED_V1)[0]
        for stage in ("teacher", "training", "selection", "evaluation")
    }
    assert kernel_nodes["teacher"]["node_id"] in kernel_nodes["training"]["dependencies"]
    assert kernel_nodes["training"]["node_id"] in kernel_nodes["selection"]["dependencies"]
    assert kernel_nodes["selection"]["node_id"] in kernel_nodes["evaluation"]["dependencies"]
    original = stable_hash(_semantic_plan_payload(plan))

    mutations = [
        (None, "contract_version", "kcbwdm_signed_v1.v2"),
        ("kernel", "base_kernel", "linear"),
        ("kernel", "anchor", "unanchored"),
        ("kernel", "bandwidth_policy", "changed_policy"),
        ("kernel", "ridge_lambda", 0.02),
        (None, "sign_policy", "residual_alignment_gt_0"),
        ("teacher", "stop_threshold", 0.002),
        ("selector", "score_threshold", 0.1),
    ]
    for section, field, value in mutations:
        changed = copy.deepcopy(plan)
        contract = changed["method_contracts"][KCBWDM_SIGNED_V1]
        target = contract if section is None else contract[section]
        target[field] = value
        assert stable_hash(_semantic_plan_payload(changed)) != original


def test_external_posterior_reuse_validation_fails_closed_on_tamper(
    tmp_path: Path,
) -> None:
    generator_id = "qwen2.5-1.5b-instruct"
    config = fm2_matrix_config(tmp_path, [generator_id])
    config["methods"] = [KCBWDM_SIGNED_V1]
    config["kcbwdm"] = load_matrix_config(
        PROJECT_ROOT
        / "configs/formal/fm2_qwen15_kcbwdm_development_smoke.seed13.matrix.server.yaml"
    )["kcbwdm"]
    posterior_paths = add_external_fm2_posteriors(
        config, tmp_path, generator_id=generator_id
    )
    plan = plan_for(config)
    train = next(
        node for node in nodes(plan, stage="posteriors") if node["split"] == "train_core"
    )
    execution_node = copy.deepcopy(train)
    posterior, manifest = posterior_paths["train_core"]
    execution_node["outputs"] = {
        "posteriors": str(posterior),
        "manifest": str(manifest),
    }
    _validate_external_posterior(execution_node)

    posterior.write_text(
        posterior.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="SHA"):
        _validate_external_posterior(execution_node)


def test_external_fm2_posterior_requires_complete_retrieval_row_coverage(
    tmp_path: Path,
) -> None:
    generator_id = "qwen2.5-1.5b-instruct"
    config = fm2_matrix_config(tmp_path, [generator_id])
    config["methods"] = [KCBWDM_SIGNED_V1]
    config["kcbwdm"] = load_matrix_config(
        PROJECT_ROOT
        / "configs/formal/fm2_qwen15_kcbwdm_development_smoke.seed13.matrix.server.yaml"
    )["kcbwdm"]
    posterior_paths = add_external_fm2_posteriors(
        config, tmp_path, generator_id=generator_id
    )
    _, manifest_path = posterior_paths["train_core"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["completed_rows"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MatrixPlanError, match="row coverage"):
        plan_for(config)


def test_kcbwdm_is_rejected_before_any_held_out_plan_is_built(
    tmp_path: Path,
) -> None:
    config = fm2_matrix_config(tmp_path, ["qwen2.5-1.5b-instruct"])
    config["profile"] = HELD_OUT
    config["evaluation_split"] = "held_out_test"
    config["methods"] = [KCBWDM_SIGNED_V1]
    config["kcbwdm"] = load_matrix_config(
        PROJECT_ROOT
        / "configs/formal/fm2_qwen15_kcbwdm_development_smoke.seed13.matrix.server.yaml"
    )["kcbwdm"]
    with pytest.raises(ValueError, match="not formal-v2 main-table eligible"):
        plan_for(config, held_out=True, held_out_freeze={})


def test_checked_in_fm2_full_development_matrix_builds_expected_plan(
    tmp_path: Path,
) -> None:
    config = load_matrix_config(FM2_FULL_CONFIG)
    local_inputs = fm2_shared_retrieval_inputs(tmp_path)
    for split in ("train_core", "validation"):
        config["retrieval_inputs"][split]["pool"] = local_inputs[split]["pool"]
        config["retrieval_inputs"][split]["manifest"] = local_inputs[split][
            "manifest"
        ]

    plan = plan_for(config)
    assert plan["profile"] == FULL_DEVELOPMENT
    assert plan["limits"] == {"training": None, "evaluation": None}
    assert len(plan["nodes"]) == 62
    assert len(plan["result_index"]) == 20
    assert len(validate_dag(plan["nodes"])) == 62

    retrieval_nodes = {
        node["split"]: node for node in nodes(plan, stage="retrieval_input")
    }
    train = retrieval_nodes["train_core"]
    validation = retrieval_nodes["validation"]
    assert set(train["outputs"]) == {"retrieval"}
    assert set(validation["outputs"]) == {"retrieval"}
    assert train["inputs"]["manifest"] == validation["inputs"]["manifest"]
    assert train["retrieval_binding"]["manifest_sha256"] == (
        validation["retrieval_binding"]["manifest_sha256"]
    )
    assert train["retrieval_binding"]["pool"] != (
        validation["retrieval_binding"]["pool"]
    )
    assert train["retrieval_binding"]["pool_sha256"] != (
        validation["retrieval_binding"]["pool_sha256"]
    )

    assert len(nodes(plan, stage="selection", method="no_evidence")) == 1
    assert len(nodes(plan, stage="selection", method="retrieval_topk")) == 1
    assert len(nodes(plan, stage="selection", method="bge")) == 1
    expected_generators = {
        "qwen2.5-0.5b-instruct",
        "qwen2.5-1.5b-instruct",
        "qwen2.5-7b-instruct",
        "mistral-7b-instruct-v0.3",
    }
    for method in ("infogain", CANONICAL_OURS):
        assert {
            node["generator_id"]
            for node in nodes(plan, stage="training", method=method)
        } == expected_generators
    assert {node["generator_id"] for node in nodes(plan, stage="evaluation")} == (
        expected_generators
    )
    assert not any(node["split"] == "held_out_test" for node in plan["nodes"])
    commands = "\n".join(" ".join(node["command"]) for node in plan["nodes"])
    assert "02_retrieve_bm25.py" not in commands
    assert "pyserini" not in commands.casefold()
    assert "fever_bm25_v1" not in commands


def test_checked_in_fm2_kcbwdm_full_development_matrix_reuses_posteriors(
    tmp_path: Path,
) -> None:
    config = load_matrix_config(FM2_KCBWDM_FULL_CONFIG)
    local_inputs = fm2_shared_retrieval_inputs(tmp_path)
    for split in ("train_core", "validation"):
        config["retrieval_inputs"][split]["pool"] = local_inputs[split]["pool"]
        config["retrieval_inputs"][split]["manifest"] = local_inputs[split][
            "manifest"
        ]
    generator_ids = [generator["generator_id"] for generator in config["generators"]]
    for generator_id in generator_ids:
        add_external_fm2_posteriors(config, tmp_path, generator_id=generator_id)

    plan = plan_for(config)
    assert plan["profile"] == FULL_DEVELOPMENT
    assert plan["limits"] == {"training": None, "evaluation": None}
    assert len(plan["nodes"]) == 31
    assert len(validate_dag(plan["nodes"])) == 31
    assert len(plan["result_index"]) == 4
    assert {row["generator_id"] for row in plan["result_index"]} == set(
        generator_ids
    )
    assert {row["method_id"] for row in plan["result_index"]} == {
        KCBWDM_SIGNED_V1
    }

    posterior_nodes = nodes(plan, stage="posteriors")
    assert len(posterior_nodes) == 8
    assert all(node["command"] == [] for node in posterior_nodes)
    assert all(
        node["execution_policy"] == "validate_external_posterior"
        for node in posterior_nodes
    )
    for generator_id in generator_ids:
        generator_posteriors = [
            node for node in posterior_nodes if node["generator_id"] == generator_id
        ]
        assert {node["split"] for node in generator_posteriors} == {
            "train_core",
            "validation",
        }
        assert all(
            node["posterior_binding"]["generator_id"] == generator_id
            and node["posterior_binding"]["generator_identity"]["generator_id"]
            == generator_id
            and f"/{generator_id}/" in node["outputs"]["posteriors"]
            for node in generator_posteriors
        )
    assert len(
        {
            node["posterior_binding"]["identity_fingerprint"]
            for node in posterior_nodes
        }
    ) == 8
    assert len(
        {
            node["posterior_binding"]["posterior_sha256"]
            for node in posterior_nodes
        }
    ) == 8

    commands = "\n".join(" ".join(node["command"]) for node in plan["nodes"])
    assert "03_compute_label_posteriors.py" not in commands
    teacher_nodes = nodes(plan, stage="teacher", method=KCBWDM_SIGNED_V1)
    assert len(teacher_nodes) == 4
    assert all("--max-rows" not in node["command"] for node in teacher_nodes)
    assert all(
        "/formal_v2_kcbwdm_full_development/" in node["outputs"]["teacher"]
        and "/formal_v2_full_development/" in node["inputs"]["posteriors"]
        for node in teacher_nodes
    )
    assert plan["method_contracts"][KCBWDM_SIGNED_V1]["kernel"] == {
        "base_kernel": "rbf",
        "anchor": "zero_effect",
        "bandwidth_policy": "train_core_within_query_positive_distance_median",
        "ridge_lambda": 0.01,
        "lambda_policy": "absolute",
        "target_normalization": False,
        "set_dependent_centering": False,
    }
    semantic_payload = _semantic_plan_payload(plan)
    assert semantic_payload["method_contracts"][KCBWDM_SIGNED_V1] == (
        plan["method_contracts"][KCBWDM_SIGNED_V1]
    )
    assert "smoke_only_limited_train_core" not in str(semantic_payload)
    assert not any(node["split"] == "held_out_test" for node in plan["nodes"])


def test_checked_in_fm2_kcbwdm_v2a_smoke_builds_isolated_reuse_dag(
    tmp_path: Path,
) -> None:
    config = load_matrix_config(FM2_KCBWDM_V2A_SMOKE_CONFIG)
    local_inputs = fm2_shared_retrieval_inputs(tmp_path)
    for split in ("train_core", "validation"):
        config["retrieval_inputs"][split]["pool"] = local_inputs[split]["pool"]
        config["retrieval_inputs"][split]["manifest"] = local_inputs[split][
            "manifest"
        ]
    add_external_fm2_posteriors(
        config, tmp_path, generator_id="qwen2.5-1.5b-instruct"
    )

    plan = plan_for(config)
    assert plan["profile"] == "development_smoke"
    assert plan["limits"] == {"training": 200, "evaluation": 100}
    assert len(plan["nodes"]) == 10
    assert len(validate_dag(plan["nodes"])) == 10
    assert len(plan["result_index"]) == 1
    assert plan["result_index"][0]["method_id"] == KCBWDM_LINEAR_GATE_V2
    assert plan["result_index"][0]["seed"] == 13

    posterior_nodes = nodes(plan, stage="posteriors")
    assert len(posterior_nodes) == 2
    assert all(node["command"] == [] for node in posterior_nodes)
    assert all(
        node["execution_policy"] == "validate_external_posterior"
        for node in posterior_nodes
    )
    assert all(
        "/formal_v2_full_development/" in node["outputs"]["posteriors"]
        for node in posterior_nodes
    )

    teacher = nodes(plan, stage="teacher", method=KCBWDM_LINEAR_GATE_V2)
    training = nodes(plan, stage="training", method=KCBWDM_LINEAR_GATE_V2)
    selection = nodes(plan, stage="selection", method=KCBWDM_LINEAR_GATE_V2)
    evaluation = nodes(plan, stage="evaluation", method=KCBWDM_LINEAR_GATE_V2)
    assert tuple(map(len, (teacher, training, selection, evaluation))) == (1, 1, 1, 1)
    assert "--method-name" in teacher[0]["command"]
    assert KCBWDM_LINEAR_GATE_V2 in teacher[0]["command"]
    assert teacher[0]["command"][teacher[0]["command"].index("--max-rows") + 1] == (
        "200"
    )
    assert training[0]["command"][
        training[0]["command"].index("--method-name") + 1
    ] == KCBWDM_LINEAR_GATE_V2
    assert selection[0]["command"][
        selection[0]["command"].index("--method-name") + 1
    ] == KCBWDM_LINEAR_GATE_V2
    assert selection[0]["command"][selection[0]["command"].index("--limit") + 1] == (
        "100"
    )
    assert evaluation[0]["command"][
        evaluation[0]["command"].index("--limit") + 1
    ] == "100"
    assert selection[0]["command"][1].endswith(
        "preformal/27_select_signed_v1.py"
    )
    assert "/formal_v2_kcbwdm_linear_gate_v2_development_smoke/" in (
        teacher[0]["outputs"]["teacher"]
    )
    assert "/formal_v2_full_development/" in teacher[0]["inputs"]["posteriors"]

    contract = plan["method_contracts"][KCBWDM_LINEAR_GATE_V2]
    assert contract["contract_version"] == "kcbwdm_linear_gate_v2.v1"
    assert contract["sign_policy"] == "static_linear_target_alignment_gt_0"
    assert _semantic_plan_payload(plan)["method_contracts"][
        KCBWDM_LINEAR_GATE_V2
    ] == contract
    assert not any(node["split"] == "held_out_test" for node in plan["nodes"])
    commands = "\n".join(" ".join(node["command"]) for node in plan["nodes"])
    for forbidden in (
        "03_compute_label_posteriors.py",
        "held_out_test",
        "fever",
        "pyserini",
        "lucene",
    ):
        assert forbidden not in commands.casefold()


def test_checked_in_fm2_kcbwdm_v2a_full_builds_unlimited_reuse_dag(
    tmp_path: Path,
) -> None:
    config = load_matrix_config(FM2_KCBWDM_V2A_FULL_CONFIG)
    local_inputs = fm2_shared_retrieval_inputs(tmp_path)
    for split in ("train_core", "validation"):
        config["retrieval_inputs"][split]["pool"] = local_inputs[split]["pool"]
        config["retrieval_inputs"][split]["manifest"] = local_inputs[split][
            "manifest"
        ]
    add_external_fm2_posteriors(
        config, tmp_path, generator_id="qwen2.5-1.5b-instruct"
    )

    plan = plan_for(config)
    assert plan["profile"] == FULL_DEVELOPMENT
    assert plan["limits"] == {"training": None, "evaluation": None}
    assert plan["registry_version"] == FORMAL_REGISTRY_VERSION == "formal_v2.2"
    assert plan["registry_fingerprint"] == FORMAL_REGISTRY_FINGERPRINT
    assert plan["held_out"] is False
    assert len(plan["nodes"]) == 10
    assert len(validate_dag(plan["nodes"])) == 10
    assert len(plan["result_index"]) == 1
    assert plan["result_index"][0]["method_id"] == KCBWDM_LINEAR_GATE_V2
    assert plan["result_index"][0]["generator_id"] == "qwen2.5-1.5b-instruct"
    assert plan["result_index"][0]["seed"] == 13

    posterior_nodes = nodes(plan, stage="posteriors")
    assert len(posterior_nodes) == 2
    assert len(plan["posterior_reuse_bindings"]) == 2
    assert {node["split"] for node in posterior_nodes} == {
        "train_core",
        "validation",
    }
    assert all(node["command"] == [] for node in posterior_nodes)
    assert all(
        node["execution_policy"] == "validate_external_posterior"
        for node in posterior_nodes
    )
    assert all(
        {
            "manifest_sha256",
            "manifest_fingerprint",
            "identity_fingerprint",
            "posterior_sha256",
            "dataset_identity",
            "generator_identity",
            "retrieval_protocol_identity",
            "split",
        }
        <= set(node["posterior_binding"])
        for node in posterior_nodes
    )

    teacher = nodes(plan, stage="teacher", method=KCBWDM_LINEAR_GATE_V2)
    training = nodes(plan, stage="training", method=KCBWDM_LINEAR_GATE_V2)
    selection = nodes(plan, stage="selection", method=KCBWDM_LINEAR_GATE_V2)
    evaluation = nodes(plan, stage="evaluation", method=KCBWDM_LINEAR_GATE_V2)
    assert tuple(map(len, (teacher, training, selection, evaluation))) == (1, 1, 1, 1)
    assert "--max-rows" not in teacher[0]["command"]
    assert "--limit" not in selection[0]["command"]
    assert "--limit" not in evaluation[0]["command"]
    assert "/formal_v2_kcbwdm_linear_gate_v2_full_development/" in (
        teacher[0]["outputs"]["teacher"]
    )
    assert "/formal_v2_full_development/" in teacher[0]["inputs"]["posteriors"]

    contract = plan["method_contracts"][KCBWDM_LINEAR_GATE_V2]
    assert contract == load_matrix_config(FM2_KCBWDM_V2A_SMOKE_CONFIG)[
        "kcbwdm_linear_gate_v2"
    ]
    commands = "\n".join(" ".join(node["command"]) for node in plan["nodes"])
    assert "03_compute_label_posteriors.py" not in commands
    assert "held_out_test" not in commands


def test_checked_in_fm2_kcbwdm_v2a_remaining_three_full_builds_reuse_dag(
    tmp_path: Path,
) -> None:
    config = load_matrix_config(FM2_KCBWDM_V2A_REMAINING3_FULL_CONFIG)
    local_inputs = fm2_shared_retrieval_inputs(tmp_path)
    for split in ("train_core", "validation"):
        config["retrieval_inputs"][split]["pool"] = local_inputs[split]["pool"]
        config["retrieval_inputs"][split]["manifest"] = local_inputs[split][
            "manifest"
        ]
    generator_ids = [
        generator["generator_id"] for generator in config["generators"]
    ]
    for generator_id in generator_ids:
        add_external_fm2_posteriors(config, tmp_path, generator_id=generator_id)

    plan = plan_for(config)
    assert plan["profile"] == FULL_DEVELOPMENT
    assert plan["limits"] == {"training": None, "evaluation": None}
    assert plan["held_out"] is False
    assert len(plan["nodes"]) == 24
    assert len(validate_dag(plan["nodes"])) == 24
    assert len(plan["result_index"]) == 3
    assert {row["generator_id"] for row in plan["result_index"]} == set(
        generator_ids
    )
    assert all(
        row["method_id"] == KCBWDM_LINEAR_GATE_V2 and row["seed"] == 13
        for row in plan["result_index"]
    )

    posterior_nodes = nodes(plan, stage="posteriors")
    posterior_compute_nodes = [node for node in posterior_nodes if node["command"]]
    assert len(posterior_nodes) == 6
    assert len(posterior_compute_nodes) == 0
    assert len(plan["posterior_reuse_bindings"]) == 6
    assert all(node["command"] == [] for node in posterior_nodes)
    assert all(
        node["execution_policy"] == "validate_external_posterior"
        for node in posterior_nodes
    )
    assert len(
        {
            (node["generator_id"], node["split"])
            for node in posterior_nodes
        }
    ) == 6

    teacher = nodes(plan, stage="teacher", method=KCBWDM_LINEAR_GATE_V2)
    training = nodes(plan, stage="training", method=KCBWDM_LINEAR_GATE_V2)
    selection = nodes(plan, stage="selection", method=KCBWDM_LINEAR_GATE_V2)
    evaluation = nodes(plan, stage="evaluation", method=KCBWDM_LINEAR_GATE_V2)
    assert tuple(map(len, (teacher, training, selection, evaluation))) == (
        3,
        3,
        3,
        3,
    )
    assert all("--max-rows" not in node["command"] for node in teacher)
    assert all("--limit" not in node["command"] for node in selection)
    assert all("--limit" not in node["command"] for node in evaluation)
    assert {node["generator_id"] for node in teacher} == set(generator_ids)
    assert all(
        "/formal_v2_kcbwdm_linear_gate_v2_full_development/"
        in node["outputs"]["teacher"]
        and "/formal_v2_full_development/" in node["inputs"]["posteriors"]
        for node in teacher
    )

    contract = plan["method_contracts"][KCBWDM_LINEAR_GATE_V2]
    assert contract == load_matrix_config(FM2_KCBWDM_V2A_FULL_CONFIG)[
        "kcbwdm_linear_gate_v2"
    ]
    commands = "\n".join(" ".join(node["command"]) for node in plan["nodes"])
    for forbidden in (
        "03_compute_label_posteriors.py",
        "held_out_test",
        "fever",
        "pyserini",
        "lucene",
    ):
        assert forbidden not in commands.casefold()


def test_normalized_rho_endpoint_matrix_builds_four_isolated_reuse_rows(
    tmp_path: Path,
) -> None:
    config = load_matrix_config(FM2_KCBWDM_NORMALIZED_RHO_ENDPOINTS_CONFIG)
    local_inputs = fm2_shared_retrieval_inputs(tmp_path)
    for split in ("train_core", "validation"):
        config["retrieval_inputs"][split]["pool"] = local_inputs[split]["pool"]
        config["retrieval_inputs"][split]["manifest"] = local_inputs[split][
            "manifest"
        ]
    generator_ids = [item["generator_id"] for item in config["generators"]]
    for generator_id in generator_ids:
        add_external_fm2_posteriors(config, tmp_path, generator_id=generator_id)

    runtime = "/root/miniconda3/envs/rag-cbwdm-mistral/bin/python"
    plan = plan_for(config, server_python=runtime)
    assert plan["profile"] == FULL_DEVELOPMENT
    assert plan["limits"] == {"training": None, "evaluation": None}
    assert plan["held_out"] is False
    assert len(plan["nodes"]) == 25
    assert len(validate_dag(plan["nodes"])) == 25
    assert len(plan["result_index"]) == 4
    assert {row["generator_id"] for row in plan["result_index"]} == set(
        generator_ids
    )
    assert {row["rho"] for row in plan["result_index"]} == {0.0, 1.0}
    assert {row["method_variant_id"] for row in plan["result_index"]} == {
        "rho_0",
        "rho_1",
    }
    assert all(
        row["method_id"] == KCBWDM_NORMALIZED_RHO_V1 and row["seed"] == 13
        for row in plan["result_index"]
    )
    assert plan["method_variants"] == [
        {
            "method_id": KCBWDM_NORMALIZED_RHO_V1,
            "variant_id": "rho_0",
            "parameters": {"rho": 0.0},
            "explicit": True,
        },
        {
            "method_id": KCBWDM_NORMALIZED_RHO_V1,
            "variant_id": "rho_1",
            "parameters": {"rho": 1.0},
            "explicit": True,
        },
    ]

    posterior = nodes(plan, stage="posteriors")
    assert len(posterior) == 4
    assert len(plan["posterior_reuse_bindings"]) == 4
    assert all(node["command"] == [] for node in posterior)
    assert all(
        node["execution_policy"] == "validate_external_posterior"
        for node in posterior
    )

    teacher = nodes(plan, stage="teacher", method=KCBWDM_NORMALIZED_RHO_V1)
    training = nodes(plan, stage="training", method=KCBWDM_NORMALIZED_RHO_V1)
    selection = nodes(plan, stage="selection", method=KCBWDM_NORMALIZED_RHO_V1)
    evaluation = nodes(plan, stage="evaluation", method=KCBWDM_NORMALIZED_RHO_V1)
    assert tuple(map(len, (teacher, training, selection, evaluation))) == (4, 4, 4, 4)
    for group in (teacher, training, selection, evaluation):
        assert {node["method_parameters"]["rho"] for node in group} == {0.0, 1.0}
        assert {node["method_variant_id"] for node in group} == {"rho_0", "rho_1"}
    assert all(
        node["command"][0] == runtime
        for node in teacher + training + selection + evaluation
    )
    assert all("--max-rows" not in node["command"] for node in teacher)
    assert all("--limit" not in node["command"] for node in selection)
    assert all("--limit" not in node["command"] for node in evaluation)
    assert all(
        f"/{KCBWDM_NORMALIZED_RHO_V1}/{node['method_variant_id']}/"
        in node["outputs"]["teacher"]
        for node in teacher
    )
    for node in teacher + training + selection:
        assert node["command"][node["command"].index("--rho") + 1] in {"0.0", "1.0"}

    commands = "\n".join(" ".join(node["command"]) for node in plan["nodes"])
    for forbidden in (
        "03_compute_label_posteriors.py",
        "held_out_test",
        "fever",
        "pyserini",
        "lucene",
        "--max-rows",
        "--limit",
    ):
        assert forbidden not in commands.casefold()


@pytest.mark.parametrize("rho", [-0.1, 1.1])
def test_normalized_rho_matrix_rejects_out_of_range_variant(rho: float) -> None:
    config = load_matrix_config(FM2_KCBWDM_NORMALIZED_RHO_ENDPOINTS_CONFIG)
    config["method_variants"][0]["parameters"]["rho"] = rho
    with pytest.raises(MatrixPlanError, match=r"rho must be in \[0, 1\]"):
        plan_for(config)


def test_result_index_protocol_identity_distinguishes_fever_and_fm2(
    smoke_plan: dict, tmp_path: Path
) -> None:
    fm2 = plan_for(fm2_matrix_config(tmp_path, ["qwen2.5-1.5b-instruct"]))
    fever_row = smoke_plan["result_index"][0]
    fm2_row = fm2["result_index"][0]
    required = {
        "dataset_id",
        "generator_id",
        "method",
        "seed",
        "split",
        "retrieval_protocol_id",
        "retrieval_protocol_fingerprint",
    }
    assert required <= set(fever_row)
    assert required <= set(fm2_row)
    assert fever_row["retrieval_protocol_id"] != fm2_row["retrieval_protocol_id"]
    assert (
        fever_row["retrieval_protocol_fingerprint"]
        != fm2_row["retrieval_protocol_fingerprint"]
    )


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

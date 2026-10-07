from __future__ import annotations

from pathlib import Path

import pytest

from scripts.create_generator_manifest import build_manifest
from src.datasets.fm2 import FM2_EXPECTED_ROWS
from src.formal_matrix import MATRIX_CONFIG_SCHEMA_VERSION, load_matrix_config
from src.formal_registry import MAIN_TABLE_METHODS
from src.io_utils import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "fm2_official_closed_page_v1"
BGE_DIRECTORY_SHA256 = (
    "b01f9eac1483006e54a902eb0d272f8738e94a98f9f36e4a2b2e3b3eb8c335a2"
)
INPUT_ROOT = (
    "/root/experiments/rag_cbwdm/formal_v2_external_inputs/"
    "fm2_official_closed_page_v1/prepared_v2"
)
MANIFEST = f"{INPUT_ROOT}/fm2_prepare.manifest.json"
TRAIN_POOL = f"{INPUT_ROOT}/fm2_train_official_pool.jsonl"
VALIDATION_POOL = f"{INPUT_ROOT}/fm2_dev_official_pool.jsonl"
FULL_MATRIX = (
    PROJECT_ROOT / "configs/formal/fm2_full_development.seed13.matrix.server.yaml"
)
KCBWDM_SMOKE_MATRIX = (
    PROJECT_ROOT
    / "configs/formal/fm2_qwen15_kcbwdm_development_smoke.seed13.matrix.server.yaml"
)
KCBWDM_FULL_MATRIX = (
    PROJECT_ROOT
    / "configs/formal/fm2_kcbwdm_full_development.seed13.matrix.server.yaml"
)
KCBWDM_V2A_SMOKE_MATRIX = (
    PROJECT_ROOT
    / "configs/formal/"
    "fm2_qwen15_kcbwdm_linear_gate_v2_development_smoke.seed13.matrix.server.yaml"
)
KCBWDM_V2A_FULL_MATRIX = (
    PROJECT_ROOT
    / "configs/formal/"
    "fm2_qwen15_kcbwdm_linear_gate_v2_full_development.seed13.matrix.server.yaml"
)
KCBWDM_V2A_REMAINING3_FULL_MATRIX = (
    PROJECT_ROOT
    / "configs/formal/"
    "fm2_remaining3_kcbwdm_linear_gate_v2_full_development.seed13.matrix.server.yaml"
)
KCBWDM_NORMALIZED_RHO_ENDPOINTS_MATRIX = (
    PROJECT_ROOT
    / "configs/formal/"
    "fm2_qwen15_qwen7_kcbwdm_normalized_rho_endpoints_development.seed13.matrix.server.yaml"
)
KCBWDM_NORMALIZED_RHO_REMAINING2_ENDPOINTS_MATRIX = (
    PROJECT_ROOT
    / "configs/formal/"
    "fm2_qwen05_mistral7_kcbwdm_normalized_rho_endpoints_development.seed13.matrix.server.yaml"
)

FULL_CONFIGS = {
    "fm2_qwen05_full_development.server.yaml": {
        "generator_id": "qwen2.5-0.5b-instruct",
        "model_family": "qwen2.5",
        "model_name": "/root/models/Qwen2.5-0.5B-Instruct",
        "posterior_batch_size": 64,
    },
    "fm2_qwen15_full_development.server.yaml": {
        "generator_id": "qwen2.5-1.5b-instruct",
        "model_family": "qwen2.5",
        "model_name": "/root/models/Qwen2.5-1.5B-Instruct",
        "posterior_batch_size": 16,
    },
    "fm2_qwen7_full_development.server.yaml": {
        "generator_id": "qwen2.5-7b-instruct",
        "model_family": "qwen2.5",
        "model_name": "/root/models/Qwen2.5-7B-Instruct",
        "posterior_batch_size": 8,
    },
    "fm2_mistral7_full_development.server.yaml": {
        "generator_id": "mistral-7b-instruct-v0.3",
        "model_family": "mistral",
        "model_name": "/root/models/Mistral-7B-Instruct-v0.3",
        "posterior_batch_size": 8,
    },
}


def assert_frozen_methods(config: dict) -> None:
    assert config["baselines"]["common"]["top_m"] == 4
    assert config["baselines"]["naive"]["min_docs"] == 0

    bge = config["baselines"]["bge"]
    assert bge["model_id"] == "BAAI/bge-reranker-large"
    assert bge["model_name"] == "/root/models/bge-reranker-large"
    assert bge["revision"] is None
    assert bge["sha256"] == BGE_DIRECTORY_SHA256
    assert bge["top_m"] == 4
    assert bge["min_docs"] == 4
    assert bge["local_files_only"] is True

    infogain = config["baselines"]["infogain_fever"]
    assert infogain["positive_quantile"] == 0.75
    assert infogain["negative_quantile"] == 0.25
    assert infogain["beta"] == 0.75
    assert infogain["top_m"] == 4
    assert infogain["min_docs"] == 2
    assert infogain["rank_loss_implementation"] == "vectorized"
    assert infogain["epochs"] == 3

    ours = config["selector"]
    assert ours["method"] == "rag_cbwdm_signed_v1"
    assert ours["top_m"] == 4
    assert ours["min_docs"] == 0
    assert ours["score_threshold"] == 0.0
    assert ours["runtime_implementation"] == "block_v1"
    assert ours["forward_batch_size"] == 32


@pytest.mark.parametrize(("filename", "expected"), FULL_CONFIGS.items())
def test_fm2_full_development_server_configs_are_loadable_and_frozen(
    filename: str, expected: dict
) -> None:
    path = PROJECT_ROOT / "configs" / filename
    config = load_yaml(path)

    assert config["dataset"] == "fm2"
    assert config["dataset_id"] == DATASET_ID
    assert config["dataset_family"] == "fm2"
    assert config["retrieval_protocol_id"] == DATASET_ID
    assert config["profile"] == "full_development"
    assert config["profile_limits"] == {
        "train_core": None,
        "validation": None,
        "seeds": [13],
    }
    assert config["generator"] == {
        **config["generator"],
        **expected,
    }
    assert config["task"]["labels"] == ["SUPPORTS", "REFUTES"]
    assert config["task"]["verbalizers"]["SUPPORTS"][0] == "A"
    assert config["task"]["verbalizers"]["REFUTES"][0] == "B"
    assert_frozen_methods(config)

    manifest = build_manifest(
        config_path=path,
        generator_id=expected["generator_id"],
        dataset_id=DATASET_ID,
        model_family=expected["model_family"],
    )
    assert manifest["dataset_identity"]["dataset_id"] == DATASET_ID
    assert manifest["dataset_identity"]["dataset_family"] == "fm2"
    assert manifest["generator_identity"]["generator_id"] == expected["generator_id"]
    assert manifest["generator_identity"]["model_family"] == expected["model_family"]

    lowered = path.read_text(encoding="utf-8").casefold()
    assert "held_out_test" not in lowered
    assert "fever_bm25_v1" not in lowered
    assert "pyserini" not in lowered


def test_fm2_qwen15_smoke_server_config_uses_supported_limits() -> None:
    path = PROJECT_ROOT / "configs/fm2_qwen15_development_smoke.server.yaml"
    config = load_yaml(path)
    assert config["dataset"] == "fm2"
    assert config["dataset_id"] == DATASET_ID
    assert config["profile"] == "development_smoke"
    assert config["profile_limits"] == {
        "train_core": 200,
        "validation": 100,
        "seeds": [13],
    }
    assert config["generator"]["posterior_batch_size"] == 16
    assert_frozen_methods(config)


def test_fm2_qwen15_development_smoke_matrix_is_static_server_ready() -> None:
    path = (
        PROJECT_ROOT
        / "configs/formal/fm2_qwen15_development_smoke.seed13.matrix.server.yaml"
    )
    matrix = load_matrix_config(path)

    assert matrix["schema_version"] == MATRIX_CONFIG_SCHEMA_VERSION
    assert matrix["profile"] == "development_smoke"
    assert matrix["dataset_id"] == DATASET_ID
    assert matrix["retrieval_protocol_id"] == DATASET_ID
    assert matrix["dataset_config"] == (
        "configs/fm2_qwen15_development_smoke.server.yaml"
    )
    assert matrix["training_split"] == "train_core"
    assert matrix["evaluation_split"] == "validation"
    assert set(matrix["retrieval_inputs"]) == {"train_core", "validation"}
    assert matrix["retrieval_inputs"]["train_core"] == {
        "pool": TRAIN_POOL,
        "manifest": MANIFEST,
        "server_pool": TRAIN_POOL,
        "server_manifest": MANIFEST,
    }
    assert matrix["retrieval_inputs"]["validation"] == {
        "pool": VALIDATION_POOL,
        "manifest": MANIFEST,
        "server_pool": VALIDATION_POOL,
        "server_manifest": MANIFEST,
    }
    assert matrix["generators"] == [
        {
            "generator_id": "qwen2.5-1.5b-instruct",
            "model_family": "qwen2.5",
            "config": "configs/fm2_qwen15_full_development.server.yaml",
        }
    ]
    assert tuple(matrix["methods"]) == MAIN_TABLE_METHODS
    assert matrix["learned_seeds"] == [13]
    assert matrix["bge"] == {
        "model_id": "BAAI/bge-reranker-large",
        "model_name_or_path": "/root/models/bge-reranker-large",
        "revision": None,
        "sha256": BGE_DIRECTORY_SHA256,
        "development_only": True,
        "local_files_only": True,
    }

    lowered = path.read_text(encoding="utf-8").casefold()
    for forbidden in (
        "held_out_test",
        "fever_bm25_v1",
        "scripts/02_retrieve_bm25.py",
        "pyserini",
    ):
        assert forbidden not in lowered


def test_fm2_four_generator_full_development_matrix_is_frozen() -> None:
    matrix = load_matrix_config(FULL_MATRIX)

    assert matrix["schema_version"] == MATRIX_CONFIG_SCHEMA_VERSION
    assert matrix["profile"] == "full_development"
    assert matrix["dataset_id"] == DATASET_ID
    assert matrix["retrieval_protocol_id"] == DATASET_ID
    assert matrix["dataset_config"] == (
        "configs/fm2_qwen15_full_development.server.yaml"
    )
    assert matrix["training_split"] == "train_core"
    assert matrix["evaluation_split"] == "validation"
    assert matrix["retrieval_inputs"] == {
        "train_core": {
            "pool": TRAIN_POOL,
            "manifest": MANIFEST,
            "server_pool": TRAIN_POOL,
            "server_manifest": MANIFEST,
        },
        "validation": {
            "pool": VALIDATION_POOL,
            "manifest": MANIFEST,
            "server_pool": VALIDATION_POOL,
            "server_manifest": MANIFEST,
        },
    }
    assert matrix["generators"] == [
        {
            "generator_id": "qwen2.5-0.5b-instruct",
            "model_family": "qwen2.5",
            "config": "configs/fm2_qwen05_full_development.server.yaml",
        },
        {
            "generator_id": "qwen2.5-1.5b-instruct",
            "model_family": "qwen2.5",
            "config": "configs/fm2_qwen15_full_development.server.yaml",
        },
        {
            "generator_id": "qwen2.5-7b-instruct",
            "model_family": "qwen2.5",
            "config": "configs/fm2_qwen7_full_development.server.yaml",
        },
        {
            "generator_id": "mistral-7b-instruct-v0.3",
            "model_family": "mistral",
            "config": "configs/fm2_mistral7_full_development.server.yaml",
        },
    ]
    assert tuple(matrix["methods"]) == MAIN_TABLE_METHODS
    assert matrix["learned_seeds"] == [13]
    assert matrix["bge"] == {
        "model_id": "BAAI/bge-reranker-large",
        "model_name_or_path": "/root/models/bge-reranker-large",
        "revision": None,
        "sha256": BGE_DIRECTORY_SHA256,
        "development_only": True,
        "local_files_only": True,
    }

    dataset_config = load_yaml(PROJECT_ROOT / matrix["dataset_config"])
    assert dataset_config["profile_limits"] == {
        "train_core": None,
        "validation": None,
        "seeds": [13],
    }
    assert_frozen_methods(dataset_config)
    expected_batches = {
        values["generator_id"]: values["posterior_batch_size"]
        for values in FULL_CONFIGS.values()
    }
    for generator in matrix["generators"]:
        config = load_yaml(PROJECT_ROOT / generator["config"])
        assert config["generator"]["generator_id"] == generator["generator_id"]
        assert config["generator"]["model_family"] == generator["model_family"]
        assert config["generator"]["posterior_batch_size"] == expected_batches[
            generator["generator_id"]
        ]
        assert_frozen_methods(config)

    lowered = FULL_MATRIX.read_text(encoding="utf-8").casefold()
    for forbidden in (
        "held_out_test",
        "official_test",
        "/test",
        "fever_bm25_v1",
        "scripts/02_retrieve_bm25.py",
        "bm25",
        "pyserini",
        "lucene",
    ):
        assert forbidden not in lowered


def test_only_development_fm2_matrices_are_present() -> None:
    fm2_matrices = sorted(
        path.name for path in (PROJECT_ROOT / "configs/formal").glob("fm2_*.yaml")
    )
    assert fm2_matrices == [
        "fm2_full_development.seed13.matrix.server.yaml",
        "fm2_kcbwdm_full_development.seed13.matrix.server.yaml",
        "fm2_qwen05_mistral7_kcbwdm_normalized_rho_endpoints_development.seed13.matrix.server.yaml",
        "fm2_qwen15_development_smoke.seed13.matrix.server.yaml",
        "fm2_qwen15_kcbwdm_development_smoke.seed13.matrix.server.yaml",
        "fm2_qwen15_kcbwdm_linear_gate_v2_development_smoke.seed13.matrix.server.yaml",
        "fm2_qwen15_kcbwdm_linear_gate_v2_full_development.seed13.matrix.server.yaml",
        "fm2_qwen15_qwen7_kcbwdm_normalized_rho_endpoints_development.seed13.matrix.server.yaml",
        "fm2_remaining3_kcbwdm_linear_gate_v2_full_development.seed13.matrix.server.yaml",
    ]


def test_fm2_kcbwdm_smoke_is_isolated_and_reuses_full_posteriors() -> None:
    matrix = load_matrix_config(KCBWDM_SMOKE_MATRIX)
    assert matrix["profile"] == "development_smoke"
    assert matrix["dataset_id"] == DATASET_ID
    assert matrix["dataset_config"] == "configs/fm2_qwen15_development_smoke.server.yaml"
    assert matrix["methods"] == ["kcbwdm_signed_v1"]
    assert matrix["learned_seeds"] == [13]
    assert matrix["kcbwdm"]["kernel"] == {
        "base_kernel": "rbf",
        "anchor": "zero_effect",
        "bandwidth_policy": "train_core_within_query_positive_distance_median",
        "ridge_lambda": 0.01,
        "lambda_policy": "absolute",
        "target_normalization": False,
        "set_dependent_centering": False,
    }
    assert matrix["kcbwdm"]["sign_policy"] == (
        "static_anchored_target_alignment_gt_0"
    )
    assert matrix["kcbwdm"]["selector"] == {
        "top_m": 4,
        "min_docs": 0,
        "score_threshold": 0.0,
    }
    posterior_inputs = matrix["posterior_inputs"]["qwen2.5-1.5b-instruct"]
    assert set(posterior_inputs) == {"train_core", "validation"}
    assert all(
        "/formal_v2_full_development/" in item["posteriors"]
        for item in posterior_inputs.values()
    )
    assert all(
        item["posteriors"].endswith("/posteriors.jsonl")
        and item["manifest"].endswith("/posteriors.manifest.json")
        for item in posterior_inputs.values()
    )
    smoke_dataset = load_yaml(PROJECT_ROOT / matrix["dataset_config"])
    assert smoke_dataset["profile_limits"] == {
        "train_core": 200,
        "validation": 100,
        "seeds": [13],
    }
    lowered = KCBWDM_SMOKE_MATRIX.read_text(encoding="utf-8").casefold()
    assert "held_out_test" not in lowered


def test_fm2_kcbwdm_v2a_smoke_is_isolated_and_reuses_full_posteriors() -> None:
    matrix = load_matrix_config(KCBWDM_V2A_SMOKE_MATRIX)
    assert matrix["profile"] == "development_smoke"
    assert matrix["dataset_id"] == DATASET_ID
    assert matrix["dataset_config"] == "configs/fm2_qwen15_development_smoke.server.yaml"
    assert matrix["artifact_root"] == (
        "/root/experiments/rag_cbwdm/"
        "formal_v2_kcbwdm_linear_gate_v2_development_smoke"
    )
    assert matrix["methods"] == ["kcbwdm_linear_gate_v2"]
    assert matrix["learned_seeds"] == [13]
    contract = matrix["kcbwdm_linear_gate_v2"]
    assert contract["contract_version"] == "kcbwdm_linear_gate_v2.v1"
    assert contract["sign_policy"] == "static_linear_target_alignment_gt_0"
    assert contract["kernel"] == {
        "base_kernel": "rbf",
        "anchor": "zero_effect",
        "bandwidth_policy": "train_core_within_query_positive_distance_median",
        "ridge_lambda": 0.01,
        "lambda_policy": "absolute",
        "target_normalization": False,
        "set_dependent_centering": False,
    }
    assert contract["teacher"] == {
        "top_m": 4,
        "stop_threshold": 0.001,
        "alignment_eps": 0.0,
        "b_plus": 0.01,
        "b_minus": 0.001,
        "neutral_sample_policy": "negative",
        "gain_tolerance": 1e-10,
    }
    assert contract["selector"] == {
        "top_m": 4,
        "min_docs": 0,
        "score_threshold": 0.0,
    }
    posterior_inputs = matrix["posterior_inputs"]["qwen2.5-1.5b-instruct"]
    assert set(posterior_inputs) == {"train_core", "validation"}
    assert all(
        "/formal_v2_full_development/" in item["posteriors"]
        for item in posterior_inputs.values()
    )
    smoke_dataset = load_yaml(PROJECT_ROOT / matrix["dataset_config"])
    assert smoke_dataset["profile_limits"] == {
        "train_core": 200,
        "validation": 100,
        "seeds": [13],
    }
    lowered = KCBWDM_V2A_SMOKE_MATRIX.read_text(encoding="utf-8").casefold()
    assert "held_out_test" not in lowered
    assert "fever" not in lowered


def test_fm2_kcbwdm_v2a_qwen15_full_development_matrix_is_frozen() -> None:
    matrix = load_matrix_config(KCBWDM_V2A_FULL_MATRIX)
    generator_ids = [generator["generator_id"] for generator in matrix["generators"]]

    assert matrix["schema_version"] == MATRIX_CONFIG_SCHEMA_VERSION
    assert matrix["profile"] == "full_development"
    assert matrix["dataset_id"] == DATASET_ID
    assert matrix["retrieval_protocol_id"] == DATASET_ID
    assert matrix["dataset_config"] == "configs/fm2_qwen15_full_development.server.yaml"
    assert matrix["training_split"] == "train_core"
    assert matrix["evaluation_split"] == "validation"
    assert matrix["artifact_root"] == (
        "/root/experiments/rag_cbwdm/"
        "formal_v2_kcbwdm_linear_gate_v2_full_development"
    )
    assert generator_ids == ["qwen2.5-1.5b-instruct"]
    assert matrix["methods"] == ["kcbwdm_linear_gate_v2"]
    assert matrix["learned_seeds"] == [13]
    assert matrix["kcbwdm_linear_gate_v2"] == load_matrix_config(
        KCBWDM_V2A_SMOKE_MATRIX
    )["kcbwdm_linear_gate_v2"]

    posterior_inputs = matrix["posterior_inputs"]["qwen2.5-1.5b-instruct"]
    assert set(posterior_inputs) == {"train_core", "validation"}
    for split, binding in posterior_inputs.items():
        expected_root = (
            "/root/experiments/rag_cbwdm/formal_v2_full_development/"
            f"{DATASET_ID}/qwen2.5-1.5b-instruct/posteriors/{split}"
        )
        assert binding["posteriors"] == f"{expected_root}/posteriors.jsonl"
        assert binding["manifest"] == f"{expected_root}/posteriors.manifest.json"
        assert binding["server_posteriors"] == binding["posteriors"]
        assert binding["server_manifest"] == binding["manifest"]

    dataset_config = load_yaml(PROJECT_ROOT / matrix["dataset_config"])
    assert dataset_config["profile_limits"] == {
        "train_core": None,
        "validation": None,
        "seeds": [13],
    }
    assert FM2_EXPECTED_ROWS["train"] == 10419

    lowered = KCBWDM_V2A_FULL_MATRIX.read_text(encoding="utf-8").casefold()
    for forbidden in (
        "held_out_test",
        "official_test",
        "smoke_only_limited_train_core",
        "formal_v2_kcbwdm_linear_gate_v2_development_smoke",
        "fever",
    ):
        assert forbidden not in lowered


def test_fm2_kcbwdm_four_generator_full_development_matrix_is_frozen() -> None:
    matrix = load_matrix_config(KCBWDM_FULL_MATRIX)
    generator_ids = [generator["generator_id"] for generator in matrix["generators"]]

    assert matrix["schema_version"] == MATRIX_CONFIG_SCHEMA_VERSION
    assert matrix["profile"] == "full_development"
    assert matrix["dataset_id"] == DATASET_ID
    assert matrix["retrieval_protocol_id"] == DATASET_ID
    assert matrix["dataset_config"] == "configs/fm2_qwen15_full_development.server.yaml"
    assert matrix["training_split"] == "train_core"
    assert matrix["evaluation_split"] == "validation"
    assert matrix["artifact_root"] == (
        "/root/experiments/rag_cbwdm/formal_v2_kcbwdm_full_development"
    )
    assert matrix["methods"] == ["kcbwdm_signed_v1"]
    assert matrix["learned_seeds"] == [13]
    assert generator_ids == [
        "qwen2.5-0.5b-instruct",
        "qwen2.5-1.5b-instruct",
        "qwen2.5-7b-instruct",
        "mistral-7b-instruct-v0.3",
    ]
    assert set(matrix["posterior_inputs"]) == set(generator_ids)

    posterior_paths = set()
    for generator_id in generator_ids:
        bindings = matrix["posterior_inputs"][generator_id]
        assert set(bindings) == {"train_core", "validation"}
        for split, binding in bindings.items():
            expected_root = (
                "/root/experiments/rag_cbwdm/formal_v2_full_development/"
                f"{DATASET_ID}/{generator_id}/posteriors/{split}"
            )
            assert binding["posteriors"] == f"{expected_root}/posteriors.jsonl"
            assert binding["manifest"] == f"{expected_root}/posteriors.manifest.json"
            assert binding["server_posteriors"] == binding["posteriors"]
            assert binding["server_manifest"] == binding["manifest"]
            posterior_paths.add(binding["posteriors"])
    assert len(posterior_paths) == 8

    dataset_config = load_yaml(PROJECT_ROOT / matrix["dataset_config"])
    assert dataset_config["profile_limits"] == {
        "train_core": None,
        "validation": None,
        "seeds": [13],
    }
    assert FM2_EXPECTED_ROWS["train"] == 10419
    assert matrix["kcbwdm"]["kernel"] == {
        "base_kernel": "rbf",
        "anchor": "zero_effect",
        "bandwidth_policy": "train_core_within_query_positive_distance_median",
        "ridge_lambda": 0.01,
        "lambda_policy": "absolute",
        "target_normalization": False,
        "set_dependent_centering": False,
    }

    lowered = KCBWDM_FULL_MATRIX.read_text(encoding="utf-8").casefold()
    for forbidden in (
        "held_out_test",
        "official_test",
        "smoke_only_limited_train_core",
        "formal_v2_kcbwdm_development_smoke",
    ):
        assert forbidden not in lowered


def test_fm2_kcbwdm_v2a_remaining_three_full_development_matrix_is_frozen() -> None:
    matrix = load_matrix_config(KCBWDM_V2A_REMAINING3_FULL_MATRIX)
    generator_ids = [
        generator["generator_id"] for generator in matrix["generators"]
    ]

    assert matrix["schema_version"] == MATRIX_CONFIG_SCHEMA_VERSION
    assert matrix["profile"] == "full_development"
    assert matrix["dataset_id"] == DATASET_ID
    assert matrix["retrieval_protocol_id"] == DATASET_ID
    assert matrix["dataset_config"] == "configs/fm2_qwen15_full_development.server.yaml"
    assert matrix["training_split"] == "train_core"
    assert matrix["evaluation_split"] == "validation"
    assert matrix["artifact_root"] == (
        "/root/experiments/rag_cbwdm/"
        "formal_v2_kcbwdm_linear_gate_v2_full_development"
    )
    assert generator_ids == [
        "qwen2.5-0.5b-instruct",
        "qwen2.5-7b-instruct",
        "mistral-7b-instruct-v0.3",
    ]
    assert "qwen2.5-1.5b-instruct" not in generator_ids
    assert matrix["methods"] == ["kcbwdm_linear_gate_v2"]
    assert matrix["learned_seeds"] == [13]
    assert matrix["kcbwdm_linear_gate_v2"] == load_matrix_config(
        KCBWDM_V2A_FULL_MATRIX
    )["kcbwdm_linear_gate_v2"]
    assert set(matrix["posterior_inputs"]) == set(generator_ids)

    posterior_paths = set()
    for generator_id in generator_ids:
        bindings = matrix["posterior_inputs"][generator_id]
        assert set(bindings) == {"train_core", "validation"}
        for split, binding in bindings.items():
            expected_root = (
                "/root/experiments/rag_cbwdm/formal_v2_full_development/"
                f"{DATASET_ID}/{generator_id}/posteriors/{split}"
            )
            assert binding["posteriors"] == f"{expected_root}/posteriors.jsonl"
            assert binding["manifest"] == f"{expected_root}/posteriors.manifest.json"
            assert binding["server_posteriors"] == binding["posteriors"]
            assert binding["server_manifest"] == binding["manifest"]
            posterior_paths.add(binding["posteriors"])
    assert len(posterior_paths) == 6

    dataset_config = load_yaml(PROJECT_ROOT / matrix["dataset_config"])
    assert dataset_config["profile_limits"] == {
        "train_core": None,
        "validation": None,
        "seeds": [13],
    }
    assert FM2_EXPECTED_ROWS["train"] == 10419

    lowered = KCBWDM_V2A_REMAINING3_FULL_MATRIX.read_text(
        encoding="utf-8"
    ).casefold()
    for forbidden in (
        "qwen2.5-1.5b-instruct",
        "held_out_test",
        "official_test",
        "smoke_only_limited_train_core",
        "formal_v2_kcbwdm_linear_gate_v2_development_smoke",
        "fever",
    ):
        assert forbidden not in lowered


@pytest.mark.parametrize(
    ("matrix_path", "dataset_config", "expected_generators", "excluded_generators"),
    [
        (
            KCBWDM_NORMALIZED_RHO_ENDPOINTS_MATRIX,
            "configs/fm2_qwen15_full_development.server.yaml",
            ["qwen2.5-1.5b-instruct", "qwen2.5-7b-instruct"],
            ["qwen2.5-0.5b-instruct", "mistral-7b-instruct-v0.3"],
        ),
        (
            KCBWDM_NORMALIZED_RHO_REMAINING2_ENDPOINTS_MATRIX,
            "configs/fm2_qwen05_full_development.server.yaml",
            ["qwen2.5-0.5b-instruct", "mistral-7b-instruct-v0.3"],
            ["qwen2.5-1.5b-instruct", "qwen2.5-7b-instruct"],
        ),
    ],
    ids=["qwen15-qwen7", "qwen05-mistral7"],
)
def test_fm2_normalized_rho_endpoint_matrix_is_frozen(
    matrix_path: Path,
    dataset_config: str,
    expected_generators: list[str],
    excluded_generators: list[str],
) -> None:
    matrix = load_matrix_config(matrix_path)
    generator_ids = [item["generator_id"] for item in matrix["generators"]]

    assert matrix["schema_version"] == MATRIX_CONFIG_SCHEMA_VERSION
    assert matrix["profile"] == "full_development"
    assert matrix["dataset_id"] == DATASET_ID
    assert matrix["retrieval_protocol_id"] == DATASET_ID
    assert matrix["dataset_config"] == dataset_config
    assert matrix["training_split"] == "train_core"
    assert matrix["evaluation_split"] == "validation"
    assert matrix["artifact_root"] == (
        "/root/experiments/rag_cbwdm/"
        "formal_v2_kcbwdm_normalized_rho_development"
    )
    assert generator_ids == expected_generators
    assert all(generator_id not in generator_ids for generator_id in excluded_generators)
    assert matrix["methods"] == ["kcbwdm_normalized_rho_v1"]
    assert matrix["learned_seeds"] == [13]
    assert matrix["method_variants"] == [
        {
            "variant_id": "rho_0",
            "method": "kcbwdm_normalized_rho_v1",
            "parameters": {"rho": 0.0},
        },
        {
            "variant_id": "rho_1",
            "method": "kcbwdm_normalized_rho_v1",
            "parameters": {"rho": 1.0},
        },
    ]
    assert set(matrix["posterior_inputs"]) == set(generator_ids)
    for generator_id in generator_ids:
        for split, binding in matrix["posterior_inputs"][generator_id].items():
            expected_root = (
                "/root/experiments/rag_cbwdm/formal_v2_full_development/"
                f"{DATASET_ID}/{generator_id}/posteriors/{split}"
            )
            assert binding["posteriors"] == f"{expected_root}/posteriors.jsonl"
            assert binding["manifest"] == f"{expected_root}/posteriors.manifest.json"
            assert binding["server_posteriors"] == binding["posteriors"]
            assert binding["server_manifest"] == binding["manifest"]

    contract = matrix["kcbwdm_normalized_rho_v1"]
    assert contract["contract_version"] == "kcbwdm_normalized_rho_v1.v1"
    assert contract["sign_policy"] == "static_linear_target_alignment_gt_0"
    assert contract["kernel"]["normalization_policy"] == (
        "median_positive_diag_ratio_v1"
    )
    assert contract["kernel"]["fit_scope"] == "full_train_core"
    assert contract["kernel"]["ridge_lambda"] == 0.01
    assert contract["teacher"]["stop_threshold"] == 0.001
    assert contract["selector"] == {
        "top_m": 4,
        "min_docs": 0,
        "score_threshold": 0.0,
    }
    dataset_config = load_yaml(PROJECT_ROOT / matrix["dataset_config"])
    assert dataset_config["profile_limits"] == {
        "train_core": None,
        "validation": None,
        "seeds": [13],
    }
    assert FM2_EXPECTED_ROWS["train"] == 10419

    lowered = matrix_path.read_text(encoding="utf-8").casefold()
    for forbidden in (
        "held_out_test",
        "official_test",
        "fever",
        *excluded_generators,
    ):
        assert forbidden not in lowered

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.create_generator_manifest import build_manifest
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
        "fm2_qwen15_development_smoke.seed13.matrix.server.yaml"
    ]

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.experiment_identity import (
    FORMAL_V2_MODE,
    LEGACY_MODE,
    POSTERIOR_MANIFEST_SCHEMA_VERSION,
    build_generator_identity,
    experiment_identity_payload,
    formal_v2_dataset_root,
    formal_v2_generator_root,
    formal_v2_posterior_split_root,
    infer_known_generator_id,
    resolve_dataset_identity,
    resolve_retrieval_protocol_identity,
    validate_posterior_provenance,
)
from src.run_manifest import sha256_file, stable_hash


ROOT = Path(__file__).resolve().parents[1]


def load_script(relative: str):
    path = ROOT / "scripts" / relative
    name = "identity_test_" + relative.replace("/", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def posterior_row(split: str = "train_core") -> dict:
    return {
        "schema_version": "rag_cbwdm_posteriors.v2",
        "id": "q1",
        "query": "A claim.",
        "label": "SUPPORTS",
        "split": split,
        "labels": ["SUPPORTS", "REFUTES"],
        "eta0": [0.5, 0.5],
        "candidates": [
            {
                "doc_id": "d1",
                "rank": 1,
                "title": "Title",
                "text": "Evidence.",
                "eta": [0.8, 0.2],
            },
            {
                "doc_id": "d2",
                "rank": 2,
                "title": "Other",
                "text": "Counter-evidence.",
                "eta": [0.2, 0.8],
            },
        ],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def write_formal_posterior_manifest(
    posterior: Path,
    *,
    retrieval_sha: str = "a" * 64,
    split: str = "train_core",
) -> Path:
    dataset = resolve_dataset_identity("fever2")
    generator = build_generator_identity(
        model_name_or_path="/root/models/Qwen2.5-1.5B-Instruct",
        generator_id="qwen2.5-1.5b-instruct",
        formal_v2=True,
        model_sha256="b" * 64,
        tokenizer_name_or_path="/root/models/Qwen2.5-1.5B-Instruct",
        prompt_template_version="fever_classification.v1",
        prompt_template_hash="p" * 64,
        verbalizer_hash="v" * 64,
    )
    retrieval = resolve_retrieval_protocol_identity(
        dataset_identity=dataset,
        source_artifact_sha256=retrieval_sha,
        retrieval_method="bm25",
        formal_v2=True,
    )
    identity = experiment_identity_payload(dataset, generator, retrieval)
    manifest = {
        "schema_version": POSTERIOR_MANIFEST_SCHEMA_VERSION,
        "stage": "posterior",
        "status": "completed",
        "fingerprint": stable_hash({"test": "posterior"}),
        "identity_mode": "formal_v2",
        "dataset_identity": identity["dataset_identity"],
        "generator_identity": identity["generator_identity"],
        "retrieval_protocol_identity": identity["retrieval_protocol_identity"],
        "identity_fingerprint": stable_hash(identity),
        "provenance": {
            "dataset": "fever2",
            "split": split,
            "generator_model": "/root/models/Qwen2.5-1.5B-Instruct",
            "generator_sha256": "b" * 64,
            "input_sha256": retrieval_sha,
            "config_sha256": "c" * 64,
            "prompt_template_hash": "p" * 64,
            "verbalizers_hash": "v" * 64,
        },
        "git": {"commit": "1" * 40, "branch": "test", "dirty": False},
        "output_path": str(posterior.resolve()),
        "output_sha256": sha256_file(posterior),
    }
    sidecar = posterior.with_suffix(".manifest.json")
    sidecar.write_text(json.dumps(manifest), encoding="utf-8")
    return sidecar


def test_dataset_alias_mapping() -> None:
    assert resolve_dataset_identity("fever2").dataset_id == "fever_binary_v2"
    assert (
        resolve_dataset_identity("fm2").dataset_id
        == "fm2_official_closed_page_v1"
    )


def test_known_generator_id_is_independent_of_local_path() -> None:
    hub = build_generator_identity(
        model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct"
    )
    local = build_generator_identity(
        model_name_or_path="C:\\models\\Qwen2.5-1.5B-Instruct"
    )
    assert hub.generator_id == local.generator_id == "qwen2.5-1.5b-instruct"
    assert hub.model_name_or_path != local.model_name_or_path
    assert infer_known_generator_id("/models/Qwen2.5-7B-Instruct") == (
        "qwen2.5-7b-instruct",
        "qwen2.5",
    )


def test_formal_generator_identity_requires_explicit_consistent_id() -> None:
    with pytest.raises(ValueError, match="explicit generator_id"):
        build_generator_identity(
            model_name_or_path="/models/Qwen2.5-1.5B-Instruct",
            formal_v2=True,
        )
    with pytest.raises(ValueError, match="conflicts with known model"):
        build_generator_identity(
            model_name_or_path="/models/Qwen2.5-1.5B-Instruct",
            generator_id="qwen2.5-7b-instruct",
            formal_v2=True,
        )


def test_formal_posterior_validator_accepts_matching_identity(tmp_path: Path) -> None:
    posterior = tmp_path / "posteriors.jsonl"
    write_jsonl(posterior, [posterior_row()])
    sidecar = write_formal_posterior_manifest(posterior)
    binding = validate_posterior_provenance(
        posterior,
        sidecar,
        mode=FORMAL_V2_MODE,
        expected_dataset_id="fever_binary_v2",
        expected_split="train_core",
        expected_generator_id="qwen2.5-1.5b-instruct",
        expected_retrieval_protocol_id="fever_bm25_v1",
    )
    assert binding["posterior_sha256"] == sha256_file(posterior)
    assert binding["generator_identity"]["generator_id"] == "qwen2.5-1.5b-instruct"


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("expected_dataset_id", "fm2_official_closed_page_v1", "dataset_id mismatch"),
        ("expected_split", "validation", "split mismatch"),
        ("expected_generator_id", "qwen2.5-7b-instruct", "generator_id mismatch"),
        (
            "expected_retrieval_protocol_id",
            "fm2_official_closed_page_v1",
            "retrieval_protocol_id mismatch",
        ),
    ],
)
def test_formal_posterior_validator_rejects_identity_mismatch(
    tmp_path: Path, keyword: str, value: str, message: str
) -> None:
    posterior = tmp_path / "posteriors.jsonl"
    write_jsonl(posterior, [posterior_row()])
    sidecar = write_formal_posterior_manifest(posterior)
    with pytest.raises(ValueError, match=message):
        validate_posterior_provenance(
            posterior, sidecar, mode=FORMAL_V2_MODE, **{keyword: value}
        )


def test_posterior_validator_rejects_tamper_missing_and_incomplete(
    tmp_path: Path,
) -> None:
    posterior = tmp_path / "posteriors.jsonl"
    write_jsonl(posterior, [posterior_row()])
    missing = posterior.with_name("missing.manifest.json")
    with pytest.raises(FileNotFoundError, match="sidecar"):
        validate_posterior_provenance(posterior, missing, mode=FORMAL_V2_MODE)
    sidecar = write_formal_posterior_manifest(posterior)
    posterior.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA"):
        validate_posterior_provenance(posterior, sidecar, mode=FORMAL_V2_MODE)
    write_jsonl(posterior, [posterior_row()])
    write_formal_posterior_manifest(posterior)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["status"] = "running"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="status"):
        validate_posterior_provenance(posterior, sidecar, mode=FORMAL_V2_MODE)


def test_legacy_manifest_allowed_only_in_legacy_mode(tmp_path: Path) -> None:
    posterior = tmp_path / "legacy.jsonl"
    write_jsonl(posterior, [posterior_row("train")])
    sidecar = posterior.with_suffix(".manifest.json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "rag_cbwdm_posterior_manifest.v1",
                "status": "completed",
                "output_path": str(posterior.resolve()),
                "output_sha256": sha256_file(posterior),
                "provenance": {"dataset": "fever2", "split": "train"},
            }
        ),
        encoding="utf-8",
    )
    binding = validate_posterior_provenance(
        posterior, sidecar, mode=LEGACY_MODE, expected_split="train"
    )
    assert binding["generator_identity"] == {}
    with pytest.raises(ValueError, match="schema v2"):
        validate_posterior_provenance(
            posterior, sidecar, mode=FORMAL_V2_MODE
        )


def test_formal_validator_rejects_legacy_compatible_v2_identity(
    tmp_path: Path,
) -> None:
    posterior = tmp_path / "posteriors.jsonl"
    write_jsonl(posterior, [posterior_row()])
    sidecar = write_formal_posterior_manifest(posterior)
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    manifest["identity_mode"] = "legacy_compatible"
    manifest["generator_identity"]["identity_source"] = "inferred_known_model"
    identity = {
        "schema_version": "rag_cbwdm_experiment_identity.v1",
        "dataset_identity": manifest["dataset_identity"],
        "generator_identity": manifest["generator_identity"],
        "retrieval_protocol_identity": manifest["retrieval_protocol_identity"],
    }
    manifest["identity_fingerprint"] = stable_hash(identity)
    sidecar.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="identity_mode"):
        validate_posterior_provenance(
            posterior, sidecar, mode=FORMAL_V2_MODE
        )


def test_formal_v2_path_construction_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "artifacts" / "formal_v2"
    dataset = formal_v2_dataset_root(root, "fever_binary_v2")
    generator = formal_v2_generator_root(
        root, "fever_binary_v2", "qwen2.5-1.5b-instruct"
    )
    split = formal_v2_posterior_split_root(
        root, "fever_binary_v2", "qwen2.5-1.5b-instruct", "train_core"
    )
    assert dataset == root / "fever_binary_v2"
    assert generator == dataset / "qwen2.5-1.5b-instruct"
    assert split == generator / "posteriors" / "train_core"
    assert not root.exists()


def test_posterior_producer_publishes_formal_v2_identity_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retrieval = tmp_path / "retrieval.jsonl"
    write_jsonl(
        retrieval,
        [
            {
                "id": "q1",
                "query": "A claim.",
                "label": "SUPPORTS",
                "split": "train_core",
                "candidates": [
                    {
                        "doc_id": "d1",
                        "rank": 1,
                        "title": "Title",
                        "text": "Evidence.",
                        "score": 1.0,
                    }
                ],
            }
        ],
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        json.dumps(
            {
                "dataset": "fever2",
                "paths": {"processed_dir": str(tmp_path)},
                "task": {
                    "labels": ["SUPPORTS", "REFUTES"],
                    "verbalizers": {"SUPPORTS": ["A"], "REFUTES": ["B"]},
                },
                "retrieval": {"method": "bm25", "top_n": 20},
                "generator": {
                    "model_name": "/models/Qwen2.5-1.5B-Instruct",
                    "dtype": "auto",
                    "device_map": "auto",
                    "trust_remote_code": False,
                    "posterior_batch_size": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "posteriors.jsonl"
    module = load_script("03_compute_label_posteriors.py")

    class FakeScorer:
        def __init__(self, **kwargs):
            self.model = SimpleNamespace(
                config=SimpleNamespace(_commit_hash="resolved-model")
            )
            self.tokenizer = SimpleNamespace(
                init_kwargs={"_commit_hash": "resolved-tokenizer"}
            )

        def score_prompts(self, prompts, batch_size, labels, verbalizers):
            return np.tile(
                np.asarray([[0.75, 0.25]], dtype=np.float32),
                (len(prompts), 1),
            )

    monkeypatch.setattr(module, "LabelLogitScorer", FakeScorer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "03_compute_label_posteriors.py",
            "--config",
            str(config),
            "--split",
            "train_core",
            "--retrieval",
            str(retrieval),
            "--output",
            str(output),
            "--generator-id",
            "qwen2.5-1.5b-instruct",
            "--formal-v2-identity",
        ],
    )
    module.main()
    manifest = json.loads(
        output.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == POSTERIOR_MANIFEST_SCHEMA_VERSION
    assert manifest["identity_mode"] == "formal_v2"
    assert manifest["dataset_identity"]["dataset_id"] == "fever_binary_v2"
    assert manifest["generator_identity"]["generator_id"] == "qwen2.5-1.5b-instruct"
    assert manifest["generator_identity"]["resolved_model_revision"] == "resolved-model"
    assert manifest["retrieval_protocol_identity"]["retrieval_protocol_id"] == "fever_bm25_v1"
    validate_posterior_provenance(output, mode=FORMAL_V2_MODE)


def test_infogain_formal_teacher_binds_posterior_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    posterior = tmp_path / "posteriors.jsonl"
    write_jsonl(posterior, [posterior_row()])
    sidecar = write_formal_posterior_manifest(posterior)
    output = tmp_path / "infogain_teacher.jsonl"
    module = load_script("12a_build_infogain_teacher.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "12a_build_infogain_teacher.py",
            "--posteriors",
            str(posterior),
            "--posterior-manifest",
            str(sidecar),
            "--output",
            str(output),
            "--formal-v2-identity",
        ],
    )
    module.main()
    manifest = json.loads(
        output.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "rag_cbwdm_infogain_teacher_manifest.v2"
    assert manifest["generator_id"] == "qwen2.5-1.5b-instruct"
    assert manifest["posterior_binding"]["posterior_manifest_sha256"] == sha256_file(sidecar)


def test_signed_v1_formal_teacher_binds_posterior_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    posterior = tmp_path / "posteriors.jsonl"
    retrieval = tmp_path / "retrieval.jsonl"
    write_jsonl(posterior, [posterior_row()])
    write_jsonl(retrieval, [{"id": "q1", "candidates": []}])
    sidecar = write_formal_posterior_manifest(
        posterior, retrieval_sha=sha256_file(retrieval)
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        json.dumps(
            {
                "dataset": "fever2",
                "cbwdm": {
                    "ridge_lambda": 0.01,
                    "eps_smooth": 0.001,
                    "L_type": "euclidean_posterior_shift",
                    "target_smoothing": "paper_mixture",
                    "gain_tolerance": 1e-10,
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "signed_teacher"
    module = load_script("preformal/25_materialize_signed_v1_teacher.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "25_materialize_signed_v1_teacher.py",
            "--config",
            str(config),
            "--posteriors",
            str(posterior),
            "--posterior-manifest",
            str(sidecar),
            "--retrieval",
            str(retrieval),
            "--output-dir",
            str(output),
            "--formal-v2-identity",
        ],
    )
    module.main()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "rag_cbwdm_preformal_signed_teacher.v2"
    assert manifest["generator_id"] == "qwen2.5-1.5b-instruct"
    assert manifest["posterior_binding"]["posterior_manifest_sha256"] == sha256_file(sidecar)

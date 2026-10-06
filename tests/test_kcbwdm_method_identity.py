from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

from src.baselines.common import build_selection_contract, publish_selection
from src.diagnostics.kcbwdm_signed_teacher_v1 import kcbwdm_teacher_statistics
from src.preformal.registry import (
    KCBWDM_LINEAR_GATE_V2_CONTRACT,
    KCBWDM_LINEAR_GATE_V2_METHOD,
    KCBWDM_SIGNED_V1_CONTRACT,
    KCBWDM_SIGNED_V1_METHOD,
    SIGNED_V1_CONTRACT,
    SIGNED_V1_CONTRACT_VERSION,
    SIGNED_V1_METHOD,
    assert_frozen_kcbwdm_linear_gate_v2_contract,
    assert_frozen_kcbwdm_contract,
    method_contract_version,
    validate_selection_method_contract,
    validate_training_method_contract,
)
from src.formal_provenance import sha256_path
from src.run_manifest import sha256_file, stable_hash


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(filename: str):
    path = PROJECT_ROOT / "scripts" / "preformal" / filename
    name = "kcbwdm_identity_" + filename.replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kcbwdm_contract_is_independent_and_frozen() -> None:
    assert KCBWDM_SIGNED_V1_CONTRACT is not SIGNED_V1_CONTRACT
    assert KCBWDM_SIGNED_V1_CONTRACT["teacher"] is not SIGNED_V1_CONTRACT["teacher"]
    assert KCBWDM_SIGNED_V1_CONTRACT["selector"] is not SIGNED_V1_CONTRACT["selector"]
    assert KCBWDM_SIGNED_V1_CONTRACT["method"] == KCBWDM_SIGNED_V1_METHOD
    assert KCBWDM_SIGNED_V1_CONTRACT["kernel"] == {
        "base_kernel": "rbf",
        "anchor": "zero_effect",
        "bandwidth_policy": "train_core_within_query_positive_distance_median",
        "ridge_lambda": 0.01,
        "lambda_policy": "absolute",
        "target_normalization": False,
        "set_dependent_centering": False,
    }
    assert KCBWDM_SIGNED_V1_CONTRACT["seed"] == 13

    signed_before = copy.deepcopy(SIGNED_V1_CONTRACT)
    changed = copy.deepcopy(KCBWDM_SIGNED_V1_CONTRACT)
    changed["teacher"]["top_m"] = 99
    assert SIGNED_V1_CONTRACT == signed_before
    with pytest.raises(ValueError, match="parameters are frozen"):
        assert_frozen_kcbwdm_contract({"top_m": 99})


def test_kcbwdm_v2a_contract_is_independent_and_frozen() -> None:
    assert KCBWDM_LINEAR_GATE_V2_CONTRACT is not KCBWDM_SIGNED_V1_CONTRACT
    assert KCBWDM_LINEAR_GATE_V2_CONTRACT["contract_version"] == (
        "kcbwdm_linear_gate_v2.v1"
    )
    assert KCBWDM_LINEAR_GATE_V2_CONTRACT["method"] == (
        KCBWDM_LINEAR_GATE_V2_METHOD
    )
    assert KCBWDM_LINEAR_GATE_V2_CONTRACT["sign_policy"] == (
        "static_linear_target_alignment_gt_0"
    )
    assert KCBWDM_LINEAR_GATE_V2_CONTRACT["kernel"] == (
        KCBWDM_SIGNED_V1_CONTRACT["kernel"]
    )
    assert KCBWDM_LINEAR_GATE_V2_CONTRACT["seed"] == 13

    v1_before = copy.deepcopy(KCBWDM_SIGNED_V1_CONTRACT)
    changed = copy.deepcopy(KCBWDM_LINEAR_GATE_V2_CONTRACT)
    changed["teacher"]["alignment_eps"] = 0.1
    assert KCBWDM_SIGNED_V1_CONTRACT == v1_before
    with pytest.raises(ValueError, match="parameters are frozen"):
        assert_frozen_kcbwdm_linear_gate_v2_contract({"alignment_eps": 0.1})


def test_trainer_contract_and_teacher_identity_are_method_isolated(tmp_path: Path) -> None:
    trainer = load_script("26_train_signed_v1.py")
    paths = {}
    for name in ("config", "teacher", "posteriors", "retrieval"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}", encoding="utf-8")
        paths[name] = path
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"weights")

    default_contract = trainer.training_contract(
        seed=13, model=model, **paths
    )
    explicit_signed = trainer.training_contract(
        seed=13, model=model, method_name=SIGNED_V1_METHOD, **paths
    )
    kernel_contract = trainer.training_contract(
        seed=13, model=model, method_name=KCBWDM_SIGNED_V1_METHOD, **paths
    )
    v2a_contract = trainer.training_contract(
        seed=13, model=model, method_name=KCBWDM_LINEAR_GATE_V2_METHOD, **paths
    )
    assert default_contract == explicit_signed
    assert default_contract["method"] == SIGNED_V1_METHOD
    assert default_contract["method_contract_version"] == SIGNED_V1_CONTRACT_VERSION
    assert kernel_contract["method"] == KCBWDM_SIGNED_V1_METHOD
    assert kernel_contract["method_contract_version"] == "kcbwdm_signed_v1.v1"
    assert v2a_contract["method"] == KCBWDM_LINEAR_GATE_V2_METHOD
    assert v2a_contract["method_contract_version"] == (
        "kcbwdm_linear_gate_v2.v1"
    )
    assert stable_hash(default_contract) != stable_hash(kernel_contract)
    assert stable_hash(kernel_contract) != stable_hash(v2a_contract)

    historical_training_contract = dict(kernel_contract)
    historical_training_contract.pop("method_contract_version")
    historical_training = {
        "schema_version": "rag_kcbwdm_preformal_signed_training.v2",
        "method": KCBWDM_SIGNED_V1_METHOD,
        "contract": historical_training_contract,
        "fingerprint": stable_hash(historical_training_contract),
        "checkpoint_sha256": sha256_path(model),
    }
    trainer.validate_resume_training_manifest(
        historical_training,
        method_name=KCBWDM_SIGNED_V1_METHOD,
        expected_contract=kernel_contract,
        checkpoint=model,
    )

    teacher_sha = sha256_file(paths["teacher"])
    signed_manifest = {
        "schema_version": "rag_cbwdm_preformal_signed_teacher.v2",
        "method": SIGNED_V1_METHOD,
        "teacher_sha256": teacher_sha,
        "contract": {
            "method": SIGNED_V1_METHOD,
            "method_contract_version": SIGNED_V1_CONTRACT_VERSION,
            "split": "train_core",
        },
    }
    signed_manifest["fingerprint"] = stable_hash(signed_manifest["contract"])
    trainer.validate_teacher_method_identity(
        signed_manifest,
        method_name=SIGNED_V1_METHOD,
        teacher_sha256=teacher_sha,
        split="train_core",
    )
    wrong_signed_version = copy.deepcopy(signed_manifest)
    wrong_signed_version["contract"]["method_contract_version"] = "wrong.v1"
    wrong_signed_version["fingerprint"] = stable_hash(
        wrong_signed_version["contract"]
    )
    with pytest.raises(ValueError, match="contract version mismatch"):
        trainer.validate_teacher_method_identity(
            wrong_signed_version,
            method_name=SIGNED_V1_METHOD,
            teacher_sha256=teacher_sha,
            split="train_core",
        )
    historical_contract = {
        "method": SIGNED_V1_METHOD,
        "stage": "teacher_training_only",
        "split": "train_core",
        "parameters": {
            key: SIGNED_V1_CONTRACT["teacher"][key]
            for key in (
                "top_m",
                "stop_threshold",
                "alignment_eps",
                "b_plus",
                "b_minus",
                "neutral_sample_policy",
            )
        },
    }
    historical_signed = {
        "schema_version": "rag_cbwdm_preformal_signed_teacher.v2",
        "method": SIGNED_V1_METHOD,
        "teacher_sha256": teacher_sha,
        "contract": historical_contract,
        "fingerprint": stable_hash(historical_contract),
    }
    assert trainer.validate_teacher_method_identity(
        historical_signed,
        method_name=SIGNED_V1_METHOD,
        teacher_sha256=teacher_sha,
        split="train_core",
    ) == SIGNED_V1_CONTRACT_VERSION
    with pytest.raises(ValueError, match="Teacher method mismatch"):
        trainer.validate_teacher_method_identity(
            signed_manifest,
            method_name=KCBWDM_SIGNED_V1_METHOD,
            teacher_sha256=teacher_sha,
            split="train_core",
        )
    kernel_manifest = {
        **signed_manifest,
        "method": KCBWDM_SIGNED_V1_METHOD,
        "contract": {
            **signed_manifest["contract"],
            "method": KCBWDM_SIGNED_V1_METHOD,
            "method_contract_version": "kcbwdm_signed_v1.v1",
        },
    }
    v2a_manifest = {
        **signed_manifest,
        "method": KCBWDM_LINEAR_GATE_V2_METHOD,
        "contract": {
            **signed_manifest["contract"],
            "method": KCBWDM_LINEAR_GATE_V2_METHOD,
            "method_contract_version": "kcbwdm_linear_gate_v2.v1",
        },
    }
    assert trainer.validate_teacher_method_identity(
        kernel_manifest,
        method_name=KCBWDM_SIGNED_V1_METHOD,
        teacher_sha256=teacher_sha,
        split="train_core",
    ) == "kcbwdm_signed_v1.v1"
    assert trainer.validate_teacher_method_identity(
        v2a_manifest,
        method_name=KCBWDM_LINEAR_GATE_V2_METHOD,
        teacher_sha256=teacher_sha,
        split="train_core",
    ) == "kcbwdm_linear_gate_v2.v1"
    with pytest.raises(ValueError, match="Teacher method mismatch"):
        trainer.validate_teacher_method_identity(
            kernel_manifest,
            method_name=KCBWDM_LINEAR_GATE_V2_METHOD,
            teacher_sha256=teacher_sha,
            split="train_core",
        )
    with pytest.raises(ValueError, match="Teacher method mismatch"):
        trainer.validate_teacher_method_identity(
            v2a_manifest,
            method_name=KCBWDM_SIGNED_V1_METHOD,
            teacher_sha256=teacher_sha,
            split="train_core",
        )
    with pytest.raises(ValueError, match="Teacher method mismatch"):
        trainer.validate_teacher_method_identity(
            signed_manifest,
            method_name=KCBWDM_LINEAR_GATE_V2_METHOD,
            teacher_sha256=teacher_sha,
            split="train_core",
        )


def test_selector_checkpoint_identity_is_bidirectionally_isolated() -> None:
    selector = load_script("27_select_signed_v1.py")
    def manifest(method_name: str) -> dict:
        version = method_contract_version(method_name)
        contract = {"method": method_name, "method_contract_version": version}
        return {
            "schema_version": "test.training.v1",
            "method": method_name,
            "method_contract_version": version,
            "seed": 13,
            "status": "completed",
            "contract": contract,
            "fingerprint": stable_hash(contract),
        }

    signed = manifest(SIGNED_V1_METHOD)
    kernel = manifest(KCBWDM_SIGNED_V1_METHOD)
    v2a = manifest(KCBWDM_LINEAR_GATE_V2_METHOD)
    selector.validate_checkpoint_method_identity(
        signed, method_name=SIGNED_V1_METHOD, seed=13
    )
    selector.validate_checkpoint_method_identity(
        kernel, method_name=KCBWDM_SIGNED_V1_METHOD, seed=13
    )
    selector.validate_checkpoint_method_identity(
        v2a, method_name=KCBWDM_LINEAR_GATE_V2_METHOD, seed=13
    )
    with pytest.raises(ValueError, match="Checkpoint method mismatch"):
        selector.validate_checkpoint_method_identity(
            signed, method_name=KCBWDM_SIGNED_V1_METHOD, seed=13
        )
    with pytest.raises(ValueError, match="Checkpoint method mismatch"):
        selector.validate_checkpoint_method_identity(
            kernel, method_name=SIGNED_V1_METHOD, seed=13
        )
    with pytest.raises(ValueError, match="Checkpoint method mismatch"):
        selector.validate_checkpoint_method_identity(
            kernel, method_name=KCBWDM_LINEAR_GATE_V2_METHOD, seed=13
        )
    with pytest.raises(ValueError, match="Checkpoint method mismatch"):
        selector.validate_checkpoint_method_identity(
            v2a, method_name=KCBWDM_SIGNED_V1_METHOD, seed=13
        )
    wrong_version = copy.deepcopy(v2a)
    wrong_version["method_contract_version"] = "wrong.v1"
    wrong_version["contract"]["method_contract_version"] = "wrong.v1"
    wrong_version["fingerprint"] = stable_hash(wrong_version["contract"])
    with pytest.raises(ValueError, match="contract version mismatch"):
        selector.validate_checkpoint_method_identity(
            wrong_version,
            method_name=KCBWDM_LINEAR_GATE_V2_METHOD,
            seed=13,
        )


def test_method_contract_version_tamper_and_historical_compatibility() -> None:
    current_contract = {
        "method": KCBWDM_LINEAR_GATE_V2_METHOD,
        "method_contract_version": "wrong.v1",
    }
    wrong = {
        "schema_version": "rag_kcbwdm_linear_gate_v2_training.v2",
        "method": KCBWDM_LINEAR_GATE_V2_METHOD,
        "method_contract_version": "wrong.v1",
        "contract": current_contract,
        "fingerprint": stable_hash(current_contract),
    }
    with pytest.raises(ValueError, match="contract version mismatch"):
        validate_training_method_contract(
            wrong, method_name=KCBWDM_LINEAR_GATE_V2_METHOD
        )

    for method_name, schema in (
        (SIGNED_V1_METHOD, "rag_cbwdm_preformal_signed_training.v2"),
        (KCBWDM_SIGNED_V1_METHOD, "rag_kcbwdm_preformal_signed_training.v2"),
    ):
        frozen = (
            SIGNED_V1_CONTRACT
            if method_name == SIGNED_V1_METHOD
            else KCBWDM_SIGNED_V1_CONTRACT
        )
        historical_contract = {
            "method": method_name,
            "stage": "training",
            "seed": 13,
            "epochs": frozen["selector"]["epochs"],
            "lr": frozen["selector"]["lr"],
            "batch_size": frozen["selector"]["batch_size"],
            "beta": frozen["selector"]["beta"],
            "gamma": frozen["selector"]["gamma"],
            "loss_type": frozen["selector"]["loss_type"],
            "b_plus": frozen["teacher"]["b_plus"],
            "b_minus": frozen["teacher"]["b_minus"],
            "neutral_sample_policy": frozen["teacher"][
                "neutral_sample_policy"
            ],
        }
        historical = {
            "schema_version": schema,
            "method": method_name,
            "contract": historical_contract,
            "fingerprint": stable_hash(historical_contract),
        }
        assert validate_training_method_contract(
            historical, method_name=method_name
        ) == method_contract_version(method_name)
        tampered_historical = copy.deepcopy(historical)
        tampered_historical["contract"]["epochs"] += 1
        tampered_historical["fingerprint"] = stable_hash(
            tampered_historical["contract"]
        )
        with pytest.raises(ValueError, match="changed frozen epochs"):
            validate_training_method_contract(
                tampered_historical, method_name=method_name
            )

        selection_contract = {
            "method": method_name,
            "parameters": {
                "seed": 13,
                "top_m": frozen["selector"]["top_m"],
                "min_docs": frozen["selector"]["min_docs"],
                "score_threshold": frozen["selector"]["score_threshold"],
                "uses_gold_at_inference": False,
            },
        }
        selection = {
            "schema_version": "rag_cbwdm_selection_manifest.v2",
            "method": method_name,
            "contract": selection_contract,
            "fingerprint": stable_hash(selection_contract),
        }
        assert validate_selection_method_contract(
            selection, method_name=method_name
        ) == method_contract_version(method_name)

    unknown_contract = {"method": SIGNED_V1_METHOD}
    unknown = {
        "schema_version": "unknown.training.v1",
        "method": SIGNED_V1_METHOD,
        "contract": unknown_contract,
        "fingerprint": stable_hash(unknown_contract),
    }
    with pytest.raises(ValueError, match="recognized contract version"):
        validate_training_method_contract(unknown, method_name=SIGNED_V1_METHOD)


def test_selection_manifest_records_method_contract_version(tmp_path: Path) -> None:
    source = tmp_path / "posteriors.jsonl"
    source.write_text("", encoding="utf-8")
    output = tmp_path / "selection.jsonl"
    version = method_contract_version(KCBWDM_LINEAR_GATE_V2_METHOD)
    contract = build_selection_contract(
        method=KCBWDM_LINEAR_GATE_V2_METHOD,
        method_contract_version=version,
        input_paths={"posteriors": source},
        parameters={"seed": 13},
    )
    publish_selection(output, [], contract=contract, project_root=PROJECT_ROOT)
    manifest = json.loads(
        output.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["method"] == KCBWDM_LINEAR_GATE_V2_METHOD
    assert manifest["method_contract_version"] == version
    assert manifest["contract"]["method_contract_version"] == version


def test_historical_kcbwdm_v1_selection_resume_is_strict_and_reusable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "posteriors.jsonl"
    source.write_text("", encoding="utf-8")
    output = tmp_path / "selection.jsonl"
    legacy_contract = build_selection_contract(
        method=KCBWDM_SIGNED_V1_METHOD,
        input_paths={"posteriors": source},
        parameters={
            "seed": 13,
            "top_m": 4,
            "min_docs": 0,
            "score_threshold": 0.0,
            "uses_gold_at_inference": False,
        },
    )
    publish_selection(output, [], contract=legacy_contract, project_root=PROJECT_ROOT)
    current_contract = {
        **legacy_contract,
        "method_contract_version": method_contract_version(
            KCBWDM_SIGNED_V1_METHOD
        ),
    }
    written, reused = publish_selection(
        output,
        [],
        contract=current_contract,
        project_root=PROJECT_ROOT,
        resume=True,
    )
    assert written == 0
    assert reused is True


def test_training_manifest_source_records_validated_contract_version() -> None:
    trainer = load_script("26_train_signed_v1.py")
    source = inspect.getsource(trainer.main)
    assert '"method_contract_version": validated_contract_version' in source


def test_kcbwdm_statistics_drop_legacy_reference() -> None:
    stats = kcbwdm_teacher_statistics([])
    assert stats["method"] == KCBWDM_SIGNED_V1_METHOD
    assert "old_teacher_ranking_skipped_ratio_reference" not in stats


@pytest.mark.parametrize(
    "method_name",
    [KCBWDM_SIGNED_V1_METHOD, KCBWDM_LINEAR_GATE_V2_METHOD],
)
def test_teacher_manifest_records_complete_smoke_bandwidth_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method_name: str
) -> None:
    materializer = load_script("28_materialize_kcbwdm_signed_v1_teacher.py")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "dataset": "fm2",
                "cbwdm": {
                    "L_type": "euclidean_posterior_shift",
                    "ridge_lambda": 0.01,
                    "eps_smooth": 0.001,
                    "target_smoothing": "paper_mixture",
                    "gain_tolerance": 1e-10,
                },
            }
        ),
        encoding="utf-8",
    )
    row = {
        "id": "q1",
        "query": "claim",
        "label": "SUPPORTS",
        "split": "train_core",
        "labels": ["SUPPORTS", "REFUTES"],
        "eta0": [0.5, 0.5],
        "candidates": [
            {"doc_id": "d1", "rank": 1, "title": "A", "text": "a", "eta": [0.8, 0.2]},
            {"doc_id": "d2", "rank": 2, "title": "B", "text": "b", "eta": [0.6, 0.4]},
        ],
    }
    posterior = tmp_path / "posteriors.jsonl"
    posterior.write_text(json.dumps(row) + "\n", encoding="utf-8")
    retrieval = tmp_path / "retrieval.jsonl"
    retrieval.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output = tmp_path / method_name
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "28_materialize_kcbwdm_signed_v1_teacher.py",
            "--config",
            str(config),
            "--posteriors",
            str(posterior),
            "--retrieval",
            str(retrieval),
            "--output-dir",
            str(output),
            "--kernel",
            "rbf",
            "--max-rows",
            "1",
            "--method-name",
            method_name,
        ],
    )
    materializer.main()

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    bandwidth = manifest["bandwidth_provenance"]
    assert manifest["method"] == method_name
    assert manifest["contract"]["method_contract_version"] == (
        "kcbwdm_linear_gate_v2.v1"
        if method_name == KCBWDM_LINEAR_GATE_V2_METHOD
        else "kcbwdm_signed_v1.v1"
    )
    assert manifest["schema_version"] == (
        "rag_kcbwdm_linear_gate_v2_teacher_manifest.v1"
        if method_name == KCBWDM_LINEAR_GATE_V2_METHOD
        else "rag_kcbwdm_preformal_signed_teacher.v1"
    )
    assert manifest["posterior_sha256"] == sha256_file(posterior)
    assert manifest["fitted_sigma"] == bandwidth["sigma"]
    assert bandwidth == {
        **bandwidth,
        "policy": "train_core_within_query_positive_distance_median",
        "positive_distance_count": 1,
        "query_group_count": 1,
        "nonempty_query_group_count": 1,
        "source_split": "train_core",
        "implementation_version": "kcbwdm_train_core_median_v1",
        "subsampling": "none",
        "input_posterior_row_count": 1,
        "fitted_row_count": 1,
        "source_row_limit": 1,
        "fit_scope": "smoke_only_limited_train_core",
        "reusable_for_full_development": False,
    }
    statistics = json.loads((output / "statistics.json").read_text(encoding="utf-8"))
    assert statistics["method"] == method_name
    assert "old_teacher_ranking_skipped_ratio_reference" not in statistics


def test_v2a_selector_deployment_uses_only_the_learned_cross_encoder() -> None:
    selector = load_script("27_select_signed_v1.py")
    source = inspect.getsource(selector.main)
    assert "select_row_without_gold" in source
    assert "uses_gold_at_inference\": False" in source
    assert "build_local_effects" not in source
    assert "kernel_set_score" not in source
    assert "linear_gate_kernel_signed_greedy" not in source

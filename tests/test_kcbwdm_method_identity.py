from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src.diagnostics.kcbwdm_signed_teacher_v1 import kcbwdm_teacher_statistics
from src.preformal.registry import (
    KCBWDM_SIGNED_V1_CONTRACT,
    KCBWDM_SIGNED_V1_METHOD,
    SIGNED_V1_CONTRACT,
    SIGNED_V1_METHOD,
    assert_frozen_kcbwdm_contract,
)
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
    assert default_contract == explicit_signed
    assert default_contract["method"] == SIGNED_V1_METHOD
    assert kernel_contract["method"] == KCBWDM_SIGNED_V1_METHOD
    assert stable_hash(default_contract) != stable_hash(kernel_contract)

    teacher_sha = sha256_file(paths["teacher"])
    signed_manifest = {
        "method": SIGNED_V1_METHOD,
        "teacher_sha256": teacher_sha,
        "contract": {"split": "train_core"},
    }
    trainer.validate_teacher_method_identity(
        signed_manifest,
        method_name=SIGNED_V1_METHOD,
        teacher_sha256=teacher_sha,
        split="train_core",
    )
    with pytest.raises(ValueError, match="Teacher method mismatch"):
        trainer.validate_teacher_method_identity(
            signed_manifest,
            method_name=KCBWDM_SIGNED_V1_METHOD,
            teacher_sha256=teacher_sha,
            split="train_core",
        )


def test_selector_checkpoint_identity_is_bidirectionally_isolated() -> None:
    selector = load_script("27_select_signed_v1.py")
    signed = {"method": SIGNED_V1_METHOD, "seed": 13, "status": "completed"}
    kernel = {
        "method": KCBWDM_SIGNED_V1_METHOD,
        "seed": 13,
        "status": "completed",
    }
    selector.validate_checkpoint_method_identity(
        signed, method_name=SIGNED_V1_METHOD, seed=13
    )
    selector.validate_checkpoint_method_identity(
        kernel, method_name=KCBWDM_SIGNED_V1_METHOD, seed=13
    )
    with pytest.raises(ValueError, match="Checkpoint method mismatch"):
        selector.validate_checkpoint_method_identity(
            signed, method_name=KCBWDM_SIGNED_V1_METHOD, seed=13
        )
    with pytest.raises(ValueError, match="Checkpoint method mismatch"):
        selector.validate_checkpoint_method_identity(
            kernel, method_name=SIGNED_V1_METHOD, seed=13
        )


def test_kcbwdm_statistics_drop_legacy_reference() -> None:
    stats = kcbwdm_teacher_statistics([])
    assert stats["method"] == KCBWDM_SIGNED_V1_METHOD
    assert "old_teacher_ranking_skipped_ratio_reference" not in stats


def test_teacher_manifest_records_complete_smoke_bandwidth_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    output = tmp_path / "teacher"
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
        ],
    )
    materializer.main()

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    bandwidth = manifest["bandwidth_provenance"]
    assert manifest["method"] == KCBWDM_SIGNED_V1_METHOD
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
    assert "old_teacher_ranking_skipped_ratio_reference" not in statistics

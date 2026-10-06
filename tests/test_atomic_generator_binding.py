from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.artifact_binding import (
    BINDING_SCHEMA_VERSION,
    CROSS_GENERATOR_TRANSFER,
    GENERATOR_DEPENDENCY_CONDITIONED,
    GENERATOR_MANIFEST_SCHEMA_VERSION,
    MATCHED_MAIN,
    build_evaluation_binding,
    complete_training_binding,
    independent_selection_binding,
    validate_formal_teacher_binding,
    validate_formal_training_binding,
    validate_generator_identity_against_config,
    validate_generator_manifest,
    validate_selection_provenance,
)
from src.baselines.common import build_selection_contract, publish_selection
from src.experiment_identity import (
    POSTERIOR_MANIFEST_SCHEMA_VERSION,
    DatasetIdentity,
    GeneratorIdentity,
    RetrievalProtocolIdentity,
    experiment_identity_payload,
    generator_artifact_sha256,
    posterior_binding_contract,
)
from src.formal_provenance import sha256_path
from src.run_manifest import sha256_file, stable_hash
from src.preformal.registry import method_contract_version


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def make_teacher(tmp_path: Path, method: str, generator_id: str = "gen-a"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = tmp_path / f"{method}.config.json"
    write_json(config, {"dataset": "fever2"})
    posterior = tmp_path / f"{method}.posterior.jsonl"
    posterior.write_text("{}\n", encoding="utf-8")
    dataset = DatasetIdentity("fever_binary_v2", "fever", "fever2").to_dict()
    generator = GeneratorIdentity(
        generator_id=generator_id,
        model_family="test",
        model_name_or_path=f"remote/{generator_id}",
        model_revision="rev-a",
        tokenizer_name_or_path=f"remote/{generator_id}",
        tokenizer_revision="rev-a",
        model_sha256=generator_artifact_sha256(f"remote/{generator_id}", "rev-a"),
        tokenizer_sha256=generator_artifact_sha256(
            f"remote/{generator_id}", "rev-a"
        ),
        prompt_template_version="prompt-v1",
        prompt_template_hash="prompt-hash",
        verbalizer_hash="verbalizer-hash",
        dtype="auto",
        device_map="auto",
        identity_source="explicit",
    ).to_dict()
    retrieval = RetrievalProtocolIdentity(
        "fever_bm25_v1", "fever_binary_v2", "retrieval-sha"
    ).to_dict()
    identity = experiment_identity_payload(
        DatasetIdentity(**dataset),
        GeneratorIdentity(**generator),
        RetrievalProtocolIdentity(**retrieval),
    )
    posterior_manifest = tmp_path / f"{method}.posterior.manifest.json"
    posterior_payload = {
        "schema_version": POSTERIOR_MANIFEST_SCHEMA_VERSION,
        "identity_mode": "formal_v2",
        "status": "completed",
        "output_path": str(posterior),
        "output_sha256": sha256_file(posterior),
        "fingerprint": "posterior-fingerprint",
        "identity_fingerprint": stable_hash(identity),
        "dataset_identity": dataset,
        "generator_identity": generator,
        "retrieval_protocol_identity": retrieval,
        "provenance": {
            "dataset": "fever2",
            "split": "train_core",
            "generator_model": generator["model_name_or_path"],
            "generator_sha256": generator["model_sha256"],
            "input_sha256": retrieval["source_artifact_sha256"],
            "config_sha256": sha256_file(config),
            "prompt_template_hash": generator["prompt_template_hash"],
            "verbalizers_hash": generator["verbalizer_hash"],
        },
        "git": {"commit": "test-commit"},
    }
    write_json(posterior_manifest, posterior_payload)
    binding = posterior_binding_contract(
        {
            "manifest_path": str(posterior_manifest),
            "manifest_sha256": sha256_file(posterior_manifest),
            "manifest_fingerprint": posterior_payload["fingerprint"],
            "identity_fingerprint": posterior_payload["identity_fingerprint"],
            "posterior_sha256": sha256_file(posterior),
            "dataset_identity": dataset,
            "generator_identity": generator,
            "retrieval_protocol_identity": retrieval,
            "split": "train_core",
        }
    )
    teacher = tmp_path / f"{method}.teacher.jsonl"
    teacher.write_text("{}\n", encoding="utf-8")
    teacher_manifest = tmp_path / f"{method}.teacher.manifest.json"
    if method == "infogain_fever":
        contract_key = "provenance"
        contract = {"posterior_binding": binding, "teacher_role": "training"}
        teacher_sha_key = "output_sha256"
        method_fields = {"stage": "build_infogain_teacher"}
    else:
        contract_key = "contract"
        contract = {
            "method": method,
            "split": "train_core",
            "posterior_binding": binding,
        }
        teacher_sha_key = "teacher_sha256"
        method_fields = {"method": method}
    payload = {
        "schema_version": "teacher.v2",
        "status": "completed",
        "completed": True,
        "identity_mode": "formal_v2",
        "fingerprint": stable_hash(contract),
        contract_key: contract,
        "posterior_binding": binding,
        teacher_sha_key: sha256_file(teacher),
        "git": {"commit": "test-commit"},
        **method_fields,
    }
    write_json(teacher_manifest, payload)
    return teacher, teacher_manifest, config


def make_training(tmp_path: Path, method: str, generator_id: str = "gen-a"):
    teacher, teacher_manifest, config = make_teacher(tmp_path, method, generator_id)
    teacher_binding = validate_formal_teacher_binding(
        teacher,
        teacher_manifest,
        method=method,
        expected_dataset_id="fever_binary_v2",
        expected_conditioning_generator_id=generator_id,
    )
    binding = complete_training_binding(teacher_binding, seed=13, config_path=config)
    checkpoint = tmp_path / f"{method}.checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    contract = {"method": method, "artifact_binding": binding}
    version = None
    if method == "rag_cbwdm_signed_v1":
        version = method_contract_version(method)
        contract["method_contract_version"] = version
    fingerprint = stable_hash(contract)
    checkpoint_sha = sha256_path(checkpoint)
    manifest = tmp_path / f"{method}.training.json"
    write_json(
        manifest,
        {
            "schema_version": (
                "rag_cbwdm_preformal_signed_training.v2"
                if version is not None
                else "training.v2"
            ),
            "status": "completed",
            "completed": True,
            "identity_mode": "formal_v2",
            "method": method,
            **(
                {"method_contract_version": version}
                if version is not None
                else {}
            ),
            "fingerprint": fingerprint,
            "contract": contract,
            "artifact_binding": binding,
            "dataset_id": binding["dataset_id"],
            "conditioning_generator_id": binding["conditioning_generator_id"],
            "generator_identity_fingerprint": binding[
                "generator_identity_fingerprint"
            ],
            "retrieval_protocol_id": binding["retrieval_protocol_id"],
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_fingerprint": stable_hash(
                {"contract": fingerprint, "checkpoint_sha256": checkpoint_sha}
            ),
            "git": {"commit": "test-commit"},
        },
    )
    return manifest, checkpoint


def make_conditioned_selection(tmp_path: Path, method: str, generator_id: str = "gen-a"):
    training, checkpoint = make_training(tmp_path, method, generator_id)
    binding = validate_formal_training_binding(
        training,
        checkpoint,
        method=method,
        expected_dataset_id="fever_binary_v2",
        expected_conditioning_generator_id=generator_id,
        expected_seed=13,
    )
    selection = tmp_path / f"{method}.selection.jsonl"
    contract = build_selection_contract(
        method=method,
        method_contract_version=(
            method_contract_version(method)
            if method == "rag_cbwdm_signed_v1"
            else None
        ),
        input_paths={"training_manifest": training},
        parameters={"seed": 13},
        artifact_binding=binding,
    )
    publish_selection(selection, [], contract=contract, project_root=Path.cwd())
    return selection, selection.with_suffix(".manifest.json")


def make_generator_manifest(tmp_path: Path, generator_id: str):
    config = tmp_path / f"{generator_id}.generator-config.json"
    generator_config = {
        "model_name": f"remote/{generator_id}",
        "revision": "rev-a",
        "dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": False,
        "max_context_tokens": 128,
    }
    write_json(config, {"generator": generator_config})
    identity = GeneratorIdentity(
        generator_id=generator_id,
        model_family="test",
        model_name_or_path=generator_config["model_name"],
        model_revision="rev-a",
        tokenizer_name_or_path=generator_config["model_name"],
        tokenizer_revision="rev-a",
        model_sha256=generator_artifact_sha256(
            generator_config["model_name"], "rev-a"
        ),
        tokenizer_sha256=generator_artifact_sha256(
            generator_config["model_name"], "rev-a"
        ),
        prompt_template_version="prompt-v1",
        prompt_template_hash="prompt-hash",
        verbalizer_hash="verbalizer-hash",
        dtype="auto",
        device_map="auto",
        trust_remote_code=False,
        max_context_tokens=128,
        identity_source="explicit",
    ).to_dict()
    path = tmp_path / f"{generator_id}.generator-manifest.json"
    write_json(
        path,
        {
            "schema_version": GENERATOR_MANIFEST_SCHEMA_VERSION,
            "status": "completed",
            "dataset_identity": DatasetIdentity(
                "fever_binary_v2", "fever", "fever2"
            ).to_dict(),
            "generator_identity": identity,
            "generator_identity_fingerprint": stable_hash(identity),
            "config_path": str(config),
            "config_sha256": sha256_file(config),
            "git": {"commit": "test-commit"},
        },
    )
    return validate_generator_manifest(path), generator_config


@pytest.mark.parametrize("method", ["infogain_fever", "rag_cbwdm_signed_v1"])
def test_formal_trainers_accept_matching_teacher_and_reject_wrong_generator(
    tmp_path: Path, method: str
) -> None:
    teacher, manifest, _ = make_teacher(tmp_path, method)
    binding = validate_formal_teacher_binding(
        teacher,
        manifest,
        method=method,
        expected_dataset_id="fever_binary_v2",
        expected_conditioning_generator_id="gen-a",
    )
    assert binding["conditioning_generator_id"] == "gen-a"
    with pytest.raises(ValueError, match="conditioning_generator_id"):
        validate_formal_teacher_binding(
            teacher,
            manifest,
            method=method,
            expected_conditioning_generator_id="gen-b",
        )


@pytest.mark.parametrize("method", ["infogain_fever", "rag_cbwdm_signed_v1"])
def test_conditioned_selection_carries_generator_and_rejects_checkpoint_tamper(
    tmp_path: Path, method: str
) -> None:
    training, checkpoint = make_training(tmp_path, method)
    binding = validate_formal_training_binding(training, checkpoint, method=method)
    assert binding["generator_dependency"] == GENERATOR_DEPENDENCY_CONDITIONED
    assert binding["conditioning_generator_id"] == "gen-a"
    (checkpoint / "weights.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Checkpoint SHA"):
        validate_formal_training_binding(training, checkpoint, method=method)


def test_selection_tamper_and_missing_conditioning_generator_fail(tmp_path: Path) -> None:
    selection, manifest = make_conditioned_selection(tmp_path, "infogain_fever")
    selection.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Selection SHA"):
        validate_selection_provenance(selection, manifest)

    selection, manifest = make_conditioned_selection(
        tmp_path / "missing", "rag_cbwdm_signed_v1"
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["artifact_binding"]["conditioning_generator_id"] = None
    payload["contract"]["artifact_binding"]["conditioning_generator_id"] = None
    payload["fingerprint"] = stable_hash(payload["contract"])
    write_json(manifest, payload)
    with pytest.raises(ValueError, match="conditioning_generator_id"):
        validate_selection_provenance(selection, manifest)


def test_matched_transfer_and_independent_evaluation_contracts(tmp_path: Path) -> None:
    selection, manifest = make_conditioned_selection(tmp_path, "infogain_fever")
    selection_binding = validate_selection_provenance(selection, manifest)
    gen_a, _ = make_generator_manifest(tmp_path, "gen-a")
    gen_b, _ = make_generator_manifest(tmp_path, "gen-b")
    matched = build_evaluation_binding(selection_binding, gen_a)
    assert matched["experiment_type"] == MATCHED_MAIN
    assert matched["conditioning_generator_id"] == matched["evaluation_generator_id"]
    with pytest.raises(ValueError, match="matched_main"):
        build_evaluation_binding(selection_binding, gen_b)
    transfer = build_evaluation_binding(
        selection_binding, gen_b, experiment_type=CROSS_GENERATOR_TRANSFER
    )
    assert transfer["conditioning_generator_id"] == "gen-a"
    assert transfer["evaluation_generator_id"] == "gen-b"

    source = tmp_path / "retrieval.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    independent = independent_selection_binding(
        dataset_id="fever_binary_v2",
        retrieval_protocol_id="fever_bm25_v1",
        method="bge",
        source_artifact_sha256=sha256_file(source),
    )
    independent_selection = tmp_path / "independent.jsonl"
    contract = build_selection_contract(
        method="bge",
        input_paths={"retrieval": source},
        parameters={},
        artifact_binding=independent,
    )
    publish_selection(
        independent_selection, [], contract=contract, project_root=Path.cwd()
    )
    validated = validate_selection_provenance(independent_selection)
    evaluated = build_evaluation_binding(validated, gen_b)
    assert evaluated["conditioning_generator_id"] is None
    assert evaluated["evaluation_generator_id"] == "gen-b"


def test_atomic_generator_rejects_incompatible_mix(tmp_path: Path) -> None:
    generator, config = make_generator_manifest(tmp_path, "gen-a")
    validate_generator_identity_against_config(generator["generator_identity"], config)
    mixed = dict(generator["generator_identity"])
    mixed["tokenizer_revision"] = "stale-revision"
    with pytest.raises(ValueError, match="tokenizer_revision"):
        validate_generator_identity_against_config(mixed, config)


def test_legacy_selection_allowed_only_outside_formal_v2(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    selection = tmp_path / "legacy.jsonl"
    contract = build_selection_contract(
        method="legacy", input_paths={"source": source}, parameters={}
    )
    publish_selection(selection, [], contract=contract, project_root=Path.cwd())
    legacy = validate_selection_provenance(selection, formal_v2=False)
    assert legacy["artifact_binding"] is None
    with pytest.raises(ValueError, match="schema v2"):
        validate_selection_provenance(selection, formal_v2=True)


def test_conditioned_binding_schema_constant() -> None:
    assert BINDING_SCHEMA_VERSION == "rag_cbwdm_artifact_binding.v1"

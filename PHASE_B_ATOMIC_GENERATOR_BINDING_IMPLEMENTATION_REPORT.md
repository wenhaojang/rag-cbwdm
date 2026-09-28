# Phase B Atomic Generator Binding Implementation Report

## 1. Summary

Phase B extends the formal-v2 provenance chain from the Phase A posterior/teacher boundary through learned checkpoints, selection sidecars, final evaluation, and metrics. InfoGain and `rag_cbwdm_signed_v1` selections are explicitly generator-conditioned. No Evidence, retrieval Top-k, and BGE selections are explicitly generator-independent.

Formal final evaluation now consumes an atomic generator manifest. A conditioned `matched_main` evaluation fails unless `conditioning_generator_id == evaluation_generator_id`; an unequal pair is accepted only under the explicit `cross_generator_transfer` experiment type, and both IDs are recorded.

No method mathematics, training losses, selector architecture, prompts, verbalizers, ranking logic, or metric definitions were changed.

## 2. Git Baseline

- Commit: `49805575868334064ff79a40a17c2da866c4b9d8`
- Branch: `feature/fever-formal-readiness`
- Platform: Windows
- Pre-existing dirty paths were limited to the three allowlisted files and were not modified by Phase B.

## 3. Files Changed

Modified by Phase B:

- `src/experiment_identity.py`
- `src/baselines/common.py`
- `scripts/03_compute_label_posteriors.py`
- `scripts/07_eval_rag_classification.py`
- `scripts/08_select_naive_topm.py`
- `scripts/12_select_bge_reranker.py`
- `scripts/12b_train_infogain_reranker.py`
- `scripts/12c_select_infogain_reranker.py`
- `scripts/preformal/26_train_signed_v1.py`
- `scripts/preformal/27_select_signed_v1.py`
- `scripts/preformal/27a_select_no_evidence.py`

Added by Phase B:

- `src/artifact_binding.py`
- `scripts/create_generator_manifest.py`
- `tests/test_atomic_generator_binding.py`
- `PHASE_B_ATOMIC_GENERATOR_BINDING_IMPLEMENTATION_REPORT.md`

The pre-existing dirty files `RAG_CBWDM_MASTER_SERVER_RUNBOOK_2026-08-24.md`, `SERVER_REBUILD_DELTA_AUDIT.md`, and `SIGNED_ALIGNMENT_SOURCE_AUDIT.md` were not edited.

## 4. Binding Architecture

`src/artifact_binding.py` provides the shared, fail-closed formal-v2 validation layer. Its compact binding carries stable dataset, method, generator-dependency, conditioning-generator, generator fingerprint, retrieval protocol/source, posterior references, teacher references, seed, config provenance, training-manifest identity, and checkpoint tree identity without copying full upstream manifests.

The module defines:

- artifact binding schema `rag_cbwdm_artifact_binding.v1`;
- generator manifest schema `rag_cbwdm_generator_manifest.v1`;
- selection manifest schema `rag_cbwdm_selection_manifest.v2`;
- evaluation manifest schema `rag_cbwdm_evaluation_manifest.v2`;
- matched-main and cross-generator-transfer validation;
- reusable teacher, checkpoint, selection, and generator-manifest validators.

Formal validation reopens referenced upstream artifacts and checks their recorded paths, SHA-256 values, fingerprints, structured identities, config SHA, and Git provenance. Legacy mode does not synthesize formal provenance.

## 5. InfoGain Training Binding

`scripts/12b_train_infogain_reranker.py` adds an opt-in `--formal-v2-identity` path. It validates the Phase A teacher manifest, dataset identity, conditioning generator, teacher SHA, posterior manifest/SHA/fingerprint, retrieval identity, and config SHA before training. The completed v2 training manifest records the compact binding, method, seed, checkpoint path/tree SHA/fingerprint, config provenance, and Git state.

The non-formal contract remains backward-compatible, including its historical fingerprint inputs and resume behavior.

## 6. signed-v1 Training Binding

`scripts/preformal/26_train_signed_v1.py` applies the same formal-v2 teacher validation and propagation. It additionally verifies that the explicitly supplied posterior and retrieval files match the teacher's bound posterior and retrieval source SHA values. Its v2 training manifest records the full compact binding and checkpoint identity.

The frozen signed-v1 contract, CrossEncoder selector, loss, sampling, thresholds, and hyperparameters are unchanged.

## 7. InfoGain Selection Binding

`scripts/12c_select_infogain_reranker.py` validates the formal training manifest and checkpoint tree before loading the selector. The resulting selection v2 sidecar carries `generator_dependency=conditioned`, the conditioning generator, dataset, method, retrieval protocol/source, seed, training-manifest SHA/fingerprint, and checkpoint SHA/fingerprint. Selection remains gold-free.

## 8. signed-v1 Selection Binding

`scripts/preformal/27_select_signed_v1.py` performs the equivalent formal checkpoint validation and publishes the same structured conditioned binding. The selection algorithm and inference payload are unchanged; no teacher-only labels are introduced.

## 9. Generator-Independent Selection Contract

The No Evidence, naive/retrieval Top-k, and BGE selection entry points can publish v2 sidecars with:

- `generator_dependency=independent`;
- `conditioning_generator_id=null`;
- stable dataset identity;
- retrieval protocol and source-artifact SHA.

These selections remain evaluable with any frozen evaluation generator for `matched_main`. They cannot be mislabeled as cross-generator transfer because they have no conditioning generator.

## 10. Atomic Evaluation Generator Contract

`scripts/create_generator_manifest.py` publishes one atomic generator identity from a frozen config and an explicit stable `generator_id`. It binds model and tokenizer names, revisions, SHA identities, dtype, device map, trust setting, maximum context length, prompt version/hash, verbalizer hash, dataset, config path/SHA, Git state, and environment.

In formal-v2 mode, `scripts/07_eval_rag_classification.py` forbids `--model-name`. It resolves every generator load field from the validated generator manifest, verifies the identity against the frozen config, and rejects incompatible model/revision/tokenizer combinations before model loading. The current scorer requires the tokenizer path to equal the model path and fails closed otherwise.

## 11. Matched-Main Enforcement

For conditioned selections, the default formal experiment type is `matched_main`, and validation requires exact equality between conditioning and evaluation generator IDs. Both signed-v1 and InfoGain therefore reject a Qwen1.5-conditioned selection evaluated as a Qwen7 matched-main result.

Independent selections do not participate in this equality check because their conditioning generator is explicitly null; an evaluation generator is still mandatory through the generator manifest.

## 12. Cross-Generator Transfer Contract

`cross_generator_transfer` is available only in formal-v2 mode and only by explicit CLI opt-in. It requires a conditioned selection and distinct conditioning/evaluation generator IDs. Evaluation binding, metrics, and the evaluation manifest record both IDs plus `experiment_type=cross_generator_transfer`. Equal-ID runs are rejected as mislabeled transfer experiments.

## 13. Selection Provenance Validation

The reusable validator checks selection existence, path, status, output SHA, method, dataset, schema, dependency type, binding shape, contract fingerprint, and contract/manifest binding equality. Conditioned selections must contain checkpoint provenance; validation rechecks the referenced formal training manifest, checkpoint tree, config, and upstream binding. Independent selections must carry retrieval source provenance and must not carry a conditioning generator. Ambiguous v1 sidecars are rejected in formal-v2 mode.

## 14. Evaluation Manifest Contract

Formal evaluation emits `rag_cbwdm_evaluation_manifest.v2` and records dataset, method, generator dependency, conditioning generator when applicable, evaluation generator, experiment type, generator identity/fingerprint, selection JSONL SHA, selection manifest SHA/fingerprint, immutable model/tokenizer identity, prompt/verbalizer identity, config SHA, Git state, prediction SHA, and metrics SHA. The metrics JSON also records the experiment type and both generator roles without changing Accuracy or Macro-F1 computation.

## 15. Formal-v2 Path Helpers

`src/experiment_identity.py` adds deterministic helpers for:

- `<root>/<dataset>/<generator>/<method>/seed<seed>`;
- `<root>/<dataset>/<generator>/evaluation/<method>/<experiment_type>`.

The helpers use stable IDs rather than absolute model paths and do not create runtime directories.

## 16. Legacy Compatibility

All new behavior is opt-in under formal-v2 flags. Selection publication retains v1 manifests when no artifact binding is supplied. Legacy evaluation continues to support the historical model override. Legacy training fingerprint inputs remain unchanged. Historical files and manifests are neither rewritten nor migrated. Formal-v2 validators reject legacy ambiguity instead of silently upgrading it.

## 17. Tests Run

- `python -m py_compile` equivalent through the repository virtual environment for every changed/added Python file.
- Focused: `tests/test_atomic_generator_binding.py`.
- Related: `tests/test_experiment_identity.py`, `tests/test_label_logits_and_manifest.py`, `tests/test_preformal_signed_v1.py`, `tests/test_fever_baselines.py`, `tests/test_fm2_support.py`, `tests/test_fever_formal_protocol.py`, and `tests/test_disk_bm25_and_limits.py`.
- Full suite: `python -m pytest -q` equivalent through the repository virtual environment.

## 18. Test Results

- Python compilation: PASS.
- Focused Phase B: 9 passed.
- Related regression set: 103 passed, 2 subtests passed.
- Full suite: 195 passed, 2 subtests passed.
- No live model or server experiment was run.

## 19. Remaining P0

None within the scoped Phase B implementation.

## 20. Remaining P1

- Implement the explicitly excluded multi-generator/multi-dataset matrix orchestrator after Phase B review.
- Perform the separately scoped signed-v1 formal registry migration.
- Exercise the contracts in controlled server runs for each approved generator/dataset pair.

## 21. Risks / Limitations

- This change validates provenance and fail-closed behavior with synthetic artifacts; it does not execute live generator or reranker models.
- Remote Hugging Face model identities retain the repository's Phase A convention of hashing the model/revision identity when no local tree is available. Pinning resolved immutable revisions remains operationally important for production runs.
- The full experiment matrix, result tables, held-out opening, additional seeds, and transfer executions remain out of scope.

## HANDOFF_TO_CHATGPT

STATUS: COMPLETED
GIT_BASELINE: 49805575868334064ff79a40a17c2da866c4b9d8

FILES_CHANGED: src/experiment_identity.py; src/baselines/common.py; scripts/03_compute_label_posteriors.py; scripts/07_eval_rag_classification.py; scripts/08_select_naive_topm.py; scripts/12_select_bge_reranker.py; scripts/12b_train_infogain_reranker.py; scripts/12c_select_infogain_reranker.py; scripts/preformal/26_train_signed_v1.py; scripts/preformal/27_select_signed_v1.py; scripts/preformal/27a_select_no_evidence.py
FILES_ADDED: src/artifact_binding.py; scripts/create_generator_manifest.py; tests/test_atomic_generator_binding.py; PHASE_B_ATOMIC_GENERATOR_BINDING_IMPLEMENTATION_REPORT.md

BINDING_SCHEMA_VERSION: rag_cbwdm_artifact_binding.v1
GENERATOR_MANIFEST_SCHEMA_VERSION: rag_cbwdm_generator_manifest.v1
SELECTION_MANIFEST_SCHEMA_VERSION: rag_cbwdm_selection_manifest.v2
EVALUATION_MANIFEST_SCHEMA_VERSION: rag_cbwdm_evaluation_manifest.v2

INFOGAIN_TRAIN_BINDING: IMPLEMENTED_AND_FAIL_CLOSED_IN_FORMAL_V2
INFOGAIN_SELECTION_BINDING: IMPLEMENTED_AND_CHECKPOINT_VERIFIED

SIGNED_V1_TRAIN_BINDING: IMPLEMENTED_AND_FAIL_CLOSED_IN_FORMAL_V2
SIGNED_V1_SELECTION_BINDING: IMPLEMENTED_AND_CHECKPOINT_VERIFIED

GENERATOR_INDEPENDENT_SELECTION_STATUS: EXPLICIT_INDEPENDENT_BINDING_WITH_NULL_CONDITIONING_GENERATOR

ATOMIC_EVALUATION_GENERATOR_STATUS: IMPLEMENTED_WITH_STANDALONE_VALIDATED_GENERATOR_MANIFEST

MATCHED_MAIN_ENFORCEMENT: CONDITIONED_GENERATOR_IDS_MUST_MATCH
TRANSFER_MODE_STATUS: EXPLICIT_OPT_IN_IMPLEMENTED_AND_BOTH_GENERATOR_IDS_RECORDED

SELECTION_VALIDATOR_STATUS: STRICT_V2_VALIDATION_IMPLEMENTED; LEGACY_REJECTED_IN_FORMAL_V2

PY_COMPILE: PASS
FOCUSED_TESTS: PASS_9
RELATED_TESTS: PASS_103_PLUS_2_SUBTESTS
FULL_SUITE: PASS_195_PLUS_2_SUBTESTS

METHOD_MATH_CHANGED: NO
PROMPT_BYTES_CHANGED: NO
VERBALIZERS_CHANGED: NO
METRIC_DEFINITIONS_CHANGED: NO
HISTORICAL_ARTIFACTS_CHANGED: NO

REMAINING_P0: NONE_WITHIN_PHASE_B_SCOPE
REMAINING_P1: MATRIX_ORCHESTRATOR; SIGNED_V1_FORMAL_REGISTRY_MIGRATION; CONTROLLED_SERVER_EXECUTION

RECOMMENDED_NEXT_STEP: Review Phase B schemas and then implement the separately scoped formal matrix orchestrator.

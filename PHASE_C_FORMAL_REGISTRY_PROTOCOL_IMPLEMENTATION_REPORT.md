# Phase C Formal Registry and Dataset Protocol Implementation Report

## 1. Summary

Phase C adds a versioned, path-independent formal-v2 registry that freezes the main-table method set, canonical Ours, generator dependency, learned-selector seed policy, dataset retrieval protocols, split roles, held-out freeze requirements, and strict result eligibility rules.

The formal-v2 main table is exactly `no_evidence`, `retrieval_topk`, `bge`, `infogain`, and `rag_cbwdm_signed_v1`. `rag_cbwdm_signed_v1` is the sole publication-facing `Ours`. Legacy `rag_cbwdm`, signed-v2/v2.1/v2.2, oracles, diagnostics, and transfer results are explicitly non-main-table.

No experiment was run and no held-out data was opened. Method mathematics, prompts, verbalizers, and metric definitions are unchanged.

## 2. Git Baseline

- Commit: `72a6c3bb0aa29d552cb43144882b28e3a444d8d2`
- Branch: `feature/fever-formal-readiness`
- Platform: Windows
- The initial dirty set contained only the three allowlisted paths.

## 3. Files Changed

Modified by Phase C:

- `src/artifact_binding.py`
- `src/formal_config.py`
- `src/formal_readiness.py`

Added by Phase C:

- `src/formal_registry.py`
- `tests/test_formal_registry.py`
- `PHASE_C_FORMAL_REGISTRY_PROTOCOL_IMPLEMENTATION_REPORT.md`

The allowlisted pre-existing dirty files `RAG_CBWDM_MASTER_SERVER_RUNBOOK_2026-08-24.md`, `SERVER_REBUILD_DELTA_AUDIT.md`, and `SIGNED_ALIGNMENT_SOURCE_AUDIT.md` were not modified.

## 4. Formal Registry Architecture

`src/formal_registry.py` is the authoritative formal-v2 semantic registry. It contains no absolute paths, timestamps, or machine-local data. The registry exposes detached copies, explicit lookup/normalization helpers, a deterministic fingerprint, dataset-method compatibility validation, split-role validation, held-out freeze validation, and main-table result eligibility validation.

Historical FEVER v1 and signed-v1 preformal registries remain in place. Phase C does not reinterpret or rewrite their artifacts.

## 5. Main-Table Method Registry

The canonical ordered method IDs are:

1. `no_evidence`
2. `retrieval_topk`
3. `bge`
4. `infogain`
5. `rag_cbwdm_signed_v1`

Each method records `method_id`, display name, main-table eligibility, generator dependency, learned-selector status, seed policy, selection kind, diagnostic status, transfer eligibility, required provenance level, and exact historical artifact aliases where needed. `naive_topm` maps explicitly to `retrieval_topk`, and `infogain_fever` maps explicitly to `infogain`; there is no fuzzy string matching.

## 6. Canonical Ours

The formal-v2 canonical Ours is `rag_cbwdm_signed_v1`, with display label `Ours`. It is generator-conditioned, learned, trained at seeds 13/21/42, and transfer-eligible only for separately labeled transfer studies. A transfer result is never main-table eligible.

## 7. Excluded Diagnostic / Legacy Methods

The registry explicitly marks these as non-main-table:

- legacy `rag_cbwdm`;
- `rag_cbwdm_signed_v2`;
- `rag_cbwdm_signed_v21`;
- `rag_cbwdm_signed_v22`;
- `signed_gate_oracle`;
- `cbwdm_oracle`;
- `gold_oracle`.

They remain readable for historical, diagnostic, ablation, or oracle use. No implementation was deleted or renamed.

## 8. Dataset Protocol Registry

The dataset registry binds each stable dataset ID to exactly one current main-table retrieval protocol, its publication-facing retrieval label, candidate-pool semantics, compatible methods, split roles, and final split. The common table column is `Retrieval Top-k`.

Dataset-method compatibility is checked structurally. Cross-dataset retrieval protocol substitution fails closed.

## 9. FEVER Protocol

- Dataset ID: `fever_binary_v2`
- Retrieval protocol: `fever_bm25_v1`
- Publication display: `BM25 Top-k`
- Final split: `held_out_test`
- `validation`, including the development 500, is `development_calibration`, not an untouched final test.

## 10. FM2 Protocol

- Dataset ID: `fm2_official_closed_page_v1`
- Retrieval protocol: `fm2_official_closed_page_v1`
- Publication display: `Official-pool Top-k`
- Candidate ordering: official closed-page/source order
- Final split: `test`
- `dev` is development/calibration, not final evaluation.

FM2 BM25 is explicitly not implemented. `fever_bm25_v1` cannot validate as an FM2 protocol.

## 11. BGE Freeze Boundary

The registry identifies the method as `bge` but does not choose base versus large. Held-out readiness requires a separately supplied, human-frozen `model_id`, immutable revision, and SHA. Until those fields are supplied and signed off, BGE remains an unresolved freeze decision and held-out readiness is blocked.

## 12. Seed Policy

Learned methods `infogain` and `rag_cbwdm_signed_v1` use exactly seeds 13, 21, and 42.

`no_evidence`, `retrieval_topk`, and `bge` use `deterministic_no_training_seed`, an empty seed set, and no replication across learned-method seeds. Evaluation generators receive no artificial sampling seed because the label-logit scorer performs deterministic forward scoring rather than stochastic decoding.

## 13. Split Policy

FEVER roles are `train_core=learning`, `validation=development_calibration`, `preformal_eval=development_evaluation`, and `held_out_test=held_out_final_evaluation`.

FM2 roles are `train=learning`, `dev=development_calibration`, and `test=held_out_final_evaluation`.

Source policy does not claim that any runtime/server held-out artifact is historically untouched.

## 14. Held-Out Freeze Contract

The source-level freeze template requires dataset ID, formal registry version/fingerprint, generator registry fingerprint, retrieval protocol ID/fingerprint, prompt and verbalizer hashes, Top-k, evidence budget, learned seed set, exact BGE model freeze, Git commit, config SHA, freeze fingerprint, and a human/runtime sign-off referencing that exact freeze fingerprint.

Draft or incomplete metadata never authorizes execution. The contract validates metadata only and does not inspect or open held-out data.

## 15. Formal Main-Table Eligibility

`validate_main_table_result` accepts only an explicitly supplied evaluation manifest candidate. It requires:

- completed evaluation manifest v2 and evaluation binding v1;
- formal-v2 provenance;
- a registered main-table method;
- compatible dataset and retrieval protocol;
- `matched_main` experiment type;
- matching conditioning/evaluation generator IDs for conditioned methods;
- a permitted learned seed, or no seed for deterministic methods;
- the dataset's final split;
- complete selection, generator, prediction, metrics, config, prompt, verbalizer, and Git provenance;
- a complete, fingerprinted, signed-off held-out freeze.

Diagnostics, oracles, transfer runs, legacy methods, wrong protocols, mismatched generators, unfinished artifacts, development splits, and incomplete freezes fail closed. This supplies a later orchestrator with structured discovery rules and requires no filesystem-wide fuzzy matching.

## 16. Registry Fingerprint

- Schema: `rag_cbwdm_formal_registry.v2`
- Registry version: `formal_v2.0`
- Fingerprint: `98617377aa8513eda5fa3554ff57905945ca32f3e6184b601fead5bdf010a508`

The fingerprint covers the main-table set, method roles/dependencies, seed policies, artifact aliases, dataset protocol mappings, compatible methods, split policies, held-out requirements, unresolved freeze decisions, and eligibility rules. Tests verify determinism and sensitivity to semantic changes.

## 17. Formal Readiness Integration

`src/formal_readiness.py` adds a separate formal-v2 readiness entry point. It reports registry version/fingerprint, dataset protocol, canonical Ours, allowed methods, learned seed policy, held-out status, unresolved freeze decisions, and the explicit fact that source cannot prove historical untouched state.

`src/formal_config.py` exposes a non-authorizing formal-v2 held-out freeze template beside the legacy FEVER v1 freezer. The old `CANONICAL_METHODS`, readiness schema, frozen-config schema, and legacy checks are unchanged, so formal-v2 cannot silently fall back to legacy `rag_cbwdm`.

## 18. Legacy / Preformal Compatibility

Existing FEVER formal-v1 and signed-v1 preformal workflows remain readable and pass regression tests. Their registries and manifests were not migrated or rewritten. Phase C only adds parallel formal-v2 semantics.

The Phase B evaluation binding now carries the learned selection seed forward, enabling formal-v2 result eligibility to enforce the seed registry without changing evaluation behavior.

## 19. Tests Run

- Python compilation for every changed/added Python file.
- Focused: `tests/test_formal_registry.py`.
- Related: `tests/test_experiment_identity.py`, `tests/test_atomic_generator_binding.py`, `tests/test_preformal_signed_v1.py`, `tests/test_fever_formal_protocol.py`, `tests/test_fever_baselines.py`, and `tests/test_fm2_support.py`.
- Full suite: `python -m pytest -q` through the repository virtual environment.

## 20. Test Results

- Python compilation: PASS.
- Focused Phase C: 20 passed.
- Required related regressions: 90 passed.
- Full suite: 215 passed, 2 subtests passed.
- No live model, server experiment, held-out split, or FM2 test data was run/opened.

## 21. Remaining P0

- Before any held-out launch, freeze the human-selected BGE model ID/revision/SHA.
- Freeze the generator registry, prompt/verbalizer contract, evidence budget, Git/config identity, and obtain runtime/human sign-off against the exact freeze fingerprint.

These are runtime/human freeze inputs, not missing source-level Phase C mechanisms.

## 22. Remaining P1

- Implement the separately scoped formal matrix orchestrator using the new registry and eligibility helpers.
- Add formal result-table generation that consumes only validated eligible manifests.
- Execute approved seeds/generators/datasets only after the held-out gate is complete.

## 23. Risks / Limitations

- The registry proves source policy, not that a server-held split has never been accessed.
- BGE base-versus-large remains intentionally unresolved.
- No generator registry or full matrix orchestrator is implemented here.
- Historical artifact aliases are explicit compatibility mappings; new publication-facing rows use canonical formal-v2 IDs.
- A completed sign-off object in synthetic tests demonstrates validation behavior only; it is not a real runtime authorization.

## HANDOFF_TO_CHATGPT

STATUS: COMPLETED
GIT_BASELINE: 72a6c3bb0aa29d552cb43144882b28e3a444d8d2

FILES_CHANGED: src/artifact_binding.py; src/formal_config.py; src/formal_readiness.py
FILES_ADDED: src/formal_registry.py; tests/test_formal_registry.py; PHASE_C_FORMAL_REGISTRY_PROTOCOL_IMPLEMENTATION_REPORT.md

FORMAL_REGISTRY_SCHEMA_VERSION: rag_cbwdm_formal_registry.v2
FORMAL_REGISTRY_FINGERPRINT: 98617377aa8513eda5fa3554ff57905945ca32f3e6184b601fead5bdf010a508

CANONICAL_OURS: rag_cbwdm_signed_v1

MAIN_TABLE_METHODS: no_evidence; retrieval_topk; bge; infogain; rag_cbwdm_signed_v1

LEGACY_RAG_CBWDM_MAIN_TABLE_ELIGIBLE: NO
SIGNED_V2_MAIN_TABLE_ELIGIBLE: NO
SIGNED_V21_MAIN_TABLE_ELIGIBLE: NO
SIGNED_V22_MAIN_TABLE_ELIGIBLE: NO
TRANSFER_MAIN_TABLE_ELIGIBLE: NO

FEVER_DATASET_ID: fever_binary_v2
FEVER_RETRIEVAL_PROTOCOL: fever_bm25_v1
FEVER_RETRIEVAL_DISPLAY: BM25 Top-k

FM2_DATASET_ID: fm2_official_closed_page_v1
FM2_RETRIEVAL_PROTOCOL: fm2_official_closed_page_v1
FM2_RETRIEVAL_DISPLAY: Official-pool Top-k
FM2_BM25_IMPLEMENTED: NO

LEARNED_METHOD_SEEDS: infogain=[13,21,42]; rag_cbwdm_signed_v1=[13,21,42]
DETERMINISTIC_METHOD_SEED_POLICY: no_evidence/retrieval_topk/bge have no training seed and are not replicated across learned seeds

FEVER_FINAL_SPLIT_POLICY: validation is development/calibration; held_out_test is final
FM2_FINAL_SPLIT_POLICY: dev is development/calibration; test is final
HELDOUT_FREEZE_STATUS: SOURCE_CONTRACT_IMPLEMENTED; RUNTIME_FREEZE_AND_SIGN_OFF_REQUIRED

BGE_MODEL_FREEZE_STATUS: UNRESOLVED_HUMAN_DECISION; MODEL_ID_REVISION_SHA_REQUIRED

MAIN_TABLE_ELIGIBILITY_VALIDATOR: IMPLEMENTED_FAIL_CLOSED
FORMAL_READINESS_INTEGRATION: IMPLEMENTED_AS_PARALLEL_FORMAL_V2_WITH_LEGACY_V1_UNCHANGED

PY_COMPILE: PASS
FOCUSED_TESTS: PASS_20
RELATED_TESTS: PASS_90
FULL_SUITE: PASS_215_PLUS_2_SUBTESTS

METHOD_MATH_CHANGED: NO
PROMPT_BYTES_CHANGED: NO
METRIC_DEFINITIONS_CHANGED: NO
HISTORICAL_ARTIFACTS_CHANGED: NO

REMAINING_P0: BGE_MODEL_FREEZE; GENERATOR_AND_PROTOCOL_FREEZE; RUNTIME_HUMAN_SIGN_OFF_BEFORE_HELD_OUT
REMAINING_P1: FORMAL_MATRIX_ORCHESTRATOR; ELIGIBLE_RESULT_TABLE_GENERATION; APPROVED_RUNTIME_EXECUTION

RECOMMENDED_NEXT_STEP: Make the remaining human freeze decisions, publish a signed held-out freeze manifest, then implement the formal matrix orchestrator against this registry.

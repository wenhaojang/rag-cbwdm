# Phase A Experiment Identity Implementation Report

## 1. Summary

Implemented the first Phase A slice: versioned dataset, generator, and retrieval-protocol identities; posterior manifest schema v2; a strict reusable posterior provenance validator; generator-conditioned teacher binding for InfoGain and signed-v1; and deterministic helpers for the future `formal_v2` artifact hierarchy.

The implementation is opt-in strict for formal-v2 while preserving legacy behavior. It does not implement the full experiment matrix, change method mathematics, alter prompt bytes or verbalizers, migrate historical artifacts, or open a held-out split.

## 2. Git Baseline

- Required baseline: `2847de6`
- Full audited HEAD: `2847de65011240b2759a0c6fd897fadc0eede55d`
- Branch: `feature/fever-formal-readiness`
- Initial dirty allowlist:
  - `M RAG_CBWDM_MASTER_SERVER_RUNBOOK_2026-08-24.md`
  - `?? SERVER_REBUILD_DELTA_AUDIT.md`
  - `?? SIGNED_ALIGNMENT_SOURCE_AUDIT.md`

The three allowlisted paths were not modified by this implementation.

## 3. Files Changed

Modified:

- `scripts/03_compute_label_posteriors.py`
- `scripts/12a_build_infogain_teacher.py`
- `scripts/preformal/25_materialize_signed_v1_teacher.py`
- `scripts/run_fever_cbwdm.py`

Added:

- `src/experiment_identity.py`
- `tests/test_experiment_identity.py`
- `PHASE_A_EXPERIMENT_IDENTITY_IMPLEMENTATION_REPORT.md`

The `run_fever_cbwdm.py` change is a narrow compatibility change: its posterior artifact validator now accepts both manifest v1 and v2 and still rejects every other schema version.

## 4. Experiment Identity Schema

The umbrella identity schema is:

```text
rag_cbwdm_experiment_identity.v1
  dataset_identity
  generator_identity
  retrieval_protocol_identity
```

Each component has its own versioned schema. Stable identifiers are validated against a lowercase path-safe contract and remain distinct from local deployment paths.

Two modes are explicit:

- `formal_v2`: requires a posterior manifest v2, formal identity mode, explicit generator ID, all component schemas, immutable identity/provenance consistency, artifact SHA, config SHA, retrieval SHA, and Git commit.
- `legacy`: validates the historical sidecar status/path/output SHA and any available expected fields, but does not claim that missing stable identity fields exist.

## 5. Dataset Identity Contract

Schema: `rag_cbwdm_dataset_identity.v1`.

Required mappings:

- `fever2 -> fever_binary_v2`
- `fm2 -> fm2_official_closed_page_v1`

The resolver records the legacy config alias rather than rewriting config values or historical artifacts. `fever3` remains readable as its existing legacy ID for backward compatibility but is not added to the Phase A target matrix.

An explicit dataset ID that conflicts with a known alias fails closed.

## 6. Generator Identity Contract

Schema: `rag_cbwdm_generator_identity.v1`.

The contract separates:

- stable `generator_id`;
- `model_family`;
- model/tokenizer deployment paths;
- requested and resolved revisions;
- model/tokenizer hashes when available;
- prompt version/hash and verbalizer hash;
- dtype, device map, and `trust_remote_code`;
- identity source (`explicit`, `inferred_known_model`, or `unresolved_legacy`).

Known non-formal inference supports:

- `qwen2.5-1.5b-instruct`
- `qwen2.5-7b-instruct`

Inference is based on the model leaf name and is independent of `/root/models`, a Windows model root, or a Hugging Face namespace. Formal-v2 requires an explicit generator ID even for known Qwen models. A known model path paired with the wrong known ID fails closed. No future non-Qwen generator is hardcoded.

## 7. Retrieval Protocol Identity Contract

Schema: `rag_cbwdm_retrieval_protocol_identity.v1`.

The identity records:

- `retrieval_protocol_id`;
- stable `dataset_id`;
- source retrieval/candidate-pool SHA-256;
- optional retrieval model/index identity.

Current deterministic mappings are:

- FEVER binary + `bm25` -> `fever_bm25_v1`
- FM2 + `fm2_official_closed_page_v1` -> `fm2_official_closed_page_v1`

An FM2 method other than the official closed-page protocol is not silently labeled as the official protocol. Formal-v2 fails when the retrieval protocol cannot be resolved or explicitly supplied.

## 8. Posterior Manifest Changes

New completed/running posterior manifests use:

```text
rag_cbwdm_posterior_manifest.v2
```

They preserve the existing provenance payload, stage fingerprint, output SHA, Git state, environment, row progress, timestamps, prompt/verbalizer hashes, retrieval SHA, and config SHA. They additionally record:

- `identity_mode`;
- `dataset_identity`;
- `generator_identity`;
- `retrieval_protocol_identity`;
- `identity_fingerprint`.

Resolved model and tokenizer commits are copied into the structured generator identity after model load and the identity fingerprint is recomputed. JSONL row schema and prompt construction are unchanged.

New CLI options are:

- `--dataset-id`
- `--generator-id`
- `--retrieval-protocol-id`
- `--formal-v2-identity`

Without formal-v2 opt-in, known Qwen IDs may be deterministically inferred and recorded; unknown legacy models remain explicitly unresolved rather than deriving identity from an absolute path.

## 9. Posterior Validation Rules

`validate_posterior_provenance()` verifies:

1. posterior JSONL exists;
2. sidecar exists and is valid JSON;
3. status is `completed`;
4. recorded output path resolves to the intended JSONL;
5. current JSONL SHA matches `output_sha256`;
6. expected dataset, split, generator, and retrieval protocol match when supplied;
7. formal-v2 manifest and component schema versions match;
8. formal-v2 identity mode is explicit and generator identity source is `explicit`;
9. required config, retrieval, prompt, verbalizer, generator, and Git provenance exists;
10. dataset/generator/retrieval identity fields agree with legacy provenance fields;
11. the retrieval source SHA agrees with posterior input SHA;
12. the identity fingerprint matches the structured identity payload.

Every mismatch raises a clear exception. Formal-v2 never downgrades to a warning or legacy continuation.

## 10. InfoGain Teacher Integration

`scripts/12a_build_infogain_teacher.py` now auto-locates `<posterior>.manifest.json` or accepts `--posterior-manifest`. It accepts optional expected dataset/generator/retrieval IDs and `--formal-v2-identity`.

For a structured identity sidecar, the teacher manifest binds:

- posterior JSONL SHA;
- posterior manifest path/SHA/fingerprint;
- posterior identity fingerprint;
- dataset, generator, and retrieval identities;
- generator ID and split.

In formal-v2, manually supplied legacy generator/prompt/verbalizer metadata must match the authoritative posterior manifest. Missing CLI values are populated from that manifest rather than becoming a second source of truth.

The InfoGain teacher equation, labels, thresholds, and output rows are unchanged.

## 11. signed-v1 Teacher Integration

`scripts/preformal/25_materialize_signed_v1_teacher.py` has the same sidecar auto-discovery and formal-v2 options. Its formal validation also compares the config-derived stable dataset ID and training split, and it verifies that the retrieval JSONL SHA equals the retrieval source SHA bound into the posterior identity.

The signed-v1 teacher contract and manifest bind the posterior manifest identity, generator ID, and component identities. signed-v1 alignment, smoothing, Theta, admissibility, thresholds, supervision classes, and teacher trajectory mathematics are unchanged.

## 12. Formal-v2 Path Helpers

Pure deterministic helpers now produce:

```text
<formal_v2_root>/<dataset_id>/
<formal_v2_root>/<dataset_id>/<generator_id>/
<formal_v2_root>/<dataset_id>/<generator_id>/posteriors/<split>/
```

The helpers validate every path component and create no directories or artifacts. Existing outputs are not moved or renamed.

## 13. Legacy Compatibility

- Old posterior manifests are not rewritten.
- Legacy validation accepts a completed v1 sidecar with matching path and SHA.
- Formal-v2 validation rejects v1, missing identities, inferred identities, incomplete sidecars, and `legacy_compatible` v2 manifests.
- Existing teacher commands still work without a sidecar or formal-v2 options.
- Auto-discovered legacy v1 sidecars are validated but do not change the legacy teacher fingerprint/schema unless identity binding was explicitly requested.
- New structured sidecars bind automatically into generator-conditioned teachers.
- The FEVER runner accepts v1 and v2 posterior sidecars so the manifest upgrade does not invalidate new posterior outputs.

## 14. Tests Run

Interpreter used because bare `python` was not registered in the shell:

```text
.venv\Scripts\python.exe
```

Compilation:

```text
python -m py_compile
  src/experiment_identity.py
  scripts/03_compute_label_posteriors.py
  scripts/12a_build_infogain_teacher.py
  scripts/preformal/25_materialize_signed_v1_teacher.py
  scripts/run_fever_cbwdm.py
  tests/test_experiment_identity.py
```

Focused and required related tests:

```text
python -m pytest -q
  tests/test_experiment_identity.py
  tests/test_label_logits_and_manifest.py
  tests/test_preformal_signed_v1.py
  tests/test_fever_baselines.py
  tests/test_fm2_support.py
```

Full regression:

```text
python -m pytest -q
```

## 15. Test Results

- Python compilation: **PASS**
- New focused identity tests: **15 passed**
- Focused plus required related tests: **60 passed**
- Full suite: **186 passed, 2 subtests passed**

Coverage includes alias resolution, path-independent Qwen IDs, explicit formal identity enforcement, accepted matching manifests, dataset/generator/split/retrieval mismatch rejection, tamper/missing/incomplete rejection, legacy/formal separation, posterior-producer v2 publication, both teacher bindings, and collision-safe path construction.

## 16. Remaining P0 Work

Outside this first slice:

- propagate the atomic generator identity contract through selector training, selection, and final evaluation;
- enforce matched conditioning versus cross-generator transfer in downstream manifests;
- migrate provisional `rag_cbwdm_signed_v1` into a versioned formal registry/readiness contract;
- implement the collision-safe hierarchy in an orchestrator rather than only path helpers;
- complete held-out freeze/sign-off policy and the FM2 main-table retrieval protocol decision.

## 17. Remaining P1 Work

- non-Qwen prompt/tokenizer/model-family compatibility tests;
- a dataset × generator × method matrix orchestrator and summary isolation;
- an FM2 formal freeze/readiness gate;
- stronger BGE and InfoGain checkpoint/model provenance;
- deterministic-runtime recording across learned seeds;
- exact BGE model/revision freeze.

## 18. Risks / Limitations

- Formal-v2 is not yet threaded through selector checkpoints or final evaluation, so this change alone cannot run a complete formal matrix.
- Known-generator inference is deliberately limited to non-formal compatibility; it is not accepted as formal identity.
- `tokenizer_sha256` remains optional because the current producer does not separately hash tokenizer files; the model-directory SHA and resolved tokenizer revision remain recorded where available.
- The path helpers do not create, migrate, or discover artifacts.
- No server model/runtime behavior was exercised.

## HANDOFF_TO_CHATGPT

STATUS: COMPLETED
GIT_BASELINE: 2847de6
FILES_CHANGED: scripts/03_compute_label_posteriors.py; scripts/12a_build_infogain_teacher.py; scripts/preformal/25_materialize_signed_v1_teacher.py; scripts/run_fever_cbwdm.py
FILES_ADDED: src/experiment_identity.py; tests/test_experiment_identity.py; PHASE_A_EXPERIMENT_IDENTITY_IMPLEMENTATION_REPORT.md

IDENTITY_SCHEMA_VERSION: rag_cbwdm_experiment_identity.v1
POSTERIOR_MANIFEST_SCHEMA_VERSION: rag_cbwdm_posterior_manifest.v2

DATASET_IDS_SUPPORTED: fever_binary_v2; fm2_official_closed_page_v1
GENERATOR_IDS_SUPPORTED: qwen2.5-1.5b-instruct; qwen2.5-7b-instruct
LEGACY_ALIAS_POLICY: explicit aliases; fever2 maps to fever_binary_v2; fm2 maps to fm2_official_closed_page_v1; historical configs/artifacts are not rewritten

POSTERIOR_VALIDATOR_STATUS: IMPLEMENTED_STRICT_FORMAL_V2_AND_LEGACY_COMPATIBLE
INFOGAIN_BINDING_STATUS: IMPLEMENTED
SIGNED_V1_BINDING_STATUS: IMPLEMENTED
FORMAL_V2_PATH_HELPER_STATUS: IMPLEMENTED_NO_ARTIFACT_CREATION

PY_COMPILE: PASS
FOCUSED_TESTS: PASS_15
RELATED_TESTS: PASS_60_INCLUDING_FOCUSED

METHOD_MATH_CHANGED: NO
PROMPT_BYTES_CHANGED: NO
VERBALIZERS_CHANGED: NO
HISTORICAL_ARTIFACTS_CHANGED: NO

REMAINING_P0: downstream atomic generator identity and matched-conditioning enforcement; signed-v1 formal registry; hierarchy orchestration; held-out freeze; FM2 retrieval protocol decision
REMAINING_P1: non-Qwen compatibility; matrix orchestrator; FM2 formal gate; stronger BGE/InfoGain provenance; deterministic runtime and model freeze

RECOMMENDED_NEXT_STEP: wire the same immutable posterior/generator binding into selector training, selection, and final evaluation before adding the full matrix orchestrator

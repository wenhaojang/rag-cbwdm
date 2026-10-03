# FM2 Formal-v2 Foundation Implementation Report

## 1. Files Changed

Source:

- `src/datasets/fm2.py`
- `src/fm2_formal.py` (new)
- `src/formal_registry.py`
- `src/formal_matrix.py`
- `src/metrics.py`
- `scripts/prepare_fm2.py`
- `scripts/03_compute_label_posteriors.py`
- `scripts/07_eval_rag_classification.py`
- `scripts/08_select_naive_topm.py`
- `scripts/preformal/29_summarize_results.py`

Tests:

- `tests/test_fm2_support.py`
- `tests/test_formal_matrix.py`
- `tests/test_formal_registry.py`

No FEVER config, experiment artifact, model file or result was changed. No formal four-generator server config or matrix YAML was created.

## 2. Patch A — FM2 Formal Data/Split/Retrieval Contract

The FM2 adapter now defines stable identity and role constants:

```text
dataset_id             = fm2_official_closed_page_v1
dataset_family         = fm2
retrieval_protocol_id  = fm2_official_closed_page_v1

official train -> formal train_core
official dev   -> formal validation
official test  -> formal held_out_test
```

This is a role mapping only. `adapt_raw_row()` still processes each official row exactly once and retains the official split in stable IDs such as `fm2:train:<id>` and in `official_source_split`. It writes the formal role to the shared `split` field so all downstream workers use the formal namespace. There is no random split, sampling or row copy.

`scripts/prepare_fm2.py` now publishes `rag_cbwdm_fm2_prepare_manifest.v2` with dataset identity, retrieval protocol identity, source commit, role mapping, official source split, formal role, raw SHA, candidate-pool path/SHA and the existing page-disjoint/candidate-pool contract.

The new `src/fm2_formal.py` is the authoritative fail-closed validator. It checks:

- manifest exists, is valid JSON, has the exact schema and completed status;
- dataset ID/family and retrieval protocol;
- pinned official source commit and expected raw-source SHA;
- official split to formal-role mapping;
- recorded pool path and current pool SHA;
- source-order and construction-gold-free contract flags;
- stable row ID, non-empty query and exact row split/official split;
- non-empty candidate array;
- deterministic candidate document ID;
- unique document IDs;
- one-based, contiguous `rank` and `source_rank` matching physical source order;
- non-empty candidate text;
- absence of gold-evidence diagnostic fields in row, pool metadata and candidates;
- row count against the manifest.

The existing page-disjoint audit remains in `prepare_fm2.py`. Development preparation still defaults to official train/dev. Nothing in the validator or planner reads test unless the requested formal role is explicitly `held_out_test`.

## 3. Patch B — Split Checks and Generic Rank Semantics

`03_compute_label_posteriors.py` rejects retrieval rows whose `split` differs from `--split`. `07_eval_rag_classification.py` similarly rejects selection rows whose split differs from the requested evaluation split. This prevents a correctly hashed artifact from being recorded under the wrong formal role.

`ClassificationMetrics` now always emits the canonical fields:

- `avg_original_retrieval_rank`
- `median_original_retrieval_rank`

For FEVER-family datasets it additionally emits the unchanged compatibility aliases:

- `avg_original_bm25_rank`
- `median_original_bm25_rank`

FM2 evaluation selects generic retrieval semantics and therefore does not publish BM25-named rank fields. Existing callers using `original_bm25_ranks` remain supported; new code passes `original_retrieval_ranks`.

`08_select_naive_topm.py` still selects exactly `candidates[:top_m]`. Only its description/metadata changed from unconditional BM25 wording to `ranking=source_order`. The preformal summary prints “Avg retrieval rank”, reads the generic metric first, falls back to old metrics, and preserves the legacy BM25 field in summary rows only when the source FEVER metric contains it.

## 4. Patch C — Formal Matrix Retrieval Binding and Result Provenance

FEVER keeps its existing string-valued retrieval input contract and paths.

For FM2, each matrix retrieval input must now be an object containing at least:

```yaml
retrieval_inputs:
  train_core:
    pool: <local pool used for planning validation>
    manifest: <local authoritative FM2 prepare manifest>
    server_pool: <optional server execution path>
    server_manifest: <optional server execution path>
  validation:
    pool: <local pool used for planning validation>
    manifest: <local authoritative FM2 prepare manifest>
    server_pool: <optional server execution path>
    server_manifest: <optional server execution path>
```

Planning validates both artifacts before emitting an executable DAG. The plan stores the validated dataset/protocol/split/role, pool and manifest SHA, row count and server paths in its semantic retrieval binding. At execution, the external retrieval node validates the server pool and manifest again and compares their identities, hashes and row count with the planned binding. Missing, incomplete, changed or mislabeled inputs fail closed.

Every `result_index` row now includes:

- `method` (while retaining `method_id`);
- `retrieval_protocol_id`;
- `retrieval_protocol_fingerprint`.

These fields are included in `_semantic_plan_payload`, so they affect the plan fingerprint. A detached result row can distinguish FM2 official-pool retrieval from FEVER BM25.

The DAG sharing contract is unchanged:

- shared once per dataset/split: No Evidence, Retrieval Top-k, BGE;
- generator-conditioned: posterior, InfoGain teacher/train/select, signed-v1 teacher/train/select, evaluation.

## 5. Patch D — Tests

Added or extended coverage includes:

- exact official-to-formal role mapping without resampling;
- canonical pool/manifest acceptance;
- missing and incomplete manifest rejection;
- wrong dataset, retrieval protocol, source split, formal role and pool SHA;
- reordered candidates, duplicate/non-contiguous ranks, duplicate document IDs and gold leakage;
- posterior/evaluator split mismatch rejection;
- source-order Top-k `[A,B,C,D]` behavior independent of score/text;
- safe selection of two candidates when `top_m=4` and the pool length is two;
- generic FM2 retrieval-rank metrics without BM25 aliases;
- stable FEVER generic plus BM25 compatibility metrics;
- result-index protocol identity for FEVER and FM2;
- FM2 full-development exclusion of held-out paths/nodes/commands;
- required FM2 external retrieval manifest;
- four-generator × five-method × seed13 topology;
- shared deterministic selections and generator-conditioned learned branches;
- absence of BM25 retrieval script, Pyserini and `fever_bm25_v1` in FM2 worker commands;
- DAG validation and output-collision protection through the existing planner validator.

## 6. Test Results

Focused tests:

```text
tests/test_fm2_support.py
tests/test_formal_matrix.py
tests/test_experiment_identity.py
tests/test_atomic_generator_binding.py
tests/test_formal_registry.py

104 passed in 8.06s
```

Full lightweight suite:

```text
286 passed, 2 subtests passed in 20.73s
```

The complete suite confirms no observed regression in FEVER, Qwen/Mistral generator identity, signed-v1/InfoGain formal binding, resume/provenance or held-out planning protections.

## 7. Audit Plan Versus Actual Code

The audit predicted 62 nodes for a four-generator, five-method, one-seed FM2 matrix. The implemented dry-run test computes the real planner topology and confirms exactly 62 nodes and 20 result rows. No code was changed merely to force that number.

The 62 nodes are:

- 1 dataset input;
- 2 external retrieval inputs;
- 3 shared deterministic selections;
- per generator: 1 generator manifest, 2 posterior nodes, 2 teachers, 2 training nodes, 2 learned selections and 5 evaluations (14 × 4).

No material topology discrepancy with the audit was found.

## 8. Remaining Work

This patch deliberately does not create Patch E artifacts. Remaining work is:

- create reviewed Qwen-0.5B, Qwen-1.5B, Qwen-7B and Mistral-7B FM2 server configs;
- create development-smoke/full-development matrix YAMLs using the new pool+manifest input shape;
- prepare official train/dev pools on the server and validate their v2 manifests;
- inspect the real candidate-count distribution before freezing every `min_docs` policy;
- freeze and verify local generator/tokenizer paths, revisions and SHA-256;
- verify A/B single-token verbalizers for all four tokenizers;
- freeze and verify `/root/models/bge-reranker-large`, including revision and SHA;
- run one-row posterior/evaluation smokes before any full-development job;
- retain official test/`held_out_test` behind final freeze and signoff;
- build an FM2-specific final held-out freeze workflow later; `src/formal_config.py` remains FEVER-specific.

## 9. Readiness Answers

### Is it safe to create FM2 server configs now?

**Yes.** The source now defines the formal roles, authoritative external retrieval contract, split checks, generic metrics and matrix provenance needed for those configs. Config review must still supply only train_core/validation pool+manifest bindings for development.

### Can Qwen2.5-1.5B FM2 development_smoke run immediately?

**Not immediately from this repository alone.** The code path is ready, but no formal FM2 smoke matrix was created in this patch and the local repository contains no prepared official pool/model artifacts. It becomes safe to run after the server train/dev pools and v2 manifests, Qwen generator manifest and BGE identity are validated and a reviewed smoke matrix is created. No official test artifact is needed.

### What server artifact validation remains before full-development?

At minimum:

1. official train/dev raw SHA and prepared v2 manifest validation;
2. pool path/SHA, formal role, source order, row count and page-disjoint integrity;
3. candidate-count distribution and method-specific `min_docs` feasibility;
4. four generator/tokenizer revisions and SHA, plus A/B tokenization;
5. BGE-large path/revision/SHA and local-files-only behavior;
6. one-row posterior and evaluator smoke for every generator;
7. dry-run review proving 20 result rows, shared deterministic artifacts and no held-out/BM25 FM2 command.

Official test/`held_out_test` remains out of scope and must not be accessed before final freeze/signoff.

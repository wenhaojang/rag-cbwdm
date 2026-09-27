# Multi-Generator × Multi-Dataset Formal Experiment Readiness Audit

Audit date: 2026-09-28

Audit mode: source-only; no server artifact access and no experiment execution

Repository: `C:\Users\wenhao\Desktop\CBWDM\rag_cbwdm`

Audited Git HEAD: `a0a0bd90e1d81caad6e7636c5eba7231b724eff1`

Audited branch: `feature/fever-formal-readiness`

Readiness labels in this report are engineering labels only:

- **READY**: the source contains a complete path for the stated scope; runtime artifacts still require execution and validation.
- **PARTIAL**: substantial source support exists, but one or more formal-experiment requirements are missing or ambiguous.
- **MISSING**: no source path implementing the requirement was found.

## 1. Executive Summary

Overall status: **PARTIAL, not ready for the complete 2-dataset × 3-generator × 5-method formal matrix**.

The repository has strong reusable components: FEVER data preparation and persistent BM25, a shared label-posterior and evaluation path, No Evidence, source-rank Top-k, BGE reranking, InfoGain, signed-v1 teacher/training/selection, checksummed manifests, formal FEVER split machinery, and an FM2 adapter with official-source checksums and a closed-page candidate protocol. Accuracy and Macro-F1 are already emitted by the shared evaluator.

The principal blockers are orchestration and identity, not a missing signed-v1 method:

1. Posterior JSONL rows do not contain generator identity. Their sidecar manifests contain good provenance, but default/formal paths do not encode a stable generator ID and downstream stages do not uniformly require and validate the posterior sidecar.
2. The formal FEVER runner and freezing/readiness system remain FEVER-specific and use the older `rag_cbwdm` method registry, not signed-v1 as provisional Ours.
3. The source supports model replacement through Hugging Face causal-LM loading, but no non-Qwen model-family compatibility contract or test proves prompt/verbalizer behavior.
4. FM2 is supported under `fm2_official_closed_page_v1`; it does not currently have an open-domain BM25 corpus/index path. Calling its source-order Top-k baseline “BM25/RAG” would be incorrect.
5. Existing output naming can collide across generators, and evaluation model overrides do not independently override model/tokenizer revisions.
6. Formal split claims must distinguish the development-used FEVER validation data from the held-out split. FM2 test is source-supported and gated, but its untouched runtime state cannot be proven by this source-only audit.

Recommended first implementation step: define and enforce a stable `(dataset_id, generator_id)` experiment identity in generator manifests and collision-safe artifact roots, then make every generator-conditioned downstream stage validate the upstream posterior manifest before any new model is run.

## 2. Current Development Status

- The development setting is **FEVER (`fever2` in config) + Qwen2.5-1.5B-Instruct**.
- The current provisional Ours is **`rag_cbwdm_signed_v1`**, the deployable selector trained from the signed-v1 teacher.
- signed-v2, signed-v2.1, and signed-v2.2 are diagnostics/ablations and are not default main-table methods.
- No v2.3 continuation is in scope.
- The source contains older production/formal paths named `rag_cbwdm`. Their presence must not silently redefine provisional Ours.
- This audit does not promote any development result to a formal result and does not verify any server runtime artifact.

## 3. Target Formal Experiment Matrix

The intended main table is:

| Dataset | Generator | No Evidence | Retrieval Top-k | BGE | InfoGain | Ours |
|---|---|---:|---:|---:|---:|---:|
| FEVER | Qwen2.5-1.5B-Instruct | required | BM25 Top-k | required | matched | matched signed-v1 |
| FEVER | Qwen2.5-7B-Instruct | required | BM25 Top-k | required | matched | matched signed-v1 |
| FEVER | non-Qwen instruct model | required | BM25 Top-k | required | matched | matched signed-v1 |
| FM2 | Qwen2.5-1.5B-Instruct | required | official-pool Top-k | required | matched | matched signed-v1 |
| FM2 | Qwen2.5-7B-Instruct | required | official-pool Top-k | required | matched | matched signed-v1 |
| FM2 | non-Qwen instruct model | required | official-pool Top-k | required | matched | matched signed-v1 |

“Matched” means posteriors, teacher, learned selector, selection, and final evaluation are associated with the same generator identity. A selector conditioned on one generator and evaluated with another belongs only in the transfer ablation in Section 17.

Every main-table cell should report at least Accuracy and Macro-F1. `src/metrics.py` implements both. The repository does not contain a dataset-specific official FM2 scorer, and the current FEVER evaluator is a binary classification evaluator rather than the official FEVER evidence-score pipeline. Therefore the source supports a common classification table, not a claim that the two datasets' official task metrics are identical.

## 4. Repository Inventory

### Dataset and protocol configuration

- FEVER: `configs/fever2_minimal.yaml`, `fever2_server_smoke.yaml`, `fever2_server_pilot.yaml`, `fever2_server_pilot_5000_500.yaml`, `fever2_server_formal.yaml`, `fever2_server_preformal_signed_v1.yaml`, and `fever3_minimal.yaml`.
- FM2: `configs/fm2_server_smoke.yaml` and `configs/fm2_server_preformal_signed_v1.yaml`.
- No source file for FactScore was found in the audited inventory.

### Data and retrieval

- FEVER preparation: `scripts/00_prepare_fever.py`, `scripts/01_prepare_fever_corpus.py`.
- FEVER formal split construction: `scripts/01a_build_fever_formal_splits.py`, `src/formal_splits.py`.
- Persistent BM25: `scripts/02a_build_bm25_index.py`, `scripts/02_retrieve_bm25.py`, `src/retrieval_bm25.py`.
- FM2 acquisition/adaptation: `scripts/download_fm2.py`, `scripts/prepare_fm2.py`, `src/datasets/fm2.py`.

### Posterior and evaluation

- Generator posterior production: `scripts/03_compute_label_posteriors.py`.
- Prompt registry: `src/prompts.py`.
- Next-token label scoring: `src/label_logits.py`.
- Final generator classification: `scripts/07_eval_rag_classification.py`.
- Metrics: `src/metrics.py`.

### Baselines

- No Evidence: evaluator `--no-evidence` and `scripts/preformal/27a_select_no_evidence.py`.
- Source-rank Top-k: `scripts/08_select_naive_topm.py`.
- BGE: `scripts/12_select_bge_reranker.py`, `src/baselines/bge_reranker.py`.
- InfoGain: `scripts/12a_build_infogain_teacher.py`, `12b_train_infogain_reranker.py`, `12c_select_infogain_reranker.py`, `src/baselines/infogain.py`, `src/baselines/infogain_fever.py`, `src/baselines/infogain_selector.py`.
- Shared selection publication: `src/baselines/common.py`, `src/selection_schema.py`.

### Ours and diagnostics

- Legacy CBWDM teacher/training/selection: scripts `04`, `05`, `06`, `10`, and `11`.
- signed-v1: `src/diagnostics/signed_teacher_v1.py`, `src/diagnostics/signed_selector_v1.py`, and preformal scripts `25`, `26`, `27`, `29`, `30`.
- signed-v2/v2.1/v2.2: diagnostic modules and tests under `src/diagnostics/`, `scripts/diagnostics/`, and `tests/test_signed_selector_v2*.py`.

### Formal control and provenance

- FEVER orchestrator: `scripts/run_fever_cbwdm.py` and shell wrapper.
- Calibration/freezing/readiness: scripts `15`, `15a`, `16`, `17`; `src/calibration/`, `src/formal_config.py`, `src/formal_readiness.py`.
- Atomic manifests/hashes: `src/run_manifest.py`, `src/formal_provenance.py`.
- signed-v1 preformal registry/policy: `src/preformal/registry.py`, `src/preformal/splits.py`, `src/preformal/fairness.py`.

### Tests and environments

- Dataset/baseline/formal/signed coverage is present in `tests/test_fm2_support.py`, `test_fever_baselines.py`, `test_fever_formal_protocol.py`, `test_preformal_signed_v1.py`, and signed diagnostic tests.
- Base dependencies are in `requirements.txt`; baseline Transformers bounds are in `requirements-baselines.txt`; Pyserini 2.3.0 is in `requirements-retrieval.txt`.
- `environment/server/` contains reconstruction documentation and captured environment files, but this audit does not validate a live server environment.

## 5. FEVER Readiness

Overall: **PARTIAL**.

| Capability | Status | Source finding |
|---|---|---|
| Raw loader | READY | FEVER preparation scripts exist. |
| Train/validation/test split | READY at source level | Formal split builder publishes `train_core`, `validation`, `held_out_test` with overlap/conflict checks. |
| Retrieval corpus | READY | Wikipedia sentence corpus preparation exists. |
| BM25 retrieval | READY | Persistent Pyserini/Lucene index and retrieval contracts exist. |
| BGE reranking | READY | Shared retrieval-schema reranker and selection publisher exist. |
| Posterior generation | READY | Batched next-token label posterior path exists with a sidecar manifest. |
| InfoGain teacher | READY for FEVER roles | `train`/`train_core` role handling and held-out rejection exist. |
| Ours teacher | READY for signed-v1 source path | signed-v1 teacher consumes generator posteriors and train labels. |
| Selector training | READY for signed-v1 preformal path | Seeds 13/21/42 and checksummed checkpoint manifest exist. |
| Selection | READY | signed-v1 inference is gold-free and publishes shared selection schema. |
| Final generator evaluation | READY per single configured model | Shared evaluator produces predictions and metrics. |
| Metrics | READY for classification | Accuracy, Macro-F1, per-class and evidence-budget diagnostics; no official FEVER evidence score. |
| Formal config | PARTIAL | FEVER formal config exists, but is Qwen-7B/legacy-`rag_cbwdm` oriented; signed-v1 has a preformal config, not a complete multi-generator formal config. |
| Manifest/checksum/fingerprint | READY within individual stages | Strong stage-level support, but cross-stage generator binding is inconsistent. |

Missing for the target matrix:

- generator-qualified formal paths and a stable generator ID;
- a signed-v1-aware formal freezing/readiness registry;
- matched per-generator InfoGain/Ours orchestration;
- a single formal config/manifest contract for each generator row;
- runtime evidence that the held-out artifact remains unconsumed;
- non-Qwen compatibility validation.

## 6. FM2 Readiness

Overall: **PARTIAL**.

| Capability | Status | Source finding |
|---|---|---|
| Raw loader | READY | `src/datasets/fm2.py` validates canonical schema, SHA-256 and row counts. |
| Train/validation/test split | READY at source level | Official `train`, `dev`, `test`; page disjointness is audited; test must be explicitly requested. |
| Retrieval corpus | PARTIAL | Official `retrieved_evidence` closed-page pool exists; no compatible open-domain corpus snapshot exists. |
| BM25 retrieval | MISSING for FM2 | FEVER Lucene corpus/index is not proven compatible; FM2 config correctly uses `fm2_official_closed_page_v1`. |
| BGE reranking | READY for official pool | Generic query/candidate schema is supported. |
| Posterior generation | READY | Shared prompt registry and posterior script support `dataset=fm2`. |
| InfoGain teacher | READY at component level | Generic adapter and FM2 tests exist; parameters are explicitly cross-dataset reference values, not an FM2 optimum. |
| Ours teacher | READY at component level | signed-v1 training split can be `train`; shared posterior schema is consumed. |
| Selector training | READY at component level | signed-v1 and InfoGain paths accept FM2-compatible roles/schema. |
| Selection | READY at component level | No Evidence, source-rank Top-k, BGE, InfoGain and signed-v1 operate on the shared schema. |
| Final generator evaluation | READY per single configured model | Shared evaluator supports FM2 prompt contract. |
| Metrics | READY for classification | Same Accuracy/Macro-F1 implementation; no separate official FM2 metric adapter was found. |
| Formal config | PARTIAL | Smoke and preformal signed-v1 configs exist; no frozen multi-generator FM2 formal config/readiness gate/orchestrator exists. |
| Manifest/checksum/fingerprint | PARTIAL | Preparation and stages are checksummed; no end-to-end FM2 formal manifest binds all stages and generator identity. |

Protocol boundary: the official FM2 pool is gold-free with respect to selector payload, but is closed-page and sourced from `retrieved_evidence`. Its Top-k baseline must be labeled **official-pool/source-order Top-k**, not BM25. A later FM2 BM25 result requires a separately versioned corpus, document-ID contract, index, and retrieval manifest.

## 7. Generator Abstraction Audit

### What `--model-name` actually reaches

- Posterior computation: yes; `scripts/03_compute_label_posteriors.py` overrides `generator.model_name` and accepts separate model/tokenizer revisions.
- Query-only posterior: yes; `eta0` is scored by the same `LabelLogitScorer` without evidence.
- Query+evidence posterior: yes; every candidate prompt uses the same scorer/model.
- InfoGain: indirectly. Its teacher consumes already-produced posteriors, so generator identity must flow through posterior provenance; the reranker itself is a separate encoder.
- Ours: indirectly. signed-v1 teacher consumes generator-specific `eta0`/`eta_j`; its selector is a separate encoder.
- Final evaluation: yes for model name, but revision/tokenizer revision still come from the config rather than separate CLI overrides.

### Actual inference abstraction

`src/label_logits.py` loads `AutoTokenizer` and `AutoModelForCausalLM`. It does not call `generate()` and does not use `apply_chat_template()`. It scores next-token logits for configured label verbalizers and softmaxes the per-label aggregate. Therefore temperature, `do_sample`, `top_p`, `top_k`, and `max_new_tokens` do not apply to the current classifier. Both posterior and final evaluation are algorithmically deterministic forward passes, subject to ordinary device/kernel numerical nondeterminism.

### Qwen2.5 readiness

- Qwen2.5-1.5B-Instruct: **PARTIAL**. It is the configured development generator and the code path supports it, but this source-only audit does not prove a current formal runtime artifact.
- Qwen2.5-7B-Instruct: **PARTIAL**. The FEVER formal config names it and the same causal-LM interface applies, but no live model, throughput, memory, or completed artifact is verified here.

### Non-Qwen HF causal/instruct readiness

Status: **PARTIAL**. The generic HF loader is a useful foundation, but compatibility is not automatic:

- prompts are plain text, not model-family chat templates;
- configured `A`/`B` variants must include at least one single-token verbalizer for the chosen tokenizer; multi-token variants are skipped and all-multi-token labels fail;
- prompt semantics may be poor for models trained to require a chat wrapper;
- no explicit `padding_side` is set; the attention-mask logic locates the final non-padding position, but model-family behavior still needs a test;
- absent pad token is replaced with EOS; a tokenizer with neither fails;
- `trust_remote_code`, dtype, device map, context length, model revision, and tokenizer revision must be frozen per model;
- no family-specific output parser is needed because there is no free-form generation, but next-token label calibration remains model/tokenizer-specific;
- only `AutoModelForCausalLM`-compatible models are supported; encoder-decoder or API-only generators are not;
- local model paths are deployment details and must not be used as experiment identities.

## 8. Posterior Provenance Audit

Authoritative production path: `scripts/03_compute_label_posteriors.py` reads a retrieval JSONL, builds the dataset-specific query-only and query+evidence prompts, calls `LabelLogitScorer`, and writes `eta0` plus candidate `eta` rows.

The completed sidecar manifest records:

- dataset and split;
- generator model name/path and a generator SHA (local directory hash or stable hash of remote model name/revision);
- requested and resolved model commit where available;
- tokenizer name/revision and resolved tokenizer commit where available;
- dtype, device map, and `trust_remote_code`;
- prompt template version/hash;
- labels, verbalizers, and verbalizer hash;
- context length, batch size, candidate limit;
- source retrieval path/SHA;
- config path/SHA;
- stage fingerprint, output SHA, Git state, and environment.

Temperature, sampling configuration, decoding seed, and `max_new_tokens` are absent because the implementation does not decode or sample. Their absence is not itself a reproducibility defect for this next-token-logit method.

Critical limitation: posterior JSONL rows contain schema, ID, query, label, split, labels, `eta0`, and candidate `eta`, but no generator identity or manifest fingerprint. A JSONL file separated from its sidecar cannot reliably identify its generator. The sidecar is strong, but downstream consumers do not uniformly require it.

The default path `outputs/posteriors/<dataset>_<split>_posteriors.jsonl` and the formal runner path `artifacts/formal/<dataset>_<role>_posteriors.jsonl` omit generator identity. Reusing one run directory for another generator can overwrite or ambiguously reuse the same logical location. The run directory fingerprint detects some incompatible resumes, but path-level separation is still inadequate for a matrix and does not protect copied/orphaned files.

Verdict: **PARTIAL / HIGH-PRIORITY PROVENANCE GAP**. Provenance is reliable only when the completed sidecar stays attached and is actively validated. It is not reliable from the posterior JSONL alone.

Required formal invariant:

```text
posterior JSONL SHA
  -> completed posterior manifest
  -> dataset_id + split + generator_id
  -> immutable model/tokenizer identity + prompt/verbalizer contract
  -> retrieval SHA + config SHA + Git commit
```

InfoGain-specific gap: `12a_build_infogain_teacher.py` records posterior SHA and accepts generator/prompt/verbalizer fields as manual CLI metadata, but it does not automatically read and validate the posterior sidecar. signed-v1 records posterior SHA in the teacher contract but likewise does not bind a required posterior manifest identity.

## 9. Generator-Dependency Matrix

| Method | Generator-specific posterior? | Generator-specific teacher? | Generator-specific selector training? | Selection reusable across generators? | Generator-specific final eval? |
|---|---:|---:|---:|---|---:|
| No Evidence | No | No | No | Yes; empty selection | Yes |
| FEVER BM25/source-rank Top-k | No | No | No | Yes, if retrieval/corpus protocol is identical | Yes |
| FM2 official-pool Top-k | No | No | No | Yes, if candidate-pool SHA is identical | Yes |
| BGE | No | No | No | Yes, if retrieval and BGE contract are identical | Yes |
| InfoGain | **Yes** | **Yes** | **Yes** | Not for matched main results; only transfer ablation | Yes |
| signed-v1 Ours | **Yes** | **Yes** | **Yes** | Not for matched main results; only transfer ablation | Yes |
| legacy `rag_cbwdm` | **Yes** | **Yes** | **Yes** | Not for matched results | Yes |
| Oracle/signed-gate oracle | Yes | Yes, gold-dependent | No deployable claim | Diagnostic only | If evaluated, diagnostic only |

For InfoGain, the teacher's document information gain uses the gold-label posterior difference, so changing generator posteriors changes supervision. For signed-v1, `eta0`, candidate `eta_j`, signed alignment, admissibility, effective gain, Theta trajectory, and teacher supervision all derive from generator posteriors. The source therefore confirms the generator-conditioned interpretation.

## 10. Baseline Readiness

### No Evidence — READY at component level

`scripts/07_eval_rag_classification.py --no-evidence` directly evaluates the query-only prompt. A schema-normalized empty selection path also exists in `scripts/preformal/27a_select_no_evidence.py`. It must be evaluated separately with every generator.

### Vanilla RAG / retrieval Top-k — dataset-dependent

- FEVER: **READY**. BM25 retrieval produces ranked candidates; `scripts/08_select_naive_topm.py` preserves the first Top-k; the evaluator concatenates selected evidence into the generator prompt. This is a valid BM25 Top-k Vanilla RAG baseline.
- FM2: **PARTIAL relative to the requested name**. The same selector consumes official source order, not BM25. It is a valid official-pool Top-k RAG baseline, but not an FM2 BM25 baseline.
- The Naive selector currently writes metadata named `bm25_source_rank`, which is misleading for FM2 and should be generalized without changing historical artifacts.

### BGE — READY at component level, formal identity PARTIAL

The code loads a Hugging Face sequence-classification reranker, caches scores, then publishes Top-k selections. Both `BAAI/bge-reranker-base` (smoke/history) and `BAAI/bge-reranker-large` (pilot/formal/FM2 configs) are supported by the same interface. This audit does not choose between them; the exact model/revision/hash is a human freeze decision.

BGE selection is generator-independent. Its score-cache contract includes retrieval SHA, model string, revision, dtype, length, normalization, template and limit. The score-cache manifest does not have the same full model-weight SHA/Git/environment coverage as the strongest formal manifests, so it should be strengthened before final publication.

### InfoGain — PARTIAL for multi-generator formal use

The teacher, multitask reranker training, checkpoint save/load, gold-free selection, and final evaluation paths all exist. Training manifests bind teacher SHA/fingerprint, hyperparameters, model string/revision, and seed. Selection binds actual checkpoint config/head/encoder files.

Gaps are automatic generator-provenance validation, generator-qualified paths, full checkpoint identity in the training manifest, and a multi-generator orchestrator. FM2 uses an adapter and explicitly non-optimal cross-dataset reference thresholds; that is suitable for engineering smoke/preformal comparison, not an undocumented FM2-tuned claim.

### Ours — PARTIAL for formal matrix, complete component chain

The intended chain is:

```text
generator-specific retrieval posteriors
  -> signed-v1 gold-dependent training teacher
  -> signed-v1 selector training (seed 13/21/42)
  -> gold-free signed-v1 selection
  -> matching generator evaluation
  -> Accuracy/Macro-F1
```

The source implements each component. The missing layer is a frozen, collision-safe, multi-dataset/multi-generator formal contract that registers signed-v1 as Ours. signed-v2/v2.1/v2.2 remain outside the default main table.

## 11. Artifact Collision Risks

Risk level: **HIGH** for a multi-generator matrix.

| Artifact | Current collision/ambiguity |
|---|---|
| Posterior | Default and formal filenames encode dataset/split but not generator. |
| Teacher | Generic names such as `teacher.jsonl` or `<dataset>_<split>_teacher.jsonl` rely on the containing run directory. |
| InfoGain checkpoint | Generic `checkpoint/`; generator-conditioning is only indirect through teacher fingerprint. |
| signed-v1 checkpoint | Generic `checkpoint/`; seed may be in the chosen parent path but is not enforced by a global hierarchy. |
| Selection | Common names encode method/Top-k, but usually not generator-conditioning source or evaluation generator. |
| Predictions/metrics | Current run-level filenames generally omit stable generator ID. |
| BGE score cache | Can be safe by fingerprint on resume, but a shared path remains awkward across model variants. |
| Formal summaries | Current FEVER summary/readiness assumes one configured generator and older canonical method registry. |

Dataset and method are often recoverable from a run directory or manifest; generator and seed are not consistently encoded in every path. Git commit and config fingerprint are manifest fields rather than directory dimensions, which is acceptable if manifests are mandatory. They should not be added as long directory names unless needed for immutable snapshots.

## 12. Recommended Artifact Hierarchy

A minimal backward-compatible structure is to leave historical runs untouched and introduce a new formal root:

```text
artifacts/formal_v2/
  <dataset_id>/
    shared/
      dataset/
      retrieval/<retrieval_protocol_id>/
      bge/<bge_model_id>/
    <generator_id>/
      generator.manifest.json
      posteriors/<split>/posteriors.jsonl
      no_evidence/<split>/predictions.jsonl
      retrieval_topk/<split>/selection.jsonl
      bge/<split>/selection.jsonl
      infogain/
        teacher/
        seed13/{checkpoint,selection,eval}/
        seed21/{checkpoint,selection,eval}/
        seed42/{checkpoint,selection,eval}/
      ours_signed_v1/
        teacher/
        seed13/{checkpoint,selection,eval}/
        seed21/{checkpoint,selection,eval}/
        seed42/{checkpoint,selection,eval}/
      summaries/
```

Generator-independent selection may live below `shared/`, but generator-specific evaluation outputs must live below `<generator_id>`. For clarity, a small link/reference manifest under each generator may point to shared retrieval/BGE selections by SHA rather than copy them. Existing `outputs/runs/<run_name>` behavior can remain supported; the new orchestrator should generate unambiguous run names and refuse a generator/dataset mismatch.

## 13. Generator ID Contract

Recommended immutable logical fields:

```yaml
generator_id: qwen2.5-1.5b-instruct
model_family: qwen2.5
model_path: /root/models/Qwen2.5-1.5B-Instruct
model_revision: <resolved revision>
model_sha256: <weights/tree hash>
tokenizer_revision: <resolved revision>
tokenizer_sha256: <tokenizer/tree hash>
prompt_contract: <version and hash>
verbalizer_hash: <hash>
```

`generator_id` is a stable experiment label, not a path. `model_family` selects compatibility behavior/tests. `model_path` is a deployment-specific locator. Repointing a stable ID to different weights must fail unless the immutable revision/hash contract also changes. The third model should remain an undecided non-Qwen ID until its family, exact checkpoint, license/deployment constraints, and label-token test are approved.

## 14. Dataset ID Contract

Current naming mixes “FEVER” as the dataset family and `fever2` as the binary pipeline/config ID. Historical artifacts should not be renamed.

Recommended mapping:

| Stable `dataset_id` | Meaning | Existing config label |
|---|---|---|
| `fever_binary_v2` | Current SUPPORTS/REFUTES pipeline with NEI dropped and formal split contract | `fever2` |
| `fm2_official_closed_page_v1` | FM2 official train/dev/test plus official retrieved-evidence pool | `fm2` + retrieval protocol |

If compatibility requires retaining `fever2` and `fm2`, add an explicit alias table to the formal manifest rather than silently changing old rows. Dataset identity must include split-manifest/candidate-pool SHA; a short name alone is insufficient.

## 15. Seed and Determinism Policy

- Learned selectors, InfoGain and signed-v1 Ours: run seeds **13, 21, 42** per dataset × generator. Record Python, NumPy, Torch, CUDA seed/determinism settings and checkpoint SHA.
- No Evidence: one selection artifact; evaluate once per generator.
- FEVER BM25/source-rank Top-k: one retrieval/selection artifact per frozen retrieval contract; evaluate once per generator.
- FM2 official-pool Top-k: one selection artifact per candidate-pool SHA; evaluate once per generator.
- BGE: model is in eval/inference mode and selection is deterministic under the frozen backend; use one selection artifact per BGE contract, then evaluate once per generator. Do not create artificial three-seed duplicates.
- Posterior and final evaluation: no generation sampling is used. Record numerical backend/environment and output hashes; do not invent decoding seeds.
- Learned-training stochasticity exists through initialization and shuffled training order. `scripts/preformal/26_train_signed_v1.py` seeds Python, NumPy, Torch and CUDA and uses seed-specific shuffling. InfoGain seeds Python and Torch; its formal policy should explicitly cover CUDA and any deterministic-kernel choice.

No formal blocker arises from `temperature`, `do_sample`, `top_p`, `top_k`, or `max_new_tokens`, because the current path never invokes free-form decoding.

## 16. Split / Contamination Audit

### FEVER

The 500-example validation used throughout signed-v1/v2/v2.1/v2.2 development is development data and must not be called an untouched formal test set.

The repository contains formal split machinery producing `train_core`, `validation`, and `held_out_test`. The preformal Strategy A path subtracts the pilot validation from the larger validation source, audits overlaps, and forbids held-out use. Source code therefore provides a possible untouched held-out protocol. This source-only audit cannot prove that a server copy of `held_out_test` has never been consumed; that claim requires the frozen split manifest, run manifests, access history/provenance, and a human sign-off before opening it.

### FM2

Official train/dev/test are source-supported, their canonical files have pinned hashes/row counts, and Wikipedia-page disjointness is enforced. Test is not prepared by default and configs set test consumption to forbidden until parameter freeze. The intended policy is train for learning, dev for development/calibration, test once after freeze.

Again, source gates cannot establish the historical untouched state of a server file. The formal launch must record that all generator IDs, model hashes, prompts, verbalizers, Top-k budgets, thresholds, seeds, and candidate-pool SHA were frozen before test preparation/evaluation.

Formal split status: **PARTIAL—protocols exist, untouched runtime state is unverified and the FEVER development validation is contaminated for final-test claims**.

## 17. Cross-Generator Transfer Ablation

Optional ablation:

```text
posterior/teacher/selector conditioned on generator A
  -> selection_A
  -> final evaluation with generator B
```

This measures selector transfer/robustness. It cannot replace the matched main-table result for generator B.

The present source can mechanically perform the final evaluator override, but the current artifact layout can confuse conditioning generator with evaluation generator. A transfer artifact must record both:

```yaml
conditioning_generator_id: qwen2.5-1.5b-instruct
evaluation_generator_id: qwen2.5-7b-instruct
experiment_type: cross_generator_transfer
selection_sha256: ...
conditioning_posterior_manifest_sha256: ...
```

Main results must require `conditioning_generator_id == evaluation_generator_id` for InfoGain and Ours. Transfer outputs should be stored in a separate `transfer/<source>_to_<target>/` subtree and excluded from main-summary discovery by default.

## 18. Execution DAGs

### Shared dataset stages

```text
raw dataset
  -> canonical split manifest
  -> candidate protocol
       FEVER: corpus -> Lucene/BM25 index -> retrieval
       FM2: canonical official closed-page pool
```

### Ours, for one dataset × generator

```text
retrieval/candidate pool
  -> generator-matched eta0 and eta_j posteriors
  -> signed-v1 teacher on training split
  -> signed-v1 selector training [seed]
  -> gold-free selection on evaluation split
  -> same-generator final label-logit evaluation
  -> Accuracy + Macro-F1 + diagnostics
```

### InfoGain, for one dataset × generator

```text
retrieval/candidate pool
  -> generator-matched eta0 and eta_j posteriors
  -> generator-conditioned InfoGain teacher on training split
  -> InfoGain reranker training [seed]
  -> gold-free selection on evaluation split
  -> same-generator final label-logit evaluation
  -> Accuracy + Macro-F1 + diagnostics
```

### Generator-independent selection methods

```text
retrieval/candidate pool
  +-> empty selection ------------------------------+
  +-> source-rank Top-k ----------------------------+-> per-generator evaluation -> metrics
  +-> BGE score cache -> BGE Top-k -----------------+
```

The retrieval/source-order and BGE selection artifacts may be shared across generators only when their upstream SHA and selection contracts are identical. Evaluation can never be shared across generators.

## 19. Formal Blockers

### P0 — must be closed before formal matrix execution

1. **Generator identity and collisions.** Posterior/evaluation paths and row payloads do not enforce a stable generator identity. Add a generator manifest and collision-safe roots; require sidecar validation.
2. **Matched conditioning enforcement.** InfoGain and signed-v1 downstream stages bind file SHAs but do not uniformly resolve and compare the upstream posterior generator contract. Fail closed on mismatches.
3. **Formal Ours registry.** FEVER freezing/readiness still treats legacy `rag_cbwdm` as canonical; signed-v1 is only preformal. Freeze provisional Ours explicitly without changing its mathematics.
4. **Revision override correctness.** Final evaluation permits model-name override while taking revisions from the base config. Make the generator contract atomic so model path, model/tokenizer revisions/hashes, prompt, dtype, and trust settings cannot be mixed.
5. **Split freeze.** Prove and sign off untouched FEVER held-out and FM2 test status; never use the development 500 as final test.
6. **FM2 table protocol decision.** Either label the main cell official-pool Top-k or implement/freeze a separate FM2 BM25 corpus/index. Do not report the current path as BM25.

### P1 — needed before broad multi-generator execution

1. Add non-Qwen family compatibility tests for prompt bytes, label tokenization, padding/EOS behavior, and next-token scoring.
2. Add a dataset/generator-aware orchestrator and summary that cannot discover transfer/diagnostic artifacts as matched main results.
3. Strengthen InfoGain teacher auto-provenance, training checkpoint hashing, and BGE model/cache provenance.
4. Add an FM2 formal freeze/readiness gate parallel to FEVER, while preserving the distinct candidate protocol.
5. Freeze the exact BGE variant and all encoder revisions/hashes.
6. Record full deterministic-training settings and environment identity for all learned seeds.

### P2 — cleanup/convenience

1. Generalize FEVER/BM25-specific script descriptions and selection metadata where the implementation is now dataset-generic.
2. Add manifest schema helpers to reduce repeated validation code.
3. Add human-readable matrix summaries and artifact-index files.
4. Keep historical artifact names readable through aliases; do not rename old outputs.

## 20. Minimal Refactor Plan

| Priority | Problem | Minimal proposed fix | Risk | Backward compatibility |
|---|---|---|---|---|
| P0 | No stable generator/dataset identity | Add a versioned experiment identity object and use it in stage contracts/paths | Medium | New formal root; old paths remain readable |
| P0 | Posterior sidecar not mandatory downstream | Shared loader validates JSONL SHA, manifest fingerprint, generator/dataset/split | Low-Medium | Allow legacy only outside formal profile |
| P0 | Evaluator override can mix revisions | Accept/resolve an atomic generator manifest instead of independent name override | Low | Keep CLI alias with deprecation outside formal mode |
| P0 | signed-v1 absent from formal registry | Add signed-v1 as frozen Ours in a new registry version | Medium | Do not mutate legacy formal manifests |
| P0 | FM2 retrieval naming ambiguity | Introduce `retrieval_protocol_id` and method display label | Low | Preserve historical method field plus alias metadata |
| P1 | Non-Qwen unproven | Add model-family prompt adapter contract and tiny tokenizer/model compatibility tests | Medium | Default plain prompt remains Qwen-compatible |
| P1 | Per-dataset orchestration fragmented | Add a thin generic matrix orchestrator over existing scripts | Medium | Existing scripts remain authoritative workers |
| P1 | Incomplete baseline provenance | Hash BGE weights/runtime; auto-link InfoGain posterior sidecar; hash InfoGain checkpoint tree | Low-Medium | Manifest schema version bump |
| P2 | FEVER-specific names in generic paths | Add neutral aliases/documentation | Low | Retain old import and method aliases |

No method equation, threshold, lambda, gamma, or signed-v1 training objective needs to change for this plan.

## 21. Recommended Execution Order

1. **Phase A — identity/provenance:** freeze `dataset_id`, `generator_id`, retrieval protocol, model/tokenizer hashes, and collision-safe hierarchy.
2. **Phase B — formal contract wiring:** mandatory posterior-sidecar validation, atomic evaluation generator contract, signed-v1 formal registry, transfer exclusion.
3. **Phase C — compatibility:** smoke the existing Qwen 1.5B contract, Qwen 7B contract, then one chosen non-Qwen family using tiny bounded inputs only after code review.
4. **Phase D — FM2 protocol:** decide official-pool Top-k versus a separately engineered BM25 protocol; add FM2 formal freeze/readiness.
5. **Phase E — one complete combination:** FEVER × Qwen1.5B × all five methods in a new non-held-out smoke/preformal root; verify manifests and summaries.
6. **Phase F — multi-generator FEVER:** run matched posteriors/teachers/selectors for learned methods and reuse only generator-independent selections.
7. **Phase G — FM2:** repeat the matched matrix under the explicitly named FM2 retrieval protocol.
8. **Phase H — formal seeds and held-out:** after parameter freeze and audit sign-off, train seeds 13/21/42 for learned methods and open each held-out split once.
9. **Optional Phase I — transfer:** run A→B selector transfer in its isolated subtree after matched main results are complete.

## 22. Exact Files That Would Need Modification

This audit changed none of the following. A minimal future implementation would likely modify or add:

### Existing source files

- `scripts/03_compute_label_posteriors.py` — emit/link stable dataset/generator IDs and expose a reusable manifest validator.
- `scripts/07_eval_rag_classification.py` — consume an atomic generator contract and record conditioning/evaluation identities.
- `scripts/08_select_naive_topm.py` — replace unconditional BM25-specific metadata with retrieval-protocol metadata.
- `scripts/12_select_bge_reranker.py` — strengthen model/cache identity.
- `scripts/12a_build_infogain_teacher.py` — automatically load and validate posterior manifest provenance.
- `scripts/12b_train_infogain_reranker.py` — publish full checkpoint tree SHA and deterministic-runtime contract.
- `scripts/preformal/25_materialize_signed_v1_teacher.py` — validate the posterior sidecar generator/dataset contract.
- `scripts/preformal/26_train_signed_v1.py` — carry generator/dataset identity into training/checkpoint manifests.
- `scripts/preformal/27_select_signed_v1.py` — carry conditioning generator ID and enforce main/transfer mode.
- `scripts/run_fever_cbwdm.py` — either generalize formal path construction or delegate to a new matrix orchestrator.
- `src/run_manifest.py` and/or `src/formal_provenance.py` — shared versioned identity/manifest validation helpers.
- `src/formal_config.py`, `src/formal_readiness.py` — versioned signed-v1 and multi-generator formal contracts.
- `src/preformal/registry.py` — factor reusable method/seed definitions from FEVER-specific preformal policy.
- `src/prompts.py`, `src/label_logits.py` — only if the selected non-Qwen family requires a versioned chat/prompt adapter; do not change existing prompt bytes silently.

### New recommended files

- `src/experiment_identity.py` — stable dataset/generator/retrieval identity schemas.
- `scripts/run_formal_matrix.py` — thin dataset × generator × method DAG orchestrator.
- generator-specific formal config files under `configs/formal/` rather than overwriting historical configs.
- an FM2 formal freeze/readiness entrypoint or a dataset-generic successor to scripts `16`/`17`.

### Tests requiring extension

- `tests/test_label_logits_and_manifest.py`
- `tests/test_fever_formal_protocol.py`
- `tests/test_fever_baselines.py`
- `tests/test_fm2_support.py`
- `tests/test_preformal_signed_v1.py`
- new tests for experiment identity, collision refusal, matched-vs-transfer enforcement, and one non-Qwen tokenizer contract.

The exact future change set depends on the human decisions in Section 23; this list is an implementation forecast, not an authorization to edit these files.

## 23. Questions Requiring Human Decision

1. Which exact non-Qwen causal/instruct checkpoint, revision, license, and deployment path will be frozen?
2. Must that family use its native chat template, or is the current plain classification prompt the controlled cross-model protocol?
3. For FM2, will the main-table retrieval column be named “official-pool Top-k,” or is a new open-domain BM25 protocol required?
4. Which BGE checkpoint is final: base or large? What exact revision/hash is accepted?
5. Is `fever2` retained as the stable public dataset ID, or mapped to a more explicit `fever_binary_v2` alias in new manifests?
6. What evidence and sign-off establish that FEVER `held_out_test` and FM2 `test` remain untouched on the server?
7. Should the common primary endpoints be Accuracy and Macro-F1 only, with official dataset metrics reported separately when available?
8. Is one generator-independent BGE selection shared across generator rows, or duplicated as references under each generator root for operational simplicity?
9. Are cross-generator transfer runs excluded from main summaries by separate root, explicit manifest flag, or both?

### Existing reusable artifacts by evidence class

- **SOURCE-CONFIRMED:** FEVER loaders/splits/corpus/BM25; FM2 canonical adapter and official pool; posterior/evaluator code; No Evidence, source-rank Top-k, BGE, InfoGain, signed-v1 components; shared metrics; manifest/hash helpers; configs and tests listed above.
- **SERVER-RUNTIME-ONLY:** any artifact/result described in runbooks or historical reports as produced on `/root/...`; those documents are evidence of intended/reported execution, not direct artifact verification in this audit.
- **UNKNOWN:** current existence, completeness, generator origin, or untouched state of any live server posterior, checkpoint, held-out split, prediction, metric, or `/root/experiments` artifact.

## HANDOFF_TO_CHATGPT

STATUS: PARTIAL_FORMAL_READINESS
GIT_HEAD: a0a0bd90e1d81caad6e7636c5eba7231b724eff1
PROVISIONAL_OURS: rag_cbwdm_signed_v1 deployable selector
DEVELOPMENT_DATASET: FEVER binary pipeline (config dataset=fever2)
DEVELOPMENT_GENERATOR: Qwen2.5-1.5B-Instruct

FEVER_READINESS: PARTIAL
FM2_READINESS: PARTIAL

QWEN15_READINESS: PARTIAL_SOURCE_SUPPORTED_RUNTIME_NOT_VERIFIED
QWEN7_READINESS: PARTIAL_SOURCE_SUPPORTED_RUNTIME_NOT_VERIFIED
NON_QWEN_READINESS: PARTIAL_GENERIC_HF_CAUSAL_LM_WITH_UNPROVEN_PROMPT_TOKENIZER_COMPATIBILITY

POSTERIOR_PROVENANCE_STATUS: PARTIAL_HIGH_PRIORITY_SIDECAR_STRONG_JSONL_AND_PATH_NOT_GENERATOR_SELF_IDENTIFYING
GENERATOR_ID_STATUS: MISSING_STABLE_CONTRACT
DATASET_ID_STATUS: PARTIAL_ALIAS_AND_PROTOCOL_CONTRACT_NEEDED
ARTIFACT_COLLISION_RISK: HIGH
FORMAL_SPLIT_STATUS: PARTIAL_DEVELOPMENT_VALIDATION_NOT_UNTOUCHED_HELDOUT_RUNTIME_STATE_UNVERIFIED

NO_EVIDENCE_STATUS: READY_COMPONENT_PER_GENERATOR_EVAL_REQUIRED
BM25_RAG_STATUS: FEVER_READY_FM2_BM25_MISSING_OFFICIAL_POOL_TOPK_AVAILABLE
BGE_STATUS: READY_COMPONENT_FORMAL_MODEL_IDENTITY_PARTIAL
INFOGAIN_STATUS: PARTIAL_COMPLETE_COMPONENT_CHAIN_NEEDS_GENERATOR_PROVENANCE_ENFORCEMENT
OURS_STATUS: PARTIAL_SIGNED_V1_CHAIN_EXISTS_FORMAL_MULTI_GENERATOR_REGISTRY_MISSING

P0_BLOCKERS: stable generator/dataset identity and collision-safe paths; mandatory posterior-manifest binding; matched-conditioning enforcement; signed-v1 formal registry; atomic evaluation revision contract; held-out freeze proof; FM2 retrieval protocol decision
P1_ITEMS: non-Qwen compatibility tests; generic matrix orchestrator; FM2 formal gate; stronger BGE/InfoGain provenance; frozen encoder variants and deterministic runtime
P2_ITEMS: neutral naming aliases; shared manifest helpers; artifact index and human-readable matrix summaries

FILES_THAT_WOULD_CHANGE: scripts/03_compute_label_posteriors.py; scripts/07_eval_rag_classification.py; scripts/08_select_naive_topm.py; scripts/12_select_bge_reranker.py; scripts/12a_build_infogain_teacher.py; scripts/12b_train_infogain_reranker.py; scripts/preformal/25_materialize_signed_v1_teacher.py; scripts/preformal/26_train_signed_v1.py; scripts/preformal/27_select_signed_v1.py; scripts/run_fever_cbwdm.py; src/run_manifest.py; src/formal_provenance.py; src/formal_config.py; src/formal_readiness.py; src/preformal/registry.py; conditional src/prompts.py and src/label_logits.py; relevant configs/tests; proposed src/experiment_identity.py and scripts/run_formal_matrix.py
RECOMMENDED_FIRST_IMPLEMENTATION_STEP: define and enforce stable dataset_id plus generator_id manifests and generator-qualified artifact roots, then require downstream posterior-sidecar validation

REPORT_PATH: C:\Users\wenhao\Desktop\CBWDM\rag_cbwdm\MULTI_GENERATOR_MULTI_DATASET_READINESS_AUDIT.md
CODE_CHANGED: NO
CONFIG_CHANGED: NO
TEST_CHANGED: NO
ALLOWLIST_UNCHANGED: YES

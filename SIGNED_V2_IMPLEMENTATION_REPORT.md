# Signed-v2 Dual-Head Implementation Report

## 1. Scope

This change implements the isolated development method `rag_cbwdm_signed_v2`. It addresses the signed-v1 factorization failure by assigning direction/admissibility and CBWDM utility to separate scalar logits while sharing one pretrained cross-encoder backbone.

The implementation is local-only. No server experiment, model download, data download, FEVER inference, training run, preformal run, or held-out evaluation was performed. The signed-v1 teacher mathematics and artifacts remain authoritative and are reused as inputs.

The following are deliberately unchanged:

- `src/cbwdm_score.py`, including `build_local_effects`, paper-mixture smoothing, Theta, and marginal gain;
- every signed-v1 source file, training contract, checkpoint contract, inference rule, and post-hoc rule;
- `src/preformal/registry.py` and every formal/preformal method registry;
- InfoGain, BGE, generator, FEVER data, and split implementations.

## 2. Files Added / Modified

Added:

- `src/diagnostics/signed_v2_model.py`
- `src/diagnostics/signed_selector_v2.py`
- `scripts/diagnostics/train_signed_selector_v2.py`
- `scripts/diagnostics/select_signed_selector_v2.py`
- `scripts/diagnostics/analyze_signed_selector_v2.py`
- `tests/test_signed_selector_v2.py`
- `SIGNED_V2_IMPLEMENTATION_REPORT.md`

Modified existing files: none.

No shared core change was needed. The v2-only modules reuse `build_selector_input`, `build_signed_training_groups`, `cbwdm_multitask_loss`, selection schema helpers, artifact publication, hashing, and diagnostic path enforcement without changing their APIs or defaults.

## 3. Architecture

### Shared backbone

`SignedV2DualHeadSelector` loads one Hugging Face `AutoModel` encoder from the supplied local MiniLM cross-encoder checkpoint. The tokenizer is loaded through the existing `AutoTokenizer` path. State-aware text is built by the existing signed-v1 `src.selector_cross_encoder.build_selector_input`, preserving its exact claim/candidate/already-selected ordering, truncation bounds, and gold-free content.

The shared representation is the encoder `pooler_output` when available, otherwise the first-token representation `last_hidden_state[:, 0]`. One shared dropout is applied before both heads. The implementation never creates two encoder instances.

### Gate head

`gate_head = Linear(hidden_size, 1)` produces `gate_logit(q, S, z)`. It predicts whether the authoritative signed-teacher alignment is strictly above `alignment_eps`.

### Utility head

`utility_head = Linear(hidden_size, 1)` produces `utility_logit(q, S, z)`. It is trained only on authoritative-admissible candidates and retains signed-v1 CBWDM gain/BCE/ranking semantics within that masked space.

### Initialization

The encoder is loaded through `AutoModel.from_pretrained`; pretrained encoder weights are not reinitialized. Both new heads are separate parameter tensors and are deterministically initialized under `head_seed` with `Normal(0, encoder.config.initializer_range)` weights and zero biases. The default head seed is the fixed experiment seed 13.

The original sequence-classification scalar head is not reused as the utility head. Robust extraction and semantic identification of that classifier differs across supported Transformer architectures; silently guessing a classifier attribute would weaken checkpoint safety. The implementation instead retains all pretrained encoder weights and gives both semantically new heads the same explicit deterministic initialization.

### Checkpoint schema

The checkpoint is explicitly distinct from signed-v1:

```text
checkpoint/
  encoder/                 # encoder.save_pretrained
  tokenizer/               # tokenizer.save_pretrained
  dual_heads.pt            # separate gate_head and utility_head states
  signed_v2_config.json    # method, architecture, initialization, training contract
```

Identifiers are:

```text
method = rag_cbwdm_signed_v2
architecture = rag_cbwdm_signed_v2_dual_head_v1
checkpoint schema = rag_cbwdm_signed_v2_checkpoint.v1
training manifest schema = rag_cbwdm_signed_v2_training_manifest.v1
```

Loading rejects missing files, a non-v2 method, a non-v2 schema, or an unsupported architecture. Selection additionally checks the full checkpoint tree SHA256 against the completed v2 training manifest.

## 4. Training Targets

For every remaining candidate in every retained signed-v1 teacher state:

```text
gate_target_j = 1  iff  authoritative_alignment_j > alignment_eps
gate_target_j = 0  otherwise
```

The authoritative alignment is recomputed directly from the posterior row and signed-v1 teacher parameters via `X, d = src.cbwdm_score.build_local_effects(...)` and `alignment_j = X_j^T d`. The v2 group builder verifies the result against the alignment stored in the teacher artifact (absolute/relative tolerance `1e-12`), verifies the stored `admissible` flag against the strict comparison, and checks consistency with `explicit_harmful_negative` classification. It does not derive a raw-alignment gate.

The utility mask is exactly:

```text
utility_mask_j = authoritative_alignment_j > alignment_eps
```

Within the mask, signed-v1 effective gains and thresholds are retained:

- positive utility target: `effective_gain > b_plus`;
- ordinary negative utility target: `effective_gain < b_minus`;
- neutral: between the two thresholds;
- `neutral_sample_policy=negative`: neutral participates in utility BCE as target 0 but remains excluded from ranking;
- `explicit_harmful_negative`: always outside the utility mask and therefore excluded from utility BCE and utility ranking. It trains only the gate.

If a state contains no admissible candidate, utility loss is a differentiable zero and gate BCE still trains on every remaining candidate.

## 5. Loss

Let `g_j` be a gate logit, `u_j` a utility logit, `a_j` the authoritative gate target, and `A={j:a_j=1}`.

Gate loss over all remaining candidates:

```text
L_gate = BCEWithLogits({g_j}, {a_j})
```

Utility BCE over admissible candidates only, using signed-v1 gain thresholds and neutral policy:

```text
L_utility_BCE = BCEWithLogits({u_j : j in A}, utility_targets)
```

For admissible positive set `P` and admissible ordinary low-gain negative set `N`, signed-v1 ranking is reused:

```text
L_utility_rank = log(1 + sum_{n in N, p in P} exp(gamma * (u_n - u_p)))
```

If either `P` or `N` is empty, `L_utility_rank=0`. Neutrals and explicit harmful negatives do not enter ranking.

The utility combination remains the signed-v1 implementation:

```text
L_utility = beta * L_utility_BCE + (1 - beta) * L_utility_rank
```

The v2 total is:

```text
L_total = lambda_gate * L_gate + lambda_utility * L_utility
```

First-round values are frozen by the training CLI:

```text
beta=0.25
gamma=1.0
lambda_gate=1.0
lambda_utility=1.0
b_plus=0.01
b_minus=0.001
neutral_sample_policy=negative
alignment_eps=0.0
```

There is no class weighting, focal loss, step weighting, margin/temperature grid, hard-negative mining, or dynamic lambda.

## 6. Inference

At each state, the existing gold-free text constructor encodes every remaining candidate once through the shared backbone and returns both logits.

The first-round inference parameters are frozen:

```text
gate_threshold=0.0
utility_threshold=0.0
top_m=4
min_docs=0
```

The action rule is:

1. Build `A_hat={j:gate_logit_j >= gate_threshold}`.
2. If `A_hat` is empty, STOP with `no_predicted_admissible_candidate`.
3. Otherwise choose the maximum utility logit within `A_hat`.
4. If that maximum is strictly below `utility_threshold`, STOP with `utility_below_threshold`.
5. Otherwise select it, update the selected-evidence state, and continue.
6. After four selections, STOP with `top_m_reached`.

Utility ties follow signed-v1's deterministic rule: prefer lower source/BM25 rank; stable input order resolves any remaining exact tie.

Every selected document stores `gate_score`, `utility_score`, `selection_step`, source rank/score, and the existing title/text fields. The generic `selector_score` field is retained only for evaluator compatibility and is explicitly documented in both the selected row and selection metadata as an alias of `utility_score`.

Each step records `predicted_gate_score`, `predicted_utility_score`, STOP state/reason, remaining count, and `all_candidate_scores`. Every candidate score item contains `doc_id`, `gate_score`, `utility_score`, rank/source rank, and the learned admissibility decision; there is no ambiguous single `score` field.

## 7. Gold Leakage Audit

The deployable inference functions accept query text, current selected documents, candidate documents, and dual logits only. They do not accept gold label, gold alignment, `d`, teacher supervision, eta-derived gold direction, or gold evidence keys.

The posterior row remains the common container, but feature construction reads only `query` and `candidates`; selection scoring uses only candidate title/text and the already-selected state. `make_selection_row` copies the label only after selection for evaluation compatibility.

The gold-free test changes the row's label from SUPPORTS to REFUTES while keeping query, state, and candidates fixed. It verifies identical model input strings, selection steps, and selected document IDs, plus:

```text
uses_gold_at_test = false
selection_metadata.uses_gold_at_inference = false
```

Authoritative and raw alignments occur only in the read-only analysis helpers. Raw alignment never enters model training targets, inference features, gating, ranking, or STOP decisions.

## 8. Backward Compatibility with signed-v1

No signed-v1 file or shared core file was modified. v1 retains its original single-logit model, training groups, BCE/ranking behavior, checkpoint schema, selection semantics, threshold behavior, artifact root, and post-hoc alignment implementation.

v2 has separate module names, CLI entry points, manifest/checkpoint schemas, method name, and artifact root (`signed_teacher_v2`). Existing signed-v1 tests and directly related selector/CBWDM tests pass unchanged.

## 9. Tests Run

Local Python: repository `.venv`; no network or large model fixture was used.

```powershell
.\.venv\Scripts\python.exe -m py_compile src\diagnostics\signed_v2_model.py src\diagnostics\signed_selector_v2.py scripts\diagnostics\train_signed_selector_v2.py scripts\diagnostics\select_signed_selector_v2.py scripts\diagnostics\analyze_signed_selector_v2.py
```

Result: PASS.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_signed_teacher_v1.py tests/test_signed_selector_v2.py
```

Result: PASS, 25 tests after the final v2 test set (signed-v1 regression plus v2 tests).

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_signed_teacher_v1.py tests/test_signed_selector_v2.py tests/test_selector_loss_and_schema.py tests/test_method_failure_diagnostics.py tests/test_cbwdm_score.py
```

Result: PASS, 41 tests.

The tests cover dual-head shapes and parameter independence, strict gate targets, direct authoritative alignment reconstruction, utility masking, no-admissible numerical behavior, gate-before-utility inference, all three STOP paths, deterministic tie-breaking, gold-free features/actions, checkpoint round-trip, explicit v2 manifest/schema identity, post-hoc gate/harm/budget metrics, and v1 regressions.

```powershell
git diff --check
```

Result: PASS. A separate no-index diff check over every new source/test file also found no new whitespace errors.

## 10. Known Limitations

- v2 has not run the real FEVER pilot; all current validation is static/unit-level with tiny local fixtures.
- seed21 and seed42 have not been run.
- v2 has not entered preformal and is not registered in `src/preformal/registry.py`.
- held-out test and FM2 have not been run.
- raw-gate is not implemented; the gate target remains authoritative smoothed signed-v1 alignment.
- step balancing, step reweighting, and later-step oversampling are not implemented.
- no class weighting, focal loss, threshold tuning, lambda tuning, or hard-negative mining is implemented.
- both heads use deterministic fresh initialization; the old architecture-specific sequence-classification head is not transplanted into the utility head.
- actual scientific benefit, calibration, gate separability, stopping behavior, and generator accuracy require the proposed seed13 pilot.

## 11. Exact Server Commands Proposed

These commands are proposals based on the implemented CLIs' actual local `--help`. They were not executed. They reuse the existing signed-v1 train-core teacher and never overwrite v1 artifacts.

```bash
cd /root/rag-cbwdm
RUN=/root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13
PY=/root/miniconda3/envs/rag-cbwdm-baselines/bin/python
V2="$RUN/artifacts/diagnostics/method_failure_audit/signed_teacher_v2"

"$PY" scripts/diagnostics/train_signed_selector_v2.py \
  --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml \
  --run-dir "$RUN" \
  --teacher "$RUN/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/teacher/head6586_train_core/teacher.jsonl" \
  --posteriors "$RUN/artifacts/formal/fever2_train_core_posteriors.jsonl" \
  --retrieval "$RUN/artifacts/formal/fever2_train_core_bm25_top20.jsonl" \
  --output-dir "$V2/training/head6586_seed13_dual_head_v1" \
  --model-name /root/models/ms-marco-MiniLM-L-6-v2 \
  --epochs 3 --lr 2e-5 --batch-size 8 --max-length 512 \
  --alignment-eps 0 --b-plus 0.01 --b-minus 0.001 \
  --beta 0.25 --gamma 1.0 --lambda-gate 1.0 --lambda-utility 1.0 \
  --neutral-sample-policy negative --seed 13 --device auto --resume

"$PY" scripts/diagnostics/select_signed_selector_v2.py \
  --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml \
  --run-dir "$RUN" \
  --posteriors "$RUN/artifacts/formal/fever2_validation_posteriors.jsonl" \
  --checkpoint-dir "$V2/training/head6586_seed13_dual_head_v1/checkpoint" \
  --output "$V2/selections/head6586_seed13_dual_head_v1/selection.jsonl" \
  --top-m 4 --min-docs 0 --gate-threshold 0 --utility-threshold 0 \
  --batch-size 32 --validation-limit 500 --device auto --resume

"$PY" scripts/07_eval_rag_classification.py \
  --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml \
  --split validation \
  --selection "$V2/selections/head6586_seed13_dual_head_v1/selection.jsonl" \
  --output "$V2/evaluation/head6586_seed13_dual_head_v1/predictions.jsonl" \
  --metrics-output "$V2/evaluation/head6586_seed13_dual_head_v1/metrics.json" \
  --model-name /root/models/Qwen2.5-1.5B-Instruct \
  --method-name rag_cbwdm_signed_v2 --limit 500 --resume

"$PY" scripts/diagnostics/analyze_signed_selector_v2.py \
  --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml \
  --run-dir "$RUN" \
  --posteriors "$RUN/artifacts/formal/fever2_validation_posteriors.jsonl" \
  --selection "$V2/selections/head6586_seed13_dual_head_v1/selection.jsonl" \
  --predictions "$V2/evaluation/head6586_seed13_dual_head_v1/predictions.jsonl" \
  --metrics "$V2/evaluation/head6586_seed13_dual_head_v1/metrics.json" \
  --output-dir "$V2/reports/head6586_seed13_dual_head_v1" \
  --alignment-eps 0
```

## 12. HANDOFF_TO_CHATGPT

STATUS: COMPLETED
GIT_HEAD: 6586ce01e6c3c7d8cb93bbce3c0b1cd6dd67c215
METHOD: rag_cbwdm_signed_v2
ARCHITECTURE: rag_cbwdm_signed_v2_dual_head_v1
SHARED_BACKBONE: One AutoModel MiniLM encoder shared by independent Linear(hidden_size,1) gate and utility heads
GATE_TARGET: 1 iff authoritative X_j^T d from build_local_effects is strictly greater than alignment_eps; otherwise 0
UTILITY_TARGET: signed-v1 effective gain/BCE/ranking semantics restricted to authoritative-admissible candidates
UTILITY_MASK: authoritative alignment > alignment_eps; explicit_harmful_negative is excluded
LAMBDA_GATE: 1.0
LAMBDA_UTILITY: 1.0
GATE_THRESHOLD: 0.0 with inclusive gate_logit >= threshold
UTILITY_THRESHOLD: 0.0 with STOP iff best admissible utility_logit < threshold
TOP_M: 4
MIN_DOCS: 0
RAW_GATE_IMPLEMENTED: NO
STEP_BALANCING_IMPLEMENTED: NO
V1_BEHAVIOR_CHANGED: NO
GOLD_USED_AT_INFERENCE: NO
TESTS: PASS; 41 focused v1/v2/selector/CBWDM tests
FILES_ADDED: src/diagnostics/signed_v2_model.py, src/diagnostics/signed_selector_v2.py, scripts/diagnostics/train_signed_selector_v2.py, scripts/diagnostics/select_signed_selector_v2.py, scripts/diagnostics/analyze_signed_selector_v2.py, tests/test_signed_selector_v2.py, SIGNED_V2_IMPLEMENTATION_REPORT.md
FILES_MODIFIED: NONE
SERVER_TRAIN_COMMAND: "$PY" scripts/diagnostics/train_signed_selector_v2.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir "$RUN" --teacher "$RUN/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/teacher/head6586_train_core/teacher.jsonl" --posteriors "$RUN/artifacts/formal/fever2_train_core_posteriors.jsonl" --retrieval "$RUN/artifacts/formal/fever2_train_core_bm25_top20.jsonl" --output-dir "$V2/training/head6586_seed13_dual_head_v1" --model-name /root/models/ms-marco-MiniLM-L-6-v2 --epochs 3 --lr 2e-5 --batch-size 8 --max-length 512 --alignment-eps 0 --b-plus 0.01 --b-minus 0.001 --beta 0.25 --gamma 1.0 --lambda-gate 1.0 --lambda-utility 1.0 --neutral-sample-policy negative --seed 13 --device auto --resume
SERVER_SELECT_COMMAND: "$PY" scripts/diagnostics/select_signed_selector_v2.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir "$RUN" --posteriors "$RUN/artifacts/formal/fever2_validation_posteriors.jsonl" --checkpoint-dir "$V2/training/head6586_seed13_dual_head_v1/checkpoint" --output "$V2/selections/head6586_seed13_dual_head_v1/selection.jsonl" --top-m 4 --min-docs 0 --gate-threshold 0 --utility-threshold 0 --batch-size 32 --validation-limit 500 --device auto --resume
SERVER_ANALYZE_COMMAND: "$PY" scripts/diagnostics/analyze_signed_selector_v2.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir "$RUN" --posteriors "$RUN/artifacts/formal/fever2_validation_posteriors.jsonl" --selection "$V2/selections/head6586_seed13_dual_head_v1/selection.jsonl" --predictions "$V2/evaluation/head6586_seed13_dual_head_v1/predictions.jsonl" --metrics "$V2/evaluation/head6586_seed13_dual_head_v1/metrics.json" --output-dir "$V2/reports/head6586_seed13_dual_head_v1" --alignment-eps 0
REPORT_PATH: C:\Users\wenhao\Desktop\CBWDM\rag_cbwdm\SIGNED_V2_IMPLEMENTATION_REPORT.md
ALLOWLIST_UNCHANGED: YES
GIT_DIFF_STAT: 7 new files, 2364 insertions(+); no existing file modified

# Learned-Selector Runtime Optimization Implementation Report

## 1. Pre-change bottlenecks

InfoGain performed one optimizer update, tokenizer call, and encoder forward per query group, while constructing every valid DIG pair as a separate Python scalar tensor. Signed-v1 performed one tokenizer call and one forward per state group, accumulated `(loss / 8).backward()` across eight groups, and stepped once per eight groups. At the reported 310,640 signed groups and three epochs, that meant 931,920 tokenizer/forward calls and 116,490 optimizer steps.

The full source audit is in `RUNTIME_OPTIMIZATION_AUDIT.md`. Artifact-dependent candidate and pair-count distributions were not fabricated: the local checkout lacks the server-scale training artifacts. The artifact-backed benchmark harness now computes those exact distributions when run on the server inputs.

## 2. Modified files

- `RUNTIME_OPTIMIZATION_AUDIT.md`
- `RUNTIME_OPTIMIZATION_IMPLEMENTATION_REPORT.md`
- `src/baselines/infogain_fever.py`
- `src/baselines/infogain_selector.py`
- `src/selector_cross_encoder.py`
- `scripts/12b_train_infogain_reranker.py`
- `scripts/preformal/26_train_signed_v1.py`
- `src/formal_matrix.py`
- `scripts/benchmarks/benchmark_learned_selector_runtime.py`
- `scripts/benchmarks/benchmark_runtime_kernels.py`
- `tests/test_runtime_optimization.py`
- `tests/test_formal_matrix.py`

The three pre-existing unrelated dirty paths were not modified.

## 3. Mathematical and training semantics

No method, teacher, threshold, neutral handling, CE/filter loss, beta/gamma weight, group order, epoch count, learning rate, seed policy, selection rule, evaluation rule, or held-out rule changed.

InfoGain now builds upper-triangular unordered pair indices, masks exact-equal DIG pairs, orients each score margin toward the larger DIG, applies the same `softplus(-margin)`, and means over the same valid pairs. DIG comparison uses float64 tensors to preserve Python-float equality/order. No-valid-pair behavior remains `rank_scores.sum() * 0.0`, a differentiable zero. One query group still means one forward, backward, and optimizer step. `zero_grad(set_to_none=True)` replaces the default zero fill.

Signed-v1 keeps the identical shuffled group list and partitions consecutive groups into blocks of eight. Each block's texts are flattened and tokenized once, encoded tensors are forwarded in candidate microbatches, scores are split at recorded offsets, and the unchanged `cbwdm_multitask_loss` is evaluated per group. The block loss is exactly `sum(group_losses) / 8`, including the final incomplete block. There is one backward and optimizer step per block, preserving update frequency and the frozen optimizer-group batch meaning.

`InfoGainPointwiseReranker` and `CrossEncoderSelector` now expose `encode_texts` and `forward_encoded`; their old `forward`/`score_texts` APIs remain compatible.

## 4. InfoGain legacy versus optimized benchmark

The local machine has CPU-only PyTorch 2.13.0 and no CUDA. A fixed-seed-13 synthetic training-kernel benchmark was run from a temporary directory; it is not a representative transformer or RTX 4090 benchmark.

| Groups | Legacy seconds | Vectorized seconds | Legacy groups/s | Vectorized groups/s | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 1.2494 | 0.5861 | 800.4 | 1,706.1 | 2.13x |
| 5,000 | 6.1306 | 2.8858 | 815.6 | 1,732.6 | 2.12x |

Optimizer steps, forward calls, tokenizer calls, final loss, and sampled loss trajectory were identical between legacy and vectorized runs. These numbers isolate Python pair-loss overhead; end-to-end transformer speedup will be smaller and must be measured on the server.

## 5. Signed-v1 legacy versus optimized benchmark

The same synthetic deterministic scorer/control-flow benchmark produced:

| Groups | Variant | Forward batch | Seconds | Groups/s | Forward calls | Tokenizer calls | Speedup |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1,000 | legacy | n/a | 0.8575 | 1,166.2 | 1,000 | 1,000 | 1.00x |
| 1,000 | block-v1 | 16 | 0.6584 | 1,518.8 | 433 | 125 | 1.30x |
| 1,000 | block-v1 | 32 | 0.6752 | 1,481.1 | 250 | 125 | 1.27x |
| 1,000 | block-v1 | 64 | 0.7073 | 1,413.9 | 127 | 125 | 1.21x |
| 1,000 | block-v1 | 128 | 0.6459 | 1,548.3 | 125 | 125 | 1.33x |
| 5,000 | legacy | n/a | 4.1977 | 1,191.1 | 5,000 | 5,000 | 1.00x |
| 5,000 | block-v1 | 16 | 3.3416 | 1,496.3 | 2,179 | 625 | 1.26x |
| 5,000 | block-v1 | 32 | 3.2215 | 1,552.1 | 1,254 | 625 | 1.30x |
| 5,000 | block-v1 | 64 | 3.5035 | 1,427.1 | 637 | 625 | 1.20x |
| 5,000 | block-v1 | 128 | 3.3479 | 1,493.5 | 625 | 625 | 1.25x |

All variants used exactly 125/625 optimizer steps, respectively, and had identical final losses and sampled trajectories. This confirms control-flow accounting but is not evidence that batch 128 fits or is fastest on the real transformer.

## 6. GPU memory and utilization

Local GPU memory and utilization are unavailable because CUDA is absent. No RTX 4090 claim is made. Both workers now record peak allocated CUDA bytes plus forward/tokenizer/optimizer counts, throughput, final loss, and trajectory in the training manifest. GPU utilization is explicitly recorded as unsampled rather than invented.

`scripts/benchmarks/benchmark_learned_selector_runtime.py` runs the actual workers for 1,000 and 5,000 groups, compares InfoGain legacy/vectorized and signed legacy/block-v1 at 16/32/64/128, records OOM without aborting later cases, and writes only beneath a caller-supplied non-formal benchmark root.

## 7. Recommended forward batch size

Use 32 as the conservative provisional formal default and the first RTX 4090 candidate. It is explicit in the plan and training fingerprint, not hidden in code-only behavior. Do not promote 64 or 128 based on the synthetic CPU result. Run the artifact-backed harness on the RTX 4090; retain 32 unless a larger value is OOM-safe and measurably faster with the actual length distribution.

## 8. Speedup summary

- InfoGain loss kernel: about 2.12x in the local 1,000/5,000-group synthetic benchmark.
- Signed control flow: about 1.30x at forward batch 32 in the local 5,000-group synthetic benchmark.
- Real RTX 4090 end-to-end speedup: pending the artifact-backed benchmark; no unsupported estimate is reported.

## 9. Correctness tests

- Twenty randomized float32 cases compare independent legacy-reference and vectorized InfoGain total/rank/filter losses and both parameter-gradient tensors at `atol=rtol=1e-6`.
- A no-valid-pair case verifies differentiable zero behavior.
- A deterministic ten-group signed test includes an incomplete two-group final block and verifies block composition `[8, 2]`, identical optimizer-step count, tokenizer-call reduction, and parameter updates within `1e-6`.
- Formal-plan tests verify runtime implementation and forward batch are explicit in generated worker commands and the semantic plan fingerprint.
- Focused runtime/formal tests: 44 passed.
- Related learned-selector tests: 132 passed.
- Full suite: 265 passed, 2 subtests passed.

With dropout enabled in normal training, block flattening changes padding shapes and random-number consumption, so bitwise identity with legacy group-by-group forwards is not expected. Group order, block composition, loss definitions, normalization, and optimizer steps remain exact. The deterministic dropout-free update gate passed.

## 10. Provenance and fingerprint changes

InfoGain contract now records:

- `implementation_version: infogain_vectorized_rank_v1`
- optimizer group batch 1
- `rank_loss_implementation`
- `vectorized_infogain_rank_loss`

Signed-v1 contract now records:

- `implementation_version: signed_optimizer_block_v1`
- `runtime_implementation`
- optimizer group batch 8
- candidate `forward_batch_size`
- optional benchmark `max_groups`

The formal matrix plan includes the same `training_runtime` object in its semantic fingerprint and emits explicit worker flags. Therefore legacy checkpoints cannot be mistaken for optimized checkpoints during resume.

## 11. Existing Qwen2.5-1.5B seed-13 checkpoint

The existing checkpoint remains a valid legacy-runtime result and is not modified or deleted. To use the optimized runtime in the subsequent main matrix, seed 13 must be trained into a new output path because the new fingerprint intentionally rejects legacy checkpoint reuse. Seeds 21/42 and their artifacts remain untouched for robustness analysis.

## 12. Applicability to Qwen0.5B, Qwen7B, and Mistral

The runtime changes are selector-side and do not alter generator/posterior/teacher semantics. They are source-correct for future generator branches as long as each branch keeps Phase B generator conditioning and uses a new runtime-fingerprinted checkpoint. Before production use, run the real 1,000/5,000-group RTX 4090 benchmark for the actual selector/text-length distribution and confirm the chosen forward batch does not OOM.

## 13. Benchmark artifacts and exclusions

The local synthetic JSON was written only to the Windows temporary directory, not the repository or any formal artifact root. No full-development matrix, live server model, held-out split, or Qwen seed 21/42 training was run.

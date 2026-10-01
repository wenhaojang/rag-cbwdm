# Learned-Selector Runtime Optimization Audit

## Scope and environment

This is a source audit of the formal learned-selector training paths at Git baseline `dd150fb41618985cef889dc0ad872e717a84a3d8`. The local Windows checkout has CPU-only PyTorch (`2.13.0+cpu`) and no CUDA device. The server-scale teacher/posterior artifacts are not present locally, so artifact-dependent distributions are identified as measurements to be produced by the benchmark harness rather than inferred or fabricated. No held-out path or formal artifact was opened or changed.

## InfoGain call chain and current runtime semantics

The formal path is:

`scripts/12b_train_infogain_reranker.py` -> `group_teacher_rows` / `pointwise_input` / `infogain_multitask_loss` in `src/baselines/infogain_fever.py` -> `InfoGainPointwiseReranker.forward` in `src/baselines/infogain_selector.py`.

- Group count is the number of distinct `query_id` values after optional `--max-groups` truncation. The formal worker materializes all teacher rows and then groups them in first-seen query order.
- Candidate count for a group is the number of teacher rows with that `query_id`. Its exact distribution is artifact-dependent and must be measured from the server teacher JSONL. The source imposes no fixed candidate count.
- Per epoch, forward calls = group count.
- Per epoch, `optimizer.step()` calls = group count. One query group is exactly one optimizer update.
- Per epoch, tokenizer calls = group count because every `model.forward(texts)` tokenizes that group once.
- For group size `n`, the loop examines `n(n-1)/2` unordered pairs. The valid pair count is `C(n,2) - sum_v C(count(dig=v),2)`: pairs with exactly equal DIG are excluded. The exact valid-pair distribution is artifact-dependent.
- Filter CE labels are unchanged: `dig >= b_pos` is positive, `dig <= b_neg` is negative, values between are neutral and excluded from filter CE.
- The ranking bottleneck is Python construction of one scalar tensor per valid pair, followed by `torch.stack` and `mean`.

## Signed-v1 call chain and current runtime semantics

The formal path is:

`scripts/preformal/26_train_signed_v1.py` -> `build_signed_training_groups` in `src/diagnostics/signed_teacher_v1.py` -> `build_selector_input` / `CrossEncoderSelector.score_texts` / `cbwdm_multitask_loss` in `src/selector_cross_encoder.py`.

- `build_signed_training_groups` emits one group for every teacher state that still has at least one remaining candidate with non-empty text. It includes terminal states when they retain candidates. Thus the reported 310,640 groups are state-level groups, not 310,640 source claims: the count is the sum of eligible states across teacher rows.
- Candidate count is the number of text-bearing `remaining_candidates` at that state. Starting from `N` retrieved candidates, successive states normally have `N, N-1, ...`; missing-text candidates are removed. The exact server distribution requires the server teacher/posterior/retrieval artifacts and is not available in this checkout.
- The frozen selector `batch_size=8` is an optimizer-group accumulation factor, not a transformer text batch size.
- Current per epoch forward calls = 310,640. Each group calls `score_texts` once with `batch_size=len(group)`, so it also performs one tokenizer call and normally one model forward per group.
- Current per epoch optimizer steps = `ceil(310,640/8) = 38,830`; across three epochs this is 116,490 steps. Forward/tokenizer calls across three epochs are 931,920 each.
- For each group the code runs `(loss / 8).backward()`. It steps after each eight groups and after the final incomplete block.
- The final incomplete block is still normalized by the constant 8, not by its actual group count. Its gradient is therefore `sum(group_gradients)/8`. Any block implementation must preserve that exact normalization.
- Group order is a fresh list shuffled by `random.Random(seed + epoch)`, independent of global RNG state.

## Optimization boundary

Safe source-level targets are: tensorize only InfoGain's valid unordered pair loss; separate tokenization from forward while retaining public APIs; flatten each fixed eight-group signed optimizer block into one tokenizer call; microbatch encoded tensors for model memory; sum the original per-group losses and divide by exactly eight; perform one backward and one optimizer step per block.

Unsafe changes excluded from this work are teacher/group construction, thresholds, neutral policy, loss weights, group ordering, epoch/lr/seed policy, selection/evaluation, mixed precision, `torch.compile`, fused optimizers, and held-out/formal registry policy.

## Required artifact-side measurements

The benchmark harness must report, for the actual server training artifacts: group count, candidate-count histogram, InfoGain valid-pair histogram, forward/tokenizer/optimizer counts, wall time, groups/s, texts/s, peak CUDA memory, final loss, and loss trajectory. This local audit does not claim RTX 4090 throughput or recommend a final forward microbatch size because CUDA is unavailable here.

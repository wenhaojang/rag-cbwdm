# RAG-CBWDM Project Status Report

## 1. Executive Summary

This audit checked the current `rag_cbwdm` repository against `RAG_CBWDM_PROJECT_AUDIT_CODEX_TASK.md`. The project already contains a usable staged FEVER RAG-CBWDM pipeline: FEVER claim/corpus preparation, BM25 retrieval, generator label posterior computation, CBWDM teacher construction, feature-MLP selector training/selection, state-aware cross-encoder selector training/selection, naive top-M selection, oracle diagnostic conversion, and final generator-based classification evaluation.

The main code paths compile, all listed scripts expose `--help`, and the configured conda environment can import the core dependencies. Existing local data and outputs are small/smoke-test scale and should not be treated as final experimental results. The most important open items before a formal run are: add the BGE reranker baseline script, change Qwen configs to `trust_remote_code: false` for production, add a unified run script, add result summarization/diagnostic scripts, normalize selection output schema to always include `num_docs`, and consider batching in posterior computation.

No experimental code was changed during this audit. This report file is the only file added.

## 2. Current Repository Structure

Key top-level entries:

```text
.git/
.agents/
.venv/
configs/
data/
outputs/
scripts/
src/
tests/
运行过的指令/
.gitignore
requirements.txt
RAG_CBWDM_PROJECT_AUDIT_CODEX_TASK.md
PROJECT_STATUS_REPORT.md
```

Important tracked/source areas:

```text
configs/
  fever2_minimal.yaml
  fever3_minimal.yaml

scripts/
  download_fever_hf.py
  00_prepare_fever.py
  01_prepare_fever_corpus.py
  02_retrieve_bm25.py
  03_compute_label_posteriors.py
  04_build_cbwdm_teacher.py
  05_train_selector.py
  06_select_with_selector.py
  07_eval_rag_classification.py
  08_select_naive_topm.py
  09_select_cbwdm_oracle_from_teacher.py
  10_train_cross_encoder_selector.py
  11_select_with_cross_encoder.py

src/
  io_utils.py
  retrieval_bm25.py
  prompts.py
  label_logits.py
  cbwdm_score.py
  selector_dataset.py
  selector_model.py
  selector_cross_encoder.py
  metrics.py
```

`tree /F /A` was run. Its full output is large because local ignored directories such as `.venv/`, `data/`, `outputs/`, `__pycache__/`, checkpoints, and model artifacts are present.

Current `git status --short` at audit time:

```text
 M configs/fever2_minimal.yaml
 M configs/fever3_minimal.yaml
 M requirements.txt
?? RAG_CBWDM_PROJECT_AUDIT_CODEX_TASK.md
?? scripts/03_compute_label_posteriors.py
?? scripts/04_build_cbwdm_teacher.py
?? scripts/05_train_selector.py
?? scripts/06_select_with_selector.py
?? scripts/07_eval_rag_classification.py
?? scripts/08_select_naive_topm.py
?? scripts/09_select_cbwdm_oracle_from_teacher.py
?? scripts/10_train_cross_encoder_selector.py
?? scripts/11_select_with_cross_encoder.py
?? scripts/download_fever_hf.py
?? src/cbwdm_score.py
?? src/label_logits.py
?? src/metrics.py
?? src/prompts.py
?? src/selector_cross_encoder.py
?? src/selector_dataset.py
?? src/selector_model.py
?? tests/fixtures/posteriors/
?? 运行过的指令/
```

`.gitignore` exists and already contains the required baseline ignores:

```gitignore
__pycache__/
*.pyc
.venv/
data/
outputs/
checkpoints/
*.faiss
*.pt
*.bin
*.safetensors
.DS_Store
```

## 3. Implemented Pipeline and Script Responsibilities

`scripts/download_fever_hf.py`: downloads/materializes FEVER v1.0 claim splits and wiki pages from Hugging Face datasets into `data/raw/fever`. Supports claim/wiki debug limits, wiki shard size, custom cache dir, and skipping wiki.

`scripts/00_prepare_fever.py`: converts raw FEVER claims into normalized JSONL rows with `id`, `query`, `label`, and `split`; handles FEVER-2 label dropping and FEVER-3 label mapping.

`scripts/01_prepare_fever_corpus.py`: converts FEVER wiki-pages JSONL shards into a sentence-level corpus with `doc_id`, `title`, and `text`.

`scripts/02_retrieve_bm25.py`: builds a BM25 index from the prepared corpus and writes top-N retrieval candidates for each prepared claim.

`scripts/03_compute_label_posteriors.py`: loads a causal LM generator and computes no-evidence posterior `eta0` plus per-candidate posterior `eta` using next-token label verbalizer logits. Current implementation scores rows/candidates sequentially, not batched.

`scripts/04_build_cbwdm_teacher.py`: builds greedy CBWDM teacher trajectories from posterior rows. It records selected document ids and per-step gains.

`scripts/05_train_selector.py`: trains the feature-based MLP selector from posterior rows and teacher trajectories.

`scripts/06_select_with_selector.py`: runs greedy evidence selection with the trained feature-MLP selector.

`scripts/07_eval_rag_classification.py`: evaluates selected evidence by recomputing generator label posteriors and writing prediction JSONL plus metrics JSON. Also supports `--no-evidence`.

`scripts/08_select_naive_topm.py`: writes a naive baseline selection using the first M BM25 candidates.

`scripts/09_select_cbwdm_oracle_from_teacher.py`: converts gold-label CBWDM teacher selections into selection JSONL. This is diagnostic only and should not be reported as a deployable test-time method.

`scripts/10_train_cross_encoder_selector.py`: trains a state-aware listwise cross-encoder selector from teacher gains.

`scripts/11_select_with_cross_encoder.py`: runs greedy evidence selection with a trained cross-encoder selector checkpoint.

BGE reranker baseline: not implemented in `scripts/`. A search for `bge`, `FlagReranker`, `FlagEmbedding`, and `reranker` found no project code. Recommended script name: `scripts/12_select_bge_reranker.py`.

## 4. Configurations and Key Settings

`configs/fever2_minimal.yaml`:

- `dataset: fever2`
- Labels: `SUPPORTS`, `REFUTES`
- Drops raw `NOT ENOUGH INFO`
- Generator: `Qwen/Qwen2.5-7B-Instruct`
- Retrieval: BM25 `top_n: 20`
- CBWDM: `top_m: 4`, `ridge_lambda: 0.01`, `eps_smooth: 0.001`, `store_all_gains: true`
- Selector: feature MLP, hidden dim 128, dropout 0.1, 5 epochs, batch size 64

`configs/fever3_minimal.yaml`:

- `dataset: fever3`
- Labels: `SUPPORTS`, `REFUTES`, `NOT_ENOUGH_INFO`
- Maps raw `NOT ENOUGH INFO` to `NOT_ENOUGH_INFO`
- Generator/retrieval/CBWDM/selector settings match the FEVER-2 config.

Both configs include:

```yaml
generator:
  model_name: Qwen/Qwen2.5-7B-Instruct
  dtype: auto
  device_map: auto
  trust_remote_code: true
  max_context_tokens: 4096
```

Production note: Qwen2.5 usually does not require `trust_remote_code=True`. For formal server runs, set `trust_remote_code: false` explicitly to avoid Hugging Face remote custom code review/security concerns.

## 5. Dependency and Environment Check

Configured environment command used:

```cmd
D:\anaconda3\condabin\conda.bat run -n rag_cbwdm python ...
```

`requirements.txt` currently contains:

```text
pyyaml
rank_bm25
numpy
torch
transformers
accelerate
```

Import checks:

```text
import yaml, numpy, torch, transformers: PASS
datasets version: 2.19.2, PASS
from FlagEmbedding import FlagReranker: PASS
```

`datasets` and `FlagEmbedding` are importable in the current conda environment but are not listed in `requirements.txt`. For reproducible server setup, add them or document them separately. If BGE baseline is added, include `FlagEmbedding` in server dependency instructions.

Syntax check:

```text
python -m py_compile ... all listed src and scripts: PASS
```

CLI `--help` check:

```text
All listed scripts from download_fever_hf.py through 11_select_with_cross_encoder.py: PASS
```

Key CLI parameters found:

- Common: `--config`, `--split`, `--limit`, path overrides such as `--output`, `--retrieval`, `--posteriors`, `--teacher`, `--checkpoint`, `--output-dir`
- Retrieval: `--top-n`
- Posterior/eval: `--model-name`, `--max-candidates`, `--no-evidence`
- Teacher/selection: `--top-m`, `--ridge-lambda`, `--stop-threshold`, `--eps-smooth`, `--l-type`, `--store-all-gains`, `--score-threshold`, `--min-docs`
- Feature selector training: `--epochs`, `--batch-size`, `--lr`, `--hidden-dim`, `--dropout`, `--limit-states`, `--max-candidates-per-state`
- Cross-encoder: `--checkpoint-dir`, `--max-length`, `--device`, `--max-train-groups`, `--max-candidates-per-group`, `--teacher-temperature`

## 6. Data and Existing Outputs

Local raw FEVER data exists:

```text
data/raw/fever/train.jsonl: exists, 0.37 MB
data/raw/fever/dev.jsonl: exists, 0.34 MB
data/raw/fever/wiki-pages/: 4 shards
```

Wiki shards:

```text
wiki-000.jsonl
wiki-001.jsonl
wiki-002.jsonl
wiki-003.jsonl
```

Existing output directories:

```text
outputs/check_stage12/
outputs/check_stage3/
outputs/check_stage4/
outputs/check_stage5/
outputs/check_stage6/
outputs/check_stage7/
outputs/qwen15b_fever2_small/
outputs/real_fever2_small/
```

Interpretation of existing outputs:

- `outputs/check_stage*`: smoke tests and schema checks. Not formal results.
- `outputs/check_stage7`: cross-encoder selector smoke run with local tiny BERT artifacts. Not formal cross-encoder evidence.
- `outputs/real_fever2_small`: 100-example small FEVER-2 run using tiny generator/posteriors. Useful for pipeline closure only, not meaningful experimental accuracy.
- `outputs/qwen15b_fever2_small`: small 12-example Qwen2.5-1.5B-style run with naive, CBWDM selector, CBWDM oracle, and no-evidence metrics. Too small for formal conclusions.

Observed metrics:

```text
outputs/check_stage6/eval/toy_metrics.json: num=2, accuracy=0.5, avg_docs=2.0
outputs/check_stage6/eval/toy_no_evidence_metrics.json: num=2, accuracy=1.0, avg_docs=0.0
outputs/qwen15b_fever2_small/eval/fever2_dev_cbwdm_oracle_metrics.json: num=12, accuracy=0.6667, avg_docs=4.0
outputs/qwen15b_fever2_small/eval/fever2_dev_cbwdm_selector_metrics.json: num=12, accuracy=0.6667, avg_docs=4.0
outputs/qwen15b_fever2_small/eval/fever2_dev_naive_top4_metrics.json: num=12, accuracy=0.6667, avg_docs=4.0
outputs/qwen15b_fever2_small/eval/fever2_dev_no_evidence_metrics.json: num=12, accuracy=1.0, avg_docs=0.0
outputs/real_fever2_small/eval/fever2_dev_cbwdm_selector_metrics.json: num=100, accuracy=0.47, avg_docs=4.0
outputs/real_fever2_small/eval/fever2_dev_no_evidence_metrics.json: num=100, accuracy=0.53, avg_docs=0.0
```

These numbers should be treated as smoke-test observations only.

## 7. Output Schema Checks

Schema checks sampled up to five rows per matching file.

Processed FEVER claim JSONL:

```text
Pattern: outputs/**/processed/*train*.jsonl and *dev*.jsonl
Required: id, query, label, split
Result: PASS for sampled files
FEVER-2 NEI check: PASS in sampled rows
```

Corpus JSONL:

```text
Pattern: outputs/**/fever_corpus.jsonl
Required: doc_id, title, text
Result: PASS for sampled files
```

Retrieval JSONL:

```text
Pattern: outputs/**/retrieval/*.jsonl
Required row fields: id, query, label, split, candidates
Required candidate fields: doc_id, rank, score, title, text
Result: PASS for sampled files
```

Posterior JSONL:

```text
Pattern: outputs/**/posteriors/*.jsonl
Required: id, labels, eta0, candidates
Required candidate eta distribution
Result: mostly PASS
Issue: Qwen small posterior files include eta sums such as 0.998291015625,
1.001953125, and 1.00146484375 in sampled rows. This is likely fp16/rounding
drift but exceeds a strict 1e-3 tolerance.
```

Teacher JSONL:

```text
Pattern: outputs/**/teacher/*.jsonl
Required: id, teacher_selected_doc_ids, steps
Required step fields: best_doc_id, best_gain, candidate_gains
Theta monotonicity in sampled rows: PASS where theta fields exist
Result: PASS
```

Selection JSONL:

```text
Pattern: outputs/**/selection/*.jsonl plus selected JSONL smoke files
Required: id, selected_doc_ids, selected_docs, method, num_docs
Result: PARTIAL
```

Files sampled with `num_docs` present:

```text
outputs/check_stage7/selection/fever2_dev_cross_encoder_selected.jsonl
```

Files sampled missing `num_docs`:

```text
outputs/check_stage5/toy_selected.jsonl
outputs/qwen15b_fever2_small/selection/fever2_dev_selected_cbwdm_oracle.jsonl
outputs/qwen15b_fever2_small/selection/fever2_dev_selected_cbwdm_selector.jsonl
outputs/qwen15b_fever2_small/selection/fever2_dev_selected_naive_top4.jsonl
outputs/real_fever2_small/selection/fever2_dev_selected_cbwdm_selector.jsonl
```

Prediction JSONL:

```text
Pattern: outputs/**/eval/*.jsonl
Required: id, gold, pred, correct, labels, probs, selected_doc_ids
Result: PASS for sampled files
```

Metrics JSON:

```text
Pattern: outputs/**/eval/*.json
Required: num_examples, num_correct, accuracy, avg_num_docs
Result: PASS for all checked metrics files
```

## 8. Full Experiment Workflow

1. Download FEVER data

```cmd
D:\anaconda3\condabin\conda.bat run -n rag_cbwdm python scripts\download_fever_hf.py --output-root data/raw/fever
```

Linux:

```bash
conda activate rag_cbwdm
python scripts/download_fever_hf.py --output-root data/raw/fever
```

Input: Hugging Face FEVER datasets. Output: `data/raw/fever/train.jsonl`, `dev.jsonl`, `wiki-pages/*.jsonl`. Key params: `--claim-limit`, `--wiki-limit`, `--wiki-shard-size`, `--cache-dir`, `--skip-wiki`.

2. Prepare FEVER-2 / FEVER-3 claims

```cmd
D:\anaconda3\condabin\conda.bat run -n rag_cbwdm python scripts\00_prepare_fever.py --config configs\fever2_minimal.yaml --splits train dev --output-dir outputs\run_name\processed
```

Input: raw FEVER claim JSONL. Output: processed claim JSONL. Key params: `--config`, `--limit`, `--raw-train`, `--raw-dev`, `--raw-test`, `--output-dir`, `--splits`.

3. Prepare FEVER wiki-pages corpus

```cmd
D:\anaconda3\condabin\conda.bat run -n rag_cbwdm python scripts\01_prepare_fever_corpus.py --config configs\fever2_minimal.yaml --output outputs\run_name\processed\fever_corpus.jsonl
```

Input: `data/raw/fever/wiki-pages/*.jsonl`. Output: sentence corpus JSONL. Key params: `--wiki-pages-dir`, `--limit`, `--output`.

4. BM25 retrieval

```cmd
D:\anaconda3\condabin\conda.bat run -n rag_cbwdm python scripts\02_retrieve_bm25.py --config configs\fever2_minimal.yaml --split train --queries outputs\run_name\processed\fever2_train.jsonl --corpus outputs\run_name\processed\fever_corpus.jsonl --output outputs\run_name\retrieval\fever2_train_bm25_top20.jsonl --top-n 20
```

Input: processed claims and corpus. Output: retrieval JSONL. Key params: `--split`, `--top-n`, `--limit`, `--queries`, `--corpus`, `--output`.

5. Compute generator label posteriors

```cmd
D:\anaconda3\condabin\conda.bat run -n rag_cbwdm python scripts\03_compute_label_posteriors.py --config configs\fever2_minimal.yaml --split train --retrieval outputs\run_name\retrieval\fever2_train_bm25_top20.jsonl --output outputs\run_name\posteriors\fever2_train_posteriors.jsonl --max-candidates 20
```

Input: retrieval JSONL. Output: posterior JSONL with `eta0` and candidate `eta`. Key params: `--model-name`, `--max-candidates`, `--limit`. Note: currently sequential scoring; batching is recommended for formal runs.

6. Build CBWDM teacher

```cmd
D:\anaconda3\condabin\conda.bat run -n rag_cbwdm python scripts\04_build_cbwdm_teacher.py --config configs\fever2_minimal.yaml --split train --posteriors outputs\run_name\posteriors\fever2_train_posteriors.jsonl --output outputs\run_name\teacher\fever2_train_cbwdm_teacher.jsonl --top-m 4
```

Input: posterior JSONL. Output: teacher trajectory JSONL. Key params: `--top-m`, `--ridge-lambda`, `--stop-threshold`, `--eps-smooth`, `--l-type`, `--store-all-gains`.

7. Train feature MLP selector

```cmd
D:\anaconda3\condabin\conda.bat run -n rag_cbwdm python scripts\05_train_selector.py --config configs\fever2_minimal.yaml --posteriors outputs\run_name\posteriors\fever2_train_posteriors.jsonl --teacher outputs\run_name\teacher\fever2_train_cbwdm_teacher.jsonl --output outputs\run_name\selector\feature_mlp_selector.pt
```

Input: train posteriors and teacher trajectories. Output: `.pt` checkpoint. Key params: `--epochs`, `--batch-size`, `--lr`, `--hidden-dim`, `--dropout`, `--limit-states`, `--max-candidates-per-state`.

8. Feature MLP selector evidence selection

```cmd
D:\anaconda3\condabin\conda.bat run -n rag_cbwdm python scripts\06_select_with_selector.py --config configs\fever2_minimal.yaml --posteriors outputs\run_name\posteriors\fever2_dev_posteriors.jsonl --checkpoint outputs\run_name\selector\feature_mlp_selector.pt --output outputs\run_name\selection\fever2_dev_selected_cbwdm_selector.jsonl --top-m 4
```

Input: dev/test posteriors and selector checkpoint. Output: selection JSONL. Key params: `--top-m`, `--score-threshold`, `--min-docs`, `--max-candidates`, `--limit`.

9. Train state-aware cross-encoder selector

```cmd
D:\anaconda3\condabin\conda.bat run -n rag_cbwdm python scripts\10_train_cross_encoder_selector.py --config configs\fever2_minimal.yaml --posteriors outputs\run_name\posteriors\fever2_train_posteriors.jsonl --teacher outputs\run_name\teacher\fever2_train_cbwdm_teacher.jsonl --retrieval outputs\run_name\retrieval\fever2_train_bm25_top20.jsonl --output-dir outputs\run_name\cross_encoder_selector --model-name cross-encoder/ms-marco-MiniLM-L-6-v2
```

Input: train posteriors, teacher trajectories, optional retrieval text. Output: checkpoint directory and training config. Key params: `--model-name`, `--epochs`, `--lr`, `--batch-size`, `--max-length`, `--device`, `--teacher-temperature`.

10. Cross-encoder selector evidence selection

```cmd
D:\anaconda3\condabin\conda.bat run -n rag_cbwdm python scripts\11_select_with_cross_encoder.py --config configs\fever2_minimal.yaml --posteriors outputs\run_name\posteriors\fever2_dev_posteriors.jsonl --checkpoint-dir outputs\run_name\cross_encoder_selector\checkpoint --output outputs\run_name\selection\fever2_dev_cross_encoder_selected.jsonl --top-m 4
```

Input: dev/test posteriors and cross-encoder checkpoint. Output: selection JSONL. Key params: `--checkpoint-dir`, `--top-m`, `--max-length`, `--batch-size`, `--device`, `--score-threshold`, `--min-docs`.

11. Naive top-M baseline

```cmd
D:\anaconda3\condabin\conda.bat run -n rag_cbwdm python scripts\08_select_naive_topm.py --config configs\fever2_minimal.yaml --retrieval outputs\run_name\retrieval\fever2_dev_bm25_top20.jsonl --output outputs\run_name\selection\fever2_dev_selected_naive_top4.jsonl --top-m 4
```

Input: retrieval JSONL. Output: selection JSONL. Key params: `--top-m`, `--method-name`, `--limit`.

12. CBWDM oracle diagnostic

```cmd
D:\anaconda3\condabin\conda.bat run -n rag_cbwdm python scripts\09_select_cbwdm_oracle_from_teacher.py --config configs\fever2_minimal.yaml --teacher outputs\run_name\teacher\fever2_dev_cbwdm_oracle_teacher.jsonl --posteriors outputs\run_name\posteriors\fever2_dev_posteriors.jsonl --output outputs\run_name\selection\fever2_dev_selected_cbwdm_oracle.jsonl --top-m 4
```

Input: teacher JSONL built with gold labels. Output: selection JSONL. Note: diagnostic upper-bound only.

13. BGE reranker baseline

Status: to be added. Recommended script: `scripts/12_select_bge_reranker.py`. It should consume retrieval JSONL, score claim/document pairs with `FlagReranker`, and emit the same selection schema as other selectors.

14. Final evaluation

```cmd
D:\anaconda3\condabin\conda.bat run -n rag_cbwdm python scripts\07_eval_rag_classification.py --config configs\fever2_minimal.yaml --split dev --selection outputs\run_name\selection\fever2_dev_selected_cbwdm_selector.jsonl --output outputs\run_name\eval\fever2_dev_cbwdm_selector_preds.jsonl --metrics-output outputs\run_name\eval\fever2_dev_cbwdm_selector_metrics.json
```

Input: selection JSONL. Output: prediction JSONL and metrics JSON. Key params: `--model-name`, `--limit`, `--max-docs`, `--method-name`, `--no-evidence`.

## 9. Current Baselines and Methods

Implemented:

- BM25 retrieval candidate generation
- No-evidence generator baseline via `07_eval_rag_classification.py --no-evidence`
- Naive BM25 top-M baseline via `08_select_naive_topm.py`
- CBWDM oracle diagnostic via `04_build_cbwdm_teacher.py` plus `09_select_cbwdm_oracle_from_teacher.py`
- CBWDM-trained feature MLP selector via `05_train_selector.py` and `06_select_with_selector.py`
- State-aware cross-encoder selector via `10_train_cross_encoder_selector.py` and `11_select_with_cross_encoder.py`

Missing:

- BGE reranker baseline script.
- Unified result summarization across methods/runs.
- Automatic per-class and prediction-distribution diagnostics.
- Difference/example export for CBWDM oracle vs naive top-M.

## 10. Current Limitations and Open Issues

Must fix or align before formal experiments:

- Add BGE reranker baseline as `scripts/12_select_bge_reranker.py`.
- Change formal Qwen configs to `trust_remote_code: false`.
- Add a unified run script, e.g. `scripts/run_fever2_full.ps1` and/or `scripts/run_fever2_full.sh`, to chain the full experiment.
- Add result summary script, e.g. `scripts/13_summarize_results.py`.
- Normalize selection schema so all selection outputs include `num_docs`.
- Decide whether `datasets` and `FlagEmbedding` should be added to `requirements.txt`.

Performance/scaling concerns:

- `03_compute_label_posteriors.py` scores prompts sequentially. For train 20k, top20/top50/top100, this will be a major bottleneck. Add batching over prompts/candidates before formal runs.
- Retrieval/posterior/selection files currently preserve full candidate text. This is convenient for small runs but can produce very large JSONL files for top100 and large FEVER runs. Consider storing `doc_id`, rank/score, and eta only, then recovering text from a corpus/doc store when needed.
- Current posterior probability sums can drift slightly under Qwen/fp16. Either renormalize before writing JSONL or loosen schema tolerance to account for dtype rounding.

Diagnostics to add:

- Label distribution, per-class accuracy, and prediction distribution.
- Oracle-vs-naive difference examples.
- Retrieval recall against FEVER evidence where feasible.
- Selection length distribution and overlap with BM25 ranks.

Diagnostic-only warnings:

- CBWDM oracle uses gold labels and is an upper-bound diagnostic, not a real test-time method.
- tiny-gpt2/local tiny BERT smoke tests have no experimental meaning.
- The current 12-example and 100-example runs are too small for formal claims.

## 11. Server Migration Plan

Upload project code/config only:

```text
configs/
scripts/
src/
tests/fixtures/ if needed for smoke tests
requirements.txt
.gitignore
PROJECT_STATUS_REPORT.md
```

Do not upload by default:

```text
.venv/
outputs/
data/ unless the server cannot download FEVER
Hugging Face cache
model weight cache
__pycache__/
*.pt
*.bin
*.safetensors
*.faiss
```

Environment setup:

```bash
conda create -n rag_cbwdm python=3.10 -y
conda activate rag_cbwdm
pip install -r requirements.txt
```

If the server CUDA version is fixed, install the matching PyTorch build from the official PyTorch index before or during requirements installation.

Hugging Face cache:

```bash
export HF_HOME=/path/to/big_disk/hf_cache
export HF_HUB_CACHE=/path/to/big_disk/hf_cache/hub
```

FEVER data options:

- Server has internet: run `scripts/download_fever_hf.py`.
- Server has no internet: upload `data/raw/fever/train.jsonl`, `data/raw/fever/dev.jsonl`, and `data/raw/fever/wiki-pages/`.

Models to prepare:

```text
Generator:
- Qwen/Qwen2.5-1.5B-Instruct for small/debug runs
- Qwen/Qwen2.5-7B-Instruct for formal experiments

Cross-encoder selector:
- cross-encoder/ms-marco-MiniLM-L-6-v2
- Optional: microsoft/deberta-v3-base

BGE baseline:
- BAAI/bge-reranker-base
- Optional: BAAI/bge-reranker-large
```

Windows local command pattern:

```cmd
D:\anaconda3\condabin\conda.bat run -n rag_cbwdm python scripts\...
```

Linux server command pattern:

```bash
conda activate rag_cbwdm
python scripts/...
```

## 12. Resource Estimate

Conservative formal FEVER-2 run:

```text
Corpus: FEVER wiki-pages sentence corpus
Retrieval: BM25 top20/top50
Scale: train 20k, dev 5k
Generator: Qwen2.5-7B
GPU: A100 40GB likely workable; A100 80GB more stable
Disk: about 100GB minimum, 100-200GB recommended for intermediate JSONL/cache
RAM: 64GB recommended
```

InfoGain-style aligned run:

```text
Corpus: 2018 Wikipedia / about 100-word passages / about 21M passages
Retrieval: top100 candidates
Generator workload: many millions of Qwen2.5-7B forward passes
Dense retrieval: may require 60GB+ vector space plus FAISS/doc store
Disk: 300GB+ minimum, 500GB safer
RAM: 64-128GB
GPU: A100 80GB or multi-GPU strongly preferred
```

## 13. Recommended Next Steps

1. Normalize output schema: add `num_docs` to all selection writers and optionally regenerate small selection outputs.
2. Set `trust_remote_code: false` in formal Qwen configs.
3. Add `scripts/12_select_bge_reranker.py`.
4. Add a single run script for FEVER-2 full pipeline with run-root variables.
5. Add `scripts/13_summarize_results.py` for metrics tables, per-class accuracy, prediction distribution, and method comparison.
6. Add batching to `03_compute_label_posteriors.py`.
7. Decide whether to store full candidate text in large posteriors or switch to ID-only rows plus corpus lookup.
8. Run a medium pilot, e.g. train 1k/dev 200 with Qwen2.5-7B, before launching train 20k/dev 5k.

## Appendix A. Important Commands

Syntax check:

```cmd
D:\anaconda3\condabin\conda.bat run -n rag_cbwdm python -m py_compile src\io_utils.py src\retrieval_bm25.py src\prompts.py src\label_logits.py src\cbwdm_score.py src\selector_dataset.py src\selector_model.py src\selector_cross_encoder.py src\metrics.py scripts\download_fever_hf.py scripts\00_prepare_fever.py scripts\01_prepare_fever_corpus.py scripts\02_retrieve_bm25.py scripts\03_compute_label_posteriors.py scripts\04_build_cbwdm_teacher.py scripts\05_train_selector.py scripts\06_select_with_selector.py scripts\07_eval_rag_classification.py scripts\08_select_naive_topm.py scripts\09_select_cbwdm_oracle_from_teacher.py scripts\10_train_cross_encoder_selector.py scripts\11_select_with_cross_encoder.py
```

Dependency checks:

```cmd
D:\anaconda3\condabin\conda.bat run -n rag_cbwdm python -c "import yaml, numpy, torch, transformers; print('basic deps ok')"
D:\anaconda3\condabin\conda.bat run -n rag_cbwdm python -c "import datasets; print('datasets', datasets.__version__)"
D:\anaconda3\condabin\conda.bat run -n rag_cbwdm python -c "from FlagEmbedding import FlagReranker; print('FlagEmbedding ok')"
```

Example help check:

```cmd
D:\anaconda3\condabin\conda.bat run -n rag_cbwdm python scripts\03_compute_label_posteriors.py --help
```

Repository status:

```cmd
git status --short
```

## Appendix B. Files Modified During This Audit

Added:

```text
PROJECT_STATUS_REPORT.md
```

No existing source, config, dependency, data, or output files were modified during this audit.

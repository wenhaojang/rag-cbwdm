# RAG-CBWDM Method-Failure Diagnostics Server Runbook

This runbook is validation-only. It reuses the completed formal retrieval and Stage 03 posterior artifacts. It must never target `artifacts/formal/`, never use `held_out_test`, and never use `--overwrite`.

Authoritative variables:

```bash
REPO=/root/rag-cbwdm
RUN=/root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13
PY=/root/miniconda3/envs/rag-cbwdm-baselines/bin/python
CONFIG=/root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml
```

## Gate 0 — environment and code confirmation

Run and review every command before Gate 1:

```bash
cd /root/rag-cbwdm && git rev-parse HEAD
```

Expected base commit: `4b914d200d9dbd5517f9b6107801a8a0117a3cdc` (plus the reviewed diagnostics commit, if these files have been synchronized as a later commit).

```bash
cd /root/rag-cbwdm && git status --short
/root/miniconda3/envs/rag-cbwdm-baselines/bin/python -c "import numpy, torch, transformers, yaml; print('baselines env OK')"
/root/miniconda3/envs/rag-cbwdm-retrieval/bin/python -c "import pyserini; print('retrieval env OK')"
test -d /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13 && echo "RUN OK"
test -f /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/formal/fever2_train_core_posteriors.jsonl && test -f /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/formal/fever2_validation_posteriors.jsonl && test -f /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/formal/calibration_candidates.json && echo "required artifacts OK"
```

Stop if the code revision is unexpected, the baselines environment cannot import its dependencies, `RUN` is wrong, or required formal artifacts are missing.

## Gate 1 — read-only supplement only

This command performs no generator inference, teacher rebuilding, selector training, or evaluation:

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/diagnostics/build_method_failure_server_supplement.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13 --output-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit
```

Review both outputs:

```bash
ls -lh /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/RAG_CBWDM_METHOD_FAILURE_AUDIT_SERVER_SUPPLEMENT.md /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/RAG_CBWDM_METHOD_FAILURE_AUDIT_SERVER_SUPPLEMENT.json
```

**DO NOT proceed to signed-gate oracle until supplement is reviewed.**

## Gate 2 — signed-gate oracle

First run only 10 validation queries in its own directory:

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/diagnostics/run_signed_gate_oracle.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13 --output-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_gate/smoke10 --validation-limit 10 --alignment-eps 0 --generator-model /root/models/Qwen2.5-1.5B-Instruct --device auto --resume
```

Inspect `trajectories.jsonl`, `selection.jsonl`, `predictions.jsonl`, `metrics.json`, and `manifest.json`. Confirm that every selected alignment is positive, selected documents contain title/text/source rank, prediction labels use `SUPPORTS/REFUTES`, and no path points into `artifacts/formal/`.

Full 500 validation command — **RUN ONLY AFTER REVIEW**:

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/diagnostics/run_signed_gate_oracle.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13 --output-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_gate/full500 --validation-limit 500 --alignment-eps 0 --generator-model /root/models/Qwen2.5-1.5B-Instruct --device auto --resume
```

No threshold search is authorized. `top_m` and teacher stop threshold remain config defaults unless a separately reviewed command explicitly sets them.

## Gate 3 — true joint-posterior oracle

Run only 10 validation queries first. `--candidate-limit 20` preserves the current BM25 pool while making the contract explicit:

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/diagnostics/run_joint_posterior_oracle.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13 --output-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/joint_posterior_oracle/smoke10 --validation-limit 10 --candidate-limit 20 --generator-model /root/models/Qwen2.5-1.5B-Instruct --device auto --resume
```

Inspect the ordered state contexts, cache metadata, positive-marginal stopping, prediction metrics, and `comparison.json`. Resume once to confirm completed output is checksum-reused.

Full 500 validation command — **RUN ONLY AFTER SIGNED-GATE REVIEW**:

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/diagnostics/run_joint_posterior_oracle.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13 --output-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/joint_posterior_oracle/full500 --validation-limit 500 --candidate-limit 20 --generator-model /root/models/Qwen2.5-1.5B-Instruct --device auto --resume
```

At 20 candidates and `top_m=4`, the upper bound is `20+19+18+17=74` joint-state posteriors per query, batched by state. Cache reuse avoids re-scoring already completed states without changing document order.

# signed_teacher_v1 Server Runbook

This runbook is experimental, validation-only after training, and fully isolated below `$RUN/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/`. It never modifies `artifacts/formal/`, production teachers/checkpoints/selections/evaluations, or `signed_gate/full500/`.

Authoritative paths:

```bash
REPO=/root/rag-cbwdm
RUN=/root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13
PY=/root/miniconda3/envs/rag-cbwdm-baselines/bin/python
CONFIG=/root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml
GENERATOR=/root/models/Qwen2.5-1.5B-Instruct
SELECTOR_INIT=/root/models/ms-marco-MiniLM-L-6-v2
```

Before Gate A:

```bash
cd /root/rag-cbwdm && git rev-parse HEAD && git branch --show-current && git status --short
/root/miniconda3/envs/rag-cbwdm-baselines/bin/python -m unittest tests.test_cbwdm_score tests.test_method_failure_diagnostics tests.test_signed_teacher_v1 tests.test_selector_loss_and_schema -v
test -d /root/models/Qwen2.5-1.5B-Instruct && test -d /root/models/ms-marco-MiniLM-L-6-v2 && test -f /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_gate/full500/manifest.json && echo "inputs OK"
```

Expected development base is `221394fff5455e3305d1c533916b1bec47ac05d1` plus this reviewed change.

## Gate A — full500 residual audit (CPU only)

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/diagnostics/analyze_signed_gate_full500.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13 --signed-gate-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_gate/full500 --output-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_gate/full500_error_audit --supplement-json /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/RAG_CBWDM_METHOD_FAILURE_AUDIT_SERVER_SUPPLEMENT.json
```

Review `SIGNED_GATE_FULL500_ERROR_AUDIT.md` and `.json`, including 106 zero-doc reasons, the 59-error taxonomy, and gold-evidence alignment. **STOP FOR REVIEW. Do not train.**

## Gate B — signed teacher build (CPU only)

Schema smoke on 10 validation queries:

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/diagnostics/build_signed_teacher_v1.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13 --split validation --output-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/teacher/schema_smoke10 --limit 10 --top-m 4 --teacher-stop-threshold 0.001 --alignment-eps 0 --b-plus 0.01 --b-minus 0.001 --neutral-sample-policy negative --resume
```

After inspecting every state/candidate field, build the 5000 train-core teacher without Qwen calls:

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/diagnostics/build_signed_teacher_v1.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13 --split train_core --output-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/teacher/full_train_core --top-m 4 --teacher-stop-threshold 0.001 --alignment-eps 0 --b-plus 0.01 --b-minus 0.001 --neutral-sample-policy negative --resume
```

Review `statistics.json`, particularly semantic terminal groups and ranking-skipped ratio versus the recorded old-teacher reference 77.48%. **STOP FOR REVIEW.**

## Gate C — selector training smoke and 20-validation pipeline

Train only 100 groups into an isolated checkpoint:

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/diagnostics/train_signed_selector_v1.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13 --teacher /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/teacher/full_train_core/teacher.jsonl --posteriors /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/formal/fever2_train_core_posteriors.jsonl --retrieval /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/formal/fever2_train_core_bm25_top20.jsonl --output-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/training/smoke100 --model-name /root/models/ms-marco-MiniLM-L-6-v2 --epochs 3 --lr 2e-5 --batch-size 8 --beta 0.25 --gamma 1.0 --teacher-temperature 0.1 --b-plus 0.01 --b-minus 0.001 --neutral-sample-policy negative --seed 13 --max-train-groups 100 --device auto --resume
```

Select 20 validation examples, allowing zero docs:

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/diagnostics/select_signed_selector_v1.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13 --posteriors /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/formal/fever2_validation_posteriors.jsonl --checkpoint-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/training/smoke100/checkpoint --output /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/selections/smoke20/selection.jsonl --top-m 4 --min-docs 0 --score-threshold 0.0 --validation-limit 20 --device auto --resume
```

Evaluate with the production evaluator and local generator:

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/07_eval_rag_classification.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --split validation --selection /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/selections/smoke20/selection.jsonl --output /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/evaluations/smoke20/predictions.jsonl --metrics-output /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/evaluations/smoke20/metrics.json --model-name /root/models/Qwen2.5-1.5B-Instruct --method-name signed_selector_v1 --limit 20 --resume
```

Run post-hoc diagnostics (gold is used only after selection):

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/diagnostics/analyze_signed_selector_v1.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13 --selection /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/selections/smoke20/selection.jsonl --predictions /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/evaluations/smoke20/predictions.jsonl --metrics /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/evaluations/smoke20/metrics.json --oracle-selection /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_gate/full500/selection.jsonl --output-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/reports/smoke20
```

Confirm zero-doc output is schema-valid, logits/scores are finite, and the manifest says `uses_gold_at_inference=false`. **STOP FOR REVIEW.**

## Gate D — full selector training

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/diagnostics/train_signed_selector_v1.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13 --teacher /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/teacher/full_train_core/teacher.jsonl --posteriors /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/formal/fever2_train_core_posteriors.jsonl --retrieval /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/formal/fever2_train_core_bm25_top20.jsonl --output-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/training/full --model-name /root/models/ms-marco-MiniLM-L-6-v2 --epochs 3 --lr 2e-5 --batch-size 8 --beta 0.25 --gamma 1.0 --teacher-temperature 0.1 --b-plus 0.01 --b-minus 0.001 --neutral-sample-policy negative --seed 13 --device auto --resume
```

## Gate E — primary full500 validation

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/diagnostics/select_signed_selector_v1.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13 --posteriors /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/formal/fever2_validation_posteriors.jsonl --checkpoint-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/training/full/checkpoint --output /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/selections/primary_min0_threshold0/selection.jsonl --top-m 4 --min-docs 0 --score-threshold 0.0 --validation-limit 500 --device auto --resume
```

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/07_eval_rag_classification.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --split validation --selection /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/selections/primary_min0_threshold0/selection.jsonl --output /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/evaluations/primary_min0_threshold0/predictions.jsonl --metrics-output /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/evaluations/primary_min0_threshold0/metrics.json --model-name /root/models/Qwen2.5-1.5B-Instruct --method-name signed_selector_v1 --limit 500 --resume
```

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/diagnostics/analyze_signed_selector_v1.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13 --selection /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/selections/primary_min0_threshold0/selection.jsonl --predictions /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/evaluations/primary_min0_threshold0/predictions.jsonl --metrics /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/evaluations/primary_min0_threshold0/metrics.json --oracle-selection /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_gate/full500/selection.jsonl --output-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/reports/primary_min0_threshold0
```

Then **STOP FOR REVIEW**.

## Gate F — legacy stopping control

Only after Gate E review, reuse the same full checkpoint. Do not retrain.

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/diagnostics/select_signed_selector_v1.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13 --posteriors /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/formal/fever2_validation_posteriors.jsonl --checkpoint-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/training/full/checkpoint --output /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/selections/legacy_min2_threshold0/selection.jsonl --top-m 4 --min-docs 2 --score-threshold 0.0 --validation-limit 500 --device auto --resume
```

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/07_eval_rag_classification.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --split validation --selection /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/selections/legacy_min2_threshold0/selection.jsonl --output /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/evaluations/legacy_min2_threshold0/predictions.jsonl --metrics-output /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/evaluations/legacy_min2_threshold0/metrics.json --model-name /root/models/Qwen2.5-1.5B-Instruct --method-name signed_selector_v1 --limit 500 --resume
```

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/diagnostics/analyze_signed_selector_v1.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13 --selection /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/selections/legacy_min2_threshold0/selection.jsonl --predictions /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/evaluations/legacy_min2_threshold0/predictions.jsonl --metrics /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/evaluations/legacy_min2_threshold0/metrics.json --oracle-selection /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_gate/full500/selection.jsonl --output-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/reports/legacy_min2_threshold0
```

Optional fixed4 control is supported with `--min-docs 0 --disable-score-threshold`, but it is **NOT AUTHORIZED FOR THE FIRST SERVER PASS**.

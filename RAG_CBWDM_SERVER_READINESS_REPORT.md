# RAG-CBWDM Server Readiness Report

## Overall status: READY FOR SERVER SMOKE

主方法 P0 链路已实现并通过不下载大模型的数值、schema 和 tiny-local fixture 验证；
尚不满足 formal experiment 条件。目标 Linux CUDA、真实 Qwen posterior 吞吐和阈值校准
必须在服务器 smoke/pilot 完成。

## 初始仓库状态

- 根目录：`<WORKSPACE>/rag_cbwdm`
- 分支：`main`
- HEAD：`4848ba8`（`Initialize FEVER preprocessing and BM25 retrieval`）
- 开始时已有用户修改：`configs/fever2_minimal.yaml`、`configs/fever3_minimal.yaml`、
  `requirements.txt`，以及论文、旧报告、Stage 3–11 源码等 untracked 文件。
- 本轮未 reset、clean、切分支、提交、删除 data/output/checkpoint，也未修改论文和
  `_external/InfoGain-RAG/**`。

## 实现摘要

### Theory/teacher

- 代码的 row-oriented 矩阵与论文 column-oriented `X_i,S` 严格转置等价。
- `Theta` 使用线性求解；raw gain 低于负 tolerance 会失败，而不是截断掩盖。
- mixture smoothing 同时应用于 `eta0` 与 gold target；旧 clip 行为有显式兼容模式。
- teacher 升级为 `rag_cbwdm_teacher.v2`，记录算法语义、参数、gain 和 stop decision。
- 完整 BW Hessian 未实现；真实方法名是 `euclidean_posterior_shift`。

### Posterior batching/resume

- `score_prompts` 一次 tokenize 多个 prompt，attention mask 定位每行最后有效 token，
  verbalizer 聚合后以 float32 softmax 输出并验证 finite/sum-to-one。
- 每个 query 的 query-only prompt 只计算一次，候选保持输入顺序并按 batch 评分。
- `output.jsonl.partial + output.manifest.json`；逐行 flush/fsync，完成后 `os.replace`
  原子发布。resume 检查 ID、输入/config/model/prompt/verbalizer fingerprint；只允许
  `--overwrite` 重算。
- manifest 记录数据/模型/revision/dtype/device/prompt/labels/hash/git/time/行数/
  candidate prompt 数。真实 resolved Hugging Face commit 仍依赖目标模型可用性。

### Selector/schema/evaluator

- 新增原稿 CE + positive/negative LogSumExp ranking loss，旧 listwise loss 不变。
- 训练日志包含 total/CE/rank、正负中性数、有效/跳过 ranking group 和 learning rate。
- cross-encoder 输入有 claim、candidate、实际 selected state；测试时不读 gold/teacher。
- feature、cross-encoder、naive、oracle 新输出统一 selection schema；oracle 明确不可部署。
- metrics v2 增加 per-class、confusion、prediction distribution、选文长度统计、
  evidence chars、NaN/Inf 和 missing prediction count。

### Server engineering

- `scripts/check_environment.py`
- `scripts/run_fever_cbwdm.sh` + Python orchestrator
- smoke/pilot/formal 分层 config
- run manifest、逐阶段 command/log、dry-run、resume、定向重跑

## 本地验证结果

| 检查 | 结果 |
|---|---|
| `py_compile`（所有本轮生产脚本/模块） | PASS |
| `python -m unittest discover` | PASS，12 tests |
| NumPy Theta/gain/smoothing/FEVER-2/FEVER-3/Rayleigh 对照 | PASS |
| batched-vs-single posterior mock | PASS |
| manuscript loss/空正负集合/legacy listwise | PASS |
| teacher v2/selection/manifest schema | PASS |
| fixture → teacher → tiny local BERT train → greedy selection | PASS，2 rows |
| selection schema validator | PASS |
| runner 全 stage `--dry-run` | PASS |
| `bash -n scripts/run_fever_cbwdm.sh`（Git Bash） | PASS |
| environment check | PASS for CPU checks；Torch `2.12.1+cpu`，无 CUDA |
| `pytest` | 未执行：当前 Conda/venv 未安装 pytest；requirements 已加入 |
| 真实 Qwen posterior/evaluator | 未执行：避免下载大模型 |
| Linux/CUDA/GPU/OOM/吞吐 | 未执行，必须服务器 smoke |

fixture 训练的两个 group 都没有同时存在正、负候选，因此 ranking 分量安全跳过；单元测试
另行覆盖了同时存在正负候选的非零 ranking loss 和反向传播。

## 数据与输出规模

本地检查时：

- `data/`：6 files，21,900,179 bytes。
- `outputs/`：78 files，53,786,313 bytes（包含此前输出及本轮 tiny fixture）。

这些仅是本地 smoke/fixture 资产，不是 formal 结果。

## 主要瓶颈与建议

1. posterior 是主要时间/显存瓶颈。按 query 将 `eta0 + candidates` 一次交给批接口；
   先从 batch 2 开始，依据 `nvidia-smi` 加倍。
2. BM25 corpus/retrieval 主要消耗 CPU/RAM/磁盘；可先完成并固定 hash。
3. cross-encoder 每一步重评分剩余候选，复杂度随 `top_n × top_m` 增长；candidate batch
   可独立调整。
4. manifest/日志位于 run 目录；posterior 另有细粒度 manifest，机器中断后不会把
   partial 误认作正式文件。

## Smoke / pilot / formal

- Smoke：Qwen2.5-1.5B、100 train/dev、top-10、top-M=2、batch 2/4。
- Pilot：1k–5k train、500 dev、top-20、top-M=4；只在 train/dev 校准 loss 和 stop。
- Formal：Qwen2.5-7B、完整固定 split、top-20、top-M=4、seeds 13/21/42；开始前冻结
  revisions、数据 hashes 和全部阈值。

服务器首轮精确命令：

```bash
python scripts/check_environment.py \
  --config configs/fever2_server_smoke.yaml \
  --output-root /mnt/experiments/rag_cbwdm \
  --cache-root /mnt/fast-cache/huggingface

bash scripts/run_fever_cbwdm.sh \
  --config configs/fever2_server_smoke.yaml \
  --run-name fever2_smoke_seed13 \
  --stages prepare,corpus,retrieve,posterior,teacher,train,select,eval \
  --limit 100 \
  --seed 13 \
  --output-root /mnt/experiments/rag_cbwdm \
  --cache-root /mnt/fast-cache/huggingface \
  --posterior-batch-size 2 \
  --selector-batch-size 4
```

## Blockers

### P0（formal blocker）

1. categorical BW local Hessian 缺少无歧义理论 closed form；当前只有 Euclidean 分支。
2. `b_plus/b_minus/gamma/beta` 与 selector stop threshold 尚未通过 train/dev pilot 冻结。
3. generator/selector config 的 revision 仍为 `null`；formal 必须固定 resolved revision。

### P1

4. 尚未在目标 Linux CUDA 验证 Qwen2.5-1.5B/7B、batch、OOM resume 和吞吐。
5. 真实 FEVER 全量输入 hashes、磁盘峰值和 runtime 未测。
6. 当前本地环境缺 pytest；服务器安装 requirements 后必须补跑 `pytest -q`。

### P2

7. formal 前应分析 state-aware 输入 token 长度和 truncation 覆盖率。
8. evaluator 的真实大模型端到端准确率与 metrics v2 尚未在服务器生成。

## 本轮实际 changed files

本轮新增/修改：

```text
RAG_CBWDM_THEORY_CODE_ALIGNMENT.md
RAG_CBWDM_SERVER_READINESS_REPORT.md
SERVER_RUNBOOK.md
configs/fever2_server_smoke.yaml
configs/fever2_server_pilot.yaml
configs/fever2_server_formal.yaml
requirements.txt
scripts/03_compute_label_posteriors.py
scripts/04_build_cbwdm_teacher.py
scripts/06_select_with_selector.py
scripts/07_eval_rag_classification.py
scripts/08_select_naive_topm.py
scripts/09_select_cbwdm_oracle_from_teacher.py
scripts/10_train_cross_encoder_selector.py
scripts/11_select_with_cross_encoder.py
scripts/check_environment.py
scripts/run_fever_cbwdm.py
scripts/run_fever_cbwdm.sh
src/cbwdm_score.py
src/label_logits.py
src/metrics.py
src/prompts.py
src/run_manifest.py
src/selection_schema.py
src/selector_cross_encoder.py
tests/test_cbwdm_score.py
tests/test_label_logits_and_manifest.py
tests/test_selector_loss_and_schema.py
```

`outputs/server_readiness_fixture/**` 是被 `.gitignore` 忽略的本地验证产物，不计入源码
changed-file list。开始前已存在且本轮未改的用户文件不冒充本轮产物。

最终 `git status --short`（其中包含开始前已有修改）：

```text
 M configs/fever2_minimal.yaml
 M configs/fever3_minimal.yaml
 M requirements.txt
?? INFOGAIN_RAG_INTEGRATION_AUDIT.md
?? PROJECT_STATUS_REPORT.md
?? RAG-CBWDM.tex
?? RAG_CBWDM_SERVER_READINESS_CODEX_TASK.md
?? RAG_CBWDM_SERVER_READINESS_REPORT.md
?? RAG_CBWDM_THEORY_CODE_ALIGNMENT.md
?? SERVER_RUNBOOK.md
?? configs/fever2_server_formal.yaml
?? configs/fever2_server_pilot.yaml
?? configs/fever2_server_smoke.yaml
?? scripts/03_compute_label_posteriors.py
?? scripts/04_build_cbwdm_teacher.py
?? scripts/05_train_selector.py
?? scripts/06_select_with_selector.py
?? scripts/07_eval_rag_classification.py
?? scripts/08_select_naive_topm.py
?? scripts/09_select_cbwdm_oracle_from_teacher.py
?? scripts/10_train_cross_encoder_selector.py
?? scripts/11_select_with_cross_encoder.py
?? scripts/check_environment.py
?? scripts/download_fever_hf.py
?? scripts/run_fever_cbwdm.py
?? scripts/run_fever_cbwdm.sh
?? src/cbwdm_score.py
?? src/label_logits.py
?? src/metrics.py
?? src/prompts.py
?? src/run_manifest.py
?? src/selection_schema.py
?? src/selector_cross_encoder.py
?? src/selector_dataset.py
?? src/selector_model.py
?? tests/fixtures/posteriors/
?? tests/test_cbwdm_score.py
?? tests/test_label_logits_and_manifest.py
?? tests/test_selector_loss_and_schema.py
?? 运行过的指令/
```

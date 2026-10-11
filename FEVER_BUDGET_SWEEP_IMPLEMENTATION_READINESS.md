# FEVER Budget-Controlled Accuracy–Tokens 实现就绪报告

**日期：** 2026-10-12  
**分支：** `feature/fever-formal-readiness`  
**参考基线：** `f275a3b9545b2b4a8c05918a4112ffdc1008a612`  
**状态：** 代码实现完成；本地合成验证通过；服务器正式 artifact preflight 尚未执行；未运行任何正式 generator evaluation。

## 1. 实现结论

FEVER document-budget-controlled sweep 已实现为 formal-v2 的严格增量层，而不是独立的宽松实验系统。框架包含30点计划、16个新 selection、10个 k=4 只读 alias、完整 evidence-vector 去重、动态 generator job 冻结、阶段性 fail-fast gate 和 Figure 2 CSV 汇总。

可以确认：

- 不需要重新训练任何 selector/reranker。
- 不需要重新生成 teacher、posterior 或 BM25 retrieval。
- 默认 selection 物化不加载 BGE、InfoGain 或 CBWDM 模型：RAG 从冻结 retrieval 生成前缀；BGE 从完整 score cache 重建；InfoGain 从完整 candidate score audit 重建；signed/KCBWDM 使用经过结构验证的 causal-prefix replay。
- 若 state-aware k=4 轨迹不能满足严格重放证明，框架提供同 checkpoint 的 `--inference-budget` 推理回退，不改变训练 top4 contract。
- runner 不硬编码18个 GPU jobs。它在服务器逐行比较5000条 ordered evidence/rendered-input vector 后生成 job list。根据用户提供的真实预检分布，预期结果是18个，但只有正式 audit 产出的数量具有执行效力。

## 2. 新增和修改文件

### 新增

- `configs/formal/fever_budget_sweep_document_caps_234.seed13.yaml`：冻结两个 generator、五种方法、k=2/3/4、三类正式来源根、BM25 SHA、48组 token analysis 来源和独立输出根。
- `src/fever_budget_sweep.py`：预算计划、只读 preflight、selection 物化、k=4 evaluation gate、向量去重、alias、job freeze、evaluation gate 和 CSV summary 的核心实现。
- `scripts/diagnostics/run_fever_budget_sweep.py`：阶段隔离 CLI。
- `tests/test_fever_budget_sweep.py`：预算实验专项测试。
- `FEVER_BUDGET_SWEEP_IMPLEMENTATION_READINESS.md`：本报告。

### 修改

- `src/baselines/infogain_protocol.py`：新增 k=2/3/4 的 corrected InfoGain budget protocol family/ID，同时保持原 k=4 ID 不变（`src/baselines/infogain_protocol.py:23`）。
- `scripts/12c_select_infogain_reranker.py`：为合法 `.2/min2/k{2,3,4}` 组合写入对应 budget-aware protocol ID；非冻结组合不冒充 final protocol（`scripts/12c_select_infogain_reranker.py:55`）。
- `src/preformal/registry.py`：新增只允许 `0 <= inference_budget <= frozen_top_m` 的解析器，训练 contract 不变（`src/preformal/registry.py:197`）。
- `scripts/preformal/27_select_signed_v1.py`：新增可选 `--inference-budget`；默认不传时保持历史 top4 fingerprint 行为，显式预算时 manifest 同时记录 `trained_max_docs=4` 和 inference budget（`scripts/preformal/27_select_signed_v1.py:86`）。

未修改任何 `plots/` 文件、旧正式 selection、checkpoint、metrics、predictions、manifest 或既有审计报告。

## 3. 冻结配置与目录隔离

配置冻结：

- dataset：`fever_binary_v2`
- split：validation
- rows：5000
- seed：13
- generators：Qwen2.5-1.5B、Qwen2.5-7B
- methods：RAG、BGE、corrected InfoGain、signed CBWDM、KCBWDM v2A
- budgets：2、3、4
- retrieval SHA：`a71fffda269f9112c6c243cee3384df072455a604eb801c838b69e32993786eb`
- 新根：`/root/experiments/rag_cbwdm/formal_v2_fever_budget_sweep_document_caps_234`

配置明确列出四个 protected roots；runner 拒绝把新根设置为旧 formal 或48组 token analysis 目录本身及其子目录。规划器强制30个唯一逻辑点和16个唯一 selection task（`src/fever_budget_sweep.py:143`, `src/fever_budget_sweep.py:268`）。

## 4. CLI 阶段行为

入口：

```bash
python scripts/diagnostics/run_fever_budget_sweep.py \
  --config configs/formal/fever_budget_sweep_document_caps_234.seed13.yaml \
  <stage>
```

六个 stage 为互斥选项（`scripts/diagnostics/run_fever_budget_sweep.py:33-38`）：

| Stage | 行为 | 是否写新根 | 是否加载模型 |
|---|---|---:|---:|
| `--preflight` | 只读核验8份 selection 来源、10份 k4 evaluation、generator/checkpoint/training binding、BGE cache、retrieval、48组 token analysis | 否 | 否；但会校验本地模型/tokenizer identity SHA |
| `--dry-run` | 展开30点、16 selection task、每点 selection 策略及待审计 evaluation disposition | 默认否；显式 `--plan-output` 时写计划 | 否 |
| `--materialize-selections` | 在新根发布16份 selection，使用现有 atomic publish/resume | 是 | 默认否；`--state-aware-inference` 回退时仅加载冻结 selector |
| `--audit-selections` | 重验来源和新 selection，构建5000行 evidence vector，生成 k4 alias、去重结果和冻结 job list | 是 | 否 |
| `--run-evaluations` | 只执行已冻结 job plan；运行前重新核验 audit fingerprint 和所有输入 SHA | 是 | 是，generator GPU；本轮未执行 |
| `--summarize` | 验证完成 metrics/token basis/alias，输出30行 Figure 2 CSV 和 summary manifest | 是 | 否 |

`--materialize-selections --resume` 使用 selection contract fingerprint 和 SHA 恢复。evaluation 命令始终带 `--resume`。audit/job/summary 使用原子 JSON/CSV 写入；任何输入 SHA 漂移都会 fail closed。

## 5. 五种 selection 的实现口径

### RAG / BM25

类型：**固定排序的合法前缀**。

- preflight 验证 retrieval schema、completed 状态、config/query SHA、manifest fingerprint、冻结 pool SHA、5000行顺序、每行至少4候选。
- k=2/3 直接从 retrieval candidate order 构造新 selection；不读取 label 决定证据。
- 两个 generator 共享两份 selection。

### BGE

类型：**完整 reranking 的合法前缀**。

- 验证 cache manifest/fingerprint/SHA、retrieval SHA、5000行 score rows。
- 用 cache 对全候选重新构造 k=4，并要求与正式 Top4 IDs 逐行一致。
- 只有正式 `score_threshold=null, min_docs=4` 通过后，才发布 k=2/3、`min_docs=k` 的 selection。
- 不加载 BGE 模型。

### Corrected InfoGain

类型：**原生 budget window 协议**，不是旧集合截断。

每个 k 严格执行：全候选 rank score 排序 → Top-k window → `filter_probability >= 0.2` → 仅在窗口内按排名补齐至2篇。preflight 会先使用 trace 重建 k=4，并要求5000行与正式 corrected selection 完全一致。核心 protocol ID 和重建逻辑分别在 `src/baselines/infogain_protocol.py:23` 与 `src/fever_budget_sweep.py` 的 InfoGain materializer 中。

### Signed CBWDM

类型：**state-aware greedy 的严格 causal-prefix replay**。

默认要求 k=4 trace 逐步满足：step 连续、selected ID 与 selected docs 一致、candidate score 排序正确、remaining count 更新正确、停止 step 合法。k 限制最大选择次数；若原轨迹在 k 前自然停止，则保留自然 stop；若选到第 k 篇，则新 selection 记录 `inference_budget_reached`。

训练 top4、threshold、checkpoint、teacher 不变。若严格 trace proof 不可用，可在 preflight/materialize 增加 `--state-aware-inference`，以同 checkpoint 实际运行最多 k 步；该回退命令由正式 source manifest 中的 posterior、training manifest 和 checkpoint 路径生成（`src/fever_budget_sweep.py:1149`）。

### KCBWDM v2A

与 signed 使用相同 deployable state-aware selector budget 机制；linear gate、kernel set objective 和 marginal-gain teacher 仍属于冻结训练 provenance，不在 validation 使用 gold 重算。默认严格 replay；必要时同样使用 `--state-aware-inference`。不重新训练、不改变 gate/threshold/bandwidth。

## 6. k=4 alias 等价性检查

`--preflight` 和 `--audit-selections` 对10个 k=4 点验证：

1. dataset/split/5000 ID 与顺序、query/label alignment；
2. retrieval manifest fingerprint、config/query/output SHA 和冻结 pool SHA；
3. selection output/manifest SHA、contract fingerprint、所有 contract input SHA；
4. selected IDs、顺序、title/text 与 retrieval pool 一致；
5. method-specific top4、threshold/min_docs、InfoGain trace、CBWDM stop/step 语义；
6. conditioned training manifest、checkpoint、teacher/posterior binding；
7. generator config/model/tokenizer/prompt/verbalizer identity；
8. native evaluation binding、config/selection/generator identity、predictions/metrics/manifest SHA；
9. predictions 的 ID/query/gold/method/selected IDs 与 selection 逐行一致；
10. 已完成48组 Evidence Token CSV/audit 的 selection/evaluation/generator SHA、tokenizer identity 和 token basis。

实现位置：native evaluation gate 在 `src/fever_budget_sweep.py:622`，48组 token analysis gate 在 `src/fever_budget_sweep.py:694`，只读总 preflight 在 `src/fever_budget_sweep.py:757`。

全部通过后才在**新根**写 alias manifest。alias 保存 source predictions/metrics/evaluation manifest 的路径与 SHA、当前 selection SHA、evidence-vector hash、generator-input fingerprint 和逐项检查结果。后续 summary 会再次核验 alias 及 source SHA（`src/fever_budget_sweep.py:1737`）。旧 matched-main manifest 不会被改写，也不会伪装成 budget-frontier。

## 7. Evidence-vector 去重与 GPU jobs

每个逻辑点的向量逐行包含：

- sample ID、query；
- ordered doc IDs；
- 与正式 evaluator 完全一致的 rendered evidence text；
- 实际 num_docs。

向量之外还绑定 generator manifest/config、model/tokenizer、prompt、verbalizer 和 context limit fingerprint。只有两者同时一致才可复用。向量生成和去重分别在 `src/fever_budget_sweep.py:1370`、`src/fever_budget_sweep.py:1400`。

去重优先使用已验证 k=4；其余 k=2、k=3 按首次出现的唯一向量生成 job。audit 动态写出 `evaluation_jobs.json` 和只供复核的 `evaluation_commands.sh`，不会执行它们（`src/fever_budget_sweep.py:1532`）。真正运行前还会重验 audit/job fingerprint 及 selection、selection manifest、generator manifest、generator config SHA（`src/fever_budget_sweep.py:1760`）。

依据用户提供的服务器分布：9个 generator-method 组合具有三个不同向量，每组新增2次；Qwen7B signed 三个预算完全一致、新增0次，因此**预期为18次 generator evaluation**。专项测试用该分布计算得到18，没有在 runner 中写死此数字。

若正式 provenance gate 失败、任何 vector 或 generator-input fingerprint 不同，实际 job 数会由 audit 相应调整或直接拒绝继续。

## 8. Evidence Tokens 与 Figure 2 CSV

新 evaluation 本身使用 `07_eval_rag_classification.py` 的 generator tokenizer 和正式 evidence rendering 统计 tokens。k=4/重合 alias 使用已完成48组离线分析的结果，但先按 selection/evaluation/generator SHA 和 tokenizer identity 严格绑定。

summary 输出：

`/root/experiments/rag_cbwdm/formal_v2_fever_budget_sweep_document_caps_234/summaries/fever_budget_accuracy_tokens.csv`

字段完全覆盖：

`dataset_id, generator_id, method_id, budget_k, accuracy, macro_f1, avg_num_docs, avg_evidence_tokens, selection_path, evaluation_path, source_or_alias, analysis_status`

CSV 恰好30行，不插值、不删除平台期，可直接作为 Figure 2 输入。summary 实现在 `src/fever_budget_sweep.py:1805`。

## 9. 训练、模型和服务器环境

### 不需要

- selector/reranker training；
- teacher/posterior materialization；
- BM25 retrieval/index；
- 默认 selection 路径中的 BGE/InfoGain/CBWDM 模型加载。

### 服务器需要

- 配置中的 `/root/miniconda3/envs/rag-cbwdm-mistral/bin/python` 环境；
- 当前项目代码部署到 `/root/rag-cbwdm`；
- 三类 frozen formal 根和48组 token analysis 根可读；
- Qwen2.5-1.5B/7B 模型与 tokenizer 路径和 manifest identity 一致；
- 正式 generator evaluation 阶段需要 GPU；
- 只有使用 state-aware inference fallback 时，selection 阶段才需要加载 frozen cross-encoder/checkpoint，建议 GPU但不涉及训练。

## 10. 测试结果

专项测试覆盖：

- 30点唯一矩阵与16 selection identity；
- RAG/BGE prefix；
- InfoGain 窗口过滤反例、k4重建和 budget ID；
- signed/KCBWDM causal prefix、自然提前停止和同 checkpoint fallback command；
- train top4 / inference budget 解耦；
- validation gold isolation；
- ordered IDs/text/rendered evidence 去重；
- Qwen7B signed 三预算重合 fixture；
- protected-root isolation；
- k4 alias SHA tamper detection；
- job plan fingerprint/input SHA gate；
- evaluator rendering parity；
- 30行 Figure 2 CSV schema。

最终验证结果：

- Budget Sweep 专项：20项，全部包含在最终全仓通过结果中；
- 全仓：538 passed，2 subtests passed；
- `compileall`：通过；
- `git diff --check`：通过。

本地没有 `/root/experiments/rag_cbwdm`，因此未执行真实 `--preflight`，不能声称10个 k=4 alias 已在服务器验收。

## 11. 推荐服务器执行顺序

### Phase 0：部署后只读检查

```bash
cd /root/rag-cbwdm

/root/miniconda3/envs/rag-cbwdm-mistral/bin/python \
  scripts/diagnostics/run_fever_budget_sweep.py \
  --config configs/formal/fever_budget_sweep_document_caps_234.seed13.yaml \
  --preflight

/root/miniconda3/envs/rag-cbwdm-mistral/bin/python \
  scripts/diagnostics/run_fever_budget_sweep.py \
  --config configs/formal/fever_budget_sweep_document_caps_234.seed13.yaml \
  --dry-run \
  --plan-output /root/experiments/rag_cbwdm/formal_v2_fever_budget_sweep_document_caps_234/plan/plan.json
```

若 preflight 的 causal trace proof 失败，但所有 checkpoint/provenance 正常，可先用 `--preflight --state-aware-inference` 检查回退输入。不要降低 gate 或手工截断 JSONL。

### Phase 1：发布16个 selection

```bash
/root/miniconda3/envs/rag-cbwdm-mistral/bin/python \
  scripts/diagnostics/run_fever_budget_sweep.py \
  --config configs/formal/fever_budget_sweep_document_caps_234.seed13.yaml \
  --materialize-selections \
  --resume
```

若必须回退为同 checkpoint 推理，则额外加 `--state-aware-inference`。该选项只影响 signed/KCBWDM k=2/3 selection，不训练模型、不运行 generator。

### Phase 2：正式 selection audit 和 job freeze

```bash
/root/miniconda3/envs/rag-cbwdm-mistral/bin/python \
  scripts/diagnostics/run_fever_budget_sweep.py \
  --config configs/formal/fever_budget_sweep_document_caps_234.seed13.yaml \
  --audit-selections

jq '{logical_point_count,new_selection_count,native_k4_alias_count,
     evidence_vector_alias_count,evaluation_job_count}' \
  /root/experiments/rag_cbwdm/formal_v2_fever_budget_sweep_document_caps_234/audit/selection_audit.json
```

预期但不强制：`evaluation_job_count=18`。任何其他值应先检查 audit 中的逐点 vector hash 和原因，而不是手工改 job list。

### Phase 3：等待人工确认后运行 GPU evaluation

```bash
/root/miniconda3/envs/rag-cbwdm-mistral/bin/python \
  scripts/diagnostics/run_fever_budget_sweep.py \
  --config configs/formal/fever_budget_sweep_document_caps_234.seed13.yaml \
  --run-evaluations
```

本轮禁止并且没有执行该命令。

### Phase 4：生成 Figure 2 CSV

```bash
/root/miniconda3/envs/rag-cbwdm-mistral/bin/python \
  scripts/diagnostics/run_fever_budget_sweep.py \
  --config configs/formal/fever_budget_sweep_document_caps_234.seed13.yaml \
  --summarize
```

确认 CSV 30行、所有 `analysis_status=completed` 后，再由后续独立任务接入 Figure 2；本轮没有生成或修改任何图。

## 12. 当前放行状态

**代码实现：READY。**  
**本地合成与回归：READY。**  
**服务器输入 provenance：REQUIRES SERVER PREFLIGHT。**  
**正式 GPU evaluation：NOT RUN / AWAITING USER CONFIRMATION。**

下一步应仅执行 Phase 0 的只读服务器 preflight，并把输出与动态 job count 返回确认；在用户确认前不进入 Phase 3。

# FEVER Budget Sweep Extra Generators 实现与安全审计

**本地基线：** `b04412a540535ac6fd6ef14ef5d96520e4e42d2f`

**分支：** `feature/fever-formal-readiness`

**目标配置：** `configs/formal/fever_budget_sweep_extra_05b_mistral_caps234.seed13.yaml`
**本地状态：** `LOCAL_CODE_READY_FOR_SERVER_PREFLIGHT`

## 1. 结论

代码已在原六阶段 FEVER Budget Sweep 框架内完成最小泛化，没有复制 runner，也没有改变五种方法的预算协议。新配置严格限定 Qwen2.5-0.5B 与 Mistral-7B-v0.3；旧配置仍限定 Qwen2.5-1.5B 与 Qwen2.5-7B。两份配置分别展开30个唯一逻辑点和16个 selection task。

本地没有 `/root/experiments/rag_cbwdm`，因此不能证明服务器 checkpoint、score audit、causal trace、模型文件或10个 k4 source 当前仍完整。下一状态必须是服务器只读 preflight；本地结论不等同于正式实验可启动。

## 2. 实现摘要

- 受控 generator 集合与配置驱动计划：`src/fever_budget_sweep.py:81`、`src/fever_budget_sweep.py:213`、`src/fever_budget_sweep.py:382`。
- 输出根同时禁止位于 protected root 内部或成为其父目录：`src/fever_budget_sweep.py:171`。
- 上一轮 RAG/BGE k2/k3 selection 可作为严格只读复用源；逐行重建比较、formal binding、input SHA 与 contract fingerprint 全部通过后才使用：`src/fever_budget_sweep.py:1166`。
- InfoGain 仍执行 full ranking → Top-k window → filter 0.2 → window 内补足 min_docs=2：`src/fever_budget_sweep.py:1482`。
- signed/KCBWDM 仍使用验证过的 causal-prefix replay，或显式 `--state-aware-inference` 同 checkpoint 回退：`src/fever_budget_sweep.py:1567`。
- alias 仍在同一 generator、同一 method 内按 ordered evidence vector 去重，并绑定 prompt/verbalizer/model/tokenizer/context identity：`src/fever_budget_sweep.py:1980`、`src/fever_budget_sweep.py:2044`、`src/fever_budget_sweep.py:2107`。
- summary 加入完整矩阵、有限值、预算上界和 generator path 检查：`src/fever_budget_sweep.py:2404`。
- 新增只读输入的30+30→60合并审计：`src/fever_budget_sweep.py:2593`、`scripts/diagnostics/merge_fever_budget_summaries.py:34`。

旧配置的计划指纹在修改前后均为：

`db6a27554db3aca0085fe6ac944a9a5ccda80e972ef7427596d6b896c12b93bf`

## 3. 30点任务矩阵

`GPU` 列是 audit 后的上限，不是预先冻结的 job 数。

| Generator | Method | k | Selection 来源 | k4 alias | GPU |
|---|---|---:|---|---|---:|
| Qwen0.5B | RAG | 2 | 共享旧 selection，严格验证；缺失时冻结 retrieval 重建 | 否 | 0/1 |
| Qwen0.5B | RAG | 3 | 同上 | 否 | 0/1 |
| Qwen0.5B | RAG | 4 | 正式 native selection | 是 | 0 |
| Qwen0.5B | BGE | 2 | 共享旧 selection，严格验证；缺失时冻结 score cache 重建 | 否 | 0/1 |
| Qwen0.5B | BGE | 3 | 同上 | 否 | 0/1 |
| Qwen0.5B | BGE | 4 | 正式 native selection | 是 | 0 |
| Qwen0.5B | InfoGain | 2 | 本 generator 完整 score audit 离线重建 | 否 | 0/1 |
| Qwen0.5B | InfoGain | 3 | 同上 | 否 | 0/1 |
| Qwen0.5B | InfoGain | 4 | corrected native selection | 是 | 0 |
| Qwen0.5B | signed CBWDM | 2 | k4 causal replay / 同 checkpoint inference | 否 | 0/1 |
| Qwen0.5B | signed CBWDM | 3 | 同上 | 否 | 0/1 |
| Qwen0.5B | signed CBWDM | 4 | 正式 native selection | 是 | 0 |
| Qwen0.5B | KCBWDM v2A | 2 | k4 causal replay / 同 checkpoint inference | 否 | 0/1 |
| Qwen0.5B | KCBWDM v2A | 3 | 同上 | 否 | 0/1 |
| Qwen0.5B | KCBWDM v2A | 4 | 正式 native selection | 是 | 0 |
| Mistral7B | RAG | 2 | 与 Qwen0.5B 共享同一合法 selection | 否 | 0/1 |
| Mistral7B | RAG | 3 | 同上 | 否 | 0/1 |
| Mistral7B | RAG | 4 | 正式 native selection | 是 | 0 |
| Mistral7B | BGE | 2 | 与 Qwen0.5B 共享同一合法 selection | 否 | 0/1 |
| Mistral7B | BGE | 3 | 同上 | 否 | 0/1 |
| Mistral7B | BGE | 4 | 正式 native selection | 是 | 0 |
| Mistral7B | InfoGain | 2 | 本 generator 完整 score audit 离线重建 | 否 | 0/1 |
| Mistral7B | InfoGain | 3 | 同上 | 否 | 0/1 |
| Mistral7B | InfoGain | 4 | corrected native selection | 是 | 0 |
| Mistral7B | signed CBWDM | 2 | k4 causal replay / 同 checkpoint inference | 否 | 0/1 |
| Mistral7B | signed CBWDM | 3 | 同上；可能与 k4 完全等价 | 否 | 0/1 |
| Mistral7B | signed CBWDM | 4 | 正式 native selection | 是 | 0 |
| Mistral7B | KCBWDM v2A | 2 | k4 causal replay / 同 checkpoint inference | 否 | 0/1 |
| Mistral7B | KCBWDM v2A | 3 | 同上；可能与 k4 完全等价 | 否 | 0/1 |
| Mistral7B | KCBWDM v2A | 4 | 正式 native selection | 是 | 0 |

16个 selection task 的组成：

- RAG：2个共享预算 task。
- BGE：2个共享预算 task。
- InfoGain：2 generators × 2 budgets = 4。
- signed CBWDM：2 × 2 = 4。
- KCBWDM v2A：2 × 2 = 4。

10个 k4 点只写新根中的 alias manifest，不重跑 evaluation。20个 k2/k3 点最多产生20个 GPU jobs。如果且仅如果 Mistral signed 与 KCBWDM 的 k3 分别逐条等于各自 k4，且其余18点均为新 evidence vector，则 job 数为18；更多平台期会进一步减少，任一预期平台期不成立则相应增加。

## 4. 安全审计

1. **旧配置兼容性：** generator/method/budget 顺序、30点、16 task、路径和 plan fingerprint 保持不变。
2. **输出隔离：** 新根为 `formal_v2_fever_budget_sweep_extra_05b_mistral_caps234`；旧 Budget Sweep 根加入 protected roots。父子方向的路径重叠均拒绝。
3. **无训练路径：** runner 没有新增 training/teacher/posterior 命令；state-aware fallback 只调用冻结 selector checkpoint。
4. **无 gold selection：**所有 budget row 都要求 `uses_gold_at_test=false`，协议测试验证 label 改变不影响 selection。
5. **无跨 generator evaluation/alias：** RAG/BGE 只共享 selection；evaluation point、generator manifest/config、输入 fingerprint 与输出路径仍逐 generator 绑定。
6. **动态 job：** job plan 只由完整5000行 evidence vector audit 产生；代码和配置都没有常量18。
7. **k4 anchor：** preflight 同时验证 selection/evaluation/generator provenance 与48组 token analysis；60点工具再次比较20个 k4 的四项数值和 source path。
8. **输入不可覆盖：**60点工具拒绝让 output CSV/manifest 覆盖任一输入文件。

## 5. 本地测试

- Budget Sweep 专项：`50 passed`。
- formal registry/matrix、InfoGain final protocol、Evidence Token analysis 相关回归：`142 passed`。
- 全仓：`568 passed, 2 subtests passed`。
- 新旧配置 `--dry-run`：均为 `points=30`、`new_selections=16`。
- `compileall`：通过。

Windows 沙箱首次全仓运行因系统 TEMP 不可写产生54个环境失败；把测试进程的 TEMP/TMP 指向仓库 `scratch/` 后，另有3个 FM2 测试因临时目录名含 `fever` 触发其安全 gate。改用中性可写临时目录后的最终全仓结果为全部通过。

## 6. 服务器待核实风险

- 两个 generator config 指向的本地 model/tokenizer 是否仍存在且 SHA/revision 未漂移。
- 10份 native k4 evaluation 与 `main_48_v3` 当前文件 SHA 是否仍一致。
- Qwen0.5B/Mistral InfoGain selection 是否确实保留完整 candidate score audit，并绑定正确 checkpoint/training manifest。
- signed/KCBWDM 的5000行 causal trace 是否包含默认 replay 所需的逐步 candidate scores；否则必须审查 `--state-aware-inference` 回退。
- 上一轮4份共享 RAG/BGE budget selection 是否仍完整。全部缺失可合法重建；半缺失或不等价会 fail closed。
- Mistral signed/KCBWDM 的 k3/k4 平台期与实际 GPU job 数只能由服务器 audit 证明。

## 7. 下一步

严格按 `FEVER_BUDGET_EXTRA_GENERATORS_DEPLOYMENT.md` 先运行只读 preflight 和 dry-run。只有 preflight PASS、selection materialization 完成且 audit job plan 经人工检查后，才能批准 GPU 阶段。

**最终状态：`LOCAL_CODE_READY_FOR_SERVER_PREFLIGHT`**

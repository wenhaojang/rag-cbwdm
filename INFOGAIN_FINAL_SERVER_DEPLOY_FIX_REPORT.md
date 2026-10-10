# InfoGain 最终协议实验：服务器最小部署修复报告

日期：2026-10-11
修复基线：`5de32367b8d975c80b7b96b3a9adbca39718ade4`
目标分支：`feature/fever-formal-readiness`

## 1. 结论与执行边界

本次修改只涉及 InfoGain 最终协议实验的服务器配置、无模型加载的启动前验证、CPU/GPU tmux 启动可靠性和相应测试。没有修改冻结的 `Top-4 → threshold=0.2 → min_docs=2` 选择规则，没有重训 InfoGain，没有重算 teacher/posterior，没有访问 held-out/test，没有修改检索数据或伪造 manifest，也没有启动正式 CPU selection 或 GPU evaluation。

最终实验根目录保持不变：

```text
/root/experiments/rag_cbwdm/formal_v2_infogain_top4_filter02_final
```

## 2. 故障根因

首次 CPU tmux 在第一组 `fever_qwen05` 的 retrieval provenance 校验前期失败。计划文件中的 FEVER retrieval 指向不存在的旧布局：

```text
/root/experiments/rag_cbwdm/formal_v2_full_development/fever_binary_v2/shared/retrieval/
```

真实且已冻结的 FEVER BM25 候选池位于 `formal_v2_external_inputs`。旧 launcher 直接在 tmux worker 中依次运行 Python stage，只在固定位置写一个总退出码；Python 在进入组内 `_run_logged()` 前抛出的 traceback 没有持久化，因此 tmux 消失后只能看到非零退出码，无法看到根因。

## 3. 修改文件与具体差异

### `configs/formal/infogain_top4_filter02_final.development.server.yaml`

- FEVER 四组通过 YAML anchor 统一改用真实 `validation.jsonl` 和 `validation.manifest.json`。
- 固定 FEVER 完整 SHA-256：`a71fffda269f9112c6c243cee3384df072455a604eb801c838b69e32993786eb`。
- FM2 路径保持不变；新增已知 SHA-256 前缀约束 `f371600059769a24`。
- checkpoint、generator、selection、evaluation 路径和资源参数未改动。

### `scripts/infogain_final_protocol.py`

- 新增 `source-preflight` stage；它对八组逐组 fail-fast，并且不加载 InfoGain 模型。
- 复用已有 `validate_retrieval_source()`、`validate_fm2_retrieval_manifest()` 和 `validate_formal_training_binding()`，不另造 provenance 规则。
- FEVER 校验：文件与 manifest 存在、schema、`completed=true`、`split=validation`、5000 行、文件实际 SHA、manifest SHA 字段和计划冻结 SHA 一致。
- FM2 校验：已有正式校验器的 dataset/formal role/protocol/候选池合同/逐行校验仍全部执行；另验证 1169 行和冻结 SHA 前缀。
- 八组 checkpoint 均验证目录、training manifest、`infogain_config.json`、`heads.pt`、encoder 目录及 encoder 权重；完整 formal-v2 training binding、dataset、generator、seed、retrieval protocol 和 retrieval source SHA 必须一致。
- selection 路径必须位于最终实验根目录内，并显式排除旧 full-development、旧 threshold-0.2 和已完成 FEVER KCBWDM 根目录。若已有完整 selection+manifest，只允许通过完整 audit 后 resume；单边残留会拒绝启动。
- 新增 `cpu-pipeline` 与 `gpu-pipeline`：每一阶段都有独立 log、`.exitcode`、`.status.json`；失败立即停止并为后续阶段写 `.skipped`；最终总状态写 `pipeline_status.json` 和 `overall.exitcode`。
- launcher 聚合日志保存合并后的 stdout/stderr。子进程 traceback、尚未进入原 `_run_logged()` 的异常以及启动器自身异常均有持久记录。

### `scripts/run_infogain_final_cpu_tmux.sh`

- 创建 tmux 前，在前台执行全部八组 `source-preflight`；任一组失败时不创建 tmux。
- 启动前后各检查一次固定 session 名，避免现有会话和并发启动竞态。
- 每次运行创建唯一目录：`$RUN_ROOT/attempts/cpu_<UTC时间>_<PID>`；旧失败目录和旧固定退出码不会被覆盖。
- tmux worker 的完整控制台输出立即追加到 `launcher.console.log`，内部 pipeline 另写 `launcher.log` 与阶段日志。

### `scripts/run_infogain_final_gpu_tmux.sh`

- 同样使用唯一 GPU attempt 目录、重复会话保护和完整控制台日志。
- 创建 tmux 前在前台执行全八组 GPU readiness preflight。
- worker 严格按 `preflight → smoke(32) → evaluation → summary` 执行，任一阶段失败即停止。
- generator、checkpoint、evaluation 参数和 GPU 资源行为未变。

### `tests/test_infogain_final_protocol.py`

新增/加强以下回归覆盖：

- FEVER 四组真实路径、manifest 路径和完整冻结 SHA；FM2 四组冻结 SHA 前缀。
- 缺失 retrieval manifest 时给出明确错误。
- 文件/manifest SHA 篡改及计划冻结 SHA 不一致时拒绝执行。
- 八组 source preflight 在任一组失败时立即停止。
- launcher 子进程失败时保留 traceback、stage log、聚合 log、状态 JSON 和非零退出码。
- CPU/GPU shell launcher 均在 `tmux new-session` 前执行前台 gate，并使用唯一 attempt 目录。
- 原有 Top-4 过滤协议和 `filter_threshold=None` 兼容测试继续保留并通过。

## 4. 修复后的八组 retrieval source

| Group | Dataset | Generator | Retrieval / manifest | Row count | SHA 条件 |
|---|---|---|---|---:|---|
| `fever_qwen05` | `fever_binary_v2` | Qwen2.5-0.5B | `.../formal_v2_external_inputs/fever_binary_v2/fever_bm25_v1/full_seed13/validation.jsonl` / `validation.manifest.json` | 5000 | 完整 SHA `a71fff…86eb` |
| `fever_qwen15` | `fever_binary_v2` | Qwen2.5-1.5B | 同一 FEVER 冻结候选池 | 5000 | 完整 SHA `a71fff…86eb` |
| `fever_qwen7` | `fever_binary_v2` | Qwen2.5-7B | 同一 FEVER 冻结候选池 | 5000 | 完整 SHA `a71fff…86eb` |
| `fever_mistral7` | `fever_binary_v2` | Mistral-7B | 同一 FEVER 冻结候选池 | 5000 | 完整 SHA `a71fff…86eb` |
| `fm2_qwen05` | `fm2_official_closed_page_v1` | Qwen2.5-0.5B | `.../formal_v2_external_inputs/fm2_official_closed_page_v1/prepared_v2/fm2_dev_official_pool.jsonl` / `fm2_prepare.manifest.json` | 1169 | manifest 全量 SHA 校验且前缀 `f371600059769a24` |
| `fm2_qwen15` | `fm2_official_closed_page_v1` | Qwen2.5-1.5B | 同一 FM2 官方候选池 | 1169 | 同上 |
| `fm2_qwen7` | `fm2_official_closed_page_v1` | Qwen2.5-7B | 同一 FM2 官方候选池 | 1169 | 同上 |
| `fm2_mistral7` | `fm2_official_closed_page_v1` | Mistral-7B | 同一 FM2 官方候选池 | 1169 | 同上 |

FEVER 的完整路径为：

```text
/root/experiments/rag_cbwdm/formal_v2_external_inputs/fever_binary_v2/fever_bm25_v1/full_seed13/validation.jsonl
/root/experiments/rag_cbwdm/formal_v2_external_inputs/fever_binary_v2/fever_bm25_v1/full_seed13/validation.manifest.json
```

## 5. 服务器部署后、启动前验证

部署后先确认 commit 和工作区，再只运行无模型加载的 source preflight：

```bash
cd /root/rag-cbwdm
git status --short
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD

RUN_ROOT=/root/experiments/rag_cbwdm/formal_v2_runs/infogain_top4_filter02_final_20261011
CHECK_DIR="$RUN_ROOT/attempts/manual_source_preflight_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$CHECK_DIR"
/root/miniconda3/envs/rag-cbwdm-mistral/bin/python \
  scripts/infogain_final_protocol.py \
  --plan configs/formal/infogain_top4_filter02_final.development.server.yaml \
  --stage source-preflight \
  --audit-output "$CHECK_DIR/source_preflight.json" \
  2>&1 | tee "$CHECK_DIR/source_preflight.log"
printf '%s\n' "${PIPESTATUS[0]}" > "$CHECK_DIR/source_preflight.exitcode"
```

验收：退出码为 0，JSON 顶层 `status=PASS`，恰有八组，每组 `selection_output_state` 为 `absent_ready` 或经完整 audit 确认的 `completed_compatible_resume`。该命令只读 retrieval/checkpoint/manifest 并检查输出路径，不加载 selector 模型，也不生成 selection。

## 6. CPU tmux：启动、进度与失败排查

得到下一步确认后启动：

```bash
cd /root/rag-cbwdm
bash scripts/run_infogain_final_cpu_tmux.sh
```

定位本次独立记录并查看进度：

```bash
RUN_ROOT=/root/experiments/rag_cbwdm/formal_v2_runs/infogain_top4_filter02_final_20261011
CPU_ATTEMPT=$(cat "$RUN_ROOT/latest_cpu_attempt.txt")
printf '%s\n' "$CPU_ATTEMPT"
tmux has-session -t infogain-final-cpu-selection
tmux attach -t infogain-final-cpu-selection
tail -F "$CPU_ATTEMPT/launcher.log"
```

退出后不要根据 tmux 是否存在判断成败；读取持久状态：

```bash
cat "$CPU_ATTEMPT/pipeline_status.json"
cat "$CPU_ATTEMPT/overall.exitcode"
cat "$CPU_ATTEMPT/01_source_preflight.exitcode"
cat "$CPU_ATTEMPT/02_selection.exitcode"
cat "$CPU_ATTEMPT/03_protocol_audit.exitcode"
```

失败排查顺序：

```bash
sed -n '1,240p' "$CPU_ATTEMPT/launcher.console.log"
sed -n '1,240p' "$CPU_ATTEMPT/launcher.log"
sed -n '1,240p' "$CPU_ATTEMPT/02_selection.log"
find "$CPU_ATTEMPT" -maxdepth 1 -type f -printf '%f\n' | sort
```

组内 selector 的既有详细日志仍位于：

```text
/root/experiments/rag_cbwdm/formal_v2_runs/infogain_top4_filter02_final_20261011/selection_logs/<group>.log
```

CPU 验收：`overall.exitcode=0`、三阶段退出码均为 0、`protocol_audit.json` 为 PASS 且含八组、八组 selection/manifest 均存在并通过完整协议与 provenance audit。

## 7. GPU 后续 preflight、smoke 与正式 evaluation

只有 CPU selection 与 protocol audit 验收完成后才可运行。单独的只读 GPU readiness preflight：

```bash
cd /root/rag-cbwdm
/root/miniconda3/envs/rag-cbwdm-mistral/bin/python \
  scripts/infogain_final_protocol.py \
  --plan configs/formal/infogain_top4_filter02_final.development.server.yaml \
  --stage preflight \
  --audit-output /root/experiments/rag_cbwdm/formal_v2_runs/infogain_top4_filter02_final_20261011/audit/evaluation_preflight.manual.json
```

后续手工分阶段命令为：

```bash
# 仅 fever_qwen05，32 条 smoke
/root/miniconda3/envs/rag-cbwdm-mistral/bin/python \
  scripts/infogain_final_protocol.py \
  --plan configs/formal/infogain_top4_filter02_final.development.server.yaml \
  --stage smoke --smoke-limit 32

# 八组正式 validation evaluation
/root/miniconda3/envs/rag-cbwdm-mistral/bin/python \
  scripts/infogain_final_protocol.py \
  --plan configs/formal/infogain_top4_filter02_final.development.server.yaml \
  --stage evaluation

# 八组结果汇总
/root/miniconda3/envs/rag-cbwdm-mistral/bin/python \
  scripts/infogain_final_protocol.py \
  --plan configs/formal/infogain_top4_filter02_final.development.server.yaml \
  --stage summary
```

推荐在确认 GPU 可运行后使用可靠 tmux launcher，一次执行全部 fail-fast 阶段：

```bash
bash scripts/run_infogain_final_gpu_tmux.sh
RUN_ROOT=/root/experiments/rag_cbwdm/formal_v2_runs/infogain_top4_filter02_final_20261011
GPU_ATTEMPT=$(cat "$RUN_ROOT/latest_gpu_attempt.txt")
tail -F "$GPU_ATTEMPT/launcher.log"
cat "$GPU_ATTEMPT/pipeline_status.json"
cat "$GPU_ATTEMPT/overall.exitcode"
```

GPU 验收：四阶段退出码均为 0；八组 evaluation 的 `predictions.jsonl`、`metrics.json`、`metrics.manifest.json` 完整且 binding/SHA/行数通过；summary 生成恰好八行结果。

## 8. 结果汇总办法

成功 summary 输出到：

```text
/root/experiments/rag_cbwdm/formal_v2_infogain_top4_filter02_final/summary/infogain_final_results.csv
/root/experiments/rag_cbwdm/formal_v2_infogain_top4_filter02_final/summary/infogain_final_results.md
/root/experiments/rag_cbwdm/formal_v2_infogain_top4_filter02_final/summary/infogain_final_results.tex
```

每行按 dataset × generator 唯一标识，记录 `accuracy`、`macro_f1`、样本数、missing/NaN、平均证据文档数、0–4 文档直方图及证据字符/token 统计。汇总前会再次检查 completed manifest、metrics SHA、dataset/generator/seed/experiment binding 和期望行数，不以文件存在本身视为成功。

## 9. 本地测试结果

在 Windows 本地仓库、项目 `.venv` 中执行：

```text
针对性集合：92 passed in 4.91s
全仓 pytest：475 passed, 2 subtests passed in 25.13s
Python compileall：PASS
bash -n（Git Bash）：PASS
git diff --check：PASS
```

针对性集合为：

```text
tests/test_infogain_final_protocol.py
tests/test_atomic_generator_binding.py
tests/test_fever_baselines.py
tests/test_fm2_support.py
tests/test_label_logits_and_manifest.py
```

## 10. 仍需服务器实机验证的风险

- 本地没有 `/root/experiments/...` 数据与 checkpoint，因此没有声称本地或本次会话已通过服务器八组 source preflight。部署后必须执行第 5 节命令。
- 新 preflight 会额外验证 training binding 中记录的 retrieval source SHA 与当前候选池实际 SHA 完全一致；这比仅检查路径更严格。如果服务器历史 training manifest 与冻结候选池不一致，会按设计拒绝启动，而不是绕过。
- 本地不能验证服务器文件权限、磁盘空间、tmux 版本、CUDA/驱动和四个 generator 模型目录；CPU/GPU 各自的前台 gate 会在创建 tmux 前暴露对应问题。
- `launcher.console.log` 捕获 worker 从进程启动开始的全部 stdout/stderr；`launcher.log` 是 pipeline 聚合日志。排障时两者都应保留。
- 本次未 commit、未 push，且未启动正式 CPU/GPU 实验；需要确认后再执行本地 GitHub push 和服务器 `git pull --ff-only`。

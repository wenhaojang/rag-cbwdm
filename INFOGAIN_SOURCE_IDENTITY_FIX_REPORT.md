# InfoGain source identity 最小修复报告

日期：2026-10-11
修复基线：`7b554055019f062378b1ad88f2346bf54a807df5`

## 故障原因

`source_preflight_group()` 错把训练阶段的 retrieval source SHA 与本次 selection 的 validation retrieval SHA 当成同一身份：

```text
training_manifest.artifact_binding.retrieval_source_sha256
    == validation retrieval pool SHA
```

该前提不成立。以 FEVER 为例，checkpoint 的训练来源是 `train_core.jsonl`（`45530c239eb429dd6f956da9d54d5b777ed6afd6f1bd128474c4fbd66335fbea`），而最终 selection 输入是 `validation.jsonl`（`a71fffda269f9112c6c243cee3384df072455a604eb801c838b69e32993786eb`）。二者属于同一 dataset/retrieval protocol 的不同阶段输入，SHA 应分别验证，而不应相等。

## 修改

- 删除 `source_preflight_group()` 中训练 retrieval SHA 必须等于 selection retrieval SHA 的跨阶段断言。
- 训练侧继续完整调用 `validate_formal_training_binding()`，验证 checkpoint、training manifest、训练来源绑定、dataset、conditioning generator、seed 13 和 retrieval protocol。
- selection 输入侧继续独立调用 `validate_retrieval_source()`，验证 FEVER validation BM25 pool 或 FM2 official closed-page pool、manifest、split/role、行数和实际 SHA。
- source-preflight 报告现在分别记录：
  - `training_retrieval_source_sha256`
  - `selection_retrieval_sha256`
  - 原 `retrieval_sha256` 保留为 selection 输入 SHA 的兼容字段。
- 核对 `12c_select_infogain_reranker.py`：正式 selection contract 已正确将训练来源身份保存在 `contract.artifact_binding.retrieval_source_sha256`，将当前 validation 输入保存在 `contract.inputs.retrieval.sha256`，无需修改。
- 核对 `audit_group()`、`preflight_group()` 和 evaluation binding：训练 binding 由 `validate_selection_provenance()` 验证，当前 selection 输入路径/SHA 由 `audit_group()` 独立验证，GPU evaluation 通过 selection manifest SHA/fingerprint 绑定已审计 selection；没有其他跨阶段 SHA 相等断言，无需降低或删除校验。

没有修改 YAML、Top-4 → threshold=0.2 → min_docs=2 协议、checkpoint、retrieval 数据/manifest、tmux launcher 或正式实验目录。

## 回归覆盖

- FEVER 和 FM2 均验证：训练 SHA 与 selection SHA 不同时，只要两条 provenance 链各自有效，source-preflight 通过。
- FEVER 和 FM2 均验证：训练 provenance 校验失败或 selection 输入 SHA 校验失败，source-preflight 拒绝执行。
- 验证 selection manifest 同时、独立保存训练来源 SHA 与 validation 输入 SHA。
- 验证 training manifest 中的 retrieval source SHA 被篡改时，`validate_formal_training_binding()` 拒绝。
- 原 InfoGain 协议、artifact binding、FEVER、FM2 相关测试继续通过。

测试结果：

```text
专项测试：100 passed in 5.39s
全仓 pytest：483 passed, 2 subtests passed in 24.76s
Python compileall：PASS
bash -n：PASS
git diff --check：PASS
```

## 服务器重新执行 source-preflight

部署后先只运行无模型加载的八组 source-preflight：

```bash
cd /root/rag-cbwdm
git rev-parse HEAD

RUN_ROOT=/root/experiments/rag_cbwdm/formal_v2_runs/infogain_top4_filter02_final_20261011
CHECK_DIR="$RUN_ROOT/attempts/manual_source_identity_fix_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$CHECK_DIR"

/root/miniconda3/envs/rag-cbwdm-mistral/bin/python \
  scripts/infogain_final_protocol.py \
  --plan configs/formal/infogain_top4_filter02_final.development.server.yaml \
  --stage source-preflight \
  --audit-output "$CHECK_DIR/source_preflight.json" \
  2>&1 | tee "$CHECK_DIR/source_preflight.log"

rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$CHECK_DIR/source_preflight.exitcode"
test "$rc" -eq 0
```

验收条件：退出码为 0，`source_preflight.json` 顶层为 `status=PASS` 且包含八组；每组分别给出 training 与 selection retrieval SHA。该命令不加载 reranker 模型、不生成 selection，也不启动 CPU/GPU 正式实验。

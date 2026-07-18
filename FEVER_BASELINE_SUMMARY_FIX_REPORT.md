# FEVER baseline summary 真实服务器错误修复报告

日期：2026-07-18

证据边界：当前 Codex 工作区是 Windows 本地代码副本，没有挂载服务器的
`/root` 和 `/tmp`；因此未直接改动或重新读取服务器 artifact。缺字段文件的
定位依据是用户提供的真实 traceback/run 信息、当前发现逻辑、旧 runner 输出
路径以及 Git 中 evaluator/method/schema 的版本历史交叉核对。服务器命令会在
部署后对原目录做最终只读发现和 summary 重跑。

## 结论

已修复 `scripts/13_summarize_fever_baselines.py` 对真实 run 的错误 artifact
发现和缺字段崩溃问题。修复不会修改或重算任何实验 artifact，也不需要重新运行
generator、reranker 或 selector。

目标 run：

```text
/root/experiments/rag_cbwdm/fever2_full_corpus_e2e100_v02_seed13
```

正式输出固定为：

```text
artifacts/baselines/summary/baseline_summary.json
artifacts/baselines/summary/baseline_summary.csv
artifacts/baselines/summary/baseline_summary.md
```

## 根因和缺字段文件

触发 `KeyError: 'macro_f1'` 的文件是：

```text
/root/experiments/rag_cbwdm/fever2_full_corpus_e2e100_v02_seed13/artifacts/fever2_dev_no_evidence_metrics.json
```

它是 baseline suite 之前生成的旧 no-evidence 评估 artifact。该旧 schema 有
`accuracy`、`per_class` 等字段，但没有 `macro_f1`。它的 `method` 一直是
`no_evidence`，因此旧版 summarizer 把它与正式 baseline 文件
`artifacts/eval/no_evidence_metrics.json` 合并到同一个方法组。

同目录旧主评估文件 `artifacts/fever2_dev_metrics.json` 也属于不含
`macro_f1` 的旧 schema，但其当时的 method 是
`cbwdm_cross_encoder_selector`，不在六个 canonical method 中，因此它不是
这次 `KeyError` 的直接触发文件。两个旧 root metrics 都不应进入正式 baseline
summary。

旧实现使用了两个宽泛的文件发现规则：

```python
run_dir.glob("artifacts/eval/*_metrics.json")
run_dir.glob("artifacts/*_metrics.json")
```

第二个规则错误纳入了上述旧主评估文件。随后下面的强制索引没有 schema
兼容或缺失值处理，直接崩溃：

```python
macro_f1s = [float(item["macro_f1"]) for item in items]
```

Git 历史也与该 schema 差异一致：旧 evaluator metrics schema 先于 FEVER
baseline suite 存在，`macro_f1` 是 baseline suite 接入时才加入
`src/metrics.py` 的。

## `discover_metrics()` 审计

结论：旧实现确实错误纳入了无关文件。

- 已确认直接触发错误的误纳入文件是
  `artifacts/fever2_dev_no_evidence_metrics.json`：它是旧版 root
  no-evidence metrics，不是本轮 manifest 指定的正式 baseline artifact。
- `artifacts/fever2_dev_metrics.json` 也被宽泛 glob 发现，但因为 method
  非 canonical，没有进入六方法聚合。
- 当前两层 glob 不会递归进入
  `artifacts/baselines/infogain_reranker/`，所以真实故障并不是 training
  metrics 或深层 resource metrics 直接触发。
- 但是旧逻辑完全不检查 manifest 的 `stage`、`status` 和 `method`，所以只要
  training/resource 文件出现在 glob 范围内，或通过 `--metrics` 显式传入，
  仍可能被错误纳入。
- 旧逻辑还会用文件名 stem 猜方法；现在已删除这种方法判定方式。

新发现流程只扫描：

```text
<run>/artifacts/eval/*.manifest.json
```

并且只接受同时满足以下条件的 artifact：

```text
stage == evaluation
status == completed
method 属于六个 canonical baseline
manifest 指向的 metrics 文件存在
```

runner 会进一步显式传入六个 `METHOD=evaluation-manifest-path`，因此正式运行
不依赖任何 `*metrics*.json` glob。对现有旧 manifest，如果还没有
`metrics_path`，使用同名 sidecar 路径兼容；新生成的 evaluation manifest
会显式记录 `metrics_path` 和 `predictions_path`。

六个 canonical method 的唯一映射为：

```text
no_evidence
naive_topm
bge
infogain_fever
rag_cbwdm
cbwdm_oracle
```

## Schema 兼容策略

每个正式 evaluation metrics 在聚合前先归一化。

- canonical 顶层字段优先，例如 `macro_f1`。
- 兼容旧字段名 `f1_macro`。
- 兼容 `metrics`、`classification_metrics`、`evaluation`、`results`、
  `aggregate`、`summary` 等嵌套容器中的 canonical/旧字段。
- 对其他嵌套位置做唯一值查找；若找到多个不同值，标记 ambiguous，不猜值。
- 数值不存在、类型错误、非 finite 或歧义时输出 `null`，并在
  `missing_fields` 和 `reason` 中记录原因，绝不填 `0`。
- 任一 metrics 缺字段不会再触发 `KeyError`。
- `avg_num_docs=0` 按合法数值保留，因此 `no_evidence` 正确输出 `0.0`。
- `cbwdm_oracle` 的 metadata 由 canonical method contract 强制设为：

```text
deployable=false
diagnostic_only=true
```

即使 metrics payload 中的值错误，也不会把 oracle 纳入 deployable best。

fairness audit 在写任何正式 summary 文件之前验证。总体状态或六个方法中的
任一个不是 `comparable` 时，程序拒绝生成正式 summary。

## 修改文件

- `scripts/13_summarize_fever_baselines.py`
  - 改为 manifest 驱动的 canonical artifact 发现。
  - 增加显式 canonical method contract。
  - 增加旧字段和嵌套 schema 归一化。
  - 缺字段输出 `null`、`missing_fields` 和 `reason`。
  - 强制固定 run 内 summary 输出目录。
  - fairness 非 comparable 时在写文件前失败。
  - 三种 summary 输出均使用固定文件名。
- `scripts/07_eval_rag_classification.py`
  - 新 evaluation manifest 显式记录 metrics/predictions 路径。
- `scripts/run_fever_cbwdm.py`
  - summary stage 显式传入六个 evaluation manifest。
  - validator 从真实 `artifacts/baselines/summary/` 验证三个输出。
  - validator 检查 completed/comparable、canonical method 顺序和 oracle
    metadata。
- `tests/test_fever_baselines.py`
  - 新增真实故障形状和全部要求场景的回归测试。
- `FEVER_BASELINE_SUMMARY_FIX_REPORT.md`
  - 本报告。

未修改 `src/metrics.py` 的评价数学逻辑，也未修改任何真实实验 artifact。

## 回归测试

新增测试覆盖：

1. 六种 canonical method；
2. 正式 metrics 缺少 `macro_f1` 时输出 `null` 和 reason；
3. 旧 root evaluation、training metrics、resource metrics 被排除；
4. 顶层 `f1_macro` 和嵌套 `metrics.f1_macro` 映射；
5. `no_evidence` 的 `avg_num_docs=0`；
6. oracle 强制 diagnostic/non-deployable；
7. summary 固定输出目录及三个文件；
8. fairness 非 comparable 时不写正式输出；
9. runner validator 使用真实 summary 目录。

本地验证结果：

```text
python -m compileall -q src scripts tests
PASS

python -m pytest -q
34 passed

python -m unittest discover
Ran 19 tests
OK

bash -n scripts/run_fever_cbwdm.sh
PASS

git diff --check
PASS
```

本地没有重新运行任何模型。为了执行测试，仅在工作区已有、未跟踪的 `.venv`
中安装了 `requirements.txt` 的测试依赖；没有修改依赖清单。

## 服务器重跑命令

部署本次代码修改后，在原有 Conda 环境中只重跑 summary：

```bash
cd /root/rag-cbwdm
conda activate rag-cbwdm

RUN=/root/experiments/rag_cbwdm/fever2_full_corpus_e2e100_v02_seed13

python scripts/13_summarize_fever_baselines.py \
  --run-dir "$RUN" \
  --output-dir "$RUN/artifacts/baselines/summary" \
  --fairness-audit "$RUN/artifacts/baselines/baseline_fairness_audit.json"
```

该命令只读取已存在的 evaluation metrics/manifest 和 fairness audit，不会加载
或运行模型。

输出验证：

```bash
python - <<'PY'
import json
from pathlib import Path

run = Path("/root/experiments/rag_cbwdm/fever2_full_corpus_e2e100_v02_seed13")
out = run / "artifacts" / "baselines" / "summary"
expected = {
    "baseline_summary.json",
    "baseline_summary.csv",
    "baseline_summary.md",
}
assert expected <= {path.name for path in out.iterdir()}

payload = json.loads((out / "baseline_summary.json").read_text(encoding="utf-8"))
assert payload["status"] == "completed"
assert payload["comparable"] is True
assert [row["method"] for row in payload["methods"]] == [
    "no_evidence",
    "naive_topm",
    "bge",
    "infogain_fever",
    "rag_cbwdm",
    "cbwdm_oracle",
]
oracle = payload["methods"][-1]
assert oracle["deployable"] is False
assert oracle["diagnostic_only"] is True
print(out)
PY
```

若某个正式 evaluation metrics 自身确实缺字段，命令仍会成功生成三份 summary；
对应字段为 `null`，方法状态为 `missing_metrics`，原因记录在 `reason` 中。

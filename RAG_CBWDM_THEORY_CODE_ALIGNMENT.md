# RAG-CBWDM 理论—代码对齐报告

## 依据与结论

仓库中只有 `RAG-CBWDM.tex`，不存在 `RAG_CBWDM.tex`。本次依据文件的
SHA-256 为 `E5E90BB0EC5E10DB1233EE3B19C3961DF11301D5AD28BDD1D7B8800EDA98C38E`；
未修改论文文件。

当前可验证实现应准确称为 **RAG-CBWDM Euclidean posterior-shift variant**。
它完整实现了原稿允许的 `L_i=I` 分支、闭式集合目标、gold-label greedy teacher、
状态依赖 selector 训练和测试时逐步重评分，但没有实现 categorical BW local metric。
原稿只给出 `H_q = 1/2 Hessian F_q(eta_0)`（限制在 simplex tangent space）以及
`L_q^T L_q=H_q` 的抽象定义，没有给出足以唯一编码的 categorical closed form。
代码因此拒绝 `bw_local_hessian`，不做静默回退。

## 公式—实现映射

| 论文对象 | 原稿公式/位置 | Python 文件 | 函数/变量 | shape | 一致性 | 本轮结果 |
|---|---|---|---|---|---|---|
| `eta_i0` | query-only posterior；约 638–644 行 | `scripts/03_compute_label_posteriors.py` | `score_retrieval_row`, `eta0` | `[K]` | 是 | 每个 query 只构造/计算一次 |
| `eta_ij` | query + candidate posterior | 同上 | candidate `eta` | `[K]` | 是 | 同一 query 内批处理并保持候选顺序 |
| smoothing | `eta0^eps=(1-eps)eta0+eps 1/K`；约 743–758 行 | `src/cbwdm_score.py` | `smooth_probability_vector` | `[K]` | 是 | 改为 mixture smoothing |
| `e_yi` | gold one-hot，同段给出 `e_y^eps` | 同上 | `one_hot`, `target` | `[K]` | 是 | `paper_mixture` 同时平滑 target |
| `L_i` | `L_i^T L_i=H_i`；identity 特例约 988–1009 行 | 同上 | `canonical_l_type` | `[K,K]` | identity 是 | 正式命名 `euclidean_posterior_shift`；保留 `identity` 别名 |
| `H_i` | BW divergence 在 `eta_i0` 的局部 Hessian | — | — | tangent-space operator | 未实现 | 原稿 closed form 不充分，明确抛错 |
| `x_ij` | `L_i(eta_ij-eta_i0)`；约 763–780 行 | `src/cbwdm_score.py` | `X_all[j,:]` | 代码 `[J,K]` | 是（转置存储） | 验证有限性、归一化和维度 |
| `d_i` | `L_i(e_yi-eta_i0)`；约 638–652 行 | 同上 | `d` | `[K]` | 是 | 使用同一 smoothed baseline |
| `X_i,S` | 列拼接 `[x_ij:j in S]` | 同上 | `X_all[indices,:].T`（概念上） | 论文 `[K,|S|]`；代码工作矩阵 `[|S|,K]` | 是 | 报告转置关系，避免符号同名误判 |
| `c_i,S` | `X_i,S^T d_i` | 同上 | `c = X @ d` | `[|S|]` | 是 | 代码 `X` 是论文矩阵的转置 |
| `G_i,S` | `X_i,S^T X_i,S+lambda I` | 同上 | `G = X @ X.T + ridge I` | `[|S|,|S|]` | 是 | 不显式求逆 |
| `Theta_i(S)` | `c^T G^{-1}c` | 同上 | `theta_for_indices` | scalar | 是 | `np.linalg.solve`，失败才告警并 pinv |
| `Delta_i(j|S)` | `Theta(S∪{j})-Theta(S)`；约 1033–1045 行 | 同上 | `marginal_gain` | scalar | 是 | 保存 raw gain；仅 tolerance 内归零 |
| greedy stop | 最大 gain 低于 `kappa_stop` 或达到 `T`；约 1046–1062 行 | 同上 | `greedy_teacher` | trajectory | 是 | 输出 stop reason/decision |
| `P_it` | `{j: Delta>b_plus}`；约 1064–1115 行 | `src/selector_cross_encoder.py` | `positive` | `[J] bool` | 是 | 严格大于 |
| `N_it` | `{j: Delta<b_minus}` | 同上 | `negative` | `[J] bool` | 是 | 验证 `b_plus>b_minus>0` |
| CE | gain 阈值生成二元标签的 BCE | 同上 | `ce_loss` | scalar | 是 | server 默认 `negative`；`ignore` 为显式消融 |
| ranking | `log(1+sum_p sum_n exp(gamma(r_n-r_p)))` | 同上 | `rank_loss` | scalar | 是 | 稳定 `logaddexp/logsumexp` |
| combined | `beta L_CE+(1-beta)L_rank` | 同上 | `cbwdm_multitask_loss` | scalar | 是 | 空正/负集合安全返回 0 分量 |
| test-time selector | `r_psi(q,S,z)`，选后更新 `S` 并重评分；约 1159–1178、1279–1287 行 | `scripts/11_select_with_cross_encoder.py` | `select_row` | trajectory | 是 | 不读取 gold/teacher，支持 top-M/score stop |

## Matrix orientation

论文采用列向量拼接：

```text
X_paper = [x_j]              # K × |S|
c = X_paper^T d              # |S|
G = X_paper^T X_paper + λI   # |S| × |S|
```

代码为适配 NumPy 按候选索引，保存 `X_rows = X_paper^T`：

```text
X = X_rows[indices, :]        # |S| × K
c = X @ d
G = X @ X.T + λI
```

这是严格的转置等价，不是公式方向错误。随机小矩阵测试进一步比较了闭式值与
`max_a 2a^Tc-a^TGa` 的直接最优值。

## Smoothing 语义

`paper_mixture` 是 v2 默认：

1. 验证所有 posterior 有限、非负且和为 1。
2. `eta0_used=(1-eps)eta0+eps/K`。
3. `target_used=(1-eps)e_y+eps/K`。
4. 原稿没有要求 candidate posterior smoothing，因此 `eta_ij` 保持 generator 原值。
5. `x_ij=eta_ij-eta0_used`，`d_i=target_used-eta0_used`。
6. `eps=0` 返回数值完全相同的副本。

`legacy_clip_all` 仅用于复现旧实验：clip/renormalize `eta0` 和所有 candidates，
gold target 不平滑。teacher schema 会记录模式，禁止混淆两类 gain。

## Teacher、selector 与 inference

teacher schema 为 `rag_cbwdm_teacher.v2`，逐行记录算法类型、真实 `l_type`、
`lambda/ridge_lambda`、平滑模式、阈值、tolerance、每步完整 candidate gains、
raw gain、选择前状态、选择结果和停止原因。

旧 feature MLP 的 best-candidate BCE 和旧 cross-encoder `listwise_distill` 保持可用；
它们不是论文的联合目标。新服务器配置显式选择 `loss_type: cbwdm_multitask`。
其中 `neutral_sample_policy: negative` 按原稿二元标签把非 positive 候选纳入 CE negative，
而 ranking negative 仍严格使用 `gain < b_minus`；`ignore` 只作为显式可选消融。
阈值 `b_plus/b_minus`、`gamma`、`beta` 在原稿中没有最终数值，配置值均标记为
smoke/pilot 初值，必须仅用 train/dev 调参后冻结。

测试时输入只含 claim、already-selected evidence 与 candidate evidence。claim/candidate
位于输入前部并各有字符预算，selected state 保持真实顺序且有独立预算；tokenizer
再按 `max_length` 截断。每选一篇后会对所有剩余候选重新评分。

## 已修复差异

- clip smoothing 改为论文 mixture smoothing，并同时平滑 gold direction。
- `identity` 的真实含义在 config、teacher 和报告中统一为 Euclidean posterior shift。
- 未实现 BW 模式从潜在模糊含义改为明确错误。
- gain 增加 raw 值、数值 tolerance 和理论单调性失败诊断。
- 新增论文联合 selector loss，旧 listwise 保持兼容。
- posterior 改为 batch、float32 检查、断点续跑和 cache fingerprint。
- teacher/selection/metrics schema 版本化并统一。

## 仍未解决

1. 缺少可唯一实现的 categorical BW local Hessian 理论公式，因此不能宣称完整 BW metric。
2. `b_plus/b_minus/gamma/beta` 和 inference score threshold 尚未经过 train/dev pilot 校准。
3. 尚未在目标 Linux CUDA GPU 上验证 Qwen 的显存、吞吐、模型 revision 与 batch size。
4. 当前 state truncation 是确定性的字符预算加 tokenizer 截断；formal 前应检查真实 token
   长度分布并决定是否采用更精细的 per-field token budgeting。

因此，当前实现可以称为“RAG-CBWDM 主流程的 Euclidean posterior-shift 实现”，
不能称为“完整 categorical BW local-metric 实现”。

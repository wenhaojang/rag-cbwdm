# InfoGain-RAG 与 RAG-CBWDM 集成审计

审计日期：2026-07-17（Asia/Shanghai）

审计对象：

- InfoGain-RAG 官方仓库：`MaYufei-NPU/InfoGain-RAG`
- 当前工程：`rag_cbwdm`
- 论文：*InfoGain-RAG: Boosting Retrieval-Augmented Generation via Document Information Gain-based Reranking and Filtering*，arXiv:2509.12765v1

## 1. Executive Decision

```text
InfoGain-RAG 作为 baseline：
GO WITH ADAPTER

RAG-CBWDM 插入官方代码：
GO WITH NONTRIVIAL REFACTOR

推荐主集成架构：
采用方案 D 的“共享 retrieval / generator scoring / evaluator + 可插拔 teacher 和 selector”，
同时把官方仓库固定为只读参考；短期先实现方案 B 作为公平 baseline，不以官方脚本直接运行作为正式结果。
```

关键依据：

1. 两者都利用固定 generator 在 query-only 与 query-document 条件下的输出变化监督文档选择，数学动机直接相关。
2. InfoGain 的监督对象是固定 ground-truth answer sequence 的**标量置信度差**；当前 FEVER 管线保存的是离散标签 `Y` 的**完整 next-token posterior 向量**。在 FEVER 上可以从向量取 gold-label 分量构造分类版 DIG，但不是同一个现成字段。
3. 官方公开代码没有实现论文所述的 sliding-window confidence 聚合、首 token 加权、DIG 标量、`b1/b2` 标注、论文式 LogSumExp margin loss、阈值过滤和至少保留 2 篇文档的完整闭环。
4. 官方 `vllm_logits.py` 能提取 ground-truth answer token 的 query-only / query-document log-probability序列，但只保存 `rag_logprob` 和 `non_rag_logprob`，没有从其计算最终 confidence 或 DIG。
5. 官方 multi-loss 脚本实现的是“第一篇正样本 vs 其余负样本”的 RankNet softplus，以及仅前两篇文档的二分类 CE；这与论文的正负集合极值、LogSumExp margin 目标不等价。
6. 当前 CBWDM teacher 是集合状态相关的：每一步重算剩余文档的 marginal gain；InfoGain reranker 是独立 query-document pointwise scorer，排序目标仍只约束单文档分数。
7. 当前 cross-encoder selector 已把 `already selected evidence` 显式编码进输入，并以每一步 teacher gains 做 listwise distillation；这是把官方 reranker改成 state-aware selector 的自然落点。
8. retrieval 行和官方 `question/top_passages` 行可无损映射主要文本字段；posterior 到官方 sequence confidence 不可无损映射，必须重算或定义 FEVER 专用 DIG。
9. 当前 evaluator 能统一消费 `selected_docs` 并使用固定 FEVER generator 评估，因此适合做公平比较的统一终点。
10. 官方仓库缺少数据、checkpoint、requirements/environment、完整 CLI 和正式复现实验说明，不能把“脚本能语法编译”等同于“官方 baseline 可复现”。

## 2. Repository Snapshots

### 2.1 当前 RAG-CBWDM

| 项目 | 快照 |
|---|---|
| 根目录 | `<WORKSPACE>/rag_cbwdm` |
| branch | `main` |
| commit | `4848ba8fa6a08922b563c5939b39b69a0246d724` |
| commit 日期 | `2026-07-05T16:50:05+08:00` |
| commit 标题 | `Initialize FEVER preprocessing and BM25 retrieval` |
| dirty | 是；审计开始前已有 3 个 tracked 修改和多项 untracked pipeline 文件 |
| Python（项目 Conda） | 3.10.20 |
| 关键版本 | NumPy 2.2.6；PyTorch 2.12.1+cpu；Transformers 5.13.0；Accelerate 1.14.0；PyYAML 6.0.3 |
| 数据规模 | `data/` 6 文件，21,900,179 bytes |
| 输出规模 | `outputs/` 70 文件，53,656,871 bytes |

审计前 `git status --short` 的核心事实：

- 用户已修改：`configs/fever2_minimal.yaml`、`configs/fever3_minimal.yaml`、`requirements.txt`。
- Stage 3–7/selector/eval 等多数新增源码尚未提交。
- 本审计没有 reset、clean、切换分支、覆盖或格式化这些文件。

本地隔离规则写入 `.git/info/exclude`：

```gitignore
_external/
scratch/
```

### 2.2 InfoGain-RAG 官方仓库

| 项目 | 快照 |
|---|---|
| remote | `https://github.com/MaYufei-NPU/InfoGain-RAG.git` |
| 默认 branch | `main` |
| 本地审计 commit | `8151a4d81971eee219b497089aa350df0c0e8301` |
| clone 时 `origin/main` | `8151a4d81971eee219b497089aa350df0c0e8301` |
| commit 日期 | `2025-09-17T15:42:25+08:00` |
| commit 标题 | `Update README.md` |
| dirty | 否 |
| license | Apache-2.0 |
| 本地目录 | `_external/InfoGain-RAG/`（被主仓库 local exclude） |

仓库只有 16 个 tracked 文件、约 90 KB；没有：

- `requirements.txt`、`environment.yml`、`setup.py/pyproject.toml`、Dockerfile；
- 数据文件、checkpoint、下载脚本或校验和；
- 单一可复现入口、配置文件、正式训练参数文件；
- README 中的逐步运行说明；
- FM2 数据处理或分类版 DIG 实现；
- 论文实验使用的 Contriever/BM25/DPR 检索实现和多检索器合并入口。

README 仅列出 Python/PyTorch/Transformers/CUDA 的最低版本；实际代码额外依赖 `vllm`、`pandas`、`pyarrow`、`jsonlines`、`openai`、`requests`、`scikit-learn`、`tqdm`。

## 3. InfoGain-RAG Architecture

### 3.1 文件清单与职责

| 文件 | 真实职责 | 是否为可独立 CLI |
|---|---|---|
| `README.md` | 方法摘要、最低依赖、论文链接 | 否 |
| `calculate_dig/utils.py` | JSONL/gzip I/O 与随机分层采样辅助 | 否 |
| `calculate_dig/data/trivia_qa.py` | TriviaQA query/answer 预处理、与 `top_passages` 合并、按 answer 展开到 Parquet | 否 |
| `calculate_dig/data/natural_qa.py` | Natural Questions short answer 提取、候选合并 | 否；多进程入口有参数错误 |
| `calculate_dig/data/popqa.py` | 从 `popQA.tsv` 生成 query dataset | 否 |
| `calculate_dig/data/query_doc_pair.py` | `question/answers/top_passages` JSON 转 query-document Parquet | 否 |
| `calculate_dig/vllm_logits.py` | Qwen2.5-7B vLLM 初始化、teacher-forced answer token logprob 提取、分块写 Parquet | 是，但参数/路径均硬编码 |
| `train/roberta_train_ce_loss.py` | RoBERTa-large pointwise二分类 CE | 否；import 即加载 tokenizer、数据并训练 |
| `train/roberta_train_ranknet_loss.py` | 第一篇正样本对其余文档的 RankNet loss | 是，但数据/模型/输出硬编码 |
| `train/roberta_train_multi_loss_v2.py` | 两个 head 的 RankNet + CE 加权训练 | 是，但数据/模型/输出硬编码 |
| `generate_and_judfe/rerank_passage_bert_multi.py` | 使用 multi-loss checkpoint 的 rank head 给 passage 打分并降序排序 | 是 |
| `generate_and_judfe/gen_res.py` | 多种远端 API 的 RAG/no-RAG 答案生成 | 是 |
| `generate_and_judfe/judge_res.py` | 500 条固定长度的宽松字符串包含评估 | 是 |
| `generate_and_judfe/tools.py` | 多家 API client，key/base URL 占位且硬编码 | 否 |
| `generate_and_judfe/run_single_model.sh` | 针对 TriviaQA/NQ/PopQA 的大量固定文件名实验调用 | shell |
| `LICENSE` | Apache-2.0 | 否 |

目录名 `generate_and_judfe` 是上游原始拼写，审计未更名。

### 3.2 官方代码实际数据流

```text
TriviaQA / NQ / PopQA 原始数据
  → dataset-specific Python 函数
  → {raw_idx, query, answers}
  + 外部预先检索的 {question, answers, top_passages}
  → 每个 (query, answer, passage) 一行 Parquet
  → vllm_logits.py 把 ground-truth answer 作为 assistant 内容拼入 prompt
  → prompt_logprobs 中切出 answer token logprob
  → rag_logprob[] / non_rag_logprob[] Parquet
  → [公开仓库缺失：平滑、加权、confidence、DIG、b1/b2、抽样]
  → 外部已有 query_documents_15.jsonl 或 CE Parquet
  → RoBERTa-large CE / RankNet / multi-loss
  → rerank_passage_bert_multi.py 写 passage.score 并排序
  → [公开仓库缺失：0.2 过滤、top-4、min-2]
  → gen_res.py 固定读取前 4 篇生成答案
  → judge_res.py 宽松包含匹配
```

### 3.3 DIG scoring 的真实实现

`calculate_dig/vllm_logits.py`：

- 模型固定为 `Qwen/Qwen2.5-7B-Instruct`（line 10）。
- import 时全局调用 `init_llm()`（line 99），配置 4-way tensor parallel、fp16、2048 context（lines 68–82）。
- RAG prompt 输入 `passage_title`、`passage_text`、`query`；query-only prompt不含 passage。
- ground-truth `d["answer"]` 作为 assistant message 拼入两类 prompt（lines 106–125），因此是 teacher forcing，不是先生成再 judge 的答案。
- `parse_prob()` 读取 answer span 内每个 token 的 prompt logprob（lines 226–234）。
- 每个 query-document-answer pair 写出 `rag_logprob: list[float]` 与 `non_rag_logprob: list[float]`。
- 支持一次给 vLLM 一批 prompt，但外层按硬编码 10,000 行写 Parquet；没有 CLI、断点 manifest 或输入 hash。
- `uuid.uuid4()` 产生 `idx`，未设 seed；同一数据重跑 id 不稳定。
- query-only logprob 对每个 query-document pair 重复计算，没有按 query cache。

没有实现：

- logprob → token probability；
- sliding-window smoothing；
- 首 token importance weighting；
- token 序列到单个 confidence 的聚合；
- `DIG = confidence_rag - confidence_no_rag`；
- `b1/b2`、proficient/challenging query 划分；
- DIG 三类或二分类 label；
- 数值 log-space 聚合策略；
- classification label 的专用处理；
- 明确的多 answer 合并；代码把多个 gold answer 展开为多行。

因此，官方代码只完成了“DIG 的原始 token logprob 采集前半段”，不是完整 DIG dataset builder。

### 3.4 论文与官方 reranker 代码

论文描述：

- CE：用 `b1/b2` 分隔高正 DIG 和负 DIG 样本；
- margin：同一 query 内正/负集合，使用极值约束与 LogSumExp 近似；
- multi-task：CE 与 margin 加权；
- RoBERTa-large、top-100、Adam、论文报告学习率 `5e-6`、loss 权重 `0.75`；
- 推理 top-4，score threshold `0.2`，阈值后不足时至少保留 2 篇。

公开 multi-loss 代码：

- `QueryDocumentDataset` 只读取 `{"query": str, "documents": list[str]}`；默认第一篇是唯一正样本。
- 输入模板是 `query:{query} passage:{doc}`，固定 max length 512。
- backbone `roberta-large`，共享 1024→768 层，独立 rank scalar head 与 binary 2-class head。
- rank loss 为第一篇分数分别对后续每篇的 `softplus(-(s_pos-s_neg))` 均值。
- CE head 只取前两篇文档；固定 label `[1, 0]`。
- `LAMBDA=0.9`，总损失为 `0.9*ranknet + 0.1*CE`。
- AdamW 学习率 `1e-5`，1 epoch，batch size 1，CosineAnnealingLR。
- 输入固定绝对路径 `/etc/ssd1/.../query_documents_15.jsonl`。
- checkpoint 目录假定已经存在。
- inference 只使用 rank head；binary head 和过滤判断均未使用。

结论：公开 multi-loss 是一个“正样本第一位”的简化 RankNet+CE 实验脚本，不是论文公式的完整实现。代码和论文在 backbone、query-document编码和“两个任务”这一层一致；在 label construction、margin 数学形式、权重、学习率、推理过滤和数据规模接口上不一致。

### 3.5 官方推理与评估缺口

- `rerank_passage_bert_multi.py:97` 只处理输入第 300–499 条，却在 lines 115–117 写回全部最多 500 条；前 300 条没有 reranker score/排序。
- 该脚本只降序排序，不做 threshold、top-4 或 min-2。
- `gen_res.py` 硬截断到 500 条；RAG 模式无条件访问前 4 篇，候选少于 4 时重试后写 `"None"`。
- `judge_res.py` 固定循环 500 次。
- `has_answer_loose()` 使用双向 substring，不是论文声称的严格 Exact Match。
- `run_single_model.sh` 用文件名中的 `0.16` 暗示阈值，但仓库中没有生成这些预过滤文件的代码。
- 没有单检索器与多检索器的独立实现；只有预先准备好的不同固定文件名。

## 4. Current RAG-CBWDM Architecture

### 4.1 当前真实流程

```text
00_prepare_fever.py
  FEVER raw → {id, query, label, split, metadata}
01_prepare_fever_corpus.py
  wiki page sentence → {doc_id, title, text, metadata}
02_retrieve_bm25.py + src/retrieval_bm25.py
  → 每 query 的 candidates[{doc_id, rank, score, title, text}]
03_compute_label_posteriors.py + src/label_logits.py
  → eta0: [C], candidate.eta: [C]
04_build_cbwdm_teacher.py + src/cbwdm_score.py
  → 集合级 greedy trajectory + 每一步全部 candidate_gains
05/06
  → feature MLP 训练与状态特征贪心选择
10/11 + src/selector_cross_encoder.py
  → state-aware text cross-encoder listwise distillation 与贪心选择
08
  → naive BM25 top-M
09
  → 使用 gold label 的 CBWDM oracle diagnostic
07
  → 拼接 selected docs，固定 generator 重新计算 FEVER label posterior并评估
```

### 4.2 数据与检索

- FEVER-2：`SUPPORTS/REFUTES`，丢弃 raw `NOT ENOUGH INFO`。
- FEVER-3：映射为 `SUPPORTS/REFUTES/NOT_ENOUGH_INFO`。
- corpus：sentence 级；title 和 text 分开保存，BM25 tokenization 使用 `title + text`。
- candidate：`doc_id: str`、`rank: int`（1-based）、`score: float`、`title: str`、`text: str`。
- 配置默认 BM25 top-N=20，最终 selector top-M=4。
- 当前仓库没有 BGE baseline；只有旧报告中的建议文件名。

### 4.3 Generator posterior

`src/label_logits.py::LabelLogitScorer.score_prompt`：

- 取 causal LM 最后一位置的 next-token logits；
- 每个 label 可有多个 single-token verbalizer；
- 同一 label 的 token logits用 `logsumexp` 聚合；
- 所有 label score 经 softmax 归一化成 `[C]` 概率向量。

`scripts/03_compute_label_posteriors.py::iter_posterior_rows`：

- `eta0(q)=P(Y|q)`；
- `eta(q,d)=P(Y|q,d)`；
- query-only 与每个单文档逐条调用，没有 batching；
- 输出保存 `labels`、`eta0` 和每个 candidate 的 `eta`；
- 没有保存完整词表 logits、ground-truth sequence token logprob 或 prompt hash。

关键限定：

- 这是 label-space posterior，不是任意 answer sequence 的生成概率。
- `eta0` 与 candidate `eta` 都是归一化的完整标签分布。
- teacher 阶段没有计算真实 `P(Y|q,Z_S)`；集合效应由单文档 posterior shift 的代数目标近似。
- final evaluator 才把多篇已选文档按 selection order 拼接并重新调用 generator。

### 4.4 CBWDM teacher

`src/cbwdm_score.py`：

```text
X_j = eta(q,d_j) - eta0(q)
d   = one_hot(y_gold) - eta0(q)
c   = X_S d
G   = X_S X_S^T + lambda I
Theta(S) = c^T G^-1 c
gain(j | S) = Theta(S ∪ {j}) - Theta(S)
```

- 当前只实现 `L_type="identity"`；注释中的 BW Hessian/local metric 尚未实现。
- `eps_smooth` 先裁剪并归一化 posterior。
- ridge 默认 `0.01`。
- 每一步保存 `current_indices/current_doc_ids`、`theta_before`、`best_gain`、`theta_after` 和所有 `candidate_gains`。
- 默认 top-M=4，`best_gain < stop_threshold` 才停止；默认 threshold=0。
- teacher 使用 gold label，不能作为可部署测试时方法。

### 4.5 Selectors

Feature MLP：

- 输入含 `eta0`、candidate `eta`、shift、绝对 shift、熵、retrieval rank/score、step、已选数量、selected shifts 均值与相似度。
- 训练 target 是每个 teacher state 下的 `is_best∈{0,1}`，使用 `BCEWithLogitsLoss`；连续 `gain` 只保存在 metadata 中，不直接回归。
- 推理每一步重新构造状态特征，再选最高分。
- 它不是纯 pointwise reranker；把 state features 清零/固定 step=0 后才可退化。

State-aware cross-encoder：

- 输入明确包含 `Claim`、`Already selected evidence`、`Candidate evidence`。
- 每个 teacher step 构成一个 group，target 为该状态下所有 candidate gains。
- loss 是 gains 经 temperature softmax 后与 model score softmax 的 listwise cross entropy。
- 推理选一篇后把其全文加入 selected block，再给剩余文档重新打分。
- checkpoint 使用 Hugging Face sequence-classification model 的单 scalar regression head。

### 4.6 评估

已实现：

- no-evidence；
- naive BM25 top-M；
- CBWDM gold-label oracle diagnostic；
- feature MLP selector；
- state-aware cross-encoder selector；
- 统一 FEVER generator classification evaluator。

指标：

- accuracy、num correct/num examples；
- average selected docs；
- average evidence chars；
- prediction JSONL 含 `gold/pred/correct/probs/selected_doc_ids`。

没有：

- BGE baseline；
- InfoGain pointwise baseline；
- per-class 指标和统一结果 aggregation；
- 正式大样本 selector 结果；现有输出均为 smoke/small run。

## 5. Paper-to-Code Verification

| 论文对象 | 论文定义/流程 | 官方代码 | 差异与影响 | 复现判断 |
|---|---|---|---|---|
| answer `y` | 固定 ground-truth answer | `d["answer"]` 被作为 assistant answer | 一致；不是生成后 judge | 可确认 |
| teacher forcing | 评估正确答案生成 confidence | answer token位于 prompt 内并读取 prompt logprob | 本质为 teacher forcing | 可确认 |
| token probability | normalized LLM logits | 保存 token logprob | 未 exponentiate/聚合 | 不完整 |
| sliding window | 缓解长度偏差 | 无 | 长 answer confidence 无法按论文复现 | 阻断 |
| early-token weighting | 前若干 token 权重 0.8 | 无 | 论文核心 confidence 定义缺失 | 阻断 |
| DIG | `p(y|x,d)-p(y|x)` | 只保存两组 logprob list | 没有标量 confidence/DIG | 阻断 |
| `b1/b2` | 高正/负样本边界 | 无 | CE label 数据构造缺失 | 阻断 |
| proficient/challenging query | 依 query-only表现划分 | 无 | 数据采样流程缺失 | 阻断 |
| CE | 正/负 DIG 二分类 | 独立 CE 脚本读预制 Parquet；multi-loss固定首正次负 | label 生成不在仓库内 | 部分 |
| margin | 正负集合极值 + LogSumExp margin | 首正对每个其余负的 RankNet softplus | 数学目标不同 | 不一致 |
| joint training | 单 reranker、CE+margin | 两个独立 head，共享 backbone | 是联合优化，但 inference 仅 rank head | 部分 |
| hyperparameters | LR `5e-6`、weight `0.75` | LR `1e-5`、`LAMBDA=0.9` | 与论文不一致 | 不忠实 |
| inference filter | threshold 0.2、top-4、min-2 | 只排序；生成固定前4篇 | filter代码缺失 | 阻断 |
| evaluation | Exact Match | 双向 substring、固定500条 | 不是严格 EM | 不一致 |

数值风险：

- 官方保存 logprob 是正确的稳定原始量，但论文 confidence 的后续实现缺失，无法判断作者实际是概率域、log 域还是其他加权均值。
- 若直接对长序列概率连乘，fp16/float32 都有下溢风险；公开代码尚未走到连乘阶段。
- answer span 靠 chat-template token 数和 `len(tokens)-2` 推断，跨 tokenizer/template 可能 off-by-one；代码没有 span assertion。
- `temperature=0.8/top_p=0.85` 只影响额外生成 token，理论上不应影响 prompt logprob；该 token又没有使用，属于不必要随机计算。

## 6. Interface and Schema Mapping

### 6.1 字段级映射

| 语义 | InfoGain 真实字段 | RAG-CBWDM 字段 | 类型/shape | 映射 |
|---|---|---|---|---|
| query id | `raw_idx` 或随机 `idx` | `id` | str/int | 可映射；不要使用随机 `idx` |
| query | 训练/DIG `query`；推理 `question` | `query` | str | 无损 rename |
| ground truth | `answers: list`，展开后 `answer: str` | `label: str` | open QA sequence vs class | FEVER 需专用 scoring，不是简单 rename |
| candidate id | `passage_id` / `top_passages[].id` | `candidates[].doc_id` | str/int | 无损 stringify |
| title | `passage_title` / `.title` | `.title` | str | 无损 |
| text | `passage_text` / `.text` | `.text` | str | 无损 |
| rank | `passage_rank` | `.rank` | int；官方多为0-based，本地1-based | adapter 明确 ±1 |
| retrieval score | 预处理通常无 | `.score` / `.retrieval_score` | float | 官方脚本不依赖，可保留扩展字段 |
| query-only | `non_rag_logprob: list[T]` | `eta0: list[C]` | token sequence vs label distribution | 有损/不可直接换 |
| query-doc | `rag_logprob: list[T]` | `candidate.eta: list[C]` | token sequence vs label distribution | 有损/不可直接换 |
| DIG | 公开代码无字段 | 公开本地无 DIG 标量 | float | 必须新增/重算 |
| binary label | CE Parquet `label` | 无 InfoGain label | int {0,1} | 从 FEVER gold-component DIG+阈值构造 |
| rank group | `{"query","documents":[positive,...]}` | teacher `steps[].candidate_gains` | group | 需按 step/objective adapter |
| reranker score | `top_passages[].score` | `selected_docs[].selector_score` | float | 无损 |
| selected ids | 没有独立字段；靠排序/过滤后的 top_passages | `selected_doc_ids` | list[str] | adapter |
| generated answer | output row `answer` | prediction `pred` | free text vs label | 任务不同 |
| evaluation | console count `EM` | metrics JSON accuracy | int vs structured | 需统一 evaluator |

### 6.2 Schema probe

使用 1 条本地 retrieval fixture 的结果：

```text
current retrieval → official question/top_passages: PASS（结构）
current posterior → official DIG raw fields: FAIL（预期）
缺失：answer, rag_logprob, non_rag_logprob
official-shaped reranked row → unified selection schema: PASS（结构）
```

该 PASS 仅表示 JSON 字段能转换；没有把 retrieval score 冒充正式 InfoGain score，也没有伪造 sequence probability。

### 6.3 逐阶段对照矩阵

| 阶段 | InfoGain-RAG 文件/函数 | RAG-CBWDM 文件/函数 | InfoGain 输入 | CBWDM 输入 | InfoGain 输出 | CBWDM 输出 | 统计语义一致 | 代码可直接复用 | adapter | 主要风险 |
|---|---|---|---|---|---|---|---|---|---|---|
| 数据准备 | `data/*.py` | `00_prepare_fever.py` | QA dataset-specific | FEVER JSONL | query/answers | id/query/label/split | 否 | 否 | dataset adapter | 官方函数不完整 |
| corpus | 外部 2018 Wiki | `01_prepare_fever_corpus.py` | 100-word passage（论文） | FEVER sentence | top_passages | doc JSONL | 否 | 否 | passage builder | 粒度影响公平性 |
| first retrieval | 仓库缺失 | `retrieval_bm25.py` | 外部 Contriever/BM25/DPR top100 | BM25 top20 | top_passages | candidates | 可对齐实验 | 当前可复用 | rename/rank | 候选集不同 |
| query-only scoring | `vllm_logits.py` | `LabelLogitScorer` + Stage 3 | answer sequence | label verbalizers | logprob[T] | eta0[C] | FEVER可类比 | 否 | scoring adapter/重算 | 概率对象不同 |
| query-doc scoring | 同上 | 同上 | answer+single passage | label+single evidence | logprob[T] | eta[C] | FEVER可类比 | 否 | 同上 | prompt不同 |
| teacher score | 公开实现缺失 | `cbwdm_score.py` | scalar DIG | posterior shifts/set objective | 缺失 | trajectory | 否 | 否 | 新 DIG teacher | pointwise vs set-wise |
| 正负标签 | 公开实现缺失 | 无 InfoGain labels | `b1/b2` | gains | 缺失 | continuous gains | 否 | 否 | threshold builder | 阈值未知 |
| dataset | train scripts | selector dataset/groups | first doc positive | 每状态所有候选 | tensors | point/listwise groups | 否 | tokenizer部分 | dataset adapter | 样本组织不同 |
| model | RoBERTa scripts | feature MLP / HF cross-encoder | query+doc | features或query+state+doc | rank/binary heads | scalar | pointwise时相近 | backbone思路 | 新 pointwise class | checkpoint不兼容 |
| loss | CE/RankNet/multi | best-candidate BCE（MLP）或 gain listwise distill | binary/order | 每状态 best label / CBWDM gain | loss | loss | 否 | 否 | objective module | 官方≠论文 |
| inference ranking | rerank script | Stage 6/11 | independent pairs | state-dependent candidates | sorted passages | greedy order | 否 | I/O可复用 | selection adapter | 复杂度不同 |
| filtering | 代码缺失 | threshold/min_docs | paper 0.2/min2 | configurable | 缺失 | early stop | 可配置成相似 | 否 | InfoGain policy | score calibration |
| set construction | external/prebuilt | greedy loop | top-k after point scores | sequential marginal | passages | selected docs | 否 | 否 | policy interface | 冗余信息 |
| final generation | `gen_res.py` | Stage 7 | free-answer API | FEVER label posterior | answer | pred/probs | 否 | 否 | 统一 evaluator | prompt/model差异 |
| metrics | `judge_res.py` | `metrics.py` | loose substring | exact label match | console int | JSON metrics | 否 | 否 | evaluator adapter | 官方“EM”不严格 |
| aggregation | shell fixed names | metrics per method | 手工文件 | per-run JSON | 无统一表 | metrics JSON | 可统一 | 当前 evaluator | summary script | 缺少 seeds/manifest |

## 7. Core Compatibility Findings

### 7.1 sequence confidence 与 label posterior

四个不同层次的结论：

1. **数学定义可类比：是。** 在 FEVER 中定义
   `DIG_cls(d|q,y)=eta_y(q,d)-eta0_y(q)`，
   即取 gold label 分量，可以得到 InfoGain 式标量差。
2. **数据可直接复用：否。** 当前 posterior 没有 answer token logprob；官方 raw logprob 也没有完整 label posterior。
3. **代码可直接复用：否。** 官方 prompt/template/vLLM span extraction 面向 answer sequence；本地 scorer 面向 single-token label verbalizer。
4. **实验可公平比较：可以，但必须固定 generator、prompt、候选集和最终 evaluator，并明确这是“FEVER-adapted InfoGain”，不是官方 open-QA pipeline 原样复现。**

开放式 QA 中，当前 next-token label posterior 不足以复现 InfoGain sequence confidence；必须新增完整 answer teacher-forcing scorer。

### 7.2 pointwise DIG 与 set-wise CBWDM

- InfoGain：每个 `(q,d)` 的 target 与 score 不依赖已选集合；margin loss也只是同 query 内单文档分数的相对约束。
- CBWDM：`gain(d|S)` 依赖当前 `S`，同一文档在不同 step 的 target 可变化。
- 把 CBWDM teacher 的 step 0 gain 当 InfoGain target是可行消融，但会丢失：
  - 文档冗余/互补性；
  - 已选集合改变后的 marginal utility；
  - 动态停止条件；
  - 选择顺序对后续候选的影响。
- 官方 pointwise reranker 可以改成 state-aware selector，但需要：
  - 输入新增 `selected_doc_ids/selected_docs/step`；
  - 编码从 `query+candidate` 改为 `query+selected set+candidate`；
  - 数据从每 query 单 group 改为每 `(query,teacher step)` group；
  - target 改为该 step 的 candidate marginal gains；
  - inference 改为每选一篇后重新 score remaining。
- 当前 `src/selector_cross_encoder.py` 已经完成这套结构，无需在官方仓库重做。

### 7.3 FEVER 与 open-domain QA

- FEVER：`Y` 是有限标签变量；可以把正确 label 的 posterior gain视为 DIG 的分类特例。
- Open QA：`y` 是多 token answer sequence；label-space posterior方案不能直接扩展。
- 官方论文虽报告 FM2 fact verification，但公开仓库没有 FM2 adapter或 classification confidence代码，不能据论文结果推断公开代码已支持 FEVER。

### 7.4 gold-answer dependence 与 deployability

- InfoGain DIG data construction 使用 gold answer；只用于训练监督，训练好的 pointwise reranker测试时不需要 gold。
- CBWDM teacher 使用 gold label；feature/cross-encoder selector测试时不使用 gold。
- `scripts/09_select_cbwdm_oracle_from_teacher.py` 在 dev/test直接消费 gold teacher，只能报告 diagnostic upper bound。
- 公平主表应比较 deployable InfoGain reranker vs deployable CBWDM selector；两者各自 teacher/oracle单列。

### 7.5 shared cache

可以共享：

- retrieval rows；
- query、doc text、rank/score；
- generator identity、prompt version、tokenizer version；
- FEVER `eta0/eta` 作为 CBWDM cache；
- 从 `eta_y` 派生的 FEVER-DIG scalar。

不能共享：

- 用 next-token label posterior替代开放 QA answer sequence confidence；
- 不同 prompt/model生成的 DIG；
- CBWDM set state target与InfoGain pointwise target。

建议 cache key 至少包含：

```text
dataset/split/query_id/doc_id
generator_revision/tokenizer_revision
prompt_template_hash
label_verbalizer_hash or answer text
scoring_mode (label_next_token | answer_sequence)
dtype
```

## 8. Integration Options

评分：5 最好；“失败风险”5 表示风险最高。

| 方案 | 复现忠实度 | baseline 公平性 | 代码污染（5=少） | 数据复用 | 维护成本（5=低） | 服务器迁移 | 失败风险 | 推荐 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A 官方仓库独立运行 | 2 | 2 | 5 | 1 | 2 | 1 | 5 | 低 |
| B 移植 DIG/reranker 逻辑到当前仓库 | 3 | 5 | 3 | 5 | 4 | 4 | 3 | 高（短期） |
| C 在官方仓库内插入 CBWDM | 2 | 2 | 1 | 2 | 1 | 1 | 5 | 不推荐 |
| D 统一可插拔框架 | 4 | 5 | 4 | 5 | 5 | 5 | 3 | 最高（主架构） |

说明：

- A 不能获得高复现忠实度，因为公开仓库本身缺失论文关键步骤；“不改上游”并不能补足缺失实现。
- B 最适合先做公平 baseline：固定当前 FEVER retrieval/generator/evaluator，只替换 teacher/selector/policy。
- C 同时继承官方硬编码、缺失模块和任务 schema，且要重写 CBWDM，收益最低。
- D 是最终主方案；实现成本高于 B，建议从 B 的接口自然演进，不一次性重构所有现有脚本。

## 9. Recommended Architecture

建议保持现有脚本可运行，新增薄层而不是移动现有模块：

```text
src/
  scoring/
    cache_schema.py
    fever_dig.py
    answer_sequence_confidence.py       # 后续 open-QA 才需要
  selectors/
    infogain_pointwise.py
    selection_policy.py
  adapters/
    infogain_official_io.py

scripts/
  12_build_fever_infogain_teacher.py
  13_train_infogain_reranker.py
  14_select_with_infogain.py
  15_export_infogain_official.py        # 仅复现对照
  16_import_infogain_selection.py       # 仅复现对照
  17_summarize_results.py
```

推荐接口：

```python
TeacherRow:
  id, query, labels, gold_label
  candidates[
    doc_id, eta, utility, binary_label
  ]
  teacher_type: "fever_dig" | "cbwdm"

Selector.score(query, candidate, state=None) -> float

SelectionPolicy.select(candidates, scorer, top_m, threshold, min_docs)
  -> selected_doc_ids, selected_docs
```

`fever_dig.py` 的最小定义：

```text
dig_j = candidate.eta[gold_index] - eta0[gold_index]
```

必须在实验命名中写 `fever_label_dig`，避免声称它复现了论文的 answer-sequence smoothing confidence。

## 10. Exact File-by-File Change Plan

| 文件 | 动作 | 目的 | 依赖 | 风险 | 影响现有实验 |
|---|---|---|---|---|---|
| `src/scoring/cache_schema.py` | 新增 | 版本化 shared scoring cache | io_utils | 低 | 否 |
| `src/scoring/fever_dig.py` | 新增 | 从 eta gold 分量计算分类 DIG/label | NumPy | 中 | 否 |
| `src/scoring/answer_sequence_confidence.py` | 后续新增 | 正式复现滑窗与首 token加权 | torch/transformers | 高 | 否 |
| `src/selectors/infogain_pointwise.py` | 新增 | RoBERTa/HF scalar+binary pointwise reranker | transformers | 中 | 否 |
| `src/selectors/selection_policy.py` | 新增 | top-M/threshold/min-doc统一策略 | 无 | 低 | 否 |
| `src/adapters/infogain_official_io.py` | 新增 | 官方 question/top_passages 与统一 schema互转 | io_utils | 中 | 否 |
| `scripts/12_build_fever_infogain_teacher.py` | 新增 | 生成 FEVER-DIG teacher JSONL | fever_dig | 中 | 否 |
| `scripts/13_train_infogain_reranker.py` | 新增 | pointwise CE+论文式 margin训练 | selector/dataset | 高 | 否 |
| `scripts/14_select_with_infogain.py` | 新增 | 统一 selection JSONL | policy | 中 | 否 |
| `scripts/15_export_infogain_official.py` | 可选新增 | 方案 A 的结构导出 | adapter | 低 | 否 |
| `scripts/16_import_infogain_selection.py` | 可选新增 | 官方输出导回 evaluator | adapter | 低 | 否 |
| `scripts/17_summarize_results.py` | 新增 | seeds/method统一表 | metrics JSON | 低 | 否 |
| `scripts/03_compute_label_posteriors.py` | 小改 | 加 batch/cache/provenance | scorer/cache | 中 | 输出需版本化 |
| `scripts/07_eval_rag_classification.py` | 保持；可加 manifest | 统一 evaluator | 现有 | 低 | 保持向后兼容 |
| `src/cbwdm_score.py` | 保持 | CBWDM teacher基准 | 现有 | 无 | 否 |
| `src/selector_cross_encoder.py` | 保持 | state-aware主方法 | 现有 | 无 | 否 |
| `_external/InfoGain-RAG/**` | 只读 | 上游证据/参考 | 固定 commit | 不可直接跑 | 否 |

## 11. Minimal Reproduction Plan

### 11.1 Level 1：CPU fixture

输入：

- `tests/fixtures/posteriors/fever2_toy_posteriors.jsonl`
- `outputs/check_stage12/retrieval/fever2_dev_bm25_top3.jsonl`

拟实施后的命令：

```bash
python scripts/12_build_fever_infogain_teacher.py \
  --posteriors tests/fixtures/posteriors/fever2_toy_posteriors.jsonl \
  --output scratch/infogain_teacher.jsonl

python scripts/14_select_with_infogain.py \
  --retrieval outputs/check_stage12/retrieval/fever2_dev_bm25_top3.jsonl \
  --checkpoint scratch/tiny_infogain \
  --output scratch/infogain_selection.jsonl
```

成功判据：

- `dig == eta[gold]-eta0[gold]` 数值单测；
- adapter round-trip 保留 id/doc_id/title/text/rank；
- selection 必有 `id/selected_doc_ids/selected_docs/method/num_docs`；
- threshold/min-doc边界有测试。

资源：CPU、<1 GB 额外磁盘、分钟级。

### 11.2 Level 2：单 GPU 小样本

输入：

- 同一 FEVER split 的 1k–5k train、500 dev；
- 固定 BM25 top20；
- 同一 Qwen2.5-1.5B或7B generator cache；
- `roberta-base` 或 `roberta-large` selector。

实验：

- naive top4；
- FEVER-DIG pointwise；
- CBWDM feature selector；
- CBWDM state-aware cross-encoder；
- oracle仅 diagnostic。

成功判据：

- 所有方法消费完全相同 candidate ids；
- evaluator model/prompt/hash一致；
- 2–3 seeds 可完成；
- InfoGain score threshold在 dev 校准，不看 test；
- 无 NaN，selection schema全通过。

资源估计：

- 7B generator：24 GB显存量级更稳，量化/分片另计；
- RoBERTa-large：16–24 GB显存通常足够小 batch；
- 20–50 GB磁盘；
- posterior scoring 数小时，selector训练分钟到数小时。实际值需服务器实测。

### 11.3 Level 3：正式服务器

输入：

- 固定 FEVER-2/3 train/dev/test；
- 固定 sentence corpus、BM25 top20（若改 top100则所有方法一起改）；
- Qwen2.5-7B同一 revision；
- 3 seeds。

输出：

- versioned retrieval/posterior/teacher/selection/prediction/metrics；
- manifest 记录 git commit、model revisions、prompt hash、seed；
- 汇总主表和 oracle附表。

成功判据：

- 所有 deployable 方法不读取 dev/test gold label；
- candidate集合、context budget、generator完全一致；
- 结果可由 clean server checkout 重跑；
- 报告 mean±std 和 avg docs/context chars。

资源建议：

- 1×A100 40/80 GB 或等价；posterior生成是主要瓶颈；
- RAM 64 GB；
- 当前 FEVER top20路线 100–200 GB磁盘较稳；
- 若对齐论文 2018 Wiki/top100/多检索器，按 300–500 GB规划，并单独建立检索索引。

未执行正式时间/显存 benchmark；以上是规划估计，不是实测。

## 12. Baseline Fairness Protocol

正式对比必须固定：

1. dataset 原始版本、label mapping 和 split；
2. corpus snapshot 与 passage粒度；
3. first-stage retriever、index 和 candidate ids；
4. candidate top-N；
5. generator model/revision/dtype；
6. query-only/query-doc/final prompt；
7. FEVER answer scoring/verbalizers；
8. final context 的最大文档数和 token budget；
9. 文档拼接顺序；
10. evaluator 与 metric；
11. train/dev threshold校准规则；
12. random seeds。

报告规则：

- `CBWDM oracle` 单列 upper bound，不进入 deployable主表。
- InfoGain 的 DIG teacher同样是 gold-supervised training signal，但其测试 reranker可进入主表。
- 同时报告 accuracy、avg docs、avg evidence chars和 selector inference cost。
- `official-code-inspired`、`paper-faithful reimplementation`、`FEVER-adapted InfoGain` 三种标签必须区分。
- 若官方缺失模块由本项目补写，不得称为“官方代码原样结果”。

## 13. Issues and Priorities

### P0：阻止正确实验或造成结论错误

| 证据 | 问题 | 影响 | 建议 |
|---|---|---|---|
| upstream `vllm_logits.py:226-278` | 只有 token logprob，无论文 confidence/DIG | 无法构造论文监督 | 实现并单测公式；向作者确认参数 |
| upstream 全仓 `rg DIG` | 无 `b1/b2` 和 label builder | 训练输入来源不可复核 | 明确重实现并记录阈值 |
| upstream `roberta_train_multi_loss_v2.py:180-200` | RankNet+首两篇CE ≠ 论文 LogSumExp margin | 不能把脚本结果称论文 faithful | 新实现论文式 loss；保留旧脚本为 ablation |
| upstream `rerank_passage_bert_multi.py:97-117` | 只 rerank 300:500，却写全部500 | 输出混合未排序与已排序 | 修复范围并加全行 score assertion |
| upstream rerank 全文件 | 无 0.2 filter/top4/min2 | 缺少核心推理方法 | 独立 policy 模块 |
| upstream `judge_res.py:17-43` | substring 被标作 EM、固定500 | 指标虚高/越界 | 使用统一 evaluator |

### P1：正式实验前必须处理

| 证据 | 问题 | 影响 | 建议 |
|---|---|---|---|
| upstream train scripts absolute paths | 作者内部路径 | 无法迁移 | CLI/config化 |
| upstream `natural_qa.py:65,96,106,135` | worker需要2参数但 pool只传1参数 | NQ preprocessing失败 | `starmap` 或 `partial` |
| upstream `vllm_logits.py:99` | import即加载4-GPU 7B | import/单测危险 | 移入 main/factory |
| upstream `gen_res.py:177` | 固定500；前4篇无长度校验 | 静默丢数据/重试30秒 | 删除截断、校验 |
| local `03_compute_label_posteriors.py:93-116` | query/candidate逐条评分 | 正式运行瓶颈 | batch+query-only cache |
| local `cbwdm_score.py:58-62` | 仅 identity metric | 与完整 BW目标可能不一致 | 明确方法命名或实现预期 metric |
| local selection scripts | `num_docs` 历史输出不统一 | adapter/eval容易分叉 | 新输出统一 schema，旧输出兼容读 |
| current repo | BGE baseline缺失 | baseline集合不完整 | 先补 BGE 或明确不比较 |

### P2：工程与可维护性

- 上游 `tools.py` 重复 import `requests`，API key/base URL硬编码。
- 上游随机 UUID、shuffle与采样没有完整 seed/provenance。
- 上游没有 requirements、lockfile、checkpoint metadata。
- 当前 posterior未记录 generator/tokenizer revision与prompt hash。
- 当前 metrics没有 per-class/confusion matrix。
- 当前输出全文复制到每个阶段，top100时会显著放大磁盘。

## 14. Open Questions

需向作者确认：

1. 论文公式中 sliding-window size、首 token数量、全部 importance weights 的确切值。
2. `b1/b2` 的数值和三类样本进入 CE/margin dataset 的完整采样代码。
3. 论文 margin loss 的正式实现是否另有未公开版本。
4. 论文报告 `beta=0.75` 与公开 `LAMBDA=0.9` 的版本关系。
5. 推理最终使用 rank head、binary head还是组合 score；0.2 threshold落在哪个 score空间。
6. 官方 filename 中 `0.16` 与论文 threshold `0.2` 的关系。
7. FM2 的 prompt、label verbalizer、DIG confidence和评价代码。
8. 多检索器合并时去重、rank normalization和每检索器配额。

需 GPU/真实数据验证：

- answer span index在 Qwen2.5 chat template上的准确性；
- sequence confidence 的数值稳定性；
- FEVER-DIG threshold分布；
- pointwise InfoGain 与 state-aware CBWDM 在相同 context budget下的真实差异；
- selector score阈值跨 seed/model是否可迁移。

## 15. Appendix

### 15.1 本轮实际执行

```text
git status --short
git branch --show-current
git rev-parse HEAD
git log -1
git clone --filter=blob:none https://github.com/MaYufei-NPU/InfoGain-RAG.git _external/InfoGain-RAG
git -C _external/InfoGain-RAG remote/branch/log/status
rg --files / rg -n / Get-Content（两仓逐文件静态阅读）
Python/关键依赖版本检查
对本地 22 个 Python 文件执行 compile() 语法检查
对上游 13 个 Python 文件执行 compile() 语法检查
CBWDM pure-function toy probe
schema adapter probe
```

结果：

- local syntax compile：22/22 PASS；
- upstream syntax compile：13/13 PASS；
- CBWDM pure-function：`X.shape=(3,2)`，2-step greedy PASS；
- schema export/import：PASS；
- posterior→official DIG：按设计 FAIL，缺少 sequence fields。

### 15.2 未执行

- 完整 DIG 构造；
- 下载 Qwen/RoBERTa 权重；
- GPU模型加载、训练或 inference；
- 远端 API generation；
- 官方 shell脚本；
- 正式 accuracy/EM实验；
- 上游 module import（若 import 会加载模型/数据或启动训练）。

原因：任务明确禁止重型操作；公开代码存在 import side effects、绝对路径和缺失依赖/数据。

### 15.3 检查过的文件

- 上游全部 16 个 tracked 文件；
- 当前 `configs/`、`scripts/`、`src/`、`tests/fixtures/`、`requirements.txt`、`.gitignore`；
- `PROJECT_STATUS_REPORT.md` 仅作线索，核心结论重新由当前源码核对；
- 论文 arXiv abstract、Method 3.1/3.2、Implementation Details、filtering ablation和公开 GitHub README。

### 15.4 本轮文件变更

```text
.git/info/exclude                          修改：加入 _external/ 与 scratch/
INFOGAIN_RAG_INTEGRATION_AUDIT.md          新增：本报告
scratch/infogain_integration_probe/probe.py 新增：被 exclude 的一次性 schema probe
_external/InfoGain-RAG/                    新增：被 exclude 的只读官方 clone
```

没有修改 `configs/`、`scripts/`、`src/`、`tests/`、`requirements.txt`、`data/` 或 `outputs/`。

### 15.5 证据来源

- 官方仓库：<https://github.com/MaYufei-NPU/InfoGain-RAG>
- 论文：<https://arxiv.org/abs/2509.12765>
- 审计上游 commit：`8151a4d81971eee219b497089aa350df0c0e8301`

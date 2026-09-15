# RouteFidelity-EP Sealed P0-B 预注册协议

> 冻结日期：2026-07-18  
> 协议状态：在任何 fresh route capture 之前冻结  
> 原则：只有 selected-placement regret 是 primary effect；不以 tau、P99 maximum 或 hypothetical capacity 代替实际决策损失

## 1. 研究范围与证据边界

本实验只研究：

> teacher-forced、request-local MoE route trace 在不同信息抽象下，是否保持 EP placement 的逻辑跨域 record-cost 决策。

本实验不研究 serving arrival、continuous batching、autoregressive decode、真实 backend latency、NIC bytes、TTFT、TPOT 或 P99。所有 token window 若在 secondary 分析中出现，只能称为 **request-local token chunk**。

## 2. 固定 primary cells

| Cell | Model | Route | Contract | Topology | Primary objective |
|---|---|---|---|---|---|
| P1 | `allenai/OLMoE-1B-7B-0924` | top-8, E64 | C1 rank-major unique-owner | EP8，2 domains × 4 ranks | cross-domain unique-owner records |
| P2 | `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M` | top-16, E32 | C1 rank-major unique-owner | EP8，2 domains × 4 ranks | cross-domain unique-owner records |

P1 与 P2 必须同时通过。C0 expanded expert-major 只是 negative control；C2 domain-partial、EP16、chunk tail、capacity 与其他数据域均为 secondary/exploratory，不得帮助 promotion。

## 3. Fresh data 与泄漏防护

数据固定为 WikiText-2 raw train 的 article-level documents，使用 `get_wikitext2_documents` 和 shuffle seed `20260717`。已扫描当前全部 `data_manifest.json/csv`；validation 60/60 与 test 61/61 articles 已用，不得再作 sealed。

| Split | shuffled offset | requests | hash-of-document-hashes | 用途 |
|---|---:|---:|---|---|
| calibration | 192 | 32 | `264360550fe1caa24f97411155812a746290ad5742f1fe9842555427c7a4ba2c` | placement 构造与工程调试 |
| sealed | 224 | 64 | `255af39aa7bb32c6f972195b0399a3c59b90af0460c11f93f7f780fb07391b04` | 一次性 confirmatory run |
| reserve | 288 | 128 | `5eda4c4314fbb20c64ad56493fb508d4008a7f26c5990c481bbd94c5fa7a4f57` | 本轮不读取 |

两个模型使用相同的 raw documents，每篇按各自 tokenizer 截取前 256 个 non-padding tokens。这是跨模型复现，不是跨数据域复现。

冻结顺序：

1. 先冻结本协议、documents、seeds、topology 和 primary metric；
2. 实现并通过 toy/reference tests；
3. 只打开 calibration route，构造 placement pool；
4. 冻结 placement mappings、source SHA256 和 machine protocol；
5. sealed runner 校验全部 hash 后一次性运行；
6. sealed 输出目录若已存在则拒绝覆盖或重跑。

## 4. Token home 与 topology

- 对每个 split 将 request 按冻结 document SHA256 排序，以 round-robin 分配 home rank 0–7；sealed 中每 rank 恰好 8 个 requests。
- 一个 request 的全部 tokens 具有同一 home rank。
- domains 固定为 `{0,1,2,3}` 与 `{4,5,6,7}`。
- 每个 expert placement 必须保持每 rank expert 数完全相同。

## 5. Contracts

### C0 Expanded expert-major negative control

每个 `(token, expert)` 生成一条单向 logical dispatch record：

\[
J_{C0}(P)=\sum_{t,e\in R_t}
\mathbf 1[domain(owner_P(e))\ne domain(home(t))].
\]

S1-R 精确保留每个 `(request, layer, expert)` degree，因此它必须对所有 placements 与 S4 产生完全一致的 C0 cost vector。任何不一致都使 sealed run 无效。

### C1 Rank-major unique-owner primary contract

同一 token 命中同 owner rank 的多个 experts 只生成一条 logical dispatch record：

\[
J_{C1}(P)=\sum_t
\left|\{owner_P(e):e\in R_t,\
domain(owner_P(e))\ne domain(home(t))\}\right|.
\]

Primary 仅使用每 request 的 cross-domain unique-owner records/token。不将该数量称为 physical frames、wire bytes 或 latency。

### C2 Domain-partial mathematical stress contract

C2 显式区分 combine source domain 和 token home domain，但在对齐某个真实 backend 代码前只能称 mathematical stress contract，不参与 promotion。

## 6. Representation ladder

### S0 Architecture-only uniform

仅保留 E、top-k 和 token count。它是弱 baseline，失败不算 problem gate。

### S1-R Request-conditioned exact degree

在每个 `(request, layer)` 内运行 degree-preserving double-edge swaps：

- 每 token 仍有 top-k 个不重复 experts；
- 每个 `(request, layer, expert)` occurrence count 完全不变；
- token-level coactivation 与顺序被破坏；
- 20 个 seeds：`2026071800 + i, i=0..19`。

每个 seed 必须输出 degree TV=0、duplicate expert count=0、accepted swaps/edge、hyperedge Jaccard 与pair-coactivation distance。

### S2 Request-layer hyperedge multiset

在每个 `(request, layer)` 内保留完整 unordered top-k hyperedge multiset，只打乱 token order。它只用于分离 temporal/order effect。

### S3-W Windowed hyperedge dictionary

固定 `W=32` request-local tokens，对每个 `(request, layer, chunk)` 保留 unordered hyperedge dictionary 及 counts，不保存 chunk 内顺序。`W=16/64/128` 只做 secondary sensitivity。

S3-W 在 C1 total-record objective 上与 S4 等价是构造上的结果，不得包装为 empirical discovery；它的 method gate 只检验以更小 canonical artifact 保留决策是否可行。

### S4 Full ordered route oracle

保存 request、layer、token position 与 unordered top-k expert set。Gate/rank weights、原文和logits不计入 canonical size。

## 7. Placement pool

每个模型固定 132 个 balanced placements：

1. contiguous；
2. round-robin；
3. calibration-only frequency-balanced LPT；
4. calibration-only coactivation-aware balanced placement；
5. 128 个 balanced random placements，seed 为 `2026072000 + i, i=0..127`。

Coactivation 放置使用 calibration pair counts，固定 20,000 次 balanced swap hill-climbing；tie 按 placement mapping SHA256 字典序。所有 representations 使用同一 pool，禁止按 sealed 结果生成或筛选 placement。

## 8. Canonical size accounting

- expert id 使用 `ceil(log2(E))` bits；
- 计入 request/layer/chunk boundaries、counts、dictionary/prototype ids 与 seed；
- 同时报告 raw packed bytes 与 zstd level-3 bytes，但 primary size gate 只使用 raw canonical packed bytes；
- S3-W 必须在两个模型上都满足 `size(S3-W) <= 0.70 * size(S4)`。

不得以 synthesis time 代替 size gate；未实现直接聚合采集时，不得声称 capture overhead 下降。

## 9. Primary metrics

对 representation S 和同一 placement pool：

\[
\hat P_S=\arg\min_{P}J_S(P),
\qquad
Regret(S)=\frac{J_{S4}(\hat P_S)-\min_PJ_{S4}(P)}{\min_PJ_{S4}(P)}.
\]

Tie 使用 placement SHA256 字典序，并额外报告 epsilon-optimal set sensitivity。

Primary effect 是 `Regret(S1-R)`。Kendall tau-b、Spearman rho、argmin-set agreement、top-5 overlap 与seed quantiles 只是 supporting metrics，不得代替 regret。

## 10. Primary hypotheses 与判决

### H-P Problem gate

P1 与 P2 两个 cells 中，S1-R 都必须满足：

- 20 seeds 中至少 16 个 point regret `>=5%`；
- seed-median regret `>=5%`；
- request-cluster bootstrap 后，Holm-adjusted one-sided 95% lower bound `>0`。

任一模型失败即为 `KILL_CCFC_MAINLINE`；只有 S0 失败也必须判死。

### H-M Method gate

仅在 H-P 通过后检验。P1/P2 中 S3-W 均必须满足：

- regret `<=2%`，Holm-adjusted one-sided 95% upper bound `<=2%`；
- tau-b 的 one-sided 95% lower bound `>=0.95`；
- `Regret(S1-R)-Regret(S3-W) >=3` 个百分点；
- raw canonical size `<=70%` S4。

H-P 通过但 H-M 失败时，判定为 `EXACT_REPLAY_ONLY / KILL_METHOD_NOVELTY`。

### H-T Temporal claim

只作 secondary confirmatory。仅当两个模型在固定 W=32 chunk-tail objective 上均出现 `>=5%` regret 且 CI 排除 0，才允许论文保留 temporal claim。

## 11. Bootstrap 与多重比较

- 统计单位是 request/article，运行 10,000 次 paired cluster bootstrap，seed `2026071899`；
- 每次携带被抽 request 的全部 layers/tokens，重新聚合 cost、选择 placement 和计算 regret；
- seeds 只是 synthesis uncertainty，不当作独立 request 增大样本量；
- P1/P2 primary comparisons 使用 Holm correction；
- 不报告全网格 maximum 作为任何 gate。

## 12. Invalid / kill / promote

### `INVALID_SEALED_RUN`

任一 source/protocol/data/config hash 不匹配；sealed IDs 与历史/calibration 重叠；C0 negative control 不 exact；degree TV 非 0；toy cases 失败；或 sealed 输出被覆盖/重跑。

### `KILL_CCFC_MAINLINE`

P1/P2 任一 H-P 失败；只有 uniform baseline 失败；或 effect 只来自 secondary/max scan。

### `EXACT_REPLAY_ONLY / KILL_METHOD_NOVELTY`

H-P 通过但 H-M 失败；只有近完整 trace 才能恢复排序；或 size >70% S4。

### `PROMOTE_TO_GPU_P1`

仅当 sealed run 有效、P1/P2 H-P 同时通过、H-M 同时通过且 size gate 同时通过。GPU P1 至少需要 2 GPU；第一个 backend 必须出现 actual operator-latency regret `>=5%` 且 CI 排除 0，才实现第二 backend。


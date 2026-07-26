# RouteFidelity-EP：严谨实验路径与 P0-B Sealed 验证

> 日期：2026-07-18  
> 最终判决：**`KILL_CCFC_MAINLINE`**  
> 证据边界：teacher-forced、request-local、logical C0/C1 record placement；不等于 backend frame、wire bytes、operator latency、TTFT、TPOT、TBT 或 P99。

## 1. 先给结论

[Observed] RouteFidelity-EP 的 primary problem hypothesis 在两个预注册模型上同时失败：OLMoE top-8 与 LLM-jp top-16 的 20 个 S1-R seeds 均有 **0/20** 达到 5% selected-placement regret，seed median regret 均为 **0**，Holm-adjusted one-sided lower bound 均为 **0**。

[Observed] 失败不是因为反事实没有被有效打乱。OLMoE 的 token-set Jaccard 约 0.132、pair-distance 约 0.399；LLM-jp 因 top-16/E32 稠密结构，Jaccard 约 0.402、pair-distance 约 0.085。所有 seed 均保持 request-layer expert degree 完全一致、duplicate=0，且 C0 对 132 个 placements 全部 exact。

[Observed] 两模型的 Kendall tau-b 只有约 0.33–0.41，表明 S1-R 与 S4 的全配置排序并不相似；但每个 seed 都仍选中与 exact S4 相同的 `calibration_coactivation_balanced` placement。因此低相关性没有转化为错误配置决策。

[Inferred] 这直接否定了当前 CCF-C 主叙事：在本轮固定的 C1 rank-major unique-owner contract、EP8 两域拓扑和现实 placement pool 中，request-conditioned exact expert degrees 已足够选出最优 placement，没有证据证明必须保存 token-level hypergraph/coactivation 才能保持配置决策。

[Observed] 即使绕过 problem gate，S3-W 的 bit-packed canonical artifact 也没有压缩：其大小分别是 S4 的 **109.72%**（OLMoE）与 **106.62%**（LLM-jp），远未满足 `<=70%` 的 method gate。

因此，不应继续为 RouteFidelity-EP 实现 GPU/backend P1，也不应改用 tau、receiver-P99 maximum、capacity exceedance 或 architecture-only 弱 baseline 来恢复主线。

## 2. 被验证的 Idea 到底是什么

一句话定义：

> RouteFidelity-EP 试图为不同 EP communication contract 找到比完整 route 更紧凑、但仍能保持 placement/configuration 决策的最小路由充分统计量。

核心可证伪假设不是“不同 route 表示看起来不同”，而是：

> 若只保留每个 request-layer 的 expert marginal degree、破坏 token-level coactivation，系统会频繁选错 placement，并在 exact route 上造成至少 5% regret；保留 windowed hyperedge 后能以更小 artifact 将 regret 恢复到 2% 以内。

这一定义把研究价值绑定到真实决策损失，而不是 trace 字段还原误差或相关性。

## 3. 完整实验路径

| 阶段 | 做法 | 进入下一阶段的硬条件 | 状态 |
|---|---|---|---|
| P0-A exploratory | 旧 trace 上比较 architecture、degree、hyperedge、exact route | 只能发现信号，不能晋级 | 完成；仅 exploratory |
| 数据审计 | 扫描全部历史 manifests，排除已用文章 | calibration/sealed 与历史零重叠 | 通过 |
| 协议冻结 | 冻结 cells、documents、seeds、placement pool、metric、threshold | 在 fresh capture 前完成 hash freeze | 通过 |
| Contract/reference tests | C0/C1、home-domain、placement、S1-R、regret、bootstrap toy cases | 58 个 core assertions + runner toy tests | 通过 |
| Fresh calibration | 两模型各 32 篇新文章，route-only capture | exact-full、artifact hash、home/rows 闭合 | 通过 |
| Placement freeze | calibration-only 构造 4 fixed + 128 random | 实际 mappings、source hashes、lock 冻结 | 通过 |
| Calibration engineering run | 完整 20 seeds + 10,000 bootstrap | 仅查实现问题，不改 gate/placements | 通过；发现并修复一次 GEMM warning 后重新锁定 v2 |
| Fresh sealed capture | 两模型各 64 篇新文章，一次性目录 | 不读取 route 统计、不覆盖 | 通过 |
| P0-B sealed evaluation | 两模型各 20 seeds、132 placements、10,000 request bootstrap | H-P 与 H-M 均通过才可做 GPU | **H-P 失败，终止** |

## 4. 数据治理与不可泄漏性

数据为 WikiText-2 raw train 的 article-level documents，shuffle seed 固定为 `20260717`。

| partition | offset | requests | hash-of-document-hashes | 用途 |
|---|---:|---:|---|---|
| calibration | 192 | 32 | `264360550fe1caa24f97411155812a746290ad5742f1fe9842555427c7a4ba2c` | placement 构造与工程测试 |
| sealed | 224 | 64 | `255af39aa7bb32c6f972195b0399a3c59b90af0460c11f93f7f780fb07391b04` | 一次性正式验证 |
| reserve | 288 | 128 | `5eda4c4314fbb20c64ad56493fb508d4008a7f26c5990c481bbd94c5fa7a4f57` | 本轮未读取 |

历史 registry 覆盖 28 个 manifests、186 个 unique document hashes；calibration 与 sealed 均通过零重叠检查。两个模型使用同一组 raw documents，因此这是跨模型复现，不是跨数据域复现。

每个 split 内将 request 按 document SHA256 排序，再 round-robin 分配 home rank 0–7；同 request 的所有 token 固定同一 home rank。sealed 每 rank 恰好 8 个 requests。

## 5. Primary cells 与系统 contract

| cell | model | route | placement/topology | primary contract |
|---|---|---|---|---|
| P1 | `allenai/OLMoE-1B-7B-0924` @ `6d84c...9c5` | E64, top-8 | EP8，2 domains × 4 ranks | C1 unique-owner |
| P2 | `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M` @ `1d598...055` | E32, top-16 | EP8，2 domains × 4 ranks | C1 unique-owner |

### C0 expanded expert-major negative control

每个 `(token, expert)` 产生一条 logical record。S1-R 精确保留每个 request-layer-expert degree，且同 request home 不变，因此 C0 对任意 placement 必须 exact。

### C1 rank-major unique-owner primary

同一 token 命中同 owner rank 的多个 experts，只计一条 logical record；primary objective 为每 input token 的 cross-domain unique-owner record 数。

C1 会受到 token-level expert coactivation 与 placement 共同影响，因此它是检验 RouteFidelity problem 是否真实存在的最低成本 contract。但它仍只是 mathematical/reference contract，不代表某个真实 backend 的 frame layout。

## 6. Representation ladder 与强反事实

- `S1-R`：在每个 `(request, layer)` 内做 degree-preserving double-edge swaps；20 seeds 为 `2026071800..2026071819`。
- 每 token 仍为 top-k unique experts；每 request-layer-expert occurrence count 完全不变。
- `S3-W`：W=32 的 request-layer-window hyperedge dictionary；对 additive C1 total cost 与 S4 等价是构造事实。
- `S4`：完整 ordered request/layer/token route oracle。

S1-R 的有效性同时由 accepted swaps、degree TV、duplicate count、token-set Jaccard 与 pair-distance 证明，避免使用“打乱几乎没发生”的弱 null。

## 7. Placement pool 与公平性

每个模型严格固定 132 个 balanced placements：

1. contiguous；
2. round-robin；
3. calibration-only frequency-balanced LPT；
4. calibration-only coactivation-aware balanced hill-climbing，20,000 proposals；
5. 128 个独立 frozen seeds 的 balanced random placements。

实际 expert-to-rank mappings 与 mapping hashes 在 sealed capture 前写入 v2 campaign lock。所有 representations 共享同一 candidate pool；tie 由 mapping SHA256 再按名称字典序决定。没有根据 sealed 结果生成、筛选或删除 placement。

## 8. 统计判据

对 representation S：

\[
\hat P_S=\arg\min_P J_S(P),\qquad
Regret(S)=\frac{J_{S4}(\hat P_S)-\min_P J_{S4}(P)}{\min_PJ_{S4}(P)}.
\]

H-P 要求 P1/P2 同时满足：

- 至少 16/20 seeds 的 point regret `>=5%`；
- seed-median regret `>=5%`；
- 10,000 次 paired request/article bootstrap 后，Holm-adjusted one-sided lower bound `>0`。

Synthesis seeds 作为嵌套不确定性，不当作 20 份独立样本。每次 bootstrap 重采样完整 request，携带其全部 layers/tokens，分别计算 20 个 seed 的 regret 后取 seed median。

## 9. Capture 与完整性结果

| phase/model | requests | route rows | input tokens | MoE layers | exact-full |
|---|---:|---:|---:|---:|---|
| calibration OLMoE | 32 | 1,048,576 | 8,192 | 16 | bit exact |
| calibration LLM-jp | 32 | 2,097,152 | 8,192 | 16 | bit exact |
| sealed OLMoE | 64 | 2,097,152 | 16,384 | 16 | bit exact |
| sealed LLM-jp | 64 | 4,194,304 | 16,384 | 16 | bit exact |

所有 `routes.csv`、`request_manifest.json`、`source_manifest.json` 与 `config.json` hashes 重新计算一致。原模型与 instrumented `full` logits 均 `torch.equal=True`、max/mean absolute diff 均为 0。

## 10. Sealed primary results

| model | seeds regret>=5% | seed median | min/max point regret | Holm lower | Holm p | H-P |
|---|---:|---:|---:|---:|---:|---|
| OLMoE | **0/20** | **0.00%** | 0% / 0% | 0.00% | 1.0 | FAIL |
| LLM-jp | **0/20** | **0.00%** | 0% / 0% | 0.00% | 1.0 | FAIL |

Bootstrap percentile intervals for seed-median regret：

- OLMoE：`[0, 7.77%]`；
- LLM-jp：`[0, 0.83%]`。

上界不为 0 不会挽救 hypothesis，因为 point seed robustness、median threshold 与 lower-bound gate 均失败。

两个模型的 exact best 都是 calibration-only coactivation-aware placement；20/20 S1-R seeds 也都选择它：

- OLMoE exact cost：40.2791 cross-domain unique-owner records / input token；
- LLM-jp exact cost：59.2243。

### 为什么 tau 很低却仍然判死

| model | 20-seed tau-b 范围 | selected regret |
|---|---:|---:|
| OLMoE | 0.328–0.368 | 全部 0 |
| LLM-jp | 0.377–0.413 | 全部 0 |

S1-R 对中后部 placements 的排序变化很大，但最优配置保持不变。论文关心的是系统会不会选错配置，而不是两个 132 维 score vectors 是否高度相关。用 tau 作为 primary 会制造一个看似很强、实际上没有系统后果的假阳性。

## 11. Controls 与 method gate

| model | accepted swaps/seed | pair-distance | token Jaccard | C0 | S3/S4 packed size |
|---|---:|---:|---:|---|---:|
| OLMoE | 10.50M–10.52M | 0.3987–0.3993 | 0.1316–0.1323 | all exact | 109.72% |
| LLM-jp | 6.630M–6.636M | 0.0849–0.0851 | 0.4018–0.4024 | all exact | 106.62% |

H-P 失败后，协议要求 H-M 不运行。即便作机制审计，S3 在 C1 上的 regret=0 只是构造性等价，同时其 dictionary/count overhead 使 packed artifact 比 S4 更大，因此也不满足 compact representation 的论文价值。

## 12. 能排除哪些替代解释

1. **不是旧数据过拟合**：sealed 64 篇文章与历史 186 篇、calibration 32 篇零重叠。
2. **不是 instrumentation 改了模型**：patched full 与原模型 logits 位级一致。
3. **不是 marginal 被破坏**：每 request-layer expert degree TV=0，C0 对全部 seeds/placements exact。
4. **不是反事实没打乱**：OLMoE 有显著 pair-distance/Jaccard 改变；LLM-jp 的较小可扰动空间被量化并报告。
5. **不是随机 seed 偶然**：20 个 seeds 全部 point regret=0。
6. **不是 token 假独立造成显著**：统计单位为 64 篇 request/article，Holm 后 lower bound=0。
7. **不是候选池太小的单一 baseline**：包含 132 个 frozen balanced placements 与 calibration-derived strong placements。

## 13. 仍然存在的边界

- C1 是 rank-major unique-owner mathematical contract，不是真实 DeepEP/NCCL/TensorRT backend。
- 只有一种 EP8 两域拓扑和一个固定、与内容独立的 home assignment。
- 两模型使用相同 WikiText-2 数据域。
- Primary 只验证 placement，不覆盖 buffer/protocol selection。
- LLM-jp top-16/E32 的稠密性限制了 degree-preserving null 能破坏的 pair structure。

这些边界说明结果不能证明“所有 backend 的所有 route-aware decision 都不需要 coactivation”。但它们不能恢复预注册的 CCF-C 主线：该主线明确要求两个 cells 同时通过，而两者均为 0/20。

## 14. 最终研究决策

### 不做

- 不进入 RouteFidelity GPU P1；
- 不把低 tau 改写成“problem 已成立”；
- 不用旧 P99 maximum、hypothetical capacity 或 C2 post-hoc scan 复活；
- 不修改 5%/16-of-20 gate 后重跑 sealed；
- 不声称 S3 是 compression，因为它比 S4 更大。

### 可以保留

1. **硕士论文的方法学/negative result**：完整展示为什么 trace similarity 与 decision fidelity 不等价。
2. **可复用 experiment harness**：fresh split、route capture、C0/C1 reference、balanced placement registry、exact-degree null、cluster bootstrap 与 fail-closed manifests。
3. **EP-WireScope 的 measurement component**：若之后获得 GPU，只可把 C0/C1 oracle 用作 backend record closure 测试，不再把 RouteFidelity placement optimizer 当主贡献。
4. **新 Idea 的前置筛选器**：任何新 route/topology 假设必须先证明会改变具体系统选择或造成可观测 backend regret，不能只展示分布差异。

## 15. 关键 artifacts

- 冻结协议：`RouteFidelity_EP_Sealed_P0B_Protocol_2026-07-18.md`
- Machine protocol：`experiments/idea_a_mac/outputs/route_fidelity_p0_2026-07-18/p0b_frozen_protocol/machine_protocol.json`
- v2 campaign lock：`experiments/idea_a_mac/outputs/route_fidelity_p0_2026-07-18/p0b_placement_lock_v2/campaign_lock.json`
- Runner：`experiments/idea_a_mac/run_route_fidelity_p0b.py`
- Core：`experiments/idea_a_mac/route_fidelity_p0b_core.py`
- Sealed summary：`experiments/idea_a_mac/outputs/route_fidelity_p0_2026-07-18/p0b_sealed_eval_v1/summary.json`
- Sealed report：`experiments/idea_a_mac/outputs/route_fidelity_p0_2026-07-18/p0b_sealed_eval_v1/report.md`
- Bootstrap replicates：`experiments/idea_a_mac/outputs/route_fidelity_p0_2026-07-18/p0b_sealed_eval_v1/bootstrap_regret_replicates.npz`

## 16. 最终判词

[Observed] RouteFidelity-EP 的核心 problem 在 sealed P0-B 中没有成立，compact method 也没有 size advantage。

[Inferred] 它不应再作为 CCF-C 通信优化/工作负载合成主线；继续投入 GPU 的期望信息增益低。

[Inferred] 本轮最大的研究成果不是“又得到一个 idea”，而是用一条可复核的逻辑链排除了一个看似合理但系统后果不足的方向，并留下了可直接复用的严谨实验基础设施。

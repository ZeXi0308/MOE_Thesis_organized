# GroupedOwnerEP 生死实验预注册修订（2026-07-14）

> 本修订写于 grouped-owner 正式结果产生之前。它不删除
> `FixedRateEP_生死实验预注册_2026-07-14.md`，而是记录一次 P0 实现语义审计：旧
> `peerblock_*` 把 routed expert pair 当作 combine wire item；DeepEP 的 non-expanded / local-reduction
> 路径可以先在 expert owner 上合并同一 source token 命中的本地 experts，再回传一个
> `(source token, owner)` 向量。旧实验仍可解释 expanded layout 的数值行为，但不能直接证明
> owner-reduced combine 的质量或字节收益。

## 1. 修正后的通信单位

对 layer `l`、source token `t`、expert owner `o`，定义：

$$
v_{l,t,o}=\sum_{e\in S_l(t):\operatorname{owner}(e)=o}g_{l,t,e}f_{l,e}(h_{l,t}).
$$

本轮先在 owner 内按 expert id 顺序做 gated local accumulation，再对 `v` 整体执行 BF16、FP8
或 MXFP4 编解码。一个真实 routed pair 计数为 `N`，去重后的 token-owner item 计数为 `M`；
必须同时报告 `N/M`。忽略 scale/metadata 时，50/50 FP8/MXFP4 expanded-pair layout 相对
owner-reduced uniform FP8 只有在 `0.75N < M`，即 `N/M < 4/3` 时才节省 payload。

本文所称 fixed quota 仅表示：

> 在每个实际 item 数为 `m` 的 `(layer, source, owner, tile)` 中，high lane 数固定为
> `floor(rho*m)`（或预先指定的确定性取整），其余进入 low lane。

它不是 fixed message volume；实际 routed count、最后一个短 tile、scale、alignment 和 header
仍会改变物理字节数。

## 2. 预注册问题与零假设

### H1：owner-reduced mixed precision 是否仍有质量 headroom

在完全相同的 token-owner item、相同 high/low cardinality 和相同 codec 下，criticality selector
应比 deterministic random 更低 KL。零假设是：owner local sum 已改变误差分布，动态选择不再优于
random，或 uniform FP8 已是唯一可接受点。

### H2：origin-available selector 是否足够

生产可行的第一优先级 selector 必须能在 expert compute 前由 source/origin 计算：

- `gate_mass = sum(g_e)`；
- `max_gate` / best-rank 作为固定-rank类对照；
- `inputnorm_gate = ||h_t|| * gate_mass` 作为无泄漏 cheap proxy；
- 若实现 calibration，则 `profiled_gain = ||h_t|| * sum(g_e * alpha_{l,e})`，其中
  `alpha_{l,e}=E_cal[||f_{l,e}(h)||/||h||]` 只由 calibration 拟合。

owner-output `||v_{l,t,o}||`、MXFP4 error 和 FP4->FP8 distortion gain 是 late/oracle selector，
只用于测 headroom。零假设是：只有 owner-output oracle 有效；若如此，router-prefix 主线死亡，
只能另立 owner-side late-binding kernel，并单独支付 selector/pack 开销。

### H3：结论是否依赖 expert placement

Primary placement 冻结为 contiguous EP8：OLMoE E64 每 owner 8 experts，LLM-jp E32 每 owner
4 experts。Stress placements 为 contiguous EP2/EP4 与 round-robin EP4/EP8。零假设是：收益只在
某个有利的人工 mapping 上出现。

## 3. 冻结数据与配置

为避免复用已经看过的 confirm articles，本轮分两级：

1. **implementation/dev**：OLMoE WikiText validation offset `32`，最多 8 articles，seq 128/256；
2. **frozen confirmation**：必须使用与旧 calibration、R-layout formal、FixedRateEP confirm 均无
article-hash overlap 的 document split；若当前本地缓存没有足够未看文章，则只报告 dev，禁止伪称
formal confirm。

固定 codec 为 per-vector scaled FP8 E4M3 与 block-32 MXFP4 E2M1；primary low fraction `rho=0.5`；
primary peer tile 为 64 个 token-owner items。tile 16/32/128 只能作为 stress，不得在 confirm 上挑最好点。

## 4. 必跑策略与指标

必跑策略：original full、grouped BF16、grouped uniform FP8、grouped uniform MXFP4、matched-rate
random、best-rank/max-gate、gate-mass、inputnorm-gate、grouped contribution、grouped qbenefit。
若 profiled-gain 完成，则 calibration/test 严格分离。

Primary endpoint 为相对 original full logits 的 document-level mean next-token KL；secondary 为 corpus
PPL。所有主比较以 article 为独立单位做 paired bootstrap 10,000 次，并对同一组 selector 使用 Holm
校正。另报告：

- grouped BF16 与 original full 的 KL/最大 logits 差，区分 accumulation-order error；
- 每层 local relative MSE；
- `N`、`M`、`N/M`、每 token unique owner 数与 empty/short tile 比例；
- exact high/low count、logical payload（含 scale）及 expanded-vs-reduced break-even；
- selector membership overlap，特别是 gate-mass 到 contribution 需要交换的 item 比例；
- contiguous/round-robin 与 EP2/4/8 的方向一致性。

Mac fake quant 仍不能产生 native FP4、actual wire、kernel latency、TTFT、TPOT 或 P99 结论。

## 5. 生死门

### Gate A：router-prefix 版本继续

同时满足才继续：

1. `gate_mass`、`inputnorm_gate` 或严格 held-out 的 `profiled_gain` 至少一个在两个模型上都显著
优于 matched-rate random，paired KL CI 上界 `<0`；
2. 相对 best-rank/max-gate 的方向在两个模型一致，且至少一个模型 CI 排除 0；
3. deployable selector 到 grouped contribution oracle 的剩余差距不超过
`30% * (KL_random - KL_contribution)`；
4. primary contiguous EP8 成立，且 stress placements 不出现系统性反转；
5. grouped mixed point相对 grouped uniform FP8 确实少约 25% 主 payload，计入 scale 后仍少至少 20%。

任一关键项失败，删除“router-guided critical prefix 是主方法”的表述。

### Gate B：owner-side late binding 继续

若 Gate A 失败、但 grouped contribution/qbenefit 在两个模型都显著优于 random 和 deployable selector，
只允许转为 owner-side late-binding 候选。此时 GPU artifact 必须额外证明：score + segmented select +
two-lane pack 总开销不超过 uniform FP8 combine 时间的 10%，且通信净时延仍降低至少 10%；否则判死。

### Gate C：整个 mixed-combine 方向停止

满足任一项即停止：

- grouped selector 对 random 没有稳定优势；
- `N/M >= 4/3` 且方法仍依赖 expanded pair transport；
- uniform MXFP4 已满足质量，动态 mixed precision没有独立 Pareto价值；
- 计入 scale/alignment 后相对 grouped uniform FP8 少于 20% wire；
- 真实 GPU 上 prefill/high-concurrency combine 相对 uniform FP8 改善不足 10%，或 decode 回退超过 3%；
- serving 端到端 P50 改善不足 3%、P99 改善不足 5%，或质量约束 goodput 改善不足 5%。

## 6. 当前允许的论文表述

在 Gate A/B/C 完成前，只能写：

> 我们正在检验一种 owner-reduced、peer-local fixed-composition mixed-precision combine；现有
> routed-pair 结果是 expanded-layout mechanism evidence，不是 DeepEP 真实 wire 或 latency 证明。

禁止写“已证明 QuotaEP 改善通信/TTFT/TBT”或“fixed volume”。

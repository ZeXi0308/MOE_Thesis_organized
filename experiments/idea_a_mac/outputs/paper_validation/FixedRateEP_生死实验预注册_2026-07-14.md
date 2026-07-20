# FixedRateEP / RefineEP 生死实验预注册（2026-07-14）

> 本文在 selector 开发长跑完成前写入。目标是限制事后挑策略、挑 block size、挑数据集解释结果。Mac 只做质量机制筛选，不产生通信或延迟结论。

## 1. 待检验的两个候选

### H1：peer-tile fixed-rate allocation

在每个 `(layer, token-origin -> expert-owner)` routed-pair tile 内固定 FP8/FP4 数量，但用 gate 动态决定成员。它试图取得：

- 比 per-token fixed-rank 更接近 global gate threshold 的质量；
- 比 global threshold 更规则的 lane count、buffer offset 和 collective schedule。

准确术语是 **fixed precision composition conditional on routed-pair count**，不是 fixed message volume。Mac 的 `block_*` 是中央 token-block proxy；`peerblock_*` 在单一隐含 origin 下按 synthetic owner group 分流，仍不是真实多 rank EP。

### H2：self-revealing residual refinement

所有 routed pairs 先编码 MXFP4 base；base encoder 同时暴露 quantization residual energy；固定一半 pairs 再发送 MXFP4 residual refinement。比较：

- `resenergy`：owner-local `||o-Q4(o)||²`；
- `reserr`：`g²||o-Q4(o)||²`；
- `resbenefit`：对所有 pairs 都执行 refinement 后计算的 pair-additive benefit 上界。

`resbenefit` 忽略同 token 不同 expert error 的交叉项，不是 exact combine oracle；它还会全量执行候选 codec，只能判断 headroom。

## 2. 数据隔离

| 用途 | 模型 | calibration | data | 状态 |
|---|---|---|---|---|
| block-size 与 selector dev | OLMoE top-8 | WikiText validation `[0:16]` | validation `[32:48]`，seq 256 | 允许选方法，不作确认结论 |
| OLMoE frozen confirm | OLMoE top-8 | validation `[0:16]` | validation `[16:32]`，seq 256 | 在看到结果前冻结 |
| 跨模型 frozen confirm | LLM-jp E32-k16 | validation `[0:16]` | test `[1:61]`，seq 256 | test doc 0 已用于兼容性 smoke，故排除 |

同一 WikiText article 是最小推断/重采样单位。禁止把 token 或同文档窗口当 IID 样本。

## 3. 已冻结的超参数

- direct token block：`8`（由 2026-07-14 早期 dev sweep 选择）；
- residual token block：`16`（只作候选筛选，不保证保留）；
- one-origin peer tile：`64 routed pairs`，`8` 个 contiguous synthetic owner groups；
- low-bit format：MXFP4 E2M1，32-element block scale；
- direct 50/50 与 residual 50% refinement 的 raw payload 都是平均 6 bit/element；
- 按当前 fake codec 的 scale 字节匹配后，residual 对应 direct low-bit fraction 为 `43.6%`；主要同预算比较要求 logical format bytes 差异不超过 `1%`。

不得在 OLMoE confirm 或 LLM-jp test 上重新扫 block/tile size。

## 4. 强对照与指标

强对照：full、uniform FP8、uniform MXFP4、fixed rank、global calibrated gate threshold、fixed-quota gate、contribution、unweighted error energy、gated error energy、pair-additive benefit、deterministic random anti-control、reverse-gate anti-control、metadata-matched direct mixture。

Primary endpoint：相对 full logits 的 corpus mean per-token KL。Secondary endpoint：corpus PPL/PPL delta。统计使用 document-cluster paired bootstrap 10,000 次；开发阶段多 selector 比较使用 Holm 校正。PPL 与 KL 方向冲突时，不允许只挑更好看的一个；以预注册 primary KL 决策，并报告冲突。

## 5. 生死门

### Gate A：fixed-rate quality mechanism

令 `gap = KL_rank - KL_global_gate`，仅当 `gap > 0` 时定义：

```text
recovery = (KL_rank - KL_fixed_rate) / gap
```

在两个 frozen confirm 上要求：

1. recovery point `>=70%`；bootstrap 95% 下界建议 `>=50%`；
2. fixed-rate vs rank 的 paired KL CI 上界 `<0`；
3. 对 global gate 的 non-inferiority margin 为 `0.30 × gap`；
4. 两模型方向一致，且 random/reverse-gate 明显更差；
5. one-origin peer-tile proxy 不能比中央 block proxy 出现不可解释的大幅反转。

若只在 OLMoE 成立，方法降级为模型特定 observation；若只在中央 block 成立、peer tile 失败，则停止系统故事。

### Gate B：residual codec

在相同 logical format bytes 下：

1. `resbenefit` 相对所有 direct mixed policies 的 Pareto envelope，KL 至少低 `15%`，paired CI 排除 0；否则无 residual headroom，停止；
2. 若 1 成立，`reserr` 至少追回 `gate residual -> resbenefit` gap 的 `70%`，且相对 gate residual 至少低 `10%`；否则只有不可部署 oracle；
3. `reserr` 必须优于 `resenergy`，或证明无需 gate metadata；
4. 两模型、两份 disjoint test 方向一致。

任一关键条件失败，residual 从主线删除，只保留 negative result。不能用 graceful/anytime 叙事挽救：若下一层仍等待 refinement，它没有 early-completion latency 语义。

### Gate C：真实系统（Mac 不能执行）

进入 GPU 实现后必须补：

- actual packed bytes、scale、token id、lane membership、count/offset、padding/alignment、protocol header；分析与 buffer bytes 差异 `<1%`；
- 同 backend 的 selector、quant、pack、network、unpack/dequant、gate-reduce 分解；
- LL decode 与 HT prefill 分开；
- 相对 uniform FP8 operator 核心带宽形状至少 `10%`，相对 strongest dynamic-gate kernel 同质量至少 `5%`；
- serving TTFT/TPOT P50 至少 `3%`，P99 至少 `5%`，quality-SLO goodput 至少 `5%`，平衡/低负载回退不超过 `2%`；
- 两模型、两 topology、真实 DeepEP/NCCL EP backend。

未过 Gate C 前，不写“提升 TTFT/TBT/TPOT/P99”，也不把 metadata-aware logical saving 称为 physical wire saving。

## 6. 预先声明的替代解释

- fixed-rate 变好可能只是更接近 global threshold，不是新 criticality 发现；系统价值必须来自固定 lane count 的可测开销收益；
- qerror/residual benefit 可能只是 fake codec scale granularity 的产物；native codec 排序若反转，本地结论作废；
- contiguous token order、tile offset 或 co-batch 可能决定成员；必须补 order/offset/request P95 稳定性；
- dynamic routing drift 可能放大微小局部误差；需 frozen-route 对照；
- residual 可能被更简单的 FP6/INT6 或 direct FP8/MXFP4 mixture 支配；
- decode 可能 launch-bound，第二 lane/selector 反而更慢；若只在 prefill 成立，主动收缩为 TTFT/prefill 方法。

## 7. 新颖性边界

不能声称首次 mixed precision、首次 FP4 EP、首次 base+residual、首次 importance-aware fixed budget。待查重后只可能保留：

> 在真实 MoE combine `(origin -> owner)` peer stream 上，以 routed contribution/quantization difficulty 为分配单元，将固定码率选择、可解包 lane protocol 与 EP kernel 联合设计。

如果该 EP-specific protocol 不能在真实 backend 上产生独立收益，即使 Mac KL 成立，也不足以形成 CCF-B 级 AI Infra 论文。

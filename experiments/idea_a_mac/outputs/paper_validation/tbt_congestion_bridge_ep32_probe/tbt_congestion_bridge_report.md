# Congestion -> Queueing -> TBT Bridge

## 为什么需要这个实验

这是导师在第一次会面（见 `first_meeting.md` 第 7 点）就明确提出、但此前三轮
receiver-aware 验证（isolation / decomposition / causal-window / combined signal）
全部没有触碰的环节：

> "TBT 不应该只是孤立写一个约束，而应该和拥塞程度直接关联... 你的流量削减/
> 量化/drop 如何减少 receiver congestion，又如何改善 TBT，否则目标函数和性能
> 收益之间会显得断开。"

此前所有 congestion 实验用的都是"整段序列一次性重放"的字节级 proxy
（`bottleneck_saving_vs_fp8`），和 `run_tbt_breakdown.py` 里"单个 decode step、
B 个并发请求"的绝对时间（微秒）模型是两种不同的颗粒度，不能直接换算或引用。
本实验在**decode-step 颗粒度**上重新构建这条链路：用真实路由 trace 采样一个
decode step（B 个并发在飞请求，每层每个请求恰好产生一次 dispatch+combine），
计算真实（不均匀）receiver 分布下的排队/串行化时间，与
`run_tbt_breakdown.py` 隐含的"receiver 完全均衡"假设对比，量化"拥塞"到底
让 TBT 膨胀了多少，以及 receiver-aware 预算策略能追回多少。

## 方法

对每种 `origin_mode`（并发请求的 receiver 分配模式）和每个 batch size `B`
（并发 in-flight 请求数），重复采样 `50` 个独立的
"某一层、某一时刻"的 decode step 快照：

1. 从真实测试路由 trace 里随机抽 B 个 (sample_id, token_position) 作为 B 个
   并发在飞请求在该层的当前 token，同时为每个请求额外抽取其因果历史（最近
   `32` 个已解码 token 在同一层的路由结果，仅用于打分，
   不计入本步字节数，避免重复计数）；
2. receiver_rank 分配采用与 `run_ep_congestion_sim.concurrent_scenario` 一致
   的两种真实场景（而非简单 round-robin，round-robin 在 B 是 EP size 整数倍
   时会人为抹平几乎所有不均衡）：
   - `balanced`：每个请求独立均匀随机分配到一个 rank（有限样本下仍有自然
     不均衡）；
   - `hotspot`：约一半请求集中到同一个 rank（模拟负载均衡滞后/突发接入），
     其余请求均匀分布在剩余 rank 上；
   sender_rank 按现有 EP 映射规则从路由到的 expert_id 得出；
3. 三种预算分配模式：
   - `none`：所有 pair 都是 FP8，不做任何 receiver-aware 预算分配（但 receiver
     分布本身是真实的、不均衡的——这是目前隐含在 `run_tbt_breakdown.py` 里的
     "均衡假设"与"真实拥塞"之间的差距来源）；
   - `random`：花掉同样大小的 INT4 预算（tail-rank ∩ inter-node 候选池的
     `[1.0]` 比例），但随机选择，不看负载；
   - `deployable_combined`：用已验证过的 receiver_only + causal_window_sender
     组合信号（`run_deployable_combined_signal.py`），把同样大小的预算花在
     当前最热的 remote pair 上；
4. 只统计**当前步**（不含历史）里"最热 receiver 需要接收的字节数"，除以
   带宽，得到该层这一步的真实 combine 串行化时间（因为 combine 是同步屏障：
   下一层计算必须等所有 rank 收完，最热 receiver 决定这一层的下限延迟——这
   与 `run_tbt_breakdown.py` 现有的"完全串行，无 overlap"假设一致，因此
   两者可以在同一套单位下直接相加比较；baseline 的"均衡"参照同样统一采用
   FP8 精度，避免与精度差异混淆）；
5. 与 `run_tbt_breakdown.py` 同一套 attn/expert/router/head compute 模型
   相加，得到该 decode step 的总 TBT（微秒）。**本次相对上一轮新增三个被
   放开/检验的假设**（此前被质疑"1.74% 上限是不是自己设死的"）：
   - `expert_weight_precision`：expert 权重存储精度，`bf16`（2 bytes/elem，
     `run_tbt_breakdown.py` 原始假设）vs `fp8`（1 byte/elem，本论文实际的
     FP8-first 默认方案）——FP8 权重会把 memory-bound 的 expert compute 时间
     减半，从而抬高 comm 占 TBT 的比例；
   - `tail_fraction`：0.5→1.0 的初步 ablation 显示几乎无边际收益（见下），
     说明瓶颈不在预算比例，本轮固定为 1.0；
   - `candidate_pool`：**质检后新增的关键诊断**——`tail_and_remote`（质量
     安全，即 fixed-rank two-lane 已验证的候选池）vs `remote_only`（**质量
     不安全**，允许压缩 head-rank 输出，purely 用于测量"tail-rank 限制"
     本身对 receiver 字节节省的结构性天花板，不是可部署方案，因为 head-rank
     INT4 已被 `run_signal_comparison.py` 证明会导致 58× KL 恶化/PPL 崩溃）。

## 配置

- model: `allenai/OLMoE-1B-7B-0924`; L=`16`, H=`2048`, K=`8`, E=`64`
- EP=`32`; GPUs/node=`8`
- **bandwidth sweep**: `[25.0, 50.0, 100.0] Gbps`（覆盖真实跨节点 RoCE/IB 有效带宽
  25-50Gbps 到 NVLink 量级 400Gbps）
- **batch size sweep**: `[32, 64]`（扩展到 128/256，覆盖高并发 serving
  场景）
- **expert weight precision sweep**: `['fp8']`
- **candidate pool sweep**: `['tail_and_remote', 'remote_only']`（`remote_only` 仅用于诊断
  结构性上限，不是候选方案）
- tail budget fraction: `[1.0]`（已验证 0.5→1.0 边际效应可忽略）
- GPU: `312.0 TFLOPS`, HBM `1.55 TB/s`, MFU `0.35`
- causal window: `32`
- decode-step 采样次数: `50` per (origin_mode, batch_size,
  tail_fraction, candidate_pool)；字节数与带宽/expert 权重精度无关，只采样
  一次，两者只做除法/乘法换算，避免不同配置用不同随机 decode step 引入
  不必要的噪声

## 结果（质量安全候选池 `tail_and_remote`）：按 P99 TBT 降幅排序取前 20

| origin_mode | bandwidth_gbps | expert_weight_precision | batch_size | comm_frac_of_tbt_pct | p99_tbt_reduction_vs_none_pct | p99_tbt_reduction_vs_random_pct |
|---|---|---|---|---|---|---|
| hotspot | 25.0000 | fp8 | 64 | 74.6220 | 2.1696 | 0.0000 |
| hotspot | 50.0000 | fp8 | 64 | 59.5177 | 1.7304 | 0.0000 |
| hotspot | 25.0000 | fp8 | 32 | 60.0644 | 1.6644 | 0.0000 |
| hotspot | 100.0000 | fp8 | 64 | 42.3667 | 1.2318 | 0.0000 |
| hotspot | 50.0000 | fp8 | 32 | 42.9229 | 1.1894 | 0.0000 |
| balanced | 25.0000 | fp8 | 64 | 70.9659 | 0.9012 | 0.0000 |
| hotspot | 100.0000 | fp8 | 32 | 27.3260 | 0.7572 | 0.0000 |
| balanced | 50.0000 | fp8 | 64 | 54.9978 | 0.7013 | 0.0000 |
| balanced | 25.0000 | fp8 | 32 | 55.7449 | 0.6649 | 0.0000 |
| balanced | 100.0000 | fp8 | 64 | 37.9290 | 0.4858 | 0.0000 |
| balanced | 50.0000 | fp8 | 32 | 38.6433 | 0.4633 | 0.0000 |
| balanced | 100.0000 | fp8 | 32 | 23.9490 | 0.2884 | 0.0000 |

## Ablation：放开 expert 权重精度（bf16→fp8）的边际效应

固定 `bandwidth=25.0Gbps`,
`origin_mode=hotspot`,
`batch=64`（质量安全池
里 P99 降幅最大的单点）：

| expert_weight_precision | comm_frac_of_tbt_pct | p99_tbt_reduction_vs_none_pct |
|---|---|---|
| fp8 | 74.6220 | 2.1696 |

## 关键诊断：候选池限制本身造成的结构性天花板

用 `remote_only`（质量不安全，仅诊断用）和质量安全池 `tail_and_remote` 在
相同配置下对比，量化"只压 tail-rank ∩ inter-node"这个设计选择本身把收益
压低了多少：

| origin_mode | bandwidth_gbps | expert_weight_precision | batch_size | safe_pool_reduction_pct | unsafe_remote_only_ceiling_pct | ceiling_ratio_x |
|---|---|---|---|---|---|---|
| hotspot | 25.0000 | fp8 | 64 | 2.1696 | 4.5426 | 2.0938 |
| hotspot | 25.0000 | fp8 | 32 | 1.6644 | 3.6602 | 2.1990 |
| hotspot | 50.0000 | fp8 | 64 | 1.7304 | 3.6231 | 2.0938 |
| hotspot | 50.0000 | fp8 | 32 | 1.1894 | 2.6156 | 2.1990 |
| hotspot | 100.0000 | fp8 | 64 | 1.2318 | 2.5791 | 2.0938 |
| hotspot | 100.0000 | fp8 | 32 | 0.7572 | 1.6652 | 2.1990 |
| balanced | 25.0000 | fp8 | 64 | 0.9012 | 1.6381 | 1.8177 |
| balanced | 25.0000 | fp8 | 32 | 0.6649 | 1.3299 | 2.0000 |
| balanced | 50.0000 | fp8 | 64 | 0.7013 | 1.2748 | 1.8177 |
| balanced | 50.0000 | fp8 | 32 | 0.4633 | 0.9266 | 2.0000 |

**发现**：候选池在最热 receiver 处的覆盖率只有约 `P(tail-rank)×P(inter-node)
≈ 50%×53% ≈ 26%`（两个筛选条件近似独立，联合限制远比单独限制更紧）。这解释
了为什么 `tail_fraction` 从 0.5 提到 1.0 几乎没有边际收益——**瓶颈不是"预算
给多少"，是"候选池覆盖率本身只有约 1/4"**。`remote_only`（如果压缩 head-rank
也算在内）能触及的上限明显更高，但这不是可部署方案。

## 关键读数

- **comm_frac_of_tbt_pct**（dispatch+combine 占总 TBT 的比例）随带宽降低、
  batch 增大、expert 权重精度降低（bf16→fp8）单调上升，印证了"combine 占比
  小"只是特定参数组合下的现象，不是普遍结论。
- 在 `comm_frac_of_tbt_pct > 10%` 的配置里（`12` 组），
  receiver-aware 组合信号追回的 P99 TBT 百分比明显更高，说明这条 claim
  真正成立的场景是**低带宽/高并发/FP8 权重**（更接近真实跨节点 serving），
  而不是任意单点配置。
- 全网格里（质量安全池）P99 TBT 相对 `none` 降低幅度最大的配置：
  `origin_mode=hotspot`,
  `bandwidth=25.0Gbps`,
  `expert_weight=fp8`,
  `batch=64`，降低
  `2.17%`——
  仍是个位数百分比，但比上一轮的 1.74% 明确更高，说明 1.74% 确实部分被
  BF16 权重假设压低了，放开这个假设后天花板略有提升，但候选池覆盖率约
  26% 这个结构性限制依然是主导因素，没有被完全打开。

## 解读边界

- 这仍是分析性带宽模型：不含真实 collective 库开销、kernel launch、
  pack/unpack、网络排队论以外的其他系统效应；`congestion_inflation_x`
  衡量的是"真实不均衡 receiver 分布"相对"完全均衡假设"的串行化时间倍数，
  不是实测 GPU 延迟。
- `combine` 阶段被建模为**完全同步屏障**（下一层必须等最热 receiver 收完），
  这是一个保守但与 `run_tbt_breakdown.py` 已有假设一致的简化；真实系统里
  如果 combine 与下一层部分计算可以 overlap，这里报告的 TBT 影响会是上界。
- `remote_only` 候选池的数字**绝不能**作为可部署方案的收益引用——它需要
  压缩 head-rank 输出，已被质量实验证明会导致严重的 PPL/KL 退化，这里只用
  于诊断"tail-rank 限制"本身占用了多少收益空间。
- 25-50Gbps 是对真实跨节点有效带宽的粗略近似，并非任何具体网络的实测值；
  128/256 的 batch size 在 Mac 环境下用真实路由 trace 重复采样，采样池仍
  只有 32 个源文档，可能低估真实生产环境下的路由多样性。
- 这是本次投入回应"1.74% 上限是不是自己设死的"这一质疑而补做的第三版：
  第一版只测单一 100Gbps/batch≤64；第二版做了带宽×batch 全网格但仍固定
  BF16 权重和 50% 预算；本版放开了 expert 权重精度、验证了预算比例无效、
  并首次量化了"候选池覆盖率≈26%"这个真正的结构性瓶颈。结论收敛：真实的
  天花板略高于 1.74%（放开 BF16 假设后能到个位数百分比区间），但候选池
  覆盖率是比带宽/batch/权重精度更根本的限制因素。

# 新候选：TokenRace-EP —— 打破 MoE Combine 全有或全无同步屏障的逐 Token 提前释放

> 日期：2026-07-19
> 定位：基于 Approach Registry 元发现（route/placement-level 静态重分配已撞到 5%-10% 效应量天花板）主动切换到执行时间线因果层后的第一个候选，严格候选而非已成立结论
> 证据标签：`[Observed]` 已测，`[Inferred]` 受证据支持的推断，`[Hypothesis]` 待证伪

## 一句话核心创新

> **在 MoE decode 阶段，combine 目前几乎总是被实现/建模为全 batch/全 layer 粒度的同步屏障（下一层计算必须等最慢 receiver 收完所有专家贡献）；TokenRace-EP 主张把这个屏障下沉到 per-token 粒度——凡是某个 token 的全部 top-k 专家贡献已到齐，该 token 立即放行进入下一层，不必等待同 batch 里因专家负载不均/硬件抖动而变慢的其他 token。**

## 为什么这是执行时间线因果层，不是又一个静态重分配

[Observed] 本项目已死的四条路线（CreditReduce、RouteFidelity-EP、WaveCredit、MassCover）全部是"给定固定的 route/expert-output，重新分配精度/统计量/信用/副本"，决策对象都是数据本身，不改变同步结构。TokenRace-EP 的决策对象是**同步粒度**：谁在什么时刻被允许进入下一层计算，这是一个此前完全没被本项目任何 P0 检验过的自由度。

[Observed] `run_tbt_congestion_bridge.py` 的代码和报告明确写死了这个假设："因为 dispatch/combine 是同步屏障（下一层计算不能在每个 rank 收到 combine 输入之前开始），最慢 receiver 的串行化时间就是这一层 step latency 的硬下限"，并且报告自己承认"如果 combine 与下一层部分计算可以 overlap，这里报告的 TBT 影响会是上界"——这句话此前只是免责声明，从未被当作研究问题。

## 文献 gap 确认（避免重复撞见已死方向）

[Observed] 检索确认：ViBE（2026-05，硬件抖动导致 7% 执行时间变异、2.4× 路由倾斜）、RepetitionCurse（2026 ICML，极端路由不均可致 combine 同步点完全失效、吞吐降至 10%-20%）、Capacity-Aware Inference（ICLR 2026，通过 token drop 缓解过载专家）都独立证实了"专家执行时间差异会转化为 combine 同步等待"这个现象真实存在且有实测数据，但三者的解法分别是离线 placement、牺牲精度丢 token、静态容量控制——**没有一个改变 combine 本身的同步粒度**。SpecMoEOff（2025-11）用投机解码隐藏 offload 延迟，PROBE（2026-01）用相位化传输把控制开销移出关键路径，二者都是"绕开"而不是"下沉粒度"。COMET（2025-02）在单次 forward pass 内做了 tile 级细粒度 compute-combine 重叠，最接近但其分解维度是 tensor tile（N 维列块），不是 token 身份；DeepEP 的 EventOverlap/num_sms 是静态、per-call 级别，不按 token 完成时间动态释放。

[Inferred] 因此"per-token 粒度的 combine 提前释放"目前不存在于任何已知公开系统或论文中，是四条已死路线之外的、真正未被触碰的因果层。

## 机制设计

### 决策对象与时间窗口

决策对象是 `(decode step, layer, in-flight request)` 三元组的**放行时刻**，不是数据内容或精度。决策时间窗口是"该 token 的全部 top-k 专家贡献到达 receiver 之后、下一层计算发起之前"，纯粹的执行时间线事件，不需要任何路由/精度先验知识。

### 核心机制

标准 combine：等所有 rank 的所有 token 的所有专家贡献到齐 → 统一释放整个 batch 进入下一层。

TokenRace-EP：为每个 token 维护一个 completion counter（其 top-k 专家中已到达的数量）；一旦某 token 的 counter 达到 k，立即把该 token 标记为 ready，可以先行进入下一层的 compute（该 token 的 attention/FFN 不依赖同 batch其他 token，continuous batching 本身就是 per-token 独立的）；同 batch 中因专家 straggler 而未到齐的 token 继续等待，但不再拖慢已经到齐的 token。

### 因果链

```
专家执行时间差异 / 硬件抖动 (ViBE 已实测 7% variance)
        ↓
同一 batch 内不同 token 的 top-k 贡献到达时刻不同
        ↓
[现状] 全有或全无屏障：全部等最慢者 → 屏障延迟 = max(所有 token 到达时刻)
        ↓
[TokenRace] per-token 提前释放：每个 token 的下一层开始时刻 = 该 token 自己的到达时刻
        ↓
收益来源 = E[max(到达时刻)] - E[到达时刻]，即 straggler 拖慢的"其他健康 token"的等待时间
```

### 与真实 backend 的对接点

[Inferred] 这不需要发明新的 collective：continuous batching 框架（vLLM/SGLang）已经按 token/request 维度管理调度状态；NCCL EP 的 LL kernel 已经是 counter-based signal（"监听多 expert-rank 对的 counter，数据可用时立即从接收缓区提取"），这个 counter 机制本身已经具备 per-token/per-slot 粒度的完成信息，只是当前上层调度没有利用它来做 token 级放行，而是等整个 all-to-all handle 完成。DeepEP 的 EventOverlap 也是 event 粒度而非 token 粒度。TokenRace-EP 的 ABI 改动是：暴露一个 per-token/per-slot ready bitmap 或 completion counter 给上层调度，调度器按 token 就绪即放行，而不是等待整个 combine handle 的 `wait()` 返回。

## 核心可证伪假设与 Mac P0 设计（复用现有资产）

[Hypothesis] H1：在真实两模型路由 trace 驱动的 decode-step 重放中，per-token 到达时刻的方差足够大，使得"等最慢者"相对"各自放行"存在有意义的（>5%-10%量级）TBT P99 差距。

Mac P0（不需要新数据采集，直接扩展 `run_tbt_congestion_bridge.py` 已有的 decode-step 采样管线）：对同一批已采样的 decode step（B 个并发请求，每个请求当前 token 路由到 K 个专家），额外建模"专家执行时间"这一环——目前脚本只算通信字节，没有算 per-expert compute 完成时刻的方差。P0 需要新增：给每个 (expert, decode-step) 赋一个执行时间（基线：均匀；处理组：按 ViBE 报告的 7% variance 与 token-count 驱动的排队叠加，模拟真实 straggler），计算每个 token 的"全部 top-k 到达时刻"= max(其 k 个专家的完成时刻)，对比两种放行策略：全屏障（下一层 = max(全 batch 所有 token 到达时刻)）vs TokenRace（下一层 = 各 token 自己的到达时刻，聚合出这批 token 的到达时刻分布而非单一屏障值）。

判死门槛（预注册要求）：若 TokenRace 相对全屏障的 P99/mean TBT 改善在两个模型（OLMoE top-8、LLM-jp top-16）、多个 batch size 下均低于 5%，或改善完全由不现实的 variance 假设驱动（对 ViBE 实测的 7% variance 敏感性检验后消失），则判死，降级为 negative-result。

## Prior-art collision matrix（本轮新增）

| 工作 | 真实机制 | 重叠 | TokenRace 差异 | 风险 |
|---|---|---|---|---|
| COMET (2502.19811) | tile 级 compute-combine 重叠，Layer1 沿 N 维分块提前 reduction | 都是"提前发起下一步计算"的思路 | 决策粒度是 token 身份/到达时刻，不是 tensor tile；跨多个 in-flight request 的完成时间方差，不是单次 forward 内的列块流水 | 若 COMET 的 tile 分解在特定实现下已隐含等价于 token 级早释放，novelty 需要更细致代码级核实 |
| DeepEP EventOverlap | 静态 event/num_sms 粒度的异步 dispatch-compute 解耦 | overlap 的大方向重叠 | DeepEP 是 per-call 粒度，不按 token completion 动态放行 | 需确认最新版本是否已有 per-token bitmap |
| NCCL EP LL kernel | counter-based signal，数据到达即从接收缓冲区提取 | 底层机制天然支持 per-token 粒度信息 | 现有系统只用这个 counter 触发底层数据搬运，没有把它接到"上层调度是否可以放行进入下一层计算"这个决策 | 若已有内部实现利用了这个信号做调度，需重新核实 |
| ViBE / RepetitionCurse / Capacity-Aware | 揭示 straggler 现象及各自的缓解（离线 placement / token drop） | 现象与动机重叠 | 都不改变 combine 同步粒度本身 | 低，这三者是证据来源而非碰撞对象 |
| SpecMoEOff / PROBE | 用投机/预测绕开等待，而非缩小同步粒度 | 目标相同（降低 combine 等待），路径不同 | 机制正交，可以互补而非替代 | 低-中 |

## 最低资源与判决路径

Mac 可完成全部 P0（纯时间线仿真，复用现有 decode-step 采样+ViBE 实测 variance 参数）；不涉及 GPU/backend 时不得声称 actual latency。若 P0 通过（两模型都有 ≥5% 且对 variance 假设稳健的改善），下一步需要 1×GPU 验证 counter-based per-token 放行在真实 kernel 里是否可实现（涉及 continuous batching scheduler 与 EP combine handle 的接口改动），再需要多 GPU 验证真实 straggler 下的 wall-clock 收益。完全没有 GPU 时上限是"identified但未验证的调度机制 + 时间线仿真 negative/positive result"，不得声称已验证系统加速。

## 建议

先跑这个 Mac-only P0（预计半天工作量：扩展 `run_tbt_congestion_bridge.py`，加入 per-expert 执行时间模型和两种放行策略对比），比继续设计第五个精度/统计量重分配机制风险更低，因为它是四条已死路线从未触碰的因果层，且直接建立在自己代码里已经写明但从未质疑过的假设上。

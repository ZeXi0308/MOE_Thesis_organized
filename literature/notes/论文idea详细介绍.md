# MoE 相关论文 idea 详细介绍

本文按论文分别整理每篇工作的核心 idea：解决了什么问题、为什么这个问题重要、方法思路是什么、系统怎么设计、实验证明了什么，以及它对“rank/gate-aware approximate combine”毕设方向的启发。

---

## 1. HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference

### 解决的问题

HOBBIT 关注的是 **内存受限设备上的 MoE 推理**，尤其是边缘设备、消费级 GPU、CPU+GPU 混合环境。

MoE 模型虽然每个 token 只激活少数 expert，计算量比 dense 模型低，但它的总参数量很大。部署时会遇到一个矛盾：

- 把所有 expert 都放进 GPU 显存：速度快，但显存装不下。
- 把部分 expert offload 到 CPU/内存/磁盘：显存压力小，但每次 cache miss 都要加载 expert，延迟很高。

已有 expert offloading 方法通常有两个问题：

1. **加载成本高**：被路由到的 expert 如果不在 GPU cache 里，就要从慢速存储加载。
2. **精度与速度难兼顾**：如果简单把 expert 全部低精度化，可能伤害模型效果；如果全部保留高精度，加载又慢。

所以 HOBBIT 要解决的问题是：

> 在显存放不下全部 MoE experts 的情况下，如何用 mixed precision 和 expert cache/offloading，让推理既快又尽量不掉精度。

### 核心洞察

HOBBIT 的核心洞察是：

> 并不是每个被激活的 expert 都同等重要。对于 cache miss 的 expert，可以把“不那么关键”的 expert 用低精度版本替代，从而减少加载延迟，同时尽量保持准确率。

它把 expert 分成不同重要程度：

- 高重要性 expert：尽量用高精度版本。
- 低重要性 expert：如果高精度版本不在 cache，可以临时用低精度版本替代。

这和传统 offloading 的区别是：传统方法通常只考虑“expert 在不在 GPU 上”，HOBBIT 进一步考虑“如果不在，是否可以用低精度版本快速顶上”。

### 方法思路

HOBBIT 是一个 mixed precision expert offloading system，主要包含三个层级的设计。

#### 1. Token-level dynamic expert loading

在 token 级别，根据 router 结果判断当前 token 需要哪些 expert，以及这些 expert 的重要性。

如果某个 expert cache miss：

- 重要 expert：优先加载高精度版本。
- 次要 expert：允许加载低精度版本，减少传输量和加载时间。

这个策略的本质是：用“精度替代”缓解 cache miss。

#### 2. Layer-level adaptive expert prefetching

MoE 的 routing 在相邻层之间往往存在一定相关性。HOBBIT 利用这种相关性，在当前层执行时预测后续层可能用到哪些 expert，并提前 prefetch。

目标是把 expert 加载隐藏到计算过程中：

```text
+ 当前层计算
+ 同时预取下一层可能需要的 expert
+ 下一层真正执行时 cache miss 更少
```

#### 3. Sequence-level multidimensional expert caching

在序列级别，HOBBIT 维护 expert cache。它不仅考虑 expert 最近是否被用过，还考虑多个维度：

- expert 热度；
- expert 精度版本；
- 不同层的访问规律；
- 当前序列后续可能的访问需求。

这样 cache 淘汰不只是简单 LRU，而是服务于 MoE 推理里的 expert 访问模式。

### 实验想证明什么

HOBBIT 的实验主要想证明：

1. mixed precision expert offloading 可以明显降低 decoding latency。
2. 低精度替代 cache-miss expert 不会显著伤害准确率。
3. token-level loading、layer-level prefetch、sequence-level cache policy 三者组合，比单独优化更有效。

论文在 llama.cpp 上实现，并在边缘设备和代表性 MoE 模型上评估，报告了最高约 9.93x 的 decoding speedup。

### 局限性

HOBBIT 的适用场景主要是 **内存/显存不足导致 expert offloading** 的推理环境。它不直接解决数据中心多 GPU expert parallelism 里的 all-to-all 通信瓶颈。

它优化的是：

```text
expert 权重/参数如何加载
```

而不是：

```text
expert output 如何在 combine 阶段跨 GPU 回传
```

### 对你毕设 idea 的启发

HOBBIT 对你的方向很有参考价值，因为它证明了一个重要事实：

> MoE 里 expert 的重要性是不均匀的，可以用重要性来指导 mixed precision 决策。

但你可以和它明确区分：

- HOBBIT：对 expert 权重/offloading 做 mixed precision。
- 你的方向：对 combine 阶段的 expert output 传输做 mixed precision。

可以这样定位你的创新点：

> Existing mixed-precision MoE systems focus on expert weights and offloading. We instead apply importance-aware mixed precision to the combine communication path, where expert outputs are transmitted back for weighted aggregation.

---

## 2. AdapMoE: Adaptive Sensitivity-based Expert Gating and Management for Efficient MoE Inference

### 解决的问题

AdapMoE 关注的是 **MoE 推理中 expert 激活数量和 expert 管理开销过高** 的问题，尤其是在边缘设备或资源受限平台上。

标准 MoE 推理通常固定使用 top-k routing，例如每个 token 每层都激活 top-2 或 top-4 experts。问题是：

- 并不是每个 token 都需要同样多的 experts。
- 并不是每一层对 expert 数量的敏感性都一样。
- 固定 top-k 会造成不必要的 expert 加载、计算和管理开销。

所以 AdapMoE 要解决的问题是：

> 如何根据 token 和 layer 的敏感性动态调整激活 expert 数量，从而减少推理开销，同时不明显降低模型质量。

### 核心洞察

AdapMoE 的核心观察是：

> MoE 模型中不同层、不同 token 对 expert 数量的敏感性不同。有些 token/layer 只用较少 experts 也能保持输出质量，有些则必须保留更多 experts。

这说明固定 top-k 不是最优的。更合理的策略是：

```text
简单 token / 低敏感层：少激活 expert
困难 token / 高敏感层：多激活 expert
```

### 方法思路

AdapMoE 是 algorithm-system co-design，主要包含两个方面。

#### 1. Adaptive sensitivity-based expert gating

它不是所有 token 都固定 top-k，而是根据敏感性动态决定激活多少 experts。

可以理解为：

```text
router 给出 expert 排序和分数
+ sensitivity 判断当前 token/layer 能不能少用 expert
+ 最终选择 top-1 / top-2 / top-k
```

如果某些层对 expert 削减不敏感，那么这些层可以减少 expert activation。这样能直接减少：

- expert 计算；
- expert 加载；
- expert cache miss；
- token 到 expert 的调度压力。

#### 2. Expert prefetching and cache management

AdapMoE 同时结合 expert 管理策略，减少 on-demand loading latency。

因为一旦动态减少 expert 数量，系统就可以更准确地管理哪些 expert 需要保留、哪些可以替换，从而提升 cache 命中率。

### 实验想证明什么

AdapMoE 的实验主要证明：

1. 动态减少激活 experts 可以降低平均 expert 数量。
2. sensitivity-aware 策略比简单 top-k 剪枝更稳。
3. 配合 prefetch/cache management，可以带来实际推理加速。

论文报告平均激活 expert 数减少约 25%，并达到约 1.35x speedup，同时不造成明显准确率下降。

### 局限性

AdapMoE 改的是 **routing/gating 决策本身**：它会改变一个 token 最终去哪些 experts，或者激活多少 experts。

这会带来一个风险：

```text
改变 routing = 改变模型实际计算路径
```

因此它需要仔细控制 accuracy drop。

另外，AdapMoE 主要关注 expert activation/loading 开销，不是专门处理多 GPU all-to-all combine 通信。

### 对你毕设 idea 的启发

AdapMoE 对你最有启发的是“layer sensitivity”：

> 不是每一层都同样能承受近似。近似策略应该按层区分。

你的方向可以借鉴它的思想，但做出不同选择：

- AdapMoE：减少激活 expert 数量，改变 routing/computation。
- 你的方向：不改变 routing，只改变 selected expert output 的传输精度。

这个差异很重要。你可以强调：

> We preserve the original top-k routing and expert computation, and only approximate the transport precision in the combine phase. This avoids changing the model's expert selection path directly.

换句话说，你的方法比 AdapMoE 更“保守”，因为它不直接删除 expert 计算路径，而是只在通信表示上做近似。

---

## 3. Aurora: Optimizing Mixture-of-Experts Inference Time Combining Model Deployment and Communication Scheduling

### 解决的问题

Aurora 关注的是 **数据中心多 GPU 环境下 MoE 推理的 end-to-end latency 优化**。

MoE 推理的核心瓶颈有几个：

1. **All-to-all 通信开销高**  
   token 被路由到不同 GPU 上的 experts，需要跨 GPU 发送 token hidden states，expert 计算后还要返回。

2. **同步通信导致 GPU 利用率低**  
   all-to-all 往往是同步阶段，快的 GPU 等慢的 GPU，造成空转。

3. **模型部署和通信调度耦合**  
   expert 放在哪些 GPU 上，会影响 token 传输量；通信顺序怎么排，也会影响总通信时间。

4. **异构 GPU 环境更复杂**  
   不同 GPU 或链路性能不同，简单平均放置 expert 可能不是最优。

Aurora 要解决的问题是：

> 如何联合优化 MoE inference 的 model deployment 和 all-to-all communication scheduling，使推理时间最小。

### 核心洞察

Aurora 的核心洞察是：

> MoE 推理时间不仅取决于 expert 放置，还取决于 all-to-all 中 token 传输的调度顺序；这两者应该联合优化。

很多系统只做 expert placement，比如把热门 expert 放到更好的 GPU，或者尽量减少跨节点通信。但 Aurora 进一步指出：

```text
即使 expert 放置固定，不同 token 传输顺序也会造成不同通信完成时间。
```

所以它把问题拆成两个互相关联的部分：

- model deployment：expert/model fragments 放在哪些 GPU 上；
- communication scheduling：all-to-all 里的 token 发送顺序如何安排。

### 方法思路

Aurora 针对几种 GPU cluster setting 做理论分析：

- exclusive vs. colocated models；
- homogeneous vs. heterogeneous GPUs。

也就是：

```text
单个模型独占 GPU，还是多个模型 colocate；
GPU/链路是同构，还是异构。
```

#### 1. Communication scheduling

Aurora 会安排 all-to-all 中 token transmission 的顺序，让通信完成时间更短。

直观例子：

```text
如果多个 GPU 同时向同一个瓶颈链路发送数据，就会排队。
更好的调度顺序可以减少链路冲突和等待。
```

这不是减少总 bytes，而是减少 communication makespan。

#### 2. Model deployment / expert placement

Aurora 同时优化 expert 或模型片段放在哪些 GPU 上，让整体推理时间更低。

它考虑的不是单个 expert 的局部最优，而是部署和通信调度一起看。

#### 3. Colocation 提升 GPU 利用率

Aurora 还讨论了多个模型 colocate 在 GPU 上的场景。由于 MoE all-to-all 存在同步等待，不同模型的 experts colocate 可能填补 GPU 空闲，提高利用率。

### 实验想证明什么

Aurora 的实验主要证明：

1. 联合优化 deployment + communication scheduling 比只优化其中一个更好。
2. 在同构和异构环境下都能降低 MoE inference time。
3. colocation 可以提高 GPU 利用率。

论文报告在同构集群中最高约 2.38x speedup，在异构环境中最高约 3.54x speedup，并提升 GPU utilization。

### 局限性

Aurora 主要优化的是：

```text
数据怎么调度、expert 怎么部署
```

它默认 token/expert output 的数据表示本身不变，也就是不改变通信 payload 的数值精度。

因此，Aurora 减少的是通信等待和调度浪费，不一定减少通信字节本身。

### 对你毕设 idea 的启发

Aurora 是你方向的强相关 baseline，因为它代表了现有 MoE inference 系统优化的主线：

```text
expert placement + all-to-all communication scheduling
```

你的方向可以和 Aurora 形成互补：

- Aurora：在原始 payload 下优化发送顺序和部署。
- 你的方向：减少/压缩 combine payload 本身。

可以这样定位：

> Existing works such as Aurora optimize where experts are placed and how all-to-all transfers are scheduled. Our work is orthogonal: we reduce the amount or precision of data that must be transported in the combine path.

这也说明你的实验里最好加入类似维度：

```text
只做 placement/scheduling
只做 rank-aware combine compression
两者结合
```

如果两者结合还能进一步提升，就说明你的方法不是替代 Aurora，而是补充 Aurora。

---

## 4. Lina: Accelerating Distributed MoE Training and Inference with Lina

### 解决的问题

Lina 是更早的一篇分布式 MoE 系统论文，关注 **MoE 训练和推理中的 all-to-all 通信瓶颈**。

在分布式 MoE 中，一个 MoE layer 通常包含：

```text
dispatch all-to-all:
  把 token hidden states 发送到对应 expert 所在设备

expert computation:
  每个设备计算本地 experts

combine all-to-all:
  把 expert outputs 发回原 token 所在设备
```

训练时还会有 data parallel 的 allreduce。问题是：

- all-to-all 和 allreduce 会竞争网络资源；
- all-to-all 插在模型计算中间，难以隐藏；
- all-to-all 一慢，整个 MoE layer 都被阻塞。

Lina 要解决的问题是：

> 如何在不改变 MoE 模型语义的情况下，优化 distributed MoE 里的 all-to-all 通信，使训练和推理更快。

### 核心洞察

Lina 的关键观察是：

> MoE 的 all-to-all 是关键路径上的通信，比普通 allreduce 更难隐藏；当 all-to-all 和 allreduce 竞争网络时，应该优先保证 all-to-all。

换句话说：

```text
allreduce 通常可以被切分、延后、流水化；
all-to-all 卡在 MoE layer 中间，更容易直接决定 step time。
```

因此 Lina 的策略是：尽可能让 all-to-all 优先获得网络资源。

### 方法思路

Lina 主要通过 tensor partitioning 和通信调度来优化。

#### 1. Prioritize all-to-all over allreduce

训练中，allreduce 可能和 all-to-all 同时发生。如果它们抢同一段网络，all-to-all 被拖慢，MoE layer 的前向/反向都会变慢。

Lina 会在可行时让 all-to-all 优先执行，避免被 allreduce 阻塞。

#### 2. Tensor partitioning

Lina 把大张量通信切成多个小块，使通信可以更细粒度地调度。

这样做的好处是：

```text
不是一个巨大的 allreduce 一口气占住网络，
而是把它拆开，让 all-to-all 有机会插队。
```

#### 3. Pipeline expert computation and communication

通过切分 token/expert 相关张量，Lina 可以让部分通信和部分 expert computation 形成流水。

目标是减少等待：

```text
一部分 token 到达 expert 后先算；
另一部分 token 还在通信中；
计算和通信重叠。
```

### 实验想证明什么

Lina 的实验主要证明：

1. all-to-all 是 distributed MoE 训练/推理里的关键瓶颈。
2. 优先 all-to-all、切分 allreduce tensor、流水化计算通信可以降低 step time。
3. 这些优化不改变模型语义，因此可以作为系统层加速。

### 局限性

Lina 的重点是 **通信调度和 overlap**，不是通信内容压缩。

它不会改变：

- token dispatch 的数据精度；
- expert output combine 的数据精度；
- expert 的重要性权重；
- routing 的 top-k 结构。

所以 Lina 解决的是：

```text
同样的数据，如何更好地安排通信
```

不是：

```text
是否每份数据都值得用同样精度传输
```

### 对你毕设 idea 的启发

Lina 可以作为你论文 related work 里的“通信调度类系统”代表。

你的方向可以和 Lina 明确区分：

- Lina：不改数据内容，只调度 all-to-all 和 allreduce。
- 你的方向：不主要调度通信顺序，而是按 expert contribution/rank 改变 combine payload precision。

可以这样写：

> Lina demonstrates that all-to-all is a first-order bottleneck in distributed MoE. However, it treats all token/expert payloads uniformly. Our work explores whether the combine payloads can be represented with non-uniform precision according to their contribution importance.

---

## 5. 你的 Idea A：Rank-aware Approximate Combine for Communication-Efficient MoE Inference

### 一句话概括

你的方案 A 可以概括为：

> 在 MoE inference 的 combine 阶段，不再把所有 selected expert output 都用同样精度完整传回，而是根据 expert 在 top-k routing 里的 rank、所在 layer 的敏感性、以及通信链路 tier，静态查表决定传输精度，从而在精度可控的前提下降低昂贵链路通信量。

更短一点可以写成英文题目：

> Rank-aware Approximate Combine for Communication-Efficient MoE Inference

或者更系统味一点：

> Topology-aware Rank-LUT Approximate Combine for MoE Serving

### 解决的问题

MoE 模型的一个核心优势是稀疏激活：

```text
每个 token 只激活 top-k 个 experts
```

但在分布式 expert parallelism 中，这会引入两次 all-to-all：

```text
dispatch:
  token hidden states -> expert 所在 GPU

combine:
  expert outputs -> token 原始所在 GPU，然后做 weighted sum
```

现有系统大多把 combine 阶段当作固定成本：

```text
每个被选中的 expert output 都完整、同精度传回
```

这背后有一个隐含假设：

> top-k 里每个 expert output 对最终 combine 结果都同等重要，因此都值得用同样精度传输。

你的 idea 要挑战这个假设。

在 MoE combine 中，最终输出是：

```text
y_t = sum_{e in S(t)} g_{t,e} * o_{t,e}
```

其中：

- `t` 是 token；
- `S(t)` 是 token 被路由到的 top-k experts；
- `g_{t,e}` 是 gate 权重；
- `o_{t,e}` 是 expert output。

直觉上，top-1 expert 通常比 top-2/top-3/top-4 expert 更重要。即使 top-k 都参与计算，它们对最终 hidden state 的贡献也往往不均匀。

所以你要解决的问题是：

> combine 阶段是否可以利用 expert contribution 的不均匀性，对低重要性 expert output 使用低精度传输，甚至在极端情况下丢弃，从而减少昂贵链路通信量？

### 为什么这个问题重要

MoE 推理的瓶颈不只是计算，还包括跨 GPU、跨节点、跨 rack 的通信。

特别是在大规模 serving 中：

- expert 可能分布在不同 GPU 或不同节点；
- top-k 越大，dispatch/combine 通信越重；
- 跨 rack / 跨 pod 链路带宽更昂贵、更容易成为瓶颈；
- all-to-all 是同步通信，慢链路会拖累整个 MoE layer。

已有优化大多在这些方向：

```text
expert placement
expert replication
communication scheduling
all-to-all kernel 优化
expert offloading / caching
```

但这些方法通常默认 payload 本身必须完整传输。你的切入点是：

```text
不是只问“怎么传得更快”，
而是问“每份 expert output 是否都值得用同样精度传”。
```

这是一个更细粒度的 combine payload 近似问题。

### 核心洞察

你的方案 A 的核心洞察可以拆成三层。

#### 1. Combine contribution 可能是长尾的

MoE combine 是 weighted sum：

```text
y_t = g_1 * o_1 + g_2 * o_2 + ... + g_k * o_k
```

如果某些 expert 的 gate rank 较低，或者 gate 权重较小，那么它们对 `y_t` 的影响可能较小。

可以用一个 contribution proxy 来衡量：

```text
contribution_{t,e} = g_{t,e} * ||o_{t,e}||
```

如果这个分布呈长尾，就说明：

```text
少数 expert output 贡献大；
多数低 rank expert output 贡献小。
```

这为近似传输提供了空间。

#### 2. 近似误差的真正风险不是单层线性误差，而是 routing drift

如果某个 expert output 被量化或丢弃，当前层 hidden state 会产生误差：

```text
Delta y_t
```

单层看，这个误差可能不大，因为 combine 是线性加权和。

但 MoE 模型的下一层 router 会基于新的 hidden state 做 routing。如果当前层误差改变了下游 top-k experts，就会导致：

```text
hidden state error -> downstream gate changes -> routing path changes -> error nonlinear amplification
```

这就是 routing drift。

所以你的方法不能只看单层 MSE，也必须看：

- 下游 routing 改变比例；
- end-to-end perplexity / task accuracy；
- 哪些层对近似更敏感。

这一点可以成为论文的 characterization contribution。

#### 3. Rank 比 gate value 更适合 runtime 系统实现

最开始可以想到用 gate value 做 bucket：

```text
if gate > 0.5: BF16
elif gate > 0.1: FP8
else: INT4/drop
```

但 gate value 是连续变量，且跨层、跨 token、跨模型分布不稳定。runtime 上还可能导致分桶碎片化。

你的改进是：不看 gate 数值，而看 expert 在 top-k 里的 rank。

```text
rank 1 -> 高精度
rank 2 -> 中等精度
rank 3/4 -> 低精度或 drop
```

runtime 只需要查：

```text
precision = LUT[layer_id, link_tier, rank]
```

这有几个好处：

- rank 是 router top-k 输出天然已有的信息；
- rank 数量固定，最多就是 `k` 类；
- 同 rank 的 token 可以组成规整张量一起传；
- 不需要在线优化、不需要复杂排序、不需要 per-token epsilon budget；
- 策略可以离线 profile 后静态写入 LUT。

需要强调的是：

> 决策精度时看 rank，但 combine 计算本身仍然使用原始 gate weight。

也就是说：

```text
rank 决定 o_i 用什么精度传输；
gate weight g_i 仍然参与最终 weighted sum。
```

### 方法思路

你的系统可以分成 offline profiling 和 runtime execution 两部分。

#### 1. Offline profiling：找出可近似空间

离线阶段需要做几件事。

第一，验证 contribution 长尾：

```text
收集每层每个 token 的 top-k expert output
计算 g_{t,e} * ||o_{t,e}||
画 rank-wise / layer-wise 分布
```

目标是证明：

```text
rank 越低，平均 contribution 越低；
不同层的 contribution 分布不同。
```

第二，测 layer sensitivity：

```text
对每一层单独施加近似
测 end-to-end perplexity / accuracy drop
测 downstream routing drift
```

目标是区分：

- 低敏感层：可以用低精度；
- 高敏感层：必须保守；
- routing drift 高的层：不要轻易 drop。

第三，搜索 LUT：

```text
LUT[layer, link_tier, rank] -> precision
```

precision 可以是：

```text
BF16 / FP8 / INT4 / drop
```

优化目标是：

```text
在 accuracy drop <= epsilon 的约束下，最小化昂贵链路 bytes
```

形式化写法：

```text
minimize    sum_{layer, tier, rank} bytes(layer, tier, rank, precision)
subject to  end-to-end accuracy drop <= epsilon
```

这个优化变量规模很小，因为只和：

```text
层数 * 链路 tier 数 * top-k rank 数
```

有关。可以先用 grid search / greedy search，不需要一开始就上复杂 MILP。

#### 2. Runtime execution：静态查表 + 规整传输

runtime 不做复杂决策，只执行查表：

```text
for each selected expert output:
    r = rank in top-k
    tier = link tier between expert GPU and token owner GPU
    p = LUT[layer, tier, r]
    transmit output with precision p
```

同 rank、同 precision、同 destination 的 token 可以 pack 成规整 tensor：

```text
rank-1 BF16 tensor
rank-2 FP8 tensor
rank-3 INT4 tensor
```

这样避免 ragged communication 和 kernel 发散。

#### 3. Drop 的处理：可选 gate renormalization

如果最激进 bucket 使用 drop，那么 combine 变成：

```text
y_t approx = sum_{e in kept} g_{t,e} * o_{t,e}
```

这会导致总权重小于 1，产生系统性偏差。

可以做 gate renormalization：

```text
y_t approx =
  1 / sum_{e in kept} g_{t,e}
  * sum_{e in kept} g_{t,e} * o_{t,e}
```

这样可以减少因为 drop 带来的幅值偏差。

不过 drop 是高风险策略，应该作为最激进配置。主线最好先强调：

```text
量化优先，丢弃次之。
```

### 你的方法和现有论文的区别

#### 和 HOBBIT 的区别

HOBBIT 做的是：

```text
expert 权重/offloading 的 mixed precision
```

你的方法做的是：

```text
expert output combine 通信的 mixed precision
```

HOBBIT 解决“expert 参数加载慢”，你解决“expert output 回传通信浪费”。

#### 和 AdapMoE 的区别

AdapMoE 会动态减少 activated experts，改变 routing/computation path。

你的方法不改变 top-k routing：

```text
原来选哪些 expert，仍然选哪些 expert；
原来 expert 如何计算，仍然如何计算；
只改变 combine 回传时的表示精度。
```

这让你的方法更容易作为 serving 系统上的 runtime optimization。

#### 和 Aurora 的区别

Aurora 优化：

```text
expert 放在哪
all-to-all 怎么排程
```

你的方法优化：

```text
combine payload 用什么精度传
```

两者是正交的。Aurora 可以作为 placement/scheduling baseline，你的方法可以叠加在 Aurora 上。

#### 和 Lina 的区别

Lina 优化 all-to-all 与 allreduce 的调度和 overlap。

你的方法关注 all-to-all payload 的重要性差异。

一句话：

```text
Lina/Aurora: same payload, better scheduling.
你的方法: same routing, cheaper payload.
```

### 实验设计

你的实验最好分成三组。

#### 1. Characterization experiments

目标是证明问题存在。

要画：

- rank-wise contribution 分布；
- layer-wise sensitivity heatmap；
- routing drift heatmap；
- gate-value vs rank 的相关性；
- 不同 top-k 下 contribution 长尾是否更明显。

这里最关键的是：

```text
rank 是否能作为 contribution importance 的稳定 proxy。
```

如果 rank-1/2/3/4 的 contribution 分布明显分离，你的方案会很强。

#### 2. Accuracy-traffic Pareto experiments

比较这些策略：

```text
Full BF16 combine
Uniform FP8 combine
Uniform INT4/FP4 combine
Gate-value bucket
Rank-LUT
Rank-LUT + layer sensitivity
Rank-LUT + layer sensitivity + link tier
Token-level oracle
```

指标：

- end-to-end perplexity；
- downstream task accuracy；
- accuracy drop；
- expensive-tier bytes；
- total all-to-all bytes；
- routing drift ratio。

最关键的图：

```text
x-axis: expensive-tier bytes reduction
y-axis: perplexity / accuracy drop
```

如果 Rank-LUT 比 uniform FP8/FP4 更接近 oracle，就说明你的 idea 成立。

#### 3. System / topology experiments

模拟或真实测：

- 同节点；
- 跨节点；
- 跨 rack；
- 跨 pod。

因为你的方法不一定要在所有链路启用。更合理的是：

```text
便宜链路：保守或全精度
昂贵链路：rank-aware mixed precision
```

指标：

- expensive link bytes；
- bandwidth contention；
- TBT / P99 TBT；
- throughput；
- 可选 J/token。

### 可能的论文贡献点

可以写成三条贡献。

#### Contribution 1: Characterization

> We characterize the combine phase of MoE inference and show that expert contributions are skewed across top-k ranks, while approximation-induced quality degradation is strongly related to downstream routing drift.

这条是“发现”。

#### Contribution 2: Rank-LUT approximate combine

> We propose a rank-aware static LUT that assigns transport precision based on layer sensitivity, link tier, and top-k rank, enabling low-overhead and deployment-friendly approximate combine.

这条是“方法”。

#### Contribution 3: Evaluation

> We evaluate the accuracy-traffic Pareto frontier and show that rank-aware combine reduces expensive-tier communication compared with uniform low-precision combine under the same accuracy budget.

这条是“实验结果”。

### 最大风险

#### 风险 1：top-k 太小，rank 粒度不够

如果模型是 top-2，那么 rank 只有两档：

```text
rank 1
rank 2
```

策略空间很小。你的收益可能有限。

缓解方式：

- 优先选 top-k 较大的模型，例如 top-4/top-8；
- 对 top-2 模型做 gate-value bucket 作为补充；
- 强调 rank-LUT 是部署友好的 conservative design。

#### 风险 2：rank 不能稳定代表重要性

有些 token 的 rank-2 gate 可能很高，有些 rank-2 gate 很低。

如果 rank 和 contribution 的相关性弱，rank-LUT 可能不如 gate-value bucket。

缓解方式：

- 做 rank vs gate-value ablation；
- 设计 rank + coarse gate bucket 的增强版；
- 或者保留 gate-value bucket 作为 oracle/upper bound。

#### 风险 3：省 bytes 不等于省 latency

如果 combine 通信已经被 overlap 掉，减少 bytes 未必降低 TBT。

缓解方式：

- 把主指标设为 expensive-tier bytes 和 bandwidth contention；
- 只在通信瓶颈或昂贵链路场景报告 latency；
- 可补充 J/token 或 throughput-per-watt。

#### 风险 4：drop 导致 routing drift

丢弃低 rank expert 可能会在某些层引发下游 routing 改变。

缓解方式：

- 主线先做 quantization，不主打 drop；
- drop 只在低敏感层和昂贵链路启用；
- 加 gate renormalization；
- 用 routing drift 指标筛掉高风险层。

### 最终定位

你的 idea 最稳的定位不是“又一个 MoE 通信优化”，而是：

> 一个 combine-phase characterization + deployment-friendly approximate transport policy。

也就是说，论文不要只卖“我省了多少 bytes”，还要卖这个洞察：

```text
MoE combine 阶段的 selected expert outputs 并非同等重要；
这种不均匀性可以被 top-k rank 近似捕捉；
rank 这个信号比 gate value 更适合做 runtime 规整传输策略。
```

如果实验能证明 rank-LUT 在精度-通信 Pareto 上优于 uniform low precision，同时接近 gate-value oracle，那么这个 idea 是可以立足的。

---

## 总体对比：这几篇论文和你的 idea 的关系

| 论文 | 核心问题 | 主要手段 | 是否改变 routing | 是否压缩 combine output | 和你的关系 |
|---|---|---|---|---|---|
| HOBBIT | 显存不足下 expert offloading 慢 | mixed-precision expert loading/cache/prefetch | 不主要改变 routing | 否 | 证明“expert 重要性可指导 mixed precision” |
| AdapMoE | 固定 top-k 激活开销高 | sensitivity-based adaptive gating | 是 | 否 | 证明“layer/token sensitivity 很关键” |
| Aurora | MoE inference all-to-all 和部署导致 latency 高 | model deployment + communication scheduling | 否 | 否 | 是 placement/scheduling 类强 baseline |
| Lina | 分布式 MoE all-to-all 是训练/推理瓶颈 | tensor partitioning + communication priority/overlap | 否 | 否 | 是 all-to-all 调度类经典 related work |
| 你的 idea | combine 阶段所有 expert output 同精度回传造成通信浪费 | rank/gate-aware transport precision LUT | 否 | 是 | 切入点是 importance-aware approximate combine |

这四篇论文共同说明：

1. MoE 的瓶颈确实在 expert 相关的数据移动和管理上。
2. 现有工作主要从 expert loading、routing 数量、placement、scheduling 角度优化。
3. 还没有把重点放在“combine 阶段 selected expert output 是否必须同精度传回”这个点上。

因此，你的论文可以这样立意：

> Existing MoE systems optimize which experts are loaded, where experts are placed, or how all-to-all communication is scheduled. However, they largely treat the combine payloads from selected experts as uniformly important. We observe that expert contributions in the combine phase are highly skewed, and propose a rank-aware static LUT to assign different transport precisions to expert outputs based on layer sensitivity, link tier, and top-k rank.

---

## 建议你后续读论文时重点看什么

### 读 HOBBIT

重点看：

- 它如何定义 less critical expert；
- mixed precision expert 替代如何控制 accuracy；
- cache miss latency 的建模；
- token/layer/sequence 三层系统设计。

对你最有用的是：如何把“重要性”转成系统决策。

### 读 AdapMoE

重点看：

- layer sensitivity 怎么测；
- 不同 token/layer 的 expert 数量差异；
- accuracy drop 如何评估；
- adaptive gating 和普通 top-k pruning 的差异。

对你最有用的是：如何证明“不是每层都适合近似”。

### 读 Aurora

重点看：

- MoE inference time 如何建模；
- all-to-all communication scheduling 怎么形式化；
- placement 和 communication scheduling 如何联合优化；
- 它的 baselines 和 metrics。

对你最有用的是：如何写系统优化问题和 baseline。

### 读 Lina

重点看：

- 它如何证明 all-to-all 是 bottleneck；
- dispatch/combine all-to-all 在训练和推理中分别为什么难；
- tensor partitioning 如何让通信更细粒度；
- communication overlap 的实验设计。

对你最有用的是：如何论证 MoE all-to-all 是一等公民问题，而不是小优化。

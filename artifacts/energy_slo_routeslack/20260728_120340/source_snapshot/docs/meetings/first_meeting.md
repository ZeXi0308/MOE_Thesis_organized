一、方案 A：partial combine / rank-level 近似的建议
1. 不要一开始建模完整网络链路，先简化成 receiver 端口拥塞

你原来想做 link-level / topology-level 的拥塞建模，比如哪条 spine-leaf 链路堵、哪个 rack/pod 链路贵。但 Qiaolun 觉得第一步太复杂。

他的建议是先参考类似 INFOCOM 那类工作的简化假设：

假设网络内部资源近似无限，不建模具体 switch / link 拥塞，只考虑 receiver 端口是否成为热点。

也就是说，先把问题简化成：

某个 GPU/server 作为 receiver，接收太多 expert output
→ receiver 端口成为瓶颈
→ 针对这个 receiver 做流量削减/量化/drop

这样比直接建模“哪条链路拥塞”简单很多，也更容易做第一版实验。

2. 你要先证明：drop / quantize 到底带来多大精度损失

你原来重点关注 routing drift：近似之后 gate 选择的 expert 变了，从而导致精度下降。

Qiaolun 认为这个可以做，但更基础、更重要的是先量化：

不做近似 vs 做近似
到底精度差多少？

也就是说，先回答：

如果我丢掉一部分 combine 信息，或者降低精度，它相对原模型的端到端 accuracy / perplexity / benchmark score 损失是多少？

然后再进一步分析：

这个损失里面，有多少来自 routing drift，有多少来自纯数值误差？

所以优先级应该是：

先测整体精度损失；
再拆解 routing drift 是否是主要原因。
3. top-k 内部 contribution 长尾，需要实验先画出来

你现在的核心假设是：

top-k experts 对一个 token 的 combine contribution 存在明显长尾

也就是 top-1 / top-2 很重要，尾部 expert 贡献很小，可以量化或 drop。

Qiaolun 提醒你：现有 evidence 更多是 expert-level 流量不均衡，不一定能直接证明 token 内部 top-k contribution 长尾。

他建议你可以看 AICB 这类工具。AICB 可以生成/模拟 MoE 通信流量，他之前也让学生测过不同 expert 之间的流量差异，发现加 bias 后 expert 流量会有差异。这个可以作为你前期实验或流量特征分析的参考。

4. 查表 LUT 的 key 要重新想清楚

你说 runtime 查表决定某个 layer / topology / gate bucket 对应什么精度。

Qiaolun 重点追问：

这个表的 index / key 到底是什么？

他觉得你目前的 key 设计还不够清楚。可能涉及：

layer
rank group
gate bucket
receiver group
topology position

但如果 key 太多，表会变得很难构建，也很难解释。

他的方向是：先把 key 简化，不要一开始做太复杂的 link-level / topology-level 表。

5. 优化公式现在还不像 ILP / MILP，要重写变量表示

你原来公式里把精度选择写成一个变量，比如从 BF16 / FP8 / INT4 / drop 里面选一个。

Qiaolun 建议，如果你想写成 ILP/MILP，更标准的表示应该是：

x_BF16 ∈ {0,1}
x_FP8  ∈ {0,1}
x_INT4 ∈ {0,1}
x_DROP ∈ {0,1}

x_BF16 + x_FP8 + x_INT4 + x_DROP = 1

也就是每种精度一个 0/1 变量，四个变量加起来等于 1，表示只能选一个。

这比直接写一个抽象的 b 变量更适合整数规划表达。

6. accuracy constraint 不好直接查表，需要重新设计

你原来想写：

Δaccuracy ≤ ε

但 Qiaolun 认为这个约束怎么得到不清楚，因为端到端 accuracy 不是一个很容易直接由 LUT 查出来的东西。

他建议你做离线 profile，比如：

某一层 drop 1%、2%、3%
在多个场景下最差 accuracy drop 是多少

然后用 worst-case 或 average-case 的经验表来约束：

每层允许最多丢多少
每层近似后最差精度损失是多少
总损失是否小于阈值

核心意思是：accuracy 约束要和你的决策变量建立可操作的对应关系。

7. TBT / latency 要和 congestion 建立关系

他还提醒你，TBT 不应该只是孤立写一个约束，而应该和拥塞程度直接关联。

比如：

receiver 端口拥塞越严重
→ 通信排队越严重
→ TBT 越大

所以如果你要优化 TBT，就要说明：

你的流量削减/量化/drop
如何减少 receiver congestion
又如何改善 TBT

否则目标函数和性能收益之间会显得断开。

二、方案 B：MoE energy-aware placement / replication 的建议
1. 做能耗优化，必须先有可信的 GPU 功耗模型

Qiaolun 重点问了你：

GPU 功率会随计算量、频率、负载变化，有没有公开数据集或论文可以建模？

他的建议是：如果你要做能耗方向，PPT 里需要加 reference，说明你使用的 GPU power model / dataset / paper 来源。

因为能耗优化论文很看重：

设备级别功耗模型是否可信

如果 GPU 功耗模型不可信，后面的系统级能耗优化也会被质疑。

2. objective 里的每一项要解释清楚

你写了静态 GPU 功耗、动态计算功耗、通信功耗等项。

Qiaolun 建议你在公式旁边加解释，比如箭头说明：

static GPU power
dynamic expert computation energy
all-to-all communication energy

否则别人看公式时不知道每一项具体代表什么。

3. 通信能耗系数 C_communication 要建模得更实际

你原来通信能耗项大概是：

data amount × communication energy coefficient

Qiaolun 认为这里还可以更细化。

比如在 spine-leaf 数据中心网络里，两个 GPU 之间通信经过哪些 switch，可以通过 ECMP 或拓扑路径确定。然后路径上的设备数量、链路类型、switch 能耗都可以影响通信能耗。

所以他建议你后面要想清楚：

C_communication 怎么从网络路径、设备、链路距离中得到？

这个会让方案 B 更有说服力。


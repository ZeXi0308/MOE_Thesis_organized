# TokenRace-EP 严格性审查 + 上限评估 + 后续方向（2026-07-19）

> 审查对象：`run_tokenrace_ep_p0.py`（P0）、`run_tokenrace_ep_topk_sweep.py`（P0-B）及其结论文档。
> 方法：代码级重读 + 两个独立外部调研子任务（kernel launch/re-batching 开销、DeepEP/NCCL EP/MORI-EP 代码级完成粒度）+ 两篇最相关 prior-art 全文核实。
> 标签规范同前：`[Observed]` / `[Inferred]` / `[Hypothesis]`。

## 一、P0/P0-B 实验设计本身：哪些地方站得住，哪些是薄弱环节

### 站得住的部分
1. **数据是真实的，不是合成的**：`route_index` 直接读取 OLMoE/LLM-jp 的 16 层完整真实路由（sample_id/layer/token_position/rank/expert_id），P0-B 的 top-K 截断是"用 `rank<=K` 截断真实 gate_weight 排序"，不是重新合成路由——这一点已经用代码验证过（`rank` 确认按 `gate_weight` 降序），是可信的真实数据消融，不是臆造的。
2. **预注册门槛 + 多 batch size + 多噪声场景 + 6 seed 稳健性检验**：这套流程本身符合本项目一贯的 sealed/预注册纪律，没有为了让结果好看而事后调整门槛，"卡边界"这个诚实的判决（CONDITIONAL 而非硬 PASS）本身就是严谨性的体现。
3. **zero-variance 对照组的使用是对的**：用它来排除"效应完全是硬件抖动假设堆出来的人造产物"这个混淆，OLMoE 的 zero-variance 改善（2%-10%）证明部分收益来自真实负载不均，不是纯抖动假设的产物；LLM-jp 的 zero-variance 改善接近 0（0.33%-1.14%）恰好和"该模型路由本身更均衡"这个后续 K 值解释一致，两个证据互相印证。
4. **P0-B 的剂量-响应曲线非常干净**：两个模型独立测得的 K=2/4/8（/16）曲线几乎重合，这是一个统计意义上很有说服力的信号，不太可能是巧合。

### 薄弱环节（按严重程度排序）

**1. [最严重] 系统实现代价完全没有建模——这是当前最大的漏洞。**
外部调研证实：
- GPU kernel launch overhead 实测量级为 **5-10 μs/kernel**（decode 阶段占总时间 25-50%）。
- `run_tokenrace_ep_p0.py` 里预注册的收益量级本身就是 `BASE_LAUNCH_US=6us + PER_TOKEN_US*cnt` 这个级别的差值——也就是说，**TokenRace-EP 试图省下的时间，和它必须额外付出的"逐 token 提前释放"所需的重新分批（re-batching）/ kernel relaunch 开销，处于同一个数量级**。
- 更关键的系统约束：vLLM/SGLang/TensorRT-LLM 三大主流 serving 框架**目前都不支持 layer-wise 异步推进**，都是迭代级/请求级同步调度；vLLM 的 `FULL` CUDA Graph 模式要求**固定 batch size**，任何"把 batch 内已完成的 token 提前抽出送入下一层"都会改变 shape，直接与 CUDA Graph 优化冲突——这意味着 TokenRace-EP 若要落地，很可能需要**放弃 CUDA Graph**，而放弃 CUDA Graph 本身在 decode 阶段的额外代价（decode 阶段本就 launch-bound）可能比它想省下的 straggler 等待时间还大。
- 这不是"暂缓的实现细节"，而是**P0 仿真模型完全没有包含的一个负成本项**。当前 P0/P0-B 报告的所有"收益"，本质上都只是"如果重新分批和释放判定是零成本，能省多少时间"——这是一个**上界（upper bound）**，不是可实现收益的估计。这一点必须在任何后续文档里明确标注为 `[Hypothesis]`，且是当前风险最高的未验证假设，而不是像现在文档里那样只写"下一步建议做参数敏感性扫描"。

**2. [重要] 找到了一个直接相关但方向不同的已发表工作：AMoE/AEP (arXiv:2505.08944)。**
这篇论文已经实现了 per-layer token queue + adaptive re-batching，用来消除 MoE all-to-all 的 barrier 等待，单节点吞吐提升 2.7×。它和 TokenRace-EP 的差异在于：AEP 论文自己承认可行的前提是"专家计算由 GEMM 主导，且 CPU 能在第一个 GEMM 执行期间异步启动后续 kernel"——即收益能覆盖 launch 开销**依赖于专家计算本身足够大**。TokenRace-EP 目前的仿真模型（`ATTN_SHARED_US=4us` 共享，专家计算 `6+0.35*cnt us`）和这个前提是否吻合，从未被检验过。**这篇论文必须作为最强 prior art 被正面回应**，而不是像之前四条路线一样在文献碰撞阶段就被简单归为"重叠过高，淘汰"——因为 AEP 解决的是 dispatch/all-to-all barrier，TokenRace-EP 解决的是 combine 之后的 layer-advance barrier，两者机制位置不同，仍可能有差异化空间，但目前的文档里完全没有提及这篇论文，这是一个必须补上的文献缺口。

**3. [中等] Prior art 碰撞：DynaMoE (2603.01697)、Capacity-Aware Inference (2503.05066, ICLR 2026)、From Tokens to Layers (2510.08055, MLSys 2026)。**
逐篇核实后结论是：
- `Capacity-Aware Inference` 解决的是**专家间负载不均衡**（用容量丢弃/扩展候选专家来缩小最慢专家的方差），**没有**打破跨层同步屏障本身，仍在同步批处理框架下运作——**不构成直接碰撞**，但值得作为"另一种缓解 straggler 效应的思路"在相关工作里对比引用，因为它在 OLMoE 上已经拿到 30% 加速（代价 0.9% 质量），是 TokenRace-EP 需要在同预算下击败或证明互补的强基线。
- `From Tokens to Layers` 解决的是 prefill/decode 共置调度（layer-group 粒度，为了减少专家权重重复加载），**不是** token 级异步跨层——**不构成直接碰撞**。
- `DynaMoE` 的标题（"Layer-Wise Adaptive Capacity"）更可能是路由级动态容量调整，而不是异步释放机制，但由于 PDF 正文未能解码确认，**这一项仍是未closed的风险**，需要后续找到可读版本（HTML/其他镜像）做最终确认，不能就此排除碰撞。

综合结论：**目前没有发现与"逐 token 提前释放打破 combine 同步屏障"完全同构的已发表工作**，但 AMoE/AEP 是必须正面处理的最近邻，且这个结论本身依赖于"DynaMoE 不是碰撞"这个尚未 100% 确认的前提。

**4. [中等] Hypothesis-level 参数从未做过敏感性扫描。**
`BASE_LAUNCH_US=6.0`、`PER_TOKEN_US=0.35`、`ATTN_SHARED_US=4.0` 全部是"illustrative"常数，P0 结论文档自己也承认"需要确认结论不是对这两个参数具体取值过度敏感"，但截至目前从未真正跑过这个敏感性扫描——这是文档里承诺但尚未兑现的部分，应该在投入单 GPU 之前完成，而且做起来成本很低（在 Mac 上跑一个二维网格：launch_us × per_token_us，观察 P99 improvement 的等高线，确认"OLMoE 稳健通过、LLM-jp 卡边界"这个格局在合理参数范围内不会反转）。

**5. [较小] 统计口径的一个可以说清楚但目前没写清楚的细节。**
`full_barrier` 的 P99 是对 `n_decode_steps` 个 decode-step 级标量取的百分位（每个 decode step 里所有请求共享同一个 barrier 值），`token_race` 的 P99 是对 `n_decode_steps × B` 个请求级标量取的百分位。这两者的样本量不同（相差 B 倍），但由于每个 barrier 值在等价意义上被 B 个请求"共享"（即把 barrier 值重复 B 次求百分位，其 CDF 与只取一次求百分位是完全一致的），这个口径实际上是**自洽合理**的，不是 bug，但报告里从未明确解释这一点，容易被审稿人质疑"两侧样本量不一致是否引入偏差"，建议在方法节里补一句说明消除这个疑虑。

## 二、这个 idea 现在的上限判断

`[Inferred]` 在当前证据下，TokenRace-EP 相对于前四条已死路线（CreditReduce/RouteFidelity-EP/WaveCredit/MassCover）确实是质的进步：它是第一条切换到"执行时间线/同步粒度"这个全新因果层、且拿到了跨模型一致、单调、可解释（K 越大收益越小）的正效应曲线的路线。**如果**（且仅如果）"重新分批开销可控"这个假设成立，它在 K=2-8 的主流生产 MoE 配置上有 10%-22% 的 TBT P99 改善空间，且已经用真实路由数据和统计稳健性检验支撑；这本身对硕士论文而言已经是一个足够扎实的核心。

但目前的证据边界必须诚实地画在这里：**所有数字目前都只是"零重批开销假设下的收益上界"，还没有任何证据表明这个上界在真实系统里能保留哪怕一部分。** 这和之前四条路线的"数值仿真显示有信号，但支撑不了系统主张"是同一类风险，只是这一次信号本身干净得多。真正决定这条路线能不能到 CCF-C 的，不是路由层面的仿真（已经做得足够好），而是接下来这一个问题：**在真实 GPU 上，"提前释放已完成 token、重新组装成更小的下一层 batch"这个操作的实际开销，是否显著小于它消除的 straggler 等待时间。** 这是一个单 GPU 就能回答的、干净的二元生死问题，不需要多机、不需要 RDMA。

## 三、建议的下一步（按信息增益排序）

1. **[最高优先级，单 GPU，几小时量级] 重批开销微基准（micro-benchmark）**：在一张 GPU 上，用一个简化的 MoE FFN kernel，测量"把 N 个 token 从一个正在进行的 batch 中动态抽出、重新组装成新 batch、送入下一层"所需的真实 wall-clock 开销（对比小 N=1,2,4,8 情形），直接替换掉 P0 模型里的 `Hypothesis`级常数。这一步单独就能判断 TokenRace-EP 生死——如果重批开销系统性地大于仿真里省下的量级（个位数到二十几微秒/层），直接判死，不再继续。
2. **[高优先级，Mac，几十分钟] 参数敏感性扫描**：把 `BASE_LAUNCH_US`/`PER_TOKEN_US` 换成步骤1实测的真实值（或在没有 GPU 结果前先做一个网格扫描），重新跑 P0/P0-B，确认"OLMoE 通过、LLM-jp 卡边界"的格局在真实参数下是否依然成立，还是被参数选择完全决定。
3. **[中优先级，文献] 正面处理 AMoE/AEP (2505.08944) 和确认 DynaMoE (2603.01697) 的真实机制**：读 AEP 全文的 Section 4-5（异步调度设计）和评测部分，明确写出 TokenRace-EP 与它在决策粒度（layer-advance vs dispatch-barrier）上的本质差异；同时找到 DynaMoE 的可读版本，排除或确认其与本方案的碰撞。
4. **[中优先级，Mac，如果1的结果是正面的] CUDA Graph 兼容性路径设计**：既然 vLLM/SGLang/TRT-LLM 的 CUDA Graph 都要求固定 shape，需要明确设计"TokenRace-EP 在放弃/部分放弃 CUDA Graph 后，decode 阶段整体是否仍然净赚"——这需要把"CUDA Graph 带来的加速"也作为对照基线的一部分，而不能假设它是免费的。

## 四、一句话总结

TokenRace-EP 是五条候选里第一个机制新颖性和路由层证据都站得住的方向，但当前所有正数收益都建立在"重新分批零成本"这个从未被检验、且已被外部证据（launch overhead 5-10us、CUDA Graph 固定 shape 约束）证明是高风险的假设之上——下一步不是继续优化仿真或者去找第三个模型对照，而是必须用一次单 GPU 微基准把这个假设的真伪确定下来，这决定了整条路线是升级为 PROMISING 还是被第五次判死。

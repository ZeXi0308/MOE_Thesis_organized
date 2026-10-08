# 专家扫描摊销与完整请求成本模型

当前落地动作只有原生 scheduler 的 token budget Q；专家预算、KV预算、输出契约及所有结果保留。目的不是把已有 chunked prefill 改名，而是用可验证的瓶颈模型判断优化空间在哪里，并以强静态动作建立基线。

## 1. 状态、动作与资源

调度前状态包含运行/等待请求、各请求已计算位置、真实KV空闲块和所有权、各层当前专家驻留集合，以及此前已经完成的服务时间。动作 Q 仅限定本步总 scheduled tokens；现有原生调度器决定具体请求，随后沿该策略独立计算 route/KV/output。不能拿另一 Q 的未来路由作为反事实。

资源约束为 M_expert + M_KV + M_activation(Q) + M_other <= M_GPU。固定专家/KV预算不等于总峰值显存完全相同：本轮4096相对512实际peak allocated多239MiB，已入账。部署时应以所有候选都满足的总显存上限比较。

## 2. 可以证明的局部字节下界

本层本次调用的实际所需专家并集为 U，入口驻留集合为 R，容量 C，每个专家权重大小为 w。所有top-k贡献均需保留时，至少搬运 w*|U\R|，并且 |U\R| >= max(|U|-C,0)。现有普通expert-major计划在本次调用中达到入口miss-once：每个缺失专家只加载一次；这已由实际调用日志核对，不是假设可任意消除的重复加载。

当prefill的U接近全64、C=24时，单次调用的必要扫描已接近40个专家。换缓存身份很难继续降低这一项。优化可改的是调用/扫描次数，以及扫描与计算的暴露关系。这个下界只管本次所需集合和缓存，不能把固定轨迹下界当作另一调度策略的真实future。

## 3. 完整成本必须包括少掉的decode合批机会

所有步骤按“含prefill”与“纯decode”互斥划分：

T_episode = sum(T_engine_prefill-bearing) + sum(T_engine_decode-only) + T_outside_engine.

每个完整请求的flow从原始arrival到实际completion。expert copy events可能含host间隙或等待，不再与engine wall叠加。有限backlog改变Q后，prefill更快也意味着prefill期间顺带执行的decode token减少；后续pure-decode成本可能增加，不能只以prefill的局部节省推算收益。

本轮等输出长度实测：Q=512→4096，含prefill22→3步，专家搬运172.965→23.807GB；pure decode49→61步，搬运142.753→178.703GB。两次均值的前者引擎成本4.523→0.901s、后者4.445→5.600s，净episode8.972→6.505s。这里是互斥实测分账，不是从已知路由推导未执行动作的收益。

## 4. 多目标边界与下一方法的资格

部署目标应明确为：在总显存与预先指定的输出停顿预算下，降低完整请求flow/提高完成服务量。没有业务时限时，报告吞吐—停顿的实测Pareto关系，不事后挑一个时限制造SLO收益。

固定Q=4096在当前32输出探针降低wall27.5%、mean flow33.0%，但最坏ITL约238→327ms；不能称所有延迟同时变好。Q=2048也未在所有延迟统计量上单调居中。任何自适应策略必须胜过满足同一时限的最强静态Q，并计入校准、决策、fallback和未完成请求。

## 5. 当前结论与新颖性边界

已实现且实际有重复正信号：相同引擎资源配置下的token-budget优化。尚未实现/证明：超越最佳static与Sarathi-style profiling的在线机制、自然答案质量等价、真实大模型显存不足域、跨模型稳定收益。

[Sarathi-Serve §4.3](https://www.usenix.org/system/files/osdi24-agrawal.pdf)已明确以TBT约束选择token budget；[ExpertFlow §3.3–3.4](https://arxiv.org/html/2410.17954v2)已在MoE offload中做route-guided rebatching与缓存。因此，本轮收益是可用的优化基础，不能单靠公式或27.5%就宣称CCF B/C级独立贡献。

下一步已在执行新的source32..47自然EOS/max512六臂。只有完整结果及质量通过，才值得检查从上述资源与成本边界能否提取超越既有固定profiling的增量动作。

## 自然完成增补

新的16题六臂已完整自然EOS，均8/16同题正确、无512cap。2048相对512平均wall降低12.1%、meanflow降低14.8%、实际token/s提高10.1%；输出token减少3.2%，不视为等工作量加速。4096重复存在耗时波动且meanflow配对可退化。六臂最大停顿都匹配当前请求KV抢占，说明后续成本模型还需纳入调度引起的prompt重处理与缺席成本，不能仅按单步扫描吞吐选择最大Q。当前把2048设为已实跑的强静态参考，未形成超越它的新方法。

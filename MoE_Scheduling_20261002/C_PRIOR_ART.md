# 专家/KV容量交换：直接先例后的裁决

2026-10-02。静态六格有边际信号后，只沿具体动作核对一手全文；没有运行这些论文的系统。

[VAMP, Dynamic HBM Repartitioning for Multi-Turn MoE Serving, v1, 2026-09-11](https://arxiv.org/html/2609.13537v1#S4) 已在 KV 分配无法满足时，比较专家卸载、prefix eviction、request preemption及组合动作的预计未来成本，将专家区域转成 KV。§IV-C/V-A 使用 CUDA VMM，保留地址且不用搬运既有 KV；当前控制器只扩 KV，未在压力退去时恢复专家驻留。§IV-D 将整段卸载权重提前载入，和本地按实际专家并集分页不同。

**本研究判断：这足以覆盖当前 C 的中心动作与决策时机。** 换成CPU快照、LRU expert cache、单卡/单轮输入，或用在线测量替代配置系数，均不自动构成独立创新。静态23槽的+3.04%输出率不是动态交换证据，也不能替代与VAMP的差异说明。

[FluxMoE v1 §III–IV](https://arxiv.org/html/2604.02715v1) 已明确以专家分页释放空间给运行态；[PagedWeight v1](https://arxiv.org/html/2607.16184v1) 则以运行时权重量化交换精度与KV容量。这里仅用于确认“权重与KV联合资源分配”不是新问题，不将三种实现混为同一算法。

决定：停止普通grow-only动作的GPU诊断与控制器开发。`live_grow_src` 已开始的源码留作**未上传、未执行、未验证的参照草稿**，不能引用为可运行成果。独立输入48–63仍执行原24/23/20/20/23/24静态协议，检验资源曲线是否仅属于开发输入；它是测量/强基线工作，不是C方法主结果。后续若重启容量动作，必须先有本质不同、可执行且有价值的中心机制，不能把双向操作的直观扩展先当新颖性结论。

附带核查：[From Expert Reduction to Behavioral Divergence](https://arxiv.org/abs/2607.28097) 已研究专家归约语义造成轨迹变化；[vLLM batch invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/) 已提供MoE相关支持。当前expert-major分组做BF16 partial sum，容量变化会改变归约，故保留内容/质量差异，不将“固定归约以保持数值”本身包装为另一个创新。未做本机端到端归因实验，不能认定全部输出差异都由该项导致。

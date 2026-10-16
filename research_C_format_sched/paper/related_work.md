# 限定查新与论文定位（2026-10-15）

只核对任务指定论文及官方实现；下表是论文 T2 的半页依据，非全面综述。

| 工作（核对版本） | 信号与动作 | 目标、边界及本课题定位 |
| --- | --- | --- |
| [PagedWeight, v1](https://arxiv.org/html/2607.16184v1)，§3 | KV 压力、路由质量权重、prompt 残差；运行时增减 AP bit-plane 并异步搬运 | 质量–显存–吞吐；已存在在线专家精度管理，非全驻留 FP8/FP4 kernel 交叉点策略 |
| [MxMoE, ICML 2025](https://proceedings.mlr.press/v267/duanmu25a.html)，[§4.2–4.3](https://arxiv.org/html/2505.05799v1) | 离线敏感度/激活频率、实测 tile 成本；ILP 分配精度与 tile，异构 GroupGEMM | 已有激活分布和硬件成本共同决定混精度的直接依据；动态激活量化不等于逐步重分配权重格式 |
| [DynaExq, v4，2026-09-10](https://arxiv.org/html/2511.15015v4)，§3.2、§3.5、§5.3 | 长窗口路由计数 EMA、每层 top-n；在线切换专家版本，异步 H2D，迟滞 | 固定显存预算下质量与延迟/吞吐；**已有“运行时按负载切换专家精度并提速”的宽义直接先例**。尚未在本文确认逐步 M_e、两套全驻留及能耗交叉点决策的相同方案 |
| [MC-MoE / Mixture Compressor, v2](https://arxiv.org/html/2410.06270v2)，§3 | 离线专家重要性分配位宽；在线按 token 重要性和路由分数裁剪专家 | 在线动作主要是裁剪，不能把它描述成在线格式切换 |
| [QServe / QoQ, v3](https://arxiv.org/abs/2405.04532v3) | W4A8KV4、渐进量化、INT4→INT8 解码和 kernel 协同 | 已说明低比特性能受反量化与 CUDA-core 开销约束；不是 MoE 逐步双格式调度 |
| [vLLM v0.26.0 / ModelOpt](https://docs.vllm.ai/en/v0.26.0/features/quantization/modelopt/) | checkpoint 量化方案与硬件/shape 支持决定后端；可能在运行时回退 | 工程能力和被测基线，不作为本课题的新方法 |

**决定：按附件既定条件采用刻画型论文定位。** DynaExq 足以否定宽泛的首次主张；本研究继续测 SM120 上的实际路径、负载交叉点、质量及匹配 SLO 的能耗，并保留双格式实现和评价。两套全驻留、逐步信号和能耗目标是待验证的研究范围，不单独证明创新。查新不等于复现：DynaExq/PagedWeight 的性能数字均为原文主张，本研究尚未验证。

版本实现的直接提醒：

- [v0.26.0 NVFP4 oracle](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py) 已包含 `FLASHINFER_B12X`；注释说明其暂不自动选择，须显式 `moe_backend="flashinfer_b12x"`。同一函数明确可能发生 shape-specific fallback。不能把“auto 选 Marlin”扩写为“SM120 不存在原生 FP4 MoE”。
- [FlashInferB12xExperts](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py) 面向 SM120/SM121，内部量化激活并调用 `B12xMoEWrapper`；是否能在租用机运行、是否正确及是否加速均待实测。两份代码也已在本地 `runs/source_snapshot/` 核对。
- ModelOpt checkpoint 的 NVFP4 名称不能替代实际算术路径；T1 必须记录后端类、profiler kernel 名、激活精度和 scale 粒度。Marlin 路径若实际使用 16-bit 激活，应记为 NVFP4 权重的 W4A16 执行。

论文来源与代码入口：[MxMoE 作者仓库](https://github.com/cat538/MxMoE)、[MC-MoE 作者仓库（已重定向）](https://github.com/Aaronhuang-778/Mixture-Compressor-MoE)、[QServe/OmniServe 作者仓库](https://github.com/mit-han-lab/omniserve)、[NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)。本次未确认 PagedWeight 或 DynaExq 的公开官方实现入口，不把“未找到”写成“未开源”。

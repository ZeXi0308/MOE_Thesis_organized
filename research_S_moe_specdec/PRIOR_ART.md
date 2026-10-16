# 最近邻与当前差异（2026-10-16）

**定位：继续建成本模型、计数器、步级控制器和热图；把已有现象准确归属。** [MoESD（v4，NeurIPS 2025）](https://arxiv.org/html/2505.19645v4) 已解释小 batch 增加专家权重读取、中等 batch 专家集饱和和随后算力饱和的收益曲线，并给出专家数与 roofline 拟合。因而“发现 MoE 的 batch 规律翻转”及“首次把 distinct 专家数纳入成本”不能作为新增贡献。当前主线是 determinants characterization：实际草稿可用率、验证形状、路由相关性与服务指标边界如何改变对 batch×γ 和 J/token 图谱的解释。真实路由校准对 kernel 时间/动作排序、在线信息对调好 batch 查表的增量仍需实测；不能只靠更换卡/模型形成贡献。

**用户指定转向条件尚未由本次查新确认。** MoESD §3.3 Algorithm 1 的优化量是离线拟合参数；§3.4 是部署讨论；§4 是 vLLM 的固定 γ 实验。本次检查 v4 正文及附录、arXiv 页面和作者/题名源码检索，未定位到 MoESD 在线选择最优 γ(B) 的系统实现或作者源码入口。这是本次证据范围，不能断言所有代码均不存在。[Cascade 原文](https://arxiv.org/html/2506.20675v1) 则明确已有 MoE 动态 γ 系统：vLLM 0.6.6、单请求重点、在线 test-and-set/关闭/回退/爬山；它是必须正面对照的方法近邻。

**基线更新：** [vLLM v0.26.0 配置](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/config/speculative.py) 已有 `num_speculative_tokens_per_batch_size`，不是只有旧式关闭阈值。[官方动态推测文档](https://github.com/vllm-project/vllm/blob/v0.26.0/docs/features/speculative_decoding/dynamic_speculative_decoding.md) 给出可设 K=0 的分段表；[工具实现](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/spec_decode/dynamic/utils.py) 不限制 K 单调方向。故应同时比较调好的 batch 表、其单阈值关闭特例和静态 γ；仅赢关闭阈值不足以说明在线 EWMA 的价值。本机读取的源码快照与这些关键字段一致（`runs/environment_20261016/source/`）。

| 工作 / 原始入口 | 状态或信号 | 动作及实现范围 | 本课题需辨认的增量 |
|---|---|---|---|
| [MoESD v4](https://arxiv.org/html/2505.19645v4) | batch、稀疏度、专家数量、接受信息 | 性能建模与固定 γ 实测；本次未定位在线 γ(B) 控制器 | trace 校准、动作预测、能耗及 v0.26 步级执行 |
| [Utility-driven SD / Cascade](https://arxiv.org/html/2506.20675v1) | 实测迭代成本与有效 token 率 | 周期试探 γ，关闭/重启、回退与爬山；vLLM 0.6.6，重点 B=1 | 显式 batch 状态与离线校准是否减少试探成本；不能称首次 MoE 自适应 |
| [AdaEAGLE](https://arxiv.org/html/2412.18910v1) | token embedding、target hidden state | 训练三层 MLP 预测下一轮长度 | 不训练预测器；明确测 CPU 控制成本 |
| [SpecDec++ v3](https://arxiv.org/html/2405.19715v3)、[作者代码](https://github.com/Kaffaljidhmah2/SpecDec_pp) | 条件接受概率预测头 | 按累计拒绝风险提前结束草拟 | MoE 验证成本与 batch 信息是否改变动作 |
| [DISCO](https://proceedings.mlr.press/v262/mamou24a.html) | draft 概率、熵与位置 | 浅层分类器决定继续草拟或验证 | 与 n-gram 可用信号和成本项不同；不应把 HF 置信阈值当其完整复现 |
| [EAGLE-3](https://arxiv.org/abs/2503.01840)、[官方代码](https://github.com/SafeAILab/EAGLE) | target 多层特征融合 | 草稿模型训练及推测生成 | 本课题的一种草稿源；权重兼容性/显存/草拟成本必须实际记录 |
| [vLLM 0.26 动态表](https://github.com/vllm-project/vllm/blob/v0.26.0/docs/features/speculative_decoding/dynamic_speculative_decoding.md) | 已调好的 batch 区间 | 任意分段 K（含 0）；文档列 EAGLE/EAGLE-3/DFlash，其他方法需验证 | 相同执行路径下，在线接受率对强 batch 表的额外决策信息 |

2026 年的直接相关工作 [MoE-Spec](https://arxiv.org/abs/2602.16052) 控制验证时的专家预算，允许质量取舍；当前主线保留完整 target 路由，只改变 γ，不实施该动作。若以后做可选专家跳过扩展，需补此近邻，不能把扩展称为无损。

## 建模时立即落实的口径

- 每层定义 `q_e = 路由到 e 的 token 数 / token 总数`，top-k 无重复时 `sum(q_e)=k`。跨 token 独立近似下 `D(S)=sum_e[1-(1-q_e)^S]`。如果旧 trace 的 `p_e` 是 assignment 占比、和为 1，则 `q_e=k*p_e`，不能直接套 `S*k` 次有放回抽样。无需假设同一 token 的不同专家独立。保留跨 token/请求相关性的 trace 联集作对照。
- `S=B(γ+1)` 仅适用于每请求完整提供并调度同长草稿的未分块 decode 调用；一般用 `S=sum_i(1+k_i)`，其中 `k_i` 是实际调度前缀，混合 prefill/decode 则直接读执行调用的有效 token 槽。主计数用该实际 S 与每层 `topk_ids` 联集；区分 padding、prefill、共享专家与已过滤 token。distinct×权重字节是逻辑工作集代理，不是直接测得的 HBM 流量。
- 拟合 `T_MoE=max(aD,bS)+c` 是固定层/精度/kernel 的组件模型，输入为实测 D 和实际 S。当前尚未测得同 kernel 拟合，不能声称 c 主导或将 c 解释成已识别的 runtime 开销；系数也不是实测带宽。计划按 B={1,4,16,64} 校准、{2,8,32} 留出检验，真实专家权重配合合成 activation 的 route replay 仍不是完整服务。用实测 D 的时间预测与 route→D 留出预测分开。
- 总迭代成本包括草拟、验证、采样、调度、控制及等待；当前完整同步 host step 包含本轮验证和下一份草稿，不能相加已重叠时间。后截断须区分生产预算 G 与验证 cap γ，完整 `(B,γ,G)` 成本不能用 `G=γ` 原生静态点替代；γ=0 也可能已支付完整草拟。只有已隔离、不重叠的 draft timing 才能作明确标注的替换估计。KV 数据可能被复用，但 attention 计算与物理读取不保证随 γ 不变；查表不能写“零成本”。
- 将 n-gram availability `v_j=P(M>=j)` 与条件前缀 survival `s_j=P(前j位接受|M>=j)` 分开，未被 EOS/输出上限截断时 `A=1+sum_j(v_j*s_j)`；两者对应同状态/生产预算时此分解不要求独立。availability 含空草稿且在控制器截断前采集，超过生产预算的 rank 是未观测。只有 availability 全为 1 且逐位条件接受概率恒定才退化成几何式。首拒绝后的 token 不是额外 tested rejection；按 rank 的前缀失败则包括更早拒绝。用真实 emission/无条件接受 survival 校准，记录 EWMA 未观测 rank 的先验及近似。
- 强基线用原生非单调 batch 表（闭区间、允许 cap=0）和原生关闭阈值；两者都改变 production budget，并由校准集选定后冻结。完整草拟后截断的 threshold 仅作执行成本消融，不能代替 native threshold，也不能把静态点拼出的选择分数称为该表已测收益。
- 附件中的 OLMoE 数字是均匀、独立、纯专家带宽模型的演算，不是实测预言校准值。Qwen 翻转点和同 SM 的跨卡性能迁移尚需验证；同指令集不意味着相同带宽、缓存、kernel 选择或转折点。

当前证据：OLMoE/n-gram eager r02 全部 56 格、112 条 performance records 已完成。γ0拟合、γ1–7同prompt执行留出的 actual-S 占用预测 MAE=1.9145；名义 B(γ+1) 为15.2994，不能由此升级为 kernel 时间或动作排序已验证。有限人口同步wave吞吐、请求级有效TPOT和device J/token给出不同排序；仅B64预热、2次重复，保留cold shape和顺序影响。下一步在同路线内测 compiled/physical padding、覆盖更高草稿源及真实kernel，再比较冻结的native表/阈值；不提前归因c或固定开销主导。

## 第二草稿源的定向核对

- [wantsleep/OLMoE_1B_7B_Eagle3](https://huggingface.co/wantsleep/OLMoE_1B_7B_Eagle3) 的模型卡明确用 **0125-Instruct** 生成训练答案，推理示例也是此版本；不能用它证明现有 **0924-Instruct** 兼容。定向搜索未找到针对 0924 的明确 checkpoint。搜索中出现的 0924 文本来自附带建模代码的示例字符串，不是训练目标声明。先用 n-gram/suffix 可推进不同非神经草拟机制，但最终图须如实标注，不能标成 EAGLE。
- [AngelSlim/Qwen3-a3B_eagle3](https://huggingface.co/AngelSlim/Qwen3-a3B_eagle3) 是 Qwen3-30B-A3B 的优先 EAGLE-3 候选；作者模型卡报告了 vLLM 0.11.2/H20 单卡该 target 的结果。后续一手 metadata 已固定 revision `266a50ea8c9dbedb729d7c18a98dd79f5b39b5c2` 并核对 config；完整文件清单、target revision 与 hash 见 `configs/draft_sources.json`。模型卡和 metadata 证据不等于 v0.26/SM120 native 运行已通过。
- [Tengyunw/qwen3_30b_moe_eagle3/config.json](https://huggingface.co/Tengyunw/qwen3_30b_moe_eagle3/blob/main/config.json) 可作为备选，声明 `LlamaForCausalLMEagle3`、hidden size 2048、vocab 151936、draft vocab 32000。[v0.26 registry](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/model_executor/models/registry.py) 已注册这一架构，[Qwen3 MoE target](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/model_executor/models/qwen3_moe.py) 实现 `SupportsEagle3`。这是源码可达性的证据，尚未验证权重映射、目标 revision、实际接受率、SM120 kernel 与显存。下一步是下载固定 revision 后在真实环境做短 native 运行，再开启矩阵。

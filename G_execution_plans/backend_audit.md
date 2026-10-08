# 当前后端可行性审计（仅源码证据）

判定：代码中存在可以进入门槛探针的不同执行路径；**尚未通过 A 的竞争力与正确性验证，也没有通过 B/C**。这份审计未运行 GPU、未访问远端。以下依据是 `native_sources/` 中从当前环境复制的源码，版本及散列以 `evidence/source_sha256.json` 为准；不能仅凭这些文件将当前本地实现的特性宣称为正式 upstream vLLM 0.26.0 的原生特性。

## 1. 合法候选与排除项

当前模型配置是 BF16 OLMoE，16 层、64 experts、top-8、hidden 2048、expert intermediate 1024、16 KV heads、head dim 128、最大上下文 4096。`model_executor/models/olmoe.py:97` 使用 `FusedMoE(..., renormalize=False)`；不可为了比较而改成归一化 top-k。

| 候选 | 当前源码兼容性依据 | 实际不同之处 | 当前证据等级 |
|---|---|---|---|
| T-small：Triton BF16，小 tile | `experts/triton_moe.py:91` 支持 CUDA，`:109` 支持未量化，`:123` 支持 SiLU；`:262` 调用配置选择 | tile 改变 kernel specialization、grid 和 expert padding，非名称变化 | 可进入 GPU 正确性及速度探针 |
| T-large：Triton BF16，大 tile | 同上；`fused_moe.py:1368` 接受 override；`:1383` 支持按 token 数从调优表选最近配置 | 与 T-small 有不同 BLOCK_SIZE_M/N/K | 可进入探针，不保证有竞争力或额外驻留成本 |
| FlashInfer CUTLASS BF16 | `experts/flashinfer_cutlass_moe.py:131` 明确接受 SM120 family 且要求扩展可用；`:149` 接受 `(None,None)` 未量化方案；`:187` 接受 SiLU | `:376` 调用 FlashInfer CUTLASS API，不是 Triton 名称别名 | 需安装版本能力查询、JIT、相同 tensor/route 正确性测试 |
| TRTLLM BF16 | `experts/trtllm_bf16_moe.py:59` 要求 capability family 100 | SM120 不满足当前设备 gate | 排除；注释中的 “Blackwell” 不等于支持全部 Blackwell 产品 |
| FlashInfer B12x | `experts/flashinfer_b12x_moe.py:58`、`:164` | 虽支持 SM120，但只支持 NVFP4 | 排除，不能改变精度 |
| FlashInfer CuteDSL | `experts/flashinfer_cutedsl_moe.py:46`、`:69` | NVFP4，且 family100 | 排除 |

候选 tile 可直接取已实现的默认合法形状，而不是开始扫描：T-small `{BLOCK_SIZE_M:16, BLOCK_SIZE_N:64, BLOCK_SIZE_K:128, GROUP_SIZE_M:1, SPLIT_K:1, num_warps:4, num_stages:4}`；T-large `{BLOCK_SIZE_M:64, BLOCK_SIZE_N:128, BLOCK_SIZE_K:64, GROUP_SIZE_M:1, SPLIT_K:1, num_warps:8, num_stages:3}`。依据是 `fused_moe.py:1313` 至 `:1354` 的默认分支。它们只是待测初始点，不能因属于默认分支就免除 SM120 实际编译与正确性验证。

`oracle/unquantized.py:67` 的 CUDA 优先顺序为 TRTLLM、CUTLASS、Triton；`:298` 对实际配置作支持检查。因此必须记录本次模型加载日志中的真实 backend；不能称 Triton 为未经核对的“当前默认”。`:277` 允许明确选择 backend，`:150` 映射 `triton` 和 `flashinfer_cutlass`。

CUTLASS 比较还需要正确处理权重布局：`oracle/unquantized.py:321` 对 gated activation 交换 w13 两半。直接把 Triton 格式 w13 原样送给 CUTLASS 会构造错误对照；合法转换仅重排同一组权重。两个 backend 同进程并存若保留两份布局，应计入真实复制成本，而不能将其记成零成本。首轮无需实现混合 backend 的逐层分派。

## 2. 组合成本已存在的共享机制

Triton 主 workspace 的形状由 M、模型 shape 和 top-k 决定，与上述 tile 选择无关：`experts/triton_moe.py:181` 返回 `(M,topk,max(N/2,K))` 与 `(M,topk,max(N,K))`；当前 gated OLMoE 的 N=2048、K=2048、topk=8。BF16 两个 workspace 合计 **64 KiB × M**。这只是源码推出的模块 workspace 下界/分项，不是整个执行计划成本。

`modular_kernel.py:1261` 以 M_full 同时作为 chunk/full 参数分配；`:1104` 将 output 与 workspace13 取 max 并复用；`:1105` 向全局 manager 请求同时存活的 workspace。`v1/worker/workspace.py:34` 每 active ubatch 只有一个 buffer，`:103` 将同时存活的视图按 256 字节对齐相加，`:132` 仅在需求超过现存容量时增长，`:168` 替换旧 buffer 并清理 allocator 空闲缓存，`:54` 支持锁定。因而不能再乘 16 层、tile 数或 graph 数。`gpu_model_runner.py:6424` 在 graph 估计前按 `max_num_tokens` 执行 profile_run；`:6838` 在正式 capture 后锁定 workspace。在固定 max_num_batched_tokens 时，删减较小 graph 桶甚至减少 graph 最大桶，都未必降低此项已分配的高水位。

CUTLASS wrapper 对 manager 可见的 workspace 只有 `(M,K)`（`experts/flashinfer_cutlass_moe.py:239`），但 FlashInfer 内部 scratch 与缓存不在此返回值中。**不能据 4 KiB×M 对比 64 KiB×M 就宣称得到完整节省**，需补 FlashInfer 实现和整个进程测量。

`compilation/cuda_graph.py:200` 让 wrapper 使用 global graph pool；`:315` capture 显式传该 pool；`:325` 和 `:336` 使用弱 output 引用降低额外保留。`gpu_model_runner.py:6728` 明确说明 FULL 和 PIECEWISE 不并行 replay、共享池覆盖，两者共享主项取 max。`v1/cudagraph_dispatcher.py:326` 按 PIECEWISE 后 FULL、各自 batch 大到小的固定顺序返回 capture 描述。这些机制压低边际组合成本，但不能证明独立 graph 数据、driver 开销或其他静态 buffer 为零。

Triton tile 仍可能改变 padding 辅助数据：`fused_moe.py:1520` 把 BLOCK_SIZE_M 传给 `moe_align_block_size`。极小 batch 的特殊路径由 `M*topk*4 <= E` 触发（`:1498`），对本模型即 M≤2；它不能替代现实服务 batch 的评估。必须实测 whole set，不把所有单 graph 的峰值相加。

## 3. 当前实现已经把 graph 估计纳入 KV 预算

启动顺序是：加载模型 → `profile_run` → 用临时最小 KV 及临时 graph pool 估计 capture 成本 → 根据可用预算计算正式 KV blocks → 正式 capture → 报告实际 capture 差额。完整 engine 外层调用顺序仍宜补 `v1/engine/core.py` 核实，单 worker 与 runner 已足以证实这些阶段和其职责。

1. `gpu_worker.py:484` 测非 KV 内存，`:506` 调 `profile_cudagraph_memory()`；`:508` 避免重复计算 graph 导致的 torch peak；`:541` 将估计 graph bytes 从 requested budget 扣除。
2. `envs.py:1986` 的 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 默认 **1**；不能按老版本假设 graph 完全不计入 KV。
3. `gpu_model_runner.py:6439` 临时分配最小 KV；`:6614` 用临时共享 graph pool；`:6658` **只采每个模式最大的两个 graph**。`:6662` 用设备 free-memory delta；`:6681` 给每个后续 graph 设置至少 **1 MiB** 的估计。`:6731` 的公式为 `max(各模式首个capture) + Σ各模式后续graph线性估计`，不是整个目标集合的精确测量。
4. `:6711` 清掉 profiling graphs，`:6725` 清掉临时 KV，正式 KV 随后重新分配。`v1/core/kv_cache_utils.py:1005` 根据可用 bytes、每层 page size、层数取整得到 blocks，`:963` 允许 override。主实验不能使用 blocks override 或固定 KV bytes。
5. `gpu_model_runner.py:6805`、`:6824` 在正式整个集合 capture 前后读取设备 free memory；`:6842` 形成组合 delta 并记录 capture 时间。**end_free 在随后 empty_cache 之前采样**，所以这个日志字段仍可能夹带 allocator 暂存，不能直接认作“全部最终 graph 独占驻留”。需同时记录 capture 后同步、清缓存的 device free、allocated、reserved 和 live workspace，并与 KV allocation 相对照。
6. `gpu_worker.py:786` 比较 estimated/actual；`:818` 计算建议 KV 值并保留 150 MiB buffer。这里生成的建议不会扩展已分配的 KV。graph 删除或 cache 清理后腾出的 bytes 也不会自动成为现有 KV pool。

当前还存在 `v1/worker/startup_plan.py`：它按配置/硬件指纹持久化一个 KV byte 值（`:41`、`:179`），下次启动写回 `cache_config.kv_cache_memory_bytes`（`:164`）以跳过 profile/graph estimate。默认关闭（`envs.py:1776`）。这只是**已有配置的启动结果复用**，没有执行配置组合搜索、共享内存边际成本或 batch 反馈。为避免混淆本研究，门槛测量明确设 `VLLM_ENABLE_STARTUP_PLAN=0`。即使后续实现组合选择，也不能把持久化预算缓存再算一项新贡献。

估计值随 graph 数变化而实际成本不变时，最终 blocks 的变化可能仅来自估计保守程度。这需要单独识别；若收益全部来自修正这种预留或默认桶，按立项规则停止论文主线。

## 4. 成本小表与 KV 灵敏度

表中“待测”是缺失证据，不是零。

| 计划集合 | 已知可共享项（源码推出） | 完整驻留/峰值 | 最终 KV blocks | 核心判别 |
|---|---|---|---|---|
| Triton 单 tile、紧凑桶 | 单 ubatch 主 workspace=64 KiB×M_highwater | 待测 | 待测 | 强紧凑基线 |
| Triton 双 tile、相同桶 | 同一个主 workspace；需另测代码/graph/对齐索引边际 | 待测 | 待测 | tile 本身是否存在速度—内存权衡 |
| Triton 双 tile、扩展桶 | graph 共享池，非单项之和；workspace 高水位可能不变 | 待测 | 待测 | 增量独立 graph 成本及 KV 转化 |
| CUTLASS 单 backend、紧凑桶 | wrapper workspace=4 KiB×M；内部 workspace 未知 | 待测 | 待测 | 当前自动选择可能已是强简单基线 |

若 BF16 KV、TP=1、无额外 KV padding，模型配置推出每 token KV 为 `16层 × 2(K,V) × 16 heads × 128 dim × 2 bytes = 128 KiB`；block_size=16 时每个跨层逻辑 block 为 **2 MiB**，4096-token 请求为 **512 MiB**。于是实际可转给 KV 的 1 GiB 对应 8192 tokens / 512 blocks / 两个满长请求；100 MiB 对应 800 tokens。此为单位换算，最终以 runtime KV specs 和实际 blocks 为准，不能由这些比例声称服务行为一定改变。

**边界结论：**B 的反例尚未获得；共享机制让“双 tile 必然牺牲显著 KV”的先验很弱。C 也未被否定，因为 driver/独立 graph 静态数据及真实服务压力尚未测量。当前只能停止后续复杂优化开发并保留明确门槛探针，不能声称“实验证明无收益”。

## 5. 下一次空闲 GPU 的最小决定性测量

使用已有锁，整组测量独占运行；一个计划集合一个新进程。先固定少量真实 tensor+topk_ids+topk_weights（包含两个代表 batch，不改路由/精度），比较 T-small/T-large/CUTLASS 并检查误差。仅保留竞争者，再测三组左右的整个 startup 集合，固定 warmup、capture 顺序、总预算、最大请求数与 token 限制。记录 profile 的各分项、实际正式 KV allocation、capture 全集合 delta、清缓存后稳定 memory、实际 blocks、启动时延。不通过 A/B/C 就不跑服务矩阵。

CPU 审计还可补的精确文件（**无需为此运行 GPU**）：`vllm/utils/flashinfer.py`、当前 FlashInfer `fused_moe.py` 及 CUTLASS backend workspace/JIT 实现（确定内部成本）；`vllm/model_executor/layers/fused_moe/moe_align_block_size.py`（对齐索引大小）；`vllm/model_executor/layers/quantization/utils/flashinfer_utils.py`（BF16权重转换）；`vllm/platforms/interface.py`/`cuda.py`（capability family gate 与 global graph pool）；`vllm/v1/engine/core.py`、`vllm/v1/kv_cache_interface.py`（正式分配调用顺序和 page size）；`vllm/compilation/breakable_cudagraph.py`（若实际启用）；设备特定 `fused_moe/configs/E=64,N=1024,device_name=*.json`（当前已调优配置）。这些是兼容性/来源核对，不应变成无 GPU 时无限扩展的分析工作。

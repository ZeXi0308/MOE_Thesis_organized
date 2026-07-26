# Receiver 旧低精度机制 GPU 复核（2026-07-22）

状态：**DEV DIAGNOSTIC / NOT PHASE 5 / NO FORMAL SCIENTIFIC VERDICT**

本次只回答一个窄问题：纠正后的 positive-net hard gate 生效后，旧 `calib_static / causal_no_hysteresis / controller` 低精度 lane 机制是否还留下足以支持继续主攻的信号。它不实现 DDRC 冻结协议要求的 native route producer、G1 route-only oracle、hierarchical bootstrap 或真实 RDMA，因此不能给 DDRC 科学 Go/No-Go。

## 1. 运行快照

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090，32607 MiB，driver 595.71.05，power limit 575 W |
| PyTorch | 2.8.0+cu128 |
| OLMoE | `allenai/OLMoE-1B-7B-0924@6d84c48581ece794365f2b8e9cfb043c68ade9c5` |
| LLM-jp | `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M@1d5983076dfc67aee4a77ec06a27027f5bab6055` |
| 数据 | MMLU：`abstract_algebra, computer_security, high_school_mathematics, machine_learning` |
| 样本 | calibration 8、test 16；这是 smoke-sized diagnostic，不是正式质量统计 |
| 执行 | batch 8，seq-len 320，EP proxy 8，4 GPUs/node proxy，contiguous placement |
| 策略 | `uniform_full, calib_static, causal_no_hysteresis, controller` |
| 硬门 | `require_positive_net_saving=true`（入口默认值） |
| codec 口径 | `serialized_tiles` 主复核；`once_per_step` 配对敏感性 |
| 仓库 HEAD | `242066439f05e183c6a0e907383075bc18871291`；worktree 非 clean，以下 source SHA 才是实际代码身份 |

实际 source SHA-256：

- `run_receiver_aware_task_quality_gpu.py`: `1fdbc8e461531cd550fa4c72d14e5850159c0c14f0f03a7028a4a294d1007b2f`
- `receiver_lane_policy.py`: `a25a066f1b43e7fb48ffd261e600c24c52d66680d9b88bd71578c5274c03834d`
- `capture_moe.py`: `efcdd79c8d02e30d9c27450eead4aae406e1bd69afa94ba6e689226f1b6ca0b8`

上传 bundle SHA-256：`dcafe7ffa7399ea4373adf6ec791057d2546370ffeaeb076c73958ab25e5c188`。远端结果包下载前 SHA-256 分别为 `8a5f9561b4519b7745f8ff934b4f0bd075bb02c0c930ac755506c60679b6ec4f` 和 `ed22c32dee85084b363908d0a199ff445b42b1a9521b69e61b573a1c6c76426b`。

## 2. 主观察

| 模型 / regime / codec | `calib_static` | `causal_no_hysteresis` | `controller` | 观察 |
|---|---:|---:|---:|---|
| OLMoE / balanced / serialized | 0 low pairs；0 saving | 0；0 | 0；0 | 分别 32/32、31/32、30/32 step 被 codec hard gate 回滚 |
| LLM-jp / balanced / serialized | 0；0 | 0；0 | 0；0 | 分别 32/32、31/32、30/32 step 被回滚 |
| OLMoE / balanced / once-per-step | 0；0 | 0；0 | 0；0 | 放宽到 amortized codec 仍无可执行 low action |
| OLMoE / hotspot / once-per-step | 7,843；**+0.3641%** | 6,512；**+0.3581%** | 5,122；**+0.3581%** | 最强是简单静态；全部远低于旧 3% 增量门 |
| LLM-jp / hotspot / once-per-step | 0；0 | 0；0 | 0；0 | 三个策略分别 32/32、31/32、31/32 step 被回滚 |

OLMoE hotspot 的同 action `serialized_tiles` 敏感性 saving 分别为 **-114.35% / -95.80% / -73.92%**。这些是 proxy 账本中的负净值，不是实际延迟变慢倍率；它们只说明该 action trace 无法支付当前 serialized codec tax。

原始表：

- [OLMoE balanced serialized](receiver_old_mechanism_olmoe_20260722/policy_task_quality_summary.csv)
- [LLM-jp balanced serialized](receiver_old_mechanism_llmjp_20260722/policy_task_quality_summary.csv)
- [OLMoE balanced once-per-step](receiver_old_mechanism_olmoe_once_20260722/policy_task_quality_summary.csv)
- [OLMoE hotspot once-per-step](receiver_old_mechanism_olmoe_once_hotspot_20260722/policy_task_quality_summary.csv)
- [LLM-jp hotspot once-per-step](receiver_old_mechanism_llmjp_once_hotspot_20260722/policy_task_quality_summary.csv)

## 3. 判定边界

- **[Observed] 旧机制负向复核：**该 smoke 中，唯一非零净收益是 OLMoE hotspot 的约 0.36%，且 `calib_static` 不弱于在线策略；LLM-jp 不复现，serialized 敏感性也不存活。因此旧“receiver 拥塞 → 动态 INT4 lane”不应继续作为主线。
- **[Unverified] DDRC：**本次没有回答跨 sender 来源分解相对强局部基线是否有 oracle headroom。DDRC 仍缺合规的 native route producer，不能借这组旧机制结果判科学 No-Go。
- **[Unverified] 质量：**每模型只有 16 道题，且多数 arm 因 hard gate 与 `uniform_full` 等价；accuracy/NLL 仅用于检查 action-matched 执行，没有统计结论。
- **[Proxy-only] 系统收益：**saving 是同 action 的解析 bottleneck-wire + measured codec tax，不是 NCCL/RDMA、TTFT、TPOT 或 P99。

## 4. 新发现的代码问题

`capture_moe.patch_mixtral_moe(..., record_routes=True, record_diagnostics=False)` 会静默产生空 route：

- `MoeRecorder.update_routing()` 自身正确检查 `record_routes`：`experiments/shared/capture_moe.py:118-135`；
- 但 Mixtral / OLMoE / Qwen 三个 forward 把 `update_routing()` 全部嵌在 `_idea_record_diagnostics` 条件内：同文件 `:742-749,852-859,929-936`；
- `capture_routes_v2.py:63-69` 通过保持 diagnostics 开启、再把其他 recorder 方法改成 no-op 绕开了该问题，旁证 route-only 本应可独立运行。

分类：**代码做错（flag-contract / silent capability failure）**，不是机制 No-Go。修法应把 `update_routing()` 从昂贵 diagnostics gate 中拆出，并对三种模型 forward 增加 `record_routes=True × record_diagnostics=False` 参数化测试。本轮遵守 Phase 1 约束，没有写修复代码。

## 5. 与 Energy-SLO 的能力核验

远端 vLLM `0.10.2` 的 `Fp8MoEMethod` 源文件 SHA-256 为 `c53680e29960ab391294e79d82d72ea4c396315087048b038244a8fdd422cb40`。源码显示：

- BF16/FP16 checkpoint 在 `process_weights_after_loading()` 中一次性量化并替换为 FP8 权重；
- FP8 MoE kernel 构造/执行固定使用 `use_fp8_w8a8=True`；
- `apply()` 没有 per-invocation BF16/FP8 action 参数。

这验证的是**当前 vLLM hot path 不提供现成 route-row 动态精度接口**。它是工程阻断证据，不是“动态精度机制科学无效”的证据；static FP8 能运行也不能替代冻结的 full-path dynamic Energy-SLO 实验。

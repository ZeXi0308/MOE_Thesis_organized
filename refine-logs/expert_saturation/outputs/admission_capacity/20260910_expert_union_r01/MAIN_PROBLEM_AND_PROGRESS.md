# 主问题固定与 route 采集通路打通

2026-09-12。分支 `agent/publish-current-moe-code`，HEAD `2a37765`。
本文件是当前主研究文档的第一版，替代此前的"候选并列"叙述。

## 一、主问题（固定，不再更换）

> **在单卡固定显存下，长上下文 MoE serving 的 KV 赤字迫使抢占；抢占代价的 96.3%
> 是"等待恢复"而非重算。如何安排准入与恢复决策，使新增工作不对已有请求造成
> 不成比例的等待，同时提高有效服务量？**

### 目标工作负载

OLMoE-1B-7B BF16、单 RTX 5090、vLLM 0.26.0、`max_num_seqs=32`、
`max_model_len=4096`、token budget 1024、FCFS、chunked prefill、无 prefix cache。
长上下文域：3072 prompt + 1024 output，32 个真实 WikiText 请求。

### 主瓶颈

KV 容量赤字。引擎自己打印 `Maximum concurrency for 4,096 tokens per request: 29.97x`，
而 `max_num_seqs=32`，所以**约 2 个请求的 KV 无法同时驻留**——这与我独立核算的
`32×4096 − 122,752 = 8,320 tokens = 2.03 个请求` 一致（本轮首次取得引擎侧确认）。

### 主评价目标

请求级联合 SLO goodput（TTFT 与 TPOT 同时满足），同时报告 TTFT、平均 TPOT、
**每请求最大 ITL**、完成时间、等待/恢复时长、拒绝与未完成数。
最大 ITL 是本问题的关键代价指标，因为"不成比例的等待"正体现在它上面。

### 可控动作（第一版只取前两个）

1. 准入上限与准入时机；
2. 抢占后的恢复顺序与恢复触发点；
3. （后续）prefill 工作量与 token 预算划分；
4. （后续）专家驻留预算。

## 二、支撑选题的实测证据

全部来自本工作区，非历史摘要：

| 证据 | 数值 | 出处 |
|---|---|---|
| 该运行域**有判别力** | cap32 → 0/32 完成，cap16 → 32/32 | `check_regime_discriminability` 自检 |
| 不成比例等待**真实存在** | 已交付 712 token 的请求停顿 4.47 s，两次重复一致（4.473/4.512） | `20260908_native_preemption_r01` |
| 代价**不是重算** | 等待 96.3%，重算 3.7%（4 个 victim-repeat 全部 93.6–97.4%） | 本会话 `pause_phases` 分解 |
| 等待的**终点是首个完成** | 首完成 20.627 s，victim 恢复 20.751 s | 本会话 raw 重建 |
| 赤字是**结构性的** | 引擎打印 29.97x 上限 vs `max_num_seqs=32` | 本轮引擎日志 |
| **过度驱逐** | 释放 237 blocks 以满足约 1.9 blocks 的缺口（122×） | 本会话核算 |
| MoE 相关性**不硬凑** | 专家权重 12.89 GiB 占显存主体，KV 仅剩 14.98 GiB | 本轮引擎日志 |

短文本 steady 域已被证明**无判别力**（见
[`20260911_aimability_r01`](../20260911_aimability_r01/REPORT.md) 与
[`20260912_feasibility_envelope_r01`](../20260912_feasibility_envelope_r01/REPORT.md)：
负载在可行包络的 1.97×，cap 只是把 TTFT 失败换成 TPOT 失败，联合达标摆幅仅 2–9/32）。
**因此主问题锚定在长上下文域**，这不是换题，是把同一问题放到有判别力的运行点上。

## 三、专家并集测量在主问题中的位置

它**不是独立选题**，而是回答主问题的一个前提：专家驻留能否让出显存以缓解 KV 赤字？

- 若 `NO_RESIDENCY_HEADROOM` → 赤字不可由资源侧消除，答案必须来自准入/恢复顺序，
  主问题被**收窄**而非否定；
- 若 `CANDIDATE_RESIDENCY_HEADROOM` → 打开"专家驻留预算"这条动作轴，再谈搬运成本。

判据与三条预测已在 [`DECISIONS.md`](DECISIONS.md) 冻结，本轮未修改。

## 四、本轮实质进展：route 采集通路打通

### 进展 1：forward hook 在本引擎下不可用（方法学约束）

原采集器在每个 router 上挂 `register_forward_hook`。**记录为零。**
诊断（`diagnose_union_hook.py`）给出直接证据：

| 观测 | 值 |
|---|---|
| `n_hook_calls` | **0** |
| 纯度判据 | **正常**（step≥1 起 `running=8, waiting=0`） |
| 请求属性 | **正确**（`num_computed_tokens=128 ≥ num_prompt_tokens=128`） |
| CUDA graph | `decode FULL: 7/7` 已捕获 |

原因：`enforce_eager=False` 下宽度 ≤32 的 decode step 是 **CUDA graph 回放**，
回放不执行 Python。图在 warmup 期捕获（hook 尚未注册）。

**这是一个可复用的方法学结论**：在 vLLM V1 + CUDA graph 下，
无法用 Python forward hook 观测 decode 路由。关掉 graph 可恢复可见性，
但会改变本仓库其他测量所依据的执行体制。

### 进展 2：引擎原生通路可用，且优于 hook

改用 `enable_return_routed_experts=True`。引擎日志确认：

```
Initializing routed experts capturer, enable_return_routed_experts: True
RoutedExpertsManager CPU buffer: 0.02 GB (slots=122752, layers=16, top_k=8, dtype=uint8)
```

它在捕获图内工作，且返回的是**真实执行的路由**，不是从 router logits 重算——
因此没有 top-k tie-break 歧义。代价是 0.02 GB CPU buffer，已记入。

### 进展 3：索引语义已验证，不是假设

`CompletionOutput.routed_experts` 仅在请求**完成**时填充，形状经两个不同输出长度
的请求交叉验证（`probe_routed_shape.py`）：

| 请求 | n_out | shape | P+n−1 | 匹配 |
|---|---:|---|---:|---|
| A | 5 | `[132, 16, 8]` | 132 | ✓ |
| B | 11 | `[138, 16, 8]` | 138 | ✓ |

$$\text{shape} = [\,P + n_{\text{out}} - 1,\ n_{\text{layers}},\ \text{top-}k\,],\quad \text{dtype}=\text{uint8},\ \text{值域}\,[0,64)$$

末轴恒无重复（合法 top-k）。索引 `0..P−1` 为 prompt 位置，`P..` 为 decode 位置；
最后一个采样 token 无前向，故 `−1`。

**join 规则（无未来信息）**：decode step 中 `num_computed_tokens = c` 的请求，
该 step 计算的正是位置 `c`，故 route 索引 = `c`。step 组成是在线调度器的真实行为，
路由是 GPU 的真实执行，二者对齐是精确的。

### 进展 4：修掉两个真实缺陷

**缺陷 A（会自证结论）**：hook 版本只按"至少一个 decode 请求"开启采集，
一个携带 chunked prefill 的 step 会把最多 1024 个 prefill token 的路由写入
本应是 decode 宽度的并集，使 `U` 变成 chunk 大小的函数并**伪造出饱和**——
正是本实验要检验的假说。已改为只在 `decode>0 且 无 prefill 且 无 waiting`
的纯 decode step 采集，并保留 hook 侧的 `expected_tokens` 身份守卫。

**缺陷 B（会误报正结果）**：`verdict()` 原本只查 `saturation_window_99 <= 2`，
而 horizons 最大 32。一个**每 4 步重搬全部专家**的层落入 `else` 被判
`CANDIDATE_RESIDENCY_HEADROOM`。合成 uniform w16 实测 `sat99=[4,4,4,4]` 正在此区间。
已改为 `CANDIDATE` 要求至少一层在全部视野内从不饱和。此修复方向**保守**，
只会让 `CANDIDATE` 更难达成，不会把负结果洗成正结果。

**缺陷 C（本轮最后定位）**：引擎给内部 `Request.request_id` 追加 8 位十六进制后缀：

| 侧 | 键 |
|---|---|
| 提交 / `RequestOutput` | `u/memory-train-article-0000001` |
| `scheduler.running` | `u/memory-train-article-0000001-884bdf35` |

裸字符串 join 静默匹配为零（`missing_route: 255 / 255` 个纯 decode step）。
已加规范化 + 一一对应断言：映射失败时**抛异常而非报 NO_DATA**，
避免把实现缺陷伪装成科学结论。

## 五、当前状态与接续点

| 项 | 状态 |
|---|---|
| 环境 | vLLM 0.26.0 / torch 2.11.0+cu130 / transformers 5.15.1，与封存基线逐位一致 |
| 模型 | 12.89 GiB，3 分片 + index 完整，revision `6d84c485` 校验通过 |
| `model_shape.json` | **`n_moe_blocks=16, num_experts=64, num_experts_per_tok=8`**（冻结文档要求的验证点通过；`block_type=OlmoeMoE`，走的是 fallback 定位） |
| KV 池 | 14.98 GiB / 122,752 tokens（与封存 15.00 GiB 一致） |
| 采集器 | 已重写为原生通路，缺陷 A/B/C 均已修，已上传 |
| 测试 | 30/30（本地与远端） |
| **route 采集** | **UNRUN**——修复后的启动命令三次因审批超时被取消 |

### 下一项可执行工作（仅一条）

```bash
cd /root/autodl-tmp/expert-union && rm -rf /root/autodl-tmp/s3 && \
setsid nohup bash run_session.sh /root/autodl-tmp/s3 short 32 \
  > /root/autodl-tmp/s3.log 2>&1 < /dev/null &
```

约 40 秒（引擎加载 12 s + 捕获 1 s + 256 步 decode）。成功判据：
`union_summary.json` 中 `n_recorded_steps > 0` 且 `skipped.missing_route == 0`。
随后 `adjudicate_expert_union.py` 机械判定 E1/E2/E3。

短域跑通后再跑 long 域（3072 prompt），因为主问题锚定在长上下文。

## 六、明确不主张

- 本轮**零科学结论**。所有 GPU 运行都是通路打通与缺陷定位，没有任何 `U` 的测量值。
- `U` 是结构信号：不等于实测 HBM 流量、不等于 fused backend 实际加载的字节、
  不等于闲置专家可回收。
- 引擎打印的 29.97x 是**容量算术**，不是可达并发实测。
- 短文本域的 NO-GO 结论仅在该已证明无判别力的运行点内成立，不外推到长上下文域。
- 96.3% 等待占比来自 4 个 victim-repeat 样本，单模型单卡，不作 p99 主张。

原始 raw 未修改，未 push，未改 `docs/current/README.md`。

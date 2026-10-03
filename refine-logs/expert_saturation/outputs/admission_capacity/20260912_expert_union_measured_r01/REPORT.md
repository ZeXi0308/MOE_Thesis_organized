# 专家驻留无法为 KV 赤字供血：相干时间失配

2026-09-12。分支 `agent/publish-current-moe-code`，HEAD `2a37765`。
RTX 5090 单卡，vLLM 0.26.0 / torch 2.11.0+cu130 / transformers 5.15.1
（与全部封存基线逐位一致）。**本轮有真实 GPU 执行。**

## 一、本轮在主问题中的位置

主问题（见 [`MAIN_PROBLEM_AND_PROGRESS.md`](../20260910_expert_union_r01/MAIN_PROBLEM_AND_PROGRESS.md)）：

> 单卡固定显存下，长上下文 MoE serving 的 KV 赤字迫使抢占；抢占代价 96.3% 是等待
> 而非重算。如何安排准入与恢复决策，使新增工作不对已有请求造成不成比例的等待？

本轮回答该问题的一个**资源侧前提**：专家驻留能否让出显存以消除赤字？
这不是独立选题——它决定动作空间是否包含"专家驻留预算"这一轴。

## 二、结论

> **闲置专家字节确实存在，且瞬时量足以覆盖 KV 赤字（116.6%–174.3%），
> 但它的相干时间只有约 1 步。KV 预算必须按请求生命周期（约 1024 步）持有。
> 两个时间尺度差约三个数量级，因此在实测带宽 56.44 GB/s 下，
> 不存在任何相干窗口同时满足"覆盖赤字"与"成本低于一个 step"。**

**专家驻留回收这条动作轴关闭**，理由是实测的量化理由，不是假设。
主问题因此被**收窄**：答案必须来自准入与恢复顺序。

## 三、方法：两个必须先解决的实现障碍

### 障碍 1：Python forward hook 在本引擎下不可用

首次采集记录为零。诊断（`diagnose_union_hook.py`）给出直接证据：

| 观测 | 值 |
|---|---|
| `n_hook_calls` | **0** |
| 纯度判据 | 正常（step≥1 起 `running=8, waiting=0`） |
| 请求属性 | 正确（`num_computed_tokens=128 ≥ num_prompt_tokens=128`） |
| CUDA graph | `decode FULL: 7/7` 已捕获 |

原因：`enforce_eager=False` 下宽度 ≤32 的 decode step 是 **CUDA graph 回放**，
回放不执行 Python；图在 warmup 期捕获，那时 hook 尚未注册。

**可复用的方法学结论**：vLLM V1 + CUDA graph 下无法用 Python forward hook 观测
decode 路由。关图可恢复可见性，但会改变本仓库其他测量所依据的执行体制。

### 障碍 2：引擎内部请求 ID 带哈希后缀

| 侧 | 键 |
|---|---|
| 提交 / `RequestOutput.request_id` | `u/memory-train-article-0000001` |
| `scheduler.running` 的 `Request.request_id` | `u/memory-train-article-0000001-884bdf35` |

裸字符串 join 静默匹配为零（255/255 个纯 decode step 全部 `missing_route`）。
已加规范化与一一对应断言：**映射失败时抛异常，不再报 `NO_DATA`**，
避免把实现缺陷伪装成科学结论。

### 采用的通路

`enable_return_routed_experts=True`。引擎日志确认
`RoutedExpertsManager CPU buffer: 0.02 GB (slots=122752, layers=16, top_k=8, dtype=uint8)`。
它在捕获图内工作，返回的是**真实执行的路由**，不是从 router logits 重算，
因此无 top-k tie-break 歧义。

### 索引语义（验证，非假设）

`probe_routed_shape.py` 用两个不同输出长度的请求交叉验证：

| 请求 | n_out | shape | P+n−1 |
|---|---:|---|---:|
| A | 5 | `[132, 16, 8]` | 132 ✓ |
| B | 11 | `[138, 16, 8]` | 138 ✓ |

$$\text{shape}=[\,P+n_{\text{out}}-1,\ 16,\ 8\,],\ \text{uint8},\ \text{值域}[0,64)$$

末轴恒无重复。join 规则：decode step 中 `num_computed_tokens = c` 的请求，
该 step 计算的正是位置 `c`，故 route 索引 `= c`。
**无未来信息**：step 组成是在线调度器的真实行为，路由是 GPU 的真实执行。

### 采集纪律

只在**纯 decode step**（`decode>0` 且无 prefill 且无 waiting）采集。
混合 step 物理上确实需要其全部 token 的并集，但把 1024-token chunk 混入 width-16
并集会使 `U` 变成 chunk 大小的函数并**伪造出饱和**——正是本实验要检验的假说。
被跳过的 step 全部计数并写入产物。

| 运行 | 引擎 step | 记录 | `missing_route` | `mixed_or_queued` |
|---|---:|---:|---:|---:|
| short cap32 | 260 | **255** | **0** | 4 |
| long cap24 | 264 | **191** | **0** | 70 |

冻结文档要求的验证点通过：`n_moe_blocks=16, num_experts=64, num_experts_per_tok=8`
（`block_type=OlmoeMoE`，走 fallback 定位）。KV 池 14.98 GiB / 122,752 tokens，
与封存 15.00 GiB 一致。

## 四、冻结预测的机械裁决

| # | 预测 | short cap32 | long cap24 |
|---|---|---|---|
| E1 | 实测闲置高于均匀零模型 | **成立**，16/16 层 | **成立**，16/16 层 |
| E2 | 宽度 ≥16 时 median idle < 0.10 | **证伪**，max 0.1875 | **证伪**，max 0.2031 |
| E3 | 99% 饱和视野 ≤ 4 步 | **证伪**，4 层从不饱和 | **证伪**，4 层从不饱和 |
| 裁决 | — | `CANDIDATE_RESIDENCY_HEADROOM` | `CANDIDATE_RESIDENCY_HEADROOM` |

**我预期的是 `NO_RESIDENCY_HEADROOM`，数据否证了我自己的预测。**
按冻结判据"某层跨视野保持闲置"，`CANDIDATE` 是正确读数，裁决不予事后改写。

均匀零模型在宽度 32 处为 $0.875^{32}=0.0139$，实测 0.031–0.1875，
即 router 集中度是独立均匀假设的 **2.2–13.5×**。负载偏斜 $C=2.17$–$4.88$。

## 五、可复现的深度结构

两个域、两个宽度下**同一结构复现**：

| 层段 | short cap32 idle | short sat99 | long cap24 idle | long sat99 |
|---|---:|---:|---:|---:|
| 0–3 浅层 | 3.1–6.2% | **2–4 步** | 9.4–12.5% | 8–16 步 |
| **4–9 中层** | **10.9–18.8%** | **16–32 / None** | **14.1–20.3%** | **32 / None** |
| 10–15 深层 | 7.8–9.4% | 8–32 步 | 10.9–15.6% | 8–32 / None |

中层同时具有**最高闲置**与**最长驻留视野**；浅层每步几乎需要全部专家。
两域中从不饱和的层都是 4 个（short: 6,7,8,9；long: 6,7,8,11）。

## 六、决定性换算：相干时间失配

赤字（引擎自报 + 独立核算一致）：

```
KV 池        122,752 tokens (14.98 GiB, 128 KiB/token)
需求         32 × 4096 = 131,072 tokens
赤字         8,320 tokens = 1.0153 GiB = 2.03 个请求
```

引擎自己打印 `Maximum concurrency for 4,096 tokens per request: 29.97x`，
与 2.03 赤字独立吻合。

每层专家字节 $=64\times3\times2048\times1024\times2=0.75$ GiB，16 层共 12.0 GiB。

| 预算 | short cap32 | long cap24 |
|---|---|---|
| 瞬时闲置（W=1） | 1.1836 GiB = **116.6%** 赤字 | 1.7695 GiB = **174.3%** 赤字 |
| 稳定闲置（W=32） | 0.0703 GiB = **6.9%** 赤字 | 0.0938 GiB = **9.2%** 赤字 |
| **缩水倍数** | **16.8×** | **18.9×** |

### 回收成本的 Pareto 前沿

实测 pinned H2D 带宽 **56.44 GB/s**（`measure_h2d_bandwidth.py`，12 MiB–1.2 GiB
五档，best-of-7；取乐观值，使不可行结论不因带宽保守而变容易）。
实测纯 decode step 时间 9.933 ms。

相干窗口 W 的闲置集必须每 W 步重新装入，摊销成本 $= \text{bytes}(W)/BW/W$：

| W | long 闲置 GiB | 覆盖赤字 | 摊销 ms/step | vs step |
|---:|---:|---:|---:|---:|
| 1 | 1.7695 | **174.3%** | 33.66 | **3.39×** |
| 2 | 0.9258 | 91.2% | 8.81 | **0.89×** |
| 4 | 0.5156 | 50.8% | 2.45 | 0.25× |
| 8 | 0.2930 | 28.9% | 0.70 | 0.07× |
| 32 | 0.0938 | 9.2% | 0.06 | 0.01× |

**前沿不与目标相交。** W=1 能覆盖但要付 3.39 个 step；W=2 只付 0.89 个 step
但只覆盖 91.2% 且仍需抢占；W≥4 成本可忽略但覆盖不足一半。

与被替代的代价对照：实测 `native32`（允许抢占）吞吐比保守 `safe29` 高 **17%**。
即便 W=2 的 0.89× step 开销也会把这 17% 完全吞掉，而它还消不掉赤字。

## 七、结论边界

| 字段 | 结论 |
|---|---|
| Verdict | `MEASUREMENT_ONLY`；`EXPERT_RESIDENCY_CANNOT_FUND_KV_DEFICIT` |
| Evidence type | 单 OLMoE、单 RTX 5090、原生 vLLM 同步 in-process 真实路由；两域两宽度 |
| 成立 | 闲置瞬时量覆盖赤字但相干时间约 1 步；稳定量仅 6.9–9.2%；实测带宽下无可行窗口 |
| **对主问题的作用** | **收窄**：动作空间不含专家驻留回收，答案须来自准入/恢复顺序 |
| 关闭的范围 | **仅**"在本运行域、按 top-k 缺失分页以换取 KV 预算" |

### 明确不主张

- `U` 是**结构信号**：不等于实测 HBM 流量、不等于 fused backend 实际加载的字节、
  不等于闲置专家的字节在本引擎 allocator 下**可驱逐**。字节总量假设完全可回收，
  是**乐观上界**。
- **不否定** FluxMoE 式整层流式、offload 运行域、WiSP 的 KV→expert 反方向，
  也不否定专家侧的**计算**组织（合批、执行顺序）。本轮只关闭"用专家显存换 KV 预算"。
- 长域测量在宽度 24，而赤字算术用 cap32。宽度越小闲置越高，
  故该口径对回收方案**有利**——负结论因此更稳健，但严格的 cap32 长域路由**未测**。
- 引擎打印的 29.97x 是容量算术，不是可达并发实测。
- 两次运行各 1 次，**无重复**；`CANDIDATE` 裁决与深度结构在两域间复现，
  但未做同域重复，不作方差主张。
- route 采集用了 `enable_return_routed_experts`，它引入 0.02 GB CPU buffer 与
  少量拷贝；本轮**不用于任何吞吐或延迟主张**。
- 96.3% 等待占比与 17% 吞吐差来自既有封存实验，本轮未重跑。

## 八、下一项可执行工作（仅一条）

主问题的动作空间现已收窄到准入与恢复顺序。既有实测给出的最强线索是：

- 抢占停顿 96.3% 是等待，且**victim 恰好等到首个请求完成才恢复**
  （首完成 20.627 s，victim 恢复 20.751 s）；
- 过度驱逐 **122×**（释放 237 blocks 以满足约 1.9 blocks 缺口）；
- 赤字恒为 **2.03 个请求**，与策略无关。

下一步最小实验：**在长上下文域做恢复顺序的三臂静态对照**——
`native32`（引擎默认：驱逐队尾、等 KV 自然释放）、`safe29`（不抢占）、
以及一条只用当前 KV 占用与已生成进度的恢复规则。
先跑 C1–C3 判别力前置检查确认该运行点可判别，再实现规则。

不新增候选，不换主问题。

## 九、复算

```bash
# 远端（GPU）
cd /root/autodl-tmp/expert-union
bash run_session.sh /root/autodl-tmp/s3 short 32
python run_expert_union.py --domain long --cap 24 \
  --prepared-dir inputs_preparation/prepared/long \
  --output-dir /root/autodl-tmp/s4-long24 --output-tokens 192 --max-steps 600
python measure_h2d_bandwidth.py /root/autodl-tmp/s3/h2d
python adjudicate_expert_union.py --union-summary <dir>/union_summary.json \
  --output-dir <dir>/adjudication
python analyze_residency_budget.py --union-summary <dir>/union_summary.json \
  --output-dir <dir>/budget --h2d-bytes-per-s 56.44e9 --step-time-ms 9.933
```

产物 `union-results.tar.gz`，sha256 `af538c32524d25d0731f9931f47b4ca63ee9d745f61bcc0200c6fa875605865d`
（远端与本地一致）。原始 raw 未修改，未 push，未改 `docs/current/README.md`。

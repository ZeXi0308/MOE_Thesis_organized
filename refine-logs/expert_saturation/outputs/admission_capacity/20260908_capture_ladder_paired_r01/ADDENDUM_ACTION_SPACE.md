# 逐步宽度微调动作空间：不存在。捕获对齐机制族在本运行域关闭

2026-09-08 追加。零 GPU 执行，全部来自本仓库已有 100 个原生 episode 的原始 step 序列。
本文件回答 [REPORT.md](REPORT.md) 指定的唯一下一最小实验，并据其预冻结判据给出裁决。

## 裁决

**`ACTION_SPACE_ABSENT`（两个到达域均成立）。** 捕获对齐机制族在本运行域关闭。

| regime | episodes | pure steps | 本已在捕获点 | **合法可填充** | 队列为空受阻 | 需抢占（非法） |
|---|---:|---:|---:|---:|---:|---:|
| bursty | 48 | 18,446 | 0.9194 | **0.0000** | 0.0000 | 0.0030 |
| steady | 52 | 34,262 | 0.7584 | **0.0039** | 0.0771 | 0.1154 |

预冻结判据为"合法可填充 step 占比 <5% 即判动作空间不存在"。实测 0.00% 与 0.39%，
远低于阈值。100 个 episode 中**没有一个**出现正净收益。

## 为什么不存在：三重独立封堵

### 1. 大多数 step 本已对齐，没有 padding 可回收

52,708 个 pure-decode step 中，**79.4% 的宽度恰好落在捕获点上**：

| 宽度 | 出现次数 | 占比 | 位置 |
|---:|---:|---:|---|
| 1 | 16,588 | 31.47% | 捕获点 |
| 8 | 13,137 | 24.92% | 捕获点 |
| 16 | 7,354 | 13.95% | 捕获点 |
| 4 | 4,797 | 9.10% | 捕获点 |
| 12 | 2,803 | 5.32% | 桶 16 内部 |

这不是巧合：`max_num_seqs` 与常用 cap 本身就取在 2 的幂上，而稳态下宽度被 cap 钉住。
**捕获尺寸集合与自然运行宽度高度重合**，可优化的错位是少数。

### 2. 向下微调需要抢占，本身非法

steady 域有 11.54% 的 step 距下一个较低捕获点仅 1–2 个请求。但把一个正在 decode 的
请求移出该 step 等于抢占或 KV 驱逐，违反冻结不变量。这些 step 被计入但标记为**非法**，
不构成动作空间。

### 3. 向上填充时队列通常是空的

剩余机会需要"宽度低于捕获点"且"此刻有排队请求可填"同时成立。steady 域 7.71% 的
step 错位却队列为空；bursty 域该项为 0.0000——到达成组本就把宽度送到捕获点上。

## 收益天花板本身也很低

即便忽略上述全部封堵，padding 的总量也不足以支撑一个机制：

```text
全部 100 个 episode 的 pure-decode 总时间 : 311.10 s
其中可归因于捕获 padding 的部分           :  19.97 s  (6.42%)
```

而每次填充都要付一次 prefill interleave，实测税为
`3.853 + 0.008826 × prefill_tokens` ms（128-token prompt 下每请求约 4.98 ms）。
填 1 个请求的成本量级与它能省下的 padding 相当或更高，这正是 100/100 个 episode
净收益非正的原因。

## 由此收束的完整因果链

三次实验（含 56 个新 GPU episode）与两次零成本重建给出一条闭合链：

1. **成本结构真实存在。** step 成本随 CUDA-graph 捕获桶呈阶梯，桶内平坦 2.1–4.0%，
   跨桶跳 12.4–30.1%；捕获边界经耗时反推与引擎日志两源确认；w17–24 平台在两次
   独立运行间最大差 0.125 ms。
2. **它有请求级后果。** 同桶内 cap12 与 cap16 付相同每步代价（比 0.9949），
   goodput 相差 3.4%–142.6%。
3. **接纳上限无法利用它。** cap 控制宽度上界而非执行宽度；steady 域 84.5% 的 step
   目标已在捕获点而实际宽度不在。同引擎配对实验裁决 `NO_GO`（steady 变号
   −44.43%/+118.74%，bursty 稳定但未胜静态 −4.63%/−0.11%）。
4. **逐步宽度动作也无法利用它。** 本文：合法机会 ≤0.39%，收益天花板 6.42%，
   单次填充成本与收益同量级。

**系统规律：`quantised cost structure != exploitable scheduling headroom`。**
量化成本阶梯是真实、可复现、可测量的，但在非抢占接纳语义下，自然运行宽度已经
基本落在量化点上，剩余错位既不合法可动、也不足以覆盖动作成本。

这条边界的价值在于它是**建设性的**：它说明要利用此类量化成本，必须改变的是
*不变量*（允许抢占/重排 batch 组装）或*量化点本身*（改捕获集合），而不是在
接纳侧增加任何控制器、档位或 predictor。

## 明确不外推

- 只覆盖 OLMoE-1B-7B / 单张 RTX 5090 / vLLM 0.26.0 / BF16 / `max_num_seqs=32` /
  固定 128 prompt + 128 output / 32 条重复文本 / 两个到达域。
- **不否定**：允许抢占或 batch 重排的动作；修改捕获集合的工程手段；offload 或
  高 HBM 压力运行域；长/变长上下文；EP/多卡；其他引擎或 kernel 后端。
- **不否定**接纳调度这一整个问题族，也不否定专家感知调度——U/C 专家信号在这三轮
  中从未被测试，仍为 `UNRUN`。
- 本文是存在性计数与有界算术，**不是**策略执行结果。它给出的是收益上界。
- `already_aligned` 高不等于"引擎已最优"；它只说明**接纳侧**没有剩余对齐动作。

## 停止与下一步

按 AGENTS.md 的判死规则，这里判死的是**精确的 formulation**：
`capture-alignment via admission-side actions, under non-preemptive semantics,
in the resident/low-HBM-pressure single-GPU regime`。

不再更换档位、阈值、反馈规则、预测器或文本抢救这条链。不实现第三个 selector。

下一轮应换问题，而不是继续加机制。基于本轮排除的内容，剩余未测且有现实依据的方向是：
**在动作侧改变不变量**（如 batch 组装期重排，需真实实现并计入完整成本），
或**换运行域**（offload / HBM 压力 / 长上下文，此时暴露路径与稀缺资源都不同）。
两者都需要新的存在性证据，不能从本轮数据推得。

## 复跑

```bash
cd refine-logs/expert_saturation/outputs/admission_capacity/20260908_capture_ladder_paired_r01
python3 check_width_nudge_space.py \
  --results-dir gpu_results \
  --results-dir ../20260908_step_cost_surface_r01/next_experiment/gpu_results \
  --results-dir ../20260906_native_knee_r01/gpu_results \
  --results-dir ../20260906_native_knee_r01/policy_probe/gpu_results \
  --output-dir <新目录>
```

输出路径须不存在。原始 cell 未修改，未 push，未改 `docs/current/README.md`。

| 结束字段 | 边界 |
|---|---|
| Verdict | `ACTION_SPACE_ABSENT`；capture-alignment 经由接纳侧动作的 formulation 关闭 |
| Evidence type | `OBSERVATIONAL_POSTHOC_ACTION_SPACE_EXISTENCE_CHECK`；100 episodes、52,708 pure steps、零新执行 |
| 关键量 | 合法可填充 0.00%/0.39%；本已对齐 79.4%；padding 占 pure decode 时间 6.42% |
| Failure category | 动作空间不存在（合法性 + 稀缺性 + 成本三重封堵），非实验无效、非问题族判死 |
| Not measured | 抢占/重排动作、修改捕获集合、offload/HBM 压力域、长上下文、EP、第二模型、U/C |
| Reopen | 新运行域中量化错位同时具备规模、合法可动性与超过动作成本的收益 |
| Claim ceiling | 一条有界且可复现的系统规律：量化成本结构不等于可利用的调度空间 |

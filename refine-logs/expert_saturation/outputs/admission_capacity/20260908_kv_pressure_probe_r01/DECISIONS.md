# KV 稀缺域存在性探针（执行前冻结）

2026-09-08。承接 [`ROUND_20260908_SUMMARY.md`](../ROUND_20260908_SUMMARY.md) 的收束：
capture-alignment 经由接纳侧动作的 formulation 已关闭，唯一有依据的下一步是**换运行域**。
冻结时本实验 GPU 执行数为 0。

## 为什么必须换运行域

对此前全部 100 个 episode 的原始 step 序列做的统计给出一个此前未被记录的事实：

```text
scheduler steps          54,622
总抢占次数                     0
总 KV 调整次数                 0
达到的最大上下文长度        255 tokens
```

**KV 从未是稀缺资源。** `max_model_len=256` 下每请求仅占 32 MiB，32 个请求共 1 GiB，
而 KV 预算约 8.5 GiB（利用率 12%）。在这个运行域里，接纳上限只能影响 batch 宽度，
无法影响任何内存决策——"准入调度"退化成了"调 batch 宽度"。三轮实验的负结果因此
只覆盖了一个**稀缺资源缺失**的运行域。

## MoE 特有的结构性张力（本轮的机制动机）

这不是人为制造的约束，而是 MoE serving 的结构后果：

| 量 | OLMoE-1B-7B | 等算力密集模型（~1B） |
|---|---|---|
| HBM 权重占用 | ~7B 参数 BF16 | ~1B 参数 BF16 |
| 每 token 计算 | ~1B 激活 | ~1B |
| 留给 KV 的空间 | 显著更少 | 显著更多 |

**MoE 为 7 倍权重付 HBM，却只拿到 1 倍算力。** 本模型 KV 成本经 config 计算为
`16 层 × 16 KV heads × 128 head_dim × 2(K,V) × 2 bytes = 128 KiB/token`。
因此"专家权重挤压 KV 容量"是 MoE 调度的真实稀缺来源，密集模型在同算力下不会同样严重。

## 唯一改变：输出长度

输入身份**逐字节不变**。派生记录见 `prepared/derivation.json`：

- `workload.json` 按字节复制，`workload_sha256` 保持
  `26073969a502a5fda1c7fbfa484fe851d24dfacaff441c9f3d80da3f1593be30`；
  全部 prompt token ids、hash、document id、到达时刻与 09-06 冻结 cohort 相同。
- `config.json` 仅 `output_tokens` `128 → 1792` 及 SLO 改变，其余键经断言相等。

于是每请求最终上下文 `128 + 1792 = 1920` tokens，32 请求峰值 KV 需求 **7.5 GiB**。

## 压力配置与预期

| 引擎 | `gpu_memory_utilization` | 估算 KV 容量 | 满长并发上限 | 相对 `max_num_seqs=32` |
|---|---:|---:|---:|---|
| 压力臂 | 0.60 | ~5.3 GiB | ~21 请求 | **必然抢占** |
| 对照臂 | 0.70 | ~8.5 GiB | ~36 请求 | 预期不抢占 |

估算基于权重约 13.8 GiB；**引擎实际分配值由 `kv_capacity.json` 落盘核实，
不使用估算值作为结论依据**。若实际容量与估算差异导致压力臂不抢占，记录为运行域
未达成，按下述规则调整而非改写结论。

## 本探针只回答一个问题

> 在 KV 稀缺域中，原生引擎是否真的抢占；若抢占，其重算成本与请求级后果有多大？

**这是存在性与成本测量，不是策略实验。** 本轮不引入任何 controller、不比较策略、
不主张收益。抢占由引擎自发产生，不是我注入的非法动作；`--allow-preemption` 只把它
从失败条件改为**被测量对象**，全部抢占事件与重算 token 数逐 step 落盘。

## 执行

```bash
# 压力臂
python run_native_capacity.py --prepared-dir prepared --output-dir results/pressure \
  --caps 8,16,32 --engine-max-seqs 32 --arrival-scales 1 --repeats 1 \
  --max-model-len 2048 --gpu-memory-utilization 0.60 --allow-preemption \
  --ttft-slo-s 2.0 --tpot-slo-s 0.020 --max-cell-seconds 300

# 对照臂（同输入、同 caps，仅显存预算不同）
python run_native_capacity.py --prepared-dir prepared --output-dir results/control \
  --caps 8,16,32 --engine-max-seqs 32 --arrival-scales 1 --repeats 1 \
  --max-model-len 2048 --gpu-memory-utilization 0.70 --allow-preemption \
  --ttft-slo-s 2.0 --tpot-slo-s 0.020 --max-cell-seconds 300
```

每臂 6 个 episode（caps 8/16/32 × steady/bursty），共 12 个。压力臂先执行并回传确认，
再执行对照臂。SLO 因输出长度变化而重设（TTFT 2 s / 平均 TPOT 20 ms）；
这是运行域重标定，**不是**按结果挑选的阈值，两臂使用同一组值。

## 冻结预测

| # | 预测 | 证伪条件 |
|---|---|---|
| K1 | 压力臂 `cap=32` 出现 >0 次抢占 | 零抢占 → KV 压力未达成，运行域仍无稀缺资源 |
| K2 | 压力臂重算 token 数 >0，且可从 `computed_adjustment` 负值独立核对 | 抢占计数 >0 但重算为 0 → 会计口径错误，需先修测量 |
| K3 | 对照臂（util 0.70）抢占次数显著低于压力臂 | 两臂相同 → 差异不由 KV 预算引起 |
| K4 | 压力臂低 cap（8）抢占次数低于高 cap（32） | 不成立 → 接纳上限无法调节 KV 压力，则该动作在本域仍无效 |

**K4 是本轮唯一与调度动作相关的预测**，也是决定是否值得继续的关键。它只检验
"接纳上限能否调节 KV 压力"这一必要条件，**不检验收益**。

## 停止与解释规则

- K1 或 K2 失败：判 `REGIME_NOT_ACHIEVED` 或 `MEASUREMENT_INVALID`，先修运行域或测量，
  不解释机制、不主张任何调度结论。
- K4 失败：接纳上限在 KV 稀缺域仍无法调节稀缺资源，则本机制族在两个运行域均关闭，
  记录为更强的负结果，不再更换参数。
- K1–K4 全部通过：仅授权设计下一个**策略对照**实验，本轮不得直接主张收益。
- 任何指标接近噪声或变号：先做受控重复，不先编机制解释。
- 长输出可能触及 `max_cell_seconds`；提前终止的 episode 保留并标为未完成，
  **不得用截断吞吐与完整运行比较**。

## 证据上限

单模型、单卡、32 条重复文本、固定 128 prompt + 1792 output、无自然 EOS、
两个到达域、每格单次。不涉及 EP/多卡、第二模型、生产服务、任务质量。
**未测 U/C 专家信号增量**，该项仍 `UNRUN`。本轮不含专家感知贡献。

抢占是原生引擎行为，本轮**不主张**任何关于 vLLM 抢占策略优劣的结论，
也不主张 recompute 与 swap 的比较——只测量当前默认后端在此配置下实际发生了什么。

## 保留

两臂全部 episode 无论结果一律保留。压力臂为预声明 canonical，对照臂为运行域对照，
均不得按结果挑选或替换。原始 cell 不修改，更正写 addendum。

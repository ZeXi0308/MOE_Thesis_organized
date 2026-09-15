# 同次引擎档位对照：已执行，档位替换未超过同次最强静态点

2026-09-08；分支 `agent/publish-current-moe-code`，HEAD `2a37765` 加已有未提交准入实现。
本文件替换本目录此前的 `PREPARED / GPU UNRUN` 记录；[冻结设计](DECISIONS.md) 未在
见到结果后改动，原方案与全部原始数据保留。

**已完成 32 个 GPU episode（1024 次完整请求执行），32/32 eligible。**

## 结论

**`MEASUREMENT_ONLY`。按 [DECISIONS.md](DECISIONS.md) 的预冻结停止规则，
capture-aligned 档位替换在本运行域为 `NO_GO`：**

- **steady 变号。** forward `−44.43%`、reverse `+118.74%`（相对 legacy）。
  预冻结规则明确：变号即不主张收益，唯一下一步是受控重复，不先解释机制。
- **bursty 稳定改善 legacy 但未胜静态上包络。** `+54.62%` / `+61.38%` 相对 legacy
  可复现，但相对同次最强静态点为 `−4.63%` / `−0.11%`。按规则只构成
  **普通反馈的工程改进线索**，不构成方法成立。

四组配对无一满足"两重复均相对 old 及 static 有 ≥3% 净增益"。因此停止本运行域的
这次档位替换，不调阈值、不换文本抢救，也不据此增加专家 predictor。

## 为什么需要这次实验

它修正了同日 [`20260908_step_cost_surface_r01`](../20260908_step_cost_surface_r01/REPORT.md)
方案的两个具体设计缺陷，二者原始数据均保留：

1. **跨日比较。** 原方案把新引擎的 aligned 与 09-06 旧引擎的 legacy 比，无法区分
   档位收益、运行漂移与引擎差异。
2. **`--caps` 同时改变预热路径。** 原 runner 的 warmup 集合由 `--caps` 派生，故两个
   实验的 CUDA graph 预热形状也不同，构成混淆变量。

本轮改为：**同一引擎内**顺序执行 legacy 与 aligned 两条 feedback，**共同预热集合
`[8,12,16,24,32]`**，另加 5 个 static 锚点与 shadow，第二引擎完整反序重复。

## 执行

一张 RTX 5090（启动前 0 计算进程、0 MiB），软件栈经只读核实与封存记录逐项一致
（python 3.12.3 / vLLM 0.26.0 / torch 2.11.0+cu130 / CUDA 13.0 / transformers 5.15.1），
模型 `OLMoE-1B-7B-0924@6d84c48581ece794365f2b8e9cfb043c68ade9c5`、BF16。

执行包 `execution.tar.gz` 本地与远端 sha256 均为 `8fb235b3502c...`；回传归档
sha256 逐位一致（forward `6c9ece20ab3e...`、reverse `24d4534d1d48...`）。按合同
**forward 回传并确认 16 个 cell 完整可读后才启动 reverse**，一次调用只起一个引擎。

两条 ladder 的动作轨迹确实分离，非同一策略的重复标注：

| cell | regime | ladder | 实际应用目标序列 |
|---|---|---|---|
| forward/cell-012 | steady | legacy | `[16, 12, 8, 12, 8, 12, 8, 12]` |
| forward/cell-014 | steady | aligned | `[24, 16, 8, 16, 8]` |
| forward/cell-013 | bursty | legacy | `[16, 12, 8, 12]` |
| forward/cell-015 | bursty | aligned | `[24, 16, 8, 16, 8]` |

## 结果

goodput 单位为同时满足 TTFT ≤200 ms 与请求平均 TPOT ≤9 ms 的请求数 / episode 墙钟。

| engine | regime | legacy | aligned | shadow | 最强静态 | aligned vs legacy | aligned vs 静态 |
|---|---|---:|---:|---:|---|---:|---:|
| forward | steady | 2.4588 | 1.3664 | 1.5346 | 2.7779 (cap16) | **−44.43%** | −50.81% |
| reverse | steady | 2.1829 | 4.7748 | 1.9320 | 3.0764 (cap24) | **+118.74%** | +55.21% |
| forward | bursty | 7.7964 | 12.0550 | 12.5151 | 12.6401 (cap32) | +54.62% | −4.63% |
| reverse | bursty | 7.8094 | 12.6030 | 12.5242 | 12.6169 (cap24) | +61.38% | −0.11% |

静态锚点自身在 steady 也不稳定，最优点在两引擎间从 cap16 移到 cap24：

| engine | regime | cap8 | cap12 | cap16 | cap24 | cap32 |
|---|---|---:|---:|---:|---:|---:|
| forward | steady | 2.025 | 1.991 | **2.778** | 2.281 | 2.702 |
| reverse | steady | 2.035 | 1.987 | 2.418 | **3.076** | 2.324 |
| forward | bursty | 2.221 | 5.005 | 12.125 | 12.578 | **12.640** |
| reverse | bursty | 2.252 | 5.025 | 12.315 | **12.617** | 12.573 |

bursty 静态高度稳定（cap16/24/32 均 12.1–12.6），steady 则在 1.99–3.08 间大幅波动。
**steady 的变号发生在一个静态排名本身就不稳定的运行域内**，这限制了任何 steady 结论。

reverse steady aligned 的 TTFT p50 为 212.9 ms，已超过 200 ms 主 SLO；它的 14 个
达标请求来自分布尾部而非中位，进一步说明该格不宜作为收益证据。

## 与同日成本重建的关系

同日重建（44 个已封存 episode，零新执行）建立的成本阶梯本身不受本轮否证，并在
本轮新引擎上继续成立；被本轮否证的是"用档位对齐去利用它"这一动作。

重建同时给出了失败的机制解释，本轮数据与之一致：**非抢占接纳 cap 只限制新接纳，
已有请求继续 decode，故实际执行宽度由到达/完成/cap 三者平衡决定，而非由档位决定。**
在 steady 域，目标已在捕获点而实际宽度不在的 step 占比高达 `0.845`。bursty 的对齐
（`0.006`）来自到达本身成组，不是档位的功劳——这正解释了为什么 bursty 的改善稳定
却仍无法越过静态点。

## 明确边界

- 单模型、单卡、32 条重复文本、固定 128 prompt / 128 output token、无自然 EOS、
  有限 episode；host 交付计时；非生产部署或稳态容量。
- "最强静态点"是同次探索性 hindsight 上包络，**不是** action Oracle，也不是可在线
  选择的策略。
- 本轮不含 padding 的硬件级归因、full-request Oracle、第二模型、EP/NCCL 或多卡证据。
- **未测 U/C 或任何专家信号增量**，该项仍为 `UNRUN`。本轮不含专家感知贡献。
- bursty 相对 legacy 的 +55%/+61% 是对一条**已被判 `CONDITIONAL_NO_GO` 的规则**的
  改善，不能当作相对强基线的收益。

## 唯一下一最小实验

**已执行，结果为 `ACTION_SPACE_ABSENT`。** 见
[`ADDENDUM_ACTION_SPACE.md`](ADDENDUM_ACTION_SPACE.md)：在 100 个 episode、52,708 个
pure-decode step 上，合法可填充机会仅 0.00%（bursty）/ 0.39%（steady），
低于预冻结的 5% 判据；79.4% 的 step 本已落在捕获点上；padding 只占 pure-decode
总时间的 6.42%，与单次填充的 prefill 税同量级。100/100 个 episode 净收益非正。

因此 `capture-alignment via admission-side actions` 这一 formulation 在本运行域关闭，
不再更换档位、阈值、反馈规则或 predictor。下一轮换问题：在动作侧改变不变量
（batch 组装期重排，需真实实现并计入完整成本），或换运行域（offload / HBM 压力 /
长上下文）。两者都需要新的存在性证据。

## 复跑与保留

```bash
cd refine-logs/expert_saturation/outputs/admission_capacity/20260908_capture_ladder_paired_r01
python3 analyze_paired.py --results-dir gpu_results --plans-dir plans --output-dir <新目录>
```

`gpu_results/{forward,reverse}/` 保留全部 raw cell、warmup raw、逐 cell 配置与检查、
stdout/stderr 与退出状态。forward 为预声明 canonical，reverse 为受控重复，
两者均未按结果挑选或替换。原始封存数据未修改，未 push，未改 `docs/current/README.md`。
GPU 任务已自然退出，设备已释放（0 MiB）。

| 结束字段 | 本轮边界 |
|---|---|
| Verdict | `MEASUREMENT_ONLY`；capture-aligned 档位替换在本运行域 `NO_GO` |
| Evidence type | `REQUEST_LEVEL` / 原生 vLLM 同步 in-process；32/32 eligible |
| Measured | 同引擎两条 ladder 的真实接纳/宽度/排队/完成差异及请求级结果 |
| Not measured | 单步宽度动作、U/C 增量、任务质量、自然 EOS、第二模型、EP、Oracle |
| Strongest baseline | 同次 static8/12/16/24/32 上包络 + legacy 反馈 + shadow |
| Failure category | 动作层失败：cap 控制上界而非执行宽度；非实验无效，非机制族判死 |
| Reopen | 新自然运行域中出现可执行且可重复的档位收益空间 |
| Claim ceiling | 一个有边界的负结果 + 一条仍成立的量化成本规律；**无方法 GO** |

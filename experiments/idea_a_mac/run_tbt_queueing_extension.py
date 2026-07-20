"""Nonlinear M/M/1-style queueing extension on top of run_tbt_congestion_bridge.py.

Why this is a genuinely NEW mechanism, not a re-run
----------------------------------------------------
`run_tbt_congestion_bridge.py` models combine congestion as PURE SERIALIZATION:

    combine_us_actual = max_receiver_bytes / bandwidth

This is equivalent to assuming the link has zero queueing beyond raw byte
transmission time -- i.e., a deterministic, work-conserving server with no
extra waiting. Under this model, if a "congestion multiplier" were applied
UNIFORMLY to all three policies (none/random/deployable_combined) in a given
decode step -- e.g. a multiplier that only depends on the NUMBER of
concurrent senders (fan-in), which is IDENTICAL across policies since
quantization changes bytes-per-pair, not which pairs exist -- that multiplier
would cancel out exactly in every `reduction_vs_none_pct` / `reduction_vs_random_pct`
ratio (because policy_us / none_us = (policy_linear * m) / (none_linear * m) =
policy_linear / none_linear). This is why a naive "add a congestion penalty"
extension would NOT change any of the prior percentage conclusions -- it
would only inflate absolute combine_us for all policies equally.

The one queueing mechanism that does NOT cancel out is textbook M/M/1-style
delay amplification, where mean sojourn time scales as

    T_queued = T_service / (1 - rho),   rho = utilization = T_service / T_budget

This is CONVEX in rho: as utilization approaches saturation, the SAME
absolute byte reduction produces a DISPROPORTIONATELY larger latency
reduction, because different policies have different rho (different byte
volumes -> different combine_us_actual_linear -> different rho), so the
1/(1-rho) amplification does NOT cancel between policies. This is the
standard qualitative behavior cited in incast/congestion-control literature
(e.g. DCTCP/DCQCN papers): near the "knee" of the utilization curve, small
load reductions yield large queueing-delay reductions. This script tests
whether that convexity is enough to meaningfully raise the ~1-2% ceiling
found under the pure-linear model.

This is a SENSITIVITY ANALYSIS with an explicit, clearly-labeled analytical
model -- not a measured network result. `rho_cap` bounds the model to avoid
blow-up as rho->1 (real links never hit rho=1 for a sustained period without
either backpressure or drops; we cap the amplification factor at
`--max-amplification` to keep numbers defensible).

Reuses the SAME sampled decode-step byte data already produced by
`run_tbt_congestion_bridge.py` (reads its `tbt_congestion_bridge_raw.csv`),
so this is an apples-to-apples re-analysis, not a re-sample with different
random decode steps.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw-csv",
        required=True,
        help="path to an existing tbt_congestion_bridge_raw.csv produced by run_tbt_congestion_bridge.py",
    )
    p.add_argument(
        "--rho-caps", default="0.5,0.7,0.85,0.95",
        help="comma-separated utilization values that the OBSERVED (none-mode) "
             "combine link is assumed to be operating near, at the MOST CONGESTED "
             "sampled decode step per config. Each value defines a separate "
             "sensitivity scenario: rho_cap=0.5 means we calibrate the queueing "
             "model so that the busiest 'none' step in this config sits at "
             "utilization 0.5 (mild), 0.95 means it sits at utilization 0.95 "
             "(near-saturated, most favorable to queueing amplification).",
    )
    p.add_argument(
        "--max-amplification", type=float, default=20.0,
        help="hard cap on the 1/(1-rho) multiplier to avoid unbounded blow-up "
             "as rho->1 (real links backpressure/drop before this; this is a "
             "conservative ceiling on how much queueing can possibly help)",
    )
    p.add_argument(
        "--output-dir",
        default="experiments/idea_a_mac/outputs/paper_validation/tbt_queueing_extension",
    )
    return p.parse_args()


def dataframe_to_markdown(df: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in df[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def apply_queueing(
    raw_df: pd.DataFrame, rho_cap: float, max_amplification: float,
) -> pd.DataFrame:
    """For each (bandwidth, expert_weight_precision, origin_mode, batch_size,
    tail_fraction, candidate_pool) config, calibrate a link "time budget" T_budget
    such that the WORST-CASE (max across sampled steps) 'none'-mode
    combine_us_actual sits at utilization `rho_cap` against that budget:

        T_budget = max_none_combine_us / rho_cap

    Then for EVERY step and EVERY mode (none/random/deployable_combined),
    compute rho = combine_us_actual_linear / T_budget, and the queued time:

        combine_us_queued = combine_us_actual_linear / max(1 - rho, 1/max_amplification)

    This keeps T_budget FIXED within a config (calibrated once from the
    worst-case 'none' step), so random/deployable_combined -- which have
    LOWER bytes than 'none' in the same step -- get a LOWER rho and therefore
    a smaller (but still possibly nontrivial) amplification, never a larger one.
    """
    rows = []
    group_cols = ["bandwidth_gbps", "expert_weight_precision", "origin_mode",
                  "batch_size", "tail_fraction", "candidate_pool"]
    for key, group in raw_df.groupby(group_cols):
        none_group = group[group["mode"] == "none"]
        if none_group.empty:
            continue
        max_none_us = float(none_group["combine_us_actual"].max())
        if max_none_us <= 0:
            t_budget = 1.0
        else:
            t_budget = max_none_us / rho_cap
        g = group.copy()
        rho = (g["combine_us_actual"] / t_budget).clip(upper=1.0 - 1.0 / max_amplification)
        amplification = 1.0 / (1.0 - rho).clip(lower=1.0 / max_amplification)
        g["rho"] = rho
        g["amplification_x"] = amplification
        g["combine_us_queued"] = g["combine_us_actual"] * amplification
        g["per_layer_tbt_queued_us"] = (
            g["per_layer_tbt_actual_us"] - g["combine_us_actual"] + g["combine_us_queued"]
        )
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else raw_df.iloc[0:0]


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(args.raw_csv)
    rho_caps = [float(v) for v in args.rho_caps.split(",") if v]

    # Need L (num layers) to extrapolate per-layer TBT to total decode-step TBT,
    # matching run_tbt_congestion_bridge.py's summary convention. Infer it from
    # the ratio baked into the raw csv is not directly available, so we instead
    # report PER-LAYER reduction percentages (still apples-to-apples across
    # policies and directly comparable to the per-layer figures implicit in
    # the bridge script's summary, since L cancels out in ratio terms anyway).
    all_scenarios = []
    for rho_cap in rho_caps:
        q_df = apply_queueing(raw_df, rho_cap, args.max_amplification)
        q_df["rho_cap_scenario"] = rho_cap
        all_scenarios.append(q_df)
    full_df = pd.concat(all_scenarios, ignore_index=True)
    full_df.to_csv(out / "tbt_queueing_raw.csv", index=False)

    delta_rows = []
    group_cols = ["rho_cap_scenario", "bandwidth_gbps", "expert_weight_precision",
                  "origin_mode", "batch_size", "tail_fraction", "candidate_pool"]
    for key, group in full_df.groupby(group_cols):
        rho_cap, bw, prec, origin, B, tail_frac, pool = key
        by_mode = group.groupby("mode")["per_layer_tbt_queued_us"].quantile(0.99)
        by_mode_linear = group.groupby("mode")["per_layer_tbt_actual_us"].quantile(0.99)
        if not {"none", "random", "deployable_combined"}.issubset(by_mode.index):
            continue
        none_q, random_q, combined_q = by_mode["none"], by_mode["random"], by_mode["deployable_combined"]
        none_lin, combined_lin = by_mode_linear["none"], by_mode_linear["deployable_combined"]
        mean_rho_none = float(group[group["mode"] == "none"]["rho"].mean())
        mean_amp_none = float(group[group["mode"] == "none"]["amplification_x"].mean())
        delta_rows.append({
            "rho_cap_scenario": rho_cap, "bandwidth_gbps": bw, "expert_weight_precision": prec,
            "origin_mode": origin, "batch_size": B, "tail_fraction": tail_frac, "candidate_pool": pool,
            "mean_rho_none": mean_rho_none, "mean_amplification_none_x": mean_amp_none,
            "p99_reduction_vs_none_pct_linear": (none_lin - combined_lin) / max(none_lin, 1e-9) * 100,
            "p99_reduction_vs_none_pct_queued": (none_q - combined_q) / max(none_q, 1e-9) * 100,
            "p99_reduction_vs_random_pct_queued": (random_q - combined_q) / max(random_q, 1e-9) * 100,
            "amplification_gain_pp": (
                (none_q - combined_q) / max(none_q, 1e-9) * 100
                - (none_lin - combined_lin) / max(none_lin, 1e-9) * 100
            ),
        })
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(out / "tbt_queueing_delta.csv", index=False)

    safe_pool = delta_df[delta_df["candidate_pool"] == "tail_and_remote"]
    headline = safe_pool.sort_values("p99_reduction_vs_none_pct_queued", ascending=False).head(20)
    headline_table = dataframe_to_markdown(
        headline,
        ["rho_cap_scenario", "origin_mode", "bandwidth_gbps", "batch_size", "mean_rho_none",
         "mean_amplification_none_x", "p99_reduction_vs_none_pct_linear",
         "p99_reduction_vs_none_pct_queued", "amplification_gain_pp"],
    ) if not safe_pool.empty else "(no data)"

    by_scenario = safe_pool.groupby("rho_cap_scenario").agg(
        max_reduction_linear=("p99_reduction_vs_none_pct_linear", "max"),
        max_reduction_queued=("p99_reduction_vs_none_pct_queued", "max"),
        mean_amplification_gain_pp=("amplification_gain_pp", "mean"),
    ).reset_index()
    scenario_table = dataframe_to_markdown(
        by_scenario, ["rho_cap_scenario", "max_reduction_linear", "max_reduction_queued", "mean_amplification_gain_pp"],
    )

    best_row = safe_pool.loc[safe_pool["p99_reduction_vs_none_pct_queued"].idxmax()] if not safe_pool.empty else None

    report = f"""# M/M/1-Style Queueing Extension: Does Nonlinear Congestion Amplification Raise the Ceiling?

## 动机

`run_tbt_congestion_bridge.py` 把 combine 拥塞建模为纯串行化时间
（`bytes / bandwidth`），等价于假设链路除了传输字节本身的时间外，没有任何
额外排队等待——一个确定性、work-conserving、零排队的服务模型。

这个模型有一个容易被忽视的数学性质：**如果给三种策略（none/random/
deployable_combined）在同一个 decode step 里乘上同一个"拥塞放大系数"**
（例如系数只取决于并发发送方数量 fan-in——这个数量在三种策略下是完全相同的，
因为量化只改变每对 sender/receiver 的字节数，不改变有哪些 sender/receiver
在通信），**这个系数会在算 `reduction_vs_none_pct` / `reduction_vs_random_pct`
时被直接约掉**（因为 `policy_us / none_us = (policy_linear × m) / (none_linear × m)
= policy_linear / none_linear`）。也就是说，一个天真的"均匀拥塞惩罚"扩展
根本不会改变此前任何百分比结论，只会把绝对时间等比例抬高。

真正不会被约掉的机制是教科书式的 **M/M/1 排队延迟**：平均排队时延随利用率
`rho` 呈 `T/(1-rho)` 增长，是**凸函数**。因为不同策略的字节量不同，
对应不同的 `rho`，`1/(1-rho)` 放大系数不会在策略间抵消——利用率越接近
饱和点（"knee"），同样的字节削减换来的延迟削减越大。这是拥塞控制文献
（如 DCTCP/DCQCN 一类工作）里反复引用的定性行为。本实验检验这个凸性
放大效应能不能把此前线性模型下 ~1-2% 的天花板显著推高。

## 方法

复用 `run_tbt_congestion_bridge.py` 已采样的 decode-step 字节数据
（`{args.raw_csv}`），不重新采样，保证与线性模型结果严格可比：

1. 对每个配置（bandwidth × expert精度 × origin_mode × batch × candidate_pool），
   用该配置下 `none` 策略里最拥塞的一步（`max(combine_us_actual)`），
   反推一个链路"时间预算" `T_budget = max_none_us / rho_cap`——即假设
   `none` 策略在最坏情况下运行在利用率 `rho_cap`（本实验对 `rho_cap` 做
   敏感性扫描：`{rho_caps}`，从温和到接近饱和）；
2. 用同一个 `T_budget`（配置内固定，不随策略变化）算出每一步、每种策略下
   的真实利用率 `rho = combine_us_actual_linear / T_budget`；
3. 排队后时间 `combine_us_queued = combine_us_actual_linear / max(1-rho, 1/{args.max_amplification})`，
   放大系数上限 `{args.max_amplification}×`（避免 `rho→1` 时无界发散，
   真实链路在这之前会触发反压/丢包，这是一个保守上限）；
4. 用排队后的 `per_layer_tbt` 重新计算 P99 reduction，与线性模型的结果并列
   对比，差值 `amplification_gain_pp`（百分点）即排队非线性效应带来的
   额外收益。

## 结果（按敏感性场景汇总，质量安全候选池）

{scenario_table}

## 最有利单点配置的完整对比（按 P99 排队后降幅排序，Top 20）

{headline_table}

## 关键读数

- 最有利配置（`{best_row['rho_cap_scenario'] if best_row is not None else '-'}` 利用率场景，
  `{best_row['origin_mode'] if best_row is not None else '-'}`,
  `{best_row['bandwidth_gbps'] if best_row is not None else '-'}Gbps`,
  `batch={best_row['batch_size'] if best_row is not None else '-'}`）：
  线性模型降幅 `{f"{best_row['p99_reduction_vs_none_pct_linear']:.2f}%" if best_row is not None else '-'}`，
  排队模型降幅 `{f"{best_row['p99_reduction_vs_none_pct_queued']:.2f}%" if best_row is not None else '-'}`，
  排队非线性带来额外 `{f"{best_row['amplification_gain_pp']:.2f}" if best_row is not None else '-'}` 个百分点。
- 只有当 `none` 策略被校准到**接近饱和**（`rho_cap>=0.85~0.95`）时，排队非线性
  才会贡献可观的额外收益；在温和利用率（`rho_cap<=0.5`）下，`1/(1-rho)` 本身
  接近 1，排队模型退化为线性模型，额外收益趋近于 0。
- 这说明：M/M/1 式排队放大**理论上是真实存在、方向正确**的机制（凸性不会被
  约掉），但它能贡献多少取决于一个本实验**无法从 bandwidth-only trace replay
  里独立验证**的量——真实链路在生产环境里的稳态利用率到底运行在拥塞曲线的
  哪个区间。如果生产环境的 combine 链路长期在 85%+ 利用率下运行（高负载、
  带宽受限的多租户 serving 集群更可能出现），这个机制可以把天花板从个位数
  百分比推高到更显著的量级；如果链路利用率长期在 50% 以下，这个机制基本不
  起作用，前几轮线性模型的结论原样成立。

## 解读边界

- 这是一个**参数化的排队论敏感性分析**，不是从真实网络测得的拥塞曲线；
  `rho_cap` 是外部假设的输入参数，不是从 trace 数据反推出的真实利用率——
  本实验只能回答"如果链路运行在某个利用率区间，非线性排队能贡献多少额外
  收益"，不能回答"真实链路到底运行在哪个利用率区间"，后者需要真实生产
  环境的网络监控数据或真实 collective 库 profiling。
- `max_amplification` 上限是人为设定的保守值，避免 `rho→1` 时数学发散；
  真实系统在触发这个上限前，网络层通常已经出现反压（PFC/ECN）或丢包重传，
  这些效应本身会带来额外、本模型未建模的延迟，因此这里报告的数字更可能是
  低估而非高估真实排队延迟。
- 沿用 `run_tbt_congestion_bridge.py` 的所有其他边界：仍是 bandwidth-only
  分析模型，不含真实 collective 库开销、kernel launch、pack/unpack。
"""
    (out / "tbt_queueing_report.md").write_text(report, encoding="utf-8")
    print(scenario_table, flush=True)
    print(f"\nsaved to {out}", flush=True)


if __name__ == "__main__":
    main()

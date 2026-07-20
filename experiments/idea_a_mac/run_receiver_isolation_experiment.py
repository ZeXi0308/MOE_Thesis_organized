"""Isolate the true value of "hotspot port selection" from the "prefer remote
pairs" confound found in run_ep_congestion_sim.py / run_quality_safe_congestion_frontier.py.

Background / motivation
------------------------
In the existing `tail_budget_profile_ports` / `tail_budget_scheduler_receiver` /
`tail_budget_greedy_ports` policies, the scoring function adds a large bonus to
every remote (inter-node) candidate pair:

    scores = scores + remote.astype(float) * (scores.max() + 1.0)

This means, for any non-zero budget, ALL remote candidates outrank ALL local
candidates regardless of how "hot" a specific sender/receiver port actually is.
The reported ~22-25% extra bottleneck saving vs. `tail_budget_random` could
therefore be driven almost entirely by "prefer remote over local", not by
genuine hotspot-port identification.

This script removes that confound. It fixes the candidate pool to
`tail-rank AND remote` (the SAME pool for every policy in this script), and
only varies WHICH remote pairs receive the limited INT4 budget:

  - `hot`    : pairs with the highest current sender/receiver load (real signal)
  - `cold`   : pairs with the lowest current sender/receiver load (reverse control)
  - `random` : uniformly random pairs, repeated over many seeds for a CI

If `hot` still beats `random` by a wide, stable margin under this controlled
pool, receiver/port-awareness has independent value beyond "compress remote
traffic first". If `hot` ~= `random` (within CI) and `cold` is also close,
the effect in the earlier reports was almost entirely the remote-preference
confound, and the paper's receiver-aware claim needs to be rewritten.

This is still a bandwidth-only analytical trace replay, not measured GPU
latency, collective scheduling, or queueing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoConfig

from run_ep_congestion_sim import add_placement, concurrent_scenario, dataframe_to_markdown


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument(
        "--test-routes",
        default="experiments/idea_a_mac/outputs/paper_validation/olmoe_signal_comparison_n32/test_routes.csv",
    )
    p.add_argument("--ep-size", type=int, default=8)
    p.add_argument("--gpus-per-node", type=int, default=4)
    p.add_argument("--num-jobs", default="4,8,16")
    p.add_argument("--origin-modes", default="balanced,hotspot")
    p.add_argument("--budget-fractions", default="0.25,0.5,0.75,1.0")
    p.add_argument("--inter-node-gbps", type=float, default=200.0)
    p.add_argument("--num-random-seeds", type=int, default=30)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        default="experiments/idea_a_mac/outputs/paper_validation/receiver_isolation",
    )
    return p.parse_args()


def layer_bottleneck_us(rows: pd.DataFrame, hidden_size: int, gpus_per_node: int, inter_node_gbps: float) -> dict[str, float]:
    """Compute the same bottleneck proxy metric as run_ep_congestion_sim.summarize,
    restricted to remote traffic (the quantity receiver-aware policies target)."""
    remote = rows[
        (rows["sender_rank"] // gpus_per_node) != (rows["receiver_rank"] // gpus_per_node)
    ].copy()
    if remote.empty:
        return {"sum_layer_bottleneck_us": 0.0, "p99_layer_receiver_bytes": 0.0,
                "mean_layer_receiver_imbalance": 1.0}
    ingress = remote.groupby(["layer", "receiver_rank"])["payload_bytes"].sum()
    egress = remote.groupby(["layer", "sender_rank"])["payload_bytes"].sum()
    ingress_max = ingress.groupby("layer").max()
    egress_max = egress.groupby("layer").max()
    layer_bottleneck = pd.concat([ingress_max, egress_max], axis=1).max(axis=1)
    bw_bytes_per_us = inter_node_gbps * 1e9 / 8 / 1e6
    ingress_mean = ingress.groupby("layer").mean()
    imbalance = ingress_max / ingress_mean.clip(lower=1.0)
    return {
        "sum_layer_bottleneck_us": float((layer_bottleneck / bw_bytes_per_us).sum()),
        "p99_layer_receiver_bytes": float(ingress.quantile(0.99)),
        "mean_layer_receiver_imbalance": float(imbalance.mean()),
    }


def build_candidate_pool(scenario: pd.DataFrame, top_k: int, gpus_per_node: int) -> pd.Series:
    """Fixed candidate pool shared by hot/cold/random: tail-rank AND inter-node."""
    tail_mask = scenario["rank"].astype(int) > (top_k - max(1, top_k // 2))
    remote_mask = (scenario["sender_rank"] // gpus_per_node) != (scenario["receiver_rank"] // gpus_per_node)
    return tail_mask & remote_mask


def score_by_current_load(layer_rows: pd.DataFrame, candidate_index: pd.Index, gpus_per_node: int) -> pd.Series:
    """Score each candidate pair by max(current sender load, current receiver load)
    among REMOTE traffic in THIS LAYER only (strict per-layer "current window"
    oracle, not a cross-layer aggregate and not a stale offline profile).
    ``layer_rows`` must already be restricted to a single layer -- scoring
    across layers would leak future/other-layer information into a per-layer
    budget decision and inflate the apparent advantage of "hot"."""
    remote_mask = (layer_rows["sender_rank"] // gpus_per_node) != (layer_rows["receiver_rank"] // gpus_per_node)
    remote_rows = layer_rows[remote_mask]
    sender_load = remote_rows.groupby("sender_rank").size().astype(float)
    receiver_load = remote_rows.groupby("receiver_rank").size().astype(float)
    cand = layer_rows.loc[candidate_index]
    s = cand["sender_rank"].map(sender_load).fillna(0.0)
    r = cand["receiver_rank"].map(receiver_load).fillna(0.0)
    return pd.Series(np.maximum(s.to_numpy(), r.to_numpy()), index=candidate_index)


def assign_isolation_bytes(
    scenario: pd.DataFrame,
    top_k: int,
    gpus_per_node: int,
    hidden_size: int,
    mode: str,
    fraction: float,
    seed: int,
) -> pd.DataFrame:
    rows = scenario.copy()
    # Baseline for every pair in this experiment is uniform FP8 (1 byte/elem);
    # the limited budget upgrades selected remote tail pairs to INT4 (0.5 byte/elem).
    rows["bytes_per_element"] = 1.0
    candidate_mask = build_candidate_pool(rows, top_k, gpus_per_node)

    selected: list[int] = []
    for layer, layer_rows in rows.groupby("layer"):
        candidates = layer_rows[candidate_mask.loc[layer_rows.index]]
        if candidates.empty:
            continue
        budget = int(round(len(candidates) * fraction))
        if budget <= 0:
            continue
        if mode == "all_candidates":
            chosen = candidates.index
        elif mode == "random":
            rng = np.random.default_rng(seed)
            n = min(budget, len(candidates))
            chosen = rng.choice(candidates.index.to_numpy(), size=n, replace=False)
        elif mode in ("hot", "cold"):
            scores = score_by_current_load(layer_rows, candidates.index, gpus_per_node)
            ascending = mode == "cold"
            chosen = scores.sort_values(ascending=ascending).index[:budget]
        else:
            raise ValueError(f"unknown mode: {mode}")
        selected.extend(list(chosen))

    rows.loc[selected, "bytes_per_element"] = 0.5
    rows["payload_bytes"] = rows["bytes_per_element"] * hidden_size
    rows.attrs["hidden_size"] = hidden_size
    return rows


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    test_routes = pd.read_csv(args.test_routes)
    cfg = AutoConfig.from_pretrained(args.model, local_files_only=args.offline)
    hidden_size = int(cfg.hidden_size)
    num_experts = int(getattr(cfg, "num_experts", getattr(cfg, "num_local_experts", 0)))
    top_k = int(getattr(cfg, "num_experts_per_tok", getattr(cfg, "num_experts_per_token", 0)))
    if num_experts <= 0 or top_k <= 0:
        raise ValueError(f"cannot resolve MoE config: num_experts={num_experts}, top_k={top_k}")

    job_counts = [int(v) for v in args.num_jobs.split(",") if v]
    origin_modes = [v.strip() for v in args.origin_modes.split(",") if v.strip()]
    fractions = [float(v) for v in args.budget_fractions.split(",") if v]

    results: list[dict[str, float | int | str]] = []
    for origin_mode in origin_modes:
        for num_jobs in job_counts:
            scenario = concurrent_scenario(test_routes, num_jobs, origin_mode, args.ep_size, num_experts)
            candidate_mask = build_candidate_pool(scenario, top_k, args.gpus_per_node)
            candidate_fraction_of_all = float(candidate_mask.mean())

            # uniform_fp8 reference (no INT4 at all)
            fp8_rows = scenario.copy()
            fp8_rows["bytes_per_element"] = 1.0
            fp8_rows["payload_bytes"] = hidden_size
            fp8_metrics = layer_bottleneck_us(fp8_rows, hidden_size, args.gpus_per_node, args.inter_node_gbps)

            for fraction in fractions:
                for mode in ("hot", "cold"):
                    policy_rows = assign_isolation_bytes(
                        scenario, top_k, args.gpus_per_node, hidden_size, mode, fraction, args.seed
                    )
                    metrics = layer_bottleneck_us(policy_rows, hidden_size, args.gpus_per_node, args.inter_node_gbps)
                    results.append({
                        "origin_mode": origin_mode, "num_jobs": num_jobs, "mode": mode,
                        "budget_fraction": fraction, "seed": -1,
                        "candidate_fraction_of_all_pairs": candidate_fraction_of_all,
                        "fp8_bottleneck_us": fp8_metrics["sum_layer_bottleneck_us"],
                        **metrics,
                    })

                # random: many seeds -> mean + CI
                random_vals = []
                for trial in range(args.num_random_seeds):
                    policy_rows = assign_isolation_bytes(
                        scenario, top_k, args.gpus_per_node, hidden_size, "random", fraction,
                        args.seed + trial,
                    )
                    metrics = layer_bottleneck_us(policy_rows, hidden_size, args.gpus_per_node, args.inter_node_gbps)
                    random_vals.append(metrics["sum_layer_bottleneck_us"])
                    results.append({
                        "origin_mode": origin_mode, "num_jobs": num_jobs, "mode": "random",
                        "budget_fraction": fraction, "seed": trial,
                        "candidate_fraction_of_all_pairs": candidate_fraction_of_all,
                        "fp8_bottleneck_us": fp8_metrics["sum_layer_bottleneck_us"],
                        **metrics,
                    })

    df = pd.DataFrame(results)
    df["bottleneck_saving_vs_fp8"] = 1.0 - df["sum_layer_bottleneck_us"] / df["fp8_bottleneck_us"].clip(lower=1e-12)
    df.to_csv(out / "receiver_isolation_raw.csv", index=False)

    # Summarize: mean/CI for random, point value for hot/cold, per (origin_mode, num_jobs, fraction)
    summary_rows = []
    for (origin_mode, num_jobs, fraction), group in df.groupby(["origin_mode", "num_jobs", "budget_fraction"]):
        hot = group[group["mode"] == "hot"]["bottleneck_saving_vs_fp8"].iloc[0]
        cold = group[group["mode"] == "cold"]["bottleneck_saving_vs_fp8"].iloc[0]
        random_vals = group[group["mode"] == "random"]["bottleneck_saving_vs_fp8"].to_numpy()
        random_mean = float(np.mean(random_vals))
        random_std = float(np.std(random_vals, ddof=1)) if len(random_vals) > 1 else 0.0
        random_ci_low = float(np.quantile(random_vals, 0.025)) if len(random_vals) > 1 else random_mean
        random_ci_high = float(np.quantile(random_vals, 0.975)) if len(random_vals) > 1 else random_mean
        hot_minus_random_z = (hot - random_mean) / max(random_std, 1e-9)
        summary_rows.append({
            "origin_mode": origin_mode,
            "num_jobs": num_jobs,
            "budget_fraction": fraction,
            "hot_saving_vs_fp8": hot,
            "cold_saving_vs_fp8": cold,
            "random_mean_saving_vs_fp8": random_mean,
            "random_ci_low": random_ci_low,
            "random_ci_high": random_ci_high,
            "hot_minus_random": hot - random_mean,
            "hot_minus_random_zscore": hot_minus_random_z,
            "hot_minus_cold": hot - cold,
            "hot_within_random_ci": bool(random_ci_low <= hot <= random_ci_high),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out / "receiver_isolation_summary.csv", index=False)

    columns = [
        "origin_mode", "num_jobs", "budget_fraction",
        "hot_saving_vs_fp8", "random_mean_saving_vs_fp8", "random_ci_low", "random_ci_high",
        "cold_saving_vs_fp8", "hot_minus_random", "hot_within_random_ci",
    ]
    table = dataframe_to_markdown(summary_df, columns)

    verdict_lines = []
    for _, row in summary_df.iterrows():
        tag = "端口感知有独立价值" if not row["hot_within_random_ci"] and row["hot_minus_random"] > 0 else \
              "端口感知无独立价值(落在random CI内)"
        verdict_lines.append(
            f"- {row['origin_mode']} / jobs={int(row['num_jobs'])} / frac={row['budget_fraction']:.2f}: "
            f"hot-random={row['hot_minus_random']:+.4f} ({tag})"
        )

    report = f"""# Receiver-Aware Isolation Experiment

## 目的

剥离 `run_ep_congestion_sim.py` / `run_quality_safe_congestion_frontier.py` 中
"优先压缩跨节点流量"这一硬编码 bonus 造成的混淆，单独检验"识别热点
sender/receiver 端口"这个信号本身是否有独立价值。

## 方法

固定候选池为 **tail-rank AND inter-node** 的 pair（对 hot/cold/random 完全相同），
只改变在这个池子内选哪些 pair 获得 INT4 预算：

- `hot`：按当前真实 remote sender/receiver 负载（max(sender_load, receiver_load)）降序选择
- `cold`：同一负载指标升序选择（反向对照）
- `random`：{args.num_random_seeds} 个随机种子重复抽样，报告均值和 95% 分位区间

baseline 为 uniform FP8（1 byte/elem），预算内 pair 升级为 INT4（0.5 byte/elem）。
瓶颈指标口径与 `run_ep_congestion_sim.summarize` 一致：只统计 remote (inter-node)
ingress/egress 的 per-layer max，再除以带宽求和。

## 配置

- model: `{args.model}`; EP=`{args.ep_size}`; GPUs/node=`{args.gpus_per_node}`
- concurrent jobs: `{job_counts}`; origin modes: `{origin_modes}`
- budget fractions (within tail∩remote pool): `{fractions}`
- inter-node bandwidth: `{args.inter_node_gbps} Gbps`
- random trials per config: `{args.num_random_seeds}`

## 结果

{table}

## 判定

{chr(10).join(verdict_lines)}

## 解读边界

- 这仍是 bandwidth-only 解析回放，不含 collective、queueing、pack/unpack、kernel。
- 若 `hot` 落在 `random` 的 95% CI 内，说明此前报告里 receiver-aware 相对 random 的
  "额外收益"主要来自 remote-vs-local 的选择，而不是"识别具体哪个端口更热"。
- 若 `hot` 稳定超出 `random` CI 且 `hot - cold` 差距明显，说明端口热度信息在
  "已经限定只压 remote"之后仍有独立、可复现的边际价值，receiver-aware 的
  claim 可以保留，但措辞必须改为"限定 remote 后进一步做端口热度选择"，
  而不是笼统的"receiver-aware 比 random 好 X%"。
"""
    (out / "receiver_isolation_report.md").write_text(report, encoding="utf-8")
    print(table, flush=True)
    print("\n".join(verdict_lines), flush=True)
    print(f"\nsaved to {out}", flush=True)


if __name__ == "__main__":
    main()

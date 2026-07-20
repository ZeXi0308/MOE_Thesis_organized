"""Test a THIRD, previously-unexamined signal for congestion-safe INT4 budgeting:
a CAUSAL, WITHIN-REQUEST sliding-window sender-load estimate, as opposed to the
two signals already tested and falsified/weakened in
`run_receiver_isolation_experiment.py` / `run_receiver_sender_decomposition.py`:

  1. cross-request offline profile (calibration corpus -> test corpus): FALSIFIED
     (Spearman ~0.39-0.50 across disjoint samples, see run_expert_popularity_stability.py)
  2. full-scenario oracle "hot" (uses the ENTIRE current job's token sequence,
     including tokens that haven't been decoded yet): NOT DEPLOYABLE, upper bound only

This script adds a THIRD candidate:

  3. causal_window: for token t in job J at layer L, score sender_rank(t) by how many
     times that same sender_rank appeared among the last `window` REMOTE tokens of the
     SAME job J at the SAME layer L, using ONLY tokens decoded strictly before t.
     This uses zero future information and zero cross-request information -- it is
     exactly the kind of signal a real serving system already has for free, because
     expert-routing decisions for already-dispatched tokens are known before combine
     for token t is scheduled.

We compare causal_window against the existing oracle "hot" (upper bound, uses future
info) and "random" (lower bound, many-seed CI), on the SAME fixed candidate pool
(tail-rank AND inter-node) and the SAME uniform-FP8 baseline / bottleneck metric as
`run_receiver_isolation_experiment.py`, so all four numbers are directly comparable.

Still a bandwidth-only analytical trace replay: no collective scheduling, queueing,
pack/unpack, or kernel launch overhead.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoConfig

from run_ep_congestion_sim import add_placement, concurrent_scenario, dataframe_to_markdown
from run_receiver_isolation_experiment import build_candidate_pool, layer_bottleneck_us


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
    p.add_argument("--windows", default="8,16,32")
    p.add_argument("--inter-node-gbps", type=float, default=200.0)
    p.add_argument("--num-random-seeds", type=int, default=30)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        default="experiments/idea_a_mac/outputs/paper_validation/causal_window_congestion",
    )
    return p.parse_args()


def score_by_causal_window(
    layer_rows: pd.DataFrame, candidate_index: pd.Index, gpus_per_node: int, window: int
) -> pd.Series:
    """Causal, within-job, within-layer sliding-window sender-load score.

    For each row (token) in `layer_rows`, the score is the count of how many times
    that row's sender_rank appeared among the last `window` REMOTE tokens of the
    SAME sample_id (job), strictly before this row's token_position. No future
    tokens and no other job's tokens are ever used. Rows with no prior window
    history (cold start at the beginning of a job) get score 0.
    """
    scores = pd.Series(0.0, index=layer_rows.index)
    candidate_set = set(candidate_index)
    for _sample_id, job_rows in layer_rows.groupby("sample_id"):
        job_rows = job_rows.sort_values("token_position")
        remote_mask = (
            (job_rows["sender_rank"] // gpus_per_node) != (job_rows["receiver_rank"] // gpus_per_node)
        ).to_numpy()
        sender_ranks = job_rows["sender_rank"].to_numpy()
        row_ids = job_rows.index.to_numpy()

        window_deque: deque = deque(maxlen=window)
        counter: Counter = Counter()
        for pos in range(len(job_rows)):
            row_id = row_ids[pos]
            if row_id in candidate_set:
                scores.loc[row_id] = float(counter.get(sender_ranks[pos], 0))
            if remote_mask[pos]:
                if len(window_deque) == window:
                    oldest = window_deque.popleft()
                    counter[oldest] -= 1
                    if counter[oldest] <= 0:
                        del counter[oldest]
                window_deque.append(sender_ranks[pos])
                counter[sender_ranks[pos]] += 1
    return scores.loc[candidate_index]


def assign_bytes(
    scenario: pd.DataFrame,
    top_k: int,
    gpus_per_node: int,
    hidden_size: int,
    mode: str,
    fraction: float,
    seed: int,
    window: int,
) -> pd.DataFrame:
    rows = scenario.copy()
    rows["bytes_per_element"] = 1.0  # baseline: uniform FP8
    candidate_mask = build_candidate_pool(rows, top_k, gpus_per_node)

    selected: list[int] = []
    for layer, layer_rows in rows.groupby("layer"):
        candidates = layer_rows[candidate_mask.loc[layer_rows.index]]
        if candidates.empty:
            continue
        budget = int(round(len(candidates) * fraction))
        if budget <= 0:
            continue
        if mode == "random":
            rng = np.random.default_rng(seed)
            n = min(budget, len(candidates))
            chosen = rng.choice(candidates.index.to_numpy(), size=n, replace=False)
        elif mode == "hot":
            # oracle upper bound: uses the FULL current-scenario remote load for this
            # layer (includes "future" tokens within the same job) -- NOT deployable,
            # kept only as a reference ceiling.
            remote_mask = (
                (layer_rows["sender_rank"] // gpus_per_node) != (layer_rows["receiver_rank"] // gpus_per_node)
            )
            remote_rows = layer_rows[remote_mask]
            sender_load = remote_rows.groupby("sender_rank").size().astype(float)
            receiver_load = remote_rows.groupby("receiver_rank").size().astype(float)
            s = candidates["sender_rank"].map(sender_load).fillna(0.0)
            r = candidates["receiver_rank"].map(receiver_load).fillna(0.0)
            scores = pd.Series(np.maximum(s.to_numpy(), r.to_numpy()), index=candidates.index)
            # deterministic tie-break jitter so ties don't silently favor row order
            rng = np.random.default_rng(seed)
            scores = scores + rng.random(len(scores)) * 1e-9
            chosen = scores.sort_values(ascending=False).index[:budget]
        elif mode == "causal_window":
            scores = score_by_causal_window(layer_rows, candidates.index, gpus_per_node, window)
            rng = np.random.default_rng(seed)
            scores = scores + rng.random(len(scores)) * 1e-9  # break zero-score cold-start ties randomly
            chosen = scores.sort_values(ascending=False).index[:budget]
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
    windows = [int(v) for v in args.windows.split(",") if v]

    results: list[dict[str, float | int | str]] = []
    for origin_mode in origin_modes:
        for num_jobs in job_counts:
            scenario = concurrent_scenario(test_routes, num_jobs, origin_mode, args.ep_size, num_experts)
            candidate_mask = build_candidate_pool(scenario, top_k, args.gpus_per_node)
            candidate_fraction_of_all = float(candidate_mask.mean())

            fp8_rows = scenario.copy()
            fp8_rows["bytes_per_element"] = 1.0
            fp8_rows["payload_bytes"] = hidden_size
            fp8_metrics = layer_bottleneck_us(fp8_rows, hidden_size, args.gpus_per_node, args.inter_node_gbps)

            for fraction in fractions:
                for mode in ("hot",):
                    policy_rows = assign_bytes(
                        scenario, top_k, args.gpus_per_node, hidden_size, mode, fraction, args.seed, window=0
                    )
                    metrics = layer_bottleneck_us(policy_rows, hidden_size, args.gpus_per_node, args.inter_node_gbps)
                    results.append({
                        "origin_mode": origin_mode, "num_jobs": num_jobs, "mode": mode, "window": -1,
                        "budget_fraction": fraction, "seed": -1,
                        "candidate_fraction_of_all_pairs": candidate_fraction_of_all,
                        "fp8_bottleneck_us": fp8_metrics["sum_layer_bottleneck_us"],
                        **metrics,
                    })

                for window in windows:
                    policy_rows = assign_bytes(
                        scenario, top_k, args.gpus_per_node, hidden_size, "causal_window", fraction, args.seed, window
                    )
                    metrics = layer_bottleneck_us(policy_rows, hidden_size, args.gpus_per_node, args.inter_node_gbps)
                    results.append({
                        "origin_mode": origin_mode, "num_jobs": num_jobs, "mode": "causal_window", "window": window,
                        "budget_fraction": fraction, "seed": -1,
                        "candidate_fraction_of_all_pairs": candidate_fraction_of_all,
                        "fp8_bottleneck_us": fp8_metrics["sum_layer_bottleneck_us"],
                        **metrics,
                    })

                random_vals = []
                for trial in range(args.num_random_seeds):
                    policy_rows = assign_bytes(
                        scenario, top_k, args.gpus_per_node, hidden_size, "random", fraction,
                        args.seed + trial, window=0,
                    )
                    metrics = layer_bottleneck_us(policy_rows, hidden_size, args.gpus_per_node, args.inter_node_gbps)
                    random_vals.append(metrics["sum_layer_bottleneck_us"])
                    results.append({
                        "origin_mode": origin_mode, "num_jobs": num_jobs, "mode": "random", "window": -1,
                        "budget_fraction": fraction, "seed": trial,
                        "candidate_fraction_of_all_pairs": candidate_fraction_of_all,
                        "fp8_bottleneck_us": fp8_metrics["sum_layer_bottleneck_us"],
                        **metrics,
                    })

    df = pd.DataFrame(results)
    df["bottleneck_saving_vs_fp8"] = 1.0 - df["sum_layer_bottleneck_us"] / df["fp8_bottleneck_us"].clip(lower=1e-12)
    df.to_csv(out / "causal_window_raw.csv", index=False)

    summary_rows = []
    for (origin_mode, num_jobs, fraction), group in df.groupby(["origin_mode", "num_jobs", "budget_fraction"]):
        hot = group[group["mode"] == "hot"]["bottleneck_saving_vs_fp8"].iloc[0]
        random_vals = group[group["mode"] == "random"]["bottleneck_saving_vs_fp8"].to_numpy()
        random_mean = float(np.mean(random_vals))
        random_ci_low = float(np.quantile(random_vals, 0.025)) if len(random_vals) > 1 else random_mean
        random_ci_high = float(np.quantile(random_vals, 0.975)) if len(random_vals) > 1 else random_mean
        oracle_gap = hot - random_mean
        row = {
            "origin_mode": origin_mode,
            "num_jobs": num_jobs,
            "budget_fraction": fraction,
            "hot_oracle_saving": hot,
            "random_mean_saving": random_mean,
            "random_ci_low": random_ci_low,
            "random_ci_high": random_ci_high,
            "oracle_minus_random_gap": oracle_gap,
        }
        for window in windows:
            causal_val = group[
                (group["mode"] == "causal_window") & (group["window"] == window)
            ]["bottleneck_saving_vs_fp8"]
            if causal_val.empty:
                continue
            causal_val = float(causal_val.iloc[0])
            row[f"causal_w{window}_saving"] = causal_val
            row[f"causal_w{window}_minus_random"] = causal_val - random_mean
            row[f"causal_w{window}_pct_of_oracle_gap"] = (
                (causal_val - random_mean) / oracle_gap if abs(oracle_gap) > 1e-9 else float("nan")
            )
            row[f"causal_w{window}_beats_random_ci"] = bool(causal_val > random_ci_high)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out / "causal_window_summary.csv", index=False)

    base_columns = [
        "origin_mode", "num_jobs", "budget_fraction",
        "random_mean_saving", "hot_oracle_saving", "oracle_minus_random_gap",
    ]
    window_columns = []
    for window in windows:
        window_columns += [f"causal_w{window}_saving", f"causal_w{window}_pct_of_oracle_gap"]
    table = dataframe_to_markdown(summary_df, base_columns + window_columns)

    verdict_lines = []
    for _, row in summary_df.iterrows():
        parts = []
        for window in windows:
            pct_col = f"causal_w{window}_pct_of_oracle_gap"
            beat_col = f"causal_w{window}_beats_random_ci"
            if pct_col not in row or pd.isna(row[pct_col]):
                continue
            tag = "显著优于random" if row.get(beat_col) else "未显著优于random"
            parts.append(f"W={window}: 拿到oracle-random差距的{row[pct_col]*100:.1f}% ({tag})")
        verdict_lines.append(
            f"- {row['origin_mode']} / jobs={int(row['num_jobs'])} / frac={row['budget_fraction']:.2f}: "
            + "; ".join(parts)
        )

    report = f"""# Causal Sliding-Window Congestion Budgeting — Isolation Experiment

## 目的

在已经证伪"跨请求离线 profile"（`run_expert_popularity_stability.py`：Spearman 仅
0.39-0.50，无法跨样本迁移）之后，检验第三种、此前完全未测试过的信号：
**同一请求内部、因果的、滑动窗口 sender 负载估计**——只用该请求 REMOTE 流量里
已经解码过的最近 `window` 个 token 的 sender_rank 分布，去给接下来的 token 打分，
不使用任何未来信息，也不跨请求借用信息。

这个信号在真实系统里几乎零成本可得：EP dispatch 阶段本来就已经知道每个 token
被路由到哪个 sender rank（expert 所在 GPU），滑动窗口统计只是把这个信息按 causal
方式攒起来，不需要额外的跨 rank 同步或提前 profile。

## 方法

三方对照，固定候选池为 **tail-rank AND inter-node**（对三者完全相同）：

- `hot`（oracle，不可部署上界）：用当前 job 在该层**全部**（含未来）token 的真实
  remote 负载打分——这是`run_receiver_isolation_experiment.py`里已验证过的"热点
  识别本身有独立价值"的那个信号，这里作为上界参照，不代表可部署方案。
- `causal_window`（本实验新增，可部署）：只用同一 job、同一层、**该 token 之前**
  的最近 `window` 个 remote token 的 sender_rank 分布打分。
- `random`：{args.num_random_seeds} 个随机种子重复抽样，报告均值和 95% 分位区间（下界）。

baseline 为 uniform FP8（1 byte/elem），预算内 pair 升级为 INT4（0.5 byte/elem）。
瓶颈指标口径与 `run_ep_congestion_sim.summarize` 一致。

`causal_w{{W}}_pct_of_oracle_gap` = `(causal_window - random_mean) / (hot_oracle - random_mean)`，
即 causal_window 在"random 下界"和"oracle 上界"之间，拿到了多大比例的差距。这是
衡量"零成本可部署信号能追回多少不可部署上界收益"的核心指标。

## 配置

- model: `{args.model}`; EP=`{args.ep_size}`; GPUs/node=`{args.gpus_per_node}`
- concurrent jobs: `{job_counts}`; origin modes: `{origin_modes}`
- budget fractions (within tail∩remote pool): `{fractions}`
- causal windows tested: `{windows}` (tokens)
- inter-node bandwidth: `{args.inter_node_gbps} Gbps`
- random trials per config: `{args.num_random_seeds}`

## 结果

{table}

## 判定

{chr(10).join(verdict_lines)}

## 解读边界

- 这仍是 bandwidth-only 解析回放，不含 collective、queueing、pack/unpack、kernel。
- `hot` oracle 使用了同 job 内的"未来" token 信息，不是可部署方案，只作为上界参照。
- `causal_window` 的 cold-start（job 开头 window 内的 token 无历史）会退化为接近
  random 的行为，这是真实系统里无法避免的启动代价，已如实计入总体指标（未做特殊剔除）。
- 若 `causal_window` 能稳定拿到 oracle-random 差距的可观比例（且显著超出 random
  95% CI），说明"因果同请求内滑动窗口"是一个真实、独立于此前两个信号（离线 profile
  已证伪、oracle 不可部署）的、可直接部署的新信号，可以作为 receiver-aware 支线的
  替代实现方式写入论文；如果始终落在 random CI 内或占比很低，则说明该信号在本场景
  的并发/热点结构下也不够用，需要另寻其他部署路径。
"""
    (out / "causal_window_report.md").write_text(report, encoding="utf-8")
    print(table, flush=True)
    print("\n".join(verdict_lines), flush=True)
    print(f"\nsaved to {out}", flush=True)


if __name__ == "__main__":
    main()

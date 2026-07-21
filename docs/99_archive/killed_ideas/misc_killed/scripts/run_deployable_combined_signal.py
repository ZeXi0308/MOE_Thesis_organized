"""The capstone experiment for the "causal-window" investigation: does a FULLY
DEPLOYABLE combined signal (receiver-side scheduler-known load + sender-side
CAUSAL sliding-window load) close most of the gap to the non-deployable oracle
upper bound, across BOTH regimes where the two signals dominate individually?

Context (see prior scripts for the individual pieces):

  - `run_receiver_sender_decomposition.py` showed: in `hotspot` scenarios,
    `receiver_only` (scheduler-known, deployable) captures ~70-100% of the
    oracle-vs-random gap; in `balanced` scenarios, `sender_only` (same-layer
    routing-dependent, NOT cheaply deployable) captures ~75-99% of the gap.
  - `run_causal_window_congestion.py` showed: a CAUSAL, within-request sliding
    window sender-load estimate (deployable, zero future/cross-request info)
    recovers ~35-67% of the oracle-random gap in `balanced` scenarios, but is
    near-useless in `hotspot` scenarios.

This suggests a natural fully-deployable combination:

    score = max(receiver_load_scheduler_known, causal_window_sender_load)

which should behave like `receiver_only` in hotspot (where causal_window adds
little) and like a partial `sender_only` proxy in balanced (where receiver_only
alone is weak). This script tests that combined score directly against:

  - `hot` oracle (uses full/future same-layer sender+receiver load; NOT deployable)
  - `receiver_only` (deployable, scheduler-known; already tested)
  - `causal_window` alone (deployable, causal; already tested)
  - `random` (lower bound, many-seed CI)

on the SAME fixed candidate pool and bottleneck metric used throughout this
investigation. Still bandwidth-only analytical replay, no collective/kernel
effects.
"""
from __future__ import annotations


# --- shared-lib bootstrap (auto) ---
import sys
from pathlib import Path as _Path

def _ensure_shared_on_path() -> None:
    here = _Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        cand = p / "experiments" / "shared"
        if (cand / "capture_moe.py").exists():
            s = str(cand)
            if s not in sys.path:
                sys.path.insert(0, s)
            return
        if (p / "capture_moe.py").exists():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
            return

_ensure_shared_on_path()
del _ensure_shared_on_path, _Path
# --- end bootstrap ---

import argparse
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoConfig

from run_ep_congestion_sim import concurrent_scenario, dataframe_to_markdown
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
    p.add_argument("--budget-fractions", default="0.25,0.5,0.75")
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--inter-node-gbps", type=float, default=200.0)
    p.add_argument("--num-random-seeds", type=int, default=30)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        default="experiments/idea_a_mac/outputs/paper_validation/deployable_combined_signal",
    )
    return p.parse_args()


def causal_sender_scores(layer_rows: pd.DataFrame, candidate_index: pd.Index, gpus_per_node: int, window: int) -> pd.Series:
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


def receiver_scores(layer_rows: pd.DataFrame, candidate_index: pd.Index, gpus_per_node: int) -> pd.Series:
    remote_mask = (layer_rows["sender_rank"] // gpus_per_node) != (layer_rows["receiver_rank"] // gpus_per_node)
    remote_rows = layer_rows[remote_mask]
    receiver_load = remote_rows.groupby("receiver_rank").size().astype(float)
    cand = layer_rows.loc[candidate_index]
    r = cand["receiver_rank"].map(receiver_load).fillna(0.0)
    return pd.Series(r.to_numpy(), index=candidate_index)


def oracle_scores(layer_rows: pd.DataFrame, candidate_index: pd.Index, gpus_per_node: int) -> pd.Series:
    remote_mask = (layer_rows["sender_rank"] // gpus_per_node) != (layer_rows["receiver_rank"] // gpus_per_node)
    remote_rows = layer_rows[remote_mask]
    sender_load = remote_rows.groupby("sender_rank").size().astype(float)
    receiver_load = remote_rows.groupby("receiver_rank").size().astype(float)
    cand = layer_rows.loc[candidate_index]
    s = cand["sender_rank"].map(sender_load).fillna(0.0).to_numpy()
    r = cand["receiver_rank"].map(receiver_load).fillna(0.0).to_numpy()
    return pd.Series(np.maximum(s, r), index=candidate_index)


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
        rng = np.random.default_rng(seed)
        if mode == "random":
            n = min(budget, len(candidates))
            chosen = rng.choice(candidates.index.to_numpy(), size=n, replace=False)
        elif mode == "hot_oracle":
            scores = oracle_scores(layer_rows, candidates.index, gpus_per_node)
            scores = scores + rng.random(len(scores)) * 1e-9
            chosen = scores.sort_values(ascending=False).index[:budget]
        elif mode == "receiver_only":
            scores = receiver_scores(layer_rows, candidates.index, gpus_per_node)
            scores = scores + rng.random(len(scores)) * 1e-9
            chosen = scores.sort_values(ascending=False).index[:budget]
        elif mode == "causal_sender_only":
            scores = causal_sender_scores(layer_rows, candidates.index, gpus_per_node, window)
            scores = scores + rng.random(len(scores)) * 1e-9
            chosen = scores.sort_values(ascending=False).index[:budget]
        elif mode == "deployable_combined":
            r = receiver_scores(layer_rows, candidates.index, gpus_per_node)
            c = causal_sender_scores(layer_rows, candidates.index, gpus_per_node, window)
            # normalize each signal to [0,1] within this layer before combining, so
            # one signal's raw magnitude doesn't trivially dominate the other.
            r_norm = (r - r.min()) / max(float(r.max() - r.min()), 1e-9)
            c_norm = (c - c.min()) / max(float(c.max() - c.min()), 1e-9)
            scores = pd.Series(np.maximum(r_norm.to_numpy(), c_norm.to_numpy()), index=candidates.index)
            scores = scores + rng.random(len(scores)) * 1e-9
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

    job_counts = [int(v) for v in args.num_jobs.split(",") if v]
    origin_modes = [v.strip() for v in args.origin_modes.split(",") if v.strip()]
    fractions = [float(v) for v in args.budget_fractions.split(",") if v]
    modes = ["hot_oracle", "receiver_only", "causal_sender_only", "deployable_combined"]

    results: list[dict[str, float | int | str]] = []
    for origin_mode in origin_modes:
        for num_jobs in job_counts:
            scenario = concurrent_scenario(test_routes, num_jobs, origin_mode, args.ep_size, num_experts)

            fp8_rows = scenario.copy()
            fp8_rows["bytes_per_element"] = 1.0
            fp8_rows["payload_bytes"] = hidden_size
            fp8_bottleneck = layer_bottleneck_us(fp8_rows, hidden_size, args.gpus_per_node, args.inter_node_gbps)["sum_layer_bottleneck_us"]

            for fraction in fractions:
                for mode in modes:
                    policy_rows = assign_bytes(
                        scenario, top_k, args.gpus_per_node, hidden_size, mode, fraction, args.seed, args.window
                    )
                    metrics = layer_bottleneck_us(policy_rows, hidden_size, args.gpus_per_node, args.inter_node_gbps)
                    results.append({
                        "origin_mode": origin_mode, "num_jobs": num_jobs, "mode": mode,
                        "budget_fraction": fraction, "fp8_bottleneck_us": fp8_bottleneck, **metrics,
                    })

                random_vals = []
                for trial in range(args.num_random_seeds):
                    policy_rows = assign_bytes(
                        scenario, top_k, args.gpus_per_node, hidden_size, "random", fraction,
                        args.seed + trial, args.window,
                    )
                    metrics = layer_bottleneck_us(policy_rows, hidden_size, args.gpus_per_node, args.inter_node_gbps)
                    random_vals.append(metrics["sum_layer_bottleneck_us"])
                results.append({
                    "origin_mode": origin_mode, "num_jobs": num_jobs, "mode": "random",
                    "budget_fraction": fraction, "fp8_bottleneck_us": fp8_bottleneck,
                    "sum_layer_bottleneck_us": float(np.mean(random_vals)),
                    "p99_layer_receiver_bytes": float("nan"),
                    "mean_layer_receiver_imbalance": float("nan"),
                })

    df = pd.DataFrame(results)
    df["bottleneck_saving_vs_fp8"] = 1.0 - df["sum_layer_bottleneck_us"] / df["fp8_bottleneck_us"].clip(lower=1e-12)
    df.to_csv(out / "deployable_combined_raw.csv", index=False)

    pivot = df.pivot_table(
        index=["origin_mode", "num_jobs", "budget_fraction"],
        columns="mode",
        values="bottleneck_saving_vs_fp8",
    ).reset_index()
    gap = (pivot["hot_oracle"] - pivot["random"]).clip(lower=1e-9)
    pivot["deployable_combined_pct_of_oracle_gap"] = (pivot["deployable_combined"] - pivot["random"]) / gap
    pivot["receiver_only_pct_of_oracle_gap"] = (pivot["receiver_only"] - pivot["random"]) / gap
    pivot["causal_sender_only_pct_of_oracle_gap"] = (pivot["causal_sender_only"] - pivot["random"]) / gap
    pivot.to_csv(out / "deployable_combined_summary.csv", index=False)

    columns = [
        "origin_mode", "num_jobs", "budget_fraction", "random", "hot_oracle",
        "receiver_only", "receiver_only_pct_of_oracle_gap",
        "causal_sender_only", "causal_sender_only_pct_of_oracle_gap",
        "deployable_combined", "deployable_combined_pct_of_oracle_gap",
    ]
    table = dataframe_to_markdown(pivot, columns)

    mean_combined_pct = float(pivot["deployable_combined_pct_of_oracle_gap"].clip(-1, 1.5).mean())
    mean_receiver_pct = float(pivot["receiver_only_pct_of_oracle_gap"].clip(-1, 1.5).mean())
    mean_causal_pct = float(pivot["causal_sender_only_pct_of_oracle_gap"].clip(-1, 1.5).mean())

    by_regime = pivot.groupby("origin_mode")[
        ["receiver_only_pct_of_oracle_gap", "causal_sender_only_pct_of_oracle_gap", "deployable_combined_pct_of_oracle_gap"]
    ].mean()

    report = f"""# Fully-Deployable Combined Signal vs Oracle Upper Bound

## 目的

检验一个**完全可部署**（不依赖跨请求离线 profile、不依赖同层未来信息）的组合信号：

    score = max(normalize(receiver_load_scheduler_known), normalize(causal_window_sender_load))

能否在 `balanced`（此前证明主要靠 sender 侧驱动）和 `hotspot`（此前证明主要靠
receiver 侧驱动）两种场景下，都追回 `hot_oracle`（用同层未来信息的不可部署上界）
相对 `random` 的大部分收益。这是把前两轮独立验证过的两个可部署信号（receiver_only
调度器已知负载 + causal_window 同请求因果滑动窗口负载）第一次合并检验。

## 方法

- `hot_oracle`：`max(sender_load, receiver_load)`，用同层全部（含未来）token 计算，
  不可部署，仅作上界参照。
- `receiver_only`：只用 receiver（token-origin GPU）当前负载，调度器已知，可部署。
- `causal_sender_only`：只用 sender（expert-owner GPU）当前负载的因果滑动窗口估计
  （window=`{args.window}`），可部署，零跨请求/未来信息。
- `deployable_combined`：上述两个可部署信号各自归一化到 [0,1] 后取 max，完全可部署。
- `random`：{args.num_random_seeds} 个随机种子均值，下界。

固定候选池为 tail-rank AND inter-node，四种打分方式共享同一候选池。

## 结果

{table}

## 按场景汇总（各信号占 oracle-random 差距的平均比例）

{dataframe_to_markdown(by_regime.reset_index(), list(by_regime.reset_index().columns))}

## 关键读数

- `deployable_combined` 平均追回 oracle-random 差距的 `{mean_combined_pct*100:.1f}%`
- 相比之下：`receiver_only` 单独平均追回 `{mean_receiver_pct*100:.1f}%`；
  `causal_sender_only` 单独平均追回 `{mean_causal_pct*100:.1f}%`

## 解读边界

- 这仍是 bandwidth-only 解析回放，不含 collective、queueing、pack/unpack、kernel。
- 若 `deployable_combined` 在两种场景下都显著超过其两个组成信号各自单独的表现，
  说明"receiver 调度信号 + sender 因果滑动窗口信号"是互补的，组合后可以用**完全
  不依赖离线 profile、不依赖跨请求假设**的方式，同时覆盖 hotspot 和 balanced 两种
  负载模式，这是一个可以写成论文核心系统设计的、自洽的两信号方案。
- 若组合信号在某一场景下仍显著落后于单信号里更强的那一个，说明简单 max 组合不是
  最优融合方式，需要场景自适应的权重或门控机制，可作为下一步改进方向。
- causal_window 存在 job 开头的冷启动代价（历史不足时退化为约等于 random），已计入
  总体指标，未做特殊剔除。
"""
    (out / "deployable_combined_report.md").write_text(report, encoding="utf-8")
    print(table, flush=True)
    print(f"\nmean_combined_pct={mean_combined_pct:.3f} mean_receiver_pct={mean_receiver_pct:.3f} mean_causal_pct={mean_causal_pct:.3f}", flush=True)
    print(f"saved to {out}", flush=True)


if __name__ == "__main__":
    main()

"""Decompose the "hot" signal in run_receiver_isolation_experiment.py into its
receiver-side and sender-side components to assess deployability.

Motivation
----------
run_receiver_isolation_experiment.py already shows that, within a fixed
tail-rank ∩ inter-node candidate pool, choosing pairs by
`max(sender_load, receiver_load)` beats random by a wide, stable margin.
But `sender_load` (which expert-owner GPU is currently busy) is only known
AFTER this layer's routing has been computed and gathered across EP ranks --
that is expensive same-layer, cross-rank information. `receiver_load` (which
token-origin GPU is currently receiving a lot of traffic) is closer to
something a request scheduler could know ahead of the current layer, because
it depends mostly on which GPU owns which in-flight requests, not on this
layer's routing outcome.

This script re-runs the SAME fixed candidate pool (tail-rank AND inter-node)
under four scoring variants:

  - `receiver_only` : score = receiver_load only (schedulable ahead-of-time)
  - `sender_only`    : score = sender_load only (requires this-layer routing)
  - `combined`        : score = max(sender_load, receiver_load) (original "hot")
  - `random`          : uniform random, many seeds, for CI

If `receiver_only` captures most of the gap between `combined` and `random`,
receiver-aware compression can plausibly be implemented as a scheduler-level
decision (cheap, deployable). If `sender_only` is required to get most of the
benefit, the mechanism depends on per-layer expert-load information, which
is much harder to act on before combine starts (the combine communication is
exactly what we are trying to compress ahead of).
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
    p.add_argument("--inter-node-gbps", type=float, default=200.0)
    p.add_argument("--num-random-seeds", type=int, default=30)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        default="experiments/idea_a_mac/outputs/paper_validation/receiver_isolation",
    )
    return p.parse_args()


def score_component(layer_rows: pd.DataFrame, candidate_index: pd.Index, gpus_per_node: int, component: str) -> pd.Series:
    remote_mask = (layer_rows["sender_rank"] // gpus_per_node) != (layer_rows["receiver_rank"] // gpus_per_node)
    remote_rows = layer_rows[remote_mask]
    sender_load = remote_rows.groupby("sender_rank").size().astype(float)
    receiver_load = remote_rows.groupby("receiver_rank").size().astype(float)
    cand = layer_rows.loc[candidate_index]
    s = cand["sender_rank"].map(sender_load).fillna(0.0).to_numpy()
    r = cand["receiver_rank"].map(receiver_load).fillna(0.0).to_numpy()
    if component == "sender_only":
        values = s
    elif component == "receiver_only":
        values = r
    elif component == "combined":
        values = np.maximum(s, r)
    else:
        raise ValueError(f"unknown component: {component}")
    return pd.Series(values, index=candidate_index)


def assign_bytes_by_component(
    scenario: pd.DataFrame,
    top_k: int,
    gpus_per_node: int,
    hidden_size: int,
    mode: str,
    fraction: float,
    seed: int,
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
        if mode == "random":
            rng = np.random.default_rng(seed)
            n = min(budget, len(candidates))
            chosen = rng.choice(candidates.index.to_numpy(), size=n, replace=False)
        else:
            scores = score_component(layer_rows, candidates.index, gpus_per_node, mode)
            chosen = scores.sort_values(ascending=False).index[:budget]
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
    components = ["receiver_only", "sender_only", "combined"]

    results: list[dict[str, float | int | str]] = []
    for origin_mode in origin_modes:
        for num_jobs in job_counts:
            scenario = concurrent_scenario(test_routes, num_jobs, origin_mode, args.ep_size, num_experts)

            fp8_rows = scenario.copy()
            fp8_rows["bytes_per_element"] = 1.0
            fp8_rows["payload_bytes"] = hidden_size
            fp8_bottleneck = layer_bottleneck_us(fp8_rows, hidden_size, args.gpus_per_node, args.inter_node_gbps)["sum_layer_bottleneck_us"]

            for fraction in fractions:
                for component in components:
                    policy_rows = assign_bytes_by_component(
                        scenario, top_k, args.gpus_per_node, hidden_size, component, fraction, args.seed
                    )
                    metrics = layer_bottleneck_us(policy_rows, hidden_size, args.gpus_per_node, args.inter_node_gbps)
                    results.append({
                        "origin_mode": origin_mode, "num_jobs": num_jobs, "component": component,
                        "budget_fraction": fraction, "fp8_bottleneck_us": fp8_bottleneck, **metrics,
                    })

                random_vals = []
                for trial in range(args.num_random_seeds):
                    policy_rows = assign_bytes_by_component(
                        scenario, top_k, args.gpus_per_node, hidden_size, "random", fraction, args.seed + trial
                    )
                    metrics = layer_bottleneck_us(policy_rows, hidden_size, args.gpus_per_node, args.inter_node_gbps)
                    random_vals.append(metrics["sum_layer_bottleneck_us"])
                results.append({
                    "origin_mode": origin_mode, "num_jobs": num_jobs, "component": "random",
                    "budget_fraction": fraction, "fp8_bottleneck_us": fp8_bottleneck,
                    "sum_layer_bottleneck_us": float(np.mean(random_vals)),
                    "p99_layer_receiver_bytes": float("nan"),
                    "mean_layer_receiver_imbalance": float("nan"),
                })

    df = pd.DataFrame(results)
    df["bottleneck_saving_vs_fp8"] = 1.0 - df["sum_layer_bottleneck_us"] / df["fp8_bottleneck_us"].clip(lower=1e-12)
    df.to_csv(out / "receiver_sender_decomposition_raw.csv", index=False)

    pivot = df.pivot_table(
        index=["origin_mode", "num_jobs", "budget_fraction"],
        columns="component",
        values="bottleneck_saving_vs_fp8",
    ).reset_index()
    pivot["receiver_share_of_combined_gain"] = (
        (pivot["receiver_only"] - pivot["random"]) / (pivot["combined"] - pivot["random"]).clip(lower=1e-9)
    )
    pivot["sender_share_of_combined_gain"] = (
        (pivot["sender_only"] - pivot["random"]) / (pivot["combined"] - pivot["random"]).clip(lower=1e-9)
    )
    pivot.to_csv(out / "receiver_sender_decomposition_summary.csv", index=False)

    columns = ["origin_mode", "num_jobs", "budget_fraction", "random", "receiver_only",
               "sender_only", "combined", "receiver_share_of_combined_gain", "sender_share_of_combined_gain"]
    table = dataframe_to_markdown(pivot, columns)

    mean_receiver_share = float(pivot["receiver_share_of_combined_gain"].clip(0, 1.5).mean())
    mean_sender_share = float(pivot["sender_share_of_combined_gain"].clip(0, 1.5).mean())

    report = f"""# Receiver vs Sender Signal Decomposition

## 目的

在已经剥离 remote-preference 混淆的固定候选池（tail-rank ∩ inter-node）内，
进一步拆解"热度信号"中 receiver 侧和 sender 侧各自的贡献，以判断
receiver-aware 压缩策略在真实系统中的可部署性：

- `receiver_only`：只按 receiver（token-origin GPU）当前负载排序。这类信息
  通常可由请求调度层提前掌握（哪个 GPU 持有哪些 in-flight 请求/token），
  不需要等这一层的路由结果。
- `sender_only`：只按 sender（expert-owner GPU）当前负载排序。这类信息
  依赖这一层刚计算出的路由结果，且需要跨 EP rank 聚合专家命中数，是
  同层实时信息，获取代价更高，且与要压缩的 combine 通信本身同批次发生。
- `combined`：`max(sender_load, receiver_load)`，即此前 isolation 实验中的
  `hot`。
- `random`：{args.num_random_seeds} 次随机种子均值。

## 结果

{table}

## 关键读数

- receiver_only 对 combined 相对 random 收益的平均贡献占比：约 `{mean_receiver_share:.2f}`
- sender_only 对 combined 相对 random 收益的平均贡献占比：约 `{mean_sender_share:.2f}`

## 解读

- 若 receiver_only 已能拿到 combined 收益的大部分（例如 ≥ 0.6），说明
  receiver-aware 的核心价值可以只靠调度层已知信息实现，不需要等本层路由，
  是一个更容易部署、更低同步开销的信号，可以作为论文里更强、更可防守的
  claim（"receiver-side scheduling signal alone captures most of the
  benefit"）。
- 若 sender_only 的贡献接近甚至超过 receiver_only，说明真正驱动收益的是
  "这一层谁被选中的专家更忙"，这与本层路由强耦合，需要在 dispatch 之后、
  combine 之前插入一次跨 rank 的轻量同步才能利用，实现代价显著更高，
  应在论文里明确写成"需要 layer-local 专家负载同步"的额外系统假设，
  不能默认为免费信号。
- 仍是 bandwidth-only 解析回放，不含 collective、queueing、同步开销本身。
"""
    (out / "receiver_sender_decomposition_report.md").write_text(report, encoding="utf-8")
    print(table, flush=True)
    print(f"\nreceiver_share_mean={mean_receiver_share:.3f} sender_share_mean={mean_sender_share:.3f}", flush=True)
    print(f"saved to {out}", flush=True)


if __name__ == "__main__":
    main()

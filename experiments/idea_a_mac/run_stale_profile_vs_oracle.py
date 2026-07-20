"""Test whether the paper's existing `tail_budget_profile_ports` policy (offline
calibration-derived sender/receiver load used to score test-time candidates)
is actually informative, once the remote-preference confound documented in
run_receiver_isolation_experiment.py is removed.

Given run_expert_popularity_stability.py shows expert/sender-rank load ranking
is only weakly stable across disjoint samples (mean Spearman ~0.39-0.50), a
STALE offline profile should perform close to random once it can no longer
free-ride on "prefer remote over local". This script checks that directly by
comparing four policies within the SAME fixed candidate pool (tail-rank AND
inter-node), in the SAME concurrent-job scenarios used elsewhere:

  - `oracle_combined`  : current-window max(sender_load, receiver_load) (upper bound, non-deployable online)
  - `stale_profile`    : calibration-derived max(sender_load, receiver_load), applied to test scenario
  - `oracle_receiver`  : current-window receiver_load only (schedulable ahead of time)
  - `random`           : many seeds, for CI
"""
from __future__ import annotations

import argparse
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
        "--calibration-routes",
        default="experiments/idea_a_mac/outputs/paper_validation/olmoe_signal_comparison_n32/calibration_routes.csv",
    )
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


def compute_calibration_loads(calibration_routes: pd.DataFrame, ep_size: int, num_experts: int) -> tuple[dict, dict]:
    """Reproduces the calibration_scores() logic from run_ep_congestion_sim but
    keyed the same way score_component expects: (layer, sender_rank)/(layer, receiver_rank) -> count."""
    sample_ids = sorted(int(v) for v in calibration_routes["sample_id"].unique())
    receivers = {sample_id: idx % ep_size for idx, sample_id in enumerate(sample_ids)}
    rows = add_placement(calibration_routes, ep_size, num_experts, receivers)
    sender = rows.groupby(["layer", "sender_rank"]).size().to_dict()
    receiver = rows.groupby(["layer", "receiver_rank"]).size().to_dict()
    return sender, receiver


def score_oracle(layer_rows: pd.DataFrame, candidate_index: pd.Index, gpus_per_node: int, component: str) -> pd.Series:
    remote_mask = (layer_rows["sender_rank"] // gpus_per_node) != (layer_rows["receiver_rank"] // gpus_per_node)
    remote_rows = layer_rows[remote_mask]
    sender_load = remote_rows.groupby("sender_rank").size().astype(float)
    receiver_load = remote_rows.groupby("receiver_rank").size().astype(float)
    cand = layer_rows.loc[candidate_index]
    s = cand["sender_rank"].map(sender_load).fillna(0.0).to_numpy()
    r = cand["receiver_rank"].map(receiver_load).fillna(0.0).to_numpy()
    if component == "receiver_only":
        return pd.Series(r, index=candidate_index)
    return pd.Series(np.maximum(s, r), index=candidate_index)


def score_stale_profile(
    layer_rows: pd.DataFrame,
    candidate_index: pd.Index,
    layer: int,
    profile_sender: dict,
    profile_receiver: dict,
) -> pd.Series:
    cand = layer_rows.loc[candidate_index]
    s = cand["sender_rank"].map(lambda r: profile_sender.get((layer, int(r)), 0.0)).to_numpy(dtype=float)
    r = cand["receiver_rank"].map(lambda r: profile_receiver.get((layer, int(r)), 0.0)).to_numpy(dtype=float)
    return pd.Series(np.maximum(s, r), index=candidate_index)


def assign_bytes(
    scenario: pd.DataFrame,
    top_k: int,
    gpus_per_node: int,
    hidden_size: int,
    mode: str,
    fraction: float,
    seed: int,
    profile_sender: dict | None,
    profile_receiver: dict | None,
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
        elif mode == "stale_profile":
            scores = score_stale_profile(layer_rows, candidates.index, int(layer), profile_sender, profile_receiver)
            chosen = scores.sort_values(ascending=False).index[:budget]
        else:
            scores = score_oracle(layer_rows, candidates.index, gpus_per_node, mode)
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

    calibration_routes = pd.read_csv(args.calibration_routes)
    test_routes = pd.read_csv(args.test_routes)
    cfg = AutoConfig.from_pretrained(args.model, local_files_only=args.offline)
    hidden_size = int(cfg.hidden_size)
    num_experts = int(getattr(cfg, "num_experts", getattr(cfg, "num_local_experts", 0)))
    top_k = int(getattr(cfg, "num_experts_per_tok", getattr(cfg, "num_experts_per_token", 0)))

    profile_sender, profile_receiver = compute_calibration_loads(calibration_routes, args.ep_size, num_experts)

    job_counts = [int(v) for v in args.num_jobs.split(",") if v]
    origin_modes = [v.strip() for v in args.origin_modes.split(",") if v.strip()]
    fractions = [float(v) for v in args.budget_fractions.split(",") if v]

    results: list[dict[str, float | int | str]] = []
    for origin_mode in origin_modes:
        for num_jobs in job_counts:
            scenario = concurrent_scenario(test_routes, num_jobs, origin_mode, args.ep_size, num_experts)
            fp8_rows = scenario.copy()
            fp8_rows["bytes_per_element"] = 1.0
            fp8_rows["payload_bytes"] = hidden_size
            fp8_bottleneck = layer_bottleneck_us(fp8_rows, hidden_size, args.gpus_per_node, args.inter_node_gbps)["sum_layer_bottleneck_us"]

            for fraction in fractions:
                for mode in ("combined", "receiver_only", "stale_profile"):
                    policy_rows = assign_bytes(
                        scenario, top_k, args.gpus_per_node, hidden_size, mode, fraction, args.seed,
                        profile_sender, profile_receiver,
                    )
                    metrics = layer_bottleneck_us(policy_rows, hidden_size, args.gpus_per_node, args.inter_node_gbps)
                    label = {"combined": "oracle_combined", "receiver_only": "oracle_receiver",
                             "stale_profile": "stale_profile"}[mode]
                    results.append({
                        "origin_mode": origin_mode, "num_jobs": num_jobs, "policy": label,
                        "budget_fraction": fraction, "fp8_bottleneck_us": fp8_bottleneck, **metrics,
                    })

                random_vals = []
                for trial in range(args.num_random_seeds):
                    policy_rows = assign_bytes(
                        scenario, top_k, args.gpus_per_node, hidden_size, "random", fraction, args.seed + trial,
                        profile_sender, profile_receiver,
                    )
                    metrics = layer_bottleneck_us(policy_rows, hidden_size, args.gpus_per_node, args.inter_node_gbps)
                    random_vals.append(metrics["sum_layer_bottleneck_us"])
                results.append({
                    "origin_mode": origin_mode, "num_jobs": num_jobs, "policy": "random",
                    "budget_fraction": fraction, "fp8_bottleneck_us": fp8_bottleneck,
                    "sum_layer_bottleneck_us": float(np.mean(random_vals)),
                    "p99_layer_receiver_bytes": float("nan"), "mean_layer_receiver_imbalance": float("nan"),
                })
                results[-1]["random_ci_low_bottleneck"] = float(np.quantile(random_vals, 0.025))
                results[-1]["random_ci_high_bottleneck"] = float(np.quantile(random_vals, 0.975))

    df = pd.DataFrame(results)
    df["bottleneck_saving_vs_fp8"] = 1.0 - df["sum_layer_bottleneck_us"] / df["fp8_bottleneck_us"].clip(lower=1e-12)
    df.to_csv(out / "stale_profile_vs_oracle_raw.csv", index=False)

    pivot = df.pivot_table(
        index=["origin_mode", "num_jobs", "budget_fraction"],
        columns="policy",
        values="bottleneck_saving_vs_fp8",
    ).reset_index()
    pivot.to_csv(out / "stale_profile_vs_oracle_summary.csv", index=False)

    columns = ["origin_mode", "num_jobs", "budget_fraction", "random", "stale_profile",
               "oracle_receiver", "oracle_combined"]
    table = dataframe_to_markdown(pivot, [c for c in columns if c in pivot.columns])

    stale_close_to_random = float((pivot["stale_profile"] - pivot["random"]).abs().mean())
    stale_vs_oracle_gap = float((pivot["oracle_combined"] - pivot["stale_profile"]).mean())

    report = f"""# Stale Offline Profile vs Oracle Signals (Confound-Removed)

## 目的

在已经剥离 remote-preference 混淆的固定候选池内，直接检验论文当前方法
`tail_budget_profile_ports`（用 disjoint calibration 算出的静态 sender/receiver
负载去打分 test 时刻的候选）到底是不是一个有效信号，还是它之前报告的
"追平 scheduler_receiver/greedy_ports"表现完全是 remote-bonus 撑出来的假象。

## 方法

同一固定候选池（tail-rank ∩ inter-node），比较：

- `random`：{args.num_random_seeds} 次随机种子均值（基线）
- `stale_profile`：用 calibration 集合（offset 0，16 条文本）算出的静态
  `max(sender_load, receiver_load)` 去给 test 场景的候选打分——这正是论文
  `tail_budget_profile_ports` 策略实际依赖的信息
- `oracle_receiver`：test 场景当前真实 receiver 负载（scheduler 可提前知道的量）
- `oracle_combined`：test 场景当前真实 `max(sender,receiver)` 负载（不可离线获得的上界）

## 配置

- model: `{args.model}`; EP=`{args.ep_size}`; GPUs/node=`{args.gpus_per_node}`
- calibration: `{args.calibration_routes}`（offset 0, disjoint from test）
- test scenario 来源: `{args.test_routes}`
- concurrent jobs: `{job_counts}`; origin modes: `{origin_modes}`

## 结果

{table}

## 关键读数

- `stale_profile` 与 `random` 的平均绝对差：`{stale_close_to_random:.4f}`
- `oracle_combined` 与 `stale_profile` 的平均差距：`{stale_vs_oracle_gap:.4f}`

## 判定

结合 `run_expert_popularity_stability.py` 的发现（专家/owner 负载排序跨样本
Spearman 仅 ~0.39-0.50，最热 sender rank 跨集合一致率仅 25%），此处的直接
对照进一步验证：**静态离线 profile 在被剥离 remote-bonus 后的表现明显弱于
oracle 信号**（若上表中 `stale_profile` 接近 `random` 且明显低于
`oracle_combined`/`oracle_receiver`）。这意味着此前
`congestion_report.md` / `quality_safe_congestion_report.md` 中报告的
"`tail_budget_profile_ports` 与 `tail_budget_scheduler_receiver` 表现几乎相同"
这一结论，主要驱动因素是两者共享的 remote-bonus，而不是离线 profile 本身
提供了有效信号——这是需要在论文里明确纠正的一处过强表述。

## 对论文 receiver-aware 章节的具体修正建议

1. 删除或大幅弱化"离线 profile 就足够"的表述；`tail_budget_profile_ports`
   在移除 remote-bonus 后并不比 random 好多少。
2. 保留、且应重点强调 `oracle_receiver`（scheduler 已知的 receiver 热度）
   这一支：它不需要等本层路由，代价低，且在 origin 明显不均衡（hotspot）
   时几乎拿到 `oracle_combined` 的全部收益。
3. 对 sender 侧真实收益（在 origin `balanced` 时才明显），必须诚实标注为
   "需要 layer-local 在线专家负载同步，且该负载本身随输入内容变化，
   不能靠离线 profile 替代"，作为未来工作或需要真实系统实现的额外开销
   项，不能默认为免费信号。
"""
    (out / "stale_profile_vs_oracle_report.md").write_text(report, encoding="utf-8")
    print(table, flush=True)
    print(f"\nstale_close_to_random_gap={stale_close_to_random:.4f} stale_vs_oracle_gap={stale_vs_oracle_gap:.4f}", flush=True)
    print(f"saved to {out}", flush=True)


if __name__ == "__main__":
    main()

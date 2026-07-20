"""Test whether expert-popularity (the sender-side signal that dominates the
"balanced" origin scenario in run_receiver_sender_decomposition.py) is stable
across disjoint sample sets. If it is, the expensive "layer-local real-time
sender load" signal can be approximated by a cheap OFFLINE per-expert
popularity profile plus a STATIC expert-placement map -- turning a hard-to-
deploy per-layer synchronization requirement into a deployable offline LUT,
much like the existing gate-threshold / PLTB profiling in this project.

Method
------
Using calibration_routes.csv (offset 0) and test_routes.csv (offset 128) from
olmoe_signal_comparison_n32 -- two disjoint WikiText-2 slices -- compute, per
layer, the expert-hit-count distribution. Compare calibration-set ranking vs
test-set ranking with Spearman correlation. Also aggregate hits into
EP-owner-rank load (expert_id -> sender_rank under the same placement rule
used elsewhere) and compare owner-rank load ranking stability, since that is
the actual unit the sender-side signal operates on.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from transformers import AutoConfig


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
    p.add_argument("--offline", action="store_true")
    p.add_argument(
        "--output-dir",
        default="experiments/idea_a_mac/outputs/paper_validation/receiver_isolation",
    )
    return p.parse_args()


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cal = pd.read_csv(args.calibration_routes)
    test = pd.read_csv(args.test_routes)
    cfg = AutoConfig.from_pretrained(args.model, local_files_only=args.offline)
    num_experts = int(getattr(cfg, "num_experts", getattr(cfg, "num_local_experts", 0)))

    cal["sender_rank"] = np.minimum(cal["expert_id"].astype(int) * args.ep_size // num_experts, args.ep_size - 1)
    test["sender_rank"] = np.minimum(test["expert_id"].astype(int) * args.ep_size // num_experts, args.ep_size - 1)

    rows = []
    expert_corrs, sender_corrs = [], []
    for layer in sorted(cal["layer"].unique()):
        cal_layer = cal[cal["layer"] == layer]
        test_layer = test[test["layer"] == layer]

        # expert-id popularity
        cal_expert = cal_layer.groupby("expert_id").size().reindex(range(num_experts), fill_value=0)
        test_expert = test_layer.groupby("expert_id").size().reindex(range(num_experts), fill_value=0)
        expert_rho, _ = spearmanr(cal_expert.to_numpy(), test_expert.to_numpy())

        # sender-rank (owner GPU) load -- the actual unit consumed by the sender signal
        cal_sender = cal_layer.groupby("sender_rank").size().reindex(range(args.ep_size), fill_value=0)
        test_sender = test_layer.groupby("sender_rank").size().reindex(range(args.ep_size), fill_value=0)
        sender_rho, _ = spearmanr(cal_sender.to_numpy(), test_sender.to_numpy())

        expert_corrs.append(expert_rho)
        sender_corrs.append(sender_rho)
        rows.append({
            "layer": int(layer),
            "expert_popularity_spearman_cal_vs_test": float(expert_rho),
            "sender_rank_load_spearman_cal_vs_test": float(sender_rho),
            "cal_top_expert": int(cal_expert.idxmax()),
            "test_top_expert": int(test_expert.idxmax()),
            "top_expert_matches": bool(cal_expert.idxmax() == test_expert.idxmax()),
            "cal_hottest_sender_rank": int(cal_sender.idxmax()),
            "test_hottest_sender_rank": int(test_sender.idxmax()),
            "hottest_sender_rank_matches": bool(cal_sender.idxmax() == test_sender.idxmax()),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out / "expert_popularity_stability.csv", index=False)

    mean_expert_rho = float(np.mean(expert_corrs))
    mean_sender_rho = float(np.mean(sender_corrs))
    top_expert_match_rate = float(df["top_expert_matches"].mean())
    hottest_rank_match_rate = float(df["hottest_sender_rank_matches"].mean())

    verdict = (
        "专家热度可离线 profile（跨样本高度稳定），sender 信号可以退化为"
        "一次性离线统计 + 静态 placement，不需要逐层在线同步"
        if mean_sender_rho > 0.5 and hottest_rank_match_rate > 0.5 else
        "专家热度跨样本不稳定，sender 信号确实依赖当层在线路由，"
        "不能简单退化为离线 LUT"
    )

    report = f"""# Expert Popularity Stability (Calibration vs Held-out Test)

## 目的

`run_receiver_sender_decomposition.py` 发现在 `balanced` origin 模式下，
sender 侧信号（expert-owner GPU 当前负载）贡献了 combined 收益的主要部分
（约 57%-99%，多数场景 > 75%）。但 sender 信号在当前实现中依赖"这一层
刚算出的路由结果"，这是逐层实时信息，同步代价高，几乎和要压缩的 combine
通信同时发生，可部署性存疑。

这里验证一个关键前提：expert 热度（谁更常被选中）是否在不同输入样本间
稳定？如果稳定，sender 信号可以退化为一次离线 profile（类似论文里已有的
PLTB layer sensitivity profile），而不需要逐层在线同步专家命中数。

## 方法

用 disjoint 的 calibration（offset 0）和 held-out test（offset 128）
WikiText-2 路由 trace，按 layer 比较：

- expert-id 级别命中次数分布的 Spearman 相关；
- 聚合到 sender_rank（owner GPU，按 `expert_id * ep_size // num_experts` 静态
  placement 规则）后的负载分布 Spearman 相关；
- 每层最热 expert / 最热 sender_rank 是否在两个集合中一致。

## 配置

- model: `{args.model}`; EP size: `{args.ep_size}`
- calibration: `{args.calibration_routes}`
- test: `{args.test_routes}`

## 结果（按层）

{dataframe_to_markdown(df)}

## 汇总

- expert-id 命中分布 Spearman 均值（跨层）：`{mean_expert_rho:.4f}`
- sender_rank 负载分布 Spearman 均值（跨层）：`{mean_sender_rho:.4f}`
- 最热 expert 跨集合一致率：`{top_expert_match_rate:.2%}`
- 最热 sender_rank 跨集合一致率：`{hottest_rank_match_rate:.2%}`

## 判定

**{verdict}**

## 意义

若上面判定为"可离线 profile"：这为论文提供一个比原始 receiver-aware 更
站得住脚、也更有新意的贡献点——**Two-Tier Congestion-Safe Budgeting**：

1. 离线阶段：像 PLTB 一样对每层做一次 expert-popularity profile，得到
   `LUT[layer, expert_id] -> owner_hotness_prior`（静态，随 checkpoint 固定，
   与 batch/request 无关）；
2. 在线阶段：调度器已知的 receiver/token-origin 热度（通常来自请求分布，
   不需要等路由结果）与①的静态 prior 相加，决定把有限 INT4 预算优先给
   哪些 remote (sender, receiver) pair；
3. 完全不需要在 dispatch 之后、combine 之前插入额外的跨 rank 专家负载
   同步，因为 sender 侧用的是离线 prior 而不是当层实时统计。

这比"receiver-aware"更准确地描述了真正驱动收益的机制，也更容易在论文里
写成一个可部署、有明确 offline/online 分工的系统设计，而不是笼统地说
"识别热点端口"。
"""
    (out / "expert_popularity_stability_report.md").write_text(report, encoding="utf-8")
    print(df.to_string(index=False), flush=True)
    print(f"\nmean_expert_rho={mean_expert_rho:.4f} mean_sender_rho={mean_sender_rho:.4f}", flush=True)
    print(f"top_expert_match_rate={top_expert_match_rate:.2%} hottest_rank_match_rate={hottest_rank_match_rate:.2%}", flush=True)
    print(verdict, flush=True)
    print(f"saved to {out}", flush=True)


if __name__ == "__main__":
    main()

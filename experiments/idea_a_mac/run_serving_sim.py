from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from transformers import AutoConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bandwidth-gbps", type=float, default=100.0)
    parser.add_argument("--per-layer-overhead-us", type=float, default=0.0)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[
            "full",
            "uniform_int4",
            "rank1_int4",
            "rank2_int4",
            "rank3_int4",
            "rank4_int4",
            "rank5_int4",
            "rank6_int4",
            "rank7_int4",
            "rank8_int4",
        ],
    )
    return parser.parse_args()


def hidden_size_for_model(model_name: str) -> int:
    cfg = AutoConfig.from_pretrained(model_name)
    return int(getattr(cfg, "hidden_size"))


def rank_index(top_k: int, suffix: str) -> int:
    if suffix == "k":
        return top_k
    return int(suffix)


def bytes_per_element(strategy: str, row: pd.Series, top_k: int) -> float:
    rank = int(row["rank"])
    group = int(row["receiver_group"])

    if strategy == "full":
        return 2.0
    if strategy == "uniform_int8":
        return 1.0
    if strategy == "uniform_int4":
        return 0.5

    match = re.fullmatch(r"rank(\d+|k)_int8", strategy)
    if match:
        return 1.0 if rank == rank_index(top_k, match.group(1)) else 2.0
    match = re.fullmatch(r"rank(\d+|k)_int4", strategy)
    if match:
        return 0.5 if rank == rank_index(top_k, match.group(1)) else 2.0
    match = re.fullmatch(r"rank(\d+|k)_drop", strategy)
    if match:
        return 0.0 if rank == rank_index(top_k, match.group(1)) else 2.0

    match = re.fullmatch(r"group(\d+)_rank(\d+|k)_int8", strategy)
    if match:
        return 1.0 if group == int(match.group(1)) and rank == rank_index(top_k, match.group(2)) else 2.0
    match = re.fullmatch(r"group(\d+)_rank(\d+|k)_int4", strategy)
    if match:
        return 0.5 if group == int(match.group(1)) and rank == rank_index(top_k, match.group(2)) else 2.0
    match = re.fullmatch(r"group(\d+)_rank(\d+|k)_drop", strategy)
    if match:
        return 0.0 if group == int(match.group(1)) and rank == rank_index(top_k, match.group(2)) else 2.0

    raise ValueError(f"unsupported strategy: {strategy}")


def simulate_strategy(
    receiver_df: pd.DataFrame,
    strategy: str,
    hidden_size: int,
    bandwidth_gbps: float,
    per_layer_overhead_us: float,
) -> dict[str, float | str]:
    top_k = int(receiver_df["rank"].max())
    rows = receiver_df.copy()
    rows["bytes_per_element"] = rows.apply(lambda row: bytes_per_element(strategy, row, top_k), axis=1)
    rows["strategy_bytes"] = rows["count"] * hidden_size * rows["bytes_per_element"]
    rows["full_bytes_calc"] = rows["count"] * hidden_size * 2.0

    group_bytes = rows.groupby(["layer", "receiver_group"], as_index=False)["strategy_bytes"].sum()
    layer_bottleneck = group_bytes.groupby("layer")["strategy_bytes"].max()
    bandwidth_bytes_per_us = bandwidth_gbps * 1e9 / 8 / 1e6
    simulated_latency_us = float((layer_bottleneck / bandwidth_bytes_per_us + per_layer_overhead_us).sum())

    full_group_bytes = rows.groupby(["layer", "receiver_group"], as_index=False)["full_bytes_calc"].sum()
    full_layer_bottleneck = full_group_bytes.groupby("layer")["full_bytes_calc"].max()

    return {
        "strategy": strategy,
        "total_bytes": float(rows["strategy_bytes"].sum()),
        "full_total_bytes": float(rows["full_bytes_calc"].sum()),
        "byte_saving": 1.0 - float(rows["strategy_bytes"].sum()) / max(float(rows["full_bytes_calc"].sum()), 1e-12),
        "bottleneck_bytes": float(layer_bottleneck.sum()),
        "full_bottleneck_bytes": float(full_layer_bottleneck.sum()),
        "bottleneck_byte_saving": 1.0
        - float(layer_bottleneck.sum()) / max(float(full_layer_bottleneck.sum()), 1e-12),
        "simulated_latency_us": simulated_latency_us,
    }


def write_report(out: Path, sim_df: pd.DataFrame, accuracy_df: pd.DataFrame | None, args: argparse.Namespace) -> None:
    full_latency = float(sim_df[sim_df["strategy"] == "full"]["simulated_latency_us"].iloc[0])
    full_bytes = float(sim_df[sim_df["strategy"] == "full"]["total_bytes"].iloc[0])
    best_rank = sim_df[sim_df["strategy"].str.fullmatch(r"rank\d+_int4", na=False)].sort_values("simulated_latency_us")

    lines = [
        "# Serving Simulation Report",
        "",
        f"model: `{args.model}`",
        f"bandwidth_gbps: `{args.bandwidth_gbps}`",
        f"per_layer_overhead_us: `{args.per_layer_overhead_us}`",
        "",
        "## Summary",
        "",
        f"- Full BF16 total expert-output bytes: `{full_bytes:.0f}`",
        f"- Full BF16 simulated bottleneck latency: `{full_latency:.3f} us`",
    ]

    if not best_rank.empty:
        row = best_rank.iloc[0]
        lines.extend(
            [
                f"- Best single-rank INT4 latency strategy in this simulation: `{row['strategy']}`",
                f"- Its byte saving: `{row['byte_saving']:.4f}`",
                f"- Its bottleneck-byte saving: `{row['bottleneck_byte_saving']:.4f}`",
            ]
        )

    if accuracy_df is not None:
        merged = sim_df.merge(accuracy_df, on="strategy", how="left", suffixes=("_sim", "_accuracy"))
        rank8 = merged[merged["strategy"] == "rank8_int4"]
        rank1 = merged[merged["strategy"] == "rank1_int4"]
        if not rank8.empty and not rank1.empty:
            lines.extend(
                [
                    "",
                    "## Accuracy-traffic comparison",
                    "",
                    f"- `rank8_int4`: KL `{float(rank8['mean_kl_vs_full'].iloc[0]):.4f}`, simulated byte saving `{float(rank8['byte_saving_sim'].iloc[0]):.4f}`",
                    f"- `rank1_int4`: KL `{float(rank1['mean_kl_vs_full'].iloc[0]):.4f}`, simulated byte saving `{float(rank1['byte_saving_sim'].iloc[0]):.4f}`",
                ]
            )

    lines.extend(["", "See `serving_simulation.csv` and generated figures for the full table."])
    (out / "serving_simulation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(out: Path, sim_df: pd.DataFrame, accuracy_df: pd.DataFrame | None) -> None:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    order = sim_df.sort_values("total_bytes")["strategy"]
    sns.barplot(data=sim_df, x="strategy", y="total_bytes", order=order)
    plt.xticks(rotation=45, ha="right")
    plt.title("Total Expert-Output Bytes by Strategy")
    plt.tight_layout()
    plt.savefig(fig_dir / "serving_total_bytes.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.barplot(data=sim_df, x="strategy", y="simulated_latency_us", order=order)
    plt.xticks(rotation=45, ha="right")
    plt.title("Simulated Receiver-Bottleneck Latency")
    plt.tight_layout()
    plt.savefig(fig_dir / "serving_latency.png", dpi=180)
    plt.close()

    if accuracy_df is not None:
        merged = sim_df.merge(accuracy_df, on="strategy", how="inner", suffixes=("_sim", "_accuracy"))
        if not merged.empty and "mean_kl_vs_full" in merged:
            plt.figure(figsize=(7, 4))
            sns.scatterplot(data=merged, x="byte_saving_sim", y="mean_kl_vs_full", hue="strategy", s=90)
            for _, row in merged.iterrows():
                plt.text(row["byte_saving_sim"], row["mean_kl_vs_full"], row["strategy"], fontsize=8)
            plt.title("Accuracy-Traffic Tradeoff")
            plt.tight_layout()
            plt.savefig(fig_dir / "serving_accuracy_tradeoff_kl.png", dpi=180)
            plt.close()
            merged.to_csv(out / "serving_accuracy_tradeoff.csv", index=False)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out = Path(args.output_dir) if args.output_dir else input_dir
    out.mkdir(parents=True, exist_ok=True)

    receiver_path = input_dir / "receiver_rank_share.csv"
    if not receiver_path.exists():
        raise FileNotFoundError(f"missing receiver profile: {receiver_path}")

    receiver_df = pd.read_csv(receiver_path)
    hidden_size = hidden_size_for_model(args.model)
    rows = [
        simulate_strategy(receiver_df, strategy, hidden_size, args.bandwidth_gbps, args.per_layer_overhead_us)
        for strategy in args.strategies
    ]
    sim_df = pd.DataFrame(rows)
    full_latency = float(sim_df[sim_df["strategy"] == "full"]["simulated_latency_us"].iloc[0])
    sim_df["latency_saving"] = 1.0 - sim_df["simulated_latency_us"] / max(full_latency, 1e-12)
    sim_df.to_csv(out / "serving_simulation.csv", index=False)

    accuracy_path = input_dir / "approx_results.csv"
    accuracy_df = pd.read_csv(accuracy_path) if accuracy_path.exists() else None
    write_report(out, sim_df, accuracy_df, args)
    plot_results(out, sim_df, accuracy_df)
    print(sim_df)


if __name__ == "__main__":
    main()

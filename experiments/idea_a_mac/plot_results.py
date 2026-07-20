from __future__ import annotations

from pathlib import Path
import argparse
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from paths import BASE_OUT


OUT = BASE_OUT
FIG = OUT / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(BASE_OUT))
    return parser.parse_args()


def plot_rank_share() -> None:
    path = OUT / "rank_share_by_layer.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    plt.figure(figsize=(7, 4))
    sns.barplot(data=df, x="layer", y="mean_share", hue="rank")
    plt.title("Mean Contribution Share by Layer and Rank")
    plt.tight_layout()
    plt.savefig(FIG / "rank_contribution_bar.png", dpi=180)
    plt.close()

    top_k = int(df["rank"].max())
    rankk = df[df["rank"] == top_k]
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=rankk.sort_values("layer"), x="layer", y="median_share", marker="o")
    plt.title("Rank-k Median Contribution Share by Layer")
    plt.tight_layout()
    plt.savefig(FIG / "rankk_share_by_layer.png", dpi=180)
    plt.close()


def plot_approx() -> None:
    path = OUT / "approx_results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    plt.figure(figsize=(7, 4))
    sns.scatterplot(data=df, x="byte_saving", y="mean_kl_vs_full", hue="strategy", s=90)
    for _, row in df.iterrows():
        plt.text(row["byte_saving"], row["mean_kl_vs_full"], row["strategy"], fontsize=8)
    plt.title("Accuracy-Byte Pareto (KL)")
    plt.tight_layout()
    plt.savefig(FIG / "accuracy_byte_pareto_kl.png", dpi=180)
    plt.close()

    if "ppl_delta" in df.columns:
        plt.figure(figsize=(7, 4))
        sns.scatterplot(data=df, x="byte_saving", y="ppl_delta", hue="strategy", s=90)
        for _, row in df.iterrows():
            plt.text(row["byte_saving"], row["ppl_delta"], row["strategy"], fontsize=8)
        plt.title("Accuracy-Byte Pareto (PPL Delta)")
        plt.tight_layout()
        plt.savefig(FIG / "accuracy_byte_pareto_ppl.png", dpi=180)
        plt.close()

    sweep_rows = []
    for _, row in df.iterrows():
        match = re.fullmatch(r"rank(\d+)_(int4|drop)", str(row["strategy"]))
        if not match:
            continue
        sweep_rows.append(
            {
                "rank": int(match.group(1)),
                "kind": match.group(2),
                "mean_kl_vs_full": row["mean_kl_vs_full"],
                "local_relative_mse": row["local_relative_mse"],
            }
        )
    if sweep_rows:
        sweep = pd.DataFrame(sweep_rows)
        plt.figure(figsize=(7, 4))
        for kind, group in sweep.groupby("kind"):
            group = group.sort_values("rank")
            plt.plot(group["rank"], group["mean_kl_vs_full"], marker="o", label=kind)
        plt.xlabel("approximated rank")
        plt.ylabel("KL vs full")
        plt.title("All-Rank Sweep: KL vs Approximated Rank")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG / "rank_sweep_kl.png", dpi=180)
        plt.close()

        plt.figure(figsize=(7, 4))
        for kind, group in sweep.groupby("kind"):
            group = group.sort_values("rank")
            plt.plot(group["rank"], group["local_relative_mse"], marker="o", label=kind)
        plt.xlabel("approximated rank")
        plt.ylabel("local relative MSE")
        plt.title("All-Rank Sweep: Local Error vs Approximated Rank")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG / "rank_sweep_local_mse.png", dpi=180)
        plt.close()


def main() -> None:
    global OUT, FIG
    args = parse_args()
    OUT = Path(args.output_dir)
    FIG = OUT / "figures"
    FIG.mkdir(parents=True, exist_ok=True)
    plot_rank_share()
    plot_approx()
    print(f"figures written to {FIG}")


if __name__ == "__main__":
    main()

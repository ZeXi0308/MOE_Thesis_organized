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

import pandas as pd
import torch

from capture_moe import patch_mixtral_moe
from modeling import DEFAULT_MODEL, load_model, load_tokenizer
from paths import resolve_output_dir
from prompts import get_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16", "auto"])
    parser.add_argument("--dataset", default="wikitext2", choices=["builtin", "wikitext2"])
    parser.add_argument("--num-receiver-groups", type=int, default=4)
    parser.add_argument("--receiver-mapping", default="contiguous", choices=["contiguous", "mod"])
    return parser.parse_args()


def summarize_receiver_variability(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (layer, rank), group in df.groupby(["layer", "rank"]):
        medians = group["median_share"]
        traffic = group["full_bytes"]
        mean_median = float(medians.mean())
        std_median = float(medians.std(ddof=0))
        rows.append(
            {
                "layer": int(layer),
                "rank": int(rank),
                "group_median_share_mean": mean_median,
                "group_median_share_min": float(medians.min()),
                "group_median_share_max": float(medians.max()),
                "group_median_share_cv": std_median / max(mean_median, 1e-12),
                "group_median_share_max_over_min": float(medians.max() / max(medians.min(), 1e-12)),
                "traffic_max_over_mean": float(traffic.max() / max(traffic.mean(), 1e-12)),
                "traffic_max_over_min": float(traffic.max() / max(traffic.min(), 1e-12)),
            }
        )
    return pd.DataFrame(rows)


def write_report(args: argparse.Namespace, out, rank_df: pd.DataFrame, receiver_df: pd.DataFrame, variability: pd.DataFrame) -> None:
    top_k = int(rank_df["rank"].max())
    rankk = rank_df[rank_df["rank"] == top_k]
    rankk_median = float(rankk["median_share"].median())
    ratio_median = float(rankk["rank1_over_rankk_median"].median())

    receiver_rankk = receiver_df[receiver_df["rank"] == top_k]
    group_medians = receiver_rankk.groupby("receiver_group")["median_share"].median()
    group_counts = receiver_rankk.groupby("receiver_group")["count"].sum()
    rankk_var = variability[variability["rank"] == top_k]
    median_group_max_over_min = float(rankk_var["group_median_share_max_over_min"].median())
    median_traffic_max_over_mean = float(rankk_var["traffic_max_over_mean"].median())

    if rankk_median < 0.10 and ratio_median > 3:
        c1 = "强成立"
    elif 0.10 <= rankk_median <= 0.20:
        c1 = "弱成立"
    else:
        c1 = "不成立或证据不足"

    group_table = "\n".join(
        f"| {int(group)} | {group_medians[group]:.6f} | {int(group_counts[group])} |"
        for group in group_medians.index
    )

    report = f"""# Receiver-Group Rank Profile Report

model: `{args.model}`
samples: `{args.num_samples}`
dataset: `{args.dataset}`
seq_len: `{args.seq_len}`
dtype: `{args.dtype}`
receiver_groups: `{args.num_receiver_groups}`
receiver_mapping: `{args.receiver_mapping}`

## C1 rank long-tail

- rank-{top_k} median share across layers: `{rankk_median:.6f}`
- rank1/rank{top_k} median ratio across layers: `{ratio_median:.6f}`
- verdict: **{c1}**

## Receiver-group heterogeneity

| receiver_group | rank-{top_k} median share across layers | selected token count |
|---:|---:|---:|
{group_table}

Summary:

- Median max/min receiver-group spread for rank-{top_k} median share: `{median_group_max_over_min:.3f}x`
- Median max/mean receiver traffic imbalance across layers for rank-{top_k}: `{median_traffic_max_over_mean:.3f}x`

Interpretation:

- `layer x rank` explains the global importance trend.
- `receiver_group` explains where that traffic and residual sensitivity lands.
- A static LUT shaped as `layer x receiver_group x rank -> precision` is therefore more deployable than one global rank-only rule when receiver-side congestion matters.
"""
    (out / "receiver_group_profile_report.md").write_text(report, encoding="utf-8")
    print(report)


def main() -> None:
    args = parse_args()
    out = resolve_output_dir(args.model, args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.model)
    model, load_seconds = load_model(args.model, dtype_name=args.dtype)
    print({"model_loaded_seconds": load_seconds, "model": args.model, "dtype": args.dtype}, flush=True)

    recorder = patch_mixtral_moe(
        model,
        policy_name="full",
        num_receiver_groups=args.num_receiver_groups,
        receiver_mapping=args.receiver_mapping,
    )

    prompts = get_prompts(args.dataset, args.num_samples)
    for idx, text in enumerate(prompts, start=1):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.seq_len)
        with torch.no_grad():
            model(**inputs)
        if idx % 32 == 0:
            print({"processed": idx, "total": len(prompts)}, flush=True)

    rank_df = pd.DataFrame(recorder.rank_rows())
    receiver_df = pd.DataFrame(recorder.receiver_rank_rows())
    variability = summarize_receiver_variability(receiver_df)

    rank_df.to_csv(out / "rank_share_by_layer.csv", index=False)
    receiver_df.to_csv(out / "receiver_rank_share.csv", index=False)
    variability.to_csv(out / "receiver_group_variability.csv", index=False)
    write_report(args, out, rank_df, receiver_df, variability)


if __name__ == "__main__":
    main()

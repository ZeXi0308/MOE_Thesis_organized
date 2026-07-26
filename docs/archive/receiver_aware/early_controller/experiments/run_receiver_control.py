"""Same-total-saving receiver control: hot vs uniform vs random vs cold.

Fixes the total byte saving (1 group per layer gets tail-INT4, rest FP8) and
varies WHICH group. Proves receiver selection (not saving amount) drives
max-receiver traffic reduction.

Traffic-only (no model forward) — max-receiver-bytes is computable from profile.
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
import random as rng
from pathlib import Path

import pandas as pd
from transformers import AutoConfig

PRECISION_BYTES = {"bf16": 2.0, "fp8": 1.0, "int4": 0.5}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--profile-csv", default="outputs/main_experiments/olmoe_wikitext256_g4/receiver_rank_share.csv")
    p.add_argument("--output-dir", default="outputs/main_experiments/olmoe_receiver_control")
    return p.parse_args()


def build_lut_selective(profile_df, top_k, num_layers, num_groups, group_selector, n_tail=4):
    """For each layer, group_selector(layer, group_traffic) -> selected group.
    Selected group gets tail-INT4 (rank > top_k-n_tail -> int4, rest fp8).
    Other groups get uniform FP8."""
    lut = {}
    for layer in range(num_layers):
        layer_rows = profile_df[profile_df["layer"] == layer]
        group_traffic = layer_rows.groupby("receiver_group")["full_bytes"].sum()
        selected = group_selector(layer, group_traffic)
        for g in range(num_groups):
            for r in range(1, top_k + 1):
                if g == selected and r > (top_k - n_tail):
                    lut[(layer, g, r)] = "int4"
                else:
                    lut[(layer, g, r)] = "fp8"
    return lut


def compute_traffic(profile_df, lut, top_k, hidden_size):
    rows = profile_df.copy()
    rows["precision"] = rows.apply(
        lambda r: lut.get((int(r["layer"]), int(r["receiver_group"]), int(r["rank"])), "bf16"), axis=1)
    rows["bpe"] = rows["precision"].map(PRECISION_BYTES)
    rows["strategy_bytes"] = rows["count"] * hidden_size * rows["bpe"]
    rows["full_bytes_calc"] = rows["count"] * hidden_size * 2.0
    group_bytes = rows.groupby(["layer", "receiver_group"])["strategy_bytes"].sum().reset_index()
    layer_max = group_bytes.groupby("layer")["strategy_bytes"].max()
    layer_mean = group_bytes.groupby("layer")["strategy_bytes"].mean()
    return {
        "total_bytes": float(rows["strategy_bytes"].sum()),
        "byte_saving": 1.0 - float(rows["strategy_bytes"].sum()) / max(float(rows["full_bytes_calc"].sum()), 1e-12),
        "max_receiver_bytes": float(layer_max.sum()),
        "max_over_mean": float((layer_max / layer_mean.clip(lower=1)).mean()),
        "p95": float(group_bytes["strategy_bytes"].quantile(0.95)),
    }


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = AutoConfig.from_pretrained(args.model)
    hidden_size = int(cfg.hidden_size)
    top_k = int(getattr(cfg, "num_experts_per_tok", 8))
    num_layers = int(cfg.num_hidden_layers)
    num_groups = 4
    profile_df = pd.read_csv(args.profile_csv)

    rng.seed(42)

    selectors = {
        "hot1 (receiver-aware)": lambda l, gt: int(gt.idxmax()),
        "cold1 (anti-receiver-aware)": lambda l, gt: int(gt.idxmin()),
        "roundrobin (uniform)": lambda l, gt: l % num_groups,
        "random": lambda l, gt: rng.choice(list(gt.index)),
    }

    results = []
    for name, selector in selectors.items():
        lut = build_lut_selective(profile_df, top_k, num_layers, num_groups, selector)
        metrics = compute_traffic(profile_df, lut, top_k, hidden_size)
        metrics["strategy"] = name
        results.append(metrics)
        print(f"  {name:30s}  saving={metrics['byte_saving']:.4f}  "
              f"max_recv={metrics['max_receiver_bytes']:.0f}  imbal={metrics['max_over_mean']:.4f}", flush=True)

    df = pd.DataFrame(results)
    # relative max_receiver vs hot1
    hot1_max = float(df[df["strategy"].str.contains("hot1")]["max_receiver_bytes"].iloc[0])
    df["max_recv_vs_hot1"] = df["max_receiver_bytes"] / hot1_max
    df.to_csv(out / "receiver_control_results.csv", index=False)
    print(f"\nsaved to {out}/receiver_control_results.csv", flush=True)
    print(f"\n{'strategy':32s} {'save%':>6s} {'max_recv':>12s} {'vs_hot1':>8s} {'imbal':>6s}")
    for _, r in df.iterrows():
        print(f"{r['strategy']:32s} {r['byte_saving']*100:5.1f}% {r['max_receiver_bytes']:12.0f} "
              f"{r['max_recv_vs_hot1']:7.3f}x {r['max_over_mean']:5.3f}")


if __name__ == "__main__":
    main()

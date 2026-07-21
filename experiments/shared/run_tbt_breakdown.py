"""TBT breakdown: combine traffic as fraction of dispatch+combine and total TBT.

Decomposes per-token-decode latency into:
  - attention compute (QKV proj + attention + output proj)
  - expert FFN compute (dispatch tokens × expert MLP FLOPs)
  - dispatch communication (all-to-all forward)
  - combine communication (all-to-all backward)

Uses real routing profile (token counts per expert) + analytical FLOPs/bytes.
Reports:
  - combine / (dispatch + combine)  — should be ~50% (symmetric)
  - combine / total TBT proxy       — the key metric
  - dispatch+combine / total TBT    — communication fraction
  - P50/P99 breakdown across decode steps (layer-level variability)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoConfig


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--profile-csv", default="outputs/main_experiments/olmoe_wikitext256_g4/receiver_rank_share.csv")
    p.add_argument("--bandwidth-gbps", type=float, default=100.0)
    p.add_argument("--gpu-tflops", type=float, default=312.0,
                   help="GPU peak FP16 TFLOPS (A100 = 312)")
    p.add_argument("--gpu-hbm-tbps", type=float, default=1.55,
                   help="GPU HBM bandwidth TB/s (A100 = 1.55)")
    p.add_argument("--gpu-mflops-util", type=float, default=0.35,
                   help="MFU at low batch (realistic ~0.3-0.4)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--sweep", action="store_true",
                   help="Run parameter sweep instead of single config")
    p.add_argument("--output-dir", default="outputs/main_experiments/olmoe_tbt_breakdown")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = AutoConfig.from_pretrained(args.model)

    H = int(cfg.hidden_size)          # 2048
    L = int(cfg.num_hidden_layers)     # 16
    E = int(cfg.num_experts)           # 64
    K = int(cfg.num_experts_per_tok)   # 8
    I = int(getattr(cfg, "intermediate_size", 1024))  # expert FFN intermediate
    num_heads = int(cfg.num_attention_heads)
    vocab = int(cfg.vocab_size)

    bw_bytes_per_us = args.bandwidth_gbps * 1e9 / 8 / 1e6  # GB/s -> bytes/us
    hbm_bytes_per_us = args.gpu_hbm_tbps * 1e12 / 1e6  # TB/s -> bytes/us

    # Load real profile
    profile = pd.read_csv(args.profile_csv)
    tokens_per_layer = profile.groupby("layer")["count"].sum() / K
    total_tokens = float(tokens_per_layer.sum())
    avg_tokens_per_layer = total_tokens / L

    def compute_breakdown(batch_size, bw_gbps, gpu_tflops, gpu_hbm_tbps, mfu):
        bw_bpus = bw_gbps * 1e9 / 8 / 1e6
        hbm_bpus = gpu_hbm_tbps * 1e12 / 1e6
        flops_bpus = gpu_tflops * 1e12 / 1e6 * mfu  # realistic flops with MFU

        B = batch_size

        # --- FLOPs per layer per decode batch ---
        # Weight reads are largely amortized across the batch, but arithmetic
        # scales with the number of active decode tokens.
        attn_flops = B * 4 * H * H  # QKV + O proj
        expert_flops = B * K * 2 * H * I * 3  # 3 linear layers (gate+up+down), 2 for MAC
        router_flops = B * E * H
        head_flops = B * vocab * H / L

        # --- Memory bytes for compute (weight reads) ---
        # Expert weights: expected unique experts touched by B*K routed tokens.
        # This is conservative for skewed routing but avoids treating expert
        # weight reads as constant when batch grows.
        expected_unique_experts = E * (1.0 - (1.0 - 1.0 / E) ** (B * K))
        unique_experts = min(E, max(K, expected_unique_experts))
        expert_weight_bytes = unique_experts * 3 * H * I * 2
        # Attn weights: 4*H*H × 2
        attn_weight_bytes = 4 * H * H * 2
        # Router weights: E*H × 2
        router_weight_bytes = E * H * 2

        # --- Communication bytes per layer ---
        dispatch_bytes = B * K * H * 2
        combine_bytes = B * K * H * 2

        # --- Times: compute is max(flops-bound, memory-bound) ---
        attn_compute = max(attn_flops / flops_bpus, attn_weight_bytes / hbm_bpus)
        expert_compute = max(expert_flops / flops_bpus, expert_weight_bytes / hbm_bpus)
        router_compute = max(router_flops / flops_bpus, router_weight_bytes / hbm_bpus)
        head_compute = head_flops / flops_bpus

        dispatch_t = dispatch_bytes / bw_bpus
        combine_t = combine_bytes / bw_bpus

        total_compute = (attn_compute + expert_compute + router_compute + head_compute) * L
        total_dispatch = dispatch_t * L
        total_combine = combine_t * L
        total_comm = total_dispatch + total_combine
        total_tbt = total_compute + total_comm

        return {
            "batch": B, "bw_gbps": bw_gbps, "gpu_tflops": gpu_tflops,
            "gpu_hbm_tbps": gpu_hbm_tbps, "mfu": mfu,
            "attn_us": attn_compute * L,
            "expert_us": expert_compute * L,
            "router_us": router_compute * L,
            "head_us": head_compute * L,
            "dispatch_us": total_dispatch,
            "combine_us": total_combine,
            "compute_us": total_compute,
            "comm_us": total_comm,
            "total_tbt_us": total_tbt,
            "combine_frac_tbt": total_combine / max(total_tbt, 1e-12) * 100,
            "comm_frac_tbt": total_comm / max(total_tbt, 1e-12) * 100,
            "unique_experts_est": unique_experts,
            "expert_bound_by": "memory" if expert_weight_bytes / hbm_bpus > expert_flops / flops_bpus else "compute",
        }

    if args.sweep:
        sweep_rows = []
        for B in [1, 4, 8, 16, 32, 64]:
            for bw in [25, 50, 100, 200, 400, 800, 1600, 4800]:
                for tflops in [150, 312, 750]:
                    for hbm in [1.0, 1.55, 2.0]:
                        r = compute_breakdown(B, bw, tflops, hbm, 0.35)
                        sweep_rows.append(r)
        sweep_df = pd.DataFrame(sweep_rows)
        sweep_df.to_csv(out / "tbt_sweep.csv", index=False)

        print(f"=== TBT Sweep: combine % of total TBT ===\n")
        pivot = sweep_df.pivot_table(
            index="batch", columns=["bw_gbps", "gpu_tflops"], values="combine_frac_tbt")
        print("A100-like (312 TF, 1.55 HBM, MFU=0.35):")
        a100 = sweep_df[(sweep_df["gpu_tflops"]==312) & (sweep_df["gpu_hbm_tbps"]==1.55)]
        for bw in [25, 50, 100, 200, 400, 800, 1600, 4800]:
            sub = a100[a100["bw_gbps"]==bw]
            print(f"  BW={bw}Gbps: " + ", ".join(f"B={int(r['batch'])}={r['combine_frac_tbt']:.1f}%" for _, r in sub.iterrows()))

        print(f"\n=== Expert compute bound by ===")
        for B in [1, 8, 32, 64]:
            r = compute_breakdown(B, 100, 312, 1.55, 0.35)
            print(f"  B={B}: expert={r['expert_us']:.2f}us ({r['expert_bound_by']}-bound), "
                  f"unique_experts≈{r['unique_experts_est']:.1f}, "
                  f"dispatch={r['dispatch_us']:.2f}us, combine={r['combine_us']:.2f}us, "
                  f"combine/TBT={r['combine_frac_tbt']:.1f}%")

        print(f"\nSaved sweep to {out}/tbt_sweep.csv ({len(sweep_df)} configs)")
        return

    # --- Single config with detailed breakdown ---
    B = args.batch_size
    r = compute_breakdown(B, args.bandwidth_gbps, args.gpu_tflops, args.gpu_hbm_tbps, args.gpu_mflops_util)

    # --- Summary ---
    print(f"=== OLMoE-1B-7B TBT Breakdown (decode, batch={B}, BW={args.bandwidth_gbps}Gbps, GPU={args.gpu_tflops}TFLOPS, MFU={args.gpu_mflops_util}) ===\n")
    print(f"Config: H={H}, L={L}, K={K}, E={E}, I={I}, HBM={args.gpu_hbm_tbps}TB/s")
    print(f"Expert compute: {r['expert_bound_by']}-bound (weight read > FLOPs at low batch)\n")

    print(f"{'Component':25s} {'total_us':>10s} {'%TBT':>6s}")
    print(f"{'-'*45}")
    for name, total in [
        ("Attention compute", r["attn_us"]),
        ("Expert FFN compute", r["expert_us"]),
        ("Router", r["router_us"]),
        ("LM head (amortized)", r["head_us"]),
        ("Dispatch comm", r["dispatch_us"]),
        ("Combine comm", r["combine_us"]),
    ]:
        print(f"{name:25s} {total:10.3f} {total/r['total_tbt_us']*100:5.1f}%")
    print(f"{'-'*45}")
    print(f"{'Total compute':25s} {r['compute_us']:10.3f} {r['compute_us']/r['total_tbt_us']*100:5.1f}%")
    print(f"{'Total comm (d+c)':25s} {r['comm_us']:10.3f} {r['comm_us']/r['total_tbt_us']*100:5.1f}%")
    print(f"{'TOTAL TBT proxy':25s} {r['total_tbt_us']:10.3f} {'100.0%':>6s}")

    print(f"\n=== Key Ratios ===")
    print(f"combine / (dispatch + combine)     = {r['combine_us']/max(r['comm_us'],1e-12)*100:.1f}%  (symmetric)")
    print(f"combine / total TBT                 = {r['combine_frac_tbt']:.1f}%  *** KEY METRIC ***")
    print(f"dispatch+combine / total TBT         = {r['comm_frac_tbt']:.1f}%  (communication fraction)")

    # --- Per-layer variability (P50/P99) ---
    # Use real profile to compute per-layer dispatch/combine bytes (they vary by token load)
    dispatch_bytes_per_layer = B * K * H * 2
    combine_bytes_per_layer = B * K * H * 2
    layer_stats = []
    for layer in range(L):
        layer_rows = profile[profile["layer"] == layer]
        tokens_this_layer = float(layer_rows["count"].sum()) / K
        # Scale bytes by actual token load relative to average
        load_factor = tokens_this_layer / avg_tokens_per_layer if avg_tokens_per_layer > 0 else 1.0
        d_bytes = dispatch_bytes_per_layer * load_factor
        c_bytes = combine_bytes_per_layer * load_factor
        d_time = d_bytes / bw_bytes_per_us
        c_time = c_bytes / bw_bytes_per_us
        # Per-layer receiver imbalance. The profile counts are accumulated over
        # all sampled tokens, so normalize them into receiver shares and then
        # apply those shares to the current serving batch's combine bytes.
        group_counts = layer_rows.groupby("receiver_group")["count"].sum()
        group_shares = group_counts / max(float(group_counts.sum()), 1e-12)
        group_bytes = group_shares * c_bytes
        max_recv_bytes = float(group_bytes.max())
        mean_recv_bytes = float(group_bytes.mean())
        max_recv_time = max_recv_bytes / bw_bytes_per_us
        imbalance_ratio = max_recv_bytes / max(mean_recv_bytes, 1e-12)
        layer_stats.append({
            "layer": layer,
            "tokens": tokens_this_layer,
            "dispatch_us": d_time,
            "combine_us": c_time,
            "max_receiver_combine_us": max_recv_time,
            "combine_frac_of_comm": c_time / max(d_time + c_time, 1e-12),
            "max_recv_over_mean_recv": imbalance_ratio,
        })

    layer_df = pd.DataFrame(layer_stats)
    layer_df.to_csv(out / "per_layer_breakdown.csv", index=False)

    p50_combine = float(layer_df["combine_us"].quantile(0.5))
    p99_combine = float(layer_df["combine_us"].quantile(0.99))
    # Max-receiver imbalance: per-layer max-receiver vs per-layer mean-receiver (same layer)
    layer_df["mean_receiver_combine_us"] = layer_df["max_receiver_combine_us"] / layer_df["max_recv_over_mean_recv"].clip(lower=1e-12)
    p50_imbalance = float(layer_df["max_recv_over_mean_recv"].quantile(0.5))
    p99_imbalance = float(layer_df["max_recv_over_mean_recv"].quantile(0.99))
    p50_maxrecv = float(layer_df["max_receiver_combine_us"].quantile(0.5))
    p99_maxrecv = float(layer_df["max_receiver_combine_us"].quantile(0.99))

    print(f"\n=== Per-layer variability (from real profile) ===")
    print(f"{'':35s} {'P50':>10s} {'P99':>10s} {'P99/P50':>8s}")
    print(f"{'Combine comm (us)':35s} {p50_combine:10.3f} {p99_combine:10.3f} {p99_combine/max(p50_combine,1e-12):7.2f}x")
    print(f"{'Max-receiver comm (us)':35s} {p50_maxrecv:10.1f} {p99_maxrecv:10.1f} {p99_maxrecv/max(p50_maxrecv,1e-12):7.2f}x")
    print(f"{'Max-receiver / mean-receiver (imbalance)':35s} {p50_imbalance:10.3f} {p99_imbalance:10.3f} {p99_imbalance/max(p50_imbalance,1e-12):7.2f}x")
    print(f"Max-receiver P99/P50 across layers = {p99_maxrecv/max(p50_maxrecv,1e-12):.2f}x  (tail inflation)")

    # Save summary
    summary = {
        "config": {"H": H, "L": L, "K": K, "E": E, "I": I,
                    "batch": B, "bw_gbps": args.bandwidth_gbps, "gpu_tflops": args.gpu_tflops,
                    "gpu_hbm_tbps": args.gpu_hbm_tbps, "mfu": args.gpu_mflops_util},
        "combine_over_dispatch_plus_combine_pct": r["combine_us"] / r["comm_us"] * 100,
        "combine_over_total_tbt_pct": r["combine_us"] / r["total_tbt_us"] * 100,
        "comm_over_total_tbt_pct": r["comm_us"] / r["total_tbt_us"] * 100,
        "compute_over_total_tbt_pct": r["compute_us"] / r["total_tbt_us"] * 100,
        "total_tbt_us": r["total_tbt_us"],
        "combine_us": r["combine_us"],
        "dispatch_us": r["dispatch_us"],
        "max_receiver_p99_over_p50": p99_maxrecv / max(p50_maxrecv, 1e-12),
    }
    with open(out / "tbt_breakdown_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {out}/")
    verdict = "很弱" if summary["combine_over_total_tbt_pct"] < 5 else \
              "勉强" if summary["combine_over_total_tbt_pct"] < 10 else \
              "可以讲" if summary["combine_over_total_tbt_pct"] < 20 else "很有说服力"
    print(f"\n>>> combine 占 total TBT = {summary['combine_over_total_tbt_pct']:.1f}% → {verdict}")


if __name__ == "__main__":
    main()

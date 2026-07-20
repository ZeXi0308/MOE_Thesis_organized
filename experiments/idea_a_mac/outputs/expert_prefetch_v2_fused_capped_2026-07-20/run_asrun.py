#!/usr/bin/env python3
"""Expert-Prefetch System Prototype v2: replace the per-expert Python-loop
compute proxy (64 separate kernel launches/layer, ~10ms) with a batched/fused
compute proxy (stack all experts' weights into one tensor, use `torch.bmm` --
O(3) kernel launches/layer total, matching the spirit of a real grouped-GEMM
MoE kernel like DeepEP/Triton-fused-MoE, at the cost of wasting FLOPs on
tokens*experts combinations that are not actually routed together).

Everything else (H2D bandwidth measurement, real route data, LRU cache +
predictive-prefetch simulation) is identical to
`run_expert_prefetch_system_prototype.py` -- only `layer_compute_s` changes.
This isolates the effect of a more realistic MoE kernel implementation on how
much of the P0-C hit-rate signal survives as end-to-end latency saving.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from run_expert_prefetch_system_prototype import (
    build_freq_fallback,
    build_transition_table,
    measure_h2d_bandwidth,
    paired_bootstrap_ci,
    simulate_document,
)


def measure_layer_compute_time_fused(model, layer_id: int, batch_tokens: int, hidden_size: int,
                                      device: str, n_repeats: int = 30) -> float:
    """Dense batched compute: stack ALL experts' gate/up/down weights into
    [num_experts, out, in] tensors, run the FULL batch through ALL experts at
    once via bmm (3 kernel launches total instead of num_experts), then keep
    only each token's actually-assigned expert output. Wastes FLOPs (computes
    every token x every expert) but eliminates per-expert kernel-launch
    overhead -- the same tradeoff a real grouped-GEMM kernel makes when it
    pads/groups tokens instead of looping."""
    layer = model.model.layers[layer_id]
    moe = layer.mlp if (hasattr(layer, "mlp") and hasattr(layer.mlp, "experts")) else layer.block_sparse_moe
    experts = moe.experts
    num_experts = len(experts)
    dtype = next(moe.parameters()).dtype

    if hasattr(experts[0], "gate_proj"):
        gate_attr, up_attr, down_attr = "gate_proj", "up_proj", "down_proj"
    else:
        gate_attr, up_attr, down_attr = "w1", "w3", "w2"  # Mixtral-style naming
    gate_w = torch.stack([getattr(e, gate_attr).weight.data for e in experts], dim=0)  # [E, inter, hidden]
    up_w = torch.stack([getattr(e, up_attr).weight.data for e in experts], dim=0)      # [E, inter, hidden]
    down_w = torch.stack([getattr(e, down_attr).weight.data for e in experts], dim=0)  # [E, hidden, inter]
    act_fn = experts[0].act_fn

    x = torch.randn(batch_tokens, hidden_size, dtype=dtype, device=device)
    top_k = int(model.config.num_experts_per_tok)
    experts_per_token = torch.randint(0, num_experts, (batch_tokens, top_k), device=device)

    def run_once():
        # broadcast the batch to every expert: [E, batch_tokens, hidden]
        x_exp = x.unsqueeze(0).expand(num_experts, -1, -1)
        gate_out = torch.bmm(x_exp, gate_w.transpose(1, 2))   # [E, B, inter]
        up_out = torch.bmm(x_exp, up_w.transpose(1, 2))       # [E, B, inter]
        hidden = act_fn(gate_out) * up_out
        down_out = torch.bmm(hidden, down_w.transpose(1, 2))  # [E, B, hidden]
        # gather each token's top_k assigned experts' outputs (mirrors what
        # the per-expert loop version returns, for an apples-to-apples cost
        # comparison of the compute PATH, not a claim that this is FLOP-optimal)
        gathered = down_out[experts_per_token.t(), torch.arange(batch_tokens, device=device).unsqueeze(0).expand(top_k, -1)]
        return gathered

    for _ in range(5):
        run_once()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        run_once()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)
    return float(np.median(times))


def run_model(model_key, model_name, calib_csv, test_csv, cache_capacity, prefetch_budget,
              n_boot, seed, device, batch_tokens):
    print(f"[{model_key}] loading model + measuring real hardware constants (fused compute)...")
    cfg = AutoConfig.from_pretrained(model_name, local_files_only=True)
    hidden_size = int(cfg.hidden_size)
    intermediate_size = int(getattr(cfg, "moe_intermediate_size", getattr(cfg, "intermediate_size", None)))
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, local_files_only=True).to(device)
    model.eval()
    num_layers = len(model.model.layers)

    h2d_time_s, expert_bytes = measure_h2d_bandwidth(hidden_size, intermediate_size, dtype, device)
    layer_compute_s = {}
    for layer_id in range(num_layers):
        layer_compute_s[layer_id] = measure_layer_compute_time_fused(model, layer_id, batch_tokens, hidden_size, device)
    del model
    torch.cuda.empty_cache()

    print(f"[{model_key}] h2d_per_expert={h2d_time_s * 1e6:.2f}us ({expert_bytes / 1e6:.2f}MB), "
          f"fused_layer_compute_median={np.median(list(layer_compute_s.values())) * 1e6:.2f}us")

    # overlap-capped runtime prefetch budget: never attempt to prefetch more
    # candidates than can be FULLY hidden inside the (now much shorter)
    # compute window -- otherwise the exposed excess prefetch time makes
    # predictive WORSE than reactive, as the naive fixed-budget=8 version does.
    median_compute = float(np.median(list(layer_compute_s.values())))
    overlap_capped_budget = max(1, int(median_compute // max(h2d_time_s, 1e-12)))
    runtime_budget = min(prefetch_budget, overlap_capped_budget)
    print(f"[{model_key}] overlap-capped runtime prefetch budget = {runtime_budget} "
          f"(vs uncapped {prefetch_budget}; overlap allows {overlap_capped_budget} free copies/layer)")

    calib = pd.read_csv(calib_csv)
    test = pd.read_csv(test_csv)
    calib_top1 = calib[calib["rank"] == 1][["sample_id", "token_position", "layer", "expert_id"]].rename(
        columns={"expert_id": "top1_expert"})
    test_top1 = test[test["rank"] == 1][["sample_id", "token_position", "layer", "expert_id"]].rename(
        columns={"expert_id": "top1_expert"})

    trans_table = build_transition_table(calib_top1, num_layers, prefetch_budget)
    freq_fallback = build_freq_fallback(calib_top1, num_layers, prefetch_budget)

    rows = []
    for sample_id, doc in test_top1.groupby("sample_id"):
        all_positions = sorted(doc["token_position"].unique().tolist())
        chunks = [all_positions[i:i + batch_tokens] for i in range(0, len(all_positions), batch_tokens)]
        for chunk_idx, chunk in enumerate(chunks):
            if len(chunk) < max(4, batch_tokens // 4):
                continue
            subset = set(chunk)
            r_react = simulate_document(doc, num_layers, cache_capacity, prefetch_budget, trans_table,
                                         freq_fallback, h2d_time_s, layer_compute_s, "reactive",
                                         token_subset=subset)
            r_pred = simulate_document(doc, num_layers, cache_capacity, prefetch_budget, trans_table,
                                        freq_fallback, h2d_time_s, layer_compute_s, "predictive",
                                        token_subset=subset)
            r_pred_capped = simulate_document(doc, num_layers, cache_capacity, runtime_budget, trans_table,
                                               freq_fallback, h2d_time_s, layer_compute_s, "predictive",
                                               token_subset=subset)
            rows.append({
                "sample_id": sample_id, "chunk_idx": chunk_idx, "batch_size": len(chunk),
                "reactive_latency_s": r_react["total_latency_s"],
                "predictive_latency_s": r_pred["total_latency_s"],
                "predictive_capped_latency_s": r_pred_capped["total_latency_s"],
                "reactive_paging_latency_s": r_react["total_paging_latency_s"],
                "predictive_paging_latency_s": r_pred["total_paging_latency_s"],
                "reactive_miss_rate": r_react["miss_rate"], "predictive_miss_rate": r_pred["miss_rate"],
                "predictive_capped_miss_rate": r_pred_capped["miss_rate"],
                "latency_saving_pct": 1.0 - r_pred["total_latency_s"] / max(r_react["total_latency_s"], 1e-12),
                "latency_saving_capped_pct": 1.0 - r_pred_capped["total_latency_s"] / max(r_react["total_latency_s"], 1e-12),
                "paging_latency_saving_pct": 1.0 - r_pred["total_paging_latency_s"] / max(
                    r_react["total_paging_latency_s"], 1e-12),
            })
    df = pd.DataFrame(rows)
    diffs = df.groupby("sample_id")["latency_saving_pct"].mean().to_numpy()
    lo, hi, mean_diff = paired_bootstrap_ci(diffs, n_boot, seed)
    diffs_capped = df.groupby("sample_id")["latency_saving_capped_pct"].mean().to_numpy()
    lo_c, hi_c, mean_diff_c = paired_bootstrap_ci(diffs_capped, n_boot, seed + 2000)

    summary = {
        "model": model_key, "cache_capacity": cache_capacity, "prefetch_budget": prefetch_budget,
        "runtime_capped_budget": runtime_budget,
        "h2d_time_us": h2d_time_s * 1e6, "fused_layer_compute_us_median": float(np.median(list(layer_compute_s.values())) * 1e6),
        "mean_reactive_latency_ms": float(df["reactive_latency_s"].mean() * 1000),
        "mean_predictive_latency_ms": float(df["predictive_latency_s"].mean() * 1000),
        "mean_latency_saving_pct": mean_diff * 100, "ci_low_pct": lo * 100, "ci_high_pct": hi * 100,
        "mean_latency_saving_capped_pct": mean_diff_c * 100, "ci_low_capped_pct": lo_c * 100, "ci_high_capped_pct": hi_c * 100,
        "mean_reactive_miss_rate": float(df["reactive_miss_rate"].mean()),
        "mean_predictive_miss_rate": float(df["predictive_miss_rate"].mean()),
        "mean_predictive_capped_miss_rate": float(df["predictive_capped_miss_rate"].mean()),
        "n_docs": len(df),
    }
    return df, pd.DataFrame([summary]), summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--olmoe-model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--olmoe-calib", required=True)
    ap.add_argument("--olmoe-test", required=True)
    ap.add_argument("--olmoe-cache-capacity", type=int, default=8)
    ap.add_argument("--olmoe-prefetch-budget", type=int, default=8)
    ap.add_argument("--llmjp-model", default="llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M")
    ap.add_argument("--llmjp-calib", required=True)
    ap.add_argument("--llmjp-test", required=True)
    ap.add_argument("--llmjp-cache-capacity", type=int, default=6)
    ap.add_argument("--llmjp-prefetch-budget", type=int, default=6)
    ap.add_argument("--batch-tokens", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_o, summ_o, s_o = run_model("olmoe", args.olmoe_model, Path(args.olmoe_calib), Path(args.olmoe_test),
                                   args.olmoe_cache_capacity, args.olmoe_prefetch_budget, args.n_boot, args.seed,
                                   device, args.batch_tokens)
    df_l, summ_l, s_l = run_model("llmjp", args.llmjp_model, Path(args.llmjp_calib), Path(args.llmjp_test),
                                   args.llmjp_cache_capacity, args.llmjp_prefetch_budget, args.n_boot, args.seed + 1,
                                   device, args.batch_tokens)

    df_o.to_csv(out / "olmoe_per_document.csv", index=False)
    df_l.to_csv(out / "llmjp_per_document.csv", index=False)
    summary = pd.concat([summ_o, summ_l], ignore_index=True)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "meta.json").write_text(json.dumps({"olmoe": s_o, "llmjp": s_l}, indent=2), encoding="utf-8")

    lines = ["# Expert-Prefetch System Prototype v2 (fused/batched compute)", ""]
    cols = ["model", "cache_capacity", "prefetch_budget", "runtime_capped_budget", "h2d_time_us",
            "fused_layer_compute_us_median",
            "mean_reactive_latency_ms", "mean_predictive_latency_ms", "mean_latency_saving_pct",
            "ci_low_pct", "ci_high_pct",
            "mean_latency_saving_capped_pct", "ci_low_capped_pct", "ci_high_capped_pct",
            "mean_reactive_miss_rate", "mean_predictive_miss_rate", "mean_predictive_capped_miss_rate", "n_docs"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in summary.iterrows():
        vals = [f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()

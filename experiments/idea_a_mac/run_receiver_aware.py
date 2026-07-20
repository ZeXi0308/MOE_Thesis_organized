"""Receiver-aware FP8+tail-INT4 evaluation with TBT proxy.

Closes the receiver/TBT loop the advisor asked about:
  - rank solves "which ranks to safely compress" (accuracy)
  - receiver solves "which compressions most reduce latency" (system)

Compares per-receiver traffic + TBT proxy under:
  1. uniform_fp8          — all receivers FP8 (50% saving, same load)
  2. fp8top4_rest_int4    — all receivers tail-INT4 (62.5% saving, same load)
  3. receiver_aware_hot1  — hottest 1 receiver/group per layer gets tail-INT4
  4. receiver_aware_hot2  — hottest 2 receivers/group per layer get tail-INT4

Metrics: max_receiver_bytes, p95, max/mean imbalance, TBT proxy, PPL, KL.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoConfig

from capture_moe import patch_mixtral_moe
from modeling import load_model, load_tokenizer
from prompts import get_prompts

PRECISION_BYTES = {"bf16": 2.0, "fp8": 1.0, "int8": 1.0, "int4": 0.5, "drop": 0.0}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--profile-csv", default="outputs/main_experiments/olmoe_wikitext256_g4/receiver_rank_share.csv")
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--dataset", default="wikitext2")
    p.add_argument("--num-receiver-groups", type=int, default=4)
    p.add_argument("--bandwidth-gbps", type=float, default=100.0)
    p.add_argument("--quant-overhead-us", type=float, default=5.0,
                   help="per-layer quant/dequant overhead in microseconds")
    p.add_argument("--output-dir", default="outputs/main_experiments/olmoe_receiver_aware")
    return p.parse_args()


def build_lut_uniform(top_k, num_layers, num_groups, precisions):
    """Same precision for all (layer, group, rank)."""
    return {(l, g, r): precisions[r - 1]
            for l in range(num_layers) for g in range(num_groups) for r in range(1, top_k + 1)}


def build_lut_receiver_aware(profile_df, top_k, num_layers, num_groups, n_hot,
                              head_prec="fp8", tail_prec="int4", n_tail=4):
    """Hot receivers (top-n_hot per layer) get tail-INT4; cold get uniform FP8."""
    lut = {}
    for layer in range(num_layers):
        layer_rows = profile_df[profile_df["layer"] == layer]
        group_traffic = layer_rows.groupby("receiver_group")["full_bytes"].sum()
        hot_groups = set(group_traffic.nlargest(min(n_hot, num_groups)).index)
        for g in range(num_groups):
            for r in range(1, top_k + 1):
                if g in hot_groups and r > (top_k - n_tail):
                    lut[(layer, g, r)] = tail_prec
                else:
                    lut[(layer, g, r)] = head_prec
    return lut


def compute_traffic_metrics(profile_df, lut, top_k, hidden_size, bandwidth_gbps, quant_overhead_us, num_layers):
    """Compute per-receiver traffic + TBT proxy from profile + LUT."""
    rows = profile_df.copy()
    rows["precision"] = rows.apply(
        lambda row: lut.get((int(row["layer"]), int(row["receiver_group"]), int(row["rank"])), "bf16"), axis=1)
    rows["bpe"] = rows["precision"].map(PRECISION_BYTES)
    rows["strategy_bytes"] = rows["count"] * hidden_size * rows["bpe"]
    rows["full_bytes_calc"] = rows["count"] * hidden_size * 2.0

    # Per (layer, group) bytes
    group_bytes = rows.groupby(["layer", "receiver_group"])["strategy_bytes"].sum().reset_index()
    full_group_bytes = rows.groupby(["layer", "receiver_group"])["full_bytes_calc"].sum().reset_index()

    # Per-layer max receiver (bottleneck)
    layer_max = group_bytes.groupby("layer")["strategy_bytes"].max()
    full_layer_max = full_group_bytes.groupby("layer")["full_bytes_calc"].max()

    # Per-layer max/mean imbalance
    layer_mean = group_bytes.groupby("layer")["strategy_bytes"].mean()
    imbalance = (layer_max / layer_mean.clip(lower=1)).mean()

    # P95 of receiver bytes across all (layer, group)
    p95 = float(group_bytes["strategy_bytes"].quantile(0.95))

    # TBT proxy: sum of per-layer max-receiver-bytes / bandwidth + per-layer overhead
    bw_bytes_per_us = bandwidth_gbps * 1e9 / 8 / 1e6
    total_max = float(layer_max.sum())
    full_total_max = float(full_layer_max.sum())
    tbt_proxy_us = total_max / bw_bytes_per_us + num_layers * quant_overhead_us
    full_tbt_us = full_total_max / bw_bytes_per_us + num_layers * quant_overhead_us

    total_bytes = float(rows["strategy_bytes"].sum())
    full_total = float(rows["full_bytes_calc"].sum())

    return {
        "total_bytes": total_bytes,
        "byte_saving": 1.0 - total_bytes / max(full_total, 1e-12),
        "max_receiver_bytes": total_max,
        "full_max_receiver_bytes": full_total_max,
        "max_receiver_saving": 1.0 - total_max / max(full_total_max, 1e-12),
        "p95_receiver_bytes": p95,
        "max_over_mean_imbalance": float(imbalance),
        "tbt_proxy_us": tbt_proxy_us,
        "full_tbt_proxy_us": full_tbt_us,
        "tbt_saving": 1.0 - tbt_proxy_us / max(full_tbt_us, 1e-12),
    }


def tokenized_inputs(tokenizer, texts, seq_len):
    for text in texts:
        yield tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)


def compute_ppl(logits, input_ids):
    if logits.shape[1] < 2:
        return float("nan")
    sl = logits[:, :-1, :].contiguous().float()
    sy = input_ids[:, 1:].contiguous()
    return float(torch.exp(F.cross_entropy(sl.view(-1, sl.size(-1)), sy.view(-1))).item())


def compute_kl(full_logits, approx_logits):
    full = full_logits[:, :-1, :].contiguous().float()
    approx = approx_logits[:, :-1, :].contiguous().float()
    p = F.softmax(full, dim=-1)
    log_q = F.log_softmax(approx, dim=-1)
    return float(F.kl_div(log_q, p, reduction="batchmean").item())


def run_lut_strategy(model, tokenizer, name, lut, texts, seq_len, baseline_logits,
                     num_groups, receiver_mapping):
    recorder = patch_mixtral_moe(model, "lut", num_receiver_groups=num_groups,
                                 receiver_mapping=receiver_mapping, lut=lut)
    ppls, kls = [], []
    for idx, inputs in enumerate(tokenized_inputs(tokenizer, texts, seq_len)):
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.detach().cpu()
        ppls.append(compute_ppl(logits, inputs["input_ids"]))
        kls.append(compute_kl(baseline_logits[idx], logits))
    return {"strategy": name, "mean_ppl": sum(ppls)/len(ppls), "mean_kl": sum(kls)/len(kls)}


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = AutoConfig.from_pretrained(args.model)
    hidden_size = int(cfg.hidden_size)
    top_k = int(getattr(cfg, "num_experts_per_tok", 8))
    num_layers = int(cfg.num_hidden_layers)
    num_groups = args.num_receiver_groups

    profile_df = pd.read_csv(args.profile_csv)
    print(f"profile: {len(profile_df)} rows, {num_layers} layers, {num_groups} groups, top_k={top_k}", flush=True)

    # Define strategies
    fp8_all = ["fp8"] * top_k
    tail_int4 = ["fp8"] * (top_k - 4) + ["int4"] * 4  # rank1-4 FP8, rank5-8 INT4

    strategies = {
        "uniform_fp8": build_lut_uniform(top_k, num_layers, num_groups, fp8_all),
        "fp8top4_rest_int4": build_lut_uniform(top_k, num_layers, num_groups, tail_int4),
        "recv_aware_hot1": build_lut_receiver_aware(profile_df, top_k, num_layers, num_groups, n_hot=1),
        "recv_aware_hot2": build_lut_receiver_aware(profile_df, top_k, num_layers, num_groups, n_hot=2),
    }

    # 1. Traffic metrics (from profile, no model forward)
    print("\n=== Traffic Metrics ===", flush=True)
    traffic_rows = []
    for name, lut in strategies.items():
        metrics = compute_traffic_metrics(profile_df, lut, top_k, hidden_size,
                                          args.bandwidth_gbps, args.quant_overhead_us, num_layers)
        metrics["strategy"] = name
        traffic_rows.append(metrics)
        print(f"  {name:25s}  saving={metrics['byte_saving']:.3f}  max_recv_saving={metrics['max_receiver_saving']:.3f}  "
              f"tbt_saving={metrics['tbt_saving']:.3f}  imbalance={metrics['max_over_mean_imbalance']:.3f}", flush=True)
    traffic_df = pd.DataFrame(traffic_rows)

    # 2. Accuracy (model forward)
    print("\n=== Accuracy (model forward) ===", flush=True)
    texts = get_prompts(args.dataset, args.num_samples)
    tokenizer = load_tokenizer(args.model)
    model, load_s = load_model(args.model, dtype_name=args.dtype)
    print(f"model loaded in {load_s:.1f}s", flush=True)

    # baseline
    patch_mixtral_moe(model, "full", num_receiver_groups=num_groups, receiver_mapping="contiguous")
    baseline_logits, ppls_full = [], []
    for inputs in tokenized_inputs(tokenizer, texts, args.seq_len):
        with torch.no_grad():
            outputs = model(**inputs)
        baseline_logits.append(outputs.logits.detach().cpu())
        ppls_full.append(compute_ppl(baseline_logits[-1], inputs["input_ids"]))
    full_ppl = sum(ppls_full) / len(ppls_full)
    print(f"  full: PPL={full_ppl:.4f}", flush=True)

    acc_rows = [{"strategy": "full", "mean_ppl": full_ppl, "mean_kl": 0.0}]
    for name, lut in strategies.items():
        row = run_lut_strategy(model, tokenizer, name, lut, texts, args.seq_len,
                               baseline_logits, num_groups, "contiguous")
        row["ppl_delta"] = row["mean_ppl"] - full_ppl
        acc_rows.append(row)
        print(f"  {name:25s}  PPL={row['mean_ppl']:.4f}  KL={row['mean_kl']:.4f}  dPPL={row['ppl_delta']:+.4f}", flush=True)

    acc_df = pd.DataFrame(acc_rows)

    # 3. Merge + save
    merged = traffic_df.merge(acc_df, on="strategy", how="left")
    merged.to_csv(out / "receiver_aware_results.csv", index=False)
    print(f"\nsaved to {out}/receiver_aware_results.csv", flush=True)

    # 4. Summary table
    print("\n=== Summary ===", flush=True)
    print(f"{'strategy':25s} {'save%':>6s} {'maxRecv%':>8s} {'tbt%':>6s} {'imbal':>6s} {'PPL_d':>7s} {'KL':>7s}", flush=True)
    for _, r in merged.iterrows():
        print(f"{r['strategy']:25s} {r['byte_saving']*100:5.1f}% {r['max_receiver_saving']*100:7.1f}% "
              f"{r['tbt_saving']*100:5.1f}% {r['max_over_mean_imbalance']:5.3f} "
              f"{r.get('ppl_delta', 0):+6.3f} {r.get('mean_kl', 0):6.3f}", flush=True)


if __name__ == "__main__":
    main()

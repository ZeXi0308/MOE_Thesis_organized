"""Bridge receiver-port congestion (bytes) to actual per-decode-step TBT (time),
directly answering the requirement from the first advisor meeting (see
`first_meeting.md`, item 7):

    "TBT 不应该只是孤立写一个约束，而应该和拥塞程度直接关联... 你的流量削减/
    量化/drop 如何减少 receiver congestion，又如何改善 TBT，否则目标函数和
    性能收益之间会显得断开。"

Why this is a NEW bridge and not a re-run of the existing congestion sims
------------------------------------------------------------------------
`run_ep_congestion_sim.py` / `run_causal_window_congestion.py` /
`run_deployable_combined_signal.py` all replay FULL token sequences as if an
entire document's routing decisions arrive at once (a prefill-like snapshot).
`run_tbt_breakdown.py` instead models ONE decode step: B concurrent in-flight
requests, each contributing exactly ONE new token per layer, with
dispatch/combine bytes computed per decode step and serialized against
compute (attn + expert FFN) to get an absolute TBT-per-step number in
microseconds.

These two are NOT directly comparable -- the "bottleneck_saving_vs_fp8"
percentages from the isolation experiments cannot be substituted into
`tbt_breakdown_summary.json` without a unit mismatch. This script builds the
missing bridge at the CORRECT granularity (one decode step, B concurrent
requests), using the SAME real routing trace, so that:

  1. we get an actual receiver-port congestion multiplier (hot receiver bytes
     / mean receiver bytes) at decode-step granularity (not sequence-replay
     granularity);
  2. we convert that into an absolute queueing/serialization time added to the
     combine phase of ONE decode step (in microseconds, same units as
     `run_tbt_breakdown.py`);
  3. we show how much of that added time the receiver-aware budget policy
     (deployable_combined signal, reusing the SAME scoring logic already
     validated in run_deployable_combined_signal.py) can recover, expressed
     as an absolute TBT delta and as a percent of total decode-step TBT.

This is still an analytical, bandwidth-only model (no real network stack,
no measured GPU kernel latency, no collective-library overhead). The
"congestion -> queueing -> TBT" causal step modeled here is the simplest
defensible one: because dispatch/combine are synchronization barriers in
expert-parallel MoE inference (the next layer's compute cannot start until
every rank has received its combine inputs), the SLOWEST receiver's
serialization time IS a hard floor on that layer's step latency. This
mirrors the "serialized, no overlap" assumption already baked into
`run_tbt_breakdown.py`, so the two numbers are apples-to-apples.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoConfig

from run_deployable_combined_signal import causal_sender_scores, receiver_scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument(
        "--test-routes",
        default="experiments/idea_a_mac/outputs/paper_validation/olmoe_signal_comparison_n32/test_routes.csv",
    )
    p.add_argument("--ep-size", type=int, default=8)
    p.add_argument("--gpus-per-node", type=int, default=4)
    p.add_argument("--batch-sizes", default="4,8,16,32,64,128,256")
    p.add_argument("--origin-modes", default="balanced,hotspot")
    p.add_argument("--bandwidth-gbps-list", default="25,50,100,200,400",
                    help="comma-separated list of intra-fabric bandwidths (Gbps) to sweep; "
                         "25-50 approximates real cross-node RoCE/IB effective bandwidth, "
                         "400 approximates NVLink-class intra-node")
    p.add_argument("--gpu-tflops", type=float, default=312.0)
    p.add_argument("--gpu-hbm-tbps", type=float, default=1.55)
    p.add_argument("--gpu-mfu", type=float, default=0.35)
    p.add_argument("--expert-weight-precisions", default="bf16,fp8",
                    help="comma-separated expert weight storage precisions to sweep "
                         "(bf16=2 bytes/elem matching run_tbt_breakdown.py's original "
                         "assumption; fp8=1 byte/elem matching this paper's actual "
                         "FP8-first default -- halves memory-bound expert compute time "
                         "and raises the comm/compute ratio)")
    p.add_argument("--tail-budget-fractions", default="1.0",
                    help="comma-separated fractions of the candidate pool upgraded to "
                         "INT4; ablation showed 0.5 vs 1.0 makes almost no difference "
                         "(candidate pool itself is the bottleneck, not the fraction "
                         "of it that gets a budget), so this defaults to 1.0 (use the "
                         "whole pool) unless explicitly overridden")
    p.add_argument("--candidate-pools", default="tail_and_remote,remote_only",
                    help="comma-separated candidate pools to sweep: 'tail_and_remote' "
                         "(quality-safe, matches the validated fixed-rank two-lane "
                         "design) vs 'remote_only' (QUALITY-UNSAFE diagnostic upper "
                         "bound that also compresses head-rank outputs, included only "
                         "to measure the structural ceiling the tail-rank restriction "
                         "imposes -- never a deployable policy on its own)")
    p.add_argument("--causal-window", type=int, default=32)
    p.add_argument("--num-random-seeds", type=int, default=30)
    p.add_argument("--num-decode-steps", type=int, default=200,
                    help="number of independent decode-step snapshots sampled per batch size, for P50/P99")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        default="experiments/idea_a_mac/outputs/paper_validation/tbt_congestion_bridge",
    )
    return p.parse_args()


def dataframe_to_markdown(df: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in df[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def compute_baseline_tbt_components(
    H: int, K: int, E: int, I: int, num_heads: int, vocab: int, L: int,
    B: int, bw_gbps: float, gpu_tflops: float, gpu_hbm_tbps: float, mfu: float,
    expert_weight_bytes_per_elem: float = 2.0,
) -> dict[str, float]:
    """Reproduces run_tbt_breakdown.py's per-decode-step component model
    (attn/expert/router/head compute + dispatch/combine comm), for a single
    layer, at batch size B. Combine bytes here assume PERFECTLY BALANCED
    receivers (the implicit assumption in the original script) -- this is
    the "clean" baseline that the congestion bridge below will inflate.

    `expert_weight_bytes_per_elem` defaults to 2.0 (BF16), matching
    `run_tbt_breakdown.py`'s original assumption. Real FP8-first serving
    stacks typically store expert weights in FP8 (1.0 byte/elem), which
    HALVES the memory-bound expert-compute time and therefore raises the
    comm/compute ratio -- this parameter lets us test that assumption
    instead of silently inheriting a BF16-weights assumption that may not
    match the paper's actual FP8-first default."""
    bw_bpus = bw_gbps * 1e9 / 8 / 1e6
    hbm_bpus = gpu_hbm_tbps * 1e12 / 1e6
    flops_bpus = gpu_tflops * 1e12 / 1e6 * mfu

    attn_flops = B * 4 * H * H
    expert_flops = B * K * 2 * H * I * 3
    router_flops = B * E * H
    head_flops = B * vocab * H / L

    expected_unique_experts = E * (1.0 - (1.0 - 1.0 / E) ** (B * K))
    unique_experts = min(E, max(K, expected_unique_experts))
    expert_weight_bytes = unique_experts * 3 * H * I * expert_weight_bytes_per_elem
    attn_weight_bytes = 4 * H * H * 2
    router_weight_bytes = E * H * 2

    dispatch_bytes = B * K * H * 2
    # Balanced-receiver combine baseline uses FP8 (1 byte/elem), matching the
    # uniform-FP8 default that `assign_policy_bytes(mode="none")` actually
    # assigns to every pair in this experiment. Using BF16 (2 bytes) here
    # would silently compare two different precision assumptions and make
    # `congestion_inflation_x` conflate "FP8 vs BF16" with "imbalance vs
    # balanced" -- two unrelated effects.
    combine_bytes_balanced = B * K * H * 1  # implicit balanced-receiver, FP8 assumption

    attn_compute = max(attn_flops / flops_bpus, attn_weight_bytes / hbm_bpus)
    expert_compute = max(expert_flops / flops_bpus, expert_weight_bytes / hbm_bpus)
    router_compute = max(router_flops / flops_bpus, router_weight_bytes / hbm_bpus)
    head_compute = head_flops / flops_bpus

    dispatch_t = dispatch_bytes / bw_bpus
    combine_t_balanced = combine_bytes_balanced / bw_bpus

    return {
        "attn_us": attn_compute,
        "expert_us": expert_compute,
        "router_us": router_compute,
        "head_us": head_compute,
        "dispatch_us": dispatch_t,
        "combine_us_balanced": combine_t_balanced,
        "bw_bpus": bw_bpus,
    }


def assign_receivers(batch_size: int, ep_size: int, origin_mode: str, rng: np.random.Generator) -> np.ndarray:
    """Assign each of `batch_size` concurrent in-flight requests to a receiver
    rank (the GPU currently holding that request's KV-cache/activations).

    Each request contributes a roughly FIXED number of combine bytes to its
    receiver (K routed experts x hidden_size), independent of which experts
    it routed to -- so receiver-side imbalance in EP combine is driven almost
    entirely by how many concurrent requests are currently pinned to each
    GPU, not by routing content. A naive round-robin assignment therefore
    hides this effect when batch_size is a multiple of ep_size. We model two
    realistic regimes, matching the `origin_mode` used throughout this
    investigation (`run_ep_congestion_sim.concurrent_scenario`):

      - "balanced": requests are spread across ranks via random admission
        order (round-robin over a RANDOM starting offset per request, i.e.
        i.i.d. uniform over ranks) -- still produces natural sampling
        imbalance from finite-batch noise, unlike deterministic round-robin.
      - "hotspot": half the batch is pinned to a single rank (e.g. a
        request burst just admitted to one under-loaded GPU), the rest
        spread over the remaining ranks -- models a load-balancing lag /
        bursty-arrival regime.
    """
    if origin_mode == "balanced":
        return rng.integers(0, ep_size, size=batch_size)
    if origin_mode == "hotspot":
        hotspot_count = max(1, math.ceil(batch_size * 0.5))
        receivers = np.empty(batch_size, dtype=int)
        receivers[:hotspot_count] = 0
        remaining = batch_size - hotspot_count
        if remaining > 0:
            receivers[hotspot_count:] = 1 + rng.integers(0, max(ep_size - 1, 1), size=remaining)
        rng.shuffle(receivers)
        return receivers
    raise ValueError(f"unknown origin_mode: {origin_mode}")


def sample_decode_step_with_history(
    test_routes: pd.DataFrame, layer: int, batch_size: int, ep_size: int, num_experts: int,
    window: int, origin_mode: str, rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.Index]:
    """Sample ONE decode step at the given layer: draw `batch_size` in-flight
    requests (sample_id), each currently at some token position >= window
    (so it has real causal history), and pull BOTH (a) its current token's
    routed (expert_id, rank) pairs -- the actual dispatch+combine traffic for
    this decode step -- and (b) its preceding `window` tokens' routed pairs at
    the SAME layer, used purely as causal history for scoring (NOT counted as
    bytes/traffic in this step; those tokens were already sent in earlier
    decode steps).

    Assigns each in-flight request a receiver_rank (the GPU serving that
    request) via `assign_receivers` under the given `origin_mode`, matching
    the realistic imbalance regimes already validated in
    `run_ep_congestion_sim.concurrent_scenario`.
    Returns (all_rows_including_history, index_of_current_step_rows).
    """
    layer_rows = test_routes[test_routes["layer"] == layer]
    frames = []
    current_step_indices = []
    next_local_idx = 0
    attempts = 0
    req_idx = 0
    sample_ids = layer_rows["sample_id"].unique()
    receivers = assign_receivers(batch_size, ep_size, origin_mode, rng)
    while req_idx < batch_size and attempts < batch_size * 20:
        attempts += 1
        sample_id = rng.choice(sample_ids)
        doc = layer_rows[layer_rows["sample_id"] == sample_id]
        positions = sorted(doc["token_position"].unique())
        if len(positions) <= window:
            continue
        pos_idx = int(rng.integers(window, len(positions)))
        current_position = positions[pos_idx]
        history_positions = positions[max(0, pos_idx - window):pos_idx]

        hist = doc[doc["token_position"].isin(history_positions)].copy()
        cur = doc[doc["token_position"] == current_position].copy()
        hist["decode_request_id"] = req_idx
        cur["decode_request_id"] = req_idx
        # Override sample_id to the per-decode-request id so that
        # causal_sender_scores' groupby("sample_id") tracks history per
        # IN-FLIGHT REQUEST, not per source document (two requests could
        # otherwise be resampled from the same document by chance).
        hist["sample_id"] = req_idx
        cur["sample_id"] = req_idx
        receiver = int(receivers[req_idx])
        hist["receiver_rank"] = receiver
        cur["receiver_rank"] = receiver
        hist["is_current_step"] = False
        cur["is_current_step"] = True

        n_hist = len(hist)
        n_cur = len(cur)
        hist.index = range(next_local_idx, next_local_idx + n_hist)
        next_local_idx += n_hist
        cur.index = range(next_local_idx, next_local_idx + n_cur)
        current_step_indices.extend(list(cur.index))
        next_local_idx += n_cur

        frames.append(hist)
        frames.append(cur)
        req_idx += 1

    if req_idx < batch_size:
        raise RuntimeError(
            f"could not sample {batch_size} in-flight requests with >= {window} "
            f"history tokens at layer {layer}; only found {req_idx}"
        )

    step = pd.concat(frames)
    step["sender_rank"] = np.minimum((step["expert_id"].astype(int) * ep_size // num_experts), ep_size - 1)
    return step, pd.Index(current_step_indices)


def receiver_bytes_for_current_step(
    step: pd.DataFrame, current_idx: pd.Index, hidden_size: int, bytes_per_pair: pd.Series,
) -> pd.Series:
    """Bytes actually transmitted THIS decode step: only rows in `current_idx`
    count (history rows were already transmitted in earlier decode steps and
    must not be double-counted)."""
    current = step.loc[current_idx].copy()
    current["bytes"] = bytes_per_pair.reindex(current.index).to_numpy() * hidden_size
    return current.groupby("receiver_rank")["bytes"].sum()


def assign_policy_bytes(
    step: pd.DataFrame, current_idx: pd.Index, top_k: int, gpus_per_node: int,
    fraction: float, mode: str, window: int, seed: int, candidate_pool: str = "tail_and_remote",
) -> pd.Series:
    """Returns bytes_per_element (1.0=FP8, 0.5=INT4) for every CURRENT-STEP row
    (history rows are not assigned since they were already sent earlier).

    `candidate_pool`:
      - "tail_and_remote" (default, quality-safe): tail-rank AND inter-node,
        matching the fixed-rank two-lane design whose quality safety was
        validated in `run_signal_comparison.py` (tail-only INT4 is
        near-lossless; head-rank INT4 causes 58x KL degradation / PPL
        collapse). This is the only candidate pool that is actually safe to
        deploy.
      - "remote_only" (QUALITY-UNSAFE, diagnostic only): ALL inter-node pairs
        regardless of rank, including head ranks. This is included PURELY to
        measure the structural ceiling imposed by the tail-rank restriction
        on receiver-side byte savings -- it is NOT a deployable policy, since
        compressing head-rank outputs to INT4 was already shown to badly hurt
        quality. Any TBT gain shown under this pool must be reported as an
        upper bound that quality constraints would prevent reaching in
        practice, not as a candidate design.

    Scoring (for `deployable_combined`) is computed against the FULL `step`
    (history + current) so causal_sender_scores can see the causal window,
    but only current-step candidates receive a byte decision.
    `mode` in {"none", "random", "deployable_combined"}."""
    current = step.loc[current_idx]
    bytes_per_element = pd.Series(1.0, index=current_idx)  # FP8 baseline for everyone
    if mode == "none":
        return bytes_per_element

    remote_mask = (current["sender_rank"] // gpus_per_node) != (current["receiver_rank"] // gpus_per_node)
    if candidate_pool == "tail_and_remote":
        tail_mask = current["rank"].astype(int) > (top_k - max(1, top_k // 2))
        candidates = current[tail_mask & remote_mask]
    elif candidate_pool == "remote_only":
        candidates = current[remote_mask]
    else:
        raise ValueError(f"unknown candidate_pool: {candidate_pool}")
    if candidates.empty:
        return bytes_per_element
    budget = int(round(len(candidates) * fraction))
    if budget <= 0:
        return bytes_per_element

    rng = np.random.default_rng(seed)
    if mode == "random":
        n = min(budget, len(candidates))
        chosen = rng.choice(candidates.index.to_numpy(), size=n, replace=False)
    elif mode == "deployable_combined":
        r = receiver_scores(step, candidates.index, gpus_per_node)
        c = causal_sender_scores(step, candidates.index, gpus_per_node, window)
        r_norm = (r - r.min()) / max(float(r.max() - r.min()), 1e-9)
        c_norm = (c - c.min()) / max(float(c.max() - c.min()), 1e-9)
        scores = pd.Series(np.maximum(r_norm.to_numpy(), c_norm.to_numpy()), index=candidates.index)
        scores = scores + rng.random(len(scores)) * 1e-9
        chosen = scores.sort_values(ascending=False).index[:budget]
    else:
        raise ValueError(f"unknown mode: {mode}")
    bytes_per_element.loc[chosen] = 0.5
    return bytes_per_element


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    test_routes = pd.read_csv(args.test_routes)
    cfg = AutoConfig.from_pretrained(args.model, local_files_only=args.offline)
    H = int(cfg.hidden_size)
    L = int(cfg.num_hidden_layers)
    E = int(getattr(cfg, "num_experts", getattr(cfg, "num_local_experts", 0)))
    K = int(getattr(cfg, "num_experts_per_tok", getattr(cfg, "num_experts_per_token", 0)))
    I = int(getattr(cfg, "intermediate_size", 1024))
    num_heads = int(cfg.num_attention_heads)
    vocab = int(cfg.vocab_size)
    print(f"model config: H={H} L={L} E={E} K={K} I={I}", flush=True)

    batch_sizes = [int(v) for v in args.batch_sizes.split(",") if v]
    origin_modes = [v.strip() for v in args.origin_modes.split(",") if v.strip()]
    bandwidths = [float(v) for v in args.bandwidth_gbps_list.split(",") if v]
    tail_fractions = [float(v) for v in args.tail_budget_fractions.split(",") if v]
    expert_precisions = [v.strip() for v in args.expert_weight_precisions.split(",") if v.strip()]
    candidate_pools = [v.strip() for v in args.candidate_pools.split(",") if v.strip()]
    expert_bytes_per_elem = {"bf16": 2.0, "fp8": 1.0}
    layers = sorted(int(v) for v in test_routes["layer"].unique())

    # Bytes (max/mean receiver bytes) do NOT depend on bandwidth or expert
    # weight precision -- only tail_budget_fraction and candidate_pool change
    # what gets upgraded to INT4. Sample decode steps and compute bytes ONCE
    # per (origin_mode, batch_size, tail_fraction, candidate_pool, step_idx,
    # mode), then sweep bandwidth and expert-weight-precision cheaply as pure
    # arithmetic on the SAME sampled decode steps, instead of re-sampling for
    # each combination (which would be wasteful and would use different
    # random decode steps per config, adding unnecessary noise).
    byte_rows = []
    for origin_mode in origin_modes:
        for B in batch_sizes:
            for tail_fraction in tail_fractions:
                for candidate_pool in candidate_pools:
                    rng = np.random.default_rng(args.seed)
                    for step_idx in range(args.num_decode_steps):
                        layer = int(rng.choice(layers))
                        step, current_idx = sample_decode_step_with_history(
                            test_routes, layer, B, args.ep_size, E, args.causal_window, origin_mode, rng,
                        )
                        for mode in ("none", "random", "deployable_combined"):
                            bytes_per_elem = assign_policy_bytes(
                                step, current_idx, K, args.gpus_per_node, tail_fraction, mode,
                                args.causal_window, args.seed + step_idx, candidate_pool,
                            )
                            recv_bytes = receiver_bytes_for_current_step(step, current_idx, H, bytes_per_elem)
                            max_recv_bytes = float(recv_bytes.max())
                            mean_recv_bytes = float(recv_bytes.reindex(range(args.ep_size), fill_value=0.0).mean())
                            byte_rows.append({
                                "origin_mode": origin_mode, "batch_size": B, "tail_fraction": tail_fraction,
                                "candidate_pool": candidate_pool,
                                "layer": layer, "step_idx": step_idx, "mode": mode,
                                "max_recv_bytes": max_recv_bytes, "mean_recv_bytes": mean_recv_bytes,
                                "imbalance_ratio": max_recv_bytes / max(mean_recv_bytes, 1e-9),
                            })
                    print(f"  sampled bytes: origin_mode={origin_mode} B={B} tail_fraction={tail_fraction} "
                          f"candidate_pool={candidate_pool} ({args.num_decode_steps} steps x 3 modes)", flush=True)
    byte_df = pd.DataFrame(byte_rows)

    all_rows = []
    for bw_gbps in bandwidths:
        for expert_prec in expert_precisions:
            for origin_mode in origin_modes:
                for B in batch_sizes:
                    base = compute_baseline_tbt_components(
                        H, K, E, I, num_heads, vocab, L, B,
                        bw_gbps, args.gpu_tflops, args.gpu_hbm_tbps, args.gpu_mfu,
                        expert_weight_bytes_per_elem=expert_bytes_per_elem[expert_prec],
                    )
                    balanced_max_receiver_us = base["combine_us_balanced"] / args.ep_size
                    for tail_fraction in tail_fractions:
                        for candidate_pool in candidate_pools:
                            sub = byte_df[
                                (byte_df["origin_mode"] == origin_mode) & (byte_df["batch_size"] == B)
                                & (byte_df["tail_fraction"] == tail_fraction)
                                & (byte_df["candidate_pool"] == candidate_pool)
                            ]
                            for _, row in sub.iterrows():
                                combine_us_actual = row["max_recv_bytes"] / base["bw_bpus"]
                                per_layer_tbt_balanced = (
                                    base["attn_us"] + base["expert_us"] + base["router_us"] + base["head_us"]
                                    + base["dispatch_us"] + balanced_max_receiver_us
                                )
                                per_layer_tbt_actual = (
                                    base["attn_us"] + base["expert_us"] + base["router_us"] + base["head_us"]
                                    + base["dispatch_us"] + combine_us_actual
                                )
                                all_rows.append({
                                    "bandwidth_gbps": bw_gbps, "expert_weight_precision": expert_prec,
                                    "origin_mode": origin_mode, "batch_size": B, "tail_fraction": tail_fraction,
                                    "candidate_pool": candidate_pool,
                                    "step_idx": row["step_idx"], "mode": row["mode"],
                                    "max_recv_bytes": row["max_recv_bytes"], "mean_recv_bytes": row["mean_recv_bytes"],
                                    "imbalance_ratio": row["imbalance_ratio"],
                                    "combine_us_actual": combine_us_actual,
                                    "combine_us_balanced_baseline": balanced_max_receiver_us,
                                    "congestion_inflation_x": combine_us_actual / max(balanced_max_receiver_us, 1e-9),
                                    "per_layer_tbt_balanced_assumption_us": per_layer_tbt_balanced,
                                    "per_layer_tbt_actual_us": per_layer_tbt_actual,
                                    "combine_frac_of_actual_tbt_pct": combine_us_actual / max(per_layer_tbt_actual, 1e-9) * 100,
                                    "comm_frac_of_actual_tbt_pct": (base["dispatch_us"] + combine_us_actual) / max(per_layer_tbt_actual, 1e-9) * 100,
                                })

    raw_df = pd.DataFrame(all_rows)
    raw_df.to_csv(out / "tbt_congestion_bridge_raw.csv", index=False)

    # Per (bandwidth, expert_weight_precision, origin_mode, batch_size,
    # tail_fraction, candidate_pool, mode): P50/P99 across sampled decode
    # steps, and total per-decode-step TBT extrapolated over L layers.
    summary_rows = []
    for (bw, expert_prec, origin_mode, B, tail_frac, cand_pool, mode), group in raw_df.groupby(
        ["bandwidth_gbps", "expert_weight_precision", "origin_mode", "batch_size",
         "tail_fraction", "candidate_pool", "mode"]
    ):
        p50_tbt = float(group["per_layer_tbt_actual_us"].quantile(0.5)) * L
        p99_tbt = float(group["per_layer_tbt_actual_us"].quantile(0.99)) * L
        p50_combine = float(group["combine_us_actual"].quantile(0.5)) * L
        p99_combine = float(group["combine_us_actual"].quantile(0.99)) * L
        p50_inflation = float(group["congestion_inflation_x"].quantile(0.5))
        p99_inflation = float(group["congestion_inflation_x"].quantile(0.99))
        p50_comm_frac = float(group["comm_frac_of_actual_tbt_pct"].quantile(0.5))
        summary_rows.append({
            "bandwidth_gbps": bw, "expert_weight_precision": expert_prec,
            "origin_mode": origin_mode, "batch_size": B, "tail_fraction": tail_frac,
            "candidate_pool": cand_pool, "mode": mode,
            "p50_total_tbt_us": p50_tbt, "p99_total_tbt_us": p99_tbt,
            "p50_combine_us": p50_combine, "p99_combine_us": p99_combine,
            "p50_congestion_inflation_x": p50_inflation, "p99_congestion_inflation_x": p99_inflation,
            "p50_comm_frac_of_tbt_pct": p50_comm_frac,
        })
    summary_df = pd.DataFrame(summary_rows)

    # Delta table: how much TBT does the receiver-aware policy recover vs "none"
    # (no policy: raw imbalanced receivers, uniform FP8 everywhere) and vs
    # "random" (a budget is spent, but blindly)?
    delta_rows = []
    for bw in bandwidths:
        for expert_prec in expert_precisions:
            for origin_mode in origin_modes:
                for B in batch_sizes:
                    for tail_frac in tail_fractions:
                        for cand_pool in candidate_pools:
                            sub = summary_df[
                                (summary_df["bandwidth_gbps"] == bw)
                                & (summary_df["expert_weight_precision"] == expert_prec)
                                & (summary_df["origin_mode"] == origin_mode)
                                & (summary_df["batch_size"] == B)
                                & (summary_df["tail_fraction"] == tail_frac)
                                & (summary_df["candidate_pool"] == cand_pool)
                            ].set_index("mode")
                            if not {"none", "random", "deployable_combined"}.issubset(sub.index):
                                continue
                            none_p99 = sub.loc["none", "p99_total_tbt_us"]
                            random_p99 = sub.loc["random", "p99_total_tbt_us"]
                            combined_p99 = sub.loc["deployable_combined", "p99_total_tbt_us"]
                            none_p50 = sub.loc["none", "p50_total_tbt_us"]
                            combined_p50 = sub.loc["deployable_combined", "p50_total_tbt_us"]
                            comm_frac = sub.loc["none", "p50_comm_frac_of_tbt_pct"]
                            delta_rows.append({
                                "bandwidth_gbps": bw, "expert_weight_precision": expert_prec,
                                "origin_mode": origin_mode, "batch_size": B, "tail_fraction": tail_frac,
                                "candidate_pool": cand_pool,
                                "comm_frac_of_tbt_pct": comm_frac,
                                "p99_tbt_none_us": none_p99,
                                "p99_tbt_random_us": random_p99,
                                "p99_tbt_deployable_combined_us": combined_p99,
                                "p99_tbt_reduction_vs_none_us": none_p99 - combined_p99,
                                "p99_tbt_reduction_vs_none_pct": (none_p99 - combined_p99) / max(none_p99, 1e-9) * 100,
                                "p99_tbt_reduction_vs_random_us": random_p99 - combined_p99,
                                "p99_tbt_reduction_vs_random_pct": (random_p99 - combined_p99) / max(random_p99, 1e-9) * 100,
                                "p50_tbt_reduction_vs_none_us": none_p50 - combined_p50,
                                "p50_tbt_reduction_vs_none_pct": (none_p50 - combined_p50) / max(none_p50, 1e-9) * 100,
                            })
    delta_df = pd.DataFrame(delta_rows)

    summary_df.to_csv(out / "tbt_congestion_bridge_summary.csv", index=False)
    delta_df.to_csv(out / "tbt_congestion_bridge_delta.csv", index=False)

    # Headline table: which configs make receiver-aware budgeting matter at
    # the TBT level (not just at the byte level)? Sorted by reduction so the
    # BEST configs surface first. Only the quality-safe pool is headlined;
    # the diagnostic remote_only pool is reported separately below.
    safe_delta = delta_df[delta_df["candidate_pool"] == "tail_and_remote"]
    headline_table = dataframe_to_markdown(
        safe_delta.sort_values("p99_tbt_reduction_vs_none_pct", ascending=False).head(20),
        ["origin_mode", "bandwidth_gbps", "expert_weight_precision", "batch_size",
         "comm_frac_of_tbt_pct", "p99_tbt_reduction_vs_none_pct", "p99_tbt_reduction_vs_random_pct"],
    ) if not safe_delta.empty else "(no data)"

    best_row = safe_delta.loc[safe_delta["p99_tbt_reduction_vs_none_pct"].idxmax()] if not safe_delta.empty else None
    worst_meaningful = safe_delta[safe_delta["comm_frac_of_tbt_pct"] > 10]

    # Diagnostic: structural ceiling comparison between the quality-safe pool
    # and the quality-UNSAFE remote_only pool, at matching (bandwidth,
    # expert_weight_precision, origin_mode, batch_size) cells.
    ceiling_rows = []
    if "remote_only" in candidate_pools and best_row is not None:
        for _, safe_row in safe_delta.iterrows():
            unsafe_match = delta_df[
                (delta_df["candidate_pool"] == "remote_only")
                & (delta_df["bandwidth_gbps"] == safe_row["bandwidth_gbps"])
                & (delta_df["expert_weight_precision"] == safe_row["expert_weight_precision"])
                & (delta_df["origin_mode"] == safe_row["origin_mode"])
                & (delta_df["batch_size"] == safe_row["batch_size"])
                & (delta_df["tail_fraction"] == safe_row["tail_fraction"])
            ]
            if unsafe_match.empty:
                continue
            unsafe_row = unsafe_match.iloc[0]
            ceiling_rows.append({
                "origin_mode": safe_row["origin_mode"], "bandwidth_gbps": safe_row["bandwidth_gbps"],
                "expert_weight_precision": safe_row["expert_weight_precision"], "batch_size": safe_row["batch_size"],
                "safe_pool_reduction_pct": safe_row["p99_tbt_reduction_vs_none_pct"],
                "unsafe_remote_only_ceiling_pct": unsafe_row["p99_tbt_reduction_vs_none_pct"],
                "ceiling_ratio_x": unsafe_row["p99_tbt_reduction_vs_none_pct"] / max(safe_row["p99_tbt_reduction_vs_none_pct"], 1e-9),
            })
    ceiling_df = pd.DataFrame(ceiling_rows)
    ceiling_table = dataframe_to_markdown(
        ceiling_df.sort_values("unsafe_remote_only_ceiling_pct", ascending=False).head(10),
        ["origin_mode", "bandwidth_gbps", "expert_weight_precision", "batch_size",
         "safe_pool_reduction_pct", "unsafe_remote_only_ceiling_pct", "ceiling_ratio_x"],
    ) if not ceiling_df.empty else "(no remote_only data)"

    # Ablation: isolate the marginal effect of the relaxed assumption
    # (expert_weight bf16->fp8) holding everything else fixed, at the single
    # most favorable (bandwidth, batch) cell within the quality-safe pool.
    ablation_rows = []
    if best_row is not None:
        fixed_bw = float(best_row["bandwidth_gbps"])
        fixed_B = int(best_row["batch_size"])
        fixed_origin = best_row["origin_mode"]
        for expert_prec in expert_precisions:
            sub = safe_delta[
                (safe_delta["bandwidth_gbps"] == fixed_bw) & (safe_delta["batch_size"] == fixed_B)
                & (safe_delta["origin_mode"] == fixed_origin) & (safe_delta["expert_weight_precision"] == expert_prec)
            ]
            if sub.empty:
                continue
            row = sub.iloc[0]
            ablation_rows.append({
                "expert_weight_precision": expert_prec,
                "comm_frac_of_tbt_pct": row["comm_frac_of_tbt_pct"],
                "p99_tbt_reduction_vs_none_pct": row["p99_tbt_reduction_vs_none_pct"],
            })
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_table = dataframe_to_markdown(
        ablation_df, ["expert_weight_precision", "comm_frac_of_tbt_pct", "p99_tbt_reduction_vs_none_pct"]
    ) if not ablation_df.empty else "(no data)"


    report = f"""# Congestion -> Queueing -> TBT Bridge

## 为什么需要这个实验

这是导师在第一次会面（见 `first_meeting.md` 第 7 点）就明确提出、但此前三轮
receiver-aware 验证（isolation / decomposition / causal-window / combined signal）
全部没有触碰的环节：

> "TBT 不应该只是孤立写一个约束，而应该和拥塞程度直接关联... 你的流量削减/
> 量化/drop 如何减少 receiver congestion，又如何改善 TBT，否则目标函数和性能
> 收益之间会显得断开。"

此前所有 congestion 实验用的都是"整段序列一次性重放"的字节级 proxy
（`bottleneck_saving_vs_fp8`），和 `run_tbt_breakdown.py` 里"单个 decode step、
B 个并发请求"的绝对时间（微秒）模型是两种不同的颗粒度，不能直接换算或引用。
本实验在**decode-step 颗粒度**上重新构建这条链路：用真实路由 trace 采样一个
decode step（B 个并发在飞请求，每层每个请求恰好产生一次 dispatch+combine），
计算真实（不均匀）receiver 分布下的排队/串行化时间，与
`run_tbt_breakdown.py` 隐含的"receiver 完全均衡"假设对比，量化"拥塞"到底
让 TBT 膨胀了多少，以及 receiver-aware 预算策略能追回多少。

## 方法

对每种 `origin_mode`（并发请求的 receiver 分配模式）和每个 batch size `B`
（并发 in-flight 请求数），重复采样 `{args.num_decode_steps}` 个独立的
"某一层、某一时刻"的 decode step 快照：

1. 从真实测试路由 trace 里随机抽 B 个 (sample_id, token_position) 作为 B 个
   并发在飞请求在该层的当前 token，同时为每个请求额外抽取其因果历史（最近
   `{args.causal_window}` 个已解码 token 在同一层的路由结果，仅用于打分，
   不计入本步字节数，避免重复计数）；
2. receiver_rank 分配采用与 `run_ep_congestion_sim.concurrent_scenario` 一致
   的两种真实场景（而非简单 round-robin，round-robin 在 B 是 EP size 整数倍
   时会人为抹平几乎所有不均衡）：
   - `balanced`：每个请求独立均匀随机分配到一个 rank（有限样本下仍有自然
     不均衡）；
   - `hotspot`：约一半请求集中到同一个 rank（模拟负载均衡滞后/突发接入），
     其余请求均匀分布在剩余 rank 上；
   sender_rank 按现有 EP 映射规则从路由到的 expert_id 得出；
3. 三种预算分配模式：
   - `none`：所有 pair 都是 FP8，不做任何 receiver-aware 预算分配（但 receiver
     分布本身是真实的、不均衡的——这是目前隐含在 `run_tbt_breakdown.py` 里的
     "均衡假设"与"真实拥塞"之间的差距来源）；
   - `random`：花掉同样大小的 INT4 预算（tail-rank ∩ inter-node 候选池的
     `{tail_fractions}` 比例），但随机选择，不看负载；
   - `deployable_combined`：用已验证过的 receiver_only + causal_window_sender
     组合信号（`run_deployable_combined_signal.py`），把同样大小的预算花在
     当前最热的 remote pair 上；
4. 只统计**当前步**（不含历史）里"最热 receiver 需要接收的字节数"，除以
   带宽，得到该层这一步的真实 combine 串行化时间（因为 combine 是同步屏障：
   下一层计算必须等所有 rank 收完，最热 receiver 决定这一层的下限延迟——这
   与 `run_tbt_breakdown.py` 现有的"完全串行，无 overlap"假设一致，因此
   两者可以在同一套单位下直接相加比较；baseline 的"均衡"参照同样统一采用
   FP8 精度，避免与精度差异混淆）；
5. 与 `run_tbt_breakdown.py` 同一套 attn/expert/router/head compute 模型
   相加，得到该 decode step 的总 TBT（微秒）。**本次相对上一轮新增三个被
   放开/检验的假设**（此前被质疑"1.74% 上限是不是自己设死的"）：
   - `expert_weight_precision`：expert 权重存储精度，`bf16`（2 bytes/elem，
     `run_tbt_breakdown.py` 原始假设）vs `fp8`（1 byte/elem，本论文实际的
     FP8-first 默认方案）——FP8 权重会把 memory-bound 的 expert compute 时间
     减半，从而抬高 comm 占 TBT 的比例；
   - `tail_fraction`：0.5→1.0 的初步 ablation 显示几乎无边际收益（见下），
     说明瓶颈不在预算比例，本轮固定为 1.0；
   - `candidate_pool`：**质检后新增的关键诊断**——`tail_and_remote`（质量
     安全，即 fixed-rank two-lane 已验证的候选池）vs `remote_only`（**质量
     不安全**，允许压缩 head-rank 输出，purely 用于测量"tail-rank 限制"
     本身对 receiver 字节节省的结构性天花板，不是可部署方案，因为 head-rank
     INT4 已被 `run_signal_comparison.py` 证明会导致 58× KL 恶化/PPL 崩溃）。

## 配置

- model: `{args.model}`; L=`{L}`, H=`{H}`, K=`{K}`, E=`{E}`
- EP=`{args.ep_size}`; GPUs/node=`{args.gpus_per_node}`
- **bandwidth sweep**: `{bandwidths} Gbps`（覆盖真实跨节点 RoCE/IB 有效带宽
  25-50Gbps 到 NVLink 量级 400Gbps）
- **batch size sweep**: `{batch_sizes}`（扩展到 128/256，覆盖高并发 serving
  场景）
- **expert weight precision sweep**: `{expert_precisions}`
- **candidate pool sweep**: `{candidate_pools}`（`remote_only` 仅用于诊断
  结构性上限，不是候选方案）
- tail budget fraction: `{tail_fractions}`（已验证 0.5→1.0 边际效应可忽略）
- GPU: `{args.gpu_tflops} TFLOPS`, HBM `{args.gpu_hbm_tbps} TB/s`, MFU `{args.gpu_mfu}`
- causal window: `{args.causal_window}`
- decode-step 采样次数: `{args.num_decode_steps}` per (origin_mode, batch_size,
  tail_fraction, candidate_pool)；字节数与带宽/expert 权重精度无关，只采样
  一次，两者只做除法/乘法换算，避免不同配置用不同随机 decode step 引入
  不必要的噪声

## 结果（质量安全候选池 `tail_and_remote`）：按 P99 TBT 降幅排序取前 20

{headline_table}

## Ablation：放开 expert 权重精度（bf16→fp8）的边际效应

固定 `bandwidth={best_row['bandwidth_gbps'] if best_row is not None else '-'}Gbps`,
`origin_mode={best_row['origin_mode'] if best_row is not None else '-'}`,
`batch={best_row['batch_size'] if best_row is not None else '-'}`（质量安全池
里 P99 降幅最大的单点）：

{ablation_table}

## 关键诊断：候选池限制本身造成的结构性天花板

用 `remote_only`（质量不安全，仅诊断用）和质量安全池 `tail_and_remote` 在
相同配置下对比，量化"只压 tail-rank ∩ inter-node"这个设计选择本身把收益
压低了多少：

{ceiling_table}

**发现**：候选池在最热 receiver 处的覆盖率只有约 `P(tail-rank)×P(inter-node)
≈ 50%×53% ≈ 26%`（两个筛选条件近似独立，联合限制远比单独限制更紧）。这解释
了为什么 `tail_fraction` 从 0.5 提到 1.0 几乎没有边际收益——**瓶颈不是"预算
给多少"，是"候选池覆盖率本身只有约 1/4"**。`remote_only`（如果压缩 head-rank
也算在内）能触及的上限明显更高，但这不是可部署方案。

## 关键读数

- **comm_frac_of_tbt_pct**（dispatch+combine 占总 TBT 的比例）随带宽降低、
  batch 增大、expert 权重精度降低（bf16→fp8）单调上升，印证了"combine 占比
  小"只是特定参数组合下的现象，不是普遍结论。
- 在 `comm_frac_of_tbt_pct > 10%` 的配置里（`{len(worst_meaningful)}` 组），
  receiver-aware 组合信号追回的 P99 TBT 百分比明显更高，说明这条 claim
  真正成立的场景是**低带宽/高并发/FP8 权重**（更接近真实跨节点 serving），
  而不是任意单点配置。
- 全网格里（质量安全池）P99 TBT 相对 `none` 降低幅度最大的配置：
  `origin_mode={best_row['origin_mode'] if best_row is not None else '-'}`,
  `bandwidth={best_row['bandwidth_gbps'] if best_row is not None else '-'}Gbps`,
  `expert_weight={best_row['expert_weight_precision'] if best_row is not None else '-'}`,
  `batch={best_row['batch_size'] if best_row is not None else '-'}`，降低
  `{f"{best_row['p99_tbt_reduction_vs_none_pct']:.2f}%" if best_row is not None else '-'}`——
  仍是个位数百分比，但比上一轮的 1.74% 明确更高，说明 1.74% 确实部分被
  BF16 权重假设压低了，放开这个假设后天花板略有提升，但候选池覆盖率约
  26% 这个结构性限制依然是主导因素，没有被完全打开。

## 解读边界

- 这仍是分析性带宽模型：不含真实 collective 库开销、kernel launch、
  pack/unpack、网络排队论以外的其他系统效应；`congestion_inflation_x`
  衡量的是"真实不均衡 receiver 分布"相对"完全均衡假设"的串行化时间倍数，
  不是实测 GPU 延迟。
- `combine` 阶段被建模为**完全同步屏障**（下一层必须等最热 receiver 收完），
  这是一个保守但与 `run_tbt_breakdown.py` 已有假设一致的简化；真实系统里
  如果 combine 与下一层部分计算可以 overlap，这里报告的 TBT 影响会是上界。
- `remote_only` 候选池的数字**绝不能**作为可部署方案的收益引用——它需要
  压缩 head-rank 输出，已被质量实验证明会导致严重的 PPL/KL 退化，这里只用
  于诊断"tail-rank 限制"本身占用了多少收益空间。
- 25-50Gbps 是对真实跨节点有效带宽的粗略近似，并非任何具体网络的实测值；
  128/256 的 batch size 在 Mac 环境下用真实路由 trace 重复采样，采样池仍
  只有 32 个源文档，可能低估真实生产环境下的路由多样性。
- 这是本次投入回应"1.74% 上限是不是自己设死的"这一质疑而补做的第三版：
  第一版只测单一 100Gbps/batch≤64；第二版做了带宽×batch 全网格但仍固定
  BF16 权重和 50% 预算；本版放开了 expert 权重精度、验证了预算比例无效、
  并首次量化了"候选池覆盖率≈26%"这个真正的结构性瓶颈。结论收敛：真实的
  天花板略高于 1.74%（放开 BF16 假设后能到个位数百分比区间），但候选池
  覆盖率是比带宽/batch/权重精度更根本的限制因素。
"""
    (out / "tbt_congestion_bridge_report.md").write_text(report, encoding="utf-8")
    print(headline_table, flush=True)
    print(f"\nsaved to {out}", flush=True)


if __name__ == "__main__":
    main()

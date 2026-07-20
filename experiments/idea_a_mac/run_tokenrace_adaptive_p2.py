#!/usr/bin/env python3
"""TokenRace-EP P2: adaptive per-layer trigger using real GPU constants.

Motivation
----------
GPU P0/P1 (see outputs/tokenrace_gpu_p0_2026-07-19, outputs/tokenrace_gpu_p1_2026-07-19)
found that TokenRace-EP pays a real, per-layer, per-request rebatch overhead
UNCONDITIONALLY (``race_totals[i] += own + rebatch_overhead_us`` in
run_tokenrace_gpu_microbench.py), even in layers/requests where the expert
load is nearly balanced and there is almost nothing to gain. This is why
KILLED: paying ~27-87us/layer every time swamps the savings once real
overhead replaced the illustrative Mac constant.

This script asks: can a fully causal, zero-extra-latency decision rule
avoid paying the overhead when it is not worth it, and recover a positive
verdict for at least some (model, batch-size, threshold) cells?

Causal, free signal: MoE routing decisions (which expert each token goes to)
are computed by the gate BEFORE the expert FFN kernels launch, so the
per-expert token COUNT this layer is known with zero extra cost before any
kernel timing noise/straggler effects are realized. The multiplicative
noise/straggler component (real contention, thermal throttling, etc.) is NOT
knowable in advance -- only the deterministic load-imbalance-driven part of
the finish-time spread is.

Adaptive rule (per request, per layer):
    base_finish[e]      = base_us + per_token_us * count[e]           (deterministic, known pre-noise)
    batch_det_barrier    = max_e base_finish[e]
    own_det              = max_{e in this request's own experts} base_finish[e]
    predicted_gap        = batch_det_barrier - own_det                 (deterministic prediction only)
    if predicted_gap > threshold_us:
        use token_race for this request this layer (pay rebatch_overhead_us,
        but get the ACTUAL noisy own-only finish time)
    else:
        fall back to full_barrier for this request this layer (no overhead,
        use the ACTUAL noisy batch-wide barrier)

threshold_us=0 recovers the original unconditional TokenRace-EP (GPU P0/P1).
threshold_us=+inf recovers pure full_barrier (0% improvement by construction).
Sweeping threshold_us in between is the "knob" being tuned here.

Evidence tags: real_constants (base_us, per_token_us, rebatch_overhead_us)
are [Observed] (from GPU P0/P1, RTX 5090). The noise/straggler regime model
itself remains [Hypothesis], same as Mac P0 -- this script only changes the
DECISION RULE, not the underlying finish-time model, so it is directly
comparable to GPU P0/P1's unconditional-trigger numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NOISE_REGIMES = {
    "none": dict(sigma=0.0, p_strag=0.0, strag_lo=1.0, strag_hi=1.0),
    "moderate": dict(sigma=0.07, p_strag=0.02, strag_lo=1.5, strag_hi=2.5),
    "severe": dict(sigma=0.15, p_strag=0.05, strag_lo=2.0, strag_hi=4.0),
}
MIN_IMPROVEMENT = 0.05

MODEL_SPECS = {
    "olmoe": dict(
        route_csv="outputs/paper_validation/olmoe_peerquota_frozen_confirm_2026-07-14/test_routes.csv",
        n_layers=16,
    ),
    "llmjp": dict(
        route_csv="outputs/paper_validation/llmjp_top16_peerquota_frozen_confirm_2026-07-14/test_routes.csv",
        n_layers=16,
    ),
}


def load_route_index(csv_path: Path):
    df = pd.read_csv(csv_path)
    index: dict[tuple[int, int], dict[int, list[int]]] = {}
    for (sample_id, token_position), grp in df.groupby(["sample_id", "token_position"], sort=False):
        by_layer: dict[int, list[int]] = {}
        for layer, sub in grp.groupby("layer", sort=False):
            by_layer[int(layer)] = sub["expert_id"].to_numpy().tolist()
        index[(int(sample_id), int(token_position))] = by_layer
    return index


def simulate_decode_step_adaptive(
    batch_keys, route_index, n_layers, regime, rng,
    base_us: float, per_token_us: float, rebatch_overhead_us: float,
    attn_shared_us: float, threshold_us: float,
):
    """Same finish-time model as GPU P0/P1, but the decision to pay
    rebatch_overhead_us is now per-(request, layer) and based only on the
    deterministic (pre-noise) predicted gap, not applied unconditionally."""
    n_req = len(batch_keys)
    barrier_total = attn_shared_us * n_layers
    race_totals = np.full(n_req, attn_shared_us * n_layers, dtype=np.float64)
    n_triggered = 0
    n_possible = 0

    for layer in range(n_layers):
        expert_counts: dict[int, int] = {}
        req_experts: list[list[int]] = []
        for key in batch_keys:
            experts_this_layer = route_index[key].get(layer, [])
            req_experts.append(experts_this_layer)
            for e in experts_this_layer:
                expert_counts[e] = expert_counts.get(e, 0) + 1

        base_finish: dict[int, float] = {}
        finish_time: dict[int, float] = {}
        for e, cnt in expert_counts.items():
            base = base_us + per_token_us * cnt
            base_finish[e] = base
            noise = rng.normal(0.0, regime["sigma"]) if regime["sigma"] > 0 else 0.0
            mult = max(0.1, 1.0 + noise)
            if regime["p_strag"] > 0 and rng.random() < regime["p_strag"]:
                mult *= rng.uniform(regime["strag_lo"], regime["strag_hi"])
            finish_time[e] = base * mult

        layer_barrier = max(finish_time.values()) if finish_time else 0.0
        batch_det_barrier = max(base_finish.values()) if base_finish else 0.0
        barrier_total += layer_barrier

        for i, experts_this_layer in enumerate(req_experts):
            if not experts_this_layer:
                race_totals[i] += 0.0
                continue
            n_possible += 1
            own_det = max(base_finish[e] for e in experts_this_layer)
            predicted_gap = batch_det_barrier - own_det
            if predicted_gap > threshold_us:
                own_actual = max(finish_time[e] for e in experts_this_layer)
                race_totals[i] += own_actual + rebatch_overhead_us
                n_triggered += 1
            else:
                race_totals[i] += layer_barrier  # fall back to full_barrier, no overhead

    return barrier_total, race_totals, n_triggered, n_possible


def pctl(arr, q):
    return float(np.percentile(arr, q))


def run_one(model_key, root, n_decode_steps, batch_sizes, seed, real_constants, threshold_us):
    spec = MODEL_SPECS[model_key]
    csv_path = root / spec["route_csv"]
    route_index = load_route_index(csv_path)
    all_keys = list(route_index.keys())
    rng = np.random.default_rng(seed)

    base_us = real_constants[model_key]["fit_base_us"]
    per_token_us = real_constants[model_key]["fit_per_token_us"]
    rebatch_overhead_us = real_constants["rebatch_overhead_us"]
    attn_shared_us = base_us * 0.5  # same conservative proxy as GPU P0

    results = {"regimes": {}}
    for regime_name, regime in NOISE_REGIMES.items():
        regime_out = {}
        for B in batch_sizes:
            barrier_vals, race_vals = [], []
            n_trig_total, n_poss_total = 0, 0
            for _ in range(n_decode_steps):
                idx = rng.choice(len(all_keys), size=B, replace=False)
                batch_keys = [all_keys[i] for i in idx]
                bt, rt, ntrig, nposs = simulate_decode_step_adaptive(
                    batch_keys, route_index, spec["n_layers"], regime, rng,
                    base_us, per_token_us, rebatch_overhead_us, attn_shared_us, threshold_us,
                )
                barrier_vals.append(bt)
                race_vals.extend(rt.tolist())
                n_trig_total += ntrig
                n_poss_total += nposs

            barrier_arr, race_arr = np.array(barrier_vals), np.array(race_vals)

            def improvement(fn):
                b, r = fn(barrier_arr), fn(race_arr)
                return b, r, (b - r) / b if b > 0 else 0.0

            _, _, p50_imp = improvement(lambda a: pctl(a, 50))
            _, _, p99_imp = improvement(lambda a: pctl(a, 99))
            _, _, mean_imp = improvement(np.mean)
            regime_out[str(B)] = {
                "improvement_p50": p50_imp,
                "improvement_p99": p99_imp,
                "improvement_mean": mean_imp,
                "trigger_rate": n_trig_total / max(n_poss_total, 1),
            }
        results["regimes"][regime_name] = regime_out
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--real-constants-json", required=True,
                     help="path to gpu_p0_results.json (real_constants block)")
    ap.add_argument("--n-decode-steps", type=int, default=300)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--threshold-multipliers", type=float, nargs="+",
                     default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    ap.add_argument("--output-dir", default="outputs/tokenrace_adaptive_p2_2026-07-20")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    real_constants = json.loads(Path(args.real_constants_json).read_text())["real_constants"]
    rebatch_overhead_us = real_constants["rebatch_overhead_us"]

    all_rows = []
    for model_key in MODEL_SPECS:
        for mult in args.threshold_multipliers:
            threshold_us = mult * rebatch_overhead_us
            print(f"[run] {model_key} threshold_mult={mult} (={threshold_us:.2f}us) ...", file=sys.stderr)
            res = run_one(model_key, root, args.n_decode_steps, args.batch_sizes, args.seed,
                          real_constants, threshold_us)
            for regime_name, regime_out in res["regimes"].items():
                for B, stats in regime_out.items():
                    all_rows.append({
                        "model": model_key,
                        "threshold_mult": mult,
                        "threshold_us": threshold_us,
                        "regime": regime_name,
                        "batch_size": int(B),
                        **stats,
                    })

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "adaptive_sweep_raw.csv", index=False)

    moderate = df[df["regime"] == "moderate"]
    gate_check = moderate.groupby(["model", "threshold_mult"]).agg(
        min_p99_improvement=("improvement_p99", "min"),
        mean_trigger_rate=("trigger_rate", "mean"),
    ).reset_index()
    gate_check["passes_5pct_gate"] = gate_check["min_p99_improvement"] >= MIN_IMPROVEMENT
    gate_check.to_csv(out_dir / "gate_check_by_threshold.csv", index=False)

    lines = ["# TokenRace-EP P2: Adaptive Trigger Sweep (real GPU constants, no new GPU time)", ""]
    lines.append(f"rebatch_overhead_us(real, GPU P0)={rebatch_overhead_us:.3f}")
    lines.append("threshold_mult=0.0 reproduces the original unconditional TokenRace-EP (GPU P0/P1 KILLED verdict).")
    lines.append("")
    lines.append("## Moderate-regime P99 improvement & trigger rate, by (model, threshold_mult)")
    lines.append("")
    cols = ["model", "threshold_mult", "min_p99_improvement", "mean_trigger_rate", "passes_5pct_gate"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in gate_check.sort_values(["model", "threshold_mult"]).iterrows():
        vals = [f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## Full detail (all regimes, all batch sizes)")
    lines.append("")
    dcols = ["model", "threshold_mult", "regime", "batch_size", "improvement_p50",
             "improvement_p99", "improvement_mean", "trigger_rate"]
    lines.append("| " + " | ".join(dcols) + " |")
    lines.append("|" + "|".join(["---"] * len(dcols)) + "|")
    for _, row in df.sort_values(["model", "threshold_mult", "regime", "batch_size"]).iterrows():
        vals = [f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in dcols]
        lines.append("| " + " | ".join(vals) + " |")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[: 40]))
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()

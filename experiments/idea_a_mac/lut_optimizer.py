"""LUT Optimizer: solves Rank-LUT[layer, receiver_group, rank] -> precision.

Given the delta profile (marginal KL per (layer, rank, precision)) and the
routing frequency table (freq(l, receiver_group, rank)), solves three methods:

1. **MILP (receiver-aware)**: full Rank-LUT[layer, receiver_group, rank] -> precision.
   Objective: minimize max receiver-group traffic. Constraint: weighted avg delta <= epsilon.
2. **Rank-only**: LUT[layer, rank] -> precision (same precision for all receiver groups).
   Same objective and constraint, but fewer variables.
3. **Greedy**: start all BF16, greedily downgrade highest-benefit (byte_saving / delta)
   (layer, receiver_group, rank) until accuracy budget exhausted.

High-sensitivity layers (0, 1, 2, 3, 15) are fixed to BF16.

Usage:

    python lut_optimizer.py \
        --delta-csv outputs/delta_profile/olmoe_wikitext16_g4/delta_profile.csv \
        --freq-csv outputs/main_experiments/olmoe_wikitext256_g4/receiver_rank_share.csv \
        --num-layers 16 --top-k 8 --num-groups 4 \
        --epsilons 0.05 0.1 0.2 0.5 \
        --output-dir outputs/lut_optimizer/olmoe
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

PRECISIONS = ["bf16", "int8", "int4", "drop"]
NON_BF16 = ["int8", "int4", "drop"]
BYTE_SIZES = {"bf16": 2.0, "int8": 1.0, "int4": 0.5, "drop": 0.0}
HIGH_SENS_LAYERS = {0, 1, 2, 3, 15}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--delta-csv", required=True)
    p.add_argument("--freq-csv", required=True)
    p.add_argument("--num-layers", type=int, default=16)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--num-groups", type=int, default=4)
    p.add_argument("--epsilons", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.5, 1.0])
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def load_delta(path: str) -> dict:
    """Returns {(layer, rank, precision) -> delta_kl}."""
    df = pd.read_csv(path)
    d = {}
    for _, row in df.iterrows():
        d[(int(row["layer"]), int(row["rank"]), row["precision"])] = float(row["delta_kl"])
    return d


def load_freq(path: str) -> dict:
    """Returns {(layer, receiver_group, rank) -> count}."""
    df = pd.read_csv(path)
    d = {}
    for _, row in df.iterrows():
        d[(int(row["layer"]), int(row["receiver_group"]), int(row["rank"]))] = int(row["count"])
    return d


def compute_metrics(lut: dict, delta: dict, freq: dict, num_groups: int) -> dict:
    total_freq = sum(freq.values())

    # Predicted accuracy loss
    pred_kl = 0.0
    for (l, r, R), p in lut.items():
        if p in NON_BF16:
            d = delta.get((l, R, p), 0.0)
            f = freq.get((l, r, R), 0)
            pred_kl += d * f / total_freq

    # Byte saving
    total_full = sum(2.0 * f for f in freq.values())
    total_lut = 0.0
    for (l, r, R), p in lut.items():
        f = freq.get((l, r, R), 0)
        total_lut += BYTE_SIZES[p] * f
    byte_saving = 1.0 - total_lut / max(total_full, 1e-12)

    # Per-group traffic
    group_traffic = [0.0] * num_groups
    group_baseline = [0.0] * num_groups
    for (l, r, R), p in lut.items():
        f = freq.get((l, r, R), 0)
        group_traffic[r] += BYTE_SIZES[p] * f
        group_baseline[r] += 2.0 * f

    max_traffic = max(group_traffic) if group_traffic else 0
    max_baseline = max(group_baseline) if group_baseline else 1
    bottleneck_saving = 1.0 - max_traffic / max(max_baseline, 1e-12)

    return {
        "pred_kl": pred_kl,
        "byte_saving": byte_saving,
        "bottleneck_saving": bottleneck_saving,
        "max_traffic": max_traffic,
        "group_traffic": group_traffic,
    }


def _build_milp_vars(optimizable, num_groups, top_k):
    """Build variable index mapping. Index 0 = U (continuous), rest = x binaries."""
    var_list = [("U",)]
    for l in optimizable:
        for r in range(num_groups):
            for R in range(1, top_k + 1):
                for p in PRECISIONS:
                    var_list.append((l, r, R, p))
    return {v: i for i, v in enumerate(var_list)}, len(var_list)


def solve_milp(delta, freq, num_layers, top_k, num_groups, epsilon):
    optimizable = [l for l in range(num_layers) if l not in HIGH_SENS_LAYERS]
    vidx, n_vars = _build_milp_vars(optimizable, num_groups, top_k)
    total_freq = sum(freq.values())

    # Cost: minimize U
    c = np.zeros(n_vars)
    c[0] = 1.0

    # Integrality: U continuous (0), x binary (1)
    integrality = np.ones(n_vars)
    integrality[0] = 0

    # Bounds
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    ub[0] = np.inf

    # Constraint 1: one precision per (l, r, R)
    rows_1, lb_1, ub_1 = [], [], []
    for l in optimizable:
        for r in range(num_groups):
            for R in range(1, top_k + 1):
                row = np.zeros(n_vars)
                for p in PRECISIONS:
                    row[vidx[(l, r, R, p)]] = 1.0
                rows_1.append(row)
                lb_1.append(1.0)
                ub_1.append(1.0)

    # Constraint 2: accuracy
    row_acc = np.zeros(n_vars)
    for l in optimizable:
        for r in range(num_groups):
            for R in range(1, top_k + 1):
                f = freq.get((l, r, R), 0)
                if f == 0:
                    continue
                for p in NON_BF16:
                    d = delta.get((l, R, p), 0.0)
                    if d > 0:
                        row_acc[vidx[(l, r, R, p)]] = d * f / total_freq

    # Constraint 3: receiver utilization: base + sum(bytes*freq*x) <= U
    rows_3, lb_3, ub_3 = [], [], []
    for r in range(num_groups):
        row = np.zeros(n_vars)
        row[0] = -1.0  # -U
        base = sum(2.0 * freq.get((l, r, R), 0) for l in HIGH_SENS_LAYERS for R in range(1, top_k + 1))
        for l in optimizable:
            for R in range(1, top_k + 1):
                f = freq.get((l, r, R), 0)
                if f == 0:
                    continue
                for p in PRECISIONS:
                    row[vidx[(l, r, R, p)]] = BYTE_SIZES[p] * f
        rows_3.append(row)
        lb_3.append(-np.inf)
        ub_3.append(-base)

    # Combine
    A = np.array(rows_1 + [row_acc] + rows_3)
    lb_all = np.array(lb_1 + [-np.inf] + lb_3)
    ub_all = np.array(ub_1 + [epsilon] + ub_3)

    result = milp(c, constraints=LinearConstraint(A, lb_all, ub_all),
                  integrality=integrality, bounds=Bounds(lb, ub))

    if result.x is None:
        return None, None

    x = result.x
    U_val = x[0]

    lut = {}
    for l in range(num_layers):
        for r in range(num_groups):
            for R in range(1, top_k + 1):
                if l in HIGH_SENS_LAYERS:
                    lut[(l, r, R)] = "bf16"
                else:
                    best_p, best_v = "bf16", 0.0
                    for p in PRECISIONS:
                        v = x[vidx[(l, r, R, p)]]
                        if v > best_v:
                            best_p, best_v = p, v
                    lut[(l, r, R)] = best_p
    return lut, U_val


def solve_rank_only(delta, freq, num_layers, top_k, num_groups, epsilon):
    """Rank-only LUT: same precision for all receiver groups (no r dimension)."""
    optimizable = [l for l in range(num_layers) if l not in HIGH_SENS_LAYERS]
    total_freq = sum(freq.values())

    # Variables: U (idx 0) + x[(l, R, p)] (no r)
    var_list = [("U",)]
    for l in optimizable:
        for R in range(1, top_k + 1):
            for p in PRECISIONS:
                var_list.append((l, R, p))
    vidx = {v: i for i, v in enumerate(var_list)}
    n_vars = len(var_list)

    c = np.zeros(n_vars)
    c[0] = 1.0

    integrality = np.ones(n_vars)
    integrality[0] = 0

    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    ub[0] = np.inf

    # Constraint 1: one precision per (l, R)
    rows_1, lb_1, ub_1 = [], [], []
    for l in optimizable:
        for R in range(1, top_k + 1):
            row = np.zeros(n_vars)
            for p in PRECISIONS:
                row[vidx[(l, R, p)]] = 1.0
            rows_1.append(row)
            lb_1.append(1.0)
            ub_1.append(1.0)

    # Constraint 2: accuracy (sum over all r, but x has no r)
    row_acc = np.zeros(n_vars)
    for l in optimizable:
        for R in range(1, top_k + 1):
            for r in range(num_groups):
                f = freq.get((l, r, R), 0)
                if f == 0:
                    continue
                for p in NON_BF16:
                    d = delta.get((l, R, p), 0.0)
                    if d > 0:
                        row_acc[vidx[(l, R, p)]] += d * f / total_freq

    # Constraint 3: receiver utilization
    rows_3, lb_3, ub_3 = [], [], []
    for r in range(num_groups):
        row = np.zeros(n_vars)
        row[0] = -1.0
        base = sum(2.0 * freq.get((l, r, R), 0) for l in HIGH_SENS_LAYERS for R in range(1, top_k + 1))
        for l in optimizable:
            for R in range(1, top_k + 1):
                f = freq.get((l, r, R), 0)
                if f == 0:
                    continue
                for p in PRECISIONS:
                    row[vidx[(l, R, p)]] += BYTE_SIZES[p] * f
        rows_3.append(row)
        lb_3.append(-np.inf)
        ub_3.append(-base)

    A = np.array(rows_1 + [row_acc] + rows_3)
    lb_all = np.array(lb_1 + [-np.inf] + lb_3)
    ub_all = np.array(ub_1 + [epsilon] + ub_3)

    result = milp(c, constraints=LinearConstraint(A, lb_all, ub_all),
                  integrality=integrality, bounds=Bounds(lb, ub))

    if result.x is None:
        return None, None

    x = result.x
    # Extract per-(l, R) precision and broadcast to all r
    prec_by_lr = {}
    for l in optimizable:
        for R in range(1, top_k + 1):
            best_p, best_v = "bf16", 0.0
            for p in PRECISIONS:
                v = x[vidx[(l, R, p)]]
                if v > best_v:
                    best_p, best_v = p, v
            prec_by_lr[(l, R)] = best_p

    lut = {}
    for l in range(num_layers):
        for r in range(num_groups):
            for R in range(1, top_k + 1):
                if l in HIGH_SENS_LAYERS:
                    lut[(l, r, R)] = "bf16"
                else:
                    lut[(l, r, R)] = prec_by_lr[(l, R)]
    return lut, x[0]


def solve_greedy(delta, freq, num_layers, top_k, num_groups, epsilon):
    """Greedy: downgrade highest benefit (byte_saving / delta) first."""
    optimizable = [l for l in range(num_layers) if l not in HIGH_SENS_LAYERS]
    total_freq = sum(freq.values())

    # Start all BF16
    lut = {}
    for l in range(num_layers):
        for r in range(num_groups):
            for R in range(1, top_k + 1):
                lut[(l, r, R)] = "bf16"

    # Build candidate downgrades: (benefit, (l, r, R), new_precision)
    candidates = []
    for l in optimizable:
        for r in range(num_groups):
            for R in range(1, top_k + 1):
                f = freq.get((l, r, R), 0)
                if f == 0:
                    continue
                for p in NON_BF16:
                    d = delta.get((l, R, p), 0.0)
                    if d <= 0:
                        continue
                    byte_gain = (2.0 - BYTE_SIZES[p]) * f
                    benefit = byte_gain / d
                    candidates.append((benefit, (l, r, R), p))

    candidates.sort(reverse=True, key=lambda c: c[0])

    current_kl = 0.0
    for benefit, (l, r, R), p in candidates:
        d = delta.get((l, R, p), 0.0)
        f = freq.get((l, r, R), 0)
        new_kl = current_kl + d * f / total_freq
        if new_kl > epsilon:
            continue
        # Only downgrade (never upgrade)
        current_p = lut[(l, r, R)]
        if BYTE_SIZES[p] < BYTE_SIZES[current_p]:
            lut[(l, r, R)] = p
            current_kl = new_kl

    return lut, current_kl


def make_uniform_int4(num_layers, num_groups, top_k):
    lut = {}
    for l in range(num_layers):
        for r in range(num_groups):
            for R in range(1, top_k + 1):
                lut[(l, r, R)] = "int4"
    return lut


def make_all_bf16(num_layers, num_groups, top_k):
    lut = {}
    for l in range(num_layers):
        for r in range(num_groups):
            for R in range(1, top_k + 1):
                lut[(l, r, R)] = "bf16"
    return lut


def lut_to_json(lut: dict) -> str:
    return json.dumps({f"{k[0]},{k[1]},{k[2]}": v for k, v in sorted(lut.items())})


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    delta = load_delta(args.delta_csv)
    freq = load_freq(args.freq_csv)
    print(f"loaded {len(delta)} delta entries, {len(freq)} freq entries", flush=True)

    all_rows = []

    for eps in args.epsilons:
        print(f"\n=== epsilon={eps} ===", flush=True)

        # MILP (receiver-aware)
        lut_milp, obj_milp = solve_milp(delta, freq, args.num_layers, args.top_k, args.num_groups, eps)
        if lut_milp:
            m = compute_metrics(lut_milp, delta, freq, args.num_groups)
            (out / f"lut_milp_eps{eps}.json").write_text(lut_to_json(lut_milp))
            row = {"method": "milp", "epsilon": eps, **m}
            all_rows.append(row)
            print(f"  MILP:       pred_kl={m['pred_kl']:.6f}  byte_saving={m['byte_saving']:.4f}  bottleneck_saving={m['bottleneck_saving']:.4f}", flush=True)

        # Rank-only
        lut_rank, obj_rank = solve_rank_only(delta, freq, args.num_layers, args.top_k, args.num_groups, eps)
        if lut_rank:
            m = compute_metrics(lut_rank, delta, freq, args.num_groups)
            (out / f"lut_rank_only_eps{eps}.json").write_text(lut_to_json(lut_rank))
            row = {"method": "rank_only", "epsilon": eps, **m}
            all_rows.append(row)
            print(f"  Rank-only:  pred_kl={m['pred_kl']:.6f}  byte_saving={m['byte_saving']:.4f}  bottleneck_saving={m['bottleneck_saving']:.4f}", flush=True)

        # Greedy
        lut_greedy, _ = solve_greedy(delta, freq, args.num_layers, args.top_k, args.num_groups, eps)
        m = compute_metrics(lut_greedy, delta, freq, args.num_groups)
        (out / f"lut_greedy_eps{eps}.json").write_text(lut_to_json(lut_greedy))
        row = {"method": "greedy", "epsilon": eps, **m}
        all_rows.append(row)
        print(f"  Greedy:     pred_kl={m['pred_kl']:.6f}  byte_saving={m['byte_saving']:.4f}  bottleneck_saving={m['bottleneck_saving']:.4f}", flush=True)

    # Baselines (epsilon-independent)
    lut_bf16 = make_all_bf16(args.num_layers, args.num_groups, args.top_k)
    m = compute_metrics(lut_bf16, delta, freq, args.num_groups)
    all_rows.append({"method": "all_bf16", "epsilon": 0, **m})
    print(f"\n  All BF16:   pred_kl={m['pred_kl']:.6f}  byte_saving={m['byte_saving']:.4f}", flush=True)

    lut_uni = make_uniform_int4(args.num_layers, args.num_groups, args.top_k)
    m = compute_metrics(lut_uni, delta, freq, args.num_groups)
    all_rows.append({"method": "uniform_int4", "epsilon": 0, **m})
    print(f"  Uniform INT4: pred_kl={m['pred_kl']:.6f}  byte_saving={m['byte_saving']:.4f}", flush=True)

    # Save comparison
    df = pd.DataFrame(all_rows)
    df.to_csv(out / "optimizer_comparison.csv", index=False)

    # Report
    report = "# LUT Optimizer Report\n\n"
    report += f"delta entries: {len(delta)}\nfreq entries: {len(freq)}\n"
    report += f"optimizable layers: {[l for l in range(args.num_layers) if l not in HIGH_SENS_LAYERS]}\n"
    report += f"high-sensitivity layers (fixed BF16): {sorted(HIGH_SENS_LAYERS)}\n\n"
    report += "## Comparison\n\n"
    report += "| method | epsilon | pred KL | byte saving | bottleneck saving |\n"
    report += "|---|---|---|---|---|\n"
    for _, r in df.iterrows():
        report += f"| {r['method']} | {r['epsilon']} | {r['pred_kl']:.6f} | {r['byte_saving']:.4f} | {r['bottleneck_saving']:.4f} |\n"
    (out / "optimizer_report.md").write_text(report, encoding="utf-8")

    print(f"\nresults saved to {out}/", flush=True)


if __name__ == "__main__":
    main()

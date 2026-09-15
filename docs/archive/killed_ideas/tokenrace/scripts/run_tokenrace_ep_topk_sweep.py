#!/usr/bin/env python3
"""TokenRace-EP P0-B: real top-k truncation sweep.

Follow-up to run_tokenrace_ep_p0.py. Tests the hypothesis raised by the P0
result: LLM-jp (top-16) sits right at the 5% gate on P99 while OLMoE
(top-8) clears it with a large margin; is this because larger K compresses
the token-race benefit (each token's own max-of-K is closer to the batch's
global max as K grows)?

Method: reuse the SAME real per-token, per-layer expert preference data
already captured (rank is confirmed to be gate_weight-descending, i.e.
rank<=K gives the REAL top-K subset the model would have used at a smaller
K, not a synthetic reroute). For each model, sweep K in {2, 4, 8} (OLMoE,
whose native K=8) and {2, 4, 8, 16} (LLM-jp, whose native K=16), truncating
each token's real expert list to its top-K by rank before running the exact
same full_barrier vs token_race simulation as the P0.

This is a real-data ablation, not a new synthetic model: it isolates K as
the controlled variable while holding the model's actual routing
preferences, hidden size proxy (via BASE_LAUNCH_US/PER_TOKEN_US, unchanged
from the P0), and noise regimes fixed.

Evidence tags: all outputs are [Observed] given the stated [Hypothesis]
compute/variance model (identical to run_tokenrace_ep_p0.py).
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
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from run_tokenrace_ep_p0 import (
    ATTN_SHARED_US,
    NOISE_REGIMES,
    MIN_IMPROVEMENT,
    MODEL_SPECS,
    simulate_decode_step,
    pctl,
)


def load_route_index_truncated(csv_path: Path, max_k: int):
    """Build {(sample_id, token_position): {layer: [expert_id,...]}} keeping
    only rows with rank <= max_k (real top-K subset by gate_weight order)."""
    df = pd.read_csv(csv_path)
    df = df[df["rank"] <= max_k]
    index: dict = {}
    for (sample_id, token_position), grp in df.groupby(["sample_id", "token_position"], sort=False):
        by_layer: dict[int, list[int]] = {}
        for layer, sub in grp.groupby("layer", sort=False):
            by_layer[int(layer)] = sub["expert_id"].to_numpy().tolist()
        index[(int(sample_id), int(token_position))] = by_layer
    return index


def run_model_k(model_key: str, root: Path, k: int, n_decode_steps: int, batch_size: int, seed: int) -> dict:
    spec = MODEL_SPECS[model_key]
    csv_path = root / spec.route_csv
    route_index = load_route_index_truncated(csv_path, k)
    all_keys = list(route_index.keys())
    rng = np.random.default_rng(seed)

    out = {}
    for regime_name, regime in NOISE_REGIMES.items():
        barrier_vals, race_vals = [], []
        for _ in range(n_decode_steps):
            idx = rng.choice(len(all_keys), size=batch_size, replace=False)
            batch_keys = [all_keys[i] for i in idx]
            barrier_total, race_totals = simulate_decode_step(
                batch_keys, route_index, spec.n_layers, regime, rng
            )
            barrier_vals.append(barrier_total)
            race_vals.extend(race_totals.tolist())
        barrier_arr = np.array(barrier_vals)
        race_arr = np.array(race_vals)

        def imp(fn):
            b, r = fn(barrier_arr), fn(race_arr)
            return (b - r) / b if b > 0 else 0.0

        out[regime_name] = {
            "improvement_p50": imp(lambda a: pctl(a, 50)),
            "improvement_p99": imp(lambda a: pctl(a, 99)),
            "improvement_mean": imp(np.mean),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--n-decode-steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260719)
    ap.add_argument("--output-dir", default="outputs/tokenrace_ep_p0b_topk_sweep_2026-07-19")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_plan = {
        "olmoe": [2, 4, 8],
        "llmjp": [2, 4, 8, 16],
    }

    results = {}
    for model_key, ks in sweep_plan.items():
        results[model_key] = {}
        for k in ks:
            print(f"[run] {model_key} K={k} ...", file=sys.stderr)
            results[model_key][str(k)] = run_model_k(
                model_key, root, k, args.n_decode_steps, args.batch_size, args.seed
            )

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    lines = ["# TokenRace-EP P0-B：真实 top-K 截断消融（K 对收益的影响）\n"]
    lines.append(f"batch_size={args.batch_size}, n_decode_steps={args.n_decode_steps}, seed={args.seed}\n")
    for model_key, ks in sweep_plan.items():
        lines.append(f"\n## {MODEL_SPECS[model_key].name}\n")
        lines.append("| K | none P99 | moderate P50 | moderate P99 | moderate mean | severe P99 |")
        lines.append("|---|---|---|---|---|---|")
        for k in ks:
            r = results[model_key][str(k)]
            lines.append(
                f"| {k} | {r['none']['improvement_p99']*100:.2f}% | "
                f"{r['moderate']['improvement_p50']*100:.2f}% | "
                f"{r['moderate']['improvement_p99']*100:.2f}% | "
                f"{r['moderate']['improvement_mean']*100:.2f}% | "
                f"{r['severe']['improvement_p99']*100:.2f}% |"
            )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] -> {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()

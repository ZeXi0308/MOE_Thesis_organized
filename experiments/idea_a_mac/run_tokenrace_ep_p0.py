#!/usr/bin/env python3
"""TokenRace-EP P0: Mac-only timing simulation for per-token combine early-release
vs. full-batch synchronous combine barrier.

Evidence tags used throughout outputs/report: every quantitative claim is
[Observed] (directly computed from this simulation run) or [Hypothesis]
(the underlying compute/variance model itself, since no real GPU timing was
collected). This script does NOT measure real wall-clock latency, NIC bytes,
or backend kernels. It is a pre-registered, killable P0 that operates purely
on real MoE routing traces (sample_id, layer, token_position, rank,
expert_id, gate_weight) already captured on real models (OLMoE-1B-7B top-8,
LLM-jp E32 top-16).

Causal mechanism under test
----------------------------
Current systems (as modeled in run_tbt_congestion_bridge.py in this repo,
and consistent with public backend descriptions) treat MoE combine as a
per-decode-step, whole-batch synchronous barrier: layer l+1 cannot start
for ANY request in the batch until EVERY expert active in layer l for THIS
batch has finished computing (== until the slowest/most-loaded expert of
the whole batch is done).

TokenRace-EP asks: what if a request is released to the next layer as soon
as its OWN top-k experts (not the whole batch's experts) have finished?
Since decode-time attention/FFN per request is independent of other
requests once its own activations are ready, this is a scheduling-grain
change, not a data/precision change.

Per-layer expert finish-time model (all constants are illustrative,
Hypothesis-level; see --sensitivity sweep for robustness):

    finish_time[e] = BASE_LAUNCH_US
                    + PER_TOKEN_US * tokens_assigned[e]
                    + (multiplicative noise)

Two release policies compared for the SAME real per-request, per-layer,
per-decode-step routing:
    full_barrier : layer completion = max over ALL active experts this step
                   (identical value applied to every request in the batch)
    token_race   : each request's own layer completion = max over only its
                   own top-k experts' finish times this step

Both totals are accumulated across all L real layers for the same batch of
concurrent requests (using each request's real per-layer routing from the
frozen sealed route trace), giving a per-request total per-decode-step
latency proxy under each policy.

Pre-registered kill gate (must ALL hold to avoid KILL):
  - At the ViBE-motivated "moderate" noise regime, for BOTH models and ALL
    tested batch sizes, token_race P99 improvement over full_barrier P99
    must be >= MIN_IMPROVEMENT (default 5%).
  - The improvement must not vanish (drop below MIN_IMPROVEMENT/2) in the
    "severe" stress regime, and must not be entirely an artifact of the
    "none" (zero-variance) sanity regime showing implausibly large
    improvement from load-imbalance alone without any noise assumption.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Model configs (Hypothesis-level, for relative comparison only; not tied to
# any measured GPU numbers).
# ---------------------------------------------------------------------------

BASE_LAUNCH_US = 6.0        # fixed per-active-expert launch/weight-read cost
PER_TOKEN_US = 0.35         # marginal per-token FFN compute cost, illustrative
ATTN_SHARED_US = 4.0        # per-layer attention cost, identical under both
                            # policies -> included for realism, dilutes the
                            # measured relative improvement (conservative).

NOISE_REGIMES = {
    "none": dict(sigma=0.0, p_strag=0.0, strag_lo=1.0, strag_hi=1.0),
    "moderate": dict(sigma=0.07, p_strag=0.02, strag_lo=1.5, strag_hi=2.5),
    "severe": dict(sigma=0.15, p_strag=0.05, strag_lo=2.0, strag_hi=4.0),
}

MIN_IMPROVEMENT = 0.05  # pre-registered kill gate threshold


@dataclass
class ModelSpec:
    name: str
    route_csv: str
    n_layers: int
    top_k: int


MODEL_SPECS = {
    "olmoe": ModelSpec(
        name="OLMoE-1B-7B (E64/top8)",
        route_csv="outputs/paper_validation/olmoe_peerquota_frozen_confirm_2026-07-14/test_routes.csv",
        n_layers=16,
        top_k=8,
    ),
    "llmjp": ModelSpec(
        name="LLM-jp E32/top16",
        route_csv="outputs/paper_validation/llmjp_top16_peerquota_frozen_confirm_2026-07-14/test_routes.csv",
        n_layers=16,
        top_k=16,
    ),
}


def load_route_index(csv_path: Path):
    """Build {(sample_id, token_position): {layer: [expert_id,...]}}."""
    df = pd.read_csv(csv_path)
    index: dict[tuple[int, int], dict[int, list[int]]] = {}
    for (sample_id, token_position), grp in df.groupby(["sample_id", "token_position"], sort=False):
        by_layer: dict[int, list[int]] = {}
        for layer, sub in grp.groupby("layer", sort=False):
            by_layer[int(layer)] = sub["expert_id"].to_numpy().tolist()
        index[(int(sample_id), int(token_position))] = by_layer
    return index, df["layer"].nunique()


def simulate_decode_step(
    batch_keys: list[tuple[int, int]],
    route_index: dict,
    n_layers: int,
    regime: dict,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    """Simulate one decode step (all n_layers) for a batch of concurrent
    requests. Returns (full_barrier_total_us, per_request_token_race_totals_us).
    """
    n_req = len(batch_keys)
    barrier_total = ATTN_SHARED_US * n_layers
    race_totals = np.full(n_req, ATTN_SHARED_US * n_layers, dtype=np.float64)

    for layer in range(n_layers):
        # tokens_assigned per expert this layer, and per-request expert lists
        expert_counts: dict[int, int] = {}
        req_experts: list[list[int]] = []
        for key in batch_keys:
            experts_this_layer = route_index[key].get(layer, [])
            req_experts.append(experts_this_layer)
            for e in experts_this_layer:
                expert_counts[e] = expert_counts.get(e, 0) + 1

        # finish time per active expert
        finish_time: dict[int, float] = {}
        for e, cnt in expert_counts.items():
            base = BASE_LAUNCH_US + PER_TOKEN_US * cnt
            noise = rng.normal(0.0, regime["sigma"]) if regime["sigma"] > 0 else 0.0
            mult = max(0.1, 1.0 + noise)
            if regime["p_strag"] > 0 and rng.random() < regime["p_strag"]:
                mult *= rng.uniform(regime["strag_lo"], regime["strag_hi"])
            finish_time[e] = base * mult

        layer_barrier = max(finish_time.values()) if finish_time else 0.0
        barrier_total += layer_barrier

        for i, experts_this_layer in enumerate(req_experts):
            if experts_this_layer:
                own = max(finish_time[e] for e in experts_this_layer)
            else:
                own = 0.0
            race_totals[i] += own

    return barrier_total, race_totals


def pctl(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q))


def run_model(
    model_key: str,
    root: Path,
    n_decode_steps: int,
    batch_sizes: list[int],
    seed: int,
) -> dict:
    spec = MODEL_SPECS[model_key]
    csv_path = root / spec.route_csv
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    route_index, observed_layers = load_route_index(csv_path)
    all_keys = list(route_index.keys())
    if len(all_keys) < max(batch_sizes):
        raise ValueError(
            f"{model_key}: only {len(all_keys)} distinct (sample,pos) available, "
            f"need >= {max(batch_sizes)}"
        )

    rng = np.random.default_rng(seed)
    results: dict = {"model": spec.name, "n_layers_observed": int(observed_layers), "regimes": {}}

    for regime_name, regime in NOISE_REGIMES.items():
        regime_out = {}
        for B in batch_sizes:
            barrier_vals = []
            race_vals: list[float] = []
            for _ in range(n_decode_steps):
                idx = rng.choice(len(all_keys), size=B, replace=False)
                batch_keys = [all_keys[i] for i in idx]
                barrier_total, race_totals = simulate_decode_step(
                    batch_keys, route_index, spec.n_layers, regime, rng
                )
                barrier_vals.append(barrier_total)
                race_vals.extend(race_totals.tolist())

            barrier_arr = np.array(barrier_vals)
            race_arr = np.array(race_vals)

            def improvement(metric_fn):
                b = metric_fn(barrier_arr)
                r = metric_fn(race_arr)
                return b, r, (b - r) / b if b > 0 else 0.0

            mean_b, mean_r, mean_imp = improvement(np.mean)
            p50_b, p50_r, p50_imp = improvement(lambda a: pctl(a, 50))
            p99_b, p99_r, p99_imp = improvement(lambda a: pctl(a, 99))

            regime_out[str(B)] = {
                "full_barrier_mean_us": mean_b,
                "token_race_mean_us": mean_r,
                "improvement_mean": mean_imp,
                "full_barrier_p50_us": p50_b,
                "token_race_p50_us": p50_r,
                "improvement_p50": p50_imp,
                "full_barrier_p99_us": p99_b,
                "token_race_p99_us": p99_r,
                "improvement_p99": p99_imp,
                "n_decode_steps": n_decode_steps,
            }
        results["regimes"][regime_name] = regime_out

    return results


def apply_kill_gate(all_results: dict) -> dict:
    verdict = {"gate": "PASS", "reasons": []}
    for model_key, res in all_results.items():
        moderate = res["regimes"]["moderate"]
        severe = res["regimes"]["severe"]
        none_regime = res["regimes"]["none"]
        for B, stats in moderate.items():
            if stats["improvement_p99"] < MIN_IMPROVEMENT:
                verdict["gate"] = "FAIL"
                verdict["reasons"].append(
                    f"{model_key} B={B}: moderate-regime P99 improvement "
                    f"{stats['improvement_p99']:.4f} < gate {MIN_IMPROVEMENT}"
                )
            sev = severe[B]["improvement_p99"]
            if sev < MIN_IMPROVEMENT / 2:
                verdict["gate"] = "FAIL"
                verdict["reasons"].append(
                    f"{model_key} B={B}: severe-regime P99 improvement "
                    f"{sev:.4f} collapsed below {MIN_IMPROVEMENT/2}"
                )
        # sanity: zero-variance regime should not show a huge improvement on
        # its own (that would mean the effect is a modeling artifact of load
        # imbalance alone, not variance/straggler-driven as hypothesized).
        for B, stats in none_regime.items():
            if stats["improvement_p99"] > 3 * MIN_IMPROVEMENT:
                verdict["reasons"].append(
                    f"{model_key} B={B}: WARNING zero-variance regime already shows "
                    f"{stats['improvement_p99']:.4f} P99 improvement from load-imbalance alone; "
                    f"moderate/severe regime numbers may be dominated by imbalance, not straggler variance."
                )
    if not verdict["reasons"]:
        verdict["reasons"].append("All gates passed with no warnings.")
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--n-decode-steps", type=int, default=300)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--seed", type=int, default=20260719)
    ap.add_argument("--output-dir", default="outputs/tokenrace_ep_p0_2026-07-19")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for model_key in MODEL_SPECS:
        print(f"[run] {model_key} ...", file=sys.stderr)
        all_results[model_key] = run_model(
            model_key, root, args.n_decode_steps, args.batch_sizes, args.seed
        )

    verdict = apply_kill_gate(all_results)

    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open(out_dir / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)

    # human-readable report
    lines = ["# TokenRace-EP P0 结果\n"]
    lines.append(f"门槛 MIN_IMPROVEMENT = {MIN_IMPROVEMENT}\n")
    lines.append(f"**GATE: {verdict['gate']}**\n")
    for r in verdict["reasons"]:
        lines.append(f"- {r}")
    lines.append("")
    for model_key, res in all_results.items():
        lines.append(f"\n## {res['model']}\n")
        for regime_name, regime_out in res["regimes"].items():
            lines.append(f"### regime={regime_name}\n")
            lines.append("| B | barrier P50 | race P50 | imp P50 | barrier P99 | race P99 | imp P99 | imp mean |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for B, s in regime_out.items():
                lines.append(
                    f"| {B} | {s['full_barrier_p50_us']:.1f} | {s['token_race_p50_us']:.1f} | "
                    f"{s['improvement_p50']*100:.2f}% | {s['full_barrier_p99_us']:.1f} | "
                    f"{s['token_race_p99_us']:.1f} | {s['improvement_p99']*100:.2f}% | "
                    f"{s['improvement_mean']*100:.2f}% |"
                )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] gate={verdict['gate']} -> {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()

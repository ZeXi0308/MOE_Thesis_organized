#!/usr/bin/env python3
"""Offline TokenRace P1 corrected recompute (no GPU / no torch).

Corrects two accounting issues from 2026-07-19/20:

1. Graph advantage must credit the *barrier* path (subtract per layer),
   not be added into token_race rebatch overhead.
2. Rebatch should not be charged unconditionally to every request; early
   requests (own < layer_barrier) pay own+rebatch, late requests pay
   layer_barrier with no rebatch.

Uses constants from existing gpu_p0 / gpu_p1 JSON and paper_validation
route CSVs. See docs/archive/killed_ideas/errata/判死结论勘误_2026-07-21.md.
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
from pathlib import Path

import numpy as np
import pandas as pd

NOISE_REGIMES = {
    "moderate": dict(sigma=0.07, p_strag=0.02, strag_lo=1.5, strag_hi=2.5),
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


def pctl(arr, q):
    return float(np.percentile(arr, q))


def simulate_decode_step(
    batch_keys, route_index, n_layers, regime, rng,
    base_us: float, per_token_us: float, rebatch_overhead_us: float,
    attn_shared_us: float, barrier_graph_credit_us_per_layer: float = 0.0,
    rebatch_mode: str = "early_only",
):
    """rebatch_mode:
      - unconditional: every request pays own + rebatch (legacy P0/P1)
      - early_only: early requests pay own+rebatch; late pay barrier (corrected)
    barrier_graph_credit_us_per_layer: subtracted from barrier each layer
      (correct placement of CUDA graph advantage).
    """
    n_req = len(batch_keys)
    barrier_total = attn_shared_us * n_layers
    race_totals = np.full(n_req, attn_shared_us * n_layers, dtype=np.float64)

    for layer in range(n_layers):
        expert_counts: dict[int, int] = {}
        req_experts: list[list[int]] = []
        for key in batch_keys:
            experts_this_layer = route_index[key].get(layer, [])
            req_experts.append(experts_this_layer)
            for e in experts_this_layer:
                expert_counts[e] = expert_counts.get(e, 0) + 1

        finish_time: dict[int, float] = {}
        for e, cnt in expert_counts.items():
            base = base_us + per_token_us * cnt
            noise = rng.normal(0.0, regime["sigma"]) if regime["sigma"] > 0 else 0.0
            mult = max(0.1, 1.0 + noise)
            if regime["p_strag"] > 0 and rng.random() < regime["p_strag"]:
                mult *= rng.uniform(regime["strag_lo"], regime["strag_hi"])
            finish_time[e] = base * mult

        layer_barrier = max(finish_time.values()) if finish_time else 0.0
        barrier_total += max(0.0, layer_barrier - barrier_graph_credit_us_per_layer)

        for i, experts_this_layer in enumerate(req_experts):
            if not experts_this_layer:
                continue
            own = max(finish_time[e] for e in experts_this_layer)
            if rebatch_mode == "unconditional":
                race_totals[i] += own + rebatch_overhead_us
            elif rebatch_mode == "early_only":
                if own < layer_barrier - 1e-12:
                    race_totals[i] += own + rebatch_overhead_us
                else:
                    race_totals[i] += layer_barrier
            else:
                raise ValueError(f"unknown rebatch_mode={rebatch_mode}")

    return barrier_total, race_totals


def run_scenario(
    root: Path, model_key: str, base_us: float, per_token_us: float,
    rebatch_overhead_us: float, graph_credit: float, rebatch_mode: str,
    n_decode_steps: int, batch_sizes: list[int], seed: int,
):
    spec = MODEL_SPECS[model_key]
    route_index = load_route_index(root / spec["route_csv"])
    all_keys = list(route_index.keys())
    attn_shared_us = base_us * 0.5
    rng = np.random.default_rng(seed)
    regime = NOISE_REGIMES["moderate"]
    out = {}
    for B in batch_sizes:
        barrier_vals, race_vals = [], []
        for _ in range(n_decode_steps):
            idx = rng.choice(len(all_keys), size=B, replace=False)
            batch_keys = [all_keys[i] for i in idx]
            b_total, r_totals = simulate_decode_step(
                batch_keys, route_index, spec["n_layers"], regime, rng,
                base_us, per_token_us, rebatch_overhead_us, attn_shared_us,
                barrier_graph_credit_us_per_layer=graph_credit,
                rebatch_mode=rebatch_mode,
            )
            barrier_vals.append(b_total)
            race_vals.extend(r_totals.tolist())
        b_arr, r_arr = np.asarray(barrier_vals), np.asarray(race_vals)

        def imp(fn):
            b, r = fn(b_arr), fn(r_arr)
            return (b - r) / b if b > 0 else 0.0

        out[str(B)] = {
            "improvement_p50": imp(lambda a: pctl(a, 50)),
            "improvement_p99": imp(lambda a: pctl(a, 99)),
            "improvement_mean": imp(np.mean),
            "passes_5pct_p99": imp(lambda a: pctl(a, 99)) >= MIN_IMPROVEMENT,
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--p0-json", default="outputs/tokenrace_gpu_p0_2026-07-19/gpu_p0_results.json")
    ap.add_argument("--p1-json", default="outputs/tokenrace_gpu_p1_2026-07-19/gpu_p1_results.json")
    ap.add_argument("--n-decode-steps", type=int, default=1000)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--seed", type=int, default=20260721)
    ap.add_argument("--output-dir", default="outputs/tokenrace_p1_corrected_recompute_2026-07-21")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    p0 = json.loads((root / args.p0_json).read_text())
    p1 = json.loads((root / args.p1_json).read_text())

    # Prefer P1 base_consts (re-fit); fall back to P0 real_constants
    if "base_consts" in p1:
        consts = p1["base_consts"]
    else:
        consts = p0["real_constants"]

    graph_adv = float(p1["avg_graph_advantage_us"])
    mg = p1["multigroup_rebatch_us"]
    rebatch_2 = float((mg["olmoe"]["2"] + mg["llmjp"]["2"]) / 2.0)
    rebatch_4 = float((mg["olmoe"]["4"] + mg["llmjp"]["4"]) / 2.0)
    # Legacy P0 best-case gather+launch from P0 json if present
    rebatch_p0 = float(p0.get("real_constants", {}).get("rebatch_overhead_us", rebatch_2))

    scenarios = {
        "legacy_p1_ii_graph_on_race_unconditional": dict(
            rebatch=rebatch_2 + graph_adv, graph_credit=0.0, rebatch_mode="unconditional",
            note="OLD P1 scenario(ii): graph added to race overhead (incorrect placement)",
        ),
        "corrected_barrier_graph_credit_early_only_2group": dict(
            rebatch=rebatch_2, graph_credit=graph_adv, rebatch_mode="early_only",
            note="CORRECT: barrier minus graph credit; early-only rebatch; 2-group rebatch",
        ),
        "corrected_barrier_graph_credit_early_only_4group": dict(
            rebatch=rebatch_4, graph_credit=graph_adv, rebatch_mode="early_only",
            note="CORRECT accounting with 4-group rebatch tax",
        ),
        "corrected_no_graph_early_only_p0_rebatch": dict(
            rebatch=rebatch_p0, graph_credit=0.0, rebatch_mode="early_only",
            note="No graph asymmetry; early-only rebatch at P0 measured overhead",
        ),
        "legacy_unconditional_p0_rebatch": dict(
            rebatch=rebatch_p0, graph_credit=0.0, rebatch_mode="unconditional",
            note="Legacy unconditional P0-style for comparison",
        ),
    }

    rows = []
    results = {}
    for scen_name, cfg in scenarios.items():
        results[scen_name] = {"note": cfg["note"], "models": {}}
        for model_key in MODEL_SPECS:
            base_us = float(consts[model_key]["fit_base_us"])
            per_token_us = float(consts[model_key]["fit_per_token_us"])
            model_out = run_scenario(
                root, model_key, base_us, per_token_us,
                cfg["rebatch"], cfg["graph_credit"], cfg["rebatch_mode"],
                args.n_decode_steps, args.batch_sizes, args.seed,
            )
            results[scen_name]["models"][model_key] = model_out
            for B, stats in model_out.items():
                rows.append({
                    "scenario": scen_name,
                    "note": cfg["note"],
                    "model": model_key,
                    "batch_size": int(B),
                    "rebatch_overhead_us": cfg["rebatch"],
                    "barrier_graph_credit_us": cfg["graph_credit"],
                    "rebatch_mode": cfg["rebatch_mode"],
                    **stats,
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "corrected_scenarios.csv", index=False)
    (out_dir / "results.json").write_text(json.dumps({
        "graph_advantage_us": graph_adv,
        "rebatch_2group_us": rebatch_2,
        "rebatch_4group_us": rebatch_4,
        "rebatch_p0_us": rebatch_p0,
        "scenarios": results,
    }, indent=2), encoding="utf-8")

    # Verdict: does any corrected scenario pass 5% on BOTH models for all B?
    corrected_names = [n for n in scenarios if n.startswith("corrected_")]
    any_full_pass = False
    lines = [
        "# TokenRace-EP P1 Corrected Offline Recompute (2026-07-21)",
        "",
        f"graph_advantage_us={graph_adv:.3f}, rebatch_2={rebatch_2:.3f}, rebatch_4={rebatch_4:.3f}",
        "",
        "Interpretation: correcting graph placement credits the barrier (smaller B),",
        "which typically makes negative improvements *more* negative. Early-only rebatch",
        "is less pessimistic than unconditional. Expectation: does **not** resurrect TokenRace.",
        "",
        "| scenario | model | B | P50 | P99 | pass5% |",
        "|---|---|---:|---:|---:|---|",
    ]
    for _, row in df.sort_values(["scenario", "model", "batch_size"]).iterrows():
        lines.append(
            f"| {row['scenario']} | {row['model']} | {row['batch_size']} | "
            f"{row['improvement_p50']*100:.2f}% | {row['improvement_p99']*100:.2f}% | "
            f"{row['passes_5pct_p99']} |"
        )

    lines.append("")
    lines.append("## Gate summary (corrected scenarios only)")
    for scen in corrected_names:
        sub = df[df["scenario"] == scen]
        ok = bool(sub["passes_5pct_p99"].all())
        any_full_pass = any_full_pass or ok
        min_p99 = float(sub["improvement_p99"].min())
        lines.append(f"- `{scen}`: all_cells_pass_5pct={ok}, min_p99={min_p99*100:.2f}%")

    if any_full_pass:
        verdict = (
            "UNEXPECTED_PARTIAL_OPEN: at least one corrected scenario passes 5% on all "
            "model×B cells — re-examine before maintaining hard KILL."
        )
    else:
        verdict = (
            "MAINTAIN_KILLED_AFTER_CORRECTION: no corrected scenario passes the 5% P99 "
            "gate across all model×B cells. P1 accounting was sloppy but fixing it does "
            "not resurrect TokenRace; if anything barrier graph credit deepens negatives."
        )
    lines.append("")
    lines.append(f"**Verdict:** {verdict}")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "decision.json").write_text(json.dumps({
        "verdict": verdict,
        "any_corrected_full_pass": any_full_pass,
    }, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()

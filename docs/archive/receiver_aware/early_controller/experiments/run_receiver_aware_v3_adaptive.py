#!/usr/bin/env python3
"""Receiver-Aware v3: causally-valid regime detection + adaptive controller.

v2 established a precise, uncomfortable fact: the "right" policy flips
depending on congestion regime.
  - hotspot (structural/persistent): calib_static (pure offline profile,
    zero online signal) already captures almost all the achievable saving.
  - balanced (transient/random): the realistic online signal
    (causal_prev_step) is WORSE than random -- staleness turns the "free"
    signal into a liability.

v2 never built a controller that could tell these two regimes apart at
*deploy time* -- it only reported the two regimes separately because it knew
the ground-truth origin_mode label. A real system does not get that label.
v3 closes that gap with a lightweight, causally-valid regime detector:

  1. Detector calibration (offline, once): using ONLY the calibration
     scenarios (already used to build calib_static; no test-set leakage),
     measure the receiver-load-vector autocorrelation between consecutive
     global steps for a "hotspot" style workload and a "balanced" style
     workload, and record the two reference levels. Set a decision threshold
     at their midpoint.
  2. Online detection (per test scenario, causal): using only the FIRST
     `detect_frac` fraction of that scenario's own global steps (a short
     warm-up window, no knowledge of the rest), compute the same
     autocorrelation statistic and compare to the threshold.
  3. Policy selection: if detected "structural", use calib_static for the
     REST of the scenario (near-oracle in that regime, per v2). If detected
     "transient", fall back to `random` for the rest (avoids causal's
     negative bias in that regime, per v2).

This is compared against three static (non-adaptive) baselines applied
blindly to the SAME pooled mixed-regime test set (50% hotspot + 50% balanced
scenarios, unknown to the controller): always calib_static, always
causal_prev_step, always random. The adaptive controller does not know which
regime it is in ahead of time -- it has to detect it from data, same as a
real deployed system would.

Evidence tag: [Observed], same route data / scenario generator as v2, only
the decision rule is new.
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
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoConfig

from run_receiver_aware_v2_systematic import (
    build_scenario,
    load_model_config,
    placement_map,
    remote_loads_by_step,
    select_indices,
    step_us_from_counts,
)


def receiver_concentration(remote_rows: pd.DataFrame, g_max: int, ep_size: int) -> float:
    """Herfindahl concentration index of TOTAL load-by-receiver over the
    observation window (g <= g_max): sum(share_r^2), in [1/ep_size, 1].
    Hotspot-style workloads concentrate load on 1-2 receivers by construction
    (high value); balanced/round-robin workloads spread load evenly across
    all receivers (value close to 1/ep_size, the uniform floor). This is a
    directly diagnostic, causally-valid statistic measurable from any short
    online observation window, no lag/autocorrelation assumptions needed."""
    rows = remote_rows[remote_rows["g"] <= g_max]
    if rows.empty:
        return 1.0 / ep_size
    counts = rows.groupby("receiver_rank").size().reindex(range(ep_size), fill_value=0).to_numpy(dtype=float)
    total = counts.sum()
    if total <= 0:
        return 1.0 / ep_size
    shares = counts / total
    return float((shares ** 2).sum())


def calibrate_threshold(routes: pd.DataFrame, doc_pool: list[int], calib_docs: list[int],
                         num_experts: int, ep_size: int, placement: str, gpus_per_node: int) -> float:
    levels = {}
    for origin_mode in ("hotspot", "balanced"):
        arrivals = np.zeros(len(calib_docs), dtype=int)
        scn = build_scenario(routes, calib_docs, arrivals, origin_mode, ep_size)
        scn["sender_rank"] = placement_map(scn["expert_id"].to_numpy(), num_experts, ep_size, placement)
        remote = scn[(scn["sender_rank"] // gpus_per_node) != (scn["receiver_rank"] // gpus_per_node)]
        g_max = int(remote["g"].max()) if not remote.empty else 0
        levels[origin_mode] = receiver_concentration(remote, g_max, ep_size)
    return 0.5 * (levels["hotspot"] + levels["balanced"]), levels


def run_adaptive_cell(
    routes: pd.DataFrame, doc_pool: list[int], test_docs: list[int], num_experts: int, top_k: int,
    ep_size: int, gpus_per_node: int, placement: str, origin_mode: str, fraction: float,
    inter_node_gbps: float, hidden_size: int, static_profile: dict, threshold: float,
    detect_frac: float, num_jobs: int, num_scenario_seeds: int, num_random_controls: int,
    scenario_seed_base: int,
) -> list[dict]:
    bw_bytes_per_us = inter_node_gbps * 1e9 / 8 / 1e6
    bytes_to_us = hidden_size / bw_bytes_per_us
    rows = []
    for scenario_seed in range(num_scenario_seeds):
        rng = np.random.default_rng(scenario_seed_base + scenario_seed)
        chosen_docs = rng.choice(test_docs, size=min(num_jobs, len(test_docs)),
                                  replace=len(test_docs) < num_jobs).tolist()
        num_layers = int(routes["layer"].max()) + 1
        max_stagger = max(1, num_layers // 2)
        arrivals = rng.integers(0, max_stagger + 1, size=len(chosen_docs))
        scn = build_scenario(routes, chosen_docs, arrivals, origin_mode, ep_size)
        scn["sender_rank"] = placement_map(scn["expert_id"].to_numpy(), num_experts, ep_size, placement)

        remote_rows = scn[(scn["sender_rank"] // gpus_per_node) != (scn["receiver_rank"] // gpus_per_node)]
        base_ingress = remote_rows.groupby(["g", "receiver_rank"]).size().astype(float)
        base_egress = remote_rows.groupby(["g", "sender_rank"]).size().astype(float)
        fp8_step_us = pd.concat(
            [base_ingress.groupby(level="g").max(), base_egress.groupby(level="g").max()], axis=1
        ).max(axis=1) * bytes_to_us
        fp8_total = float(fp8_step_us.sum())

        tail_mask = remote_rows["rank"].astype(int) > (top_k - max(1, top_k // 2))
        cand_rows = remote_rows[tail_mask]
        loads_by_step = remote_loads_by_step(remote_rows)

        g_max_all = int(remote_rows["g"].max()) if not remote_rows.empty else 0
        detect_cutoff = int(round(g_max_all * detect_frac))
        detected_ac = receiver_concentration(remote_rows, detect_cutoff, ep_size)
        detected_regime = "structural" if detected_ac >= threshold else "transient"
        adaptive_info_mode = "calib_static" if detected_regime == "structural" else "random"

        def total_saving(selection_mode: str, info_mode: str, rand_seed: int) -> float:
            idx = select_indices(cand_rows, loads_by_step, static_profile, selection_mode, info_mode, fraction, rand_seed)
            step_us = step_us_from_counts(base_ingress, base_egress, cand_rows, idx, bytes_to_us)
            return 1.0 - float(step_us.sum()) / max(fp8_total, 1e-12)

        adaptive_saving = (
            total_saving("hot", "calib_static", 0) if adaptive_info_mode == "calib_static"
            else total_saving("random", "random", 777 + scenario_seed)
        )
        calib_static_saving = total_saving("hot", "calib_static", 0)
        causal_saving = total_saving("hot", "causal_prev_step", 0)
        random_savings = [total_saving("random", "random", seed) for seed in range(num_random_controls)]

        rows.append({
            "origin_mode": origin_mode, "placement": placement, "budget_fraction": fraction,
            "scenario_seed": scenario_seed, "detected_autocorr": detected_ac,
            "detected_regime": detected_regime, "adaptive_saving": adaptive_saving,
            "always_calib_static_saving": calib_static_saving,
            "always_causal_saving": causal_saving,
            "always_random_saving_mean": float(np.mean(random_savings)),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--model-key", default="olmoe")
    ap.add_argument("--routes", required=True)
    ap.add_argument("--ep-size", type=int, default=8)
    ap.add_argument("--gpus-per-node", type=int, default=4)
    ap.add_argument("--num-jobs", type=int, default=16)
    ap.add_argument("--placement", default="contiguous")
    ap.add_argument("--budget-fraction", type=float, default=0.5)
    ap.add_argument("--inter-node-gbps", type=float, default=200.0)
    ap.add_argument("--calib-jobs", type=int, default=12)
    ap.add_argument("--detect-frac", type=float, default=0.3)
    ap.add_argument("--num-scenario-seeds", type=int, default=24)
    ap.add_argument("--num-random-controls", type=int, default=20)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    num_experts, top_k = load_model_config(args.model)
    routes = pd.read_csv(args.routes)
    hidden_size = int(AutoConfig.from_pretrained(args.model, local_files_only=True).hidden_size)
    doc_pool = sorted(routes["sample_id"].unique().tolist())
    calib_docs = doc_pool[:args.calib_jobs]
    test_docs = doc_pool[args.calib_jobs:]

    threshold, levels = calibrate_threshold(routes, doc_pool, calib_docs, num_experts, args.ep_size,
                                             args.placement, args.gpus_per_node)
    print(f"[{args.model_key}] regime detector calibration: hotspot_autocorr={levels['hotspot']:.4f}, "
          f"balanced_autocorr={levels['balanced']:.4f}, threshold={threshold:.4f}")

    calib_arrivals = np.zeros(len(calib_docs), dtype=int)
    all_rows = []
    for origin_mode in ("hotspot", "balanced"):
        calib_scn = build_scenario(routes, calib_docs, calib_arrivals, origin_mode, args.ep_size)
        calib_scn["sender_rank"] = placement_map(calib_scn["expert_id"].to_numpy(), num_experts, args.ep_size, args.placement)
        calib_remote = calib_scn[(calib_scn["sender_rank"] // args.gpus_per_node) != (calib_scn["receiver_rank"] // args.gpus_per_node)]
        n_steps_calib = max(calib_scn["g"].nunique(), 1)
        static_profile = {
            "sender": calib_remote.groupby("sender_rank").size() / n_steps_calib,
            "receiver": calib_remote.groupby("receiver_rank").size() / n_steps_calib,
        }
        rows = run_adaptive_cell(
            routes, doc_pool, test_docs, num_experts, top_k, args.ep_size, args.gpus_per_node,
            args.placement, origin_mode, args.budget_fraction, args.inter_node_gbps, hidden_size,
            static_profile, threshold, args.detect_frac, args.num_jobs, args.num_scenario_seeds,
            args.num_random_controls, scenario_seed_base=20260720_00 if origin_mode == "hotspot" else 20260720_50,
        )
        all_rows.extend(rows)
        print(f"[{args.model_key}] {origin_mode}: done ({len(rows)} scenario seeds)")

    df = pd.DataFrame(all_rows)
    df.insert(0, "model", args.model_key)
    df.to_csv(out / f"{args.model_key}_adaptive_raw.csv", index=False)

    detect_acc = df.groupby("origin_mode").apply(
        lambda g: float((g["detected_regime"] == ("structural" if g.name == "hotspot" else "transient")).mean())
    )
    pooled_mean = df[["adaptive_saving", "always_calib_static_saving", "always_causal_saving",
                       "always_random_saving_mean"]].mean()

    lines = [f"# Receiver-Aware v3: Adaptive Regime-Detection Controller ({args.model_key})", "",
             f"regime-detector calibration: hotspot_autocorr={levels['hotspot']:.4f}, "
             f"balanced_autocorr={levels['balanced']:.4f}, threshold={threshold:.4f}", "",
             "## Detection accuracy (per true origin_mode, causally detected from first "
             f"{args.detect_frac*100:.0f}% of steps only)", ""]
    for om, acc in detect_acc.items():
        lines.append(f"- {om}: correct-regime detection rate = {acc:.4f}")
    lines.append("")
    lines.append("## Pooled mean saving across BOTH regimes (50/50 mix, unknown to controller in advance)")
    lines.append("")
    for k, v in pooled_mean.items():
        lines.append(f"- {k}: {v:.4f}")
    lines.append("")
    lines.append("## Per-regime breakdown")
    lines.append("")
    per_regime = df.groupby("origin_mode")[["adaptive_saving", "always_calib_static_saving",
                                             "always_causal_saving", "always_random_saving_mean"]].mean()
    cols = list(per_regime.reset_index().columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in per_regime.reset_index().iterrows():
        vals = [f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")

    (out / f"{args.model_key}_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()

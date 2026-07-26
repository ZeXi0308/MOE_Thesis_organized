#!/usr/bin/env python3
"""Receiver-Aware direct-benefit controller: threshold-fit, hysteresis-gated,
homogeneous one-shot lane selection with receiver/sender credit -- replacing
the v3 regime classifier (HHI autocorrelation -> calib_static/random switch).

READ THIS BEFORE RUNNING OR CITING RESULTS.

Why v3 is retired, not reused
------------------------------
v3's causally-corrected audit (2026-07-20) found the adaptive HHI classifier's
advantage over the strongest fixed baseline (``always_causal_prev_step``) has
a 95% CI crossing zero at every ``detect_frac``, and that a MORE accurate
classifier produced a WORSE net outcome (warm-up tax grew faster than
detection accuracy). The failure is structural: v3 commits an entire scenario
to one of two canned policies from a single scenario-level statistic. This
script replaces that with a LANE-level (per (sender_rank, receiver_rank) pair)
decision made independently every global step, so there is no scenario-level
classification to get wrong, and no warm-up-prefix look-back is possible by
construction (every step's decision only ever reads step g-1's realized load).

What "direct-benefit" means here
---------------------------------
The decision threshold is fit by directly measuring, on calibration scenarios
ONLY, the realized wall-clock saving of putting a lane into the low-bit state
for one step MINUS that state's real measured codec (pack+unpack) overhead
(read from ``run_receiver_codec_break_even_gpu.py``'s ``codec_break_even.csv``
-- real Triton-kernel timings, not a proxy). It is not a classifier trained to
separate "hotspot" from "balanced" traces; it never sees an origin_mode label,
calibration or test.

What "homogeneous one-shot lane" means here
---------------------------------------------
When a lane is in the low state at step g, EVERY candidate (token, rank) pair
routed across that lane at that step uses the SAME single low-bit format in
one shot (no progressive/residual encoding, no per-vector mixing). The
2026-07-20 progressive-quality-gate and codec-break-even audits both found
progressive/residual representations strictly worse (higher KL; break-even
0.9-59 Gbps depending on tile size, comfortably below modern EP link speeds)
than a single homogeneous low-bit choice.

What "receiver/sender credit" means here
-------------------------------------------
Each rank carries a scalar EWMA "credit" of its own recent (g-1 and earlier)
ingress/egress load. A lane (sender, receiver) may only flip full->low when
max(sender_credit, receiver_credit) rises above a fitted high threshold, and
only flips low->full when it falls below a strictly lower threshold (two-
sided hysteresis / Schmitt trigger), and must dwell in its current state for
at least ``dwell_min`` steps before flipping again. This directly targets the
v3 failure mode: bounding the flap rate is what a more-accurate-but-unbounded
classifier could not do.

STILL BANDWIDTH-MODEL REPLAY -- evidence boundary
----------------------------------------------------
This script has NO real NCCL/RDMA queueing, ECN, or incast measurement. Step
latency is the same analytic byte-accounting model as
``run_receiver_aware_v2_systematic.py`` (max ingress/egress byte-weight per
(g, rank) times a fixed inter-node Gbps). The one genuinely-more-real
ingredient is the codec pack/unpack overhead, which is real GPU-measured and
is added as a per-step tax on top of the wire-time model (see
``lane_step_us``). A positive result here is NECESSARY but NOT SUFFICIENT
evidence for a real multi-GPU/RDMA claim -- it only shows the controller
design is internally coherent and beats fixed baselines under this replay
model. Do not report this as an incast or tail-latency result.

Known confounds most likely to flip a positive result to negative on real
hardware or under closer scrutiny:
  1. Tile-size dependence of codec overhead: break-even ranged 0.9-59 Gbps
     across (rows, hidden) tile sizes in the 2026-07-20 audit. This script
     looks up the overhead row nearest to ``--assumed-lane-rows``; if actual
     per-step per-lane row counts differ a lot (they will, across scenarios),
     the overhead charged here can be systematically wrong in either
     direction. Always check ``metadata.json["codec_lookup"]`` for the actual
     (rows, hidden) matched.
  2. Single global controller: this script's "prev step load" is computed
     from the FULL scenario's realized traffic, i.e. the controller has
     pooled/global visibility into every rank's g-1 load. A real per-rank
     controller may have degraded or delayed visibility into other ranks'
     state; the saving reported here is thus an upper bound on what a
     genuinely decentralized controller could achieve.
  3. Hysteresis/dwell parameters are grid-fit on calibration docs/scenarios
     only. If the test pool's regime mix (hotspot/balanced ratio, arrival
     stagger) differs materially from calibration, frozen parameters may be
     mistuned in a way this script's calibration-only fitting cannot detect
     -- this is exactly the generalization risk that undid v3's threshold.
  4. Quality is NOT modeled here. The low state's downstream quality cost is
     whatever the project has already measured for uniform INT4 combine
     degradation (KL ~= 0.257 on OLMoE in the 2026-07-20 codec-break-even
     report) -- a fixed external reference, not something this script checks.
     A saving result here says nothing about whether that KL is acceptable;
     that judgement must be made jointly with the quality-debt mechanism.

Usage
-----
  python run_receiver_direct_benefit_controller.py \\
      --olmoe-routes outputs/paper_validation/olmoe_layer_budget_n32/calibration_routes.csv \\
      --llmjp-routes outputs/route_fidelity_p0_2026-07-18/p0b_calibration_llmjp/routes.csv \\
      --codec-csv outputs/receiver_codec_break_even_2026-07-20/codec_break_even.csv \\
      --output-dir outputs/receiver_direct_benefit_2026-07-20
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
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoConfig

from run_receiver_aware_v2_systematic import (
    build_scenario,
    load_model_config,
    placement_map,
)

LOW_BYTE_WEIGHT = 0.5  # INT4 vs the FP8 baseline's 1.0 (see fake_quant.BYTE_SIZES).
FULL_BYTE_WEIGHT = 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--olmoe-model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--olmoe-routes", required=True)
    p.add_argument("--llmjp-model", default="llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M")
    p.add_argument("--llmjp-routes", required=True)
    p.add_argument("--codec-csv", required=True, help="codec_break_even.csv from run_receiver_codec_break_even_gpu.py")
    p.add_argument("--assumed-lane-rows", type=int, default=32,
                    help="row-count tile used to look up a representative codec overhead per lane per step")
    p.add_argument("--ep-size", type=int, default=8)
    p.add_argument("--gpus-per-node", type=int, default=4)
    p.add_argument("--placement", default="contiguous")
    p.add_argument("--num-jobs", type=int, default=16)
    p.add_argument("--inter-node-gbps", type=float, default=200.0)
    p.add_argument("--calib-jobs", type=int, default=12)
    p.add_argument("--num-calib-scenario-seeds", type=int, default=10)
    p.add_argument("--num-test-scenario-seeds", type=int, default=30)
    p.add_argument("--num-bootstrap", type=int, default=2000)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


# --------------------------------------------------------------------------
# Codec overhead lookup (real GPU-measured, from codec_break_even.csv)
# --------------------------------------------------------------------------

def load_codec_overhead(codec_csv: str, hidden: int, assumed_rows: int) -> dict[str, object]:
    df = pd.read_csv(codec_csv)
    sub = df[(df["mode"] == "uniform_int4")].copy()
    if sub.empty:
        raise RuntimeError(f"no uniform_int4 rows found in {codec_csv}")
    sub["hidden_dist"] = (sub["hidden"] - hidden).abs()
    best_hidden = sub.loc[sub["hidden_dist"].idxmin(), "hidden"]
    sub = sub[sub["hidden"] == best_hidden].copy()
    sub["rows_dist"] = (sub["rows"] - assumed_rows).abs()
    row = sub.loc[sub["rows_dist"].idxmin()]
    return {
        "matched_hidden": int(row["hidden"]),
        "matched_rows": int(row["rows"]),
        "sender_pack_us": float(row["sender_pack_us"]),
        "receiver_unpack_us": float(row["receiver_unpack_us"]),
        "requested_hidden": hidden,
        "requested_rows": assumed_rows,
    }


# --------------------------------------------------------------------------
# Scenario pool construction (reuses v2's real-route-driven scenario builder)
# --------------------------------------------------------------------------

def make_scenario(routes: pd.DataFrame, doc_pool: list[int], rng: np.random.Generator,
                   num_jobs: int, origin_mode: str, ep_size: int, placement: str,
                   num_experts: int, gpus_per_node: int, max_stagger: int) -> pd.DataFrame:
    chosen_docs = rng.choice(doc_pool, size=min(num_jobs, len(doc_pool)),
                              replace=len(doc_pool) < num_jobs).tolist()
    arrivals = rng.integers(0, max_stagger + 1, size=len(chosen_docs))
    scn = build_scenario(routes, chosen_docs, arrivals, origin_mode, ep_size)
    scn["sender_rank"] = placement_map(scn["expert_id"].to_numpy(), num_experts, ep_size, placement)
    remote = scn[(scn["sender_rank"] // gpus_per_node) != (scn["receiver_rank"] // gpus_per_node)].copy()
    return remote


def prev_loads(remote: pd.DataFrame, g: int, ep_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Realized sender/receiver byte-weighted load at step g (all rows weight
    1.0; this is the ground truth used ONLY to build the g-1 signal fed to a
    later step's decision -- never used to score the same step it describes)."""
    rows = remote[remote["g"] == g]
    sender = np.zeros(ep_size)
    receiver = np.zeros(ep_size)
    if rows.empty:
        return sender, receiver
    s_counts = rows.groupby("sender_rank").size()
    r_counts = rows.groupby("receiver_rank").size()
    sender[s_counts.index.to_numpy()] = s_counts.to_numpy()
    receiver[r_counts.index.to_numpy()] = r_counts.to_numpy()
    return sender, receiver


# --------------------------------------------------------------------------
# Lane-level step latency with homogeneous one-shot low-bit weighting plus a
# real per-step codec overhead tax on the bottleneck rank.
# --------------------------------------------------------------------------

def lane_step_us(remote: pd.DataFrame, low_lane_by_step: dict[int, set[tuple[int, int]]],
                  bytes_to_us: float, pack_us: float, unpack_us: float) -> float:
    if remote.empty:
        return 0.0
    if any(low_lane_by_step.values()):
        decisions = pd.DataFrame(
            [(g, s, r) for g, pairs in low_lane_by_step.items() for (s, r) in pairs],
            columns=["g", "sender_rank", "receiver_rank"],
        )
        decisions["is_low"] = True
        merged = remote.merge(decisions, on=["g", "sender_rank", "receiver_rank"], how="left")
        merged["is_low"] = merged["is_low"].fillna(False)
    else:
        merged = remote.copy()
        merged["is_low"] = False
    merged["w"] = np.where(merged["is_low"], LOW_BYTE_WEIGHT, FULL_BYTE_WEIGHT)

    ingress = merged.groupby(["g", "receiver_rank"])["w"].sum()
    egress = merged.groupby(["g", "sender_rank"])["w"].sum()
    ingress_max = ingress.groupby(level="g").max()
    egress_max = egress.groupby(level="g").max()
    step_wire = pd.concat([ingress_max, egress_max], axis=1).max(axis=1) * bytes_to_us

    # Codec tax: any step where at least one active low lane exists pays one
    # representative pack+unpack overhead (conservative -- see confound #1
    # in the module docstring: this does not scale with lane count).
    low_steps = {g for g, pairs in low_lane_by_step.items() if pairs}
    tax = pd.Series(0.0, index=step_wire.index)
    tax.loc[tax.index.isin(low_steps)] = pack_us + unpack_us
    return float((step_wire + tax).sum())


def full_precision_us(remote: pd.DataFrame, bytes_to_us: float) -> float:
    return lane_step_us(remote, {}, bytes_to_us, 0.0, 0.0)


def uniform_low_us(remote: pd.DataFrame, bytes_to_us: float, pack_us: float, unpack_us: float) -> float:
    all_g = remote["g"].unique().tolist()
    all_lanes = set(zip(remote["sender_rank"].tolist(), remote["receiver_rank"].tolist()))
    low_by_step = {g: all_lanes for g in all_g}
    return lane_step_us(remote, low_by_step, bytes_to_us, pack_us, unpack_us)


# --------------------------------------------------------------------------
# Policies: each returns low_lane_by_step for a scenario, given ONLY causal
# (g-1 and earlier) information -- except calib_static, which uses zero
# online information at all (pure offline profile).
# --------------------------------------------------------------------------

def policy_causal_prev_step_no_hysteresis(remote: pd.DataFrame, ep_size: int, threshold: float) -> dict:
    """Ablation: same causal g-1 score as the main controller, but flips
    every step with NO hysteresis and NO dwell -- isolates what hysteresis
    buys over a naive per-step causal threshold."""
    out: dict[int, set[tuple[int, int]]] = {}
    steps = sorted(remote["g"].unique().tolist())
    for g in steps:
        s_prev, r_prev = prev_loads(remote, g - 1, ep_size)
        lanes_g = remote.loc[remote["g"] == g, ["sender_rank", "receiver_rank"]].drop_duplicates()
        low = set()
        for s, r in lanes_g.itertuples(index=False):
            if max(s_prev[s], r_prev[r]) >= threshold:
                low.add((s, r))
        out[g] = low
    return out


def policy_calib_static(remote: pd.DataFrame, static_profile: dict[tuple[int, int], float], threshold: float) -> dict:
    out: dict[int, set[tuple[int, int]]] = {}
    for g in sorted(remote["g"].unique().tolist()):
        lanes_g = remote.loc[remote["g"] == g, ["sender_rank", "receiver_rank"]].drop_duplicates()
        low = {(s, r) for s, r in lanes_g.itertuples(index=False)
               if static_profile.get((s, r), 0.0) >= threshold}
        out[g] = low
    return out


def policy_direct_benefit_credit(
    remote: pd.DataFrame, ep_size: int, alpha: float,
    threshold_high: float, threshold_low: float, dwell_min: int,
) -> dict:
    """Main proposal: EWMA credit per rank, two-sided hysteresis, minimum
    dwell time, decided independently per lane every step from g-1 info."""
    steps = sorted(remote["g"].unique().tolist())
    sender_credit = np.zeros(ep_size)
    receiver_credit = np.zeros(ep_size)
    state: dict[tuple[int, int], str] = {}
    dwell: dict[tuple[int, int], int] = {}
    out: dict[int, set[tuple[int, int]]] = {}
    for g in steps:
        s_prev, r_prev = prev_loads(remote, g - 1, ep_size)
        sender_credit = alpha * s_prev + (1 - alpha) * sender_credit
        receiver_credit = alpha * r_prev + (1 - alpha) * receiver_credit

        lanes_g = remote.loc[remote["g"] == g, ["sender_rank", "receiver_rank"]].drop_duplicates()
        low = set()
        for s, r in lanes_g.itertuples(index=False):
            lane = (int(s), int(r))
            cur = state.get(lane, "full")
            d = dwell.get(lane, dwell_min)
            score = max(sender_credit[s], receiver_credit[r])
            if cur == "full" and score >= threshold_high and d >= dwell_min:
                state[lane] = "low"
                dwell[lane] = 0
            elif cur == "low" and score <= threshold_low and d >= dwell_min:
                state[lane] = "full"
                dwell[lane] = 0
            else:
                dwell[lane] = d + 1
            if state.get(lane, "full") == "low":
                low.add(lane)
        out[g] = low
    return out


# --------------------------------------------------------------------------
# Calibration: build a static profile + grid-fit (alpha, dwell, thresholds)
# --------------------------------------------------------------------------

@dataclass
class CalibResult:
    static_profile: dict[tuple[int, int], float]
    best_params: dict[str, float]
    grid_rows: list[dict[str, object]] = field(default_factory=list)


def build_static_profile(remote_scenarios: list[pd.DataFrame]) -> dict[tuple[int, int], float]:
    frames = [r[["g", "sender_rank", "receiver_rank"]] for r in remote_scenarios if not r.empty]
    if not frames:
        return {}
    pooled = pd.concat(frames, ignore_index=True)
    n_steps = max(pooled["g"].nunique(), 1)
    counts = pooled.groupby(["sender_rank", "receiver_rank"]).size() / n_steps
    return {(int(s), int(r)): float(v) for (s, r), v in counts.items()}


def fit_controller(
    calib_remotes: list[pd.DataFrame], ep_size: int, bytes_to_us: float,
    pack_us: float, unpack_us: float,
) -> CalibResult:
    static_profile = build_static_profile(calib_remotes)
    all_receiver_loads = []
    for remote in calib_remotes:
        for g in remote["g"].unique():
            _, r_prev = prev_loads(remote, g, ep_size)
            all_receiver_loads.extend(r_prev[r_prev > 0].tolist())
    all_receiver_loads = np.array(all_receiver_loads) if all_receiver_loads else np.array([0.0])

    alphas = (0.3, 0.6)
    dwell_mins = (1, 3, 6)
    high_quantiles = (0.60, 0.75, 0.90)
    gap_ratios = (0.5, 0.7)

    grid_rows: list[dict[str, object]] = []
    best = None
    for alpha, dwell_min, hq, gap in itertools.product(alphas, dwell_mins, high_quantiles, gap_ratios):
        threshold_high = float(np.quantile(all_receiver_loads, hq))
        threshold_low = threshold_high * gap
        savings = []
        savings_over_causal = []
        for remote in calib_remotes:
            if remote.empty:
                continue
            base_us = full_precision_us(remote, bytes_to_us)
            low_by_step = policy_direct_benefit_credit(remote, ep_size, alpha, threshold_high, threshold_low, dwell_min)
            ctrl_us = lane_step_us(remote, low_by_step, bytes_to_us, pack_us, unpack_us)
            causal_low = policy_causal_prev_step_no_hysteresis(remote, ep_size, threshold_high)
            causal_us = lane_step_us(remote, causal_low, bytes_to_us, pack_us, unpack_us)
            if base_us > 0:
                savings.append(1.0 - ctrl_us / base_us)
                savings_over_causal.append((1.0 - ctrl_us / base_us) - (1.0 - causal_us / base_us))
        if not savings:
            continue
        mean_saving = float(np.mean(savings))
        lcb_saving = mean_saving - float(np.std(savings))
        mean_over_causal = float(np.mean(savings_over_causal))
        row = {
            "alpha": alpha, "dwell_min": dwell_min, "high_quantile": hq, "gap_ratio": gap,
            "threshold_high": threshold_high, "threshold_low": threshold_low,
            "mean_saving": mean_saving, "lcb_saving": lcb_saving,
            "mean_saving_over_causal": mean_over_causal,
        }
        grid_rows.append(row)
        if best is None or lcb_saving > best["lcb_saving"]:
            best = row
    if best is None:
        raise RuntimeError("calibration grid produced no valid configuration")
    return CalibResult(static_profile=static_profile, best_params=best, grid_rows=grid_rows)


# --------------------------------------------------------------------------
# Evaluation on held-out test scenarios: paired bootstrap across seeds.
# --------------------------------------------------------------------------

def evaluate(
    model_key: str, routes: pd.DataFrame, num_experts: int, hidden_size: int,
    ep_size: int, gpus_per_node: int, placement: str, num_jobs: int,
    inter_node_gbps: float, calib_jobs: int, num_calib_seeds: int, num_test_seeds: int,
    codec_info: dict, num_bootstrap: int,
) -> dict:
    bw_bytes_per_us = inter_node_gbps * 1e9 / 8 / 1e6
    bytes_to_us = hidden_size / bw_bytes_per_us
    pack_us = codec_info["sender_pack_us"]
    unpack_us = codec_info["receiver_unpack_us"]

    doc_pool = sorted(routes["sample_id"].unique().tolist())
    calib_docs = doc_pool[:calib_jobs]
    test_docs = doc_pool[calib_jobs:]
    if len(test_docs) < 4:
        raise RuntimeError(f"{model_key}: not enough documents left for test after calibration split")
    num_layers = int(routes["layer"].max()) + 1
    max_stagger = max(1, num_layers // 2)

    calib_remotes = []
    for seed in range(num_calib_seeds):
        rng = np.random.default_rng(9000 + seed)
        origin_mode = "hotspot" if seed % 2 == 0 else "balanced"
        calib_remotes.append(make_scenario(routes, calib_docs, rng, num_jobs, origin_mode, ep_size,
                                            placement, num_experts, gpus_per_node, max_stagger))

    calib = fit_controller(calib_remotes, ep_size, bytes_to_us, pack_us, unpack_us)
    p = calib.best_params
    static_threshold = float(np.quantile(list(calib.static_profile.values()) or [0.0], p["high_quantile"]))

    rows = []
    for seed in range(num_test_seeds):
        rng = np.random.default_rng(20260720_9000 + seed)
        origin_mode = "hotspot" if seed % 2 == 0 else "balanced"
        remote = make_scenario(routes, test_docs, rng, num_jobs, origin_mode, ep_size,
                                placement, num_experts, gpus_per_node, max_stagger)
        if remote.empty:
            continue
        base_us = full_precision_us(remote, bytes_to_us)
        if base_us <= 0:
            continue

        ctrl_low = policy_direct_benefit_credit(remote, ep_size, p["alpha"], p["threshold_high"], p["threshold_low"], p["dwell_min"])
        ctrl_us = lane_step_us(remote, ctrl_low, bytes_to_us, pack_us, unpack_us)

        causal_low = policy_causal_prev_step_no_hysteresis(remote, ep_size, p["threshold_high"])
        causal_us = lane_step_us(remote, causal_low, bytes_to_us, pack_us, unpack_us)

        static_low = policy_calib_static(remote, calib.static_profile, static_threshold)
        static_us = lane_step_us(remote, static_low, bytes_to_us, pack_us, unpack_us)

        uniform_us = uniform_low_us(remote, bytes_to_us, pack_us, unpack_us)

        rows.append({
            "model": model_key, "origin_mode": origin_mode, "scenario_seed": seed,
            "saving_controller": 1.0 - ctrl_us / base_us,
            "saving_causal_no_hysteresis": 1.0 - causal_us / base_us,
            "saving_calib_static": 1.0 - static_us / base_us,
            "saving_uniform_low": 1.0 - uniform_us / base_us,
        })

    df = pd.DataFrame(rows)
    rng = np.random.default_rng(4242)
    n = len(df)
    diffs = df["saving_controller"].to_numpy() - df["saving_causal_no_hysteresis"].to_numpy()
    boot = np.array([diffs[rng.integers(0, n, size=n)].mean() for _ in range(num_bootstrap)]) if n > 1 else np.array([diffs.mean() if n else 0.0])
    ci_low, ci_high = (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))) if n > 1 else (float(diffs.mean()), float(diffs.mean()))

    return {
        "model": model_key,
        "fitted_params": p,
        "codec_lookup": codec_info,
        "test_raw": df,
        "mean_saving_controller": float(df["saving_controller"].mean()) if n else 0.0,
        "mean_saving_causal_no_hysteresis": float(df["saving_causal_no_hysteresis"].mean()) if n else 0.0,
        "mean_saving_calib_static": float(df["saving_calib_static"].mean()) if n else 0.0,
        "mean_saving_uniform_low": float(df["saving_uniform_low"].mean()) if n else 0.0,
        "controller_minus_best_causal_ablation_mean": float(diffs.mean()) if n else 0.0,
        "controller_minus_best_causal_ablation_ci_low": ci_low,
        "controller_minus_best_causal_ablation_ci_high": ci_high,
        "go_no_go": bool(ci_low > 0.0),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    models = {
        "olmoe": (args.olmoe_model, args.olmoe_routes),
        "llmjp": (args.llmjp_model, args.llmjp_routes),
    }
    all_results = {}
    all_raw = []
    for model_key, (model_name, route_path) in models.items():
        num_experts, _top_k = load_model_config(model_name)
        hidden_size = int(AutoConfig.from_pretrained(model_name, local_files_only=True).hidden_size)
        codec_info = load_codec_overhead(args.codec_csv, hidden_size, args.assumed_lane_rows)
        routes = pd.read_csv(route_path)
        result = evaluate(
            model_key, routes, num_experts, hidden_size, args.ep_size, args.gpus_per_node,
            args.placement, args.num_jobs, args.inter_node_gbps, args.calib_jobs,
            args.num_calib_scenario_seeds, args.num_test_scenario_seeds, codec_info, args.num_bootstrap,
        )
        raw = result.pop("test_raw")
        all_raw.append(raw)
        all_results[model_key] = result
        print(f"[{model_key}] controller={result['mean_saving_controller']:.4f} "
              f"causal_no_hysteresis={result['mean_saving_causal_no_hysteresis']:.4f} "
              f"calib_static={result['mean_saving_calib_static']:.4f} "
              f"uniform_low={result['mean_saving_uniform_low']:.4f} "
              f"controller-causal_CI=[{result['controller_minus_best_causal_ablation_ci_low']:.4f},"
              f"{result['controller_minus_best_causal_ablation_ci_high']:.4f}] "
              f"go_no_go={result['go_no_go']}")

    pd.concat(all_raw, ignore_index=True).to_csv(out / "raw_test_scenarios.csv", index=False)
    metadata = {
        "results": all_results,
        "evidence_boundary": (
            "Analytic byte-count + real Triton codec pack/unpack overhead replay. "
            "No real NCCL/RDMA, no queueing, no incast. Single global controller "
            "with pooled g-1 visibility (upper bound on decentralized deployment). "
            "Threshold/hysteresis fit on calibration docs only; quality cost of the "
            "low state is NOT modeled here (see uniform INT4 KL from prior rounds)."
        ),
        "confounds": [
            "codec overhead lookup uses a single (rows, hidden) tile per model; "
            "actual per-lane per-step row counts vary and are not re-measured",
            "global vs per-rank visibility gap not modeled",
            "calibration-test regime-mix mismatch not stress-tested here",
            "downstream quality of the low state is an external fixed reference, not measured in this script",
        ],
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    lines = ["# Receiver-Aware Direct-Benefit Controller (homogeneous one-shot lane + receiver/sender credit)", ""]
    for model_key, result in all_results.items():
        lines.append(f"## {model_key}")
        lines.append(f"- fitted params: {result['fitted_params']}")
        lines.append(f"- codec lookup: {result['codec_lookup']}")
        lines.append(f"- mean saving: controller={result['mean_saving_controller']:.4f}, "
                      f"causal_no_hysteresis={result['mean_saving_causal_no_hysteresis']:.4f}, "
                      f"calib_static={result['mean_saving_calib_static']:.4f}, "
                      f"uniform_low={result['mean_saving_uniform_low']:.4f}")
        lines.append(f"- controller - causal_no_hysteresis: mean={result['controller_minus_best_causal_ablation_mean']:.4f}, "
                      f"95% CI=[{result['controller_minus_best_causal_ablation_ci_low']:.4f}, "
                      f"{result['controller_minus_best_causal_ablation_ci_high']:.4f}]")
        lines.append(f"- GO/NO-GO (CI excludes 0 and controller beats hysteresis-free causal baseline): "
                      f"{'GO' if result['go_no_go'] else 'NO-GO'}")
        lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Follow-up to ``analyze_receiver_aware_bandit_offline_replay.py``: rebuilds
the SAME 4 low-bit policies (controller / causal_no_hysteresis /
calib_static / uniform_low), plus the required no-action uniform_full baseline,
on the SAME real routing data and ALREADY-FITTED parameters
used by ``run_receiver_direct_benefit_controller.py`` (frozen thresholds
copied verbatim from its saved ``metadata.json`` -- not refit here, so this
cannot leak), but this time tracks a SECOND quantity per policy per
scenario: ``low_frac`` -- the fraction of remote (token, rank) traffic that
was actually sent through a low-bit lane at least once. This lets us build a
reward that is no longer quality-blind:

    reward(policy, scenario; lambda) = saving(policy, scenario)
                                        - lambda * low_frac(policy, scenario)
                                          * KL_uniform_low(model)

``KL_uniform_low`` is this project's own REAL measured quality number for
100%-low-precision combine (GPU 第三轮 homogeneous-INT4 result: OLMoE
mean token KL = 0.257494, LLM-jp = 0.196984) -- not invented. ``low_frac`` is
used as a LINEAR proxy for "how much of the realized KL this policy would be
responsible for" (policy sends low_frac fraction of traffic at low
precision, uniform_low sends 100%): this is an explicit, flagged
approximation (real KL is very unlikely to be exactly linear in traffic
fraction), not a measured per-policy KL -- no GPU time was spent measuring
quality for anything except the uniform_low endpoint, which IS real.

Because the correct lambda (nats of KL per unit of wire-time-saving-fraction)
is not something this project has calibrated, this script SWEEPS lambda
across a wide range and reports, for each lambda: (a) which arm has the
highest mean reward per (model, origin_mode) cell, and (b) whether the
winning arm becomes origin_mode-DEPENDENT (i.e. whether a context/regime-
aware or bandit controller could ever have something non-trivial to learn)
once quality is priced in, rather than being uniform_low everywhere as it
was in the quality-blind version of this experiment.

This is diagnostic, not decisive: the exact crossover lambda depends on an
assumption (linearity) this project has not verified, so the correct
reading is "does pricing quality in at ANY plausible lambda change the
qualitative conclusion", not "here is the true optimal lambda".
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

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/Users/leandrozhao/Desktop/毕设论文资料/experiments/idea_a_mac/outputs")
ROUTE_PATHS = {
    "olmoe": (BASE / "receiver_aware_v2" / "olmoe_routes.csv", 64, 2048),
    "llmjp": (BASE / "receiver_aware_v2" / "llmjp_routes.csv", 32, 512),
}
EP_SIZE = 8
GPUS_PER_NODE = 4
NUM_JOBS = 16
CALIB_JOBS = 12
NUM_CALIB_SEEDS = 10
NUM_TEST_SEEDS = 30
INTER_NODE_GBPS = 200.0
LOW_BYTE_WEIGHT = 0.5
FULL_BYTE_WEIGHT = 1.0

# Frozen from run_receiver_direct_benefit_controller's own saved metadata.json
# (outputs/receiver_direct_benefit_2026-07-20_full/metadata.json) -- copied
# verbatim, NOT refit here.
FITTED_PARAMS = {
    "olmoe": {"alpha": 0.6, "dwell_min": 1, "high_quantile": 0.6, "gap_ratio": 0.5,
              "threshold_high": 1110.0, "threshold_low": 555.0,
              "pack_us": 26.12175941467285, "unpack_us": 25.37951946258545},
    "llmjp": {"alpha": 0.6, "dwell_min": 1, "high_quantile": 0.6, "gap_ratio": 0.5,
              "threshold_high": 2178.6, "threshold_low": 1089.3,
              "pack_us": 17.097280025482178, "unpack_us": 16.322879791259766},
}
# Real measured quality cost of 100%-low-precision combine (GPU 第三轮
# homogeneous-INT4 section of GPU第三轮有效性实验结果_ReceiverProgressive与Codec_2026-07-20.md).
KL_UNIFORM_LOW = {"olmoe": 0.257494, "llmjp": 0.196984}

LAMBDA_GRID = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
ARM_NAMES = ["uniform_full", "controller", "causal_no_hysteresis", "calib_static", "uniform_low"]


# ---------------------------------------------------------------------------
# Copied verbatim (pure numpy/pandas) from run_receiver_aware_v2_systematic.py
# / run_receiver_direct_benefit_controller.py.
# ---------------------------------------------------------------------------

def placement_map(expert_id: np.ndarray, num_experts: int, ep_size: int, mapping: str) -> np.ndarray:
    if mapping == "contiguous":
        return np.minimum(expert_id * ep_size // num_experts, ep_size - 1)
    if mapping == "round_robin":
        return expert_id % ep_size
    raise ValueError(mapping)


def build_scenario(routes: pd.DataFrame, doc_ids: list[int], arrivals: np.ndarray, origin_mode: str, ep_size: int) -> pd.DataFrame:
    frames = []
    for job_id, (doc_id, arrival) in enumerate(zip(doc_ids, arrivals)):
        rows = routes[routes["sample_id"] == doc_id].copy()
        rows["job_id"] = job_id
        rows["g"] = rows["layer"].astype(int) + int(arrival)
        if origin_mode == "balanced":
            receiver = job_id % ep_size
        elif origin_mode == "hotspot":
            hotspot_jobs = max(1, math.ceil(len(doc_ids) * 0.5))
            receiver = 0 if job_id < hotspot_jobs else 1 + ((job_id - hotspot_jobs) % max(ep_size - 1, 1))
        else:
            raise ValueError(origin_mode)
        rows["receiver_rank"] = receiver
        frames.append(rows[["job_id", "g", "rank", "expert_id", "receiver_rank"]])
    return pd.concat(frames, ignore_index=True)


def make_scenario(routes: pd.DataFrame, doc_pool: list[int], rng: np.random.Generator,
                   num_jobs: int, origin_mode: str, ep_size: int, num_experts: int,
                   gpus_per_node: int, max_stagger: int) -> pd.DataFrame:
    chosen_docs = rng.choice(doc_pool, size=min(num_jobs, len(doc_pool)),
                              replace=len(doc_pool) < num_jobs).tolist()
    arrivals = rng.integers(0, max_stagger + 1, size=len(chosen_docs))
    scn = build_scenario(routes, chosen_docs, arrivals, origin_mode, ep_size)
    scn["sender_rank"] = placement_map(scn["expert_id"].to_numpy(), num_experts, ep_size, "contiguous")
    remote = scn[(scn["sender_rank"] // gpus_per_node) != (scn["receiver_rank"] // gpus_per_node)].copy()
    return remote


def prev_loads(remote: pd.DataFrame, g: int, ep_size: int) -> tuple[np.ndarray, np.ndarray]:
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


def lane_step_us_and_low_frac(
    remote: pd.DataFrame, low_lane_by_step: dict[int, set[tuple[int, int]]],
    bytes_to_us: float, pack_us: float, unpack_us: float,
) -> tuple[float, float]:
    if remote.empty:
        return 0.0, 0.0
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

    low_steps = {g for g, pairs in low_lane_by_step.items() if pairs}
    tax = pd.Series(0.0, index=step_wire.index)
    tax.loc[tax.index.isin(low_steps)] = pack_us + unpack_us

    low_frac = float(merged["is_low"].mean())
    return float((step_wire + tax).sum()), low_frac


def policy_causal_prev_step_no_hysteresis(remote: pd.DataFrame, ep_size: int, threshold: float) -> dict:
    out: dict[int, set[tuple[int, int]]] = {}
    for g in sorted(remote["g"].unique().tolist()):
        s_prev, r_prev = prev_loads(remote, g - 1, ep_size)
        lanes_g = remote.loc[remote["g"] == g, ["sender_rank", "receiver_rank"]].drop_duplicates()
        low = {(s, r) for s, r in lanes_g.itertuples(index=False) if max(s_prev[s], r_prev[r]) >= threshold}
        out[g] = low
    return out


def policy_calib_static(remote: pd.DataFrame, static_profile: dict[tuple[int, int], float], threshold: float) -> dict:
    out: dict[int, set[tuple[int, int]]] = {}
    for g in sorted(remote["g"].unique().tolist()):
        lanes_g = remote.loc[remote["g"] == g, ["sender_rank", "receiver_rank"]].drop_duplicates()
        low = {(s, r) for s, r in lanes_g.itertuples(index=False) if static_profile.get((s, r), 0.0) >= threshold}
        out[g] = low
    return out


def policy_direct_benefit_credit(remote: pd.DataFrame, ep_size: int, alpha: float,
                                  threshold_high: float, threshold_low: float, dwell_min: int) -> dict:
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
                state[lane] = "low"; dwell[lane] = 0
            elif cur == "low" and score <= threshold_low and d >= dwell_min:
                state[lane] = "full"; dwell[lane] = 0
            else:
                dwell[lane] = d + 1
            if state.get(lane, "full") == "low":
                low.add(lane)
        out[g] = low
    return out


def build_static_profile(remote_scenarios: list[pd.DataFrame]) -> dict[tuple[int, int], float]:
    frames = [r[["g", "sender_rank", "receiver_rank"]] for r in remote_scenarios if not r.empty]
    if not frames:
        return {}
    pooled = pd.concat(frames, ignore_index=True)
    n_steps = max(pooled["g"].nunique(), 1)
    counts = pooled.groupby(["sender_rank", "receiver_rank"]).size() / n_steps
    return {(int(s), int(r)): float(v) for (s, r), v in counts.items()}


# ---------------------------------------------------------------------------
# Main: rebuild raw saving+low_frac table, then apply quality-constrained
# reward at each lambda and re-run the UCB1/regime-free bandit diagnostic.
# ---------------------------------------------------------------------------

def rebuild_raw_table(model_key: str) -> pd.DataFrame:
    route_path, num_experts, hidden_size = ROUTE_PATHS[model_key]
    routes = pd.read_csv(route_path)
    p = FITTED_PARAMS[model_key]
    bw_bytes_per_us = INTER_NODE_GBPS * 1e9 / 8 / 1e6
    bytes_to_us = hidden_size / bw_bytes_per_us
    num_layers = int(routes["layer"].max()) + 1
    max_stagger = max(1, num_layers // 2)
    doc_pool = sorted(routes["sample_id"].unique().tolist())
    calib_docs, test_docs = doc_pool[:CALIB_JOBS], doc_pool[CALIB_JOBS:]

    calib_remotes = []
    for seed in range(NUM_CALIB_SEEDS):
        rng = np.random.default_rng(9000 + seed)
        origin_mode = "hotspot" if seed % 2 == 0 else "balanced"
        calib_remotes.append(make_scenario(routes, calib_docs, rng, NUM_JOBS, origin_mode,
                                            EP_SIZE, num_experts, GPUS_PER_NODE, max_stagger))
    static_profile = build_static_profile(calib_remotes)
    static_threshold = float(np.quantile(list(static_profile.values()) or [0.0], p["high_quantile"]))

    rows = []
    for seed in range(NUM_TEST_SEEDS):
        rng = np.random.default_rng(20260720_9000 + seed)
        origin_mode = "hotspot" if seed % 2 == 0 else "balanced"
        remote = make_scenario(routes, test_docs, rng, NUM_JOBS, origin_mode,
                                EP_SIZE, num_experts, GPUS_PER_NODE, max_stagger)
        if remote.empty:
            continue
        base_us, _ = lane_step_us_and_low_frac(remote, {}, bytes_to_us, 0.0, 0.0)
        if base_us <= 0:
            continue

        all_g = remote["g"].unique().tolist()
        all_lanes = set(zip(remote["sender_rank"].tolist(), remote["receiver_rank"].tolist()))
        policies = {
            "uniform_full": {},
            "controller": policy_direct_benefit_credit(remote, EP_SIZE, p["alpha"], p["threshold_high"], p["threshold_low"], p["dwell_min"]),
            "causal_no_hysteresis": policy_causal_prev_step_no_hysteresis(remote, EP_SIZE, p["threshold_high"]),
            "calib_static": policy_calib_static(remote, static_profile, static_threshold),
            "uniform_low": {g: all_lanes for g in all_g},
        }
        row = {"model": model_key, "origin_mode": origin_mode, "scenario_seed": seed}
        for name, low_by_step in policies.items():
            us, low_frac = lane_step_us_and_low_frac(remote, low_by_step, bytes_to_us, p["pack_us"], p["unpack_us"])
            row[f"saving_{name}"] = 1.0 - us / base_us
            row[f"low_frac_{name}"] = low_frac
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    out_dir = BASE / "receiver_aware_quality_constrained_reward_2026-07-20"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_frames = []
    for model_key in ROUTE_PATHS:
        raw_frames.append(rebuild_raw_table(model_key))
    raw = pd.concat(raw_frames, ignore_index=True)
    raw.to_csv(out_dir / "raw_saving_and_low_frac.csv", index=False)

    print("=== Sanity check: recomputed saving vs original receiver_direct_benefit_2026-07-20_full ===")
    print(raw.groupby("model")[[f"saving_{a}" for a in ARM_NAMES]].mean().to_string())
    print()

    winner_rows = []
    for model_key in ROUTE_PATHS:
        kl = KL_UNIFORM_LOW[model_key]
        sub_model = raw[raw.model == model_key]
        for lam in LAMBDA_GRID:
            for origin_mode in sorted(sub_model["origin_mode"].unique()) + ["mixed"]:
                cell = sub_model if origin_mode == "mixed" else sub_model[sub_model.origin_mode == origin_mode]
                means = {}
                for arm in ARM_NAMES:
                    reward = cell[f"saving_{arm}"] - lam * cell[f"low_frac_{arm}"] * kl
                    means[arm] = float(reward.mean())
                winner = max(means, key=means.get)
                winner_rows.append({
                    "model": model_key, "lambda": lam, "origin_mode": origin_mode,
                    "winner": winner, **{f"reward_{a}": v for a, v in means.items()},
                })
    winners = pd.DataFrame(winner_rows)
    winners.to_csv(out_dir / "quality_constrained_winners.csv", index=False)

    print("=== Winning arm per (model, lambda, origin_mode) ===")
    piv = winners.pivot_table(index=["model", "lambda"], columns="origin_mode", values="winner", aggfunc="first")
    print(piv.to_string())

    print("\n=== Does the winner ever differ ACROSS origin_mode at the same lambda "
          "(i.e. would a regime-aware or bandit controller have something non-trivial to learn)? ===")
    for model_key in ROUTE_PATHS:
        for lam in LAMBDA_GRID:
            sub = winners[(winners.model == model_key) & (winners.lambda_ if hasattr(winners, "lambda_") else winners["lambda"] == lam)]
            per_regime = sub[sub.origin_mode != "mixed"]
            distinct = per_regime["winner"].nunique()
            if distinct > 1:
                print(f"  {model_key} lambda={lam}: winners differ across regimes -> "
                      f"{dict(zip(per_regime.origin_mode, per_regime.winner))}")

    (out_dir / "metadata.json").write_text(json.dumps({
        "lambda_grid": LAMBDA_GRID, "kl_uniform_low": KL_UNIFORM_LOW,
        "fitted_params_source": "outputs/receiver_direct_benefit_2026-07-20_full/metadata.json (copied verbatim)",
        "reward_formula": "saving(policy) - lambda * low_frac(policy) * KL_uniform_low(model)",
        "caveat": "low_frac is a LINEAR proxy for policy-attributable quality damage; "
                  "only the uniform_low endpoint (100% low_frac -> KL_uniform_low) is a real measured quality number; "
                  "intermediate points are an explicit, unverified linearity assumption",
    }, indent=2), encoding="utf-8")
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Zero-new-GPU-time deepening of Receiver-aware: offline bandit-regret
replay over the ALREADY-COLLECTED 4-arm reward table in
``outputs/receiver_direct_benefit_2026-07-20_full/raw_test_scenarios.csv``
(produced by ``run_receiver_direct_benefit_controller.py`` on real routing
traces + real Triton codec overhead).

What this tests
----------------------------------------------------------------------------
That CSV already records, for EVERY test scenario, the realized saving of
ALL FOUR candidate policies (controller, causal_no_hysteresis, calib_static,
uniform_low) -- i.e. it is a full-information bandit dataset (every arm's
counterfactual reward is known for every round), which makes OFFLINE bandit
simulation exact (no importance-weighting/off-policy-correction needed,
unlike a typical only-logged-arm bandit log).

This directly operationalizes two of the reconstruction directions raised
in the audit report:
  (b) "regret-minimizing / contextual bandit controller instead of a regime
      classifier"
  (d) "a robust policy that does NOT need to classify regime accurately"

Design: run UCB1 over the 4 arms two ways per model:
  1. REGIME-AWARE: one independent UCB1 instance per origin_mode (balanced,
     hotspot), i.e. bandit gets to see which regime each round is in --
     this is the closest bandit analogue of "classify then switch".
  2. REGIME-FREE: ONE UCB1 instance sees BOTH regimes mixed together in
     random round order, with NO regime label at all -- this is the direct
     test of reconstruction direction (d): can a policy that never
     classifies regime still find a good arm fast, purely from reward
     feedback?

Both are compared against: always-controller (current deployed choice),
always-uniform_low (cheapest fixed baseline), random arm selection, and the
per-scenario oracle (upper bound, not achievable without foresight).
Cumulative regret vs. oracle is reported, averaged over many random round
orderings (since with only 15-30 rounds per cell, a single ordering is
noisy) -- this IS the bootstrap-equivalent discipline for a bandit regret
estimate.

Known confound this experiment cannot rule out
----------------------------------------------------------------------------
This is a REPLAY over a FIXED, already-recorded scenario pool (bandwidth
replay only, no real RDMA/queueing -- same evidence boundary as the
underlying controller experiment), and with only 15 scenarios per regime the
regret curves have not remotely converged; this experiment is diagnostic of
whether the DIRECTION is promising, not a publishable regret bound.
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

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/Users/leandrozhao/Desktop/毕设论文资料/experiments/idea_a_mac/outputs")
SRC = BASE / "receiver_direct_benefit_2026-07-20_full" / "raw_test_scenarios.csv"
ARMS = ["saving_controller", "saving_causal_no_hysteresis", "saving_calib_static", "saving_uniform_low"]
ARM_NAMES = ["controller", "causal_no_hysteresis", "calib_static", "uniform_low"]
N_ORDER_REPLICATES = 500
SEED = 20260720


def ucb1_regret_curve(rewards: np.ndarray, order: np.ndarray) -> np.ndarray:
    """rewards: (n_rounds, n_arms) full-information table. order: permutation
    of round indices defining the sequence bandit sees them in. Returns
    cumulative regret vs. oracle (best-per-round) over rounds, in that order."""
    n_rounds, n_arms = rewards.shape
    counts = np.zeros(n_arms)
    means = np.zeros(n_arms)
    cum_regret = np.zeros(n_rounds)
    running_regret = 0.0
    oracle = rewards.max(axis=1)
    for t, idx in enumerate(order):
        if t < n_arms:
            arm = t  # forced initial exploration of each arm once
        else:
            ucb = means + np.sqrt(2.0 * np.log(t + 1) / np.maximum(counts, 1))
            arm = int(np.argmax(ucb))
        reward = rewards[idx, arm]
        counts[arm] += 1
        means[arm] += (reward - means[arm]) / counts[arm]
        running_regret += float(oracle[idx] - reward)
        cum_regret[t] = running_regret
    return cum_regret


def simulate_cell(rewards: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    n_rounds = rewards.shape[0]
    oracle_sum = rewards.max(axis=1).sum()

    ucb_finals = []
    for _ in range(N_ORDER_REPLICATES):
        order = rng.permutation(n_rounds)
        curve = ucb1_regret_curve(rewards, order)
        ucb_finals.append(curve[-1])
    ucb_mean_regret = float(np.mean(ucb_finals))

    random_regrets = []
    for _ in range(N_ORDER_REPLICATES):
        choices = rng.integers(0, rewards.shape[1], size=n_rounds)
        chosen_reward = rewards[np.arange(n_rounds), choices].sum()
        random_regrets.append(oracle_sum - chosen_reward)
    random_mean_regret = float(np.mean(random_regrets))

    fixed_regrets = {name: float(oracle_sum - rewards[:, i].sum()) for i, name in enumerate(ARM_NAMES)}
    best_fixed = min(fixed_regrets, key=fixed_regrets.get)

    return {
        "n_rounds": n_rounds,
        "oracle_sum": float(oracle_sum),
        "ucb1_mean_cum_regret": ucb_mean_regret,
        "random_mean_cum_regret": random_mean_regret,
        "best_fixed_arm": best_fixed,
        "best_fixed_cum_regret": fixed_regrets[best_fixed],
        **{f"fixed_{name}_cum_regret": v for name, v in fixed_regrets.items()},
        "ucb1_beats_best_fixed": bool(ucb_mean_regret < fixed_regrets[best_fixed]),
        "ucb1_beats_random": bool(ucb_mean_regret < random_mean_regret),
    }


def main() -> None:
    df = pd.read_csv(SRC)
    rng = np.random.default_rng(SEED)
    rows = []

    for model in sorted(df["model"].unique()):
        sub_model = df[df.model == model]

        # (1) REGIME-AWARE: one bandit per origin_mode.
        for origin_mode in sorted(sub_model["origin_mode"].unique()):
            sub = sub_model[sub_model.origin_mode == origin_mode]
            rewards = sub[ARMS].to_numpy()
            result = simulate_cell(rewards, rng)
            result.update({"model": model, "setting": "regime_aware", "origin_mode": origin_mode})
            rows.append(result)

        # (2) REGIME-FREE: single bandit, both regimes mixed, no label used.
        rewards_mixed = sub_model[ARMS].to_numpy()
        result = simulate_cell(rewards_mixed, rng)
        result.update({"model": model, "setting": "regime_free_mixed", "origin_mode": "mixed"})
        rows.append(result)

    out_df = pd.DataFrame(rows)
    out_dir = BASE / "receiver_aware_bandit_offline_replay_2026-07-20"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dir / "results.csv", index=False)
    (out_dir / "metadata.json").write_text(json.dumps({
        "arms": ARM_NAMES, "n_order_replicates": N_ORDER_REPLICATES,
        "evidence_boundary": "full-information offline bandit replay over already-collected "
                              "bandwidth+codec-overhead scenario rewards; no real RDMA/queueing; "
                              "only 15-30 rounds per cell, regret curves far from converged",
    }, indent=2), encoding="utf-8")

    print(out_df[[
        "model", "setting", "origin_mode", "n_rounds", "ucb1_mean_cum_regret",
        "best_fixed_arm", "best_fixed_cum_regret", "random_mean_cum_regret",
        "ucb1_beats_best_fixed", "ucb1_beats_random",
    ]].to_string(index=False))
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()

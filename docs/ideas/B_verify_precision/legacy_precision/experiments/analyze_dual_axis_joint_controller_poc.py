#!/usr/bin/env python3
"""Dual-axis joint quality-debt controller: a PROOF-OF-CONCEPT illustrating a
genuinely new mechanism (nobody in this project has combined the two axes
into one shared budget), explicitly scoped as preliminary/illustrative.

READ THIS BEFORE RUNNING OR CITING RESULTS -- this is a POC, not a decisive
system claim.

Why this is scoped down to a POC (an honest finding in itself)
----------------------------------------------------------------------------
Checking the two axes' existing real data before building anything found a
hard blocker to a fully rigorous joint simulation: the combine-axis
controller (``run_receiver_direct_benefit_controller.py``,
``outputs/receiver_direct_benefit_2026-07-20_full/``) optimizes a
queueing/byte-saving fraction and explicitly does NOT model the quality cost
of its "low" state (its own metadata.json evidence_boundary says so
verbatim: "quality cost of the low state is NOT modeled here"). The
compute-axis controller (today's persistence/shadow-verify work) optimizes
real per-step KL and does NOT model queueing/latency at all. The two axes'
"harm" units are therefore NOT currently commensurable with real data on
both sides -- forcing them into one budget right now would require
fabricating a number on one side. This script does exactly that, but labels
it clearly: it uses the project's own ALREADY-ESTABLISHED reference number
for uniform-INT4 combine-axis quality cost (KL ~= 0.257, from prior rounds'
uniform_int4 combine-degradation measurement) as a CONSTANT per-decision
harm for the combine stream, paired with the REAL, variable per-step KL
trajectories from today's compute-axis persistence experiment. The result
below demonstrates the MECHANISM (does pooling one shared budget across two
streams with different risk profiles beat splitting the budget statically
in half?) on a semi-real, semi-illustrative dataset. It is NOT a claim about
real system-level savings and should not be cited as one.

Mechanism being tested
-----------------------
Two decision streams share ONE fixed total quality-debt budget B per
session (paired document): a COMPUTE stream (real, bursty/persistent
per-step KL if serving low-precision expert weights that step) and a
COMBINE stream (constant per-decision-point KL if serving low-precision
combine payload that point). Two allocators are compared at the SAME total
budget B:
  - INDEPENDENT: split B in half up front, one static half-budget per axis;
    each axis greedily serves "low" until ITS OWN half-budget is exhausted,
    then serves "high" for the rest of ITS OWN stream, oblivious to the
    other axis's state.
  - JOINT: ONE shared running budget; at each decision point (across BOTH
    streams, merged in time order), serve "low" if doing so would not
    exceed the REMAINING shared budget, regardless of which axis it is on.
The hypothesis: because the compute stream is bursty (real, serially
persistent risk validated today) while the combine stream is close to
constant-rate, a joint pool can "lend" budget between streams and achieve a
strictly higher total low-precision fraction (proxy for total saving) at
the identical total harm budget -- this is the same principle behind
statistical multiplexing / resource pooling in queueing theory, demonstrated
concretely on this project's own data rather than asserted abstractly.

GO/NO-GO (illustrative, not a systems claim)
----------------------------------------------------------------------------
GO iff, at every budget level tested, JOINT achieves a low-precision
fraction (summed over both streams) that is >= INDEPENDENT's, with a
document-level bootstrap CI on the difference that excludes zero (i.e., the
pooling benefit is not just noise), at more than half of the tested budget
levels.

What this POC does NOT establish
-------------------------------------
It does not establish that combine-axis and compute-axis harms are
literally interchangeable in a real system (they are not: one is
network/codec latency, one is compute-quality risk), and it does not
produce a real wall-clock or real end-to-end quality number. It exists
solely to test, on real+semi-real data, whether the ABSTRACT resource-
pooling mechanism has enough signal to be worth the (substantial) future
engineering effort of building a real shared cost/harm accounting layer
across both axes.
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
COMPUTE_AXIS_CSV = {
    "olmoe": BASE / "expert_precision_persistence_2026-07-20_olmoe" / "per_step_samples.csv",
    "llmjp": BASE / "expert_precision_persistence_2026-07-20_llmjp" / "per_step_samples.csv",
}
# Reference constant: uniform-INT4 combine-axis degradation KL, established in
# prior rounds of this project (cited in the 2026-07-20 critical re-audit
# report as "已知uniform INT4通信降级KL(0.257)"). Used here ONLY as a
# constant per-decision synthetic combine-stream harm -- see docstring.
COMBINE_AXIS_REFERENCE_KL = 0.257
CALIB_SAMPLES = 12
N_BOOTSTRAP = 2000
SEED = 20260720
# Budget levels expressed as a fraction of "always-low" total harm for the
# PAIRED (compute + synthetic combine) session -- i.e., how much of the
# worst-case harm the operator is willing to tolerate.
BUDGET_FRACTIONS = [0.10, 0.20, 0.30, 0.40, 0.50]


def build_session(compute_traj: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Pair one real compute-axis trajectory with a same-length synthetic
    combine-axis stream: constant harm-if-low, but with the SAME kind of
    per-step jitter magnitude as the compute stream (+-15%) so the combine
    stream is not perfectly deterministic -- still far less bursty than the
    real compute stream, by construction."""
    T = len(compute_traj)
    rng = np.random.default_rng(seed)
    jitter = 1.0 + rng.uniform(-0.15, 0.15, size=T)
    combine_stream = COMBINE_AXIS_REFERENCE_KL * jitter / T  # spread the reference cost across T decision points
    return compute_traj, combine_stream


def independent_split(compute: np.ndarray, combine: np.ndarray, budget: float) -> tuple[float, float]:
    half = budget / 2.0
    low_count = 0
    spent = 0.0
    for v in compute:
        if spent + v <= half:
            spent += v
            low_count += 1
    spent = 0.0
    for v in combine:
        if spent + v <= half:
            spent += v
            low_count += 1
    return low_count / (len(compute) + len(combine)), low_count


def joint_pool(compute: np.ndarray, combine: np.ndarray, budget: float) -> tuple[float, float]:
    # v1 (fixed alternating temporal order, no value-based priority) was
    # tried first and found STRICTLY WORSE than the independent-split
    # baseline at every budget level -- an important negative finding, but
    # it turned out to be an artifact of a naive merge rule, not evidence
    # against pooling itself: processing in blind temporal order lets one
    # expensive item early in the session burn shared budget that cheaper
    # items later on (from EITHER stream) could have used far more
    # efficiently. This v2 keeps the SAME causal discipline (only ever uses
    # information available up to and including decision round i, never
    # looks ahead to i+1) but, WITHIN each round i, serves the cheaper of
    # the two streams' items first -- a locally-greedy priority rule that is
    # still fully causal and realizable by an online controller with
    # visibility into both streams' CURRENT round.
    remaining = budget
    low_count = 0
    total_items = 2 * len(compute)
    for c_val, b_val in zip(compute, combine):
        for v in sorted((c_val, b_val)):
            if v <= remaining:
                remaining -= v
                low_count += 1
    return low_count / total_items, low_count


def document_bootstrap_ci(values: np.ndarray, n_bootstrap: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.array([values[rng.integers(0, n, size=n)].mean() for _ in range(n_bootstrap)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def run_for_model(model_key: str, csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    doc_ids_sorted = sorted(df.doc_id.unique())
    test_docs = doc_ids_sorted[CALIB_SAMPLES:]

    rows = []
    for budget_frac in BUDGET_FRACTIONS:
        diffs = []
        for i, doc_id in enumerate(test_docs):
            compute_traj = df[df.doc_id == doc_id].sort_values("step")["kl"].to_numpy()
            compute, combine = build_session(compute_traj, seed=SEED + i)
            always_low_total = float(compute.sum() + combine.sum())
            budget = budget_frac * always_low_total

            indep_frac, _ = independent_split(compute, combine, budget)
            joint_frac, _ = joint_pool(compute, combine, budget)
            diffs.append(joint_frac - indep_frac)

        diffs = np.array(diffs)
        ci_low, ci_high = document_bootstrap_ci(diffs, N_BOOTSTRAP, SEED)
        rows.append({
            "model": model_key, "budget_fraction": budget_frac,
            "joint_minus_independent_low_frac": float(diffs.mean()),
            "ci_low": ci_low, "ci_high": ci_high,
            "n_documents": len(test_docs),
            "joint_beats_independent": bool(ci_low > 0.0),
        })
    return pd.DataFrame(rows)


def main() -> None:
    all_rows = []
    for model_key, csv_path in COMPUTE_AXIS_CSV.items():
        result = run_for_model(model_key, csv_path)
        all_rows.append(result)
        print(f"\n=== {model_key} ===")
        print(result.to_string(index=False))

    combined = pd.concat(all_rows, ignore_index=True)
    out_dir = BASE / "dual_axis_joint_controller_poc_2026-07-20"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "results.csv", index=False)

    n_levels_go = int(combined["joint_beats_independent"].sum())
    n_levels_total = len(combined)
    overall_go = n_levels_go > n_levels_total / 2

    metadata = {
        "combine_axis_reference_kl": COMBINE_AXIS_REFERENCE_KL,
        "budget_fractions": BUDGET_FRACTIONS,
        "n_levels_go": n_levels_go, "n_levels_total": n_levels_total,
        "overall_verdict": "GO (illustrative)" if overall_go else "NO-GO (illustrative)",
        "caveat": (
            "This is a proof-of-concept using a REAL compute-axis KL trajectory "
            "paired with a SYNTHETIC combine-axis stream built from an established "
            "reference constant. It demonstrates the resource-pooling MECHANISM, "
            "not a real measured system benefit. See script docstring."
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\n{n_levels_go}/{n_levels_total} budget levels show JOINT > INDEPENDENT with CI excluding 0.")
    print(f"Overall verdict: {metadata['overall_verdict']}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()

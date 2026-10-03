#!/usr/bin/env python3
"""Aimability: how much information about the *effect-time* batch state is
present in the *decision-time* observable state.

Motivation
----------
Four independent scheduling mechanisms in this repository failed with the same
signature: a hindsight opportunity exists, but no online rule recovers it
(StableBatch Oracle 57 / online -7 / LODO 0-of-16; the capture ladder's
sign-flipping -44%/+119%/-20%/-57%). The usual reading is "the controller was
not good enough". This analyser tests a stronger and falsifiable alternative:

    An action is aimable only if the state that determines its value is
    already realised at decision time.

The capture-alignment action sets an admission cap at step k, but its value is
realised as the decode width at step k+lag landing on (or off) a CUDA-graph
capture point. So the decisive quantity is

    I( bucket(k+lag) ; observable_state(k) )

which is a *model-free upper bound on every possible predictor*. If this
information is at the finite-sample noise floor for the lag at which caps
actually bind, then no controller -- however clever -- can aim the action, and
the failures are structural rather than a tuning shortfall.

What this is NOT
----------------
- Not a policy result. Nothing is executed; no counterfactual width is claimed.
- Mutual information is an upper bound on predictability, not an achievable
  accuracy. A high value does not imply a working controller exists.
- Per-episode estimation only. Pooling episodes with different caps would
  manufacture information from between-episode differences rather than from
  within-episode predictability.
- The shuffle floor is reported alongside every estimate because discrete MI
  is biased upward at finite sample size; only the excess over the floor is
  interpretable.
"""

import argparse
import glob
import json
import math
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

# vLLM V1 capture sizes at or below max_num_seqs=32. Cross-checked against the
# engine's own logged `cudagraph_capture_sizes` in 20260908_step_cost_surface_r01.
CAPTURE_SIZES = (1, 2, 4, 8, 16, 24, 32)
LAGS = (1, 2, 4, 8, 16, 32, 64, 128)
N_SHUFFLE = 20


def capture_bucket(width):
    for size in CAPTURE_SIZES:
        if width <= size:
            return size
    return CAPTURE_SIZES[-1]


def entropy(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def mutual_information(xs, ys):
    """Discrete MI in bits. Both sequences must be equal length."""
    n = len(xs)
    if n == 0:
        return 0.0
    joint = Counter(zip(xs, ys))
    px, py = Counter(xs), Counter(ys)
    total = 0.0
    for (x, y), c in joint.items():
        pxy = c / n
        total += pxy * math.log2(pxy / ((px[x] / n) * (py[y] / n)))
    return max(0.0, total)


def shuffle_floor(xs, ys, rng, n_shuffle=N_SHUFFLE):
    """Finite-sample MI floor: destroy the pairing, keep both marginals."""
    ys_copy = list(ys)
    vals = []
    for _ in range(n_shuffle):
        rng.shuffle(ys_copy)
        vals.append(mutual_information(xs, ys_copy))
    return st.mean(vals), max(vals)


def load_cells(results_dirs):
    cells = []
    for root in results_dirs:
        for path in sorted(glob.glob(str(Path(root) / "**" / "cell-*.json"),
                                     recursive=True)):
            data = json.load(open(path))
            steps = data.get("scheduler_steps") or []
            if len(steps) < 2 * max(LAGS):
                continue
            cells.append(dict(
                path=path, regime=data.get("regime"), policy=data.get("policy", "static"),
                cap=data.get("target_cap"), steps=steps))
    return cells


def episode_aimability(cell, rng):
    """Per-episode MI between decision-time state and effect-time bucket."""
    steps = cell["steps"]
    width = [s["decode_requests"] for s in steps]
    bucket = [capture_bucket(w) for w in width]
    running = [s.get("running_before", 0) for s in steps]
    waiting = [s.get("waiting_before", 0) for s in steps]
    n = len(steps)

    out = {}
    for lag in LAGS:
        if n - lag < 64:            # too few pairs to estimate MI at all
            continue
        idx = range(n - lag)
        target = [bucket[k + lag] for k in idx]
        h_target = entropy(target)
        if h_target <= 1e-9:        # effect-time bucket is constant: nothing to predict
            out[lag] = dict(n_pairs=len(target), h_target=0.0, degenerate=True)
            continue

        # Predictor A: the bucket you are in right now (persistence).
        src_b = [bucket[k] for k in idx]
        mi_b = mutual_information(src_b, target)
        floor_b, floor_b_max = shuffle_floor(src_b, target, rng)

        # Predictor B: richer observable state the scheduler genuinely has.
        src_rich = [(bucket[k], min(running[k], 32), min(waiting[k], 8)) for k in idx]
        mi_r = mutual_information(src_rich, target)
        floor_r, floor_r_max = shuffle_floor(src_rich, target, rng)

        persistence = sum(bucket[k] == bucket[k + lag] for k in idx) / len(target)
        # Chance persistence if the two were independent with the same marginals.
        pb_src, pb_tgt = Counter(src_b), Counter(target)
        chance = sum((pb_src[v] / len(target)) * (pb_tgt[v] / len(target))
                     for v in set(src_b) | set(target))

        out[lag] = dict(
            n_pairs=len(target), h_target=h_target, degenerate=False,
            mi_bucket=mi_b, mi_bucket_floor=floor_b, mi_bucket_floor_max=floor_b_max,
            mi_rich=mi_r, mi_rich_floor=floor_r, mi_rich_floor_max=floor_r_max,
            # Excess information over the finite-sample floor, as a fraction of
            # the effect-time uncertainty. This is the number that matters.
            frac_resolved_bucket=max(0.0, mi_b - floor_b) / h_target,
            frac_resolved_rich=max(0.0, mi_r - floor_r) / h_target,
            persistence=persistence, persistence_chance=chance)
    return out


def action_lag(cell):
    """Steps between writing a new admission target and that target first
    actually constraining admission.

    A cap only binds when there is something to admit and the running set is
    already at the target. Writing a target that never binds costs nothing and
    must not be counted as an action.
    """
    steps = cell["steps"]
    caps = [s.get("target_cap") for s in steps]
    lags, writes, binds = [], 0, 0
    for k in range(1, len(steps)):
        if caps[k] == caps[k - 1] or caps[k] is None:
            continue
        writes += 1
        for j in range(k, len(steps)):
            if (steps[j].get("waiting_before", 0) > 0
                    and steps[j].get("actual_active", 0) >= caps[j]):
                lags.append(j - k)
                binds += 1
                break
    return dict(n_cap_writes=writes, n_bound=binds,
                lag_steps_p50=st.median(lags) if lags else None,
                lag_steps_min=min(lags) if lags else None,
                lag_steps_max=max(lags) if lags else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", action="append", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=20260911)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    cells = load_cells(args.results_dir)
    if not cells:
        raise SystemExit("no usable cells found")

    per_cell, lag_stats = [], []
    for cell in cells:
        aim = episode_aimability(cell, rng)
        lag = action_lag(cell)
        per_cell.append(dict(cell=cell["path"], regime=cell["regime"],
                             policy=cell["policy"], cap=cell["cap"],
                             n_steps=len(cell["steps"]), aimability=aim,
                             action_lag=lag))
        if lag["lag_steps_p50"] is not None and cell["policy"] != "static":
            lag_stats.append(lag["lag_steps_p50"])

    # Aggregate the aimability curve across episodes (per-episode estimates only).
    curve = {}
    for lag in LAGS:
        rows = [c["aimability"][lag] for c in per_cell
                if lag in c["aimability"] and not c["aimability"][lag]["degenerate"]]
        if not rows:
            continue
        curve[lag] = dict(
            n_episodes=len(rows),
            frac_resolved_bucket_p50=st.median(r["frac_resolved_bucket"] for r in rows),
            frac_resolved_rich_p50=st.median(r["frac_resolved_rich"] for r in rows),
            frac_resolved_rich_p90=sorted(r["frac_resolved_rich"] for r in rows)[
                int(0.9 * (len(rows) - 1))],
            mi_rich_p50=st.median(r["mi_rich"] for r in rows),
            mi_rich_floor_p50=st.median(r["mi_rich_floor"] for r in rows),
            h_target_p50=st.median(r["h_target"] for r in rows),
            persistence_p50=st.median(r["persistence"] for r in rows),
            persistence_chance_p50=st.median(r["persistence_chance"] for r in rows),
            n_episodes_above_floor=sum(
                1 for r in rows if r["mi_rich"] > r["mi_rich_floor_max"]))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(capture_sizes=list(CAPTURE_SIZES), lags=list(LAGS),
                   n_shuffle=N_SHUFFLE, seed=args.seed,
                   n_cells=len(cells), aimability_curve=curve,
                   observed_action_lag_steps=dict(
                       n_episodes=len(lag_stats),
                       p50=st.median(lag_stats) if lag_stats else None,
                       min=min(lag_stats) if lag_stats else None,
                       max=max(lag_stats) if lag_stats else None),
                   per_cell=per_cell),
              open(out / "aimability.json", "w"), indent=1)

    lines = ["# Aimability: information at decision time about effect time", "",
             f"{len(cells)} sealed episodes. Nothing executed; no policy claim.",
             "`frac resolved` is mutual information in excess of the per-episode",
             "shuffle floor, divided by the effect-time bucket entropy. It upper-bounds",
             "*every* predictor, so a value at zero means no controller can aim the action.",
             "", "## Aimability decay", "",
             "| lag (steps) | episodes | frac resolved (bucket only) | frac resolved (rich state) | "
             "MI rich (bits) | shuffle floor (bits) | H(target) | persistence | chance | "
             "episodes above floor |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for lag, v in sorted(curve.items()):
        lines.append(
            f'| {lag} | {v["n_episodes"]} | {v["frac_resolved_bucket_p50"]:.4f} | '
            f'{v["frac_resolved_rich_p50"]:.4f} | {v["mi_rich_p50"]:.4f} | '
            f'{v["mi_rich_floor_p50"]:.4f} | {v["h_target_p50"]:.4f} | '
            f'{v["persistence_p50"]:.4f} | {v["persistence_chance_p50"]:.4f} | '
            f'{v["n_episodes_above_floor"]}/{v["n_episodes"]} |')

    al = st.median(lag_stats) if lag_stats else None
    lines += ["", "## Lag at which an admission cap actually binds", "",
              f"- dynamic episodes with at least one binding cap write: {len(lag_stats)}",
              f"- median binding lag: {al} steps"
              + (f" (range {min(lag_stats)}-{max(lag_stats)})" if lag_stats else ""),
              "", "A cap write that never binds is not an action and is excluded.", ""]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

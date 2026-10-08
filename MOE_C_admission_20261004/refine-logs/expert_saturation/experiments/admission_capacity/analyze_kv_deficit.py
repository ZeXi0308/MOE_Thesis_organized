#!/usr/bin/env python3
"""Apply the KV deficit primitives to sealed native-vLLM capacity ledgers.

Reads only already-recorded artifacts.  Writes a JSON summary; never modifies
raw data.  The depletion forecast uses an explicitly bounded observed prefix;
release/wait/bridge reconstruction separately uses observed future outcomes.

Usage:
    python3 analyze_kv_deficit.py \
        --cell <dir with raw.json and config.json> [--cell ...] \
        --output <path.json>
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kv_deficit_model import (  # noqa: E402
    BridgeRequirement,
    DeficitLedger,
    DepletionForecast,
    StaggerFeasibility,
    StaticFeasibility,
    decode_drain_rate,
)

# Fixed window start and length after the declared whole cohort has started.
# Validate this window before issuing a forecast at its final call's return.
FIT_START_MARGIN = 25
FIT_LENGTH = 400


def load_json(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    with path.open() as fh:
        return json.load(fh)


def find_raw(cell: Path) -> Path:
    for name in ("raw.json", "raw.json.gz"):
        if (cell / name).exists():
            return cell / name
    raise FileNotFoundError(f"no raw ledger under {cell}")


def last_admission_step(scheduler_steps, declared_requests: int) -> int:
    """Stop at the first prefix in which the declared cohort has started prefill.

    This is the moment the admission action window closes for a closed cohort:
    afterwards there is no waiting new request left to defer.
    """
    if type(declared_requests) is not int or declared_requests <= 0:
        raise ValueError("closed cohort needs a positive declared request count")
    seen = set()
    for step in scheduler_steps:
        for sched in step["scheduled"]:
            rid = sched["internal_request_id"]
            if rid in seen:
                continue
            if sched["computed_before"] == 0 and sched["prefill_tokens"] > 0:
                seen.add(rid)
                if len(seen) == declared_requests:
                    return step["step"]
    raise ValueError("declared closed cohort has not all started prefill")


def validate_fit_window(raw, lo: int, hi: int, usable: int):
    """Only use stable, successful pure-decode work through the window cutoff."""
    steps = [s for s in raw["scheduler_steps"] if lo <= s["step"] <= hi]
    if [s["step"] for s in steps] != list(range(lo, hi + 1)):
        raise ValueError("fit window is incomplete or has duplicate/misaligned steps")
    cohort = {s["internal_request_id"] for s in steps[0]["scheduled"]}
    if not cohort:
        raise ValueError("fit window has no active decode requests")
    for step in steps:
        work = step["scheduled"]
        if (len(work) != len(cohort) or {s["internal_request_id"] for s in work} != cohort
                or step["actual_active"] != len(cohort) or step["decode_requests"] != len(cohort)
                or step["preempted_request_ids"] or any(s["prefill_tokens"] or s["recompute_tokens"]
                    or s["scheduled_tokens"] != 1 or s["decode_tokens"] != 1 for s in work)):
            raise ValueError("fit window is not pure decode with one stable request set")
    calls = [c for c in raw["engine_steps"]
             if c["scheduler_step_start"] <= hi and c["scheduler_step_end"] > lo]
    covered = [s for c in calls for s in range(max(lo, c["scheduler_step_start"]), min(hi + 1, c["scheduler_step_end"]))]
    if covered != list(range(lo, hi + 1)) or any(not c["completed"] or c["scheduler_step_end"] > hi + 1 for c in calls):
        raise ValueError("fit window lacks complete engine-call coverage at its cutoff")
    receipts = {c["returned_s"] for c in calls}
    if (any(e["finished"] and e["received_s"] in receipts for e in raw["output_events"])
            or any(lo <= e["attempted_step"] <= hi for e in raw["preemption_events"])):
        raise ValueError("fit window contains request completion or preemption")
    traces = [m for m in raw["memory_trace"] if lo <= m["attempted_step"] <= hi]
    if ([m["attempted_step"] for m in traces] != list(range(lo, hi + 1))
            or any(not m["schedule_completed"] or m["after"] is None
                or any(m[k]["pool"]["usable_blocks"] != usable for k in ("before", "after")) for m in traces)):
        raise ValueError("fit window has incomplete memory accounting or changed capacity")
    return len(cohort), max(receipts), traces[-1]


def fit_free_slope(memory_trace, lo: int, hi: int):
    pts = [
        (m["attempted_step"], m["before"]["pool"]["free_blocks"])
        for m in memory_trace
        if lo <= m["attempted_step"] <= hi
    ]
    if len(pts) < 10:
        raise ValueError(f"fit window [{lo},{hi}] has only {len(pts)} points")
    n = len(pts)
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n
    resid = max(abs(y - (slope * x + intercept)) for x, y in pts)
    return slope, intercept, resid, n


def step_start_times(scheduler_steps) -> dict[int, float]:
    return {s["step"]: s["start_s"] for s in scheduler_steps}


def step_of_time(starts: dict[int, float], t: float) -> int | None:
    best = None
    for step in sorted(starts):
        if starts[step] <= t:
            best = step
        else:
            break
    return best


def analyse_cell(cell: Path) -> dict:
    raw = load_json(find_raw(cell))
    cfg = load_json(cell / "config.json")
    qualification = load_json(cell / "safe-cap-qualification.json")

    scheduler_steps = raw["scheduler_steps"]
    memory_trace = raw["memory_trace"]
    requests = raw["requests"]
    events = raw["preemption_events"]
    pool0 = memory_trace[0]["before"]["pool"]
    usable = pool0["usable_blocks"]

    # Initialization metadata is available before the measured trajectory.
    if (qualification["status"] != "QUALIFIED" or qualification["group_count"] != 1
            or qualification["manager_group_count"] != 1 or qualification["prefix_caching"] is not False
            or qualification["coordinator_type"] != "KVCacheCoordinatorNoPrefixCache"
            or qualification["usable_blocks"] != usable):
        raise ValueError("requires qualified single-group, unshared fixed KV pool")
    block_size = qualification["block_size"]
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("invalid qualified block size")
    block_size_source = "same-cell pre-measurement safe-cap-qualification.json"

    n = cfg["requests"]
    prompt = cfg["prompt_tokens"]
    max_out = cfg["output_tokens"]

    # --- 1. admission-time static feasibility ------------------------------
    static = StaticFeasibility(n, prompt, max_out, block_size, usable)

    # --- 2. action window vs. constraint ----------------------------------
    admit_close = last_admission_step(scheduler_steps, n)

    # --- 3. depletion forecast from a fixed causal early window -------------
    fit_lo = admit_close + FIT_START_MARGIN
    fit_hi = fit_lo + FIT_LENGTH
    width, forecast_cutoff_s, anchor = validate_fit_window(raw, fit_lo, fit_hi, usable)
    if any(e["attempted_step"] <= fit_hi for e in events):
        raise ValueError("a preemption is already known by forecast time; not a first-depletion forecast")
    slope, intercept, resid, npts = fit_free_slope(memory_trace, fit_lo, fit_hi)

    nominal_drain = decode_drain_rate(width, block_size)

    forecast = DepletionForecast(
        step0=fit_hi + 1,
        free_blocks0=anchor["after"]["pool"]["free_blocks"],
        drain_rate=nominal_drain,
    )

    # --- 4. observed release schedule -------------------------------------
    first_preempt = min((e["attempted_step"] for e in events), default=None)
    starts = step_start_times(scheduler_steps)
    completions = sorted(r["completion_s"] for r in requests if r.get("completion_s"))

    # --- 5. bridge requirement and victim stalls ---------------------------
    bridge = None
    stall_checks = []
    if events:
        first_release_step = step_of_time(starts, completions[0])
        victim_yield = min(sum(e["victim_state"]["block_counts"]) for e in events)
        bridge = BridgeRequirement(
            exhaustion_step=forecast.exhaustion_step,
            first_release_step=first_release_step,
            drain_rate=nominal_drain,
            victim_yield_blocks=victim_yield,
        )

        # Restore order observed in the ledger: earliest recompute first.
        order = sorted(events, key=lambda e: e["attempted_step"], reverse=True)
        by_rid = {r["request_id"]: r for r in requests}
        for rank, ev in enumerate(order):
            rid_full = ev["victim_internal_request_id"]
            rid = raw["internal_to_source"][rid_full]
            req = by_rid.get(rid)
            preempt_t = starts[ev["attempted_step"]]
            later = [t for t in completions if t >= preempt_t]
            if rank < len(later):
                reconstructed_wait = later[rank] - preempt_t
            else:
                reconstructed_wait = None
            stall_checks.append(
                {
                    "request_id": rid,
                    "preempt_step": ev["attempted_step"],
                    "preempt_time_s": preempt_t,
                    "restore_rank": rank,
                    "released_blocks": sum(ev["victim_state"]["block_counts"]),
                    "computed_tokens_at_preempt": ev["victim_state"]["computed_tokens"],
                    "reconstructed_wait_s": reconstructed_wait,
                    "scope": "Observed-future completion schedule reconstruction; not an online wait prediction.",
                    "observed_completion_s": req["completion_s"] if req else None,
                }
            )
    return {
        "cell": cell.name,
        "block_size": block_size,
        "block_size_source": block_size_source,
        "pool": pool0,
        "width": width,
        "static_feasibility": static.as_dict(),
        "admission_window_closes_at_step": admit_close,
        "depletion": {
            "fit_window": [fit_lo, fit_hi],
            "fit_points": npts,
            "fitted_slope_blocks_per_step": slope,
            "nominal_slope_blocks_per_step": nominal_drain,
            "max_abs_residual_blocks": resid,
            "forecast_available_after_step": fit_hi,
            "forecast_available_at_s": forecast_cutoff_s,
            "forecast_anchor_step": fit_hi + 1,
            "forecast_anchor_free_blocks": anchor["after"]["pool"]["free_blocks"],
            "forecast_lead_from_available_step": forecast.lead_time_steps(fit_hi + 1),
            "forecast_exhaustion_step": forecast.exhaustion_step,
            "observed_first_preemption_step": first_preempt,
            "lead_time_steps_over_admission_window": forecast.lead_time_steps(admit_close),
            # Observed, not forecast: the step at which the first request
            # completes and starts returning blocks. This does not establish
            # a general sufficient condition for avoiding preemption.
            "observed_first_release_step": step_of_time(
                step_start_times(scheduler_steps),
                min(r["completion_s"] for r in requests if r.get("completion_s")),
            ),
        },
        "bridge": dict(bridge.as_dict(), scope="Fixed-path victim estimate uses observed future first release and victim block yield; not an online estimate or a policy-independent bound.") if bridge else None,
        "observed_preemptions": len(events),
        "victims": stall_checks,
        "stagger": StaggerFeasibility(n, prompt, max_out, block_size, usable).as_dict(),
        "observation_end_s": raw["observation_end_s"],
        "min_free_blocks_observed": min(
            m["before"]["pool"]["free_blocks"] for m in memory_trace
        ),
    }


def attach_measured_stalls(result: dict, pause_ledger: Path | None) -> None:
    """Fold in the independently computed pause partition, if supplied."""
    if pause_ledger is None or not pause_ledger.exists():
        return
    data = load_json(pause_ledger)
    parts = {
        r["request_id"]: r["partition"]
        for r in data["requests"]
        if r.get("partition")
    }
    led = DeficitLedger()
    for victim in result["victims"]:
        part = parts.get(victim["request_id"])
        if not part:
            continue
        wait = part["before_first_recompute_call_s"]
        rc = part["recompute_calls_span_s"]
        victim["observed_wait_s"] = wait
        victim["observed_recompute_span_s"] = rc
        if victim["reconstructed_wait_s"] is not None:
            victim["wait_reconstruction_abs_error_s"] = abs(victim["reconstructed_wait_s"] - wait)
        led.add_victim(
            victim["request_id"], victim["preempt_time_s"], wait, rc, victim["released_blocks"]
        )
    if led.victims:
        result["deficit_ledger"] = led.as_dict()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", required=True, type=Path)
    ap.add_argument(
        "--pause-ledger",
        action="append",
        default=[],
        type=Path,
        help="optional analyze_pause_ledger.py output, positionally matched to --cell",
    )
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists; refusing overwrite: {args.output}")

    results = []
    for idx, cell in enumerate(args.cell):
        res = analyse_cell(cell)
        ledger = args.pause_ledger[idx] if idx < len(args.pause_ledger) else None
        attach_measured_stalls(res, ledger)
        results.append(res)

    terminal_overflow = [r["static_feasibility"]["terminal_co_residency_exceeds_pool"] for r in results]
    observed = [r["observed_preemptions"] > 0 for r in results]
    summary = {
        "scope": (
            "Post-hoc reconstruction of sealed native vLLM ledgers. Descriptive "
            "model validation only: no counterfactual policy was executed."
        ),
        "cells": results,
        "terminal_co_residency_vs_observed_preemption_agreement": f"{sum(p == o for p, o in zip(terminal_overflow, observed))}/{len(results)}",
        "forecast_errors_steps": [
            r["depletion"]["forecast_exhaustion_step"]
            - r["depletion"]["observed_first_preemption_step"]
            for r in results
            if r["depletion"]["observed_first_preemption_step"] is not None
        ],
        "wait_reconstruction_abs_errors_s": [
            v["wait_reconstruction_abs_error_s"]
            for r in results
            for v in r["victims"]
            if "wait_reconstruction_abs_error_s" in v
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "cells"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

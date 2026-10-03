#!/usr/bin/env python3
"""Read one archived original/lazy native-full ordinary adapter timing pair.

Both cells have the same policy and host timers. This is a within-pair code-cost
check; it does not re-rank the earlier native-reference performance pair.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_native_backfill_cpu_diag_r01 import (
    ADDITIVE, FIELDS, PHASES, describe_phase, ratio, validated_timing,
)
from analyze_native_backfill_only_pair_r01 import read_cell, sha256
from analyze_protection_yield_triplet_r01 import compare, require
from analyze_waiter_backfill_triplet_r01 import ordinary_actions


ARMS = ("ordinary_timed_original", "ordinary_timed_lazy")
MANIFESTS = (
    "5050566b57250692d24976932d1f4fbc31b8dc53f37904fdffb5f2ab1ddc6727",
    "275b25e8cc1db9d642d862a7f66f03a49eee53b528606ea92128b864a3196ad1",
)
SCOPE = "WITHIN_HOST_IMPLEMENTATION_COST_PAIR_NOT_NEW_POLICY_CONFIRMATION"


def timed_cell(cell: dict) -> dict:
    archive = Path(cell["archive"])
    timing_path, drain_path = (archive / name for name in
                               ("timing.json", "post-request-drain.json"))
    timing = json.loads(timing_path.read_text())
    drain = json.loads(drain_path.read_text())
    require(type(drain.get("calls")) is int and drain["calls"] >= 0
            and type(drain.get("seconds")) in (int, float)
            and drain["seconds"] >= 0, "Post-request drain record malformed")
    elapsed = timing["measurement_return_perf_s"] - timing["measurement_start_perf_s"]
    capture = cell["raw"]["observation_end_s"]
    require(elapsed > 0 and capture > 0, "Measurement duration missing")
    store = cell["store"]
    formal = validated_timing(store.get("adapter_host_timing_at_measurement_end"),
                              "measurement_end")
    final = validated_timing(store.get("adapter_host_timing"), "final")
    phases, after = {}, {}
    for phase in PHASES:
        start, end = formal[phase], final[phase]
        require(all(end[key] >= start[key] for key in FIELDS),
                f"{phase}: final timing precedes formal snapshot")
        phases[phase] = describe_phase(start, elapsed, capture)
        after[phase] = {key: end[key] - start[key] for key in ADDITIVE}
        # A per-call maximum cannot be subtracted from a previous maximum.
        after[phase].update(max_wall_ns=None, max_thread_cpu_ns=None)
    total = {key: sum(formal[phase][key] for phase in PHASES)
             for key in ADDITIVE}
    total.update(wall_share_measurement_elapsed=ratio(
                     total["wall_ns"] / 1e9, elapsed),
                 wall_share_raw_capture_duration=ratio(
                     total["wall_ns"] / 1e9, capture),
                 thread_cpu_to_wall_ratio=ratio(
                     total["thread_cpu_ns"], total["wall_ns"]))
    return dict(source_sha256=dict(timing=sha256(timing_path),
                                   post_request_drain=sha256(drain_path)),
                measurement_elapsed_s=elapsed,
                raw_capture_duration_s=capture,
                formal_measurement_phases=phases,
                formal_measurement_disjoint_sum=total,
                after_measurement_through_uninstall_additive=after,
                final_phase_aggregates=final,
                post_request_drain=drain,
                scope=store.get("adapter_host_timing_scope"))


def analyze(session: Path) -> dict:
    plan_path, receipt_path = session / "plan.json", session / "receipt.json"
    plan, receipt = (json.loads(path.read_text()) for path in
                     (plan_path, receipt_path))
    require(plan.get("measurement_scope") == SCOPE
            and tuple(cell.get("arm") for cell in plan.get("cells", [])) == ARMS
            and tuple(plan.get("package_manifest_sha256", [])) == MANIFESTS,
            "Frozen original/lazy timing plan differs")
    require(receipt.get("status") == "CELLS_COMPLETE"
            and receipt.get("plan_sha256") == sha256(plan_path)
            and tuple(cell.get("arm") for cell in receipt.get("cells", [])) == ARMS
            and all(cell.get("exit_code") == 0
                    and cell.get("archive_status") == "VERIFIED"
                    for cell in receipt["cells"]),
            "Two-cell receipt incomplete, unordered or not tied to plan")
    cells = {arm: read_cell(session, index, arm, ordinary=True)
             for index, arm in enumerate(ARMS)}
    original, lazy = (cells[arm] for arm in ARMS)
    comparison = compare(original, lazy)
    action = {arm: ordinary_actions(cells[arm]) for arm in ARMS}
    timings = {arm: timed_cell(cells[arm]) for arm in ARMS}
    orig_begin = timings[ARMS[0]]["formal_measurement_phases"]["begin"]
    lazy_begin = timings[ARMS[1]]["formal_measurement_phases"]["begin"]
    orig_cpu_per_call = orig_begin["average_thread_cpu_ns"]
    lazy_cpu_per_call = lazy_begin["average_thread_cpu_ns"]
    cpu_ratio = ratio(lazy_cpu_per_call, orig_cpu_per_call) if orig_cpu_per_call else None
    original_metrics, lazy_metrics = (cell["arm"]["metrics"]
                                      for cell in (original, lazy))
    complete = all(cell["complete"] for cell in cells.values())
    chain_complete = all(action[arm]["choices"] > 0
                         and action[arm]["actual_output_completion_chains"]
                             == action[arm]["choices"]
                         and action[arm]["native_admissions"] == action[arm]["choices"]
                         and action[arm]["raw_preemption_evidence_complete"]
                         and action[arm]["postnative_preemption_evidence_complete"]
                         for arm in ARMS)
    no_forced = all(cells[arm]["status"].get("forced_rotations") == 0
                    and cells[arm]["store"].get("applied_rotations") == 0
                    for arm in ARMS)
    criteria = dict(
        full_128_request_cohort_both=complete,
        actual_admission_output_completion_chains_both=chain_complete,
        adapter_forced_rotations_zero_both=no_forced,
        begin_thread_cpu_per_call_at_most_half_original=(
            cpu_ratio is not None and cpu_ratio <= 0.50),
        lazy_rate_at_least_97pct_original=(
            original_metrics["actual_output_tokens_s"] > 0
            and lazy_metrics["actual_output_tokens_s"] >=
            .97 * original_metrics["actual_output_tokens_s"]),
        lazy_mean_flow_at_most_105pct_original=(
            lazy_metrics["mean_flow_with_incomplete_penalty_s"] <=
            1.05 * original_metrics["mean_flow_with_incomplete_penalty_s"]),
    )
    status = ("INCOMPLETE_PAIR" if not complete else
              "NO_ACTION" if any(action[arm]["choices"] == 0 for arm in ARMS) else
              "COMPLETE_PAIR")
    return dict(
        status=status, session=str(session),
        plan_sha256=sha256(plan_path), receipt_sha256=sha256(receipt_path),
        package_manifest_sha256=list(MANIFESTS),
        arms={arm: dict(archive=cells[arm]["archive"],
                        source_sha256=cells[arm]["source_sha256"],
                        metrics=cells[arm]["arm"]["metrics"],
                        actual_preemption_count=cells[arm]["raw"].get(
                            "actual_preemption_count"),
                        forced_rotations=cells[arm]["status"].get("forced_rotations"),
                        stop_reason_counts=dict(Counter(
                            request.get("stop_reason")
                            for request in cells[arm]["raw"]["requests"])),
                        ordinary_actions=action[arm],
                        timing=timings[arm]) for arm in ARMS},
        comparison=comparison,
        criterion_values=dict(
            begin_thread_cpu_ns_per_call_original=orig_cpu_per_call,
            begin_thread_cpu_ns_per_call_lazy=lazy_cpu_per_call,
            begin_thread_cpu_per_call_ratio=cpu_ratio,
            output_rate_ratio_lazy_to_original=ratio(
                lazy_metrics["actual_output_tokens_s"],
                original_metrics["actual_output_tokens_s"]),
            mean_flow_ratio_lazy_to_original=ratio(
                lazy_metrics["mean_flow_with_incomplete_penalty_s"],
                original_metrics["mean_flow_with_incomplete_penalty_s"]),
            maximum_request_gap_s=dict(original=original_metrics["max_gap_request_max_s"],
                                       lazy=lazy_metrics["max_gap_request_max_s"])),
        predeclared_criteria=criteria, all_criteria_met=all(criteria.values()),
        interpretation=[
            "Both arms retain the same native-full ordinary policy and identical host timers; the ordered pair tests demand-driven running-view construction within this host.",
            "Four adapter timing segments are disjoint and use the formal measurement-end snapshot. Final minus snapshot adds calls, wall and thread CPU only; maxima are not subtractable. Native scheduler, GPU and asynchronous transfer work are excluded.",
            "The original-relative 97% output-rate and 105% mean-flow limits are service guardrails; no maximum-gap reduction is required for the CPU-cost hypothesis, but every request and fixed goodput point is reported.",
            "Natural output and stop trajectories may differ. Timing instrumentation has overhead, and neither cell is a native-without-adapter reference. Do not subtract this pair from the earlier different-host performance pair or claim its failed native-relative budget was repaired.",
            "A zero-action arm has status NO_ACTION; its service and timing data are retained without action-benefit attribution.",
        ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Output must be a new file")
    result = analyze(args.session)
    with args.output.open("x") as destination:
        json.dump(result, destination, indent=2, ensure_ascii=False, allow_nan=False)
        destination.write("\n")
    print(json.dumps(dict(status=result["status"],
                          criteria=result["predeclared_criteria"],
                          values=result["criterion_values"])))


if __name__ == "__main__":
    main()

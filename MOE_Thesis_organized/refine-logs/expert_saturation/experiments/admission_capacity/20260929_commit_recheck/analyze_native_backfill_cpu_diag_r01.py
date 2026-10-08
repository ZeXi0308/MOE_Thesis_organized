#!/usr/bin/env python3
"""Describe disjoint adapter host timing for one native-full ordinary diagnostic.

This reads a real archived session only. The instrumented episode is not a
performance repeat of the earlier native-full pair.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_native_backfill_only_pair_r01 import read_cell, sha256
from analyze_protection_yield_triplet_r01 import require
from analyze_waiter_backfill_triplet_r01 import ordinary_actions


ARM = "native_full_ordinary_cpu_diagnostic"
PHASES = ("begin", "hold", "schedule_pre", "schedule_post")
FIELDS = ("calls", "wall_ns", "thread_cpu_ns", "max_wall_ns",
          "max_thread_cpu_ns")
ADDITIVE = ("calls", "wall_ns", "thread_cpu_ns")


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def validated_timing(timing: dict, label: str) -> dict:
    require(isinstance(timing, dict) and set(timing) == set(PHASES),
            f"{label}: missing or extra timing phase")
    for phase in PHASES:
        row = timing[phase]
        require(isinstance(row, dict) and set(row) == set(FIELDS),
                f"{label}.{phase}: timing fields differ")
        require(all(type(row[field]) is int and row[field] >= 0
                    for field in FIELDS),
                f"{label}.{phase}: timing values must be nonnegative integers")
        require((row["calls"] == 0 and all(row[field] == 0 for field in FIELDS[1:]))
                or (row["calls"] > 0
                    and row["max_wall_ns"] <= row["wall_ns"]
                    and row["max_thread_cpu_ns"] <= row["thread_cpu_ns"]),
                f"{label}.{phase}: calls and maxima differ")
        require(row["thread_cpu_ns"] <= row["wall_ns"],
                f"{label}.{phase}: thread CPU exceeds wall time")
    return timing


def describe_phase(row: dict, measurement_elapsed_s: float,
                   raw_capture_duration_s: float) -> dict:
    calls, wall, cpu = (row[field] for field in ADDITIVE)
    return dict(**row,
                average_wall_ns=ratio(wall, calls),
                average_thread_cpu_ns=ratio(cpu, calls),
                thread_cpu_to_wall_ratio=ratio(cpu, wall),
                wall_share_measurement_elapsed=ratio(
                    wall / 1_000_000_000, measurement_elapsed_s),
                wall_share_raw_capture_duration=ratio(
                    wall / 1_000_000_000, raw_capture_duration_s))


def analyze(session: Path) -> dict:
    receipt_path = session / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    require(receipt.get("status") == "CELLS_COMPLETE"
            and len(receipt.get("cells", [])) == 1
            and receipt["cells"][0].get("arm") == ARM
            and receipt["cells"][0].get("exit_code") == 0,
            "One-cell diagnostic receipt incomplete or arm differs")
    cell = read_cell(session, 0, ARM, ordinary=True)
    archive = Path(cell["archive"])
    timing_path = archive / "timing.json"
    drain_path = archive / "post-request-drain.json"
    timing = json.loads(timing_path.read_text())
    drain = json.loads(drain_path.read_text())
    require(type(drain.get("calls")) is int and drain["calls"] >= 0
            and isinstance(drain.get("seconds"), (int, float))
            and drain["seconds"] >= 0,
            "Post-request drain record malformed")
    measurement_elapsed_s = (timing["measurement_return_perf_s"]
                             - timing["measurement_start_perf_s"])
    raw_capture_duration_s = cell["raw"]["observation_end_s"]
    require(measurement_elapsed_s > 0 and raw_capture_duration_s > 0,
            "Measurement or raw capture duration missing")
    store = cell["store"]
    formal = validated_timing(store.get("adapter_host_timing_at_measurement_end"),
                              "measurement_end")
    final = validated_timing(store.get("adapter_host_timing"), "final")
    phases = {}
    after = {}
    for phase in PHASES:
        before, end = formal[phase], final[phase]
        require(all(end[field] >= before[field] for field in FIELDS),
                f"{phase}: final timing precedes measurement-end snapshot")
        phases[phase] = describe_phase(before, measurement_elapsed_s,
                                       raw_capture_duration_s)
        after[phase] = {field: end[field] - before[field]
                        for field in ADDITIVE}
        after[phase].update(max_wall_ns=None, max_thread_cpu_ns=None)
    sums = {field: sum(formal[phase][field] for phase in PHASES)
            for field in ADDITIVE}
    sums.update(wall_share_measurement_elapsed=ratio(
                    sums["wall_ns"] / 1_000_000_000, measurement_elapsed_s),
                wall_share_raw_capture_duration=ratio(
                    sums["wall_ns"] / 1_000_000_000, raw_capture_duration_s),
                thread_cpu_to_wall_ratio=ratio(
                    sums["thread_cpu_ns"], sums["wall_ns"]))
    action = ordinary_actions(cell)
    action["gate_counts"] = store.get("ordinary_backfill_gate_counts")
    action["allow_forced_rotations"] = store.get("allow_forced_rotations")
    no_forced = (cell["status"].get("forced_rotations") == 0
                 and store.get("applied_rotations") == 0)
    require(no_forced, "Active forced rotation in native ordinary diagnostic")
    state = ("INCOMPLETE_DIAGNOSTIC" if not cell["complete"] else
             "NO_ACTION" if action["choices"] == 0 else
             "ACTIONS_WITHOUT_COMPLETE_CHAIN" if not action["actual_output_completion_chains"]
             else "COMPLETE_CPU_DIAGNOSTIC")
    return dict(status=state, session=str(session),
                source_sha256=dict(receipt=sha256(receipt_path),
                                   timing=sha256(timing_path),
                                   post_request_drain=sha256(drain_path),
                                   **cell["source_sha256"]),
                complete_requests=cell["arm"]["metrics"]["completed"],
                full_cohort_complete=cell["complete"],
                metrics=cell["arm"]["metrics"],
                actual_preemption_count=cell["raw"].get("actual_preemption_count"),
                forced_rotations=cell["status"].get("forced_rotations"),
                ordinary_actions=action,
                timing=dict(measurement_elapsed_s=measurement_elapsed_s,
                            raw_capture_duration_s=raw_capture_duration_s,
                            raw_capture_duration_source="raw.observation_end_s",
                            formal_measurement_phases=phases,
                            formal_measurement_disjoint_sum=sums,
                            after_measurement_through_uninstall_additive=after,
                            final_phase_aggregates=final,
                            post_request_drain=drain,
                            scope=store.get("adapter_host_timing_scope")),
                interpretation=[
                    "Four measured adapter segments are disjoint; begin and hold occur inside native(), while schedule_pre ends before native() and schedule_post begins after it returns.",
                    "The formal values are the measurement-return snapshot. Final minus snapshot covers subsequent calls through adapter uninstall; only calls, wall and thread CPU totals are additive. A maximum cannot be subtracted.",
                    "Host wall and thread CPU clocks exclude the native scheduler body, GPU execution and asynchronous transfer work; timer and aggregation overhead remains in the diagnostic run.",
                    "A zero-choice run is NO_ACTION, with complete requests and timing retained. No timing result is subtracted from the earlier pair or used for cross-host performance ranking.",
                ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "Output must be a new file")
    result = analyze(args.session)
    with args.output.open("x") as destination:
        json.dump(result, destination, indent=2, ensure_ascii=False,
                  allow_nan=False)
        destination.write("\n")
    print(json.dumps(dict(status=result["status"],
                          completed=result["complete_requests"],
                          choices=result["ordinary_actions"]["choices"])))


if __name__ == "__main__":
    main()

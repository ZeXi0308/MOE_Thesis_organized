#!/usr/bin/env python3
"""Localize each size arm's longest observed output gap and native recovery."""

import argparse
import json
from pathlib import Path


RULES = ("min_held_other", "max_held_other")
ALL = ("tail",) + RULES


def gap(request):
    times = sorted(set(request["token_times_s"]))
    left, right = max(zip(times, times[1:]), key=lambda pair: pair[1] - pair[0])
    return left, right


def analyze(canonical):
    raw = {arm: json.loads((Path(canonical["arms"][arm]["archive"]) /
                            "raw.json").read_text()) for arm in ALL}
    metrics = {arm: {row["request_id"]: row
                     for row in canonical["arms"][arm]["metrics"]["requests"]}
               for arm in ALL}
    chains = {}
    for arm in RULES:
        request_id = max(metrics[arm], key=lambda rid:
                         metrics[arm][rid]["max_gap_s"] or -1)
        request = next(row for row in raw[arm]["requests"]
                       if row["request_id"] == request_id)
        left, right = gap(request)
        native_preemptions = [event for event in raw[arm]["preemption_events"]
                              if event.get("internal_request_id")
                              == request["internal_request_id"]
                              and event.get("original_preemption_called") is True
                              and event.get("original_preemption_returned") is True]
        selected = [row for row in canonical["size_actions"][arm]["rows"]
                    if row["selected_victim"] == request_id]
        in_gap = [row for row in selected
                  if row["selected_preemption_s"] is not None
                  and left <= row["selected_preemption_s"] <= right]
        if len(in_gap) != 1 or abs(right-left-metrics[arm][request_id]["max_gap_s"]) > 1e-8:
            raise ValueError(f"{arm}: worst gap or unique in-gap action differs")
        action = in_gap[0]
        preempt = action["selected_preemption_s"]
        readmit = action["selected_first_recorded_running_readmission_s"]
        if readmit is None or not left <= preempt <= readmit <= right:
            raise ValueError(f"{arm}: native preempt/readmission/output order differs")
        span = right-left
        chain = dict(
            request=request_id, internal_request_id=request["internal_request_id"],
            gap_start_output_s=left, gap_end_next_output_s=right, gap_s=span,
            selected_victim_decisions=len(selected),
            actual_native_preemptions=len(native_preemptions),
            preemptions_inside_gap=sum(left <= event["method_entered_s"] <= right
                                       for event in native_preemptions),
            gap_action=dict(
                step=action["step"], preemption_s=preempt,
                output_count_at_preemption=action["selected_output_count_at_preemption"],
                selected_held_blocks=action["selected_held_blocks"],
                native_tail_held_blocks=action["native_tail_held_blocks"],
                needed_blocks=action["needed_blocks"],
                sufficient_other_count=action["sufficient_other_count"],
                changed_from_native_tail=action["changed_from_native_tail"],
                fallback_reason=action["fallback_reason"],
                first_recorded_running_readmission_s=readmit,
                next_new_output_s=action["selected_next_new_output_s"],
                gap_start_to_preemption_s=preempt-left,
                preemption_to_readmission_s=readmit-preempt,
                readmission_to_next_output_s=right-readmit,
                preemption_to_readmission_fraction_of_gap=(readmit-preempt)/span,
            ),
            all_selected_victim_episodes=[dict(
                step=row["step"], preemption_s=row["selected_preemption_s"],
                output_count_at_preemption=row["selected_output_count_at_preemption"],
                selected_held_blocks=row["selected_held_blocks"],
                native_tail_held_blocks=row["native_tail_held_blocks"],
                needed_blocks=row["needed_blocks"],
                changed_from_native_tail=row["changed_from_native_tail"],
                first_recorded_running_readmission_s=(
                    row["selected_first_recorded_running_readmission_s"]),
                next_new_output_s=row["selected_next_new_output_s"],
                preemption_to_next_output_s=row["selected_preemption_to_next_output_s"])
                for row in selected],
            same_request_complete_metrics={control: dict(
                max_gap_s=metrics[control][request_id]["max_gap_s"],
                flow_s=metrics[control][request_id]["flow_s"],
                ttft_s=metrics[control][request_id]["ttft_s"],
                actual_native_preemptions=sum(
                    event.get("internal_request_id") == next(
                        row["internal_request_id"] for row in raw[control]["requests"]
                        if row["request_id"] == request_id)
                    and event.get("original_preemption_called") is True
                    and event.get("original_preemption_returned") is True
                    for event in raw[control]["preemption_events"]))
                for control in ALL},
        )
        chains[arm] = chain
    return dict(
        status="OBSERVED_WORST_GAP_CHAINS", session=canonical["session"],
        qualification_met=canonical["qualification_met"],
        descriptive_service_budget_met=canonical["descriptive_service_budget_met"],
        arms=chains,
        interpretation_limits=[
            "A preemption-to-readmission interval is observable, but it is not entirely proven to be allocator queue wait.",
            "Serial arms evolve different states; a selected victim size and a later long pause do not prove size caused the pause.",
            "An action that selected the same request as native tail is not a size-rule victim change.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output must be new")
    result = analyze(json.loads(args.result.read_text()))
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({arm: dict(request=row["request"], gap_s=row["gap_s"],
                                selected_victim_decisions=row["selected_victim_decisions"],
                                preempt_to_readmission_s=(
                                    row["gap_action"]["preemption_to_readmission_s"]))
                      for arm, row in result["arms"].items()}))


if __name__ == "__main__":
    main()

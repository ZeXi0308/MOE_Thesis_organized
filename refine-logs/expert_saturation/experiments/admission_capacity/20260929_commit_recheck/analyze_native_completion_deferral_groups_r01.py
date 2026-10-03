#!/usr/bin/env python3
"""Group observed capacity deferrals without assigning causal waiting cost."""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def worst_gap(raw):
    worst = None
    for request in raw["requests"]:
        times = sorted(set(request["token_times_s"]))
        for left, right in zip(times, times[1:]):
            candidate = (right - left, request["request_id"], left, right)
            if worst is None or candidate[0] > worst[0]:
                worst = candidate
    return (dict(gap_s=worst[0], request=worst[1], start_s=worst[2], end_s=worst[3])
            if worst else None)


def contiguous_runs(rows):
    paired = defaultdict(list)
    for row in rows:
        paired[(row["failed_current"], row["witness"])].append(row)
    runs = []
    for pair_rows in paired.values():
        current_run = None
        for row in pair_rows:
            previous = current_run[-1] if current_run else None
            same_uninterrupted_pair = (
                previous is not None
                and previous["step"] <= row["step"] <= previous["step"] + 1
                and (previous["current_next_new_output_s"] is None
                     or previous["current_next_new_output_s"] > row["decision_s"]))
            if same_uninterrupted_pair:
                current_run.append(row)
            else:
                current_run = [row]
                runs.append(dict(rows=current_run))
    runs.sort(key=lambda run: run["rows"][0]["_ordinal"])
    result = []
    for number, run in enumerate(runs):
        events = run["rows"]
        first, last = events[0], events[-1]
        next_output = last["current_next_new_output_s"]
        result.append(dict(
            run_index=number, failed_current=first["failed_current"],
            witness=first["witness"], events=len(events),
            first_step=first["step"], last_step=last["step"],
            first_decision_s=first["decision_s"], last_decision_s=last["decision_s"],
            current_next_output_s=next_output,
            first_deferral_to_next_output_s=(next_output - first["decision_s"]
                                              if next_output is not None else None),
            last_deferral_to_next_output_s=(next_output - last["decision_s"]
                                             if next_output is not None else None),
            witness_completion_s=last["witness_completion_s"],
            witness_status=last["witness_status"],
            witness_stop_reason=last["witness_stop_reason"],
            witness_repreempted_before_completion=any(
                event["witness_repreempted_before_completion"] for event in events),
            row_ordinals=[event["_ordinal"] for event in events],
            decision_times_s=[event["decision_s"] for event in events],
        ))
    return result


def current_episodes(rows, runs):
    by_current = defaultdict(list)
    for row in rows:
        by_current[row["failed_current"]].append(row)
    run_lookup = {ordinal: run["run_index"]
                  for run in runs for ordinal in run["row_ordinals"]}
    actors = []
    for current, events in sorted(by_current.items()):
        episodes = []
        for row in events:
            previous = episodes[-1] if episodes else None
            if (previous is None or (previous["next_output_s"] is not None
                                    and row["decision_s"] >= previous["next_output_s"])):
                episodes.append(dict(events=[row], next_output_s=row["current_next_new_output_s"]))
            else:
                previous["events"].append(row)
        episode_rows = []
        for episode in episodes:
            records = episode["events"]
            first, last = records[0], records[-1]
            next_output = episode["next_output_s"]
            episode_rows.append(dict(
                first_step=first["step"], last_step=last["step"],
                first_decision_s=first["decision_s"], last_decision_s=last["decision_s"],
                actions=len(records), witnesses=sorted({r["witness"] for r in records}),
                runs=len({run_lookup[r["_ordinal"]] for r in records}),
                next_output_s=next_output,
                first_deferral_to_next_output_s=(next_output - first["decision_s"]
                                                  if next_output is not None else None),
                last_deferral_to_next_output_s=(next_output - last["decision_s"]
                                                 if next_output is not None else None),
                current_completion_s=last["current_completion_s"],
                current_status=last["current_status"],
            ))
        actors.append(dict(
            request=current, cumulative_deferrals=len(events),
            uninterrupted_no_output_episodes=len(episode_rows),
            distinct_witnesses=sorted({row["witness"] for row in events}),
            complete_request_max_gap_s=events[0]["actor_costs"]["current"]["max_gap_s"],
            complete_request_flow_s=events[0]["actor_costs"]["current"]["flow_s"],
            peer_arm_complete_request_costs=events[0]["actor_costs"]["current"]["other_arms"],
            episodes=episode_rows,
        ))
    return actors


def witness_actors(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["witness"]].append(row)
    return [dict(
        request=witness, cumulative_deferrals=len(events),
        distinct_failed_currents=sorted({row["failed_current"] for row in events}),
        final_status=events[-1]["witness_status"],
        completion_s=events[-1]["witness_completion_s"],
        stop_reason=events[-1]["witness_stop_reason"],
        final_output_count=events[-1]["witness_final_output_count"],
        completed_at_hard_cap=events[-1]["witness_completed_at_hard_cap"],
        completed_by_eos=events[-1]["witness_completed_by_eos"],
        repreempted_after_any_deferral_before_completion=any(
            row["witness_repreempted_before_completion"] for row in events),
        complete_request_max_gap_s=events[0]["actor_costs"]["witness"]["max_gap_s"],
        complete_request_flow_s=events[0]["actor_costs"]["witness"]["flow_s"],
        peer_arm_complete_request_costs=events[0]["actor_costs"]["witness"]["other_arms"],
    ) for witness, events in sorted(grouped.items())]


def analyze_arm(arm, action, raw):
    rows = sorted(action["rows"], key=lambda row: (row["step"], row["decision_s"]))
    rows = [dict(row, _ordinal=index) for index, row in enumerate(rows)]
    require(len(rows) == action["actions"], f"{arm}: action count mismatch")
    require(all(earlier["step"] <= later["step"] for earlier, later in zip(rows, rows[1:])),
            f"{arm}: reordered deferrals")
    runs = contiguous_runs(rows)
    currents = current_episodes(rows, runs)
    witnesses = witness_actors(rows)
    gap = worst_gap(raw)
    if gap is not None:
        inside = [row for row in rows if gap["start_s"] <= row["decision_s"] <= gap["end_s"]]
        gap["deferral_events_inside_gap_any_actor"] = len(inside)
        gap["deferral_events_inside_gap_as_failed_current"] = sum(
            row["failed_current"] == gap["request"] for row in inside)
        gap["deferral_events_inside_gap_as_witness"] = sum(
            row["witness"] == gap["request"] for row in inside)
        gap["deferral_witnesses_for_worst_request"] = sorted({
            row["witness"] for row in inside if row["failed_current"] == gap["request"]})
        gap["deferral_runs_with_event_inside_gap"] = sum(
            any(gap["start_s"] <= t <= gap["end_s"] for t in run["decision_times_s"])
            for run in runs)
        gap["actual_native_preemptions_of_worst_request_inside_gap"] = [dict(
            step=event["engine_call_index"], time_s=event["method_entered_s"],
            output_count=event.get("last_returned_output_count"))
            for event in raw.get("preemption_events", [])
            if event.get("request_id") == gap["request"]
            and event.get("original_preemption_called") is True
            and event.get("original_preemption_returned") is True
            and gap["start_s"] <= event["method_entered_s"] <= gap["end_s"]]
    episodes = [(actor["request"], episode) for actor in currents
                for episode in actor["episodes"]]
    delay = lambda item: item[1]["first_deferral_to_next_output_s"] or -1
    return dict(
        arm=arm, deferral_events=len(rows), contiguous_same_pair_runs=len(runs),
        distinct_failed_currents=len(currents), distinct_witnesses=len(witnesses),
        max_consecutive_same_pair_events=max((run["events"] for run in runs), default=0),
        max_cumulative_events_for_one_current=max(
            (actor["cumulative_deferrals"] for actor in currents), default=0),
        failed_currents_flow_worse_than_tail=sum(
            actor["complete_request_flow_s"]
            > actor["peer_arm_complete_request_costs"]["tail"]["flow_s"]
            for actor in currents) if arm != "tail" else 0,
        failed_currents_gap_worse_than_tail=sum(
            actor["complete_request_max_gap_s"]
            > actor["peer_arm_complete_request_costs"]["tail"]["max_gap_s"]
            for actor in currents) if arm != "tail" else 0,
        witnesses_repreempted_before_completion=sum(
            actor["repreempted_after_any_deferral_before_completion"]
            for actor in witnesses),
        top_failed_currents_by_events=[dict(request=actor["request"],
                                            events=actor["cumulative_deferrals"])
                                       for actor in sorted(currents,
                                           key=lambda actor: actor["cumulative_deferrals"],
                                           reverse=True)[:5]],
        top_no_output_episodes_by_observed_delay=[dict(
            request=request, actions=episode["actions"],
            first_decision_s=episode["first_decision_s"],
            next_output_s=episode["next_output_s"],
            observed_delay_s=episode["first_deferral_to_next_output_s"])
            for request, episode in sorted(episodes, key=delay, reverse=True)[:5]],
        worst_complete_request_gap=gap,
        runs=runs, failed_currents=currents, witnesses=witnesses,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True,
                        help="Frozen three-arm canonical result JSON")
    parser.add_argument("--output", type=Path, required=True,
                        help="New grouped summary JSON")
    args = parser.parse_args()
    require(not args.output.exists(), "Output must be new")
    canonical = json.loads(args.result.read_text())
    require(set(canonical["arms"]) == {"tail", "prefix_work", "prefix_finish"},
            "Unexpected three-arm canonical result")
    result = dict(
        status="GROUPED_OBSERVED_DEFERRALS",
        canonical_result=str(args.result), session=canonical["session"],
        arms={},
        grouping_rule="A run requires the same failed current and witness on the same or adjacent scheduler steps, allowing other pairs' actions between them, with no recorded current output between decisions. Current episodes merge actions until that current's next actual output; elapsed time is observed, not attributed to deferral.",
        limitations=[
            "Decision timestamps inside a gap mark temporal overlap, not cause of the gap.",
            "The first decision-to-next-output interval can include allocator, waiting, and other work; it is not a sum of deferral costs.",
            "Repeated action records for one witness share its final outcome and must not be counted as independent completed requests.",
        ],
    )
    for arm in ("tail", "prefix_work", "prefix_finish"):
        raw_path = Path(canonical["arms"][arm]["archive"]) / "raw.json"
        raw = json.loads(raw_path.read_text())
        result["arms"][arm] = analyze_arm(
            arm, canonical["deferral_actions"][arm], raw)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({arm: dict(events=row["deferral_events"],
                                runs=row["contiguous_same_pair_runs"],
                                worst_gap=row["worst_complete_request_gap"])
                      for arm, row in result["arms"].items()}))


if __name__ == "__main__":
    main()

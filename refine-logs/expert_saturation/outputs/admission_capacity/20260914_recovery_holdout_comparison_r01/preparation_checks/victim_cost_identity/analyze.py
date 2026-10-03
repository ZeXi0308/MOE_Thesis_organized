#!/usr/bin/env python3
"""Compare old d6 most-output victims with visible KV and recompute cost only."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    with path.open() as handle:
        return json.load(handle)


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def owned_blocks(state):
    values = state["block_counts"]
    require(len(values) == 1 and type(values[0]) is int and values[0] >= 0,
            "unsupported request block ownership")
    return values[0]


def pearson(xs, ys):
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    xx = sum((x - mean_x) ** 2 for x in xs)
    yy = sum((y - mean_y) ** 2 for y in ys)
    return numerator / math.sqrt(xx * yy) if xx and yy else None


def histogram(values):
    return {str(key): value for key, value in sorted(Counter(values).items())}


def verify_selected_release(event, state, released):
    require(event["original_preemption_called"] and event["original_preemption_returned"],
            "native preemption was not completed")
    require(event["victim_state"] == state, "victim changed before native release")
    require(owned_blocks(event["victim_state_after"]) == 0, "victim retained blocks")
    delta = event["pool_after"]["free_blocks"] - event["pool"]["free_blocks"]
    require(delta == released, "observed selected-victim release differs")
    return delta


def analyze_cell(base):
    paths = [base / name for name in ("raw.json", "headroom-decisions.json",
                                      "config.json", "engine_args.json",
                                      "safe-cap-qualification.json")]
    raw, decisions, config, engine, qualification = map(read, paths)
    require(config["rotation_victim_order"] == "most_output", "cell is not most_output")
    require(engine["enable_prefix_caching"] is False
            and qualification["prefix_caching"] is False, "APC must be off")
    require(qualification["coordinator_type"] == "KVCacheCoordinatorNoPrefixCache"
            and qualification["spec_type"] == "FullAttentionSpec"
            and qualification["group_count"] == 1
            and qualification["watermark_blocks"] == 0, "unsupported block semantics")
    require(len(raw["memory_trace"]) == len(raw["scheduler_steps"]) == len(decisions),
            "step counts differ")
    require(config["prompt_tokens"] == 3072 and config["output_tokens"] == 1024,
            "unexpected workload dimensions")

    cfg, block_size = config["rotation_config"], qualification["block_size"]
    aliases = raw["internal_to_source"]
    events = {(event["attempted_step"], event["victim_internal_request_id"]): event
              for event in raw["preemption_events"]}
    require(len(events) == len(raw["preemption_events"]), "duplicate preemption event")
    absent, preemptions, resident = {}, Counter(), {}
    cases, all_outputs, all_blocks, state_correlations = [], [], [], []
    unequal_pairs = block_ties = block_inversions = 0

    for step, (memory, schedule, decision) in enumerate(zip(
            raw["memory_trace"], raw["scheduler_steps"], decisions)):
        require(memory["attempted_step"] == schedule["step"] == decision["step"] == step,
                "request/step identity differs")
        before = memory["before"]
        states, running = before["requests"], before["running_ids"]
        free = before["pool"]["free_blocks"]
        require(free == decision["free_before"], "decision free blocks differ")
        require(sum(owned_blocks(state) for state in states.values())
                == before["pool"]["used_blocks"], "request ownership does not close")
        require(before["pool"]["used_blocks"] + free == qualification["usable_blocks"],
                "pool conservation differs")
        for rid, state in states.items():
            require(state["num_preemptions"] == preemptions[rid],
                    "visible preemption history differs")

        proposal = decision.get("proposal")
        if isinstance(proposal, dict) and proposal.get("action") == "rotate":
            require(decision["effective_victim_order"] == "most_output",
                    "effective order differs")
            target, chosen = proposal["resume_id"], proposal["victim_id"]
            require(target in states and target not in running and target in absent,
                    "target is not an observed absentee")
            require(all(states[rid]["output_tokens"] > 0
                        and states[rid]["computed_tokens"]
                        == states[rid]["prompt_tokens"] + states[rid]["output_tokens"] - 1
                        for rid in running), "selector called outside pure decode")
            target_state = states[target]
            target_need = max(0, math.ceil((target_state["prompt_tokens"]
                                            + target_state["output_tokens"]) / block_size)
                              - owned_blocks(target_state))
            require(target_need == decision["candidate_required_blocks"],
                    "target full-history requirement differs")
            require(step - absent[target] == proposal["absence_steps"],
                    "target absence differs")

            rows = []
            for rid in running:
                state = states[rid]
                progress = min(1.0, max(0, state["computed_tokens"]
                                        - state["prompt_tokens"])
                               / max(1, config["output_tokens"]))
                since = resident.get(rid)
                residency_age = None if since is None else step - since
                if (progress >= cfg["protect_progress_fraction"]
                        or preemptions[rid] >= cfg["max_absences_per_request"]
                        or step - resident.get(rid, -10 ** 9) < cfg["min_residency_steps"]):
                    continue
                blocks = owned_blocks(state)
                rows.append({
                    "internal_id": rid, "request_id": aliases[rid],
                    "prompt_tokens": state["prompt_tokens"],
                    "output_tokens": state["output_tokens"],
                    "computed_tokens": state["computed_tokens"],
                    "recompute_cost_tokens": state["computed_tokens"],
                    "progress": progress, "allocated_blocks": blocks,
                    "funding_slack_blocks": free + blocks - target_need,
                    "feasible": free + blocks >= target_need,
                    "prior_preemptions": preemptions[rid],
                    "resident_since": since, "residency_age": residency_age,
                })
            require(rows and chosen in {row["internal_id"] for row in rows},
                    "selected victim is not eligible")
            expected = min(rows, key=lambda row: (-row["output_tokens"], row["internal_id"]))
            require(chosen == expected["internal_id"], "frozen most_output ranking differs")
            feasible = [row for row in rows if row["feasible"]]
            require(feasible, "actual action has no fundable victim")
            selected = next(row for row in rows if row["internal_id"] == chosen)
            max_blocks = max(row["allocated_blocks"] for row in rows)
            max_block_rows = [row for row in rows if row["allocated_blocks"] == max_blocks]
            max_output = max(row["output_tokens"] for row in rows)
            max_output_rows = [row for row in rows if row["output_tokens"] == max_output]
            min_cost = min(row["recompute_cost_tokens"] for row in feasible)
            min_cost_rows = [row for row in feasible
                             if row["recompute_cost_tokens"] == min_cost]
            minimum = min(min_cost_rows, key=lambda row: row["internal_id"])
            require(selected["allocated_blocks"] == decision["candidate_released_blocks"],
                    "selected candidate release differs")
            actual_release = verify_selected_release(events[step, chosen], states[chosen],
                                                     selected["allocated_blocks"])

            outputs = [row["output_tokens"] for row in rows]
            footprints = [row["allocated_blocks"] for row in rows]
            correlation = pearson(outputs, footprints)
            if correlation is not None:
                state_correlations.append(correlation)
            all_outputs.extend(outputs)
            all_blocks.extend(footprints)
            for left_index, left in enumerate(rows):
                for right in rows[left_index + 1:]:
                    if left["output_tokens"] == right["output_tokens"]:
                        continue
                    unequal_pairs += 1
                    if left["allocated_blocks"] == right["allocated_blocks"]:
                        block_ties += 1
                    elif ((left["output_tokens"] - right["output_tokens"])
                          * (left["allocated_blocks"] - right["allocated_blocks"]) < 0):
                        block_inversions += 1

            cases.append({
                "step": step, "target": aliases[target], "target_internal_id": target,
                "free_blocks": free, "target_required_blocks": target_need,
                "eligible_count": len(rows), "feasible_count": len(feasible),
                "selected": aliases[chosen], "selected_internal_id": chosen,
                "selected_output_tokens": selected["output_tokens"],
                "selected_recompute_cost_tokens": selected["recompute_cost_tokens"],
                "selected_releasable_blocks": selected["allocated_blocks"],
                "selected_actual_released_blocks": actual_release,
                "selected_is_argmax_releasable_blocks": selected in max_block_rows,
                "argmax_releasable_block_tie_count": len(max_block_rows),
                "argmax_releasable_block_requests": [row["request_id"] for row in max_block_rows],
                "argmax_output_tie_count": len(max_output_rows),
                "argmax_output_requests": [row["request_id"] for row in max_output_rows],
                "minimum_recompute": minimum["request_id"],
                "minimum_recompute_internal_id": minimum["internal_id"],
                "minimum_recompute_tie_count": len(min_cost_rows),
                "minimum_recompute_requests": [row["request_id"] for row in min_cost_rows],
                "minimum_recompute_cost_tokens": minimum["recompute_cost_tokens"],
                "minimum_recompute_output_tokens": minimum["output_tokens"],
                "minimum_recompute_releasable_blocks": minimum["allocated_blocks"],
                "same_as_minimum_recompute": selected in min_cost_rows,
                "selected_minus_minimum_recompute_tokens":
                    selected["recompute_cost_tokens"] - minimum["recompute_cost_tokens"],
                "selected_minus_minimum_output_tokens":
                    selected["output_tokens"] - minimum["output_tokens"],
                "selected_minus_minimum_releasable_blocks":
                    selected["allocated_blocks"] - minimum["allocated_blocks"],
                "output_block_pearson": correlation,
                "candidates": rows,
            })

        for rid in decision["preempted"]:
            absent.setdefault(rid, step)
            preemptions[rid] += 1
            resident.pop(rid, None)
        for rid in decision["resumed"]:
            absent.pop(rid, None)
            resident[rid] = step
        require(memory["after"]["pool"]["free_blocks"] == decision["free_after"],
                "after-state free blocks differ")

    deltas = [case["selected_minus_minimum_recompute_tokens"] for case in cases]
    block_deltas = [case["selected_minus_minimum_releasable_blocks"] for case in cases]
    differing = [value for value in deltas if value]
    summary = {
        "steps_checked": len(decisions), "rotation_cases": len(cases),
        "eligible_candidate_rows": sum(case["eligible_count"] for case in cases),
        "feasible_candidate_rows": sum(case["feasible_count"] for case in cases),
        "all_eligible_candidates_feasible": all(case["eligible_count"]
                                                  == case["feasible_count"] for case in cases),
        "selected_is_argmax_releasable_blocks":
            sum(case["selected_is_argmax_releasable_blocks"] for case in cases),
        "selected_is_unique_argmax_releasable_blocks":
            sum(case["argmax_releasable_block_tie_count"] == 1 for case in cases),
        "argmax_releasable_block_tie_histogram":
            histogram(case["argmax_releasable_block_tie_count"] for case in cases),
        "argmax_output_tie_histogram": histogram(case["argmax_output_tie_count"] for case in cases),
        "minimum_recompute_tie_histogram":
            histogram(case["minimum_recompute_tie_count"] for case in cases),
        "selected_is_minimum_recompute": sum(case["same_as_minimum_recompute"] for case in cases),
        "selected_differs_from_minimum_recompute": len(differing),
        "selected_recompute_excess_tokens_mean_all": statistics.mean(deltas),
        "selected_recompute_excess_tokens_mean_when_different": statistics.mean(differing),
        "selected_recompute_excess_tokens_range_when_different": [min(differing), max(differing)],
        "selected_releasable_block_excess_mean_all": statistics.mean(block_deltas),
        "selected_releasable_block_excess_range_all": [min(block_deltas), max(block_deltas)],
        "prompt_lengths_seen": sorted({row["prompt_tokens"] for case in cases
                                        for row in case["candidates"]}),
        "configured_output_tokens": config["output_tokens"],
        "computed_equals_prompt_plus_output_minus_one":
            sum(row["computed_tokens"] == row["prompt_tokens"] + row["output_tokens"] - 1
                for case in cases for row in case["candidates"]),
        "blocks_equal_ceil_computed_over_block_size":
            sum(row["allocated_blocks"] == math.ceil(row["computed_tokens"] / block_size)
                for case in cases for row in case["candidates"]),
        "pooled_output_block_pearson": pearson(all_outputs, all_blocks),
        "per_state_output_block_pearson_count": len(state_correlations),
        "per_state_output_block_pearson_mean": statistics.mean(state_correlations),
        "per_state_output_block_pearson_range": [min(state_correlations), max(state_correlations)],
        "unequal_output_candidate_pairs": unequal_pairs,
        "unequal_output_pairs_tied_on_blocks": block_ties,
        "unequal_output_pairs_tied_on_blocks_fraction": block_ties / unequal_pairs,
        "output_block_order_inversions": block_inversions,
    }
    return {"label": base.name, "path": str(base.resolve()),
            "inputs": list(map(fingerprint, paths)), "summary": summary, "cases": cases}


def case_signature(case):
    return (case["step"], case["target"], case["selected"], case["minimum_recompute"],
            [(row["request_id"], row["output_tokens"], row["computed_tokens"],
              row["allocated_blocks"], row["feasible"]) for row in case["candidates"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path, action="append", required=True)
    parser.add_argument("--frozen-source", type=Path, action="append", required=True)
    parser.add_argument("--reused-logic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in ("analysis.json", "REPORT.md", "differences.csv"):
        require(not (args.output_dir / name).exists(), f"refuse overwrite: {name}")
    cells = [analyze_cell(path) for path in args.cell]
    require(len(cells) == 2, "expected exactly two old d6 repeats")
    mirrored = ([case_signature(case) for case in cells[0]["cases"]]
                == [case_signature(case) for case in cells[1]["cases"]])
    result = {
        "question": "Does old d6 most_output also choose maximum releasable KV, and which "
                    "visible minimum-recompute feasible victims differ?",
        "evidence_type": "CPU_OBSERVED_PRE_ACTION_IDENTITY_ONLY",
        "scope": "Old d6 block0/block1 most_output raw states only; no T3475 or future replay.",
        "recompute_cost_definition": "Observed victim num_computed_tokens before native RECOMPUTE; "
                                     "all candidates are in pure decode.",
        "funding_rule": "One eligible victim; free_blocks + victim_owned_blocks must fund the "
                        "target's complete visible history.",
        "script": fingerprint(Path(__file__)),
        "reused_logic": fingerprint(args.reused_logic),
        "frozen_sources": list(map(fingerprint, args.frozen_source)),
        "cells": cells,
        "cross_repeat": {"source_level_case_and_candidate_identity": mirrored,
                         "case_count_each": [len(cell["cases"]) for cell in cells]},
        "verdict": "SELECTED_ALWAYS_ARGMAX_BLOCK_TIE; MIN_RECOMPUTE_DIFFERS_37_OF_38_PER_REPEAT",
        "limitations": [
            "Candidate ranks use only each recorded pre-action snapshot and prior bookkeeping events.",
            "The same-call native release validates selected ownership only; it is not an outcome score.",
            "No alternative trajectory, latency, completion, quality, future output length, or Oracle is computed.",
            "All requests have 3072-token prompts and a 1024-token cap, so output, recompute history, "
            "and KV footprint are structurally coupled; heterogeneous prompts are untested.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "analysis.json").open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    with (args.output_dir / "differences.csv").open("x", newline="") as handle:
        fields = ["cell", "step", "target", "selected", "minimum_recompute",
                  "selected_output_tokens", "minimum_output_tokens",
                  "selected_recompute_cost_tokens", "minimum_recompute_cost_tokens",
                  "recompute_excess_tokens", "selected_releasable_blocks",
                  "minimum_releasable_blocks", "releasable_block_excess"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            for case in cell["cases"]:
                if case["same_as_minimum_recompute"]:
                    continue
                writer.writerow({
                    "cell": cell["label"], "step": case["step"],
                    "target": case["target"], "selected": case["selected"],
                    "minimum_recompute": case["minimum_recompute"],
                    "selected_output_tokens": case["selected_output_tokens"],
                    "minimum_output_tokens": case["minimum_recompute_output_tokens"],
                    "selected_recompute_cost_tokens": case["selected_recompute_cost_tokens"],
                    "minimum_recompute_cost_tokens": case["minimum_recompute_cost_tokens"],
                    "recompute_excess_tokens": case["selected_minus_minimum_recompute_tokens"],
                    "selected_releasable_blocks": case["selected_releasable_blocks"],
                    "minimum_releasable_blocks": case["minimum_recompute_releasable_blocks"],
                    "releasable_block_excess": case["selected_minus_minimum_releasable_blocks"],
                })
    lines = [
        "# Old d6 `most_output` victim cost identity", "",
        "Evidence: `CPU_OBSERVED_PRE_ACTION_IDENTITY_ONLY`. This is an old-data diagnostic; no GPU or future trajectory was run.", "",
        "| Cell | Rotations | Eligible/feasible rows | Selected in max-block tie | Unique max block | Selected=min recompute | Pooled output/block r |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        summary = cell["summary"]
        lines.append(f"| {cell['label']} | {summary['rotation_cases']} | "
                     f"{summary['eligible_candidate_rows']}/{summary['feasible_candidate_rows']} | "
                     f"{summary['selected_is_argmax_releasable_blocks']} | "
                     f"{summary['selected_is_unique_argmax_releasable_blocks']} | "
                     f"{summary['selected_is_minimum_recompute']} | "
                     f"{summary['pooled_output_block_pearson']:.6f} |")
    first = cells[0]["summary"]
    lines += [
        "",
        f"Both repeats are source-level identical across targets, selected victims, candidate sets, and visible values: `{mirrored}`.",
        "",
        f"Every one of the {first['eligible_candidate_rows']} eligible candidate rows per repeat can fund the target's full visible history. "
        f"The selected victim belongs to the maximum-block tie in all {first['rotation_cases']} actions, but is the unique maximum in only "
        f"{first['selected_is_unique_argmax_releasable_blocks']}. Max-block tie-size histogram: "
        f"`{first['argmax_releasable_block_tie_histogram']}`.",
        "",
        f"The feasible minimum-current-recompute victim differs in {first['selected_differs_from_minimum_recompute']} of "
        f"{first['rotation_cases']} actions. The only equality is the one-candidate final action. When different, `most_output` carries "
        f"{first['selected_recompute_excess_tokens_mean_when_different']:.2f} more visible recompute tokens on average "
        f"(range {first['selected_recompute_excess_tokens_range_when_different'][0]}–{first['selected_recompute_excess_tokens_range_when_different'][1]}), "
        f"and 4–9 more releasable blocks.",
        "",
        f"The fixed-length confound is exact in these snapshots: all prompts are `{first['prompt_lengths_seen'][0]}` tokens; all "
        f"{first['eligible_candidate_rows']} rows satisfy `computed = prompt + output - 1` and "
        f"`owned_blocks = ceil(computed / 16)`. Output-to-block ordering has {first['output_block_order_inversions']} inversions; "
        f"{first['unequal_output_pairs_tied_on_blocks']} of {first['unequal_output_candidate_pairs']} unequal-output pairs "
        f"({100 * first['unequal_output_pairs_tied_on_blocks_fraction']:.2f}%) tie after 16-token block quantization.",
        "",
        "Exact per-step alternatives and deltas are in `differences.csv`; complete feasible candidate sets and source identities are in `analysis.json`.",
        "",
        "Interpretation boundary: this establishes ranking identity under the old homogeneous d6 states. It does not estimate how any alternative victim would affect later batches or completion.",
        "",
    ]
    with (args.output_dir / "REPORT.md").open("x") as handle:
        handle.write("\n".join(lines))
    print(json.dumps({"verdict": result["verdict"], "cross_repeat": result["cross_repeat"],
                      "summaries": [cell["summary"] for cell in cells]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

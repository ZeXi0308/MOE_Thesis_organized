#!/usr/bin/env python3
"""Reproduce shared-prefix opportunities and delayed same-group revisits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


BLOCK = 16


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shared_reusable(a, b):
    limit = min((len(a) - 1) // BLOCK, (len(b) - 1) // BLOCK)
    for block in range(limit):
        start = block * BLOCK
        if a[start:start + BLOCK] != b[start:start + BLOCK]:
            return start
    return limit * BLOCK


def analyze(block_root, actions_path):
    receipt = read(actions_path)
    if receipt.get("schema") != "c-longqa-grain-actions-v1":
        raise ValueError("first-allocation receipt differs")
    arms, examples, raw_sha = {}, {}, {}
    for arm in ("whole", "1", "2"):
        native = block_root / arm / "native"
        paths = {name: native / name for name in ("measured-steps.json", "measured-outputs.json")}
        raw_sha[arm] = {name: sha(path) for name, path in paths.items()}
        steps, output = (read(paths[name]) for name in paths)
        rows = {r["source_index"]: r for r in output}
        acts = receipt["arms"][arm]["first_successful_allocations"]
        calls = steps["scheduler_calls"]
        if len(rows) != 150 or len(acts) != 150:
            raise ValueError(f"{arm}: expected 150 rows and allocation events")
        scheduled = [c["scheduled_tokens_total"] for c in calls]
        cumulative = [0]
        for count in scheduled:
            cumulative.append(cumulative[-1] + count)
        earlier, last_same_group, revisits = [], {}, []
        opportunity = hit_sum = completed_opportunity = completed_miss = 0
        by_source_action = {}
        for action in acts:
            source = action["source_index"]
            row = rows[source]
            if action["external_request_id"] != row["external_request_id"]:
                raise ValueError(f"{arm}: external ID/source mapping differs")
            call = action["schedule_call"]
            when = calls[call]["host_s"]
            candidates = [(shared_reusable(row["prompt_token_ids"], rows[p["source_index"]]["prompt_token_ids"]), p)
                          for p in earlier]
            possible = max((v for v, _ in candidates), default=0)
            prior_done = max((v for v, p in candidates
                              if rows[p["source_index"]]["token_times_s"][0] < when), default=0)
            hit = action["new_prefix_cached_tokens"]
            if hit > possible or hit % BLOCK:
                raise ValueError(f"{arm}: hit exceeds prefix opportunity")
            opportunity += possible
            hit_sum += hit
            completed_opportunity += prior_done
            completed_miss += max(0, prior_done - hit)
            group = action["group_id"]
            if group in last_same_group:
                previous = last_same_group[group]
                prev_call = previous["schedule_call"]
                revisits.append(dict(previous_source_index=previous["source_index"],
                    source_index=source, group_id=group,
                    previous_first_token_before_allocation=(
                        rows[previous["source_index"]]["token_times_s"][0] < when),
                    schedule_call_gap=call - prev_call,
                    interval_scheduled_tokens=cumulative[call] - cumulative[prev_call],
                    shared_reusable_tokens=shared_reusable(row["prompt_token_ids"],
                        rows[previous["source_index"]]["prompt_token_ids"]),
                    actual_prefix_hit_tokens=hit))
            last_same_group[group] = action
            by_source_action[source] = action
            earlier.append(action)
        if opportunity != 107152 or hit_sum != receipt["arms"][arm]["sum_new_prefix_cached_tokens"]:
            raise ValueError(f"{arm}: prefix opportunity/hit accounting differs")
        done = [r for r in revisits if r["previous_first_token_before_allocation"]]
        arms[arm] = dict(theoretical_prior_prompt_prefix_opportunity_tokens=opportunity,
            actual_first_allocation_prefix_hit_tokens=hit_sum,
            missing_hit_tokens=opportunity - hit_sum,
            prior_first_token_completed_opportunity_tokens=completed_opportunity,
            missing_hit_tokens_with_completed_prior_prompt=completed_miss,
            same_group_revisits=len(revisits),
            same_group_revisits_after_prior_first_token=len(done),
            median_completed_revisit_schedule_call_gap=(
                statistics.median(r["schedule_call_gap"] for r in done) if done else None),
            median_completed_revisit_interval_scheduled_tokens=(
                statistics.median(r["interval_scheduled_tokens"] for r in done) if done else None))
        if arm in ("1", "2"):
            previous, current = (38, 123) if arm == "1" else (141, 63)
            p, c = by_source_action[previous], by_source_action[current]
            example = next(r for r in revisits if r["previous_source_index"] == previous
                           and r["source_index"] == current)
            examples[arm] = dict(example,
                exact_prompt_duplicate=(rows[previous]["prompt_token_ids"] == rows[current]["prompt_token_ids"]),
                previous_first_token_host_s=rows[previous]["token_times_s"][0],
                current_first_allocation_schedule_host_s=calls[c["schedule_call"]]["host_s"],
                previous_schedule_call=p["schedule_call"], current_schedule_call=c["schedule_call"])
    return dict(schema="c-longqa-grain-revisit-v1", block_tokens=BLOCK,
        reusable_rule="complete 16-token prefix blocks, at most P-1 prompt tokens per request",
        first_token_rule="prior first output observed before current schedule call implies its full prompt prefill had completed",
        interval_rule="sum of scheduled tokens over [previous first-allocation call, current first-allocation call); not unique KV blocks or cache footprint",
        limitation="No per-block hash/eviction events were recorded; missing hits show absent reusable APC prefix, not exact eviction time or victim blocks",
        actions_sha256=sha(actions_path), raw_sha256=raw_sha, arms=arms, examples=examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-root", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        json.dump(analyze(args.block_root, args.actions), stream, indent=2)
        stream.write("\n")

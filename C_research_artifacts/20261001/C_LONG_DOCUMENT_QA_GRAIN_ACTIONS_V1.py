#!/usr/bin/env python3
"""Describe observed first KV allocations in a fixed-grain three-cell block."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text())


def analyze(block_root: Path, orders_path: Path):
    orders = read(orders_path)
    if orders.get("schema") != "c-longqa-fixed-grain-order-v1" or orders.get("request_count") != 150:
        raise ValueError("frozen fixed-grain order receipt differs")
    groups = orders["arms"]["whole"]["groups"]
    group_by_source = {index: group_id for group_id, group in enumerate(groups)
                       for index in group["source_indices_in_density_order"]}
    if len(group_by_source) != 150 or set(group_by_source) != set(range(150)):
        raise ValueError("prefix groups do not partition source indices")
    result, reference_prompts = {}, None
    for arm in ("whole", "1", "2"):
        native = block_root / arm / "native"
        status, steps, rows = (read(native / name) for name in
            ("status.json", "measured-steps.json", "measured-outputs.json"))
        if status.get("status") != "COMPLETE" or status.get("request_count") != 150:
            raise ValueError(f"{arm}: measured cell is incomplete")
        if (len(rows) != 150 or len({r["external_request_id"] for r in rows}) != 150
                or {r["source_index"] for r in rows} != set(range(150))
                or any(r["external_request_id"] != "measured/" + r["request_id"] for r in rows)):
            raise ValueError(f"{arm}: output request/source mapping differs")
        prompts = {r["source_index"]: r["prompt_token_ids"] for r in rows}
        if reference_prompts is None:
            reference_prompts = prompts
        elif prompts != reference_prompts:
            raise ValueError(f"{arm}: input prompt IDs differ from whole")
        by_external = {r["external_request_id"]: r for r in rows}
        allocations = steps.get("first_successful_allocation", [])
        if (len(allocations) != 150
                or len({a["internal_request_id"] for a in allocations}) != 150
                or {a["external_request_id"] for a in allocations} != set(by_external)
                or any(a["request_id"] != a["internal_request_id"] for a in allocations)):
            raise ValueError(f"{arm}: first allocations are not one-to-one")
        actions = [dict(rank=rank, source_index=by_external[a["external_request_id"]]["source_index"],
                        group_id=group_by_source[by_external[a["external_request_id"]]["source_index"]],
                        schedule_call=a["schedule_call"],
                        new_prefix_cached_tokens=a["new_prefix_cached_tokens"],
                        previous_computed_tokens=a["previous_computed_tokens"],
                        prompt_length_tokens=a["prompt_length_tokens"],
                        scheduled_compute_tokens=a["scheduled_compute_tokens"],
                        external_request_id=a["external_request_id"],
                        internal_request_id=a["internal_request_id"])
                   for rank, a in enumerate(allocations)]
        if any(a["prompt_length_tokens"] != len(reference_prompts[a["source_index"]])
               for a in actions):
            raise ValueError(f"{arm}: first allocation prompt length differs")
        observed = [a["source_index"] for a in actions]
        frozen = orders["arms"][arm]["source_indices_in_submission_order"]
        if (len(frozen) != 150 or sorted(frozen) != list(range(150))
                or steps.get("source_indices_in_submission_order") != frozen):
            raise ValueError(f"{arm}: recorded enqueue order differs from frozen receipt")
        segments = []
        for action in actions:
            if not segments or segments[-1]["group_id"] != action["group_id"]:
                segments.append(dict(group_id=action["group_id"], start_rank=action["rank"], length=0))
            segments[-1]["length"] += 1
        result[arm] = dict(first_allocation_matches_frozen_submission=observed == frozen,
            mismatch_count=sum(a != b for a, b in zip(observed, frozen)),
            observed_source_indices=observed, group_segment_count=len(segments),
            max_group_segment_length=max(s["length"] for s in segments),
            max_segments_per_group=max(Counter(s["group_id"] for s in segments).values()),
            segments=segments, sum_new_prefix_cached_tokens=sum(a["new_prefix_cached_tokens"] for a in actions),
            sum_previous_computed_tokens=sum(a["previous_computed_tokens"] for a in actions),
            peak_used_kv_blocks_after_schedule=status["peak_used_blocks_after_schedule"],
            first_successful_allocations=actions)
    return dict(schema="c-longqa-grain-actions-v1", request_count=150,
        orders_sha256=hashlib.sha256(orders_path.read_bytes()).hexdigest(),
        group_rule="identical first 32 prompt token IDs from frozen receipt",
        scope="First successful KV allocation is an admission proxy; not GPU start, an active-group cap, or a decode pause. No EOS/output text is used.",
        groups=[dict(group_id=i, first32_sha256=g["first32_sha256"], size=len(g["source_indices_in_density_order"]))
                for i, g in enumerate(groups)], arms=result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-root", type=Path, required=True)
    parser.add_argument("--orders", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        json.dump(analyze(args.block_root, args.orders), stream, indent=2)
        stream.write("\n")

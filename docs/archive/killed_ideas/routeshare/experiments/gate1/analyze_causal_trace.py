#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import time


def greedy(ids, sets, maximize=True):
    remaining = sorted(ids)
    pairs = []
    while remaining:
        first = remaining.pop(0)
        second = sorted(
            remaining,
            key=lambda other: (
                -len(sets[first] & sets[other]) if maximize else len(sets[first] & sets[other]),
                other,
            ),
        )[0]
        remaining.remove(second)
        pairs.append((first, second))
    return pairs


def random_pairs(ids, seed):
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    return list(zip(ids[::2], ids[1::2]))


def cost(pairs, current):
    return sum(len(current[a] | current[b]) for a, b in pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--positions", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    routes = {}
    request_order = []
    with args.trace.open() as handle:
        for line in handle:
            if '"layer_id": 0' not in line:
                continue
            row = json.loads(line)
            request = row["request_id"]
            if request not in routes:
                if len(request_order) >= args.requests:
                    continue
                request_order.append(request)
                routes[request] = {}
            position = int(row["token_position"])
            if position < args.positions:
                routes[request].setdefault(position, set()).add(int(row["expert_id"]))
    ids = [request for request in request_order if all(p in routes[request] for p in range(args.positions))]
    if len(ids) % 2:
        ids = ids[:-1]
    if len(ids) < 8:
        raise RuntimeError(f"insufficient complete requests: {len(ids)}")
    rows = []
    for position in range(1, args.positions):
        previous = {request: routes[request][position - 1] for request in ids}
        current = {request: routes[request][position] for request in ids}
        start = time.perf_counter_ns(); causal = greedy(ids, previous); causal_us = (time.perf_counter_ns()-start)/1000
        start = time.perf_counter_ns(); oracle = greedy(ids, current); oracle_us = (time.perf_counter_ns()-start)/1000
        random_plan = random_pairs(ids, 20260723 + position)
        random_cost = cost(random_plan, current)
        causal_cost = cost(causal, current)
        oracle_cost = cost(oracle, current)
        rows.append({
            "position": position,
            "random_union": random_cost,
            "causal_union": causal_cost,
            "oracle_union": oracle_cost,
            "causal_reduction": (random_cost-causal_cost)/random_cost,
            "oracle_reduction": (random_cost-oracle_cost)/random_cost,
            "causal_scheduler_us": causal_us,
            "oracle_scheduler_us": oracle_us,
        })
    payload = {
        "trace": str(args.trace), "complete_requests": len(ids), "positions": args.positions,
        "median_causal_union_reduction": statistics.median(r["causal_reduction"] for r in rows),
        "median_oracle_union_reduction": statistics.median(r["oracle_reduction"] for r in rows),
        "positive_causal_fraction": sum(r["causal_reduction"] > 0 for r in rows)/len(rows),
        "median_causal_scheduler_us": statistics.median(r["causal_scheduler_us"] for r in rows),
        "rows": rows,
        "evidence_boundary": "real native layer-0 prefill routes; previous-token causal pairing; union-count opportunity, not GPU serving latency",
    }
    args.output.write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps({key:value for key,value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()

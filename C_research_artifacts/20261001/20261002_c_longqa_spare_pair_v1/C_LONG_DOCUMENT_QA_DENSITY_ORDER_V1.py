#!/usr/bin/env python3
"""Prompt-only subtree work/request ordering; a development Smith-rule reference.

Each 16-token reusable block is one tree edge. Per-request uncached tails
are separate leaves. Visit sibling subtrees by work / request count,
keeping each subtree contiguous. No output/EOS/reference data is accepted.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time


def density_order(sequences: list[list[int]]) -> list[int]:
    nodes = [dict(children={}, terminal=None, edge=0)]
    for index, tokens in enumerate(sequences):
        if not tokens:
            raise ValueError("empty prompt")
        parent = 0
        reusable = (len(tokens) - 1) // 16
        for offset in range(0, reusable * 16, 16):
            key = tuple(tokens[offset:offset + 16])
            children = nodes[parent]["children"]
            if key not in children:
                children[key] = len(nodes)
                nodes.append(dict(children={}, terminal=None, edge=16))
            parent = children[key]
        # Distinct tail jobs, including exact-duplicate complete prompts.
        nodes[parent]["children"][("tail", index)] = len(nodes)
        nodes.append(dict(children={}, terminal=index,
                          edge=len(tokens) - 16 * reusable))
    for node in reversed(nodes):
        children = [nodes[i] for i in node["children"].values()]
        node["count"] = sum(c["count"] for c in children) if children else 1
        node["work"] = node["edge"] + sum(c["work"] for c in children)
        node["first"] = min(c["first"] for c in children) if children else node["terminal"]
    order, stack = [], [0]
    while stack:
        node = nodes[stack.pop()]
        if node["terminal"] is not None:
            order.append(node["terminal"])
        else:
            children = sorted(node["children"].values(), key=lambda i:
                (Fraction(nodes[i]["work"], nodes[i]["count"]), nodes[i]["first"]))
            stack.extend(reversed(children))
    if sorted(order) != list(range(len(sequences))):
        raise ValueError("order is not a permutation")
    return order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = (args.input_dir / "workload.json").read_bytes()
    rows = json.loads(raw)["requests"]
    started = time.perf_counter()
    order = density_order([row["prompt_token_ids"] for row in rows])
    elapsed = time.perf_counter() - started
    report = dict(schema="c-longqa-density-order-v1",
        workload_sha256=hashlib.sha256(raw).hexdigest(),
        source_indices_in_submission_order=order,
        local_reorder_s=elapsed, request_count=len(rows),
        scope="Prompt-only Smith-rule sibling ordering, contiguous block subtrees; known scheduling idea, unvalidated GPU outcome; no novel-method claim")
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(requests=len(rows), local_reorder_s=elapsed)))


if __name__ == "__main__":
    main()

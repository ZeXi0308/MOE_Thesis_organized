#!/usr/bin/env python3
"""Two-cell development control: common extra shape warmup, DFS then density."""
from pathlib import Path

import C_LONG_DOCUMENT_QA_PRIMARY_PAIR_V1 as block

BASE = block.BASE
block.ROOT = BASE / "shape-warmup-control-dev-v1"
block.FREEZE = BASE / "C_LONG_DOCUMENT_QA_SHAPE_WARMUP_FREEZE_V1.json"
block.RUN_ORDER = ("dfs", "whole")
block.PAIR_DEADLINE_S = 780
block.STAGES = {
    name: Path(f"/dev/shm/c-olmoe-instruct-20261002-shape-{name}-v1")
    for name in block.RUN_ORDER
}
block.CELL_BY_KIND = {
    kind: BASE / "C_LONG_DOCUMENT_QA_SHAPE_WARMUP_CELL_V1.py"
    for kind in ("dfs", "density")
}

original_child = block.resource.run_owned_child
original_qualify = block.qualify_cell


def run_child(command, **kwargs):
    output = Path(command[command.index("--output") + 1])
    policy = output.parent.name
    block.require(policy in block.RUN_ORDER, "unknown warmup control policy")
    return original_child([*command, "--policy", policy], **kwargs)


def qualify(native):
    result = original_qualify(native)
    extra = block.resource.read(native / "shape-warmup-source.json")
    block.require(extra["prompt_tokens"] == 1023
                  and extra["max_tokens"] == 1
                  and extra["source_index"] == 0,
                  "common shape warmup input differs")
    trace = block.resource.read(native / "warmup-large-odd-steps.json")
    block.require(len(trace["scheduler_calls"]) == 1
                  and trace["scheduler_calls"][0]["scheduled_tokens_total"] == 1023,
                  "extra shape warmup did not execute the declared batch")
    result["shape_warmup"] = "QUALIFIED"
    return result


block.resource.run_owned_child = run_child
block.qualify_cell = qualify

if __name__ == "__main__":
    raise SystemExit(block.main())

#!/usr/bin/env python3
"""One development block: unchanged density ranks with whole, one, two releases."""
from pathlib import Path

import C_LONG_DOCUMENT_QA_SHAPE_WARMUP_BLOCK_V2 as setup

block = setup.block
block.ROOT = block.BASE / "fixed-grain-spare-dev-v1"
block.FREEZE = block.BASE / "C_LONG_DOCUMENT_QA_GRAIN_SPARE_FREEZE_V1.json"
block.RUN_ORDER = ("whole", "1", "2")
block.PAIR_DEADLINE_S = 1140
block.STAGES = {
    name: Path(f"/dev/shm/c-olmoe-instruct-20261002-sparegrain-{name}-v1")
    for name in block.RUN_ORDER
}


def atomic_new(path, value):
    if path.parent == block.ROOT and "scientific_scope" in value:
        value = dict(value, scientific_scope=(
            "Three single development cells, whole then q1 then q2, with identical "
            "density ranks and common 1023-token shape warmup. Only prefix-group "
            "enqueue quantum changes. This spare block is UNRUN unless selected; the prerequisite is a completed primary control, not a spare output. No cross-GPU pairing, replication, new-method or held-out claim."))
    return setup.original_atomic_new(path, value)


block.resource.atomic_new = atomic_new

if __name__ == "__main__":
    prior = block.resource.read(block.BASE /
                                "prerequisite-primary-shape-warmup-pair.json")
    block.require(prior["status"] == "COMPLETE"
                  and prior["run_order"] == ["dfs", "whole"],
                  "completed primary common-shape control receipt is required; it is not a spare result")
    raise SystemExit(block.main())

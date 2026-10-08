#!/usr/bin/env python3
"""Repeat the common-warmup control after repairing internal request-ID mapping."""
from pathlib import Path

import C_LONG_DOCUMENT_QA_SHAPE_WARMUP_BLOCK_V1 as setup

block = setup.block
block.ROOT = block.BASE / "shape-warmup-control-dev-v2"
block.FREEZE = block.BASE / "C_LONG_DOCUMENT_QA_SHAPE_WARMUP_FREEZE_V2.json"
block.STAGES = {
    name: Path(f"/dev/shm/c-olmoe-instruct-20261002-shape-{name}-v2")
    for name in block.RUN_ORDER
}
block.CELL_BY_KIND = {
    kind: block.BASE / "C_LONG_DOCUMENT_QA_SHAPE_WARMUP_CELL_V2.py"
    for kind in ("dfs", "density")
}

original_atomic_new = block.resource.atomic_new


def atomic_new(path, value):
    if path.parent == block.ROOT and "scientific_scope" in value:
        value = dict(value, scientific_scope=(
            "Two-cell common 1023-token shape-warmup development control, "
            "DFS then whole density; V1 failed during candidate warmup because "
            "internal and external request IDs differed. All V1 outputs retained. "
            "No reversed-pair, new-method or held-out claim."))
    return original_atomic_new(path, value)


block.resource.atomic_new = atomic_new

if __name__ == "__main__":
    raise SystemExit(block.main())

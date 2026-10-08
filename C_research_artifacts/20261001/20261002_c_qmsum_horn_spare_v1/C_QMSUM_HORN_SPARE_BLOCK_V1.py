#!/usr/bin/env python3
"""Spare-host fallback for the frozen Horn then whole native512 pair."""
from pathlib import Path

import C_QMSUM_HORN_BLOCK_V1 as setup

block = setup.block
block.ROOT = block.BASE / "qmsum-classical-order-512-spare-dev-v1"
block.FREEZE = block.BASE / "C_QMSUM_HORN_SPARE_FREEZE_V1.json"
block.STAGES = {
    name: Path(f"/dev/shm/c-qmsum-horn512-spare-{name}-20261002-v1")
    for name in block.RUN_ORDER
}

if __name__ == "__main__":
    raise SystemExit(block.main())

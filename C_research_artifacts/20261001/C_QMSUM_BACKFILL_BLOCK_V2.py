#!/usr/bin/env python3
"""Same frozen two-arm probe; repair method-source indentation before inference."""
from pathlib import Path

import C_QMSUM_BACKFILL_BLOCK_V1 as setup

block = setup.block
block.ROOT = block.BASE / "qmsum-backfill-development-v2"
block.FREEZE = block.BASE / "C_QMSUM_BACKFILL_FREEZE_V2.json"
block.CELL_BY_KIND = {kind: block.BASE / "C_QMSUM_BACKFILL_CELL_V2.py"
                      for kind in ("dfs", "density")}
block.STAGES = {name: Path(f"/dev/shm/c-qmsum-backfill-{name}-20261002-v2")
                for name in block.RUN_ORDER}

if __name__ == "__main__":
    raise SystemExit(block.main())

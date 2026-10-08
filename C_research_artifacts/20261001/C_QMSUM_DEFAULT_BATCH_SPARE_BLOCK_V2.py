#!/usr/bin/env python3
"""Same8192→512 pair in a fresh root after disk prechecks prevented both V1 runs."""
from pathlib import Path
import C_QMSUM_DEFAULT_BATCH_BLOCK_V1 as original

block = original.block
block.ROOT = block.BASE / "qmsum-offline-batch-default-spare-dev-v2"
block.FREEZE = block.BASE / "C_QMSUM_DEFAULT_BATCH_SPARE_FREEZE_V2.json"
block.STAGES = {name: Path(f"/dev/shm/c-qmsum-default-batch-spare-{name}-20261002-v2")
                for name in block.RUN_ORDER}

if __name__ == "__main__":
    raise SystemExit(block.main())

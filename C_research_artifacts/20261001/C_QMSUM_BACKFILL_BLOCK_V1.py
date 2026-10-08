#!/usr/bin/env python3
"""One bounded QMSum development pair: live allocator backfill, native control."""
import json
from pathlib import Path

import C_QMSUM_BLOCK_V1 as setup

block, resource, require = setup.block, setup.resource, setup.require
block.ROOT = block.BASE / "qmsum-backfill-development-v1"
block.FREEZE = block.BASE / "C_QMSUM_BACKFILL_FREEZE_V1.json"
block.RUN_ORDER = ("backfill", "control")
block.CELL_TIMEOUT_S, block.PAIR_DEADLINE_S = 420, 900
block.CELL_BY_KIND = {name: block.BASE / "C_QMSUM_BACKFILL_CELL_V1.py"
                      for name in ("dfs", "density")}
block.STAGES = {name: Path(f"/dev/shm/c-qmsum-backfill-{name}-20261002-v1")
                for name in block.RUN_ORDER}


def validate():
    freeze = resource.read(block.FREEZE)
    expected = dict(schema="c-qmsum-backfill-freeze-v1", status="PREPARED_UNRUN",
        gpu_uuid=resource.GPU_UUID, shared_lock_inode=resource.LOCK_INODE,
        output_root=block.ROOT.name, request_count=200, output_cap=512,
        run_order=list(block.RUN_ORDER), per_cell_timeout_s=420,
        pair_deadline_s=900, stage_paths={k: str(v) for k, v in block.STAGES.items()})
    for key, value in expected.items():
        require(freeze.get(key) == value, "backfill freeze differs: " + key)
    files = freeze.get("remote_files_sha256", {})
    required = {"C_QMSUM_BACKFILL_CELL_V1.py", "C_QMSUM_BACKFILL_BLOCK_V1.py",
                "C_QMSUM_CELL_V1.py", "C_QMSUM_BLOCK_V1.py",
                "C_LONG_DOCUMENT_QA_PRIMARY_PAIR_V1.py", "C_SPARE_RESOURCE_V1.py",
                "20261002_c_qmsum_inputs_v1/workload.json"}
    require(required <= files.keys(), "missing backfill frozen dependencies")
    for name, digest in files.items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts
                and path.name != block.FREEZE.name and len(digest) == 64,
                "invalid frozen source path")
        require(resource.sha(block.BASE / path) == digest, "changed source: " + name)
    require(not block.ROOT.exists() and not block.ROOT.is_symlink(),
            "immutable backfill result root exists")
    return freeze


def qualify(native):
    # Keep original QMSum completion checks; repair only its object-only reader.
    original = resource.read

    def read(path):
        if Path(path) == native / "measured-outputs.json":
            result = json.loads(Path(path).read_text())
            require(isinstance(result, list), "outputs must be a JSON list")
            return result
        return original(path)

    resource.read = read
    try:
        result = setup.qualify(native)
    finally:
        resource.read = original
    action = resource.read(native / "measured-backfill.json")
    require(action.get("mode") == native.parent.name,
            "backfill/control action receipt differs")
    result["measured_backfill_sha256"] = resource.sha(native / "measured-backfill.json")
    return result


def atomic(path, value):
    if path.parent == block.ROOT and "scientific_scope" in value:
        value = dict(value, scientific_scope=(
            "One development pair, backfill then native control, all QMSum200. "
            "Same prompt-density rank, native128/1024/4096, EOS/512 and warmup. "
            "Only a failed WAITING allocation may trigger a live native-allocator "
            "scan of later waiting requests; no capacity controller, future EOS, "
            "novelty claim, parameter search, or unseen confirmation."))
    return setup.original_atomic(path, value)


block.validate_freeze, block.qualify_cell = validate, qualify
resource.atomic_new = atomic

if __name__ == "__main__":
    raise SystemExit(block.main())

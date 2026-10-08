#!/usr/bin/env python3
"""Reverse native QMSum chunk-budget pair: 1024 then 512, same frozen cell."""
from pathlib import Path

import C_QMSUM_CHUNK_BLOCK_V1 as forward

block, resource, require = forward.block, forward.resource, forward.require
original_atomic = forward.original_atomic
block.ROOT = block.BASE / "qmsum-native-chunk-reverse-v1"
block.FREEZE = block.BASE / "C_QMSUM_CHUNK_REVERSE_FREEZE_V1.json"
block.RUN_ORDER = ("1024", "512")
block.STAGES = {name: Path(f"/dev/shm/c-qmsum-chunk-reverse-{name}-20261002-v1")
                for name in block.RUN_ORDER}


def validate():
    freeze = resource.read(block.FREEZE)
    expected = dict(schema="c-qmsum-native-chunk-reverse-freeze-v1",
        status="PREPARED_UNRUN", gpu_uuid=resource.GPU_UUID,
        shared_lock_inode=resource.LOCK_INODE,
        output_root=block.ROOT.name, request_count=200, output_cap=512,
        run_order=list(block.RUN_ORDER), per_cell_timeout_s=420,
        pair_deadline_s=900, stage_paths={k: str(v) for k, v in block.STAGES.items()})
    for key, value in expected.items():
        require(freeze.get(key) == value, "reverse native chunk freeze differs: " + key)
    files = freeze.get("remote_files_sha256", {})
    require({"C_QMSUM_CHUNK_REVERSE_BLOCK_V1.py", "C_QMSUM_CHUNK_BLOCK_V1.py",
             "C_QMSUM_CHUNK_CELL_V1.py", "C_LONG_DOCUMENT_QA_PRIMARY_PAIR_V1.py",
             "C_SPARE_RESOURCE_V1.py", "20261002_c_qmsum_inputs_v1/workload.json"}
            <= files.keys(), "missing reverse frozen sources")
    for name, digest in files.items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts
                and path.name != block.FREEZE.name and len(digest) == 64,
                "invalid reverse frozen source path")
        require(resource.sha(block.BASE / path) == digest, "changed source: " + name)
    require(not block.ROOT.exists() and not block.ROOT.is_symlink(),
            "immutable reverse chunk result root exists")
    return freeze


def atomic(path, value):
    if path.parent == block.ROOT and "scientific_scope" in value:
        value = dict(value, scientific_scope=("Reverse order native max_num_batched_tokens "
            "1024 then 512, same QMSum200/density/model/128seq/4096KV/EOS512 and "
            "common 511+1023 warmups. One viewed-cohort order check; no unseen "
            "confirmation, method claim, parameter grid, or equal-output assumption."))
    return original_atomic(path, value)


block.validate_freeze, block.qualify_cell = validate, forward.qualify
resource.atomic_new = atomic

if __name__ == "__main__":
    raise SystemExit(block.main())

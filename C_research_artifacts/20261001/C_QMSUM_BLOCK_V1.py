#!/usr/bin/env python3
"""One complete official QMSum200 density-domain measurement under existing lock."""
from pathlib import Path

import C_LONG_DOCUMENT_QA_PRIMARY_PAIR_V1 as block

block.ROOT = block.BASE / "qmsum-whole-development-v1"
block.FREEZE = block.BASE / "C_QMSUM_FREEZE_V1.json"
block.INPUTS = block.BASE / "20261002_c_qmsum_inputs_v1"
block.RUN_ORDER = ("whole",)
block.CELL_TIMEOUT_S, block.PAIR_DEADLINE_S = 420, 480
block.CELL_BY_KIND = {kind: block.BASE / "C_QMSUM_CELL_V1.py" for kind in ("dfs", "density")}
block.STAGES = {"whole": Path("/dev/shm/c-olmoe-instruct-qmsum200-whole-20261002-v1")}
resource, require = block.resource, block.require
original_atomic = resource.atomic_new


def validate():
    frozen = resource.read(block.FREEZE)
    expected = dict(schema="c-qmsum-whole-freeze-v1", status="PREPARED_UNRUN",
        gpu_uuid=resource.GPU_UUID, shared_lock_inode=resource.LOCK_INODE,
        output_root=block.ROOT.name, request_count=200, output_cap=512,
        run_order=list(block.RUN_ORDER), per_cell_timeout_s=420, pair_deadline_s=480,
        stage_paths={name: str(path) for name, path in block.STAGES.items()})
    for key, value in expected.items():
        require(frozen.get(key) == value, "QMSum freeze differs: " + key)
    files = frozen.get("remote_files_sha256", {})
    required = {"C_QMSUM_CELL_V1.py", "C_QMSUM_BLOCK_V1.py",
        "C_LONG_DOCUMENT_QA_PRIMARY_PAIR_V1.py", "C_SPARE_RESOURCE_V1.py",
        "C_LONG_DOCUMENT_QA_DENSITY_CELL_V1.py", "C_LONG_DOCUMENT_QA_DENSITY_ORDER_V1.py",
        "C_SPARE_CELL_HELPERS_V1.py", "C_INSTRUCT_CACHED_MODEL_V1.py",
        "C_INSTRUCT_DOWNLOAD_V2.py", "C_INSTRUCT_MODEL_MANIFEST_V1.json",
        "qmsum_density_order_v1.json", "long_document_qa_density_order_v1.json",
        "20261002_c_qmsum_inputs_v1/config.json", "20261002_c_qmsum_inputs_v1/workload.json",
        "20261002_c_qmsum_inputs_v1/SOURCE_RECEIPT.json",
        "20261001_c_instruct_spare_parent_subset_v1/pkg/memory_telemetry.py",
        "20261001_c_instruct_spare_parent_subset_v1/pkg/run_recovery_cadence.py"}
    require(required <= files.keys(), "missing QMSum frozen dependencies")
    for name, digest in files.items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts
                and path.name != block.FREEZE.name and len(digest) == 64,
                "invalid frozen source path")
        require(resource.sha(block.BASE / path) == digest, "changed source: " + name)
    require(resource.PYTHON.is_file() and resource.HF_HOME.is_dir(), "runtime missing")
    require(not block.ROOT.exists() and not block.ROOT.is_symlink(), "immutable result root exists")
    return frozen


def qualify(native):
    status = resource.read(native / "status.json")
    outputs = resource.read(native / "measured-outputs.json")
    require(status.get("status") == "COMPLETE" and status.get("request_count") == 200
            and len(outputs) == 200 and {r["source_index"] for r in outputs} == set(range(200))
            and len({r["request_id"] for r in outputs}) == 200
            and all(r.get("finished") and r["arrival_s"] == 0
                    and 1 <= len(r["output_token_ids"]) <= 512 for r in outputs),
            "full200 completion differs")
    require(resource.read(native / "native-drain.json").get("status") == "QUALIFIED"
            and resource.read(native / "resolved-eos.json").get("qualified_eos_token_ids") == [50279]
            and resource.read(native / "input-tokenizer-check.json").get("requests_checked") == 200
            and resource.read(native / "model-download.json").get("status") == "VERIFIED",
            "drain, EOS, tokenizer or model differs")
    reset = resource.read(native / "prefix-cache-reset.json")
    require(reset.get("reset_succeeded") is True and reset.get("cached_hash_keys_after") == 0,
            "measured APC initial state differs")
    require(resource.read(native / "resolved-scheduler.json") == dict(max_num_seqs=128,
            max_num_batched_tokens=1024, prefix_caching=True,
            allocated_kv_blocks=4097, usable_kv_blocks=4096), "native resources differ")
    trace = resource.read(native / "measured-steps.json")
    mix = resource.read(native / "measured-service-mix.json")
    require(len(trace["scheduler_calls"]) == len(mix["scheduler_calls"]) == status["schedule_calls"]
            and all(a["scheduled_tokens_total"] == b["scheduled_tokens"]
                    for a, b in zip(trace["scheduler_calls"], mix["scheduler_calls"])),
            "native service-mix alignment differs")
    return dict(status="QUALIFIED", measured_outputs_sha256=resource.sha(native / "measured-outputs.json"),
                measured_steps_sha256=resource.sha(native / "measured-steps.json"),
                measured_service_mix_sha256=resource.sha(native / "measured-service-mix.json"))


def atomic(path, value):
    if path.parent == block.ROOT and "scientific_scope" in value:
        value = dict(value, scientific_scope="One full official QMSum200 density baseline; same model/native resources and common shape warmup. Natural domain measurement, no method comparison, novelty, or held-out confirmation.")
    return original_atomic(path, value)


block.validate_freeze, block.qualify_cell = validate, qualify
resource.atomic_new = atomic

if __name__ == "__main__":
    raise SystemExit(block.main())

#!/usr/bin/env python3
"""One native QMSum batch-budget pair: 8192 then 512, unchanged density rank."""
import json
from pathlib import Path

import C_LONG_DOCUMENT_QA_PRIMARY_PAIR_V1 as block

resource, require = block.resource, block.require
original_atomic = resource.atomic_new
block.ROOT = block.BASE / "qmsum-offline-batch-default-dev-v1"
block.FREEZE = block.BASE / "C_QMSUM_DEFAULT_BATCH_FREEZE_V1.json"
block.INPUTS = block.BASE / "20261002_c_qmsum_inputs_v1"
block.RUN_ORDER = ("8192", "512")
block.CELL_TIMEOUT_S, block.PAIR_DEADLINE_S = 420, 900
block.CELL_BY_KIND = {
    kind: block.BASE / "C_QMSUM_DEFAULT_BATCH_CELL_V1.py"
    for kind in ("dfs", "density")
}
block.STAGES = {
    name: Path(f"/dev/shm/c-qmsum-default-batch-{name}-20261002-v1")
    for name in block.RUN_ORDER
}


def validate():
    freeze = resource.read(block.FREEZE)
    expected = dict(
        schema="c-qmsum-offline-batch-default-freeze-v1",
        status="PREPARED_UNRUN",
        gpu_uuid=resource.GPU_UUID,
        shared_lock_inode=resource.LOCK_INODE,
        output_root=block.ROOT.name,
        request_count=200,
        output_cap=512,
        run_order=list(block.RUN_ORDER),
        per_cell_timeout_s=420,
        pair_deadline_s=900,
        stage_paths={k: str(v) for k, v in block.STAGES.items()},
    )
    for key, value in expected.items():
        require(freeze.get(key) == value, "default batch freeze differs: " + key)
    require(freeze.get("actual_batch_budgets") == [8192, 512],
            "default batch budgets differ")
    files = freeze.get("remote_files_sha256", {})
    require({
        "C_QMSUM_DEFAULT_BATCH_CELL_V1.py",
        "C_QMSUM_DEFAULT_BATCH_BLOCK_V1.py",
        "C_LONG_DOCUMENT_QA_PRIMARY_PAIR_V1.py",
        "C_SPARE_RESOURCE_V1.py",
        "C_LONG_DOCUMENT_QA_DENSITY_CELL_V1.py",
        "C_QMSUM_CELL_V1.py",
        "20261002_c_qmsum_inputs_v1/workload.json",
    } <= files.keys(), "missing frozen default batch sources")
    for name, digest in files.items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts
                and path.name != block.FREEZE.name and len(digest) == 64,
                "invalid frozen source path")
        require(resource.sha(block.BASE / path) == digest,
                "changed source: " + name)
    require(not block.ROOT.exists() and not block.ROOT.is_symlink(),
            "immutable default batch result root exists")
    return freeze


def qualify(native):
    status = resource.read(native / "status.json")
    outputs = json.loads((native / "measured-outputs.json").read_text())
    budget = int(native.parent.name)
    require(status.get("status") == "COMPLETE" and status.get("request_count") == 200
            and len(outputs) == 200
            and {r["source_index"] for r in outputs} == set(range(200))
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
            "formal APC initial state differs")
    require(resource.read(native / "resolved-scheduler.json") == dict(
        max_num_seqs=128, max_num_batched_tokens=budget, prefix_caching=True,
        allocated_kv_blocks=4097, usable_kv_blocks=4096),
        "actual native budget differs")

    warmup = resource.read(native / "shape-warmup-source.json")
    shapes = warmup.get("shapes", [])
    require(warmup.get("native_batch_budget") == budget
            and [x.get("prompt_tokens") for x in shapes] == [511, 1023]
            and all(x.get("source_index") == 0 and x.get("max_tokens") == 1
                    and sum(x["scheduled_tokens_per_call"]) == x["prompt_tokens"]
                    and x["scheduled_tokens_per_call"]
                    and all(1 <= n <= budget for n in x["scheduled_tokens_per_call"])
                    and x.get("reset_status") == "QUALIFIED" for x in shapes),
            "common single-request shape warmups differ")

    batch = warmup.get("batch_shape", {})
    scheduled = batch.get("scheduled_tokens_per_call", [])
    cached = batch.get("first_allocation_cached_tokens")
    require(batch.get("run_id") == "warmup-batch8190"
            and batch.get("source_indices") == [0, 1, 2]
            and batch.get("prompt_tokens_per_request") == [2730] * 3
            and batch.get("request_count") == 3 and batch.get("max_tokens") == 1
            and batch.get("prompt_tokens_total") == 8190
            and type(cached) is int and 0 <= cached <= 8190
            and scheduled and all(type(n) is int and 1 <= n <= budget for n in scheduled)
            and sum(scheduled) + cached == 8190
            and batch.get("reset_status") == "QUALIFIED",
            "common three-request batch shape differs")
    batch_outputs = json.loads((native / "warmup-batch8190-outputs.json").read_text())
    batch_steps = resource.read(native / "warmup-batch8190-steps.json")
    batch_mix = resource.read(native / "warmup-batch8190-service-mix.json")
    batch_reset = resource.read(native / "warmup-batch8190-reset.json")
    first = batch_mix.get("first_successful_allocation", [])
    calls = batch_mix.get("scheduler_calls", [])
    require(len(batch_outputs) == 3
            and {r["source_index"] for r in batch_outputs} == {0, 1, 2}
            and all(r.get("finished") and len(r["output_token_ids"]) == 1
                    for r in batch_outputs)
            and len(first) == 3
            and all(r["prompt_length_tokens"] == 2730
                    and r["previous_computed_tokens"] == 0 for r in first)
            and sum(r["new_prefix_cached_tokens"] for r in first) == cached
            and [c["scheduled_tokens"] for c in calls] == scheduled
            and all(c["decode_tokens"] == 0 for c in calls)
            and not batch_steps.get("preempted_request_ids")
            and all(not c.get("preempted_request_ids")
                    for c in batch_steps["scheduler_calls"])
            and batch_reset["drained_after"]["status"] == "QUALIFIED",
            "batch warmup outputs, allocation, service mix, or reset differ")

    trace = resource.read(native / "measured-steps.json")
    mix = resource.read(native / "measured-service-mix.json")
    require(len(trace["scheduler_calls"]) == len(mix["scheduler_calls"])
            == status["schedule_calls"]
            and all(a["scheduled_tokens_total"] == b["scheduled_tokens"] <= budget
                    for a, b in zip(trace["scheduler_calls"], mix["scheduler_calls"])),
            "native mix or token budget alignment differs")
    return dict(status="QUALIFIED", actual_max_num_batched_tokens=budget,
        measured_outputs_sha256=resource.sha(native / "measured-outputs.json"),
        measured_steps_sha256=resource.sha(native / "measured-steps.json"),
        measured_service_mix_sha256=resource.sha(native / "measured-service-mix.json"))


def atomic(path, value):
    if path.parent == block.ROOT and "scientific_scope" in value:
        value = dict(value, scientific_scope=(
            "Native max_num_batched_tokens8192 then512, same QMSum200/density/model/"
            "128seq/4096KV/EOS512 and common511+1023+8190 warmups. "
            "One development budget comparison; max_num_seqs remains128, "
            "so this is not a full-default configuration or method claim."))
    return original_atomic(path, value)


block.validate_freeze, block.qualify_cell = validate, qualify
resource.atomic_new = atomic

if __name__ == "__main__":
    raise SystemExit(block.main())

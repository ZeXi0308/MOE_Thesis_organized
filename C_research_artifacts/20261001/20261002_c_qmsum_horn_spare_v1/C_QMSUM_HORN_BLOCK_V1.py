#!/usr/bin/env python3
"""One classical-order development pair: Horn then whole, fixed native budget512."""
import json
from pathlib import Path

import C_LONG_DOCUMENT_QA_PRIMARY_PAIR_V1 as block

resource, require = block.resource, block.require
original_atomic = resource.atomic_new
block.ROOT = block.BASE / "qmsum-classical-order-512-dev-v1"
block.FREEZE = block.BASE / "C_QMSUM_HORN_FREEZE_V1.json"
block.INPUTS = block.BASE / "20261002_c_qmsum_inputs_v1"
block.RUN_ORDER = ("horn", "whole")
block.CELL_TIMEOUT_S, block.PAIR_DEADLINE_S = 420, 900
block.CELL_BY_KIND = {kind: block.BASE / "C_QMSUM_HORN_CELL_V1.py"
                      for kind in ("dfs", "density")}
block.STAGES = {name: Path(f"/dev/shm/c-qmsum-horn512-{name}-20261002-v1")
                for name in block.RUN_ORDER}


def validate():
    freeze = resource.read(block.FREEZE)
    expected = dict(schema="c-qmsum-classical-order512-freeze-v1", status="PREPARED_UNRUN",
        gpu_uuid=resource.GPU_UUID, shared_lock_inode=resource.LOCK_INODE,
        output_root=block.ROOT.name, request_count=200, output_cap=512,
        run_order=list(block.RUN_ORDER), per_cell_timeout_s=420,
        pair_deadline_s=900, stage_paths={k: str(v) for k, v in block.STAGES.items()})
    for key, value in expected.items():
        require(freeze.get(key) == value, "native chunk freeze differs: " + key)
    files = freeze.get("remote_files_sha256", {})
    require({"C_QMSUM_HORN_CELL_V1.py", "C_QMSUM_HORN_BLOCK_V1.py",
             "C_LONG_DOCUMENT_QA_PRIMARY_PAIR_V1.py", "C_SPARE_RESOURCE_V1.py",
             "C_LONG_DOCUMENT_QA_DENSITY_CELL_V1.py", "C_QMSUM_CELL_V1.py",
             "C_QMSUM_HORN_COMPACT_V1.py", "C_LONG_DOCUMENT_QA_HORN_ORDER_V1.py",
             "C_LONG_DOCUMENT_QA_TREE_ORDER_WITNESS_V1.py", "qmsum_horn_compact_v1.json",
             "20261002_c_qmsum_inputs_v1/workload.json"} <= files.keys(),
            "missing frozen native chunk sources")
    for name, digest in files.items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts
                and path.name != block.FREEZE.name and len(digest) == 64,
                "invalid frozen source path")
        require(resource.sha(block.BASE / path) == digest, "changed source: " + name)
    require(not block.ROOT.exists() and not block.ROOT.is_symlink(),
            "immutable native chunk result root exists")
    return freeze


def qualify(native):
    status = resource.read(native / "status.json")
    outputs = json.loads((native / "measured-outputs.json").read_text())
    budget = 512
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
            "formal APC initial state differs")
    require(resource.read(native / "resolved-scheduler.json") == dict(max_num_seqs=128,
            max_num_batched_tokens=budget, prefix_caching=True,
            allocated_kv_blocks=4097, usable_kv_blocks=4096), "actual native budget differs")
    warmup = resource.read(native / "shape-warmup-source.json")
    require(warmup["native_batch_budget"] == budget
            and [x["prompt_tokens"] for x in warmup["shapes"]] == [511, 1023]
            and all(sum(x["scheduled_tokens_per_call"]) == x["prompt_tokens"]
                    and max(x["scheduled_tokens_per_call"]) <= budget
                    and x["reset_status"] == "QUALIFIED" for x in warmup["shapes"]),
            "common shape warmup differs")
    trace = resource.read(native / "measured-steps.json")
    mix = resource.read(native / "measured-service-mix.json")
    require(len(trace["scheduler_calls"]) == len(mix["scheduler_calls"]) == status["schedule_calls"]
            and all(a["scheduled_tokens_total"] == b["scheduled_tokens"] <= budget
                    for a, b in zip(trace["scheduler_calls"], mix["scheduler_calls"])),
            "native mix or token budget alignment differs")
    receipt_name = ("qmsum_horn_compact_v1.json" if native.parent.name == "horn"
                    else "qmsum_density_order_v1.json")
    order = resource.read(block.BASE / receipt_name)["source_indices_in_submission_order"]
    source_by_id = {row["external_request_id"]: row["source_index"] for row in outputs}
    observed = [source_by_id[row["external_request_id"]]
                for row in mix["first_successful_allocation"]]
    require(trace["source_indices_in_submission_order"] == order == observed,
            "actual classical or whole first-admission order differs")
    return dict(status="QUALIFIED", actual_max_num_batched_tokens=budget,
        measured_outputs_sha256=resource.sha(native / "measured-outputs.json"),
        measured_steps_sha256=resource.sha(native / "measured-steps.json"),
        measured_service_mix_sha256=resource.sha(native / "measured-service-mix.json"))


def atomic(path, value):
    if path.parent == block.ROOT and "scientific_scope" in value:
        value = dict(value, scientific_scope=("Classical Horn then whole-density order at fixed native512 budget; "
            "same QMSum200/model/128seq/4096KV/EOS512 and common511+1023 warmups. "
            "One viewed strong-reference development comparison, no new method or grid."))
    return original_atomic(path, value)


block.validate_freeze, block.qualify_cell = validate, qualify
resource.atomic_new = atomic

if __name__ == "__main__":
    raise SystemExit(block.main())

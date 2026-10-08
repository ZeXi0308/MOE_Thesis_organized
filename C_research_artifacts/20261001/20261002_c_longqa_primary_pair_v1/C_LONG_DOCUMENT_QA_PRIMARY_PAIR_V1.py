#!/usr/bin/env python3
"""Four fresh LongBench150 cells on one primary GPU lock: DFS, density, density, DFS."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import time

import C_SPARE_RESOURCE_V1 as resource
from C_INSTRUCT_DOWNLOAD_V2 import GIB, memory_headroom

BASE = resource.BASE
ROOT = BASE / "primary-order-pair-dev-v1"
FREEZE = BASE / "C_LONG_DOCUMENT_QA_PRIMARY_PAIR_FREEZE_V1.json"
INPUTS = BASE / "20261001_c_long_document_qa_full_inputs_v1"
MANIFEST = BASE / "C_INSTRUCT_MODEL_MANIFEST_V1.json"
PARENT_SUBSET = BASE / "20261001_c_instruct_spare_parent_subset_v1"
RUN_ORDER = ("dfs1", "density1", "density2", "dfs2")
CELL_BY_KIND = {
    "dfs": BASE / "C_LONG_DOCUMENT_QA_DFS_CELL_V1.py",
    "density": BASE / "C_LONG_DOCUMENT_QA_DENSITY_CELL_V1.py",
}
STAGES = {
    name: Path(f"/dev/shm/c-olmoe-instruct-20261002-primary-{name}-v1")
    for name in RUN_ORDER
}
CELL_TIMEOUT_S = 360
PAIR_DEADLINE_S = 1500


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def idle(state):
    return state.get("used_mib", 65) <= 64 and state.get("compute_processes") == []


def stage_identity(stage):
    info = stage.lstat()
    require(stat.S_ISDIR(info.st_mode) and not stage.is_symlink(),
            "model stage ceased to be a directory")
    return f"{info.st_dev}:{info.st_ino}"


def cleanup_owned_stage(stage, owned_inode, *, child_reaped, gpu_empty):
    if owned_inode is None:
        return dict(status="NOT_CREATED", stage=str(stage))
    receipt = dict(stage=str(stage), owned_inode=owned_inode,
                   child_reaped=child_reaped, gpu_empty=gpu_empty)
    if not child_reaped or not gpu_empty:
        return dict(receipt, status="PRESERVED",
                    reason="child/GPU release unverified")
    try:
        require(stage.parent == Path("/dev/shm")
                and stage_identity(stage) == owned_inode,
                "owned stage path or inode changed")
        shutil.rmtree(stage)
        require(not stage.exists() and not stage.is_symlink(),
                "owned stage remains after cleanup")
        return dict(receipt, status="REMOVED")
    except Exception as exc:
        return dict(receipt, status="PRESERVED",
                    reason=f"{type(exc).__name__}: {exc}")


def validate_freeze():
    freeze = resource.read(FREEZE)
    expected = dict(schema="c-longqa-primary-pair-freeze-v1",
                    status="PREPARED_UNRUN",
                    gpu_uuid=resource.GPU_UUID,
                    shared_lock_inode=resource.LOCK_INODE,
                    output_root=ROOT.name, request_count=150,
                    run_order=list(RUN_ORDER),
                    stage_paths={name: str(STAGES[name]) for name in RUN_ORDER},
                    per_cell_timeout_s=CELL_TIMEOUT_S,
                    pair_deadline_s=PAIR_DEADLINE_S)
    for key, value in expected.items():
        require(freeze.get(key) == value, "primary pair freeze differs: " + key)
    files = freeze.get("remote_files_sha256")
    require(isinstance(files, dict) and files,
            "primary pair freeze lacks source hashes")
    required = {
        "C_LONG_DOCUMENT_QA_PRIMARY_PAIR_V1.py",
        "C_SPARE_RESOURCE_V1.py",
        "C_SPARE_CELL_HELPERS_V1.py",
        "C_INSTRUCT_CACHED_MODEL_V1.py",
        "C_INSTRUCT_DOWNLOAD_V2.py",
        "C_INSTRUCT_MODEL_MANIFEST_V1.json",
        "C_LONG_DOCUMENT_QA_DFS_CELL_V1.py",
        "C_LONG_DOCUMENT_QA_DENSITY_CELL_V1.py",
        "C_LONG_DOCUMENT_QA_DENSITY_ORDER_V1.py",
        "C_LONG_DOCUMENT_QA_PEEK_ORDER_V1.py",
        "long_document_qa_density_order_v1.json",
        "long_document_qa_peek_order_v1.json",
        "20261001_c_long_document_qa_full_inputs_v1/config.json",
        "20261001_c_long_document_qa_full_inputs_v1/workload.json",
        "20261001_c_long_document_qa_full_inputs_v1/SOURCE_RECEIPT.json",
        "20261001_c_instruct_model_metadata_v1/tokenizer.json",
        "20261001_c_instruct_model_metadata_v1/tokenizer_config.json",
        "20261001_c_instruct_spare_parent_subset_v1/pkg/memory_telemetry.py",
        "20261001_c_instruct_spare_parent_subset_v1/pkg/run_recovery_cadence.py",
        "20261001_c_peek_offline_source_v1/SOURCE_RECEIPT.json",
    }
    require(required <= files.keys(), "primary pair freeze omits required sources")
    for name, digest in files.items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts
                and path.name != FREEZE.name and len(digest) == 64,
                "unsafe or invalid frozen source: " + name)
        require(resource.sha(BASE / path) == digest,
                "frozen source/input changed: " + name)
    require(resource.PYTHON.is_file() and resource.HF_HOME.is_dir(),
            "frozen interpreter or HF_HOME directory missing")
    require(not ROOT.exists() and not ROOT.is_symlink(),
            "immutable primary pair root already exists")
    return freeze


def qualify_cell(native):
    names = ("status.json", "native-drain.json", "resolved-eos.json",
             "input-tokenizer-check.json", "prefix-cache-reset.json",
             "model-download.json", "resolved-scheduler.json",
             "measured-steps.json")
    docs = {name: resource.read(native / name) for name in names}
    status, drain, eos, tokenizer, reset, model, resolved, schedule = (
        docs[name] for name in names)
    outputs = json.loads((native / "measured-outputs.json").read_text())
    require(status.get("status") == "COMPLETE"
            and status.get("request_count") == 150
            and len(outputs) == 150
            and all(row.get("finished") is True for row in outputs),
            "150-request completion differs")
    require(drain.get("status") == "QUALIFIED"
            and eos.get("qualification_status") == "QUALIFIED"
            and tokenizer.get("status") == "QUALIFIED"
            and tokenizer.get("requests_checked") == 150
            and reset.get("reset_succeeded") is True
            and model.get("status") == "VERIFIED"
            and resolved == dict(max_num_seqs=128,
                                 max_num_batched_tokens=1024,
                                 prefix_caching=True,
                                 allocated_kv_blocks=4097,
                                 usable_kv_blocks=4096)
            and isinstance(schedule.get("scheduler_calls"), list)
            and len(schedule["scheduler_calls"]) == status.get("schedule_calls"),
            "model, EOS, tokenizer, APC, scheduler, or drain differs")
    return dict(status="QUALIFIED", measured_outputs_sha256=resource.sha(
        native / "measured-outputs.json"),
        measured_steps_sha256=resource.sha(native / "measured-steps.json"))


def run_cell(name, *, lock_fd, freeze_sha, deadline):
    cell_root = ROOT / name
    stage = STAGES[name]
    cell = CELL_BY_KIND["dfs" if name.startswith("dfs") else "density"]
    cell_root.mkdir(exist_ok=False)
    errors, before, after, life, qualification = [], None, None, None, None
    owned_inode = None
    started = resource.stamp()
    try:
        require(not stage.exists() and not stage.is_symlink(),
                "model stage already exists; never adopt it")
        before = resource.probe()
        require(idle(before), "GPU is not idle under shared lock")
        require(shutil.disk_usage(BASE).free > 256 * 1024**2,
                "less than 256 MiB free on data disk")
        manifest = resource.read(MANIFEST)
        total = sum(item["size"] for item in manifest["files"])
        metadata_bytes = sum(item["size"] for item in manifest["files"]
                             if not item["filename"].endswith(".safetensors"))
        headroom = memory_headroom()
        require(headroom >= 26 * GIB + metadata_bytes,
                "insufficient container memory headroom for model and engine")
        require(shutil.disk_usage(stage.parent).free >= metadata_bytes + GIB,
                "insufficient /dev/shm capacity for verified model stage")
        require(time.monotonic() < deadline, "primary pair deadline exhausted")
        stage.mkdir(mode=0o700, exist_ok=False)
        owned_inode = stage_identity(stage)
        os.chmod(stage, 0o700)
        require(stat.S_IMODE(stage.stat().st_mode) == 0o700,
                "model stage permissions differ from 0700")
        resource.atomic_new(cell_root / "start.json", dict(
            started_utc=started, cell=name, gpu_before=before,
            lock_inode=resource.LOCK_INODE, stage=str(stage),
            stage_inode=owned_inode, memory_headroom_bytes=headroom,
            expected_model_bytes=total, weight_cache_already_accounted=True,
            new_weight_staging_bytes=0, new_metadata_bytes=metadata_bytes,
            freeze_sha256=freeze_sha,
            runner_sha256=resource.sha(Path(__file__)),
            cell_source_sha256=resource.sha(cell)))
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=resource.GPU_UUID,
                   HF_HOME=str(resource.HF_HOME), HF_HUB_OFFLINE="1",
                   OMP_NUM_THREADS="25", MKL_NUM_THREADS="25",
                   PYTHONDONTWRITEBYTECODE="1", PYTHONOPTIMIZE="0",
                   VLLM_USE_FLASHINFER_SAMPLER="0",
                   MOE_GPU_LOCK=str(resource.LOCK))
        timeout = min(CELL_TIMEOUT_S, deadline - time.monotonic())
        require(timeout > 0, "primary pair deadline exhausted before child")
        with (cell_root / "child.log").open("x") as log:
            life = resource.run_owned_child(
                [str(resource.PYTHON), str(cell),
                 "--inputs", str(INPUTS), "--output", str(cell_root / "native"),
                 "--parent", str(PARENT_SUBSET),
                 "--model-dir", str(stage), "--model-manifest", str(MANIFEST)],
                cwd=BASE, env=env, log=log, lock_fd=lock_fd, timeout_s=timeout)
            log.flush()
            os.fsync(log.fileno())
        try:
            after = resource.probe()
        except Exception as exc:
            after = dict(probe_error=f"{type(exc).__name__}: {exc}")
            errors.append("post-child GPU probe failed")
        if not (life["exit_code"] == life["child_returncode"] == 0
                and life["child_reaped"] and not life["timed_out"]
                and life["launcher_error"] is None
                and life["interruption_signal"] is None):
            errors.append("owned child failed or timed out")
        if not idle(after):
            errors.append("GPU not empty after owned child exit")
        if not errors:
            qualification = qualify_cell(cell_root / "native")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if owned_inode is not None and after is None:
            try:
                after = resource.probe()
            except Exception as exc:
                after = dict(probe_error=f"{type(exc).__name__}: {exc}")
        child_reaped = life is None or bool(
            life["child_reaped"] or life["child_pid"] is None)
        cleanup = cleanup_owned_stage(stage, owned_inode,
            child_reaped=child_reaped, gpu_empty=after is not None and idle(after))
        resource.atomic_new(cell_root / "stage-cleanup.json", cleanup)
        if cleanup["status"] == "PRESERVED":
            errors.append("owned RAM stage preserved: " + cleanup["reason"])
        receipt = dict(status="COMPLETE" if not errors else "INCOMPLETE",
            cell=name, errors=errors, started_utc=started,
            ended_utc=resource.stamp(), gpu_before=before, gpu_after=after,
            child_lifecycle=life, stage_cleanup=cleanup,
            output_qualification=qualification, freeze_sha256=freeze_sha)
        resource.atomic_new(cell_root / "launcher-receipt.json", receipt)
        print(json.dumps(dict(cell=name, status=receipt["status"],
                              errors=errors, stage_cleanup=cleanup["status"])),
              flush=True)
    return receipt


def main():
    validate_freeze()
    info = resource.LOCK.lstat()
    require(stat.S_ISREG(info.st_mode)
            and f"{info.st_dev}:{info.st_ino}" == resource.LOCK_INODE,
            "shared primary lock inode changed")
    fd = os.open(resource.LOCK, os.O_RDWR | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        require(stat.S_ISREG(opened.st_mode)
                and f"{opened.st_dev}:{opened.st_ino}" == resource.LOCK_INODE,
                "opened shared primary lock changed")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps(dict(status="GPU_DEFERRED",
                                  reason="shared primary lock busy; no child started")),
                  flush=True)
            return 75
        ROOT.mkdir(exist_ok=False)
        started = resource.stamp()
        deadline = time.monotonic() + PAIR_DEADLINE_S
        freeze_sha = resource.sha(FREEZE)
        receipts = []
        errors = []
        try:
            resource.atomic_new(ROOT / "start.json", dict(
                started_utc=started, gpu_uuid=resource.GPU_UUID,
                lock_inode=resource.LOCK_INODE, freeze_sha256=freeze_sha,
                run_order=list(RUN_ORDER), pair_deadline_s=PAIR_DEADLINE_S,
                per_cell_timeout_s=CELL_TIMEOUT_S,
                scientific_scope="Two adjacent reversed development pairs on the same full150 cohort and primary GPU; author DFS is its default order only, not full PEEK; no novelty or held-out claim"))
            for name in RUN_ORDER:
                if time.monotonic() >= deadline:
                    errors.append("primary pair deadline exhausted before " + name)
                    break
                receipt = run_cell(name, lock_fd=fd, freeze_sha=freeze_sha,
                                   deadline=deadline)
                receipts.append(receipt)
                if receipt["status"] != "COMPLETE":
                    errors.append(name + " incomplete; subsequent cells not started")
                    break
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        summary = dict(status="COMPLETE" if not errors and
                       len(receipts) == len(RUN_ORDER) else "INCOMPLETE",
                       errors=errors, started_utc=started,
                       ended_utc=resource.stamp(), freeze_sha256=freeze_sha,
                       run_order=list(RUN_ORDER),
                       cells=[dict(name=row["cell"], status=row["status"],
                                   stage_cleanup=row["stage_cleanup"]["status"],
                                   output_qualification=row["output_qualification"])
                              for row in receipts],
                       scientific_scope="Two adjacent reversed development pairs on one primary GPU and one fixed full150 cohort; no earlier spare timing comparison, full PEEK reproduction, novelty, or held-out claim")
        resource.atomic_new(ROOT / "pair-receipt.json", summary)
        print(json.dumps(dict(status=summary["status"],
                              completed_cells=len(receipts), errors=errors)),
              flush=True)
        return 0 if summary["status"] == "COMPLETE" else 70
    finally:
        os.close(fd)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"PRIMARY_PAIR_PRECHECK_ERROR: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        sys.exit(70)

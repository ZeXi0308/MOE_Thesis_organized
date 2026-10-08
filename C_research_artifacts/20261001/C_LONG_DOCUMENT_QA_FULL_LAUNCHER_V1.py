#!/usr/bin/env python3
"""One bounded 150-request LongBench native resource observation under the spare GPU lock."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import stat
import sys

import C_SPARE_RESOURCE_V1 as resource
from C_INSTRUCT_DOWNLOAD_V2 import GIB, memory_headroom

BASE = resource.BASE
ROOT = BASE / "c-instruct-longbench-full150-dev-v1"
INPUTS = BASE / "20261001_c_long_document_qa_full_inputs_v1"
CELL = BASE / "C_LONG_DOCUMENT_QA_FULL_CELL_V1.py"
MANIFEST = BASE / "C_INSTRUCT_MODEL_MANIFEST_V1.json"
FREEZE = BASE / "C_LONG_DOCUMENT_QA_FULL_FREEZE_V1.json"
PARENT_SUBSET = BASE / "20261001_c_instruct_spare_parent_subset_v1"
STAGE = Path("/dev/shm/c-olmoe-instruct-20261001-longqa150-v1")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def idle(state):
    return state.get("used_mib", 65) <= 64 and state.get("compute_processes") == []


def stage_identity():
    info = STAGE.lstat()
    require(stat.S_ISDIR(info.st_mode) and not STAGE.is_symlink(),
            "model stage ceased to be a directory")
    return f"{info.st_dev}:{info.st_ino}"


def cleanup_owned_stage(owned_inode, *, child_reaped, gpu_empty):
    if owned_inode is None:
        return dict(status="NOT_CREATED", stage=str(STAGE))
    receipt = dict(stage=str(STAGE), owned_inode=owned_inode,
                   child_reaped=child_reaped, gpu_empty=gpu_empty)
    if not child_reaped or not gpu_empty:
        return dict(receipt, status="PRESERVED", reason="child/GPU release unverified")
    try:
        require(STAGE.parent == Path("/dev/shm") and stage_identity() == owned_inode,
                "owned stage path or inode changed")
        shutil.rmtree(STAGE)
        require(not STAGE.exists() and not STAGE.is_symlink(),
                "owned stage remains after cleanup")
        return dict(receipt, status="REMOVED")
    except Exception as exc:
        return dict(receipt, status="PRESERVED",
                    reason=f"{type(exc).__name__}: {exc}")


def main():
    freeze = resource.read(FREEZE)
    for name, digest in freeze["remote_files_sha256"].items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts,
                "unsafe frozen source path")
        require(resource.sha(BASE / path) == digest,
                "frozen source/input changed: " + name)
    require(not ROOT.exists() and not ROOT.is_symlink(),
            "immutable pilot root exists; no repeat")
    info = resource.LOCK.lstat()
    require(stat.S_ISREG(info.st_mode)
            and f"{info.st_dev}:{info.st_ino}" == resource.LOCK_INODE,
            "shared lock inode changed")
    fd = os.open(resource.LOCK, os.O_RDWR | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        require(stat.S_ISREG(opened.st_mode)
                and f"{opened.st_dev}:{opened.st_ino}" == resource.LOCK_INODE,
                "opened shared lock changed")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps(dict(status="GPU_DEFERRED", reason="shared lock busy")),
                  flush=True)
            return 75

        # The one-shot root records every post-lock probe or child failure.
        ROOT.mkdir(exist_ok=False)
        errors, before, after, life = [], None, None, None
        owned_inode = None
        started = resource.stamp()
        try:
            require(not STAGE.exists() and not STAGE.is_symlink(),
                    "model stage already exists; never adopt it")
            before = resource.probe()
            require(idle(before), "GPU is not idle under shared lock")
            require(shutil.disk_usage(BASE).free > 256 * 1024**2,
                    "less than 256 MiB free on data disk")
            manifest = resource.read(MANIFEST)
            total = sum(item["size"] for item in manifest["files"])
            metadata_bytes = sum(item["size"] for item in manifest["files"] if not item["filename"].endswith(".safetensors"))
            headroom = memory_headroom()
            require(headroom >= 26 * GIB + metadata_bytes,
                    "insufficient container memory headroom for model and engine")
            require(shutil.disk_usage(STAGE.parent).free >= metadata_bytes + GIB,
                    "insufficient /dev/shm capacity for verified model stage")
            STAGE.mkdir(mode=0o700, exist_ok=False)
            owned_inode = stage_identity()
            os.chmod(STAGE, 0o700)
            require(stat.S_IMODE(STAGE.stat().st_mode) == 0o700,
                    "model stage permissions differ from 0700")
            resource.atomic_new(ROOT / "start.json", dict(started_utc=started,
                gpu_before=before, lock_inode=resource.LOCK_INODE,
                stage=str(STAGE), stage_inode=owned_inode,
                memory_headroom_bytes=headroom, expected_model_bytes=total, weight_cache_already_accounted=True,
                new_weight_staging_bytes=0, new_metadata_bytes=metadata_bytes,
                freeze_sha256=resource.sha(FREEZE), runner_sha256=resource.sha(Path(__file__))))
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=resource.GPU_UUID,
                HF_HOME=str(resource.HF_HOME), HF_HUB_OFFLINE="1",
                OMP_NUM_THREADS="25", MKL_NUM_THREADS="25",
                PYTHONDONTWRITEBYTECODE="1", PYTHONOPTIMIZE="0",
                VLLM_USE_FLASHINFER_SAMPLER="0", MOE_GPU_LOCK=str(resource.LOCK))
            with (ROOT / "child.log").open("x") as log:
                life = resource.run_owned_child([str(resource.PYTHON), str(CELL),
                    "--inputs", str(INPUTS), "--output", str(ROOT / "native"),
                    "--parent", str(PARENT_SUBSET),
                    "--model-dir", str(STAGE), "--model-manifest", str(MANIFEST)],
                    cwd=BASE, env=env, log=log, lock_fd=fd, timeout_s=360)
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
            try:
                native = ROOT / "native"
                result = resource.read(native / "status.json")
                drain = resource.read(native / "native-drain.json")
                eos = resource.read(native / "resolved-eos.json")
                tokenizer = resource.read(native / "input-tokenizer-check.json")
                reset = resource.read(native / "prefix-cache-reset.json")
                model = resource.read(native / "model-download.json")
                resolved = resource.read(native / "resolved-scheduler.json")
                schedule = resource.read(native / "measured-steps.json")
                outputs = json.loads((native / "measured-outputs.json").read_text())
                require(result["status"] == "COMPLETE"
                        and result["request_count"] == 150
                        and len(outputs) == 150 and all(row["finished"] for row in outputs),
                        "150-request completion differs")
                require(drain["status"] == "QUALIFIED"
                        and eos["qualification_status"] == "QUALIFIED"
                        and tokenizer["status"] == "QUALIFIED"
                        and tokenizer["requests_checked"] == 150
                        and reset["reset_succeeded"] is True
                        and model["status"] == "VERIFIED"
                        and resolved == dict(max_num_seqs=128, max_num_batched_tokens=1024,
                                             prefix_caching=True, allocated_kv_blocks=4097,
                                             usable_kv_blocks=4096)
                        and isinstance(schedule.get("scheduler_calls"), list)
                        and len(schedule["scheduler_calls"]) == result["schedule_calls"],
                        "model, EOS, tokenizer, APC, scheduler observation, or native drain differs")
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            # No other launch can use the GPU or this RAM stage until cleanup ends.
            if life is None:
                child_reaped = True  # no owned child was started
                gpu_empty = before is not None and idle(before)
            else:
                child_reaped = bool(life["child_reaped"] or life["child_pid"] is None)
                gpu_empty = after is not None and idle(after)
            cleanup = cleanup_owned_stage(owned_inode,
                child_reaped=child_reaped, gpu_empty=gpu_empty)
            resource.atomic_new(ROOT / "stage-cleanup.json", cleanup)
            if cleanup["status"] == "PRESERVED":
                errors.append("owned RAM stage preserved: " + cleanup["reason"])
            receipt = dict(status="COMPLETE" if not errors else "INCOMPLETE",
                errors=errors, started_utc=started, ended_utc=resource.stamp(),
                gpu_before=before, gpu_after=after, child_lifecycle=life,
                stage_cleanup=cleanup, freeze_sha256=resource.sha(FREEZE),
                scientific_scope="150-request native LongBench multifieldqa_en resource observation; no capacity-policy or speedup claim")
            resource.atomic_new(ROOT / "launcher-receipt.json", receipt)
            print(json.dumps(dict(status=receipt["status"], errors=errors,
                                  stage_cleanup=cleanup["status"])), flush=True)
        return 0 if not errors else 70
    finally:
        # Closing after the owned child is reaped and RAM stage cleanup releases flock.
        os.close(fd)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"INSTRUCT_LAUNCH_PRECHECK_ERROR: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        sys.exit(70)

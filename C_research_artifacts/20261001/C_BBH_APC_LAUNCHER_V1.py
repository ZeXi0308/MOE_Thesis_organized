#!/usr/bin/env python3
"""One fixed native APC off/on qualification pair; not a timing benchmark."""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import C_NATIVE_PAST_FUTURE_AE_LAUNCHER_V1 as resource

BASE = resource.BASE
ROOT = BASE / "c-bbh-native-apc-pair-dev-v1"
INPUTS = BASE / "20261001_c_bbh_apc_inputs_v1"
CELL = BASE / "C_BBH_APC_CELL_V1.py"
FREEZE = BASE / "C_BBH_APC_FREEZE_V1.json"
ORDER = ("apc_off", "apc_on")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def idle(state):
    return state.get("used_mib", 65) <= 64 and state.get("compute_processes") == []


def main():
    freeze = resource.read(FREEZE)
    for name, digest in freeze["remote_files_sha256"].items():
        require(resource.sha(BASE / name) == digest, "frozen source/input changed: " + name)
    require(not ROOT.exists() and not ROOT.is_symlink(), "immutable pair root exists; no repeat")
    info = resource.LOCK.lstat()
    require(stat.S_ISREG(info.st_mode) and f"{info.st_dev}:{info.st_ino}" == resource.LOCK_INODE,
            "shared lock inode changed")
    fd = os.open(resource.LOCK, os.O_RDWR | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        require(f"{info.st_dev}:{info.st_ino}" == resource.LOCK_INODE, "opened lock changed")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps(dict(status="GPU_DEFERRED", reason="shared lock busy")), flush=True)
            return 75
        require(shutil.disk_usage(BASE).free > 256 * 1024**2, "less than 256 MiB free")
        require(idle(resource.probe()), "GPU is not idle under existing shared lock")
        ROOT.mkdir(exist_ok=False)
        resource.atomic_new(ROOT / "start.json", dict(started_utc=resource.stamp(),
            order=list(ORDER), lock_inode=resource.LOCK_INODE,
            freeze_sha256=resource.sha(FREEZE), runner_sha256=resource.sha(Path(__file__))))
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=resource.GPU_UUID, HF_HOME=str(resource.HF_HOME),
            HF_HUB_OFFLINE="1", OMP_NUM_THREADS="25", MKL_NUM_THREADS="25",
            PYTHONDONTWRITEBYTECODE="1", PYTHONOPTIMIZE="0",
            VLLM_USE_FLASHINFER_SAMPLER="0", MOE_GPU_LOCK=str(resource.LOCK))
        completed, errors = [], []
        for arm in ORDER:
            before = resource.probe()
            require(idle(before), "GPU did not remain idle between cells")
            started = resource.stamp()
            with (ROOT / (arm + ".log")).open("x") as log:
                life = resource.run_owned_child([str(resource.PYTHON), str(CELL),
                    "--arm", arm, "--inputs", str(INPUTS), "--output", str(ROOT / arm),
                    "--parent", str(resource.PARENT)], cwd=BASE, env=env, log=log,
                    lock_fd=fd, timeout_s=900)
            post = resource.probe()
            cell_errors = []
            if not (life["exit_code"] == life["child_returncode"] == 0
                    and life["child_reaped"] and not life["timed_out"]
                    and life["launcher_error"] is None and life["interruption_signal"] is None):
                cell_errors.append("owned child failed or timed out")
            if not idle(post):
                cell_errors.append("GPU not empty after owned child exit")
            try:
                result = resource.read(ROOT / arm / "status.json")
                drain = resource.read(ROOT / arm / "native-drain.json")
                args = resource.read(ROOT / arm / "engine_args.json")
                outputs = json.loads((ROOT / arm / "measured-outputs.json").read_text())
                require(result["status"] == "COMPLETE" and result["request_count"] == 32,
                        "request completion failed")
                require(drain["status"] == "QUALIFIED" and len(outputs) == 32
                        and all(row["finished"] for row in outputs), "incomplete drain or outputs")
                require(args["enable_prefix_caching"] is (arm == "apc_on"), "APC setting differs")
            except Exception as exc:
                cell_errors.append(f"{type(exc).__name__}: {exc}")
            receipt = dict(status="COMPLETE" if not cell_errors else "INCOMPLETE",
                arm=arm, errors=cell_errors, child_lifecycle=life, gpu_before=before, gpu_after=post,
                started_utc=started, ended_utc=resource.stamp(), freeze_sha256=resource.sha(FREEZE))
            resource.atomic_new(ROOT / (arm + "-receipt.json"), receipt)
            print(json.dumps(dict(arm=arm, status=receipt["status"], errors=cell_errors)), flush=True)
            if cell_errors:
                errors.extend(cell_errors)
                break
            completed.append(arm)
        receipt = dict(status="COMPLETE" if not errors and completed == list(ORDER) else "INCOMPLETE",
            completed=completed, order=list(ORDER), errors=errors,
            ended_utc=resource.stamp(), freeze_sha256=resource.sha(FREEZE),
            scope="Native APC allocation/sharing qualification; no timing or new controller claim")
        resource.atomic_new(ROOT / "pair-receipt.json", receipt)
        print(json.dumps(receipt), flush=True)
        return 0 if receipt["status"] == "COMPLETE" else 70
    finally:
        os.close(fd)


if __name__ == "__main__":
    sys.exit(main())

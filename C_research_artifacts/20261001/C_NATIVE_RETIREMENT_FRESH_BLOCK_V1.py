#!/usr/bin/env python3
"""One frozen six-cell fresh-input block on the user-supplied replacement GPU."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import stat
import sys

import C_NATIVE_RECOMPUTE_PILOT_LAUNCHER_V3 as pilot

pilot.GPU_UUID = "GPU-e4434c32-c4a4-2b81-55fa-271af38f3c36"
pilot.LOCK_INODE = "2304:29005388732"
BASE = pilot.BASE
PARENT = pilot.PARENT
ROOT = BASE / "c-native-retirement-fresh-v1"
ORDER = (("native_1", "native_full"),
         ("bound_fifo_1", "native_max_bound"),
         ("retirement_1", "native_retirement"),
         ("retirement_2", "native_retirement"),
         ("bound_fifo_2", "native_max_bound"),
         ("native_2", "native_full"))
INPUTS = BASE / "20261001_c_retirement_fresh_inputs_v1"
INPUT_SHA = {"config.json": "634e7715618879775daf312fc98d77c1b25f4cd8e97007c101604ded317738a3",
             "workload.json": "9576395c9540c71be86dd182df03ab6d39cf1c961a6a24dc851d40991cf0a985",
             "INPUT_STATS.json": "797ccd408a60ee01bed2ac370f51f7145fa4f9ea3bcaecc5a1a4535afd034ac1"}
CELL_SHA = "90c0065e8f004f98c636c1453caacaf678033144b96e9e74a708943e8b13703b"
CONTRACT_SHA = "6896c987f30106deffed0bfce22b98728cf4d78871e41ba45e06b0c1383d72a8"
HELPER_SHA = "3c64e4866a1c3e28082158bf38b64f6a1de4517d1a33a042fb7a2b9d83322b42"
PARENT_SHA = "acdf36222489773ab1fc3a4b6580adcb5951b08af4fa044cfe2c66fb9e850792"
BOUND_CELL_SHA = CELL_SHA
BOUND_ADAPTER_SHA = "b6a601d1edc05804bee70d7f6ecdb2f00e1dd017b2280163df2db88250fb50ee"
RETIREMENT_CELL_SHA = CELL_SHA
RETIREMENT_ADAPTER_SHA = "261b8f2697042f75130203019e8722c424cca3e71d82ac309c78f29ec1bc8eb6"
BOUND_SOURCE = BASE / "C_NATIVE_RETIREMENT_FRESH_CELL_V1.py"
BOUND_ADAPTER_SOURCE = BASE / "C_NATIVE_MAX_BOUND_ADMISSION.py"
RETIREMENT_SOURCE = BASE / "C_NATIVE_RETIREMENT_FRESH_CELL_V1.py"
RETIREMENT_ADAPTER_SOURCE = BASE / "C_NATIVE_RETIREMENT_ADMISSION_V1.py"
CELL_SOURCES = {"native_full": (BOUND_SOURCE, CELL_SHA), "native_max_bound": (BOUND_SOURCE, BOUND_CELL_SHA),
                "native_retirement": (RETIREMENT_SOURCE, RETIREMENT_CELL_SHA)}
MIN_FREE = 512 * 1024**2
MIN_START_FREE = 900 * 1024**2

def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)

def idle(state: dict) -> bool:
    return (state.get("gpu_uuid") == pilot.GPU_UUID
            and type(state.get("used_mib")) is int and state["used_mib"] <= 64
            and state.get("compute_processes") == [])

def environment() -> dict:
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=pilot.GPU_UUID, HF_HOME=str(pilot.HF_HOME),
               HF_HUB_OFFLINE="1", OMP_NUM_THREADS="25", MKL_NUM_THREADS="25",
               PYTHONDONTWRITEBYTECODE="1", PYTHONOPTIMIZE="0",
               VLLM_USE_FLASHINFER_SAMPLER="0", MOE_GPU_LOCK=str(pilot.LOCK))
    return env

def precheck() -> None:
    require(all(pilot.sha(INPUTS / name) == digest for name, digest in INPUT_SHA.items()),
            "frozen fresh input changed")
    require(pilot.sha(BASE / "C_NATIVE_RETIREMENT_FRESH_CONTRACT_V1.py") == CONTRACT_SHA,
            "fresh input measurement contract changed")
    require(pilot.sha(Path(pilot.__file__)) == HELPER_SHA,
            "V3 launcher helper source changed")
    require(pilot.PYTHON.is_file() and pilot.HF_HOME.is_dir()
            and pilot.sha(PARENT / "manifest.json") == PARENT_SHA
            and pilot.sha(BOUND_SOURCE) == BOUND_CELL_SHA
            and pilot.sha(BOUND_ADAPTER_SOURCE) == BOUND_ADAPTER_SHA
            and pilot.sha(RETIREMENT_SOURCE) == RETIREMENT_CELL_SHA
            and pilot.sha(RETIREMENT_ADAPTER_SOURCE) == RETIREMENT_ADAPTER_SHA,
            "frozen interpreter, cache, parent or cell source changed")
    require(not (ROOT / "block-start.json").exists()
            and not (ROOT / "block-start.json").is_symlink(),
            "matched block already started; no retry")
    require(not any((ROOT / label).exists() or (ROOT / label).is_symlink()
                    for label, _ in ORDER), "existing cell result; no retry")
    require(not ROOT.is_symlink(), "matched root must not be a symlink")
    lock = pilot.LOCK.lstat()
    require(stat.S_ISREG(lock.st_mode)
            and f"{lock.st_dev}:{lock.st_ino}" == pilot.LOCK_INODE,
            "shared GPU lock inode changed")

def qualify_cell(cell: Path, arm: str, code: int, timed_out: bool, post: dict) -> dict:
    original_cell, original_sha = pilot.CELL, pilot.CELL_SOURCE_SHA
    try:
        pilot.CELL = cell
        pilot.CELL_SOURCE_SHA = CELL_SOURCES[arm][1]
        result = pilot.qualify(code, timed_out, post)
    finally:
        pilot.CELL, pilot.CELL_SOURCE_SHA = original_cell, original_sha
    require(result["status"] == "ACCEPTED_PILOT_ONLY",
            f"{arm} cell qualification failed: {result['errors']}")
    if arm == "native_full":
        reservation_summary = {"policy": "native_no_offload"}
    elif arm == "native_max_bound":
        gate = pilot.read(cell / "max-bound-admission.json")
        require(gate.get("status") == "DRAINED" and gate.get("drained") is True
                and gate.get("admitted") == gate.get("released") == 128
                and gate.get("preemptions") == 0 and gate.get("violations") == []
                and type(gate.get("reserved_blocks_peak")) is int
                and 0 <= gate["reserved_blocks_peak"] <= 4096,
                "maximum-bound admission trace did not drain or honor capacity")
        reservation_summary = dict(status=gate["status"],
                                   reserved_blocks_peak=gate["reserved_blocks_peak"],
                                   hold_calls=gate["hold_calls"])
    elif arm == "native_retirement":
        gate = pilot.read(cell / "retirement-envelope.json")
        require(gate.get("status") == "DRAINED" and gate.get("drained") is True
                and gate.get("policy") == "conditional_cap_retirement_fifo"
                and gate.get("admitted") == gate.get("released") == 128
                and gate.get("preemptions") == 0 and gate.get("violations") == []
                and type(gate.get("full_bound_sum_peak")) is int
                and gate["full_bound_sum_peak"] >= 0
                and type(gate.get("envelope_peak_blocks")) is int
                and 0 <= gate["envelope_peak_blocks"] <= 4096
                and type(gate.get("physical_blocks_peak")) is int
                and 0 <= gate["physical_blocks_peak"] <= 4096
                and type(gate.get("incremental_admissions")) is int
                and gate["incremental_admissions"] >= 0,
                "conditional retirement trace did not drain or honor capacity")
        reservation_summary = {key: gate[key] for key in
            ("status", "policy", "admitted", "released", "preemptions",
             "full_bound_sum_peak", "envelope_peak_blocks", "physical_blocks_peak",
             "incremental_admissions", "decision_seconds")}
    else:
        raise ValueError(f"unknown arm: {arm}")
    return dict(output_tokens=result["output_tokens"],
                raw_sha256=result["raw_sha256"],
                pressure_qualified=result["pressure_qualified"],
                reservation_summary=reservation_summary)

def command(arm: str, cell: Path) -> tuple[list[str], Path]:
    source = CELL_SOURCES[arm][0]
    return ([str(pilot.PYTHON), str(source),
             "--parent-package", str(PARENT), "--inputs-dir", str(INPUTS),
             "--arm", arm, "--output-dir", str(cell)], BASE)

def run_cell(index: int, label: str, arm: str, fd: int, env: dict) -> dict:
    free = shutil.disk_usage(BASE).free
    require(free >= MIN_FREE, f"before {label}: less than 512 MiB free; preserve block")
    before = pilot.probe()
    require(idle(before), f"before {label}: GPU is not idle")
    folder = ROOT / label
    folder.mkdir(exist_ok=False)
    cell = folder / arm
    cmd, cwd = command(arm, cell)
    log_path = folder / "child.log"
    started = pilot.stamp()
    with log_path.open("x") as log:
        life = pilot.run_owned_child(cmd, cwd=cwd, env=env, log=log,
                                     lock_fd=fd, timeout_s=900)
        log.flush()
        os.fsync(log.fileno())
    try:
        post = pilot.probe()
    except Exception as exc:
        post = dict(probe_error=f"{type(exc).__name__}: {exc}")
    errors = []
    if not (life["exit_code"] == life["child_returncode"] == 0
            and life["child_reaped"] is True and life["timed_out"] is False
            and life["launcher_error"] is None and life["interruption_signal"] is None):
        errors.append("child exit, timeout, or lifecycle unqualified")
    if not idle(post):
        errors.append("GPU did not drain after child exit")
    qualified = {}
    if not errors:
        try:
            qualified = qualify_cell(cell, arm, life["exit_code"], life["timed_out"], post)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    receipt = dict(schema="c-native-retirement-fresh-block-cell-v1",
        sequence_index=index, label=label, arm=arm,
        status="ACCEPTED_FOR_MATCHED_BLOCK" if not errors else "INCOMPLETE",
        errors=errors, started_utc=started, ended_utc=pilot.stamp(),
        child_lifecycle=life, gpu_before=before, gpu_after=post,
        disk_free_before_bytes=free, log_sha256=pilot.sha(log_path),
        runner_source_sha256=pilot.sha(Path(__file__)),
        launcher_helper_sha256=HELPER_SHA, parent_manifest_sha256=PARENT_SHA,
        bound_cell_source_sha256=BOUND_CELL_SHA,
        bound_adapter_source_sha256=BOUND_ADAPTER_SHA,
        retirement_cell_source_sha256=RETIREMENT_CELL_SHA,
        retirement_adapter_source_sha256=RETIREMENT_ADAPTER_SHA,
        lock_inode=pilot.LOCK_INODE, gpu_uuid=pilot.GPU_UUID,
        input_file_sha256=INPUT_SHA, fresh_contract_sha256=CONTRACT_SHA,
        **qualified)
    pilot.atomic_new(folder / "launcher-receipt.json", receipt)
    print(json.dumps(dict(label=label, status=receipt["status"],
                          output_tokens=qualified.get("output_tokens"))), flush=True)
    return receipt

def main() -> int:
    precheck()
    env = environment()
    fd = os.open(pilot.LOCK, os.O_RDWR | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        require(stat.S_ISREG(opened.st_mode)
                and f"{opened.st_dev}:{opened.st_ino}" == pilot.LOCK_INODE,
                "opened shared GPU lock inode changed")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps(dict(status="GPU_DEFERRED", reason="shared lock busy")), flush=True)
            return 75
        require(shutil.disk_usage(BASE).free >= MIN_START_FREE,
                "under 900 MiB free on data disk; block not started")
        ROOT.mkdir(parents=True, exist_ok=True)
        pilot.atomic_new(ROOT / "block-start.json", dict(
            schema="c-native-retirement-fresh-block-v1", started_utc=pilot.stamp(),
            order=[label for label, _ in ORDER],
            runner_source_sha256=pilot.sha(Path(__file__)),
            launcher_helper_sha256=HELPER_SHA, parent_manifest_sha256=PARENT_SHA,
            lock_inode=pilot.LOCK_INODE, gpu_uuid=pilot.GPU_UUID,
            input_file_sha256=INPUT_SHA, fresh_contract_sha256=CONTRACT_SHA))
        completed = []
        error = None
        try:
            for index, (label, arm) in enumerate(ORDER):
                receipt = run_cell(index, label, arm, fd, env)
                require(receipt["status"] == "ACCEPTED_FOR_MATCHED_BLOCK",
                        f"{label}: cell incomplete; block stopped")
                completed.append(label)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
        result = dict(schema="c-native-retirement-fresh-block-v1",
            status="COMPLETE" if error is None else "INCOMPLETE",
            order=[label for label, _ in ORDER], completed=completed,
            error=error, ended_utc=pilot.stamp(),
            runner_source_sha256=pilot.sha(Path(__file__)))
        pilot.atomic_new(ROOT / "block-receipt.json", result)
        print(json.dumps(dict(status=result["status"], completed=completed, error=error)), flush=True)
        return 0 if error is None else 70
    finally:
        os.close(fd)

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"MATCHED_BLOCK_PRECHECK_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(70)

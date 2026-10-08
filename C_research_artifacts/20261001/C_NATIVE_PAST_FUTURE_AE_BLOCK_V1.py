#!/usr/bin/env python3
"""One native/Past-Future/Past-Future/native block in the batch4096 domain."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import stat
import sys

import C_NATIVE_PAST_FUTURE_AE_LAUNCHER_V1 as pilot

BASE, PARENT = pilot.BASE, pilot.PARENT
ROOT = BASE / "c-native-past-future-ae-matched-dev-v1"
ORDER = (("native_1", "native_full"),
         ("past_future_1", "native_past_future_ae"),
         ("past_future_2", "native_past_future_ae"),
         ("native_2", "native_full"))
INPUTS = BASE / "20261001_c_retirement_fresh_inputs_v1"
INPUT_SHA = {"config.json": "634e7715618879775daf312fc98d77c1b25f4cd8e97007c101604ded317738a3",
             "workload.json": "9576395c9540c71be86dd182df03ab6d39cf1c961a6a24dc851d40991cf0a985",
             "INPUT_STATS.json": "797ccd408a60ee01bed2ac370f51f7145fa4f9ea3bcaecc5a1a4535afd034ac1"}
CELL_SHA = "b9d1a9609cd355449a31c9b0800ab0d0df27960787381fe5b86d8c0e0de22cd3"
HELPER_SHA = "9a2ae6b8eb866593d3914bc3d32f54cf259b6589ec6df302470fd724cd757dfb"
PARENT_SHA = "acdf36222489773ab1fc3a4b6580adcb5951b08af4fa044cfe2c66fb9e850792"
PF_ADAPTER_SHA = "28001321554a72bf1a451d26e15e6d4e2192c26d074891c6efd9bf5091ed3dc7"
PF_PREDICTOR_SHA = "e37d50ee262ae3325521bc95edee14079119451cd96f182c8236e905bc9a7773"
CONTRACT_SHA = "6896c987f30106deffed0bfce22b98728cf4d78871e41ba45e06b0c1383d72a8"
MIN_START_FREE = 900 * 1024**2
MIN_CELL_FREE = 512 * 1024**2


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def idle(state: dict) -> bool:
    return (state.get("gpu_uuid") == pilot.GPU_UUID
            and type(state.get("used_mib")) is int and state["used_mib"] <= 64
            and state.get("compute_processes") == [])


def pilot_gate() -> dict:
    """A valid pilot lifecycle is required; its observed benefit is irrelevant."""
    cell = pilot.CELL
    receipt = pilot.read(cell / "launcher-receipt.json")
    gate = pilot.read(cell / "past-future-ae.json")
    life = receipt.get("child_lifecycle")
    require(receipt.get("status") == "ACCEPTED_PILOT_ONLY"
            and receipt.get("errors") == []
            and receipt.get("cell_source_sha256") == CELL_SHA
            and receipt.get("parent_manifest_sha256") == PARENT_SHA
            and receipt.get("launcher_source_sha256") == HELPER_SHA
            and type(life) is dict and life.get("exit_code") == life.get("child_returncode") == 0
            and life.get("child_reaped") is True and life.get("timed_out") is False
            and life.get("launcher_error") is None and life.get("interruption_signal") is None,
            "author-AE pilot did not qualify and finish under its frozen launcher")
    require(gate.get("status") == "DRAINED" and gate.get("drained") is True
            and gate.get("admitted") == gate.get("completed") == 128
            and gate.get("violations") == []
            and gate.get("policy") == "author_AE_statistical_peak_core_native_batch4096"
            and type(gate.get("physical_blocks_peak")) is int
            and 0 <= gate["physical_blocks_peak"] <= 4096,
            "author-AE pilot admission trace did not drain cleanly")
    return dict(pilot_receipt_sha256=pilot.sha(cell / "launcher-receipt.json"),
                pilot_gate_sha256=pilot.sha(cell / "past-future-ae.json"),
                pilot_preemptions=gate.get("preemptions"))


def precheck() -> dict:
    require(pilot.PARENT == PARENT and pilot.CELL_SOURCE_SHA == CELL_SHA
            and pilot.LOCK_INODE == "2304:29005388732"
            and pilot.GPU_UUID == "GPU-e4434c32-c4a4-2b81-55fa-271af38f3c36",
            "frozen launcher location/GPU identity differs")
    require(pilot.sha(Path(pilot.__file__)) == HELPER_SHA
            and pilot.sha(pilot.CELL_SOURCE) == CELL_SHA
            and pilot.sha(PARENT / "manifest.json") == PARENT_SHA
            and pilot.PYTHON.is_file() and pilot.HF_HOME.is_dir(),
            "frozen helper/cell/parent/interpreter/cache changed")
    require(all(pilot.sha(INPUTS / name) == digest for name, digest in INPUT_SHA.items()),
            "viewed 128-request input bytes changed")
    require(all(pilot.sha(BASE / name) == digest for name, digest in pilot.EXTRA_SOURCES.items())
            and pilot.EXTRA_SOURCES.get("C_NATIVE_PAST_FUTURE_AE_ADMISSION_V1.py") == PF_ADAPTER_SHA
            and pilot.EXTRA_SOURCES.get("C_PAST_FUTURE_AE_PREDICTOR_V1.py") == PF_PREDICTOR_SHA
            and pilot.EXTRA_SOURCES.get("C_NATIVE_RETIREMENT_FRESH_CONTRACT_V1.py") == CONTRACT_SHA,
            "frozen admission, predictor, or contract source changed")
    require(not (ROOT / "block-start.json").exists()
            and not (ROOT / "block-start.json").is_symlink()
            and not any((ROOT / label).exists() or (ROOT / label).is_symlink()
                        for label, _ in ORDER)
            and not ROOT.is_symlink(), "matched block already started; no retry")
    info = pilot.LOCK.lstat()
    require(stat.S_ISREG(info.st_mode)
            and f"{info.st_dev}:{info.st_ino}" == pilot.LOCK_INODE,
            "shared GPU lock inode changed")
    return pilot_gate()


def environment() -> dict:
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=pilot.GPU_UUID, HF_HOME=str(pilot.HF_HOME),
        HF_HUB_OFFLINE="1", OMP_NUM_THREADS="25", MKL_NUM_THREADS="25",
        PYTHONDONTWRITEBYTECODE="1", PYTHONOPTIMIZE="0",
        VLLM_USE_FLASHINFER_SAMPLER="0", MOE_GPU_LOCK=str(pilot.LOCK))
    return env


def qualify_cell(cell: Path, arm: str, code: int, timed_out: bool, post: dict) -> dict:
    previous = pilot.CELL
    try:
        pilot.CELL = cell
        result = pilot.qualify(code, timed_out, post)
    finally:
        pilot.CELL = previous
    require(result["status"] == "ACCEPTED_PILOT_ONLY",
            f"{arm} frozen cell qualification failed: {result['errors']}")
    config, args, env = (pilot.read(cell / name) for name in
                         ("config.json", "engine_args.json", "environment.json"))
    source = pilot.read(cell / "source-receipt.json")
    raw = pilot.read(cell / "raw.json")
    require(config.get("reservation_policy") == arm
            and config.get("max_num_batched_tokens") == 4096
            and config.get("policy_seed") == 20261001
            and config.get("seed") == 20260905
            and config.get("arrival_regime") == "poisson_v1"
            and config.get("arrival_rate_per_s") == 5.0
            and config.get("arrival_span_s") == 22.682329
            and config.get("prompt_tokens") == 3072
            and args.get("max_num_batched_tokens") == 4096
            and args.get("long_prefill_token_threshold") == 0
            and args.get("max_num_seqs") == 32
            and args.get("max_model_len") == 4096
            and args.get("seed") == 20260905
            and args.get("async_scheduling") is False
            and args.get("enable_prefix_caching") is False
            and raw.get("regime") == "poisson_v1"
            and raw.get("arrival_scale") == 1.0
            and source.get("input_sha256") == INPUT_SHA
            and config.get("development_input_receipt", {}).get("input_sha256") == INPUT_SHA
            and env.get("fresh_input_sha256") == INPUT_SHA
            and env.get("past_future_adapter_sha256") == PF_ADAPTER_SHA
            and env.get("past_future_predictor_sha256") == PF_PREDICTOR_SHA,
            "same-input batch4096/seed/source qualification differs")
    if arm == "native_full":
        require(not (cell / "past-future-ae.json").exists(),
                "native arm unexpectedly installed AE gate")
        policy = dict(policy="native_full", actual_preemption_count=
                      raw.get("actual_preemption_count"))
    elif arm == "native_past_future_ae":
        gate = pilot.read(cell / "past-future-ae.json")
        require(gate.get("status") == "DRAINED" and gate.get("drained") is True
                and gate.get("policy") == "author_AE_statistical_peak_core_native_batch4096"
                and gate.get("admitted") == gate.get("completed") == 128
                and gate.get("violations") == []
                and type(gate.get("physical_blocks_peak")) is int
                and 0 <= gate["physical_blocks_peak"] <= 4096
                and type(raw.get("actual_preemption_count")) is int
                and gate.get("preemptions") == raw["actual_preemption_count"],
                "AE gate failed capacity, completion, or state contract")
        policy = {key: gate[key] for key in
                  ("policy", "admitted", "completed", "resumptions", "preemptions",
                   "hold_calls", "decision_seconds", "physical_blocks_peak")}
        policy["gate_sha256"] = pilot.sha(cell / "past-future-ae.json")
    else:
        raise ValueError(f"unsupported arm: {arm}")
    return dict(output_tokens=result["output_tokens"],
                raw_sha256=result["raw_sha256"],
                pressure_qualified=result["pressure_qualified"], policy_summary=policy)


def run_cell(index: int, label: str, arm: str, fd: int, env: dict,
             pilot_provenance: dict) -> dict:
    free = shutil.disk_usage(BASE).free
    require(free >= MIN_CELL_FREE, f"before {label}: under 512 MiB data-disk free")
    before = pilot.probe()
    require(idle(before), f"before {label}: GPU not idle")
    folder = ROOT / label
    folder.mkdir(exist_ok=False)
    cell = folder / arm
    command = [str(pilot.PYTHON), str(pilot.CELL_SOURCE),
               "--parent-package", str(PARENT), "--inputs-dir", str(INPUTS),
               "--arm", arm, "--output-dir", str(cell)]
    log_path = folder / "child.log"
    started = pilot.stamp()
    with log_path.open("x") as log:
        life = pilot.run_owned_child(command, cwd=BASE, env=env, log=log,
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
        errors.append("owned child exit/timeout/lifetime unqualified")
    if not idle(post):
        errors.append("GPU did not drain after child exit")
    qualified = {}
    if not errors:
        try:
            qualified = qualify_cell(cell, arm, life["exit_code"], life["timed_out"], post)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    receipt = dict(schema="c-native-past-future-ae-matched-cell-v1",
        sequence_index=index, label=label, arm=arm,
        status="ACCEPTED_FOR_MATCHED_BLOCK" if not errors else "INCOMPLETE",
        errors=errors, started_utc=started, ended_utc=pilot.stamp(),
        child_lifecycle=life, gpu_before=before, gpu_after=post,
        disk_free_before_bytes=free, log_sha256=pilot.sha(log_path),
        runner_source_sha256=pilot.sha(Path(__file__)),
        helper_source_sha256=HELPER_SHA, cell_source_sha256=CELL_SHA,
        parent_manifest_sha256=PARENT_SHA, extra_source_sha256=pilot.EXTRA_SOURCES,
        input_file_sha256=INPUT_SHA, lock_inode=pilot.LOCK_INODE,
        gpu_uuid=pilot.GPU_UUID, batch_tokens=4096, arrival_scale=1.0,
        pilot_gate=pilot_provenance, **qualified)
    pilot.atomic_new(folder / "launcher-receipt.json", receipt)
    print(json.dumps(dict(label=label, status=receipt["status"],
                          output_tokens=qualified.get("output_tokens"))), flush=True)
    return receipt


def main() -> int:
    pilot_provenance = precheck()
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
                "under 900 MiB free on data disk; no block started")
        ROOT.mkdir(parents=True, exist_ok=True)
        pilot.atomic_new(ROOT / "block-start.json", dict(
            schema="c-native-past-future-ae-matched-block-v1",
            started_utc=pilot.stamp(), order=[label for label, _ in ORDER],
            runner_source_sha256=pilot.sha(Path(__file__)),
            helper_source_sha256=HELPER_SHA, cell_source_sha256=CELL_SHA,
            parent_manifest_sha256=PARENT_SHA, extra_source_sha256=pilot.EXTRA_SOURCES,
            input_file_sha256=INPUT_SHA, lock_inode=pilot.LOCK_INODE,
            gpu_uuid=pilot.GPU_UUID, batch_tokens=4096, arrival_scale=1.0,
            pilot_gate=pilot_provenance))
        completed = []
        error = None
        try:
            for index, (label, arm) in enumerate(ORDER):
                receipt = run_cell(index, label, arm, fd, env, pilot_provenance)
                require(receipt["status"] == "ACCEPTED_FOR_MATCHED_BLOCK",
                        f"{label}: invalid cell; matched block stopped")
                completed.append(label)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
        result = dict(schema="c-native-past-future-ae-matched-block-v1",
            status="COMPLETE" if error is None else "INCOMPLETE",
            order=[label for label, _ in ORDER], completed=completed,
            error=error, ended_utc=pilot.stamp(),
            runner_source_sha256=pilot.sha(Path(__file__)))
        pilot.atomic_new(ROOT / "block-receipt.json", result)
        print(json.dumps(dict(status=result["status"], completed=completed,
                              error=error)), flush=True)
        return 0 if error is None else 70
    finally:
        # Closing only after each owned child was reaped retains lock ownership.
        os.close(fd)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"AE_MATCHED_BLOCK_PRECHECK_ERROR: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        raise SystemExit(70)

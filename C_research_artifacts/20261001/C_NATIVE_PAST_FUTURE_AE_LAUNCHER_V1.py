#!/usr/bin/env python3
"""Bounded shared-lock author-AE native-port pilot; explicit batch4096 regime."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys

BASE = Path("/root/autodl-tmp/c-research-20260930")
PARENT = BASE / "20260930_c_sustained_baselines_dev_v1"
CELL_SOURCE = BASE / "C_NATIVE_PAST_FUTURE_AE_CELL_V1.py"
ROOT = BASE / "c-native-past-future-ae-pilot-v1"
CELL = ROOT / "native_past_future_ae_1"
PYTHON = Path("/root/autodl-tmp/c-vllm-v026-venv/bin/python")
HF_HOME = BASE / "hf-cache"
LOCK = Path("/root/autodl-tmp/moe-research-gpu.lock")
LOCK_INODE = "2304:29005388732"
GPU_UUID = "GPU-e4434c32-c4a4-2b81-55fa-271af38f3c36"
PARENT_SHA = "acdf36222489773ab1fc3a4b6580adcb5951b08af4fa044cfe2c66fb9e850792"
CELL_SOURCE_SHA = "b9d1a9609cd355449a31c9b0800ab0d0df27960787381fe5b86d8c0e0de22cd3"

EXTRA_SOURCES = {
    "C_NATIVE_PAST_FUTURE_AE_ADMISSION_V1.py": "28001321554a72bf1a451d26e15e6d4e2192c26d074891c6efd9bf5091ed3dc7",
    "C_PAST_FUTURE_AE_PREDICTOR_V1.py": "e37d50ee262ae3325521bc95edee14079119451cd96f182c8236e905bc9a7773",
    "C_NATIVE_MAX_BOUND_ADMISSION.py": "b6a601d1edc05804bee70d7f6ecdb2f00e1dd017b2280163df2db88250fb50ee",
    "C_NATIVE_RETIREMENT_ADMISSION_V1.py": "261b8f2697042f75130203019e8722c424cca3e71d82ac309c78f29ec1bc8eb6",
    "C_NATIVE_RETIREMENT_FRESH_CONTRACT_V1.py": "6896c987f30106deffed0bfce22b98728cf4d78871e41ba45e06b0c1383d72a8",
}

class LaunchInterrupted(Exception):
    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"launcher received signal {signum}")


def run_owned_child(command: list[str], *, cwd: Path, env: dict, log,
                    lock_fd: int, timeout_s: float = 900, grace_s: float = 5) -> dict:
    """Reap our child before returning; child retains flock if parent is killed."""
    watched = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    prior = {sig: signal.getsignal(sig) for sig in watched}
    child = None
    spawning = True
    pending_signal = None
    result = dict(exit_code=70, timed_out=False, interruption_signal=None,
                  launcher_error=None, child_pid=None, child_returncode=None,
                  child_reaped=False, lock_fd_inherited=True)

    def interrupt(signum, frame):
        nonlocal pending_signal
        pending_signal = signum
        # During Popen, wait until its child handle has been assigned.
        if not spawning:
            raise LaunchInterrupted(signum)

    try:
        for sig in watched:
            signal.signal(sig, interrupt)
        if pending_signal is not None:
            raise LaunchInterrupted(pending_signal)
        child = subprocess.Popen(command, cwd=cwd, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, pass_fds=(lock_fd,))
        result["child_pid"] = child.pid
        spawning = False
        if pending_signal is not None:
            raise LaunchInterrupted(pending_signal)
        result["exit_code"] = child.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        result.update(exit_code=124, timed_out=True)
    except LaunchInterrupted as exc:
        result.update(exit_code=128 + exc.signum, interruption_signal=exc.signum)
    except BaseException as exc:
        result["launcher_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # Repeated termination signals cannot interrupt owned-child cleanup.
        for sig in watched:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if child is not None:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=grace_s)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        # Never unlock while an owned, un-reaped child remains.
                        child.wait()
                else:
                    child.wait()
                result.update(child_returncode=child.returncode, child_reaped=True)
        finally:
            for sig, handler in prior.items():
                signal.signal(sig, handler)
    return result


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if type(value) is not dict:
        raise ValueError(f"JSON object required: {path}")
    return value


def atomic_new(path: Path, value: dict) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temp, path)
    finally:
        temp.unlink()


def probe() -> dict:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used",
        "--format=csv,noheader,nounits"], capture_output=True, text=True,
        timeout=15, check=True)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits"], capture_output=True, text=True,
        timeout=15, check=True)
    rows = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError("expected one physical GPU")
    parts = [piece.strip() for piece in rows[0].split(",")]
    if len(parts) != 2 or parts[0] != GPU_UUID:
        raise RuntimeError("approved GPU UUID changed")
    return dict(gpu_uuid=parts[0], used_mib=int(parts[1]),
                compute_processes=[line.strip() for line in apps.stdout.splitlines()
                                   if line.strip()])


def qualify(code: int, timed_out: bool, post: dict) -> dict:
    errors = []
    if code != 0 or timed_out:
        errors.append(f"child exit {code}, timeout={timed_out}")
    if post.get("probe_error") or post.get("used_mib", 65) > 64 or post.get("compute_processes", ["unknown"]):
        errors.append("GPU did not drain after child exit")
    required = ("config.json", "engine_args.json", "environment.json",
                "safe-cap-qualification.json", "memory-after-init.json",
                "baseline-result-audit.json", "raw.json", "status.json",
                "runtime-qualification.json", "warmup-native-drain.json", "measurement-native-drain.json")
    docs = {}
    for name in required:
        try:
            docs[name] = read(CELL / name)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    pressure = None
    output_tokens = None
    if len(docs) == len(required):
        c, a, e, q, m, b, raw, status, runtime, warm_drain, final_drain = (docs[name] for name in required)
        if not (c.get("requests") == c.get("cap") * 4 == 128
                and c.get("output_mode") == "eos" and c.get("output_tokens") == 1024
                and c.get("offload_gib") == 0 and c.get("store_scope") == "none"
                and c.get("fixed_kv_cache_memory_bytes") == 4097 * 2 * 1024 * 1024
                and a.get("model") == "allenai/OLMoE-1B-7B-0924"
                and a.get("max_num_seqs") == 32 and a.get("max_num_batched_tokens") == 4096
                and a.get("scheduler_reserve_full_isl") is True
                and "kv_offloading_size" not in a and "kv_transfer_config" not in a
                and e.get("vllm") == "0.26.0"
                and e.get("parent_manifest_sha256") == PARENT_SHA
                and e.get("pilot_source_sha256") == CELL_SOURCE_SHA
                and q.get("status") == "QUALIFIED" and q.get("usable_blocks") == 4096
                and m.get("kv_storage_bytes") == 4097 * 2 * 1024 * 1024
                and b.get("status") == "PASS" and b.get("requests") == 128
                and status.get("status") == raw.get("status") == "COMPLETE"
                and status.get("requests_completed") == 128
                and len(raw.get("requests", [])) == 128
                and all(r.get("status") == "completed" for r in raw["requests"])):
            errors.append("native no-connector physical/source/request contract differs")
        if not (runtime.get("status") == "QUALIFIED"
                and runtime.get("observed") == runtime.get("expected")
                and len(runtime.get("observed", {}).get("vllm_source_sha256", {})) == 8):
            errors.append("runtime identity qualification differs")
        for name, drain in (("warmup", warm_drain), ("measurement", final_drain)):
            if not (drain.get("status") == "QUALIFIED"
                    and drain.get("scheduler_connector_absent") is True
                    and drain.get("kv_transfer_config_absent") is True
                    and drain.get("kv_offloading_size") is None
                    and drain.get("has_unfinished_requests") is False
                    and drain.get("scheduler_request_count") == drain.get("running") == drain.get("waiting") == 0
                    and drain.get("total_blocks") == 4097 and drain.get("free_blocks") == 4096):
                errors.append(f"{name} native request/KV drain qualification differs")
        pressure = (type(raw.get("actual_preemption_count")) is int
                    and raw["actual_preemption_count"] > 0
                    and any(s.get("running_before") == 32
                            and type(s.get("waiting_before")) is int
                            and s["waiting_before"] > 0
                            for s in raw.get("scheduler_steps", [])))
        output_tokens = b.get("output_tokens")
    return dict(status="ACCEPTED_PILOT_ONLY" if not errors else "UNQUALIFIED",
                errors=errors, pressure_qualified=pressure, output_tokens=output_tokens,
                raw_sha256=sha(CELL / "raw.json") if (CELL / "raw.json").is_file() else None)


def main() -> int:
    for name, digest in EXTRA_SOURCES.items():
        if sha(BASE / name) != digest:
            raise RuntimeError(f"frozen baseline source changed: {name}")
    if not (PYTHON.is_file() and HF_HOME.is_dir() and CELL_SOURCE.is_file()
            and sha(PARENT / "manifest.json") == PARENT_SHA
            and sha(CELL_SOURCE) == CELL_SOURCE_SHA):
        raise RuntimeError("frozen interpreter, cache, parent or pilot source changed")
    if CELL.exists() or CELL.is_symlink():
        raise FileExistsError("immutable pilot cell already exists")
    ROOT.mkdir(parents=True, exist_ok=True)
    attempts = ROOT / "attempts"
    attempts.mkdir(exist_ok=True)
    identifier = f"{stamp().replace(':', '').replace('.', '')}-{os.getpid()}"
    attempt = attempts / identifier
    log_path = attempt.with_suffix(".log")
    receipt_path = attempt.with_suffix(".json")
    info = LOCK.lstat()
    if not stat.S_ISREG(info.st_mode) or f"{info.st_dev}:{info.st_ino}" != LOCK_INODE:
        raise RuntimeError("shared GPU lock inode changed")
    fd = os.open(LOCK, os.O_RDWR | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or f"{opened.st_dev}:{opened.st_ino}" != LOCK_INODE:
            raise RuntimeError("opened shared GPU lock inode changed")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            atomic_new(receipt_path, dict(status="GPU_DEFERRED", started_utc=stamp(),
                reason="shared physical GPU lock busy; no child started",
                lock_inode=LOCK_INODE, gpu_uuid=GPU_UUID,
                cell_source_sha256=CELL_SOURCE_SHA, parent_manifest_sha256=PARENT_SHA))
            print(json.dumps(dict(status="GPU_DEFERRED", receipt=str(receipt_path))), flush=True)
            return 75
        before = probe()
        if before["used_mib"] > 64 or before["compute_processes"]:
            raise RuntimeError("GPU was occupied after lock acquisition")
        if shutil.disk_usage(BASE).free < 512 * 1024**2:
            raise RuntimeError("under 512 MiB free on data disk; child not started")
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=GPU_UUID, HF_HOME=str(HF_HOME),
            OMP_NUM_THREADS="25", MKL_NUM_THREADS="25",
            HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONOPTIMIZE="0",
            VLLM_USE_FLASHINFER_SAMPLER="0")
        cmd = [str(PYTHON), str(CELL_SOURCE), "--parent-package", str(PARENT),
               "--output-dir", str(CELL)]
        started = stamp()
        with log_path.open("x") as log:
            lifecycle = run_owned_child(cmd, cwd=BASE, env=env, log=log, lock_fd=fd)
            log.flush()
            os.fsync(log.fileno())
        code, timed_out = lifecycle["exit_code"], lifecycle["timed_out"]
        try:
            post = probe()
        except Exception as exc:
            post = dict(probe_error=f"{type(exc).__name__}: {exc}")
        result = qualify(code, timed_out, post)
        if not lifecycle["child_reaped"] or lifecycle["launcher_error"]:
            result["status"] = "UNQUALIFIED"
            result["errors"].append("child launch/lifetime did not complete normally")
        receipt = dict(schema="c-native-past-future-ae-pilot-launch-v1", started_utc=started,
            ended_utc=stamp(), child_exit_code=code, timed_out=timed_out,
            child_lifecycle=lifecycle,
            before_gpu=before, post_exit_gpu=post, cell_source_sha256=CELL_SOURCE_SHA,
            parent_manifest_sha256=PARENT_SHA, launcher_source_sha256=sha(Path(__file__)),
            lock_path=str(LOCK), lock_inode=LOCK_INODE,
            log_sha256=sha(log_path), **result)
        atomic_new(CELL / "launcher-receipt.json" if CELL.is_dir() else receipt_path, receipt)
        print(json.dumps(dict(status=result["status"], pressure_qualified=result["pressure_qualified"],
                              child_exit_code=code, output_tokens=result["output_tokens"])), flush=True)
        return 0 if result["status"] == "ACCEPTED_PILOT_ONLY" else 70
    finally:
        # close releases the flock only when no inherited child descriptor remains.
        # Explicit LOCK_UN would also unlock a still-live child's shared description.
        os.close(fd)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PILOT_LAUNCH_PRECHECK_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(70)

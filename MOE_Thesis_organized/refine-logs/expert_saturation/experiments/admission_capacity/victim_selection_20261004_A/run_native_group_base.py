#!/usr/bin/env python3
"""Reused locked controller; victim wrapper supplies one frozen ABBA plan."""
import argparse
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
import time

MANIFEST = "7c5221fb80402843e10c459471045baf7c2ea50541d0eb4f461c26ce46046c01"
REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
ARMS = ("ordinary", "queue_fund", "native")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value, *, new=False):
    # Only live receipts are updated; all experiment outputs remain immutable.
    with Path(path).open("x" if new else "w") as out:
        json.dump(value, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write("\n")


def command(argv, *, env=None, timeout=30):
    result = subprocess.run(argv, env=env, timeout=timeout, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    require(result.returncode == 0,
            f"Command failed ({result.returncode}): {argv!r}: {result.stderr[-3000:]}")
    return result.stdout.strip()


def gpu_state(uuid):
    gpu = command(["nvidia-smi", "--query-gpu=uuid,name,memory.used,utilization.gpu,driver_version",
                   "--format=csv,noheader,nounits"])
    rows = [line.strip().split(",") for line in gpu.splitlines() if line.strip()]
    require(len(rows) == 1 and rows[0][0].strip() == uuid, "Authorized single GPU differs")
    processes = command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                         "--format=csv,noheader,nounits"])
    fields = [item.strip() for item in rows[0]]
    used, utilization = int(fields[2]), int(fields[3])
    return dict(unix_s=time.time(), uuid=uuid, name=fields[1], memory_used_mib=used,
                utilization_gpu_percent=utilization, driver=fields[4],
                process_rows=processes.splitlines(), empty=not processes and used <= 256 and utilization == 0)


def gpu_boundary(uuid):
    readings = [gpu_state(uuid)]
    if not readings[-1]["empty"] and not readings[-1]["process_rows"]:
        # Counters can retain the just-finished process's sample briefly.
        time.sleep(2)
        readings.append(gpu_state(uuid))
    return dict(readings[-1], boundary_readings=readings)


def own_cgroup_memory_file():
    unified = [line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
               if line.startswith("0::")]
    require(len(unified) == 1 and unified[0].startswith("/") and ".." not in unified[0],
            "Unified own process cgroup unavailable")
    return Path("/sys/fs/cgroup" + unified[0].rstrip("/") + "/memory.max")


def verify_package(package):
    require(package.is_dir() and not package.is_symlink(), "Candidate directory missing or symlink")
    require(digest(package / "manifest.json") == MANIFEST, "Frozen manifest changed")
    manifest = json.loads((package / "manifest.json").read_text())
    for name, expected in manifest.items():
        path = package / name
        require(path.is_file() and not path.is_symlink() and digest(path) == expected,
                f"Frozen payload changed: {name}")
    require(not list(package.glob("launch-once-*")), "Source candidate already consumed a launch")
    config = json.loads((package / "pkg/inputs/config.json").read_text())
    require(config["model"]["revision"] == config["model"]["tokenizer_revision"] == REVISION,
            "Pinned model revision changed")
    return config


def private_env(plan, cache, arm, memory_file):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("A_", "H1_", "VLLM_", "HF_", "HUGGINGFACE_", "TRANSFORMERS_")):
            env.pop(key)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONINSPECT", "PYTHONUSERBASE", "VIRTUAL_ENV"):
        env.pop(key, None)
    paths = {"XDG_CACHE_HOME": "xdg", "TMPDIR": "tmp", "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
             "TRITON_CACHE_DIR": "triton", "VLLM_CACHE_ROOT": "vllm", "TORCH_HOME": "torch",
             "CUDA_CACHE_PATH": "cuda/ComputeCache", "HF_ASSETS_CACHE": "assets",
             "HF_MODULES_CACHE": "transformers_modules", "HF_XET_CACHE": "xet"}
    for key, relative in paths.items():
        destination = cache / relative
        destination.mkdir(parents=True, exist_ok=True)
        env[key] = str(destination)
    env.update(HF_HOME=plan["hf_cache_dir"], HF_HUB_CACHE=plan["hf_cache_dir"] + "/hub",
               HUGGINGFACE_HUB_CACHE=plan["hf_cache_dir"] + "/hub",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
               PYTHONOPTIMIZE="0", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               CUDA_VISIBLE_DEVICES=plan["authorized_gpu_uuid"], VLLM_USE_FLASHINFER_SAMPLER="0",
               A_NATIVE_VICTIM_RULE="tail", A_NATIVE_VICTIM_FULL_RUNNING="off",
               A_NATIVE_VICTIM_CURRENT_GUARD="off", A_SELF_PREEMPT_CONTINUE="off",
               A_NATIVE_CAPACITY_DEFERRAL="off", A_NATIVE_OLDEST_ADMISSION=arm,
               A_NATIVE_OLDEST_REPEAT="0" if arm == "ordinary" else "1",
               H1_AUTHORIZED_GPU_UUID=plan["authorized_gpu_uuid"], H1_PYTHON=plan["python"],
               H1_HF_CACHE_DIR=plan["hf_cache_dir"], H1_LOCK_PATH=plan["lock_path"],
               H1_MAX_WALL_SECONDS=str(plan["per_cell_wall_seconds"]),
               H1_APPROVED_HOST_BYTES=str(plan["approved_host_bytes"]),
               H1_CGROUP_MEMORY_MAX_FILE=str(memory_file), H1_EXPECTED_MANIFEST_SHA256=MANIFEST)
    return env


def verify_model(plan, config, env):
    hub = Path(plan["hf_cache_dir"]) / "hub"
    snapshot = hub / "models--allenai--OLMoE-1B-7B-0924" / "snapshots" / REVISION
    require(snapshot.is_dir(), "Private pinned HF model snapshot missing")
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()))
    require(names, "Model weight index is empty")
    files = {}
    expected_files = plan["model_files"]
    require(set(names) <= set(expected_files), "Weight shards absent from frozen model hashes")
    for name, expected in expected_files.items():
        path = snapshot / name
        resolved = path.resolve(strict=True)
        require(resolved.is_relative_to(hub.resolve()) and resolved.is_file() and resolved.stat().st_size > 0,
                f"Model file absent/outside private cache: {name}")
        sha = digest(resolved)
        require(sha == expected["sha256"], f"Pinned model file identity differs: {name}")
        if expected["bytes"] is not None:
            require(resolved.stat().st_size == expected["bytes"], f"Pinned model file size differs: {name}")
        if path.is_symlink() and len(resolved.name) == 64:
            require(sha == resolved.name, f"HF content-addressed blob mismatch: {name}")
        if name in config["source"]["tokenizer_files_sha256"]:
            require(sha == config["source"]["tokenizer_files_sha256"][name], f"Tokenizer identity differs: {name}")
        files[name] = dict(bytes=resolved.stat().st_size, sha256=sha)
    check = ("import json; from huggingface_hub import snapshot_download; "
             "print(json.dumps(snapshot_download('allenai/OLMoE-1B-7B-0924', "
             f"revision='{REVISION}', local_files_only=True)))")
    resolved = json.loads(command([plan["python"], "-c", check], env=env, timeout=60))
    require(Path(resolved).resolve() == snapshot.resolve(), "Offline HF resolution uses another snapshot")
    return dict(status="PRIVATE_PINNED_SNAPSHOT_COMPLETE_OFFLINE", revision=REVISION,
                snapshot=str(snapshot), files=files,
                scope="Offline pinned revision, exact nine-file SHA256 set, tokenizer hashes, complete weight-index shards; no cross-session verifier")


def run_child(argv, env, log, wall_seconds):
    started = time.monotonic()
    with log.open("x") as stream:
        process = subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, pass_fds=(9,))
        try:
            code = process.wait(timeout=wall_seconds)
            return dict(exit_code=code, timed_out=False, elapsed_wall_s=time.monotonic() - started,
                        child_pid=process.pid)
        except BaseException as error:
            # Only this controller's own process group is terminated.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            if isinstance(error, subprocess.TimeoutExpired):
                return dict(exit_code=process.returncode, timed_out=True,
                            elapsed_wall_s=time.monotonic() - started, child_pid=process.pid)
            raise


def archive(output, destination):
    require(not destination.exists(), "Archive exists; refusing replacement")
    hashes = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}
    require(hashes, "No output files to archive")
    shutil.copytree(output, destination)
    require(hashes == {str(p.relative_to(destination)): digest(p)
                       for p in sorted(destination.rglob("*")) if p.is_file()}, "Archive readback differs")
    write_json(destination.parent / "output_sha256.json", hashes, new=True)
    return len(hashes)


def run(plan, plan_path):
    require(tuple(plan["arms"]) == ARMS, "Frozen arm order changed")
    require(plan["package_manifest_sha256"] == MANIFEST, "Wrong frozen package")
    source = Path(plan["source_package"])
    config = verify_package(source)
    session = Path(plan["session_dir"])
    require(session.is_absolute() and not session.exists(), "Session already exists or is not absolute")
    lock_path = Path(plan["lock_path"])
    require(lock_path.is_absolute() and lock_path.is_file() and not lock_path.is_symlink(),
            "Actual shared regular lock file must already exist")
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_NOFOLLOW)
    require(stat.S_ISREG(os.fstat(lock_fd).st_mode), "Lock is not regular")
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.dup2(lock_fd, 9, inheritable=True)
    receipt = None
    try:
        held = os.fstat(9)
        def check_lock():
            actual = lock_path.stat()
            require((actual.st_dev, actual.st_ino) == (held.st_dev, held.st_ino), "Shared lock path replaced")
        check_lock()
        session.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(plan_path, session / "plan.json")
        receipt = dict(status="RUNNING", started_unix_s=time.time(), cells=[],
                       plan_sha256=digest(plan_path), controller_sha256=digest(__file__),
                       held_lock_device_inode=f"{held.st_dev}:{held.st_ino}",
                       host_differences=plan["host_differences"])
        write_json(session / "receipt.json", receipt)
        deadline = time.monotonic() + plan["approved_total_wall_seconds"]
        signal.alarm(plan["approved_total_wall_seconds"])
        memory_file = own_cgroup_memory_file()
        limit = int(memory_file.read_text().strip())
        require(0 < limit <= plan["approved_host_bytes"], "Own cgroup memory exceeds authorization")
        receipt.update(cgroup_memory_max_file=str(memory_file), actual_cgroup_memory_max_bytes=limit)
        initial = gpu_boundary(plan["authorized_gpu_uuid"])
        write_json(session / "gpu-start-occupancy.json", initial, new=True)
        require(initial["empty"], "GPU busy before initialization")
        require(shutil.disk_usage(session).free >= plan["minimum_free_disk_bytes"], "Insufficient new-host disk space")
        # Source and private model preflights initialize no CUDA context.
        startup_env = private_env(plan, session / "preflight-cache", "ordinary", memory_file)
        model_start = time.monotonic()
        write_json(session / "model-identity-receipt.json", verify_model(plan, config, startup_env), new=True)
        receipt["model_verification_wall_s"] = time.monotonic() - model_start
        preflight = command([plan["python"], str(source / "pkg/preflight.py")], env=startup_env, timeout=60)
        write_json(session / "runtime-preflight.json", dict(status="PASS", output=preflight), new=True)
        for index, arm in enumerate(ARMS):
            check_lock()
            require(deadline - time.monotonic() > plan["per_cell_wall_seconds"] + 60,
                    "Group budget cannot fund next complete cell")
            state = gpu_boundary(plan["authorized_gpu_uuid"])
            require(state["empty"], f"GPU busy before cell {index}")
            cell = session / f"cell-{index:02d}-{arm}"
            cell.mkdir()
            package = cell / "candidate_native_oldest_strong_r01"
            shutil.copytree(source, package)
            verify_package(package)
            cache = cell / "runtime-cache"
            cache.mkdir()
            env = private_env(plan, cache, arm, memory_file)
            output = cell / "output"
            # The frozen candidate routes all arms through ordinary-only wrapper,
            # then chooses effective ordinary/native/fund exclusively by env.
            argv = ["bash", str(package / "pkg/run.sh"), "native_full_ordinary_only",
                    "performance", "ordinary", str(output)]
            row = dict(arm=arm, argv=argv, output_dir=str(output), started_unix_s=time.time(),
                       gpu_before=state, runtime_cache_initial="EMPTY_PRIVATE", archive_status="NOT_STARTED")
            receipt["cells"].append(row)
            write_json(session / "receipt.json", receipt)
            write_json(cell / "oldest-admission-mode.json", dict(
                mode=env['A_NATIVE_OLDEST_ADMISSION'], effective_ordinary_backfill=False,
                victim_rule=env['A_NATIVE_VICTIM_RULE'], funding_victim_rule=env['A_FUNDING_VICTIM_RULE'],
                recovery_lease_mode=env['A_RECOVERY_LEASE_MODE'],
                oldest_repeat=True, full_running="off", current_guard="off",
                self_preempt_continue="off", capacity_deferral="off"), new=True)
            try:
                row.update(run_child(argv, env, cell / "launch.log", plan["per_cell_wall_seconds"] + 2))
            finally:
                row["finished_unix_s"] = time.time()
                if output.is_dir() and any(output.iterdir()):
                    row["archive_files"] = archive(output, cell / "archive")
                    row["archive_status"] = "VERIFIED"
                after = gpu_boundary(plan["authorized_gpu_uuid"])
                row.update(gpu_process_state_after="EMPTY" if after["empty"] else "BUSY", gpu_after=after)
                write_json(session / "receipt.json", receipt)
            require(row["gpu_process_state_after"] == "EMPTY", "GPU not empty after cell; no next cell")
            require(row["exit_code"] == 0 and not row["timed_out"] and row["archive_status"] == "VERIFIED",
                    f"Cell {index} failed; no retry or next cell")
        receipt["status"] = "CELLS_COMPLETE"
        return 0
    except BaseException as error:
        if receipt is not None:
            receipt.update(status="ABORTED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        if receipt is not None:
            receipt["finished_unix_s"] = time.time()
            receipt["elapsed_wall_s"] = receipt["finished_unix_s"] - receipt["started_unix_s"]
            write_json(session / "receipt.json", receipt)
        if lock_fd != 9:
            os.close(9)
        os.close(lock_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda number, frame: (_ for _ in ()).throw(InterruptedError(f"signal {number}")))
    signal.signal(signal.SIGALRM, lambda number, frame: (_ for _ in ()).throw(TimeoutError("Group wall budget exhausted")))
    try:
        return run(json.loads(args.plan.read_text()), args.plan)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"ABORT_STRONG_NEWHOST: {type(error).__name__}: {error}", file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())


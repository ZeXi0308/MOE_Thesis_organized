#!/usr/bin/env python3
"""One-shot CPU handoff: successful r09 -> DeepSeek BF16 -> DeepSeek FP8.

No GPU access here. Only run_locked.py can request the shared GPU lock/budget.
An error records STOPPED, leaves all existing processes alone, and never retries.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


PAYLOAD = {
    "code/run_deepseek_matrix.py": "1587d261d07f0757402dbc46c63a7e183d58f4c436c8a5a102f81bfdb2836452",
    "configs/deepseek_matrix.json": "f9fd31c92f6c4b7b6b4f9574e42813267bd9a54ce1756151bec3b9a331d18dd8",
}
R09 = "olmoe_fp8_matrix_r09"
SUCCESSORS = ("deepseek_bf16_matrix_r01", "deepseek_fp8_matrix_r01")
DS_STAGES = ("native_traces", "native_tuning", "graph_dense", "graph_marginal",
             "quality_b1", "quality_b8", "quality_b32", "bench_graph")
R09_STAGES = DS_STAGES[:4] + ("microbench_policy", "rolling_path") + DS_STAGES[4:]
ACTIVE = {"PREPARING", "WAITING", "RUNNING"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def process(pid):
    """Read Linux process identity without sending signals or inspecting GPUs."""
    if not isinstance(pid, int) or pid <= 0:
        return None
    base = Path("/proc") / str(pid)
    try:
        first = (base / "stat").read_text().rsplit(")", 1)[1].split()
        command = (base / "cmdline").read_bytes().split(b"\0")
        second = (base / "stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    if first[19] != second[19]:
        raise RuntimeError(f"PID {pid} was reused during observation; stop for inspection")
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    boot_time = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                     if line.startswith("btime "))
    return {"pid": pid, "boot_id": boot, "start_ticks": int(second[19]), "state": second[0],
            "started_unix": boot_time + int(second[19]) / os.sysconf("SC_CLK_TCK"),
            "command": [os.fsdecode(arg) for arg in command if arg]}


def running(run, status, role, bound):
    pid = status.get(role + "_pid")
    previous = bound.get(role)
    if previous and previous["pid"] != pid:
        raise RuntimeError(f"{run.name}: {role} PID changed in status")
    current = process(pid)
    if current is None or current["state"] in ("Z", "X"):
        return False
    if previous:
        return all(current[key] == previous[key] for key in ("pid", "boot_id", "start_ticks"))
    argv = current["command"]
    script = "run_locked.py" if role == "controller" else (
        "run_olmoe_fp8_r09.py" if run.name == R09 else "run_deepseek_matrix.py")
    flag = "--name" if role == "controller" else "--output"
    matches = any(Path(arg).name == script for arg in argv)
    matches &= flag in argv and argv.index(flag) + 1 < len(argv)
    if matches:
        value = argv[argv.index(flag) + 1]
        matches &= value == run.name if role == "controller" else Path(value).name == run.name
    expected_start = status.get("created_unix" if role == "controller" else "started_unix")
    if not matches or expected_start is None or not -2 <= current["started_unix"] - expected_start <= 120:
        raise RuntimeError(f"{run.name}: cannot identify {role} PID {pid}; possible PID reuse")
    bound[role] = {key: current[key] for key in ("pid", "boot_id", "start_ticks")}
    return True


def completed(run, expected_stages):
    """Use the existing runner success contract, with terminal receipt integrity."""
    hashes = read_json(run / "SHA256.json")
    for name in ("status.json", "stages.json", "bench/status.json"):
        if hashes.get(name) != sha(run / name):
            raise RuntimeError(f"{run.name}: terminal SHA missing or changed: {name}")
    status, stages = read_json(run / "status.json"), read_json(run / "stages.json")
    rows = stages.get("stages", [])
    if status.get("state") != "COMPLETE" or status.get("exit_code") != 0:
        raise RuntimeError(f"{run.name}: controller did not complete successfully")
    if any(not isinstance(status.get(role + "_pid"), int) or status[role + "_pid"] <= 0
           for role in ("controller", "worker")):
        raise RuntimeError(f"{run.name}: completed record lacks controller/worker PIDs")
    if (stages.get("status") != "COMPLETE" or tuple(row.get("name") for row in rows) != expected_stages
            or any(row.get("status") != "COMPLETE" or row.get("exit_code") != 0 for row in rows)):
        raise RuntimeError(f"{run.name}: required stages are missing or unsuccessful")
    if read_json(run / "bench/status.json").get("status") != "COMPLETE":
        raise RuntimeError(f"{run.name}: full performance runner did not complete")
    return {"name": run.name, "ended_unix": status["ended_unix"],
            "lock_held_s": status["lock_held_s"], "terminal_sha256": sha(run / "SHA256.json")}


def wait_success(run, expected_stages, save, child=None, timeout=30000, poll=30):
    deadline, bound, status_deadline = time.monotonic() + timeout, {}, time.monotonic() + 30
    while True:
        code = child.poll() if child is not None else None
        path = run / "status.json"
        if not path.exists() and child is not None and code is None and time.monotonic() < status_deadline:
            time.sleep(min(poll, .5))
            continue
        status = read_json(path)
        state = status.get("state")
        if state not in ACTIVE | {"COMPLETE"}:
            raise RuntimeError(f"{run.name}: terminal state {state}; no automatic retry")
        if child is not None and status.get("controller_pid") != child.pid:
            raise RuntimeError(f"{run.name}: controller PID does not match the submitted child")
        controller = running(run, status, "controller", bound)
        worker = running(run, status, "worker", bound)
        save("WAITING_FOR_EXIT", observed_run=run.name, observed_run_state=state, process_identities=bound)
        if state in ACTIVE and (not controller or child is not None and code is not None):
            raise RuntimeError(f"{run.name}: active record has no executing controller")
        if state == "COMPLETE" and not controller and not worker and (child is None or code is not None):
            if child is not None and child.wait() != 0:
                raise RuntimeError(f"{run.name}: controller process exited unsuccessfully")
            return completed(run, expected_stages)
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{run.name}: observation deadline; existing processes left unchanged")
        time.sleep(poll)


def verify_payload(stage, root, deployed):
    for name, expected in PAYLOAD.items():
        source, target = stage / name, root / name
        if source.is_symlink() or sha(source) != expected:
            raise RuntimeError(f"Staged source changed: {name}")
        if deployed:
            if target.is_symlink() or sha(target) != expected:
                raise RuntimeError(f"Deployed source changed: {name}")
        elif target.exists() or target.is_symlink():
            raise FileExistsError(f"Only new files may be deployed; target already exists: {target}")


def available(root, names, allowed_active=None):
    for name in names:
        if (root / "runs" / name).exists():
            raise FileExistsError(f"Duplicate run directory: {name}")
    for path in (root / "runs").glob("*/status.json"):
        if path.parent.name != allowed_active and read_json(path).get("state") in ACTIVE:
            raise RuntimeError(f"Another K run is active: {path}")


def deploy(stage, root):
    verify_payload(stage, root, False)
    for name in PAYLOAD:
        source, target = stage / name, root / name
        # Exclusive creation cannot replace an existing file or symlink.
        with target.open("xb") as handle:
            handle.write(source.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
    verify_payload(stage, root, True)


def command(root, python, name, fmt):
    return [python, str(root / "code/run_locked.py"), "--root", str(root), "--name", name,
            "--wait-seconds", "21600", "--timeout", "7200", "--", python,
            str(root / "code/run_deepseek_matrix.py"), "--root", str(root),
            "--format", fmt, "--output", f"runs/{name}", "--budget-seconds", "7170"]


def handoff(root, stage, python, save):
    # The one allowed active run at entry is the predecessor, observed below.
    available(root, SUCCESSORS, allowed_active=R09)
    verify_payload(stage, root, False)
    receipts = [wait_success(root / "runs" / R09, R09_STAGES, save)]
    available(root, SUCCESSORS)
    save("DEPLOYING", completed_runs=receipts, payload_sha256=PAYLOAD)
    deploy(stage, root)
    environment = os.environ.copy()
    environment.pop("VLLM_TUNED_CONFIG_FOLDER", None)
    environment.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       VLLM_ENABLE_V1_MULTIPROCESSING="0")
    for fmt, name in zip(("bf16", "fp8"), SUCCESSORS):
        available(root, [name])
        verify_payload(stage, root, True)
        argv = command(root, python, name, fmt)
        with (stage / f"{name}_controller.log").open("x") as log:
            child = subprocess.Popen(argv, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
            save("SUBMITTED", submitted_run=name, command=argv, controller_pid=child.pid)
            receipts.append(wait_success(root / "runs" / name, DS_STAGES, save, child))
        save("RUN_COMPLETE", completed_runs=receipts)
    save("COMPLETE", completed_runs=receipts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/research_K_expert_skip"))
    parser.add_argument("--python", default="/root/miniconda3/bin/python")
    args = parser.parse_args()
    root, stage = args.root.resolve(), Path(__file__).resolve().parent
    # This CPU-only singleton lock is separate from the GPU lock in run_locked.py.
    with (stage / "handoff.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"pid": os.getpid(), "created_unix": time.time(), "root": str(root)}
        destination = stage / "handoff.json"
        with destination.open("x") as handle:
            json.dump({**state, "state": "STARTING"}, handle)
        def save(value, **fields):
            state.update(state=value, observed_unix=time.time(), **fields)
            temporary = stage / "handoff.tmp"
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(destination)
            print(json.dumps(state), flush=True)
        def interrupted(signum, frame):
            raise InterruptedError(f"Handoff received signal {signum}; submitted processes left unchanged")
        signal.signal(signal.SIGTERM, interrupted)
        try:
            handoff(root, stage, args.python, save)
        except BaseException as error:
            save("STOPPED", error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()

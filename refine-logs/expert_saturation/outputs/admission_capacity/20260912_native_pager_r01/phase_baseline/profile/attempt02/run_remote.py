"""Run one existing GPU process with four reset/common-warmup phase episodes."""
import ast
import hashlib
import json
import os
import sys
from pathlib import Path
import subprocess
import time


def main():
    staging = Path(__file__).resolve().parent
    sys.path.insert(0, str(staging))
    config = json.loads((staging / "config.json").read_text())
    root = Path(config["remote_root"])
    source, out = root / config["remote_reference"], root / config["remote_output"]
    old = json.loads((source / "execution.json").read_text())
    if old["status"] != "COMPLETED" or len(old["cells"]) != 4:
        raise RuntimeError("Previous four-cell comparison is incomplete")
    for name, expected in config["source_sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Frozen source changed: " + name)
    # Reuse exactly the prior external observer, without executing its campaign.
    old_source = (source / "run_campaign.py").read_text()
    observer = next(n for n in ast.parse(old_source).body
                    if isinstance(n, ast.FunctionDef) and n.name == "observe")
    import pynvml as nv
    nv.nvmlInit()
    gpu = nv.nvmlDeviceGetHandleByIndex(0)
    namespace = dict(Path=Path, nv=nv, gpu=gpu, time=time)
    exec(compile(ast.Module(body=[observer], type_ignores=[]), "frozen_observer", "exec"), namespace)
    if nv.nvmlDeviceGetComputeRunningProcesses(gpu):
        raise RuntimeError("GPU occupied; no episode process started")
    out.mkdir(exist_ok=False)
    for name in ("run_wisp_injection_probe.py", "run_wisp_olmoe_probe.py",
                 "wisp_paging_trace.py", "phase_prefill_policy.py", "workload.json"):
        (out / name).write_bytes((source / name).read_bytes())
    for name in ("config.json", "run_phase_repeated.py", "reuse_phase_engine.py", "profile_probe_patch.py"):
        (out / name).write_bytes((staging / name).read_bytes())
    from profile_probe_patch import patch_profile_probe
    probe_path = out / "run_wisp_injection_probe.py"
    probe_path.write_text(patch_profile_probe(probe_path.read_text()))
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in out.iterdir() if p.is_file()}
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = dict(os.environ, OMP_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1", PYTHONPATH=str(root / "source/src") + ":" + str(out))
    command = ["taskset", "-c", "0-7", "timeout", "-s", "TERM", "600",
               str(root / "venv/bin/python"), str(out / "run_phase_repeated.py")]
    launch = dict(status="RUNNING", command=command, start_unix=time.time())
    (out / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    print(json.dumps(dict(status="STARTING", output=str(out))), flush=True)
    with (out / "run.log").open("x") as log, (out / "hardware.jsonl").open("x") as hardware:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            hardware.write(json.dumps(namespace["observe"]()) + "\n")
            hardware.flush()
            time.sleep(0.1)
    launch.update(status="EXITED", exit_code=process.returncode, end_unix=time.time())
    (out / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    nv.nvmlShutdown()
    state = json.loads((out / "execution.json").read_text())
    print(json.dumps(dict(status=state["status"], error=state.get("error"),
        cells=[dict(cell=r["cell"], status=r["status"]) for r in state["cells"]])), flush=True)
    if process.returncode or state["status"] != "COMPLETED":
        raise RuntimeError("Repeated campaign stopped; all outputs retained")


if __name__ == "__main__":
    main()

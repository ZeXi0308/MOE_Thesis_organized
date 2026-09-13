"""Build the five-strategy comparison from verified remote-existing inputs."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def build_probe_sources(source):
    from static_calibration_patch import patch_static_probe, patch_static_policy
    from map_probe_patch import patch_map_probe
    from first_step_map_probe_patch import patch_first_step_map_probe
    from next_chunk_policy_patch import patch_next_chunk_probe, patch_next_chunk_policy
    probe = patch_first_step_map_probe(patch_map_probe(patch_static_probe(
        (source / "run_wisp_injection_probe.py").read_text())))
    policy = patch_static_policy((source / "phase_prefill_policy.py").read_text())
    one8 = patch_next_chunk_probe(probe)
    anchor = "from phase_prefill_policy import install as install_phase_policy"
    if one8.count(anchor) != 1:
        raise RuntimeError("One8 policy import anchor changed")
    one8 = one8.replace(anchor, "from next_chunk_phase_policy import install as install_phase_policy", 1)
    sources = {"run_wisp_injection_probe.py": probe, "run_one8_probe.py": one8,
        "phase_prefill_policy.py": policy, "next_chunk_phase_policy.py": patch_next_chunk_policy(policy)}
    for name, text in sources.items():
        compile(text, name, "exec")
    return sources


def build_workloads(data, config):
    if hashlib.sha256(data).hexdigest() != config["prepared_source"]["sha256"]:
        raise RuntimeError("Prepared source workload hash changed")
    inputs, workloads = json.loads(data), {}
    for group, item in config["workload_sources"].items():
        rows = []
        for index, (source_index, count) in enumerate(zip(item["source_indices"], (32, 32, 128))):
            row = inputs["source_requests"][source_index]
            ids = inputs["actual_prompt_token_ids"][source_index][:count]
            if len(ids) != count:
                raise RuntimeError("Prepared input prefix is too short")
            rows.append(dict(request_id=f"independent-{index}", document_id=row["document_id"],
                document_sha256=row["document_sha256"], prompt_token_ids=ids, arrival_s=0.0))
        encoded = (json.dumps(dict(requests=rows), indent=2) + "\n").encode()
        if hashlib.sha256(encoded).hexdigest() != item["sha256"]:
            raise RuntimeError("Constructed workload hash changed: " + group)
        workloads[group] = encoded
    return workloads


def main():
    staging = Path(__file__).resolve().parent
    sys.path.insert(0, str(staging))
    from gpu_preflight import check_gpu
    check_gpu(staging / "gpu_preflight_attempts" / f"{time.time_ns()}-{os.getpid()}-parent.json")
    config = json.loads((staging / "config.json").read_text())
    root = Path(config["remote_root"])
    source, out = root / config["remote_reference"], root / config["remote_output"]
    previous = json.loads((source / "execution.json").read_text())
    if previous["status"] != "COMPLETED" or len(previous["cells"]) != 4:
        raise RuntimeError("Frozen source campaign is incomplete")
    for name, digest in config["source_sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Frozen source changed: " + name)
    workloads = build_workloads(Path(config["prepared_source"]["remote"]).read_bytes(), config)
    sources = build_probe_sources(source)
    out.mkdir(exist_ok=False)
    for name in ("run_wisp_olmoe_probe.py", "wisp_paging_trace.py"):
        (out / name).write_bytes((source / name).read_bytes())
    for name in ("config.json", "run_phase_repeated.py", "reuse_phase_engine.py", "batched_expert_map.py",
                 "map_probe_patch.py", "static_calibration_patch.py", "first_step_map_probe_patch.py",
                 "next_chunk_policy_patch.py", "gpu_preflight.py"):
        (out / name).write_bytes((staging / name).read_bytes())
    for name, text in sources.items():
        (out / name).write_text(text)
    for group, data in workloads.items():
        (out / f"workload-{group}.json").write_bytes(data)
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    observer = next(n for n in ast.parse((source / "run_campaign.py").read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "observe")
    import pynvml as nv
    nv.nvmlInit()
    namespace = dict(Path=Path, nv=nv, gpu=nv.nvmlDeviceGetHandleByIndex(0), time=time)
    exec(compile(ast.Module(body=[observer], type_ignores=[]), "frozen_observer", "exec"), namespace)
    env = dict(os.environ, OMP_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
               PYTHONPATH=str(root / "source/src") + ":" + str(out))
    command = ["taskset", "-c", "0-7", "timeout", "-s", "TERM", "600",
               str(root / "venv/bin/python"), str(out / "run_phase_repeated.py")]
    launch = dict(status="RUNNING", command=command, start_unix=time.time())
    (out / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    print(json.dumps(dict(status="STARTING", output=str(out), cells=len(config["cells"]))), flush=True)
    with (out / "run.log").open("x") as log, (out / "hardware.jsonl").open("x") as hardware:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            hardware.write(json.dumps(namespace["observe"]()) + "\n")
            hardware.flush()
            time.sleep(0.1)
    launch.update(status="ABORT" if process.returncode == 91 else "EXITED", exit_code=process.returncode, end_unix=time.time())
    (out / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    nv.nvmlShutdown()
    state = json.loads((out / "execution.json").read_text())
    print(json.dumps(dict(status=state["status"], cells=len(state["cells"]), error=state.get("error"))), flush=True)
    if process.returncode:
        raise SystemExit(process.returncode)
    if json.loads((out / "execution.json").read_text())["status"] != "COMPLETED":
        raise RuntimeError("Campaign incomplete; all outputs retained")


if __name__ == "__main__":
    main()

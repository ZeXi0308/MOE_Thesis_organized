"""Run exactly one fresh engine; archive it locally before starting the other."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("forward", "reverse"):
        raise SystemExit("usage: python run_block.py forward|reverse")
    block = sys.argv[1]
    root = Path(__file__).resolve().parent
    out = root / "results"
    out.mkdir(exist_ok=True)
    metadata = out / (block + "-execution.json")
    command = [sys.executable, "-u", "run_native_capacity.py", "--prepared-dir", "prepared",
               "--output-dir", "results/" + block, "--caps", "8,12,16,24,32",
               "--engine-max-seqs", "32", "--arrival-scales", "1", "--repeats", "1",
               "--compare-feedback-ladders", "--ttft-slo-s", "0.20", "--tpot-slo-s", "0.009",
               "--warmup-condition-rounds", "1"]
    if block == "reverse":
        command.append("--reverse-conditions")
    env = dict(os.environ, HF_HUB_CACHE="/root/autodl-tmp/hf-cache/hub", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="0",
               VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_USE_FLASHINFER_SAMPLER="0")
    state = dict(block=block, status="STARTING", command=command, cwd=str(root),
                 started_unix_s=time.time(), returncode=None)
    with metadata.open("x") as f:
        json.dump(state, f, indent=2)
    print(shlex.join(command), flush=True)
    try:
        with (out / (block + ".stdout.log")).open("x") as stdout, \
             (out / (block + ".stderr.log")).open("x") as stderr:
            code = subprocess.call(command, cwd=root, env=env, stdout=stdout, stderr=stderr)
        state.update(status="EXITED", returncode=code)
    except BaseException as error:
        state.update(status="LAUNCH_OR_WAIT_FAILED", error=repr(error))
        raise
    finally:
        state["finished_unix_s"] = time.time()
        metadata.write_text(json.dumps(state, indent=2) + "\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()

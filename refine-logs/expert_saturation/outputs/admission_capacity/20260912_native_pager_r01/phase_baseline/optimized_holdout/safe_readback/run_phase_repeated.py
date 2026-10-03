"""Same-process ABBA episodes using the frozen phase probe and one LLM instance."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    base = Path(__file__).resolve().parent
    config = json.loads((base / "config.json").read_text())
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_USE_V1="1",
        VLLM_ENABLE_V1_MULTIPROCESSING="0", WISP_PLUGIN_DISABLE="0", WISP_MODE="paged",
        WISP_CAP_EXPERTS="16", WISP_PREFETCH="0", WISP_DYNAMIC="0")
    import vllm
    import batched_expert_map as map_patch
    import run_wisp_injection_probe as probe
    from reuse_phase_engine import ReusingLLM
    reuser = ReusingLLM(vllm.LLM)
    vllm.LLM = reuser
    state = dict(status="RUNNING", pid=os.getpid(), cells=[], start_unix=time.time(),
        reference_rule="First declared episode, never selected by outcome")

    def save():
        (base / "execution.json").write_text(json.dumps(state, indent=2) + "\n")

    save()
    try:
        state["map_selftest"] = map_patch.gpu_selftest()
        save()
        for cell in config["cells"]:
            pids = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                            "--format=csv,noheader"], text=True).split()
            if set(pids) - {str(os.getpid())}:
                raise RuntimeError("Foreign GPU process before episode; no next episode started")
            out = base / cell["cell"]
            out.mkdir(exist_ok=False)
            row = dict(cell, start_unix=time.time(), status="RUNNING")
            state["cells"].append(row)
            save()
            sys.argv = [str(base / "run_wisp_injection_probe.py"), "--model", config["model"],
                "--workload", str(base / "workload.json"), "--out", str(out / "result.json"),
                "--chunk", "8", "--new-prompt-length", "128", "--policy", cell["policy"]]
            map_patch.set_mode(cell["map_mode"])
            code = probe.main()
            raw = json.loads((out / "result.json").read_text())
            row.update(exit_code=code, end_unix=time.time(), status=raw.get("status"),
                       reuse_record=reuser.records[-1] if reuser.records else None)
            if code or row["status"] != "COMPLETED":
                raise RuntimeError("Episode failed: " + cell["cell"])
            action = raw["action"]
            before = dict(tokens=action["old_output_tokens"], scheduler=action["before_scheduler"],
                          pager=[w["pager_execution_state"] for w in action["before_worker"]])
            actual = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
            row["prestate_sha256"] = actual
            if "reference_prestate_sha256" not in state:
                state["reference_prestate_sha256"] = actual
            if actual != state["reference_prestate_sha256"]:
                row["status"] = "INVALID_PRESTATE"
                raise RuntimeError("Preaction state differs from first declared episode")
            save()
        state["status"] = "COMPLETED"
    except BaseException as exc:
        state.update(status="STOPPED", error=f"{type(exc).__name__}: {exc}")
    finally:
        state["end_unix"] = time.time()
        save()
    print(json.dumps(dict(status=state["status"], cells=[
        dict(cell=r["cell"], status=r["status"]) for r in state["cells"]])), flush=True)
    return 0 if state["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

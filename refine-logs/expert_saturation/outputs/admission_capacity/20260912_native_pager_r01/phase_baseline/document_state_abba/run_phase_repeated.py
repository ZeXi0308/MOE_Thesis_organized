"""Same-process ABBA episodes using the frozen phase probe and one LLM instance."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from gpu_preflight import check_gpu


def main():
    base = Path(__file__).resolve().parent
    config = json.loads((base / "config.json").read_text())
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_USE_V1="1",
        VLLM_ENABLE_V1_MULTIPROCESSING="0", WISP_PLUGIN_DISABLE="0", WISP_MODE="paged",
        WISP_CAP_EXPERTS="16", WISP_PREFETCH="0", WISP_DYNAMIC="0")
    state = dict(status="RUNNING", pid=os.getpid(), cells=[], start_unix=time.time(),
        reference_rule="First declared episode, never selected by outcome")

    def save():
        (base / "execution.json").write_text(json.dumps(state, indent=2) + "\n")

    save()
    try:
        state["last_preflight"] = str(base / "gpu_preflight" / "before_selftest.json")
        save()
        check_gpu(state["last_preflight"], allow_pid=os.getpid())
        import vllm
        import batched_expert_map as map_patch
        import run_wisp_injection_probe as probe
        from reuse_phase_engine import ReusingLLM
        reuser = ReusingLLM(vllm.LLM)
        vllm.LLM = reuser
        state["map_selftest"] = map_patch.gpu_selftest()
        save()
        for cell in config["cells"]:
            state["last_preflight"] = str(base / "gpu_preflight" / f"before_{cell['cell']}.json")
            save()
            check_gpu(state["last_preflight"], allow_pid=os.getpid())
            out = base / cell["cell"]
            out.mkdir(exist_ok=False)
            row = dict(cell, start_unix=time.time(), status="RUNNING")
            state["cells"].append(row)
            save()
            sys.argv = [str(base / "run_wisp_injection_probe.py"), "--model", config["model"],
                "--workload", str(base / f"workload-{cell['document_set']}.json"), "--out", str(out / "result.json"),
                "--chunk", str(cell["chunk"]), "--new-prompt-length", "128", "--policy", cell["policy"]]
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
            references = state.setdefault("document_prestate_sha256", {})
            group = cell["document_set"]
            if group not in references:
                references[group] = actual
            if actual != references[group]:
                row["status"] = "INVALID_PRESTATE"
                raise RuntimeError("Preaction state differs from first declared episode")
            save()
        state["status"] = "COMPLETED"
    except BaseException as exc:
        state.update(status="ABORT" if isinstance(exc, SystemExit) and exc.code == 91 else "STOPPED",
                     error=f"{type(exc).__name__}: {exc}")
    finally:
        state["end_unix"] = time.time()
        save()
    print(json.dumps(dict(status=state["status"], cells=[
        dict(cell=r["cell"], status=r["status"]) for r in state["cells"]])), flush=True)
    return 91 if state["status"] == "ABORT" else 0 if state["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

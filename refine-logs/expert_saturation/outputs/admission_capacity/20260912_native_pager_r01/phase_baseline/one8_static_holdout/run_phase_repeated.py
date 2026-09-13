"""Two randomized blocks; five strategies share one drained/reset LLM instance."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from gpu_preflight import check_gpu


def probe_entry(base, cell, config, out):
    strategy = cell["strategy"]
    if strategy not in ("static8", "static16", "static32", "phase8", "one8"):
        raise RuntimeError("Unknown strategy")
    chunk = 8 if strategy == "phase8" else 16 if strategy == "one8" else int(strategy[6:])
    policy = "static16" if strategy == "one8" else strategy
    if cell["chunk"] != chunk or cell["policy"] != policy:
        raise RuntimeError("Strategy/CLI mismatch")
    module = "run_one8_probe" if strategy == "one8" else "run_wisp_injection_probe"
    argv = [str(base / f"{module}.py"), "--model", config["model"], "--workload",
        str(base / f"workload-{cell['document_set']}.json"), "--out", str(out / "result.json"),
        "--chunk", str(chunk), "--new-prompt-length", "128", "--policy", policy]
    return module, argv + (["--next-chunk", "8"] if strategy == "one8" else [])


def main():
    base = Path(__file__).resolve().parent
    config = json.loads((base / "config.json").read_text())
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_USE_V1="1",
        VLLM_ENABLE_V1_MULTIPROCESSING="0", WISP_PLUGIN_DISABLE="0", WISP_MODE="paged",
        WISP_CAP_EXPERTS="16", WISP_PREFETCH="0", WISP_DYNAMIC="0")
    state = dict(status="RUNNING", pid=os.getpid(), cells=[], start_unix=time.time(), reference_rule=config["reference_rule"])
    def save():
        (base / "execution.json").write_text(json.dumps(state, indent=2) + "\n")
    save()
    try:
        check_gpu(base / "gpu_preflight/before_selftest.json", allow_pid=os.getpid())
        import vllm
        import batched_expert_map as map_patch
        import run_wisp_injection_probe as baseline
        import run_one8_probe as one8
        from reuse_phase_engine import ReusingLLM
        reuser = ReusingLLM(vllm.LLM)
        vllm.LLM = reuser
        probes = dict(run_wisp_injection_probe=baseline, run_one8_probe=one8)
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
            module, sys.argv = probe_entry(base, cell, config, out)
            map_patch.set_mode(cell["map_mode"])
            code = probes[module].main()
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
            if actual != config["preaction_sha256"]:
                row["status"] = "INVALID_COMMON_PRESTATE"
                raise RuntimeError("Pre-injection state differs from frozen common old-pair state")
            group = cell["document_set"]
            state.setdefault("document_prestate_sha256", {})[group] = actual
            if cell["strategy"] == "one8":
                row["prebranch_sha256"] = raw["next_chunk_branch"]["prestate_sha256"]
                ref = state.setdefault("document_prebranch_sha256", {}).setdefault(group, row["prebranch_sha256"])
                if ref != row["prebranch_sha256"]:
                    row["status"] = "INVALID_BRANCH_PRESTATE"
                    raise RuntimeError("One8 after-first-chunk state differs within document")
            elif "next_chunk_branch" in raw:
                raise RuntimeError("Baseline unexpectedly incurred a next-chunk snapshot")
            save()
        state["status"] = "COMPLETED"
    except BaseException as exc:
        state.update(status="ABORT" if isinstance(exc, SystemExit) and exc.code == 91 else "STOPPED", error=f"{type(exc).__name__}: {exc}")
    finally:
        state["end_unix"] = time.time()
        save()
    return 91 if state["status"] == "ABORT" else 0 if state["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

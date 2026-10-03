#!/usr/bin/env python3
"""One fresh engine, two common mixed warmups, two midpoint budget episodes; no method GO."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime"))
from native_capture import capture_episode
from metrics import summarize_episode_requests

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def gpu_state():
    query = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                     "--format=csv,noheader"], text=True).strip()
    others = [r for r in csv.reader(query.splitlines()) if r and int(r[0]) != os.getpid()]
    if others:
        raise RuntimeError(f"other GPU compute processes: {others}")
    return dict(compute_processes=query, gpu=subprocess.check_output(["nvidia-smi",
        "--query-gpu=name,uuid,memory.total,memory.used,temperature.gpu,power.draw,clocks.sm",
        "--format=csv,noheader"], text=True).strip())

def set_empty_budget(engine, budget):
    config = engine.vllm_config.scheduler_config
    scheduler = engine.engine_core.engine_core.scheduler
    if type(budget) is not int or budget not in (512, 1024):
        raise ValueError("frozen runtime budgets are 512 and 1024")
    if config.max_num_seqs != 8 or config.max_num_batched_tokens != 1024:
        raise ValueError("compiled engine must retain cap8 and token capacity1024")
    if not hasattr(scheduler, "max_num_scheduled_tokens") or scheduler.max_num_running_reqs != 8:
        raise ValueError("native scheduler budget field/cap differs from frozen v0.26 contract")
    if engine.has_unfinished_requests() or scheduler.get_request_counts() != (0, 0) or scheduler.requests:
        raise ValueError("runtime budget may change only after complete drain")
    previous = scheduler.max_num_scheduled_tokens
    scheduler.max_num_scheduled_tokens = budget
    return dict(previous_budget=previous, runtime_budget=budget, compiled_token_capacity=1024, cap=8)

def prepare(block):
    inputs = {}
    for cohort in ("mixed",):
        folder = ROOT / "prepared" / cohort
        config = json.loads((folder / "config.json").read_text())
        workload = json.loads((folder / "workload.json").read_text())
        if hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest() != config["workload_sha256"]:
            raise ValueError("prepared workload hash differs")
        lengths = [len(p) for p in workload["actual_prompt_token_ids"]]
        expected = [128 if i % 2 == 0 else 2048 for i in range(16)]
        if lengths != expected or config["output_tokens"] != 128 or config["requests"] != 16:
            raise ValueError("prepared cohort differs from frozen 16-request 128/2048 contract")
        arrivals = workload["arrival_traces_s"]["steady"]
        if len(arrivals) != 16 or any(abs(a - i * 0.05) > 1e-12 for i, a in enumerate(arrivals)):
            raise ValueError("prepared arrival sequence differs")
        inputs[cohort] = (config, workload)
    original = inputs["mixed"][0]
    if any(c["model"] != original["model"] or c["seed"] != original["seed"] for c, _ in inputs.values()):
        raise ValueError("model or seed differs between cohorts")
    model = original["model"]
    engine_args = dict(model=model["id"], revision=model["revision"], tokenizer_revision=model["tokenizer_revision"],
        dtype="bfloat16", seed=original["seed"], max_model_len=4096, max_num_seqs=8,
        max_num_batched_tokens=1024, enable_chunked_prefill=True, gpu_memory_utilization=0.70,
        enable_prefix_caching=False, scheduling_policy="fcfs", async_scheduling=False,
        stream_interval=1, enforce_eager=False, enable_return_routed_experts=False, disable_log_stats=False)
    warmups = [dict(cohort="mixed", budget=b) for b in (512, 1024)]
    plans = warmups[:] if block == "forward" else list(reversed(warmups))
    files = [Path(__file__).resolve(), *(ROOT / "runtime").glob("*.py"), *ROOT.glob("prepared/*/*.json")]
    return inputs, dict(block=block, warmups=warmups, plans=plans, engine_args=engine_args,
        reference_slo=dict(ttft_slo_s=5.0, tpot_slo_s=0.2), max_seconds=180,
        source_and_input_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(files)},
        evidence_ceiling="REQUEST_LEVEL_EXPLORATORY_INTERVENTION_NO_METHOD_GO")

def execute(inputs, frozen, out):
    engine, completed, attempted = None, 0, 0
    try:
        before = gpu_state()
        os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_BATCH_INVARIANT="0",
                          VLLM_USE_FLASHINFER_SAMPLER="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        import torch
        import vllm
        from importlib.metadata import version
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine
        package = Path(vllm.__file__).parent
        dump(out / "environment.json", dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
            vllm=vllm.__version__, transformers=version("transformers"), gpu_before_init=before,
            cpu_threads=torch.get_num_threads(), vllm_source_sha256={p: sha(package / p) for p in
                ("v1/core/sched/scheduler.py", "v1/engine/llm_engine.py", "v1/engine/output_processor.py",
                 "v1/worker/gpu_model_runner.py")}, execution_env={k: os.environ.get(k) for k in
                ("VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_BATCH_INVARIANT", "VLLM_USE_FLASHINFER_SAMPLER",
                 "CUDA_VISIBLE_DEVICES", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                 "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_LAUNCH_BLOCKING")}))
        if vllm.__version__ != "0.26.0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("frozen vLLM0.26.0 and exactly one visible CUDA GPU required")
        engine = LLMEngine.from_engine_args(EngineArgs(**frozen["engine_args"]), enable_multiprocessing=False)
        capacity = engine.vllm_config.cache_config
        dump(out / "engine_capacity.json", dict(initial_budget=set_empty_budget(engine, 1024),
            num_gpu_blocks=getattr(capacity, "num_gpu_blocks", None), block_size=getattr(capacity, "block_size", None),
            scheduler_config=str(engine.vllm_config.scheduler_config), cache_config=str(capacity)))
        for phase, plans in (("warmup", frozen["warmups"]), ("cell", frozen["plans"])):
            for index, plan in enumerate(plans):
                before = gpu_state()
                state = set_empty_budget(engine, plan["budget"])
                config, workload = inputs[plan["cohort"]]
                config = dict(config, cap=8, policy="static")
                stem = f"{phase}-{index:03d}"
                print(json.dumps(dict(phase="START", cell=stem, plan=plan)), flush=True)
                dump(out / "status.json", dict(status="RUNNING", phase=phase, index=index, cells_completed=completed))
                raw = capture_episode(engine, workload, config, regime="steady", arrival_scale=1,
                    run_id=stem, max_seconds=frozen["max_seconds"])
                raw.update(plan=plan, phase=phase, budget_state=state, gpu_before=before)
                dump(out / (stem + "-raw.json"), raw)
                attempted += phase == "cell"
                if raw["status"] != "COMPLETE":
                    raise RuntimeError(f"{stem} incomplete: {raw['error']}")
                dump(out / (stem + "-checks.json"), dict(gpu_after=gpu_state(), drained_budget=set_empty_budget(engine, plan["budget"])))
                groups = {"all": raw["requests"], "short": [r for r in raw["requests"] if r["prompt_tokens"] == 128],
                          "long": [r for r in raw["requests"] if r["prompt_tokens"] == 2048]}
                dump(out / (stem + "-metrics.json"), dict(reference_slo_only=True, plan=plan,
                    by_prompt_length={g: summarize_episode_requests(rows, observation_end_s=raw["observation_end_s"],
                        **frozen["reference_slo"]) for g, rows in groups.items()}, host_chunk_diagnostics=raw["host_chunk_diagnostics"]))
                completed += phase == "cell"
                print(json.dumps(dict(phase="END", cell=stem, status=raw["status"])), flush=True)
        dump(out / "status.json", dict(status="COMPLETE", cells_completed=completed, warmups_completed=2,
            scientific_verdict="MEASUREMENT_ONLY_PENDING_ANALYSIS"))
    except BaseException as exc:
        dump(out / "status.json", dict(status="INCOMPLETE" if attempted else "BLOCKED", cells_completed=completed,
            cells_attempted=attempted, error=repr(exc)))
        (out / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        if engine is not None:
            engine.engine_core.shutdown()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    inputs, frozen = prepare(args.block)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dump(args.output_dir / "config.json", frozen)
    dump(args.output_dir / "plans.json", dict(warmups=frozen["warmups"], measured=frozen["plans"]))
    (args.output_dir / "commands.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    dump(args.output_dir / "status.json", dict(status="PREPARED" if args.prepare_only else "INITIALIZING",
        cells_completed=0, cells_planned=2, gpu_initialized=False))
    if args.prepare_only:
        return
    with (args.output_dir / "stdout.log").open("x") as stdout, (args.output_dir / "stderr.log").open("x") as stderr:
        os.dup2(stdout.fileno(), 1)
        os.dup2(stderr.fileno(), 2)
        execute(inputs, frozen, args.output_dir)

if __name__ == "__main__":
    main()

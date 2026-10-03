#!/usr/bin/env python3
"""Diagnose why the expert-union hook recorded nothing.

The first collection run loaded the engine, captured graphs and exited 0, but
`n_recorded_steps` was 0. Two independent gates could have caused that:

  A. the purity predicate never became true, i.e. the driver's reading of
     `scheduler.running` / `waiting` / per-request progress is wrong for this
     vLLM build;
  B. the hook fired but its shape guard rejected every call, i.e. the tensor
     reaching `block.gate`'s forward hook is not a [T, E] logits matrix.

Guessing between them would waste GPU time, so this script instruments both
and prints the ground truth: the real attribute names on the V1 request
objects, the real per-step scheduler state, and the real shapes seen at the
hook. It records nothing into the tracker and makes no scientific claim.
"""
import json
import os
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/diag")
PREPARED = Path(sys.argv[2] if len(sys.argv) > 2
                else "/root/autodl-tmp/expert-union/inputs_preparation/prepared/short")
CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 8
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 12

OUT.mkdir(parents=True, exist_ok=True)
os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_BATCH_INVARIANT="0",
                  VLLM_USE_FLASHINFER_SAMPLER="0")

import torch
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.engine.llm_engine import LLMEngine
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

workload = json.loads((PREPARED / "workload.json").read_text())
config = json.loads((PREPARED / "config.json").read_text())
model = config["model"]

engine = LLMEngine.from_engine_args(EngineArgs(
    model=model["id"], revision=model["revision"],
    tokenizer_revision=model["tokenizer_revision"], dtype="bfloat16",
    seed=config["seed"], max_model_len=4096, max_num_seqs=32,
    max_num_batched_tokens=1024, gpu_memory_utilization=0.90,
    enable_chunked_prefill=True, enable_prefix_caching=False,
    scheduling_policy="fcfs", async_scheduling=False, stream_interval=1,
    enforce_eager=False), enable_multiprocessing=False)

runner = engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
model_obj = runner.model
blocks = [m for m in model_obj.modules() if type(m).__name__.endswith("SparseMoeBlock")]
if not blocks:
    blocks = [m for m in model_obj.modules()
              if hasattr(m, "gate") and hasattr(m, "experts")]

report = dict(n_blocks=len(blocks), block_type=type(blocks[0]).__name__,
              gate_type=type(blocks[0].gate).__name__,
              num_experts=model_obj.config.num_experts,
              num_experts_per_tok=model_obj.config.num_experts_per_tok)

# ---- Gate B: what does the hook actually see? ----------------------------
seen = []


def probe(module, inputs, output):
    def describe(x):
        if hasattr(x, "shape"):
            return dict(kind=type(x).__name__, shape=list(x.shape),
                        dtype=str(getattr(x, "dtype", None)))
        if isinstance(x, (tuple, list)):
            return dict(kind=type(x).__name__, n=len(x),
                        items=[describe(v) for v in x[:3]])
        return dict(kind=type(x).__name__)
    if len(seen) < 40:
        seen.append(dict(inputs=[describe(v) for v in inputs[:2]],
                         output=describe(output)))


handle = blocks[0].gate.register_forward_hook(probe)

# ---- Gate A: what does the scheduler actually expose? --------------------
scheduler = engine.engine_core.engine_core.scheduler
report["scheduler_attrs"] = sorted(
    a for a in dir(scheduler) if not a.startswith("_")
    and a in ("running", "waiting", "requests", "max_num_running_reqs",
              "kv_cache_manager", "finished_req_ids"))

params = SamplingParams(n=1, temperature=0.0, max_tokens=24, min_tokens=24,
                        ignore_eos=True, detokenize=False,
                        output_kind=RequestOutputKind.CUMULATIVE)
for src, ids in zip(workload["source_requests"][:CAP],
                    workload["actual_prompt_token_ids"][:CAP]):
    engine.add_request(f"d/{src['request_id']}", {"prompt_token_ids": ids}, params)

trace = []
step = 0
while engine.has_unfinished_requests() and step < STEPS:
    running = list(scheduler.running)
    waiting = list(getattr(scheduler, "waiting", []))
    if step == 0 and running:
        r = running[0]
        report["request_attrs"] = sorted(
            a for a in dir(r) if not a.startswith("_") and (
                "token" in a or "prompt" in a or "computed" in a
                or "status" in a or "output" in a))
    rows = []
    for r in running:
        rows.append(dict(
            rid=getattr(r, "request_id", None),
            num_computed_tokens=getattr(r, "num_computed_tokens", "MISSING"),
            num_prompt_tokens=getattr(r, "num_prompt_tokens", "MISSING"),
            num_tokens=getattr(r, "num_tokens", "MISSING"),
            num_output_tokens=getattr(r, "num_output_tokens", "MISSING"),
            status=str(getattr(r, "status", "MISSING"))))
    before = len(seen)
    engine.step()
    trace.append(dict(step=step, n_running=len(running), n_waiting=len(waiting),
                      hook_calls_this_step=len(seen) - before,
                      requests=rows[:4]))
    step += 1

handle.remove()
report["trace"] = trace
report["hook_samples"] = seen[:8]
report["n_hook_calls"] = len(seen)
(OUT / "diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2)[:4000])

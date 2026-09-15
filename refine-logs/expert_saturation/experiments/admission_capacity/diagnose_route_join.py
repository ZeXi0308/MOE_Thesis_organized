#!/usr/bin/env python3
"""Why did the route join miss? Print both key sets side by side.

The collector harvested routes for all 32 requests yet every pure-decode step
reported `missing_route`, so `routes.get(rid)` returned None. Either the
scheduler's `Request.request_id` differs from `RequestOutput.request_id`, or
the routes arrive after the steps that need them are already recorded.

Small and cheap on purpose: 4 requests, 8 output tokens.
"""
import json
import os
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/joindbg")
PREPARED = Path(sys.argv[2] if len(sys.argv) > 2
                else "/root/autodl-tmp/expert-union/inputs_preparation/prepared/short")
CAP, NOUT = 4, 8

OUT.mkdir(parents=True, exist_ok=True)
os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_BATCH_INVARIANT="0",
                  VLLM_USE_FLASHINFER_SAMPLER="0")

import numpy as np
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
    enforce_eager=False, enable_return_routed_experts=True),
    enable_multiprocessing=False)

scheduler = engine.engine_core.engine_core.scheduler
params = SamplingParams(n=1, temperature=0.0, max_tokens=NOUT, min_tokens=NOUT,
                        ignore_eos=True, detokenize=False,
                        output_kind=RequestOutputKind.FINAL_ONLY)
added = []
for src, ids in zip(workload["source_requests"][:CAP],
                    workload["actual_prompt_token_ids"][:CAP]):
    rid = f"u/{src['request_id']}"
    added.append(rid)
    engine.add_request(rid, {"prompt_token_ids": ids}, params)

scheduler_rids, output_rids, routed_rids, trace = set(), set(), {}, []
step = 0
while engine.has_unfinished_requests() and step < 40:
    running = list(scheduler.running)
    for r in running:
        scheduler_rids.add(r.request_id)
    row = dict(step=step, n_running=len(running),
               sched_rids=[r.request_id for r in running][:2])
    outs = engine.step()
    finished = []
    for o in outs:
        output_rids.add(o.request_id)
        if o.finished:
            has = bool(o.outputs and o.outputs[0].routed_experts is not None)
            finished.append(dict(rid=o.request_id, has_routes=has))
            if has:
                routed_rids[o.request_id] = list(
                    np.asarray(o.outputs[0].routed_experts).shape)
    row["finished"] = finished
    trace.append(row)
    step += 1

report = dict(
    added=added,
    scheduler_rids=sorted(scheduler_rids),
    output_rids=sorted(output_rids),
    routed_rids={k: v for k, v in sorted(routed_rids.items())},
    added_equals_scheduler=sorted(added) == sorted(scheduler_rids),
    added_equals_output=sorted(added) == sorted(output_rids),
    routed_keys_in_scheduler=sorted(set(routed_rids) & scheduler_rids),
    routed_keys_not_in_scheduler=sorted(set(routed_rids) - scheduler_rids),
    # The decisive question: are routes available before the steps end?
    step_of_first_route=next((r["step"] for r in trace
                              if any(f["has_routes"] for f in r["finished"])), None),
    total_steps=len(trace), trace=trace[-6:])
(OUT / "joindbg.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2)[:3000])

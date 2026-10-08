#!/usr/bin/env python3
"""Is absence rotation implementable without engine surgery?

The measured dynamics (reproduced bit-identically across two repeats):

  * preemption at steps 809 and 931, exactly 122 steps apart, which equals
    237 over-evicted blocks divided by the survivors' 1.94 blocks/step;
  * the victim's 221-step absence ends at a COMPLETION, not at the next
    preemption -- B's preemption freed 245 blocks while A needed only 237,
    yet A stayed out for 94 more steps because the survivors absorbed them.

So the actionable decision is: who gets the blocks a preemption frees. The
proposed mechanism is to rotate the mandatory absence instead of concentrating
it, which needs three capabilities:

  R1  choose WHICH running request is preempted next;
  R2  choose WHICH waiting request resumes first;
  R3  observe free KV blocks online, to decide when a swap is affordable.

This probe establishes which of the three exist as list/queue reordering on
the live scheduler, and which would need new engine code. It changes nothing:
every structure is only inspected, and the episode is 4 requests long.

Writing a controller before knowing this would risk pretending a parameter is
already effective when it is not.
"""
import json
import os
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/schedprobe")
PREPARED = Path(sys.argv[2] if len(sys.argv) > 2
                else "/root/autodl-tmp/expert-union/inputs_preparation/prepared/short")

OUT.mkdir(parents=True, exist_ok=True)
os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_BATCH_INVARIANT="0",
                  VLLM_USE_FLASHINFER_SAMPLER="0")

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

sched = engine.engine_core.engine_core.scheduler
report = dict(scheduler_class=type(sched).__name__)

# ---- R1 / R2: are running and waiting order-controllable containers? ----
report["running_type"] = type(sched.running).__name__
report["waiting_type"] = type(getattr(sched, "waiting", None)).__name__
report["waiting_api"] = sorted(a for a in dir(getattr(sched, "waiting", object()))
                               if not a.startswith("_"))[:24]

# Anything on the scheduler that looks like an explicit preempt/resume hook.
report["scheduler_methods"] = sorted(
    a for a in dir(sched) if not a.startswith("__")
    and any(t in a.lower() for t in
            ("preempt", "resume", "evict", "free", "alloc", "policy",
             "priority", "schedule", "kv_cache", "running", "waiting", "finish")))

# ---- R3: can free blocks be observed? ----
kvm = getattr(sched, "kv_cache_manager", None)
report["kv_manager_class"] = type(kvm).__name__ if kvm is not None else None
if kvm is not None:
    report["kv_manager_attrs"] = sorted(
        a for a in dir(kvm) if not a.startswith("_")
        and any(t in a.lower() for t in
                ("free", "block", "usage", "alloc", "num", "pool", "coordinator")))
    pool = getattr(kvm, "block_pool", None)
    report["block_pool_class"] = type(pool).__name__ if pool is not None else None
    if pool is not None:
        report["block_pool_attrs"] = sorted(
            a for a in dir(pool) if not a.startswith("_")
            and any(t in a.lower() for t in ("free", "num", "block", "get", "cache")))
        for name in ("get_num_free_blocks", "num_free_blocks", "num_gpu_blocks"):
            fn = getattr(pool, name, None)
            try:
                report[f"probe_{name}"] = fn() if callable(fn) else fn
            except Exception as exc:
                report[f"probe_{name}"] = f"ERR {type(exc).__name__}"

params = SamplingParams(n=1, temperature=0.0, max_tokens=10, min_tokens=10,
                        ignore_eos=True, detokenize=False,
                        output_kind=RequestOutputKind.FINAL_ONLY)
for src, ids in zip(workload["source_requests"][:4],
                    workload["actual_prompt_token_ids"][:4]):
    engine.add_request(f"q/{src['request_id']}", {"prompt_token_ids": ids}, params)

# ---- does the order of `running` decide the victim? observe, do not mutate ----
trace = []
step = 0
while engine.has_unfinished_requests() and step < 14:
    running = list(sched.running)
    pool = getattr(kvm, "block_pool", None) if kvm else None
    free = None
    if pool is not None:
        fn = getattr(pool, "get_num_free_blocks", None)
        try:
            free = fn() if callable(fn) else getattr(pool, "num_free_blocks", None)
        except Exception:
            free = None
    row = dict(step=step, free_blocks=free,
               running_order=[r.request_id.split("-")[-1] for r in running],
               n_waiting=len(getattr(sched, "waiting", [])),
               # If reordering is to control victims, the list must be plain and
               # its tail must be what the allocator sacrifices first.
               running_is_list=isinstance(sched.running, list))
    engine.step()
    trace.append(row)
    step += 1

report["trace"] = trace
verdict = dict(
    R1_choose_victim=("list reorder plausible" if report["running_type"] == "list"
                      else f'needs new code: running is {report["running_type"]}'),
    R2_choose_resumer=(f'{report["waiting_type"]} - check ordering API'),
    R3_observe_free_blocks=("yes" if any(r["free_blocks"] is not None for r in trace)
                            else "NOT observable through probed attributes"))
report["verdict"] = verdict
(OUT / "schedprobe.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
print(json.dumps(report, indent=2, default=str)[:4000])

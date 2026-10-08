#!/usr/bin/env python3
"""Probe vLLM's native routed-expert return path.

Python forward hooks on the router cannot observe decode routing in this
engine: `enforce_eager=False` means every decode step of width <= 32 is a
CUDA-graph *replay*, and a replay executes no Python. The diagnosis run
confirmed this directly (n_hook_calls = 0 while 8 requests decoded for 12
steps with correct purity state).

Turning graphs off would make routes observable but would change the
execution regime the rest of this repository measured, so the engine's own
`enable_return_routed_experts` path is preferred: it is designed for this and
works inside captured graphs.

This probe only establishes the *shape and semantics* of what comes back:
where the routed experts appear on the output object, how they are indexed
(per token? per layer? cumulative or delta?), and whether the count matches
`num_experts_per_tok`. No union statistic is computed here.
"""
import json
import os
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/probe-routed")
PREPARED = Path(sys.argv[2] if len(sys.argv) > 2
                else "/root/autodl-tmp/expert-union/inputs_preparation/prepared/short")
CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 4

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
    enforce_eager=False,
    enable_return_routed_experts=True), enable_multiprocessing=False)


def describe(x, depth=0):
    if hasattr(x, "shape"):
        return dict(kind=type(x).__name__, shape=list(x.shape),
                    dtype=str(getattr(x, "dtype", None)))
    if isinstance(x, (tuple, list)):
        return dict(kind=type(x).__name__, n=len(x),
                    first=describe(x[0], depth + 1) if x and depth < 3 else None)
    if isinstance(x, dict):
        return dict(kind="dict", keys=sorted(x)[:12])
    return dict(kind=type(x).__name__, repr=repr(x)[:120])


params = SamplingParams(n=1, temperature=0.0, max_tokens=6, min_tokens=6,
                        ignore_eos=True, detokenize=False,
                        output_kind=RequestOutputKind.DELTA)
for src, ids in zip(workload["source_requests"][:CAP],
                    workload["actual_prompt_token_ids"][:CAP]):
    engine.add_request(f"p/{src['request_id']}", {"prompt_token_ids": ids}, params)

report = dict(sampling_params_fields=[f for f in dir(params)
                                      if "rout" in f.lower() or "expert" in f.lower()],
              steps=[])
step = 0
while engine.has_unfinished_requests() and step < 12:
    outs = engine.step()
    row = dict(step=step, n_outputs=len(outs))
    for o in outs[:1]:
        row["output_attrs"] = sorted(a for a in dir(o) if not a.startswith("_")
                                     and ("rout" in a.lower() or "expert" in a.lower()))
        row["output_top"] = sorted(a for a in dir(o) if not a.startswith("_"))[:20]
        for name in ("routed_experts", "num_cached_tokens", "outputs"):
            if hasattr(o, name):
                row[f"has_{name}"] = describe(getattr(o, name))
        if getattr(o, "outputs", None):
            c = o.outputs[0]
            row["completion_attrs"] = sorted(
                a for a in dir(c) if not a.startswith("_")
                and ("rout" in a.lower() or "expert" in a.lower() or a == "token_ids"))
            for name in ("routed_experts",):
                if hasattr(c, name):
                    row[f"completion_{name}"] = describe(getattr(c, name))
                    val = getattr(c, name)
                    # Keep one small concrete sample so the indexing is unambiguous.
                    try:
                        row["sample"] = json.loads(json.dumps(val))[:2] \
                            if isinstance(val, list) else str(val)[:400]
                    except Exception:
                        row["sample"] = str(val)[:400]
    report["steps"].append(row)
    step += 1

(OUT / "probe.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
print(json.dumps(report, indent=2, default=str)[:5000])

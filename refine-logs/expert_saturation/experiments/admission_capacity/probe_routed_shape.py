#!/usr/bin/env python3
"""Pin down the shape and index semantics of vLLM's routed_experts array.

The probe established that `CompletionOutput.routed_experts` is populated only
on the FINAL output of a request (an ndarray of expert ids in [0, 64)). Before
any union statistic is computed, the axis order must be unambiguous: which
axis is token, which is layer, which is top-k, and whether prompt tokens are
included.

Two requests with deliberately different prompt and output lengths make the
token axis identifiable by arithmetic rather than by assumption.
"""
import json
import os
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/shape-routed")
PREPARED = Path(sys.argv[2] if len(sys.argv) > 2
                else "/root/autodl-tmp/expert-union/inputs_preparation/prepared/short")

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

# Distinct output lengths so the token axis is identifiable by arithmetic.
PLAN = [("A", 5), ("B", 11)]
prompt_ids = workload["actual_prompt_token_ids"][0]
for tag, n_out in PLAN:
    engine.add_request(
        f"s/{tag}", {"prompt_token_ids": prompt_ids},
        SamplingParams(n=1, temperature=0.0, max_tokens=n_out, min_tokens=n_out,
                       ignore_eos=True, detokenize=False,
                       output_kind=RequestOutputKind.FINAL_ONLY))

report = dict(prompt_tokens=len(prompt_ids),
              plan={t: n for t, n in PLAN},
              num_experts=64, num_experts_per_tok=8, n_layers=16, finals=[])
while engine.has_unfinished_requests():
    for o in engine.step():
        if not o.finished:
            continue
        c = o.outputs[0]
        arr = c.routed_experts
        if arr is None:
            report["finals"].append(dict(rid=o.request_id, routed="None"))
            continue
        arr = np.asarray(arr)
        n_out = len(c.token_ids)
        report["finals"].append(dict(
            rid=o.request_id, n_output_tokens=n_out,
            shape=list(arr.shape), dtype=str(arr.dtype),
            size=int(arr.size), min=int(arr.min()), max=int(arr.max()),
            # Identify each axis by matching it against known quantities.
            axis_matches=[dict(axis=i, size=int(s),
                               equals_output_tokens=(s == n_out),
                               equals_prompt_plus_output=(s == len(prompt_ids) + n_out),
                               equals_prompt=(s == len(prompt_ids)),
                               equals_layers=(s == 16), equals_topk=(s == 8))
                          for i, s in enumerate(arr.shape)],
            # A top-k axis must have no duplicate ids within a slice.
            last_axis_unique_always=bool(
                all(len(set(row)) == arr.shape[-1]
                    for row in arr.reshape(-1, arr.shape[-1])[:256])),
            first_slice=arr.reshape(-1, arr.shape[-1])[0].tolist(),
            second_slice=arr.reshape(-1, arr.shape[-1])[1].tolist()))

(OUT / "shape.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))

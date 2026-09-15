#!/usr/bin/env python3
"""Collect batched expert-union statistics from one native decode episode.

Pure measurement. No admission action, no paging, no router/top-k/precision
change. Reuses the frozen inputs and engine arguments of the memory-pressure
experiments so the routing observed here belongs to the same operating regime
as the sealed capacity results.

Why not a forward hook
----------------------
An earlier version registered a forward hook on each MoE router. It recorded
nothing: with `enforce_eager=False` every decode step at width <= 32 is a
CUDA-graph *replay*, and a replay runs no Python. The diagnosis was direct --
zero hook calls across 12 steps with 8 requests decoding and the purity state
correct (`diagnose_union_hook.py`). Disabling graphs would restore visibility
but would change the execution regime every other measurement in this
repository was taken under.

This version uses the engine's own `enable_return_routed_experts`, which works
inside captured graphs. Routes are therefore what actually executed, not a
recomputation from router logits, so there is no top-k tie-break ambiguity.

Index semantics (verified, not assumed)
---------------------------------------
`CompletionOutput.routed_experts` is populated on the FINAL output of a
request, as a uint8 array of shape

    [P + n_out - 1, n_layers, top_k]

`probe_routed_shape.py` confirmed the leading axis on two requests with
different output lengths (132 = 128+5-1, 138 = 128+11-1). Index `c` is the
forward pass at position `c`; positions `0..P-1` are prompt, `P..` are decode.
The final sampled token has no forward pass, hence the `-1`.

A scheduler step that advances a decoding request whose `num_computed_tokens`
is `c` computes exactly position `c`. That makes the join between step
composition and routing exact and uses no future information: the step
composition is what the online scheduler did, and the routes are what the GPU
executed.

Only compact per-step statistics are exported. Per-token routes are never
written to disk.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from expert_union_tracker import ExpertUnionTracker


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read(path):
    return json.loads(path.read_text())


def gpu_state():
    device = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,memory.total,memory.used,"
         "temperature.gpu,power.draw,clocks.sm", "--format=csv,noheader"],
        text=True).strip()
    procs = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader"], text=True).strip()
    return dict(device=device, compute_processes=procs)


def load_inputs(root, domain, prepared_dir=None):
    """Locate the frozen inputs and re-verify their identity."""
    folder = Path(prepared_dir) if prepared_dir is not None else \
        root / "inputs_preparation/prepared" / domain
    if not (folder / "workload.json").exists():
        raise FileNotFoundError(f"no workload.json under {folder}")
    config, workload = read(folder / "config.json"), read(folder / "workload.json")
    if hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest() \
            != config["workload_sha256"]:
        raise ValueError("workload changed")
    expected = 128 if domain == "short" else 3072
    for row, ids in zip(workload["source_requests"], workload["actual_prompt_token_ids"]):
        if len(ids) != expected or hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()).hexdigest() \
                != row["prompt_token_ids_sha256"]:
            raise ValueError("prompt identity mismatch")
    return config, workload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["short", "long"], required=True)
    parser.add_argument("--cap", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-tokens", type=int, default=None)
    parser.add_argument("--prepared-dir", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=4000)
    args = parser.parse_args()

    root, out = Path(__file__).resolve().parent, args.output_dir
    config, workload = load_inputs(root, args.domain, args.prepared_dir)
    if args.output_tokens:
        config = dict(config, output_tokens=args.output_tokens)
    config = dict(config, cap=args.cap, domain=args.domain, engine_max_num_seqs=32,
                  evidence_ceiling="NATIVE_INPROCESS_STRUCTURAL_ROUTE_MEASUREMENT",
                  measurement_role="expert union is a structural signal, "
                                   "not measured HBM traffic")
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "config.json", config)
    (out / "commands.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    dump(out / "status.json", dict(status="INITIALIZING"))

    try:
        before = gpu_state()
        os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="0",
                          VLLM_BATCH_INVARIANT="0", VLLM_USE_FLASHINFER_SAMPLER="0")
        import numpy as np
        import torch
        import vllm
        from importlib.metadata import version
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        if vllm.__version__ != "0.26.0" or torch.cuda.device_count() != 1:
            raise RuntimeError("requires the pinned vLLM 0.26 one-GPU environment")
        dump(out / "environment.json", dict(
            python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
            vllm=vllm.__version__, transformers=version("transformers"),
            gpu_before=before, cpu_threads=torch.get_num_threads(),
            source_sha256={n: hashlib.sha256((root / n).read_bytes()).hexdigest()
                           for n in ["run_expert_union.py", "expert_union_tracker.py"]}))

        model = config["model"]
        kwargs = dict(model=model["id"], revision=model["revision"],
                      tokenizer_revision=model["tokenizer_revision"],
                      dtype="bfloat16", seed=config["seed"], max_model_len=4096,
                      max_num_seqs=32, max_num_batched_tokens=1024,
                      gpu_memory_utilization=0.90, enable_chunked_prefill=True,
                      enable_prefix_caching=False, scheduling_policy="fcfs",
                      async_scheduling=False, stream_interval=1, enforce_eager=False,
                      enable_return_routed_experts=True)
        dump(out / "engine_args.json", kwargs)
        engine = LLMEngine.from_engine_args(EngineArgs(**kwargs),
                                            enable_multiprocessing=False)

        runner = engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        model_obj = runner.model
        blocks = [m for m in model_obj.modules()
                  if type(m).__name__.endswith("SparseMoeBlock")] or \
                 [m for m in model_obj.modules()
                  if hasattr(m, "gate") and hasattr(m, "experts")]
        n_experts = model_obj.config.num_experts
        top_k = model_obj.config.num_experts_per_tok
        dump(out / "model_shape.json", dict(
            n_moe_blocks=len(blocks), num_experts=n_experts,
            num_experts_per_tok=top_k,
            block_type=type(blocks[0]).__name__ if blocks else None))

        count = config["output_tokens"]
        sources = workload["source_requests"][:args.cap]
        prompts = workload["actual_prompt_token_ids"][:args.cap]
        params = SamplingParams(n=1, temperature=0.0, max_tokens=count,
                                min_tokens=count, ignore_eos=True, detokenize=False,
                                output_kind=RequestOutputKind.FINAL_ONLY)
        scheduler = engine.engine_core.engine_core.scheduler
        scheduler.max_num_running_reqs = args.cap
        id_map = {}
        for source, ids in zip(sources, prompts):
            rid = f"u/{source['request_id']}"
            id_map[rid] = dict(request_id=source["request_id"],
                               prompt_tokens=len(ids))
            engine.add_request(rid, {"prompt_token_ids": ids}, params)

        # ---- online phase: record step composition, harvest routes on finish ----
        # The engine appends an 8-hex-digit suffix to the internal
        # `Request.request_id` (`u/x` becomes `u/x-884bdf35`) while
        # `RequestOutput.request_id` keeps the id we submitted. Joining on the
        # raw strings silently matches nothing, so the internal id is
        # normalised back to the submitted one and the mapping is asserted to
        # be one-to-one before it is used.
        import re
        suffix = re.compile(r"-[0-9a-f]{8}$")

        def external_id(internal):
            base = suffix.sub("", internal)
            return base if base in id_map else internal

        steps, routes = [], {}
        step_index = 0
        while engine.has_unfinished_requests() and step_index < args.max_steps:
            running = list(scheduler.running)
            decoding, prefilling = [], []
            for r in running:
                (decoding if r.num_computed_tokens >= r.num_prompt_tokens
                 else prefilling).append(r)
            steps.append(dict(
                step=step_index,
                n_waiting=len(getattr(scheduler, "waiting", [])),
                n_prefilling=len(prefilling), decode_width=len(decoding),
                # `computed_before` IS the route index this step computes.
                decoding=[dict(rid=external_id(r.request_id),
                               internal_rid=r.request_id,
                               computed_before=int(r.num_computed_tokens))
                          for r in decoding]))
            for o in engine.step():
                if o.finished and o.outputs and o.outputs[0].routed_experts is not None:
                    routes[o.request_id] = np.asarray(o.outputs[0].routed_experts)
            step_index += 1

        # Fail loudly rather than reporting NO_DATA if the mapping is wrong.
        seen_ids = {d["rid"] for rec in steps for d in rec["decoding"]}
        unmatched = sorted(seen_ids - set(id_map))
        if unmatched:
            raise RuntimeError(
                f"step ids do not normalise into submitted ids: {unmatched[:3]}")

        dump(out / "gpu-after.json", gpu_state())

        # ---- offline join: union per (step, layer) over pure-decode steps ----
        tracker = ExpertUnionTracker(experts_total=n_experts, experts_per_token=top_k)
        joined, skipped = [], dict(mixed_or_queued=0, no_decode=0, missing_route=0,
                                   dropped_request_index_past_end=0,
                                   step_lost_all_requests=0)
        for rec in steps:
            if rec["decode_width"] <= 0:
                skipped["no_decode"] += 1
                continue
            if rec["n_prefilling"] or rec["n_waiting"]:
                # A mixed step genuinely needs the union over all its tokens, but
                # mixing a 1024-token prefill chunk into a width-16 decode union
                # would make U a function of chunk size, not of decode width.
                skipped["mixed_or_queued"] += 1
                continue
            picks = []
            for d in rec["decoding"]:
                arr = routes.get(d["rid"])
                if arr is None:
                    skipped["missing_route"] += 1
                    continue
                idx = d["computed_before"]
                if idx >= arr.shape[0]:
                    # The final sampled token of a request has no forward pass,
                    # so its last decode step has no routing row. Drop that one
                    # request rather than the whole step, and record how often.
                    skipped["dropped_request_index_past_end"] += 1
                    continue
                picks.append(arr[idx])
            if not picks:
                skipped["step_lost_all_requests"] += 1
                continue
            for layer in range(picks[0].shape[0]):
                tracker.record_step(
                    layer=layer,
                    selected_per_token=[tuple(sorted(int(e) for e in p[layer]))
                                        for p in picks])
            # `n_tokens_used` may be below decode_width when a request was in its
            # final step; the union is over the tokens actually joined.
            joined.append(dict(step=rec["step"], decode_width=rec["decode_width"],
                               n_tokens_used=len(picks)))

        summary = tracker.summary()
        verdict = tracker.verdict()
        dump(out / "union_summary.json", dict(
            summary=summary, verdict=verdict,
            experts_total=n_experts, experts_per_token=top_k,
            n_recorded_steps=len(joined),
            n_engine_steps=len(steps), n_requests_with_routes=len(routes),
            step_records=joined[:2000],
            collection_discipline=dict(
                source="engine enable_return_routed_experts (executed routes, "
                       "not recomputed from logits)",
                pure_decode_steps_only=True, skipped_steps=skipped,
                route_index_rule="route index = num_computed_tokens before the step; "
                                 "verified via T = prompt + n_out - 1",
                note="mixed prefill+decode steps are skipped, not recorded, so U is "
                     "a function of decode width and never of chunk size"),
            truncation=dict(max_steps=args.max_steps,
                            episode_truncated=engine.has_unfinished_requests()),
            note="union is structural: it is not measured HBM traffic, not a claim "
                 "about what the fused backend loads, and not evidence that idle "
                 "expert bytes are reclaimable in practice"))
        dump(out / "status.json", dict(status="COMPLETE", verdict=verdict["verdict"],
                                       n_recorded_steps=len(joined),
                                       scientific_verdict="MEASUREMENT_ONLY"))
        print(json.dumps(verdict, indent=2))
    except Exception as exc:
        previous = read(out / "status.json") if (out / "status.json").exists() else {}
        dump(out / "status.json", dict(previous, status="INCOMPLETE",
                                       error=f"{type(exc).__name__}: {exc}"))
        raise


if __name__ == "__main__":
    main()

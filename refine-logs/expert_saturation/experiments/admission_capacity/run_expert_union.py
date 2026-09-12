#!/usr/bin/env python3
"""Collect batched expert-union statistics from one native decode episode.

Pure measurement. No admission action, no paging, no router/top-k/precision change.
Reuses the frozen inputs, engine arguments and warmup discipline of the
memory-pressure experiment so the routing observed here belongs to the same
operating regime as the sealed capacity results.

Routing is recomputed from the model's own router logits with its own top-k rule,
per layer, per decode step, over exactly the tokens the native scheduler advanced
in that step. The union is therefore the set of experts that step genuinely needed
at that layer.

Only compact per-step statistics are exported. Per-token routes are never written
to disk: for 32 requests x 1024 tokens x 16 layers that would be tens of millions
of rows and would dominate the artifact for no analytical gain.
"""
import argparse
import csv
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
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader"], text=True).strip()
    if any(row and int(row[0]) != os.getpid() for row in csv.reader(processes.splitlines())):
        raise RuntimeError("GPU has another compute process")
    return dict(compute_processes=processes, device=subprocess.check_output(["nvidia-smi",
        "--query-gpu=name,uuid,memory.total,memory.used,temperature.gpu,power.draw,clocks.sm",
        "--format=csv,noheader"], text=True).strip())


def load_inputs(root, domain):
    folder = root / "inputs_preparation/prepared" / domain
    config, workload = read(folder / "config.json"), read(folder / "workload.json")
    if hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest() != config["workload_sha256"]:
        raise ValueError("workload changed")
    rows, tokens = workload["source_requests"], workload["actual_prompt_token_ids"]
    expected = 128 if domain == "short" else 3072
    for row, ids in zip(rows, tokens):
        if len(ids) != expected or hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest() != row["prompt_token_ids_sha256"]:
            raise ValueError("prompt identity mismatch")
    return config, workload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["short", "long"], required=True)
    parser.add_argument("--cap", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=400,
                        help="stop collecting routes after this many decode steps")
    parser.add_argument("--output-tokens", type=int, default=None,
                        help="override fixed output length to bound episode time")
    args = parser.parse_args()
    root, out = Path(__file__).resolve().parent, args.output_dir
    config, workload = load_inputs(root, args.domain)
    if args.output_tokens:
        config = dict(config, output_tokens=args.output_tokens)
    config = dict(config, cap=args.cap, domain=args.domain, engine_max_num_seqs=32,
        max_route_steps=args.max_steps,
        evidence_ceiling="NATIVE_INPROCESS_STRUCTURAL_ROUTE_MEASUREMENT",
        measurement_role="expert union is a structural signal, not measured HBM traffic",
        max_seconds=600)
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "config.json", config)
    (out / "commands.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    dump(out / "status.json", dict(status="INITIALIZING"))
    engine = None
    try:
        before = gpu_state()
        os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_BATCH_INVARIANT="0",
                          VLLM_USE_FLASHINFER_SAMPLER="0")
        import torch
        import vllm
        from importlib.metadata import version
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine
        if vllm.__version__ != "0.26.0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("requires the pinned vLLM0.26 one-GPU environment")
        dump(out / "environment.json", dict(python=sys.version, torch=torch.__version__,
            cuda=torch.version.cuda, vllm=vllm.__version__, transformers=version("transformers"),
            gpu_before=before, cpu_threads=torch.get_num_threads(),
            source_sha256={n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in
                ["run_expert_union.py", "expert_union_tracker.py"]}))
        model = config["model"]
        kwargs = dict(model=model["id"], revision=model["revision"],
            tokenizer_revision=model["tokenizer_revision"], dtype="bfloat16", seed=config["seed"],
            max_model_len=4096, max_num_seqs=32, max_num_batched_tokens=1024,
            gpu_memory_utilization=0.90, enable_chunked_prefill=True,
            enable_prefix_caching=False, scheduling_policy="fcfs", async_scheduling=False,
            stream_interval=1, enforce_eager=False, enable_return_routed_experts=False)
        dump(out / "engine_args.json", kwargs)
        engine = LLMEngine.from_engine_args(EngineArgs(**kwargs), enable_multiprocessing=False)

        runner = engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        model_obj = runner.model
        layers = [m for m in model_obj.modules() if type(m).__name__.endswith("SparseMoeBlock")]
        if not layers:
            layers = [m for m in model_obj.modules() if hasattr(m, "gate") and hasattr(m, "experts")]
        if not layers:
            raise RuntimeError("could not locate MoE blocks in the loaded model")
        top_k = getattr(layers[0], "top_k", None) or model_obj.config.num_experts_per_tok
        n_experts = model_obj.config.num_experts
        tracker = ExpertUnionTracker(experts_total=n_experts, experts_per_token=top_k)
        dump(out / "model_shape.json", dict(n_moe_blocks=len(layers), num_experts=n_experts,
            num_experts_per_tok=top_k, block_type=type(layers[0]).__name__))

        state = dict(collect=False, step=0, decode_tokens=0)
        handles = []

        def make_hook(index):
            def hook(module, inputs, output):
                # Router logits for the tokens in this forward; shape [T, E].
                if not state["collect"]:
                    return
                logits = output[0] if isinstance(output, tuple) else output
                if logits.dim() != 2 or logits.shape[-1] != n_experts:
                    return
                chosen = torch.topk(logits.float(), top_k, dim=-1).indices
                rows = [tuple(sorted(int(v) for v in row)) for row in chosen.tolist()]
                tracker.record_step(layer=index, selected_per_token=rows)
            return hook

        for index, block in enumerate(layers):
            gate = getattr(block, "gate", None)
            if gate is None:
                raise RuntimeError(f"MoE block {index} has no gate module")
            handles.append(gate.register_forward_hook(make_hook(index)))

        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        count = config["output_tokens"]
        sources = workload["source_requests"][:args.cap]
        prompts = workload["actual_prompt_token_ids"][:args.cap]
        params = SamplingParams(n=1, temperature=0.0, max_tokens=count, min_tokens=count,
                                ignore_eos=True, detokenize=False,
                                output_kind=RequestOutputKind.CUMULATIVE)
        scheduler = engine.engine_core.engine_core.scheduler
        scheduler.max_num_running_reqs = args.cap
        for source, ids in zip(sources, prompts):
            engine.add_request(f"u/{source['request_id']}", {"prompt_token_ids": ids}, params)

        # Prefill first with collection off, so the recorded unions are decode-only.
        step_records = []
        while engine.has_unfinished_requests() and state["step"] < args.max_steps:
            widths = sum(1 for r in scheduler.running
                         if r.num_computed_tokens >= r.num_prompt_tokens)
            state["collect"] = widths > 0
            before_layers = {l: len(tracker.steps.get(l, [])) for l in range(len(layers))}
            engine.step()
            if state["collect"]:
                added = {l: len(tracker.steps.get(l, [])) - before_layers[l]
                         for l in range(len(layers))}
                if any(added.values()):
                    state["step"] += 1
                    step_records.append(dict(step=state["step"], decode_width=widths,
                                             layers_recorded=sum(1 for v in added.values() if v)))
        for handle in handles:
            handle.remove()
        state["collect"] = False

        summary = tracker.summary()
        verdict = tracker.verdict()
        dump(out / "union_summary.json", dict(summary=summary, verdict=verdict,
            step_records=step_records, n_recorded_steps=state["step"],
            experts_total=n_experts, experts_per_token=top_k,
            note="union is structural: recomputed from router logits, not measured HBM traffic"))
        dump(out / "gpu-after.json", gpu_state())
        dump(out / "status.json", dict(status="COMPLETE", verdict=verdict["verdict"],
            n_recorded_steps=state["step"], scientific_verdict="MEASUREMENT_ONLY"))
        print(json.dumps(verdict, indent=2))
    except Exception as exc:
        previous = read(out / "status.json") if (out / "status.json").exists() else {}
        dump(out / "status.json", dict(previous, status="INCOMPLETE",
                                       error=f"{type(exc).__name__}: {exc}"))
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Two-request OLMoE/BF16 integration probe for WiSP and vLLM 0.11.2.

Run each mode in a fresh process. This measures host-observed token receipt
times; pager counters are not measurements of exposed H2D time. No downloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback


PROMPTS = ["The capital of France is", "Explain how a computer stores data:"]


def worker_snapshot(worker, reset=False):
    """Executed inside the model worker, including when RPC crosses processes."""
    import sys
    import torch

    torch.cuda.synchronize()
    runner = worker.model_runner
    kv = runner.kv_cache_config
    module = sys.modules.get("wisp.integrations.vllm.fused_moe")
    states = getattr(module, "_LAYER_STATES", [])
    layers = []
    for state in states:
        counters = {key: int(getattr(state, key)) for key in state.__slots__
                    if key.startswith("stats_")}
        tensors = (state.cpu_w13, state.cpu_w2)
        per_expert = sum(t[0].numel() * t.element_size() for t in tensors)
        layers.append(dict(layer_idx=state.layer_idx, cap=state.cap_experts,
            num_experts=state.num_experts, mode=state.mode, counters=counters,
            master_shapes=[list(t.shape) for t in tensors],
            master_pinned=all(t.is_pinned() for t in tensors),
            master_bytes=sum(t.numel() * t.element_size() for t in tensors),
            scratch_bytes=sum(t.numel() * t.element_size()
                              for t in (state.scratch_w13, state.scratch_w2)),
            weight_copy_payload_bytes=counters["stats_miss"] * per_expert,
            resident_experts=sorted(state.expert_to_slot)))
        if reset:
            for key in counters:
                setattr(state, key, 0)
    result = dict(pid=os.getpid(), pager_layers=layers,
        gpu_name=torch.cuda.get_device_name(),
        gpu_capability=list(torch.cuda.get_device_capability()),
        num_gpu_blocks=int(kv.num_blocks),
        kv_allocated_tensor_bytes=sum(int(t.size) for t in kv.kv_cache_tensors),
        kv_groups=[dict(layer_names=list(g.layer_names),
                        block_size=int(g.kv_cache_spec.block_size))
                   for g in kv.kv_cache_groups],
        cuda_allocated_bytes=torch.cuda.memory_allocated(),
        cuda_reserved_bytes=torch.cuda.memory_reserved(),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        counters_reset_after_snapshot=reset)
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return result


def run_requests(engine, prompt_ids, params, phase):
    """Keep partial requests in phase so exceptions preserve every observed token."""
    origin, wall_origin = time.perf_counter(), time.time()
    phase.update(status="RUNNING", origin_unix_s=wall_origin, requests=[])
    for index, ids in enumerate(prompt_ids):
        rid = f"{phase['name']}/{index}"
        row = dict(request_id=rid, prompt=PROMPTS[index], prompt_token_ids=ids,
                   arrival_s=0.0, output_token_ids=[], token_received_s=[],
                   finished=False)
        phase["requests"].append(row)
        row["submit_s"] = time.perf_counter() - origin
        engine.add_request(rid, {"prompt_token_ids": ids}, params,
                           arrival_time=wall_origin)
        row["submit_return_s"] = time.perf_counter() - origin
    by_id = {row["request_id"]: row for row in phase["requests"]}
    while engine.has_unfinished_requests():
        outputs = engine.step()
        received = time.perf_counter() - origin
        for output in outputs:
            row = by_id[output.request_id]
            if len(output.outputs) != 1 or row["finished"]:
                raise RuntimeError("Unexpected output count or output after completion")
            completion = output.outputs[0]
            ids = list(completion.token_ids)
            previous = row["output_token_ids"]
            if ids[:len(previous)] != previous or len(ids) < len(previous):
                raise RuntimeError("Cumulative token sequence changed")
            row["token_received_s"].extend([received] * (len(ids) - len(previous)))
            row["output_token_ids"] = ids
            row["finished"] = bool(output.finished)
            if output.finished:
                row.update(completion_s=received, finish_reason=completion.finish_reason,
                           stop_reason=completion.stop_reason)
    phase["wall_s"] = time.perf_counter() - origin
    for row in phase["requests"]:
        times = row["token_received_s"]
        row.update(ttft_s=times[0] if times else None,
                   tpot_s=(times[-1] - times[0]) / (len(times) - 1)
                   if len(times) > 1 else None,
                   max_itl_s=max((b - a for a, b in zip(times, times[1:])), default=None))
        if not row["finished"] or len(row["output_token_ids"]) != params.max_tokens:
            raise RuntimeError("Request incomplete or generated token count differs")
    phase["status"] = "COMPLETED"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=("vanilla", "paged"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cap-experts", type=int, default=16)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=512 * 1024**2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Reserve a new artifact; never replace a previous run.
    with args.out.open("x") as handle:
        handle.write('{}\n')
    result = dict(status="STARTING", evidence_type="NATIVE_SERVING_INTEGRATION_PROBE",
                  args={k: str(v) if isinstance(v, Path) else v
                        for k, v in vars(args).items()}, phases={})

    def save():
        pending = args.out.with_name(args.out.name + ".pending")
        pending.write_text(json.dumps(result, indent=2) + "\n")
        pending.replace(args.out)

    def terminate(signum, frame):
        raise RuntimeError(f"Interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    save()
    try:
        if not args.model.is_dir() or not (args.model / "config.json").is_file():
            raise ValueError("--model must be a complete existing local model directory")
        if not 8 <= args.cap_experts <= 64 or args.kv_cache_memory_bytes <= 0:
            raise ValueError("OLMoE requires cap 8..64 and a positive KV byte budget")
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            VLLM_USE_V1="1", VLLM_ENABLE_V1_MULTIPROCESSING="0",
            WISP_PLUGIN_DISABLE="1" if args.mode == "vanilla" else "0",
            WISP_MODE="paged", WISP_CAP_EXPERTS=str(args.cap_experts),
            WISP_PREFETCH="0", WISP_DYNAMIC="0")
        import importlib.metadata
        import torch
        import vllm
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import RequestOutputKind

        result["runtime"] = dict(python=sys.version, vllm=vllm.__version__,
            torch=torch.__version__, cuda=torch.version.cuda,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if vllm.__version__.split("+")[0] != "0.11.2":
            raise RuntimeError("This probe requires vLLM 0.11.2")
        if args.mode == "paged":
            from wisp.integrations.vllm import install_wisp_moe
            install_wisp_moe(mode="paged", cap_experts=args.cap_experts)
            result["runtime"]["wisp"] = importlib.metadata.version("wisp-moe")
        model_config = json.loads((args.model / "config.json").read_text())
        if model_config.get("model_type") != "olmoe" or model_config.get("num_hidden_layers") != 16:
            raise ValueError("This probe is restricted to the 16-layer OLMoE checkpoint")
        config = dict(model=str(args.model.resolve()), dtype="bfloat16", seed=args.seed,
            enforce_eager=True, max_model_len=256, max_num_seqs=2,
            max_num_batched_tokens=32, enable_chunked_prefill=True,
            enable_prefix_caching=False, kv_cache_memory_bytes=args.kv_cache_memory_bytes,
            gpu_memory_utilization=args.gpu_memory_utilization, block_size=16,
            tensor_parallel_size=1, trust_remote_code=False, disable_log_stats=True)
        result.update(status="LOADING", engine_config=config, model_config=model_config)
        save()
        start = time.perf_counter()
        llm = LLM(**config)
        result["load_wall_s"] = time.perf_counter() - start
        engine = llm.llm_engine
        snapshot = lambda reset=False: engine.engine_core.collective_rpc(
            worker_snapshot, args=(reset,))
        result["initialization_snapshot"] = snapshot(True)
        for worker in result["initialization_snapshot"]:
            count = len(worker["pager_layers"])
            if count != (16 if args.mode == "paged" else 0):
                raise RuntimeError(f"Unexpected pager layer count: {count}")
            if worker["num_gpu_blocks"] <= 0:
                raise RuntimeError("No actual KV blocks reported")
            if any(layer["cap"] != args.cap_experts or not layer["master_pinned"]
                   for layer in worker["pager_layers"]):
                raise RuntimeError("Pager cap or pinned-master invariant failed")
        prompt_ids = [llm.get_tokenizer().encode(prompt) for prompt in PROMPTS]
        params = SamplingParams(temperature=0, seed=args.seed, max_tokens=8,
            min_tokens=8, ignore_eos=True, detokenize=False,
            output_kind=RequestOutputKind.CUMULATIVE)
        result["sampling_config"] = dict(temperature=0, seed=args.seed, max_tokens=8,
            min_tokens=8, ignore_eos=True, detokenize=False, output_kind="CUMULATIVE")
        for name in ("warmup", "measurement"):
            result["status"] = name.upper()
            phase = result["phases"][name] = dict(name=name)
            save()
            run_requests(engine, prompt_ids, params, phase)
            phase["worker_snapshot"] = snapshot(reset=name == "warmup")
            if any(layer["counters"]["stats_forward"] <= 0
                   for worker in phase["worker_snapshot"] for layer in worker["pager_layers"]):
                raise RuntimeError("Registered pager layer did not execute in this phase")
            save()
        result.update(status="COMPLETED", claim_ceiling="Full-model integration only",
            timing_definition="Host receipt after engine.step; includes engine/host work",
            cache_state="Measurement retains warmup residency; only counters reset")
    except (Exception, KeyboardInterrupt) as exc:
        for phase in result["phases"].values():
            if phase.get("status") == "RUNNING":
                phase["status"] = "FAILED"
        result.update(status="FAILED", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
        save()
        print(f"FAILED: {exc}; retained {args.out}", file=sys.stderr)
        return 1
    save()
    print(f"COMPLETED: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

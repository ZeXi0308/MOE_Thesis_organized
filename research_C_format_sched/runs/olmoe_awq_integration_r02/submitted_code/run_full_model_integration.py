#!/usr/bin/env python3
"""Full 16-layer OLMoE integration, one real sidecar arm per fresh process.

Run three separate run_locked jobs, selecting respectively:
  configs/olmoe_integration_r01/single_fp8.json
  configs/olmoe_integration_r01/single_awq.json
  configs/olmoe_integration_r01/dual_phase.json
All use checkpoints/olmoe_awq_integration_bf16_r01/checkpoint_manifest.json.

Example (choose a fresh job name for each arm):
  python code/run_locked.py --name integration_fp8_r01 --timeout 900 -- \
    python code/run_full_model_integration.py \
    --checkpoint-manifest checkpoints/olmoe_awq_integration_bf16_r01/checkpoint_manifest.json \
    --worker-config configs/olmoe_integration_r01/single_fp8.json \
    --output runs/integration_fp8_r01/integration \
    --lock-status runs/integration_fp8_r01/status.json --run

Without --run, resolve/validate local inputs and print a CPU-only plan. No
engine, GPU, server, download or model mutation occurs during plan/import.
The two fixed prompts first receive a one-token full-model forward, then
independent greedy generation with natural EOS and a 32-token cap. These
are integration observations, not quality scores or performance benchmarks.
API: installed vLLM 0.26 LLM.generate / collective_rpc; see
https://docs.vllm.ai/en/v0.26.0/api/vllm/entrypoints/llm/
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback

from eval_quality import digest_json, verify_checkpoint, worker_plan
from prepare_quantization import read_json, sha256, write_json


PROMPTS = (
    "Question: Why do plants need sunlight? Explain briefly.\nAnswer:",
    "Question: A store has 12 apples and sells 5. How many apples remain? Explain briefly.\nAnswer:",
)
LAYERS = {str(i) for i in range(16)}
STAGES = (("full_forward", 1), ("natural_generation", 32))


def worker_memory(worker):
    """Executed by collective_rpc on the actual worker, never on CPU planning."""
    import torch
    device = worker.device
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return {"cuda_allocated_bytes": torch.cuda.memory_allocated(device),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(device),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "device_free_bytes": free, "device_total_bytes": total,
            "model_memory_usage_bytes": int(worker.model_runner.model_memory_usage),
            "scope": "live worker allocator after KV initialization; peaks include loading/warmup; device free is not model-only"}


def plan(args):
    manifest = read_json(args.checkpoint_manifest)
    if manifest["model_key"] != "olmoe" or manifest["format"] != "bf16" or manifest.get("runtime_quantization"):
        raise ValueError("all integration arms require the same complete OLMoE common-basis BF16 host")
    root = verify_checkpoint(manifest, hash_weights=args.run)
    hf_config = read_json(root / "config.json")
    if hf_config.get("num_hidden_layers") != 16:
        raise ValueError("expected the complete 16-layer OLMoE checkpoint")
    worker = worker_plan(args.worker_config, args.output, hash_weights=args.run)
    config = worker["effective_config"]
    if {str(layer["layer"]) for layer in config["layers"]} != LAYERS:
        raise ValueError("worker config must cover exactly all 16 layers")
    single = config["policy"] == "single"
    if single and config["single_format"] not in ("fp8", "awq"):
        raise ValueError("integration single arms are expert FP8 and expert AWQ")
    if not single and (config["policy"] != "phase" or config["lowbit_format"] != "awq"):
        raise ValueError("the mixed integration arm must be real scheduler phase FP8/AWQ")
    # Preserve actual policy/basis settings; only bounded observation capacity
    # changes. Two <=32-token requests cannot need 256 ordinary decode steps.
    config["record_limit"] = max(256, config.get("record_limit", 1024))
    config["record_cuda_events"] = False
    kwargs = {"model": str(root), "tokenizer": str(root), "dtype": "bfloat16",
              "tensor_parallel_size": 1, "trust_remote_code": False,
              "gpu_memory_utilization": args.gpu_memory_utilization,
              "max_model_len": 1024, "max_num_seqs": 2,
              "max_num_batched_tokens": 1024, "enable_chunked_prefill": False,
              "enable_prefix_caching": False, "kv_cache_dtype": "auto",
              "seed": 20261016, "generation_config": "vllm",
              **worker["worker_model_kwargs"]}
    return {"schema_version": 1, "status": "planned", "arm": "single_" + config["single_format"] if single else "dual_phase",
            "checkpoint_manifest": str(Path(args.checkpoint_manifest).resolve()),
            "host_checkpoint_fingerprint": manifest["checkpoint_fingerprint"],
            "host_manifest_sha256": sha256(args.checkpoint_manifest),
            "host_calibration": manifest.get("calibration"),
            "host_runtime_quantization": None, "worker": worker, "llm_kwargs": kwargs,
            "prompts": list(PROMPTS), "prompt_template": "fixed raw Q/A completion text; tokenizer adds its own special tokens; no chat-template inference",
            "stages": [{"name": name, "max_tokens": cap} for name, cap in STAGES],
            "maximum_output_tokens_per_arm": 66,
            "sampling": {"temperature": 0.0, "ignore_eos": False, "min_tokens": 0, "logprobs": 1, "seed": 20261016},
            "scope": "actual full-model forward and short natural generation; no teacher forcing, quality score or serving-performance claim"}


def serialize_output(output, expected_prompt_ids, eos_token_id, cap):
    if list(output.prompt_token_ids) != expected_prompt_ids or len(output.outputs) != 1:
        raise RuntimeError("actual output prompt/order/completion count does not match the fixed request")
    completion = output.outputs[0]
    tokens = list(completion.token_ids)
    finish, stop = completion.finish_reason, completion.stop_reason
    status = ("unfinished" if not output.finished or finish is None else
              "cap" if finish == "length" else
              "eos" if finish == "stop" and stop is None and tokens and tokens[-1] == eos_token_id else
              "stop_string" if finish == "stop" and isinstance(stop, str) else
              "stop_token" if finish == "stop" and isinstance(stop, int) else
              "stop_unspecified" if finish == "stop" else "error")
    logs, selected_finite = [], True
    if completion.logprobs is None or len(completion.logprobs) != len(tokens):
        raise RuntimeError("requested generated-token logprobs are missing")
    for token, choices in zip(tokens, completion.logprobs, strict=True):
        if choices is None or token not in choices:
            raise RuntimeError("generated token is missing from its logprob record")
        selected_finite &= math.isfinite(float(choices[token].logprob))
        logs.append({str(key): {"logprob": float(value.logprob) if math.isfinite(float(value.logprob)) else str(value.logprob),
                              "rank": value.rank, "decoded_token": value.decoded_token}
                     for key, value in choices.items()})
    cumulative = completion.cumulative_logprob
    return {"status": status, "request_id": output.request_id,
            "output_token_ids": tokens, "output_tokens": len(tokens), "text": completion.text,
            "finish_reason": finish, "stop_reason": stop,
            "cumulative_logprob": cumulative if cumulative is None or math.isfinite(cumulative) else str(cumulative),
            "logprobs": logs, "selected_token_logprobs_finite": bool(selected_finite),
            "within_cap": len(tokens) <= cap, "completed_unix_s": time.time()}


def check_execution(summary, records, arm, stage):
    """Check observed real calls after startup reset; never synthesize phases."""
    if len(summary) != 1 or len(records) != 1:
        raise RuntimeError("integration expects exactly one real worker")
    worker, rows = summary[0], records[0]
    if set(worker["layers"]) != LAYERS or set(rows) != LAYERS:
        raise RuntimeError("actual runtime did not report all 16 installed layers")
    result = {}
    for layer in sorted(LAYERS, key=int):
        stats, events = worker["layers"][layer], rows[layer]
        if stats["calls"] <= 0 or len(events) != stats["calls"]:
            raise RuntimeError(f"layer {layer}: no actual full-model calls or bounded records were lost")
        counts = Counter(event["format"] for event in events)
        if dict(counts) != stats["selection_counts"]:
            raise RuntimeError(f"layer {layer}: summary and actual records disagree")
        if arm == "dual_phase":
            if any(event["phase_source"] != "scheduler_computed_position_and_prompt_boundary" for event in events):
                raise RuntimeError(f"layer {layer}: phase did not originate from the real scheduler")
            prefill = sum(event["phase"] == "prefill" and event["format"] == "fp8" for event in events)
            decode = sum(event["phase"] == "decode" and event["format"] == "awq" for event in events)
            if not prefill or (stage == "natural_generation" and not decode):
                raise RuntimeError(f"layer {layer}: required real prefill-FP8/decode-AWQ activity absent; no forced extension permitted")
            if any((event["phase"] == "prefill" and event["format"] != "fp8") or
                   (event["phase"] == "decode" and event["format"] != "awq") for event in events):
                raise RuntimeError(f"layer {layer}: wrong observed pure-phase format")
            result[layer] = {"prefill_fp8_calls": prefill, "decode_awq_calls": decode,
                             "selection_counts": dict(counts)}
        else:
            if set(counts) != {arm.removeprefix("single_")}:
                raise RuntimeError(f"layer {layer}: unexpected single-format execution")
            result[layer] = {"calls": stats["calls"], "selection_counts": dict(counts)}
    return {"passed": True, "layers": result, "scheduler_steps": worker["scheduler_steps"],
            "scheduler_tokens": worker["scheduler_tokens"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-manifest", default="checkpoints/olmoe_awq_integration_bf16_r01/checkpoint_manifest.json")
    parser.add_argument("--worker-config", default="configs/olmoe_integration_r01/single_fp8.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--lock-status")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("gpu-memory-utilization must be between zero and one")
    if not args.run:
        print(json.dumps(plan(args), indent=2, ensure_ascii=False))
        return
    if not args.lock_status:
        parser.error("--run requires the real --lock-status from a run_locked ancestor")
    from bench_serving import verify_lock
    lock = verify_lock(args.lock_status)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = {"schema_version": 1, "status": "validating_inputs", "lock": lock,
             "started_unix_s": time.time(), "stages": {}, "requests": [
                 {"stage": name, "prompt_id": i, "prompt": text, "status": "not_started", "max_tokens": cap}
                 for name, cap in STAGES for i, text in enumerate(PROMPTS)]}
    def save():
        temporary = output / "integration.tmp"
        write_json(temporary, state)
        temporary.replace(output / "integration.json")
    def interrupted(signum, frame):
        raise InterruptedError(f"run interrupted by signal {signum}; retain all requests")
    signal.signal(signal.SIGTERM, interrupted)
    save()
    llm, current_stage = None, None
    try:
        spec = plan(args)
        state["plan"] = spec
        worker = spec["worker"]
        write_json(worker["effective_config_path"], worker["effective_config"])
        worker["effective_config_sha256"] = sha256(worker["effective_config_path"])
        os.environ.update(C_FORMAT_DUAL_CONFIG=worker["effective_config_path"],
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        code_path = str(Path(__file__).resolve().parent)
        os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (code_path, os.environ.get("PYTHONPATH"))))
        state["implementation_sha256"] = {name: sha256(Path(code_path) / name) for name in
            ("run_full_model_integration.py", "eval_quality.py", "dual_format_worker.py", "dual_format_moe.py", "format_backends.py")}
        state["versions"] = {name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers")}
        if state["versions"]["vllm"].split("+", 1)[0] != "0.26.0":
            raise RuntimeError("integration targets the inspected installed vLLM 0.26.0")
        state["status"] = "loading_full_model"
        save()
        from vllm import LLM, SamplingParams
        llm = LLM(**spec["llm_kwargs"])
        report = read_json(worker["effective_config"]["report_path"])
        if report.get("status") != "installed_before_kv_profile" or report.get("config_sha256") != worker["effective_config_sha256"] or set(report.get("layers", {})) != LAYERS:
            raise RuntimeError("all-layer worker installation is missing, failed, or has a different configuration")
        state["installation_verified"] = True
        state["memory_after_load"] = llm.collective_rpc(worker_memory, timeout=30)
        state["pre_kv_installation_memory"] = report["memory"]
        state["logical_resident_expert_weight_scale_bytes"] = sum(
            row["installation"]["resident_weight_and_scale_bytes"] for row in report["layers"].values())
        tokenizer = llm.get_tokenizer()
        prompt_ids = [list(tokenizer.encode(text, add_special_tokens=True)) for text in PROMPTS]
        for row in state["requests"]:
            row["prompt_token_ids"] = prompt_ids[row["prompt_id"]]
            if not row["prompt_token_ids"] or len(row["prompt_token_ids"]) + row["max_tokens"] > 1024:
                raise ValueError("fixed prompt exceeds context budget; no truncation or filtering")
        state["prompt_token_ids_sha256"] = digest_json(prompt_ids)
        save()
        for name, cap in STAGES:
            current_stage = name
            stage = state["stages"][name] = {"status": "running", "max_tokens": cap}
            stage["previous_statistics"] = llm.collective_rpc("dual_format_reset_statistics", timeout=30)
            rows = [row for row in state["requests"] if row["stage"] == name]
            for row in rows:
                row.update(status="in_flight", started_unix_s=time.time())
            state["status"] = name
            save()
            generated = llm.generate([{"prompt_token_ids": ids} for ids in prompt_ids],
                SamplingParams(**spec["sampling"], max_tokens=cap), use_tqdm=False)
            if len(generated) != len(rows):
                raise RuntimeError("vLLM omitted a fixed request")
            for row, result in zip(rows, generated, strict=True):
                row.update(serialize_output(result, row["prompt_token_ids"], tokenizer.eos_token_id, cap))
            save()  # Preserve real outputs before any observation/coverage failure.
            stage["worker_statistics"] = llm.collective_rpc("dual_format_summary", timeout=30)
            stage["worker_records"] = llm.collective_rpc("dual_format_take_records", timeout=30)
            stage["memory_after_stage"] = llm.collective_rpc(worker_memory, timeout=30)
            save()
            if any(row["status"] in ("error", "unfinished") or not row["selected_token_logprobs_finite"] or not row["within_cap"] or not row["output_tokens"] for row in rows):
                raise RuntimeError("incomplete/invalid output; preserve the full request population")
            stage["observed_execution"] = check_execution(stage["worker_statistics"], stage["worker_records"], spec["arm"], name)
            stage["status"] = "complete"
            save()
        state.update(status="complete", full_16_layer_forward_verified=True,
                     real_natural_generation_verified=True,
                     dual_prefill_fp8_decode_awq_verified=spec["arm"] == "dual_phase")
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        if current_stage:
            state["stages"][current_stage]["status"] = "failed"
        for row in state["requests"]:
            if row["status"] == "in_flight":
                row.update(status="interrupted" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "error", error=state["error"])
            elif row["status"] == "not_started":
                row.update(status="unfinished", error="not started before arm failure")
        (output / "runtime_traceback.txt").write_text(traceback.format_exc())
        save()
        if llm is not None:
            try:
                state["failure_worker_statistics"] = llm.collective_rpc("dual_format_summary", timeout=5)
                state["failure_worker_records"] = llm.collective_rpc("dual_format_take_records", timeout=5)
            except Exception as observation_error:
                state["failure_observation_error"] = repr(observation_error)
        raise
    finally:
        state["ended_unix_s"] = time.time()
        state["termination_counts"] = dict(Counter(row["status"] for row in state["requests"]))
        save()
    print(json.dumps({"status": state["status"], "arm": state["plan"]["arm"], "output": str(output),
                      "termination_counts": state["termination_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One native BBH base-model workload qualification; no policy comparison."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

from C_NATIVE_PAST_FUTURE_AE_CELL_V1 import (
    EXPECTED_RUNTIME, KV_BYTES, dump, gpu_state, native_empty_contract, sha)


def generate(engine, requests, max_tokens, *, run_id, output_dir):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    scheduler = engine.engine_core.engine_core.scheduler
    pool = scheduler.kv_cache_manager.block_pool
    original_schedule = scheduler.schedule
    steps, preemptions = [], []

    def schedule():
        result = original_schedule()
        preemptions.extend(list(result.preempted_req_ids or []))
        return result

    scheduler.schedule = schedule
    rows = {}
    origin, arrival_epoch = time.perf_counter(), time.time()
    try:
        for item in requests:
            request_id = run_id + "/" + item["request_id"]
            rows[request_id] = dict(item, external_request_id=request_id,
                arrival_s=0.0, add_request_s=time.perf_counter() - origin,
                output_text="", output_token_ids=[], token_times_s=[],
                finished=False, finish_reason=None, stop_reason=None)
            params = SamplingParams(n=1, temperature=0.0, max_tokens=max_tokens,
                min_tokens=0, ignore_eos=False, stop=["\n\n"],
                detokenize=True, output_kind=RequestOutputKind.CUMULATIVE)
            engine.add_request(request_id, {"prompt_token_ids": item["prompt_token_ids"]},
                               params, arrival_time=arrival_epoch)
        while engine.has_unfinished_requests():
            before = time.perf_counter() - origin
            if before > 300:
                raise TimeoutError("qualification exceeded 300 seconds; preserve partial outputs")
            outputs = engine.step()
            received = time.perf_counter() - origin
            steps.append(dict(start_s=before, return_s=received,
                output_requests=len(outputs),
                used_blocks_after_step=int(pool.num_gpu_blocks - 1 - pool.get_num_free_blocks())))
            for output in outputs:
                row = rows[output.request_id]
                if row["finished"] or len(output.outputs) != 1:
                    raise RuntimeError("duplicate/unexpected output")
                completion = output.outputs[0]
                tokens = list(completion.token_ids)
                previous = row["output_token_ids"]
                if tokens[:len(previous)] != previous or not len(previous) <= len(tokens) <= max_tokens:
                    raise RuntimeError("cumulative output token prefix or cap violated")
                row["token_times_s"].extend([received] * (len(tokens) - len(previous)))
                row.update(output_text=completion.text, output_token_ids=tokens,
                    finish_reason=completion.finish_reason,
                    stop_reason=getattr(completion, "stop_reason", None),
                    finished=bool(output.finished))
                if output.finished:
                    if completion.finish_reason not in ("stop", "length"):
                        raise RuntimeError("unexpected finish reason")
                    row["host_elapsed_s"] = received
        if len(rows) != len(requests) or not all(row["finished"] for row in rows.values()):
            raise RuntimeError("not every request completed")
        return dict(status="COMPLETE", request_count=len(rows),
            observation_end_s=time.perf_counter() - origin,
            preemptions=len(preemptions),
            output_tokens=sum(len(row["output_token_ids"]) for row in rows.values()))
    finally:
        scheduler.schedule = original_schedule
        (output_dir / (run_id + "-outputs.json")).write_text(
            json.dumps(list(rows.values()), indent=2, ensure_ascii=False) + "\n")
        dump(output_dir / (run_id + "-steps.json"), dict(steps=steps,
            preempted_request_ids=preemptions, accounting="Host-observed delivery times; stop text buffering may affect delivery. All 27 arrivals at t=0; full drain included."))


def run(inputs, output, parent):
    config = json.loads((inputs / "config.json").read_text())
    workload = json.loads((inputs / "workload.json").read_text())
    requests = workload["requests"]
    assert len(requests) == 27 and len({r["request_id"] for r in requests}) == 27
    assert [r["task"] for r in requests] == sorted(r["task"] for r in requests)
    assert all(r["example_index"] == 0 and len(r["prompt_token_ids"]) + 512 <= 4096 for r in requests)
    expected_sampling = dict(temperature=0.0, max_tokens=512, stop=["\n\n"], ignore_eos=False, min_tokens=0)
    assert all(workload["sampling"].get(key) == value for key, value in expected_sampling.items())
    output.mkdir(exist_ok=False)
    dump(output / "config.json", config)
    timing = dict(process_start_unix_s=time.time(), process_start_perf_s=time.perf_counter())
    engine = None
    try:
        before = gpu_state()
        os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_BATCH_INVARIANT="0",
                          VLLM_USE_SIMPLE_KV_OFFLOAD="0")
        import torch
        import vllm
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine
        sys.path.insert(0, str(parent / "pkg"))
        from memory_telemetry import memory_snapshot
        from run_recovery_cadence import RUNTIME_SOURCES, eos_metadata
        runtime = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
            vllm=vllm.__version__, transformers=version("transformers"),
            cpu_threads=torch.get_num_threads(),
            vllm_source_sha256={name: sha(Path(vllm.__file__).parent / name) for name in RUNTIME_SOURCES})
        dump(output / "environment.json", dict(runtime=runtime, gpu_before=before,
            cell_sha256=sha(Path(__file__)), input_sha256={name: sha(inputs / name)
                for name in ("config.json", "workload.json", "SOURCE_RECEIPT.json")}))
        if runtime != EXPECTED_RUNTIME or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("pinned runtime differs")
        revision = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
        kwargs = dict(model="allenai/OLMoE-1B-7B-0924", revision=revision,
            tokenizer_revision=revision, dtype="bfloat16", seed=20260905,
            max_model_len=4096, max_num_seqs=32, max_num_batched_tokens=1024,
            long_prefill_token_threshold=0, gpu_memory_utilization=.90,
            enable_chunked_prefill=True, enable_prefix_caching=False,
            scheduling_policy="fcfs", async_scheduling=False,
            kv_cache_memory_bytes=KV_BYTES, scheduler_reserve_full_isl=True,
            stream_interval=1, enforce_eager=False, enable_return_routed_experts=False)
        dump(output / "engine_args.json", kwargs)
        timing["engine_init_start_perf_s"] = time.perf_counter()
        engine = LLMEngine.from_engine_args(EngineArgs(**kwargs), enable_multiprocessing=False)
        timing["engine_init_end_perf_s"] = time.perf_counter()
        dump(output / "resolved-eos.json", eos_metadata(engine))
        initial = native_empty_contract(engine)
        if initial["status"] != "QUALIFIED":
            raise RuntimeError("native initial request/KV state differs")
        dump(output / "memory-after-init.json", memory_snapshot(engine, torch))
        # A fixed two-request application warmup; its generations never enter the 27-task score.
        timing["warmup_start_perf_s"] = time.perf_counter()
        warmup = generate(engine, [requests[0], requests[-1]], 16,
                          run_id="warmup", output_dir=output)
        dump(output / "warmup-summary.json", warmup)
        timing["warmup_end_perf_s"] = time.perf_counter()
        if native_empty_contract(engine)["status"] != "QUALIFIED":
            raise RuntimeError("warmup did not drain")
        torch.cuda.reset_peak_memory_stats()
        timing["measurement_start_perf_s"] = time.perf_counter()
        summary = generate(engine, requests, 512, run_id="measured", output_dir=output)
        timing["measurement_return_perf_s"] = time.perf_counter()
        drain = native_empty_contract(engine)
        dump(output / "native-drain.json", drain)
        if drain["status"] != "QUALIFIED":
            raise RuntimeError("measurement did not drain")
        dump(output / "memory-after.json", memory_snapshot(engine, torch))
        dump(output / "status.json", dict(summary, scientific_scope="27 first-task examples, development workload qualification only; no policy or full-BBH benchmark claim"))
    except Exception as exc:
        dump(output / "status.json", dict(status="INCOMPLETE", error=f"{type(exc).__name__}: {exc}"))
        raise
    finally:
        timing["shutdown_start_perf_s"] = time.perf_counter()
        try:
            if engine is not None:
                engine.engine_core.shutdown()
        finally:
            timing["process_end_perf_s"] = time.perf_counter()
            dump(output / "timing.json", timing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    run(args.inputs, args.output, args.parent)

#!/usr/bin/env python3
"""Native APC off/on qualification on 32 fixed BBH geometric_shapes requests."""
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


def block_ids(blocks):
    """Flatten the native KVCacheBlocks group IDs without inferring ownership."""
    value = blocks.get_block_ids()
    def flatten(item):
        if isinstance(item, int):
            return [int(item)]
        return [block for part in item for block in flatten(part)]
    return flatten(value)


def pool_snapshot(pool, *, include_shared_ids=False):
    """A physical-block census from native refcounts, excluding the null block."""
    active = [block for block in pool.blocks if not block.is_null and block.ref_cnt > 0]
    shared = [block for block in active if block.ref_cnt >= 2]
    receipt = dict(unique_active_blocks=len(active),
        shared_active_blocks=len(shared), active_block_references=sum(b.ref_cnt for b in active),
        free_blocks=int(pool.get_num_free_blocks()),
        cached_hash_keys=len(pool.cached_block_hash_to_block),
        hashed_blocks=sum(b.block_hash is not None for b in pool.blocks if not b.is_null),
        auxiliary_hash_blocks=len(pool.cached_block_hashes_by_block))
    receipt["free_active_accounting_consistent"] = (
        receipt["unique_active_blocks"] == pool.num_gpu_blocks - 1 - receipt["free_blocks"])
    if include_shared_ids:
        receipt["shared_block_refcounts_sample"] = [
            dict(block_id=int(b.block_id), ref_cnt=int(b.ref_cnt)) for b in shared[:4]]
    return receipt


def reset_and_verify_empty(engine):
    scheduler = engine.engine_core.engine_core.scheduler
    pool = scheduler.kv_cache_manager.block_pool
    before = pool_snapshot(pool)
    drained_before = native_empty_contract(engine)
    if drained_before["status"] != "QUALIFIED" or before["unique_active_blocks"] != 0:
        raise RuntimeError("warmups did not release every request-owned block")
    reset_succeeded = bool(engine.reset_prefix_cache())
    after = pool_snapshot(pool)
    drained_after = native_empty_contract(engine)
    receipt = dict(reset_succeeded=reset_succeeded, before=before, after=after,
        drained_before=drained_before, drained_after=drained_after)
    if (not reset_succeeded or drained_after["status"] != "QUALIFIED"
            or after["unique_active_blocks"] != 0 or after["cached_hash_keys"] != 0
            or after["hashed_blocks"] != 0 or after["auxiliary_hash_blocks"] != 0
            or not after["free_active_accounting_consistent"]):
        raise RuntimeError("prefix cache reset did not establish an empty measurement baseline")
    return receipt


def generate(engine, requests, max_tokens, *, run_id, output_dir):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    scheduler = engine.engine_core.engine_core.scheduler
    manager = scheduler.kv_cache_manager
    pool = manager.block_pool
    original_schedule = scheduler.schedule
    original_lookup = manager.get_computed_blocks
    original_allocate = manager.allocate_slots
    steps, schedule_states, preemptions = [], [], []
    lookup_events, allocation_events = [], []
    shared_witness = None
    rows = {}
    origin, arrival_epoch = time.perf_counter(), time.time()

    def lookup(*args, **kwargs):
        result = original_lookup(*args, **kwargs)
        request = args[0] if args else kwargs["request"]
        lookup_events.append(dict(schedule_index=len(schedule_states),
            request_id=request.request_id, hit_tokens=int(result[1]),
            hit_block_ids=block_ids(result[0])))
        return result

    def allocate(*args, **kwargs):
        result = original_allocate(*args, **kwargs)
        request = args[0] if args else kwargs["request"]
        ids = [] if result is None else block_ids(result)
        allocation_events.append(dict(schedule_index=len(schedule_states),
            request_id=request.request_id, succeeded=result is not None,
            returned_block_refcounts=[dict(block_id=bid, ref_cnt=int(pool.blocks[bid].ref_cnt))
                                      for bid in ids]))
        return result

    def schedule(*args, **kwargs):
        nonlocal shared_witness
        result = original_schedule(*args, **kwargs)
        preemptions.extend(list(result.preempted_req_ids or []))
        state = dict(schedule_index=len(schedule_states),
            return_s=time.perf_counter() - origin,
            **pool_snapshot(pool, include_shared_ids=True))
        if shared_witness is None and state["shared_block_refcounts_sample"]:
            shared_id = state["shared_block_refcounts_sample"][0]["block_id"]
            owners = []
            for request in scheduler.running:
                try:
                    held = block_ids(manager.get_blocks(request.request_id))
                except KeyError:
                    continue
                if shared_id in held:
                    owners.append(request.request_id)
            if len(owners) >= 2:
                shared_witness = dict(schedule_index=len(schedule_states),
                    block_id=shared_id, native_ref_cnt=int(pool.blocks[shared_id].ref_cnt),
                    request_ids=owners)
        schedule_states.append(state)
        return result

    manager.get_computed_blocks = lookup
    manager.allocate_slots = allocate
    scheduler.schedule = schedule
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
            if before > 600:
                raise TimeoutError("APC qualification exceeded 600 seconds; preserve partial outputs")
            outputs = engine.step()
            received = time.perf_counter() - origin
            steps.append(dict(start_s=before, return_s=received,
                output_requests=len(outputs),
                **pool_snapshot(pool)))
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
        if not all(s["free_active_accounting_consistent"] for s in schedule_states + steps):
            raise RuntimeError("native free-list and block refcount census disagree")
        return dict(status="COMPLETE", request_count=len(rows),
            observation_end_s=time.perf_counter() - origin,
            preemptions=len(preemptions),
            output_tokens=sum(len(row["output_token_ids"]) for row in rows.values()),
            lookup_events=len(lookup_events),
            total_lookup_hit_tokens=sum(event["hit_tokens"] for event in lookup_events),
            requests_with_lookup_hits=len({event["request_id"] for event in lookup_events
                                          if event["hit_tokens"] > 0}),
            peak_unique_active_physical_blocks=max(
                (state["unique_active_blocks"] for state in schedule_states), default=0),
            peak_shared_active_blocks=max(
                (state["shared_active_blocks"] for state in schedule_states), default=0),
            shared_witness=shared_witness)
    finally:
        scheduler.schedule = original_schedule
        manager.get_computed_blocks = original_lookup
        manager.allocate_slots = original_allocate
        (output_dir / (run_id + "-outputs.json")).write_text(
            json.dumps(list(rows.values()), indent=2, ensure_ascii=False) + "\n")
        dump(output_dir / (run_id + "-steps.json"), dict(steps=steps,
            schedule_states=schedule_states, lookup_events=lookup_events,
            allocation_events=allocation_events, shared_witness=shared_witness,
            preempted_request_ids=preemptions,
            accounting=f"Host-observed delivery; all {len(requests)} arrivals at t=0. Physical census after native scheduler.schedule excludes null and free hashed blocks. Refcounts are native request references; timings include Python instrumentation and are descriptive only."))


def run(inputs, output, parent, arm):
    if arm not in ("apc_off", "apc_on"):
        raise ValueError("arm must be apc_off or apc_on")
    config = json.loads((inputs / "config.json").read_text())
    workload = json.loads((inputs / "workload.json").read_text())
    requests = workload["requests"]
    if (config.get("schema") != "c-bbh-apc-config-v1"
            or workload.get("schema") != "c-bbh-apc-workload-v1"
            or config.get("workload_sha256") != sha(inputs / "workload.json")):
        raise RuntimeError("APC input schema or frozen workload hash differs")
    assert len(requests) == 32 and len({r["request_id"] for r in requests}) == 32
    assert all(r["task"] == "geometric_shapes" for r in requests)
    assert [r["example_index"] for r in requests] == list(range(32))
    assert all(len(r["prompt_token_ids"]) + 512 <= 4096 for r in requests)
    assert workload["arrival_traces_s"] == [0.0] * 32
    expected_sampling = dict(temperature=0.0, max_tokens=512, stop=["\n\n"], ignore_eos=False, min_tokens=0)
    assert all(workload["sampling"].get(key) == value for key, value in expected_sampling.items())
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable output already exists: {output}")
    output.mkdir(exist_ok=False)
    dump(output / "config.json", dict(config, arm=arm))
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
            enable_chunked_prefill=True, enable_prefix_caching=(arm == "apc_on"),
            scheduling_policy="fcfs", async_scheduling=False,
            kv_cache_memory_bytes=KV_BYTES, scheduler_reserve_full_isl=True,
            stream_interval=1, enforce_eager=False, enable_return_routed_experts=False)
        dump(output / "engine_args.json", kwargs)
        timing["engine_init_start_perf_s"] = time.perf_counter()
        engine = LLMEngine.from_engine_args(EngineArgs(**kwargs), enable_multiprocessing=False)
        timing["engine_init_end_perf_s"] = time.perf_counter()
        if bool(engine.vllm_config.cache_config.enable_prefix_caching) != (arm == "apc_on"):
            raise RuntimeError("resolved native APC toggle differs from arm")
        dump(output / "resolved-eos.json", eos_metadata(engine))
        initial = native_empty_contract(engine)
        if initial["status"] != "QUALIFIED":
            raise RuntimeError("native initial request/KV state differs")
        dump(output / "memory-after-init.json", memory_snapshot(engine, torch))
        # Separate first/last warmups, followed by a mandatory cache reset in both arms.
        timing["warmup_start_perf_s"] = time.perf_counter()
        warmup_first = generate(engine, [requests[0]], 16,
                                run_id="warmup-first", output_dir=output)
        warmup_last = generate(engine, [requests[-1]], 16,
                               run_id="warmup-last", output_dir=output)
        dump(output / "warmup-summary.json", dict(first=warmup_first, last=warmup_last))
        timing["warmup_end_perf_s"] = time.perf_counter()
        reset_receipt = reset_and_verify_empty(engine)
        dump(output / "prefix-cache-reset.json", reset_receipt)
        torch.cuda.reset_peak_memory_stats()
        timing["measurement_start_perf_s"] = time.perf_counter()
        summary = generate(engine, requests, 512, run_id="measured", output_dir=output)
        timing["measurement_return_perf_s"] = time.perf_counter()
        drain = native_empty_contract(engine)
        dump(output / "native-drain.json", drain)
        if drain["status"] != "QUALIFIED":
            raise RuntimeError("measurement did not drain")
        dump(output / "memory-after.json", memory_snapshot(engine, torch))
        dump(output / "status.json", dict(summary, arm=arm,
            scientific_scope="32 source-first geometric_shapes examples, development APC qualification only; instrumented host timing is descriptive, not a performance comparison"))
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
    parser.add_argument("--arm", choices=("apc_off", "apc_on"), required=True)
    args = parser.parse_args()
    run(args.inputs, args.output, args.parent, args.arm)

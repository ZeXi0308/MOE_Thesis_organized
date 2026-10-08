#!/usr/bin/env python3
"""One complete150 prompt subtree density enqueue-order development reference."""
from __future__ import annotations

import argparse
import cProfile
from collections import Counter
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

from C_SPARE_CELL_HELPERS_V1 import (
    EXPECTED_RUNTIME, KV_BYTES, dump, gpu_state, native_empty_contract, sha)

MODEL_REVISION = "7f1c97f440f06ce36705e4f2b843edb5925f4498"
MODEL_ID = "allenai/OLMoE-1B-7B-0924-Instruct"
REQUEST_COUNT = 150
OUTPUT_CAP = 64
DENSITY_REORDER = None
EXPECTED_ORDER = None


def validate_inputs(inputs: Path):
    config = json.loads((inputs / "config.json").read_text())
    workload = json.loads((inputs / "workload.json").read_text())
    if (config.get("schema") != "c-longbench-multifieldqa-en-full-config-v1"
            or workload.get("schema") != "c-longbench-multifieldqa-en-full-workload-v1"
            or config.get("workload_sha256") != sha(inputs / "workload.json")):
        raise RuntimeError("LongBench150 input schema or workload hash differs")
    if (config.get("requests") != REQUEST_COUNT or config.get("output_tokens") != OUTPUT_CAP
            or config.get("max_model_len") != 4096
            or config.get("proposed_runtime", {}).get("max_num_seqs") != 128
            or config.get("proposed_runtime", {}).get("max_num_batched_tokens") != 1024
            or config.get("proposed_runtime", {}).get("usable_kv_blocks") != 4096
            or config.get("proposed_runtime", {}).get("prefix_caching") is not True
            or config.get("model") != dict(id=MODEL_ID, revision=MODEL_REVISION,
                                            tokenizer_revision=MODEL_REVISION)):
        raise RuntimeError("native128 input/model/resource contract differs")
    requests = workload.get("requests")
    if not isinstance(requests, list) or len(requests) != REQUEST_COUNT:
        raise RuntimeError("expected exactly 150 input requests")
    if (len({r["request_id"] for r in requests}) != REQUEST_COUNT
            or [r["example_index"] for r in requests] != list(range(REQUEST_COUNT))
            or workload.get("arrival_traces_s") != [0.0] * REQUEST_COUNT):
        raise RuntimeError("request identities, source indices, or arrivals differ")
    for row in requests:
        ids = row.get("prompt_token_ids")
        if (not isinstance(ids, list) or not ids
                or not all(type(token) is int and token >= 0 for token in ids)
                or len(ids) + OUTPUT_CAP > 4096
                or not isinstance(row.get("prompt"), str) or not row["prompt"]
                or not isinstance(row.get("chat_user_content"), str)
                or not row["chat_user_content"]
                or not isinstance(row.get("question"), str) or not row["question"]
                or not isinstance(row.get("answers"), list) or not row["answers"]
                or any(not isinstance(answer, str) for answer in row["answers"])):
            raise RuntimeError("request prompt, question, references, or context bound differs")
    expected = dict(temperature=0.0, max_tokens=OUTPUT_CAP, min_tokens=0,
                    ignore_eos=False, stop=[])
    sampling = workload.get("sampling", {})
    if sampling != expected:
        raise RuntimeError("EOS-only greedy sampling contract differs")
    return config, workload, requests


def validate_local_tokenizer(model_dir: Path, requests: list[dict], output: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False)
    for row in requests:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["chat_user_content"]}],
            tokenize=False, add_generation_prompt=True)
        encoded = tokenizer.encode(row["prompt"], add_special_tokens=False)
        raw_ids = tokenizer.encode(row["raw_prompt"], add_special_tokens=True)
        rebuilt = (tokenizer.decode(raw_ids[:1750], skip_special_tokens=True)
                   + tokenizer.decode(raw_ids[-1750:], skip_special_tokens=True)
                   if len(raw_ids) > 3500 else row["raw_prompt"])
        if (rendered != row["prompt"] or encoded != row["prompt_token_ids"]
                or rebuilt != row["chat_user_content"]
                or len(raw_ids) != row["truncation"]["raw_prompt_tokens"]):
            raise RuntimeError("local tokenizer/truncation/chat differs for " + row["request_id"])
    receipt = dict(status="QUALIFIED", requests_checked=len(requests),
        tokenizer_path=str(model_dir.resolve()),
        tokenizer_eos_token_id=tokenizer.eos_token_id,
        historical_answer_suffix=None,
        scope="Exact official head/tail truncation and local one-user chat/token IDs checked before CUDA use")
    dump(output / "input-tokenizer-check.json", receipt)
    return receipt


def reset_warmup_cache(engine):
    scheduler = engine.engine_core.engine_core.scheduler
    pool = scheduler.kv_cache_manager.block_pool
    before = native_empty_contract(engine)
    if before["status"] != "QUALIFIED":
        raise RuntimeError("warmup requests did not drain")
    succeeded = bool(engine.reset_prefix_cache())
    after = native_empty_contract(engine)
    remaining_hashes = len(pool.cached_block_hash_to_block)
    receipt = dict(reset_succeeded=succeeded, drained_before=before,
                   drained_after=after, cached_hash_keys_after=remaining_hashes)
    if not succeeded or after["status"] != "QUALIFIED" or remaining_hashes:
        raise RuntimeError("prefix cache did not reset before measurement")
    return receipt


def generate(engine, requests, max_tokens, *, run_id, output_dir, max_seconds=900):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    scheduler = engine.engine_core.engine_core.scheduler
    pool = scheduler.kv_cache_manager.block_pool
    original_schedule = scheduler.schedule
    original_allocate = scheduler.kv_cache_manager.allocate_slots
    allocation_calls, allocation_failures = 0, []
    steps, schedule_calls, preempted = [], [], []
    profiled_steps = {}
    rows = {}
    submission_order, reorder_s = [], None
    origin, arrival_epoch = time.perf_counter(), time.time()

    def scheduler_state():
        return dict(used_blocks=int(pool.num_gpu_blocks - 1 - pool.get_num_free_blocks()),
                    running=len(scheduler.running), waiting=len(scheduler.waiting))

    def schedule(*args, **kwargs):
        before = scheduler_state()
        result = original_schedule(*args, **kwargs)
        after = scheduler_state()
        newly_preempted = list(result.preempted_req_ids or [])
        scheduled_tokens = result.num_scheduled_tokens
        preempted.extend(newly_preempted)
        schedule_calls.append(dict(call=len(schedule_calls), host_s=time.perf_counter() - origin,
            before=before, after=after, preempted_request_ids=newly_preempted,
            scheduled_request_count=len(scheduled_tokens),
            scheduled_tokens_total=sum(scheduled_tokens.values())))
        return result

    def allocate(request, num_new_tokens, *args, **kwargs):
        nonlocal allocation_calls
        allocation_calls += 1
        free_before = int(pool.get_num_free_blocks())
        result = original_allocate(request, num_new_tokens, *args, **kwargs)
        if result is None:
            allocation_failures.append(dict(
                schedule_call=len(schedule_calls), host_s=time.perf_counter() - origin,
                request_id=request.request_id, request_status=request.status.name,
                num_computed_tokens=request.num_computed_tokens,
                current_num_tokens=request.num_tokens,
                prompt_tokens=len(request.prompt_token_ids),
                num_new_tokens=num_new_tokens,
                num_new_computed_tokens=kwargs.get("num_new_computed_tokens", 0),
                full_sequence_must_fit=kwargs.get("full_sequence_must_fit", False),
                free_blocks_before=free_before,
                free_blocks_after=int(pool.get_num_free_blocks())))
        return result

    scheduler.kv_cache_manager.allocate_slots = allocate
    scheduler.schedule = schedule
    try:
        reorder_start = time.perf_counter()
        submission_order = DENSITY_REORDER([item["prompt_token_ids"] for item in requests])
        reorder_s = time.perf_counter() - reorder_start
        if sorted(submission_order) != list(range(len(requests))):
            raise RuntimeError("density order omitted or duplicated requests")
        if run_id == "measured" and submission_order != EXPECTED_ORDER:
            raise RuntimeError("density order differs from frozen CPU receipt")
        for index in submission_order:
            item = requests[index]
            request_id = run_id + "/" + item["request_id"]
            rows[request_id] = dict(item, external_request_id=request_id,
                arrival_s=0.0, arrival_unix_s=arrival_epoch,
                add_request_s=time.perf_counter() - origin,
                output_text="", output_token_ids=[], token_times_s=[],
                finished=False, finish_reason=None, stop_reason=None)
            params = SamplingParams(n=1, temperature=0.0, max_tokens=max_tokens,
                min_tokens=0, ignore_eos=False, stop=[], detokenize=True,
                output_kind=RequestOutputKind.CUMULATIVE)
            engine.add_request(request_id, {"prompt_token_ids": item["prompt_token_ids"]},
                               params, arrival_time=arrival_epoch)
        while engine.has_unfinished_requests():
            before = time.perf_counter() - origin
            if before > max_seconds:
                raise TimeoutError("LongBench150 observation exceeded its measured deadline; partial outputs retained")
            if run_id == "measured" and len(steps) in (379, 380, 381):
                profiler = cProfile.Profile()
                profiled_steps[len(steps)] = profiler
                outputs = profiler.runcall(engine.step)
            else:
                outputs = engine.step()
            received = time.perf_counter() - origin
            steps.append(dict(start_s=before, return_s=received,
                output_requests=len(outputs),
                used_blocks_after_step=int(pool.num_gpu_blocks - 1 - pool.get_num_free_blocks())))
            for output in outputs:
                row = rows[output.request_id]
                if row["finished"] or len(output.outputs) != 1:
                    raise RuntimeError("duplicate or unexpected completion")
                completion = output.outputs[0]
                tokens = list(completion.token_ids)
                previous = row["output_token_ids"]
                if (tokens[:len(previous)] != previous
                        or not len(previous) <= len(tokens) <= max_tokens):
                    raise RuntimeError("cumulative output prefix or cap violated")
                row["token_times_s"].extend([received] * (len(tokens) - len(previous)))
                row.update(output_text=completion.text, output_token_ids=tokens,
                    finish_reason=completion.finish_reason,
                    stop_reason=getattr(completion, "stop_reason", None),
                    finished=bool(output.finished))
                if output.finished:
                    if completion.finish_reason not in ("stop", "length"):
                        raise RuntimeError("unexpected finish reason")
                    row["host_elapsed_s"] = received
                    row["finish_unix_s"] = time.time()
        if len(rows) != len(requests) or not all(row["finished"] for row in rows.values()):
            raise RuntimeError("not every request completed")
        counts = Counter(row["finish_reason"] for row in rows.values())
        return dict(status="COMPLETE", request_count=len(rows),
            observation_end_s=time.perf_counter() - origin,
            preemptions=len(preempted), schedule_calls=len(schedule_calls),
            allocation_calls=allocation_calls, allocation_failure_count=len(allocation_failures),
            subtree_density_reorder_s=reorder_s,
            peak_used_blocks_after_schedule=max((c["after"]["used_blocks"] for c in schedule_calls), default=0),
            peak_running_after_schedule=max((c["after"]["running"] for c in schedule_calls), default=0),
            peak_waiting_after_schedule=max((c["after"]["waiting"] for c in schedule_calls), default=0),
            output_tokens=sum(len(row["output_token_ids"]) for row in rows.values()),
            finish_reason_counts=dict(counts))
    finally:
        scheduler.schedule = original_schedule
        scheduler.kv_cache_manager.allocate_slots = original_allocate
        if run_id == "measured":
            profiles = {}
            for step_index, profiler in profiled_steps.items():
                entries = []
                for entry in profiler.getstats():
                    code = entry.code
                    entries.append(dict(function=code if isinstance(code, str) else code.co_name,
                        filename=None if isinstance(code, str) else code.co_filename,
                        first_line=None if isinstance(code, str) else code.co_firstlineno,
                        calls=entry.callcount, recursive_calls=entry.reccallcount,
                        cumulative_call_s=entry.totaltime, self_call_s=entry.inlinetime))
                profiles[str(step_index)] = sorted(entries,
                    key=lambda entry: entry["cumulative_call_s"], reverse=True)
            dump(output_dir / "selected-step-python-profile.json", dict(
                scope="Diagnostic cProfile of synchronous engine.step at fixed steps379/380/381; inclusive nested call times must not be summed; profiling overhead retained; no CUDA kernel timing or policy performance claim",
                profiled_steps=profiles))
        (output_dir / (run_id + "-outputs.json")).write_text(
            json.dumps(list(rows.values()), indent=2, ensure_ascii=False) + "\n")
        dump(output_dir / (run_id + "-steps.json"), dict(steps=steps,
            scheduler_calls=schedule_calls,
            subtree_density_reorder_s=reorder_s,
            source_indices_in_submission_order=[requests[i]["source_index"] for i in submission_order],
            allocation_calls=allocation_calls, allocation_failures=allocation_failures,
            preempted_request_ids=preempted,
            accounting="Native schedule before/after block and queue counts; host-observed step delivery; all requests arrive at t=0 and drain. Warmups are separate files."))


def run(inputs: Path, output: Path, parent: Path, model_dir: Path,
        model_manifest: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable output already exists: {output}")
    output.mkdir(exist_ok=False)
    timing = dict(process_start_unix_s=time.time(), process_start_perf_s=time.perf_counter())
    engine = None
    global DENSITY_REORDER, EXPECTED_ORDER
    try:
        config, _, requests = validate_inputs(inputs)
        from C_LONG_DOCUMENT_QA_DENSITY_ORDER_V1 import density_order
        DENSITY_REORDER = density_order
        EXPECTED_ORDER = json.loads((Path(__file__).parent /
            "long_document_qa_density_order_v1.json").read_text())["source_indices_in_submission_order"]
        dump(output / "density-reorder-source.json", dict(
            source_sha256=sha(Path(__file__).parent / "C_LONG_DOCUMENT_QA_DENSITY_ORDER_V1.py"),
            scope="Prompt-only subtree work/request ordering; known Smith-rule heuristic development reference, no novelty claim"))
        # The launcher owns the stage; the cached-model helper verifies pinned
        # shards and stages local hardlinks before GPU initialization.
        from C_INSTRUCT_CACHED_MODEL_V1 import prepare_model
        timing["model_prepare_start_perf_s"] = time.perf_counter()
        receipt = prepare_model(model_dir, model_manifest, output)
        timing["model_prepare_end_perf_s"] = time.perf_counter()
        if not isinstance(receipt, dict) or not model_dir.is_dir():
            raise RuntimeError("model preparation did not return a receipt/local directory")
        dump(output / "config.json", config)
        os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_BATCH_INVARIANT="0",
                          VLLM_USE_SIMPLE_KV_OFFLOAD="0", HF_HUB_OFFLINE="1")
        validate_local_tokenizer(model_dir, requests, output)
        before = gpu_state()
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
            vllm_source_sha256={name: sha(Path(vllm.__file__).parent / name)
                                for name in RUNTIME_SOURCES})
        dump(output / "environment.json", dict(runtime=runtime, gpu_before=before,
            cell_sha256=sha(Path(__file__)), model_manifest_sha256=sha(model_manifest),
            input_sha256={name: sha(inputs / name)
                for name in ("config.json", "workload.json", "SOURCE_RECEIPT.json")}))
        if runtime != EXPECTED_RUNTIME or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("pinned runtime differs")
        local_model = str(model_dir.resolve())
        kwargs = dict(model=local_model, tokenizer=local_model,
            revision=MODEL_REVISION, tokenizer_revision=MODEL_REVISION,
            dtype="bfloat16", seed=20260905, max_model_len=4096,
            max_num_seqs=128, max_num_batched_tokens=1024,
            long_prefill_token_threshold=0, gpu_memory_utilization=.90,
            enable_chunked_prefill=True, enable_prefix_caching=True,
            scheduling_policy="fcfs", async_scheduling=False,
            kv_cache_memory_bytes=KV_BYTES, scheduler_reserve_full_isl=True,
            stream_interval=1, enforce_eager=False,
            enable_return_routed_experts=False)
        dump(output / "engine_args.json", kwargs)
        timing["engine_init_start_perf_s"] = time.perf_counter()
        engine = LLMEngine.from_engine_args(EngineArgs(**kwargs), enable_multiprocessing=False)
        timing["engine_init_end_perf_s"] = time.perf_counter()
        scheduler_config = engine.vllm_config.scheduler_config
        pool = engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool
        resolved = dict(max_num_seqs=scheduler_config.max_num_seqs,
            max_num_batched_tokens=scheduler_config.max_num_batched_tokens,
            prefix_caching=bool(engine.vllm_config.cache_config.enable_prefix_caching),
            allocated_kv_blocks=pool.num_gpu_blocks, usable_kv_blocks=pool.num_gpu_blocks - 1)
        dump(output / "resolved-scheduler.json", resolved)
        if (resolved["max_num_seqs"] != 128 or resolved["max_num_batched_tokens"] != 1024
                or resolved["prefix_caching"] is not True
                or resolved["allocated_kv_blocks"] != 4097):
            raise RuntimeError("resolved native128 concurrency/batch/APC/KV configuration differs")
        resolved_eos = eos_metadata(engine)
        eos = resolved_eos.get("hf_eos_token_id")
        eos_ids = [eos] if type(eos) is int else eos if isinstance(eos, list) else []
        if (resolved_eos.get("status") != "READ" or not eos_ids
                or any(type(token) is not int or token < 0 for token in eos_ids)):
            resolved_eos["qualification_status"] = "UNQUALIFIED"
            dump(output / "resolved-eos.json", resolved_eos)
            raise RuntimeError("model EOS metadata is unavailable or has no valid EOS ID")
        resolved_eos["qualification_status"] = "QUALIFIED"
        resolved_eos["qualified_eos_token_ids"] = eos_ids
        dump(output / "resolved-eos.json", resolved_eos)
        initial = native_empty_contract(engine)
        if initial["status"] != "QUALIFIED":
            raise RuntimeError("native initial request/KV state differs")
        dump(output / "memory-after-init.json", memory_snapshot(engine, torch))
        timing["warmup_start_perf_s"] = time.perf_counter()
        first = generate(engine, [requests[0]], 16,
                         run_id="warmup-first", output_dir=output)
        last = generate(engine, [requests[-1]], 16,
                        run_id="warmup-last", output_dir=output)
        dump(output / "warmup-summary.json", dict(first=first, last=last))
        timing["warmup_end_perf_s"] = time.perf_counter()
        dump(output / "prefix-cache-reset.json", reset_warmup_cache(engine))
        torch.cuda.reset_peak_memory_stats()
        timing["measurement_start_perf_s"] = time.perf_counter()
        summary = generate(engine, requests, OUTPUT_CAP,
                           run_id="measured", output_dir=output, max_seconds=180)
        timing["measurement_return_perf_s"] = time.perf_counter()
        drain = native_empty_contract(engine)
        dump(output / "native-drain.json", drain)
        if drain["status"] != "QUALIFIED":
            raise RuntimeError("measurement did not drain")
        dump(output / "memory-after.json", memory_snapshot(engine, torch))
        dump(output / "status.json", dict(summary,
            eos_qualification_status=resolved_eos["qualification_status"],
            scientific_scope="Diagnostic full150 same density order with Python call profiling at steps379/380/381; not an uninstrumented performance comparison"))
    except Exception as exc:
        dump(output / "status.json", dict(status="INCOMPLETE",
            error=f"{type(exc).__name__}: {exc}"))
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
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    args = parser.parse_args()
    run(args.inputs, args.output, args.parent, args.model_dir, args.model_manifest)

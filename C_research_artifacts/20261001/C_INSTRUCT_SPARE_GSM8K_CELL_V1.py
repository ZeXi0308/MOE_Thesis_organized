#!/usr/bin/env python3
"""One native Instruct GSM8K workload qualification; no strategy comparison."""
from __future__ import annotations

import argparse
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
REQUEST_COUNT = 16
OUTPUT_CAP = 1024


def validate_inputs(inputs: Path):
    config = json.loads((inputs / "config.json").read_text())
    workload = json.loads((inputs / "workload.json").read_text())
    if (config.get("schema") != "c-instruct-gsm8k-qualification-config-v1"
            or workload.get("schema") != "c-instruct-gsm8k-qualification-workload-v1"
            or config.get("workload_sha256") != sha(inputs / "workload.json")):
        raise RuntimeError("Instruct GSM8K input schema or workload hash differs")
    requests = workload.get("requests")
    if not isinstance(requests, list) or len(requests) != REQUEST_COUNT:
        raise RuntimeError("expected exactly 16 input requests")
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
                or not isinstance(row.get("gold"), str) or not row["gold"]):
            raise RuntimeError("request prompt, question, gold, or context bound differs")
    expected = dict(temperature=0.0, max_tokens=OUTPUT_CAP, min_tokens=0,
                    ignore_eos=False, stop=[])
    sampling = workload.get("sampling", {})
    if any(sampling.get(key) != value for key, value in expected.items()):
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
        # The pinned historical evaluator appends this answer cue after the
        # model chat template's assistant-generation prefix.
        if rendered + "Answer:" != row["prompt"] or encoded != row["prompt_token_ids"]:
            raise RuntimeError("local tokenizer/chat template differs for " + row["request_id"])
    receipt = dict(status="QUALIFIED", requests_checked=len(requests),
        tokenizer_path=str(model_dir.resolve()),
        tokenizer_eos_token_id=tokenizer.eos_token_id,
        historical_answer_suffix="Answer:",
        scope="Exact local chat-template plus historical answer cue and token IDs checked before CUDA use")
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


def generate(engine, requests, max_tokens, *, run_id, output_dir):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    scheduler = engine.engine_core.engine_core.scheduler
    pool = scheduler.kv_cache_manager.block_pool
    original_schedule = scheduler.schedule
    steps, preempted = [], []
    rows = {}
    origin, arrival_epoch = time.perf_counter(), time.time()

    def schedule(*args, **kwargs):
        result = original_schedule(*args, **kwargs)
        preempted.extend(list(result.preempted_req_ids or []))
        return result

    scheduler.schedule = schedule
    try:
        for item in requests:
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
            if before > 900:
                raise TimeoutError("GSM8K qualification exceeded 900 seconds; partial outputs retained")
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
            preemptions=len(preempted),
            output_tokens=sum(len(row["output_token_ids"]) for row in rows.values()),
            finish_reason_counts=dict(counts))
    finally:
        scheduler.schedule = original_schedule
        (output_dir / (run_id + "-outputs.json")).write_text(
            json.dumps(list(rows.values()), indent=2, ensure_ascii=False) + "\n")
        dump(output_dir / (run_id + "-steps.json"), dict(steps=steps,
            preempted_request_ids=preempted,
            accounting="Host-observed step delivery; all requests arrive at t=0 and drain. Warmup generations excluded from measurement."))


def run(inputs: Path, output: Path, parent: Path, model_dir: Path,
        model_manifest: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable output already exists: {output}")
    output.mkdir(exist_ok=False)
    timing = dict(process_start_unix_s=time.time(), process_start_perf_s=time.perf_counter())
    engine = None
    try:
        config, _, requests = validate_inputs(inputs)
        # The launcher owns the model staging directory; this validates its pin and
        # downloads through normal TLS before any GPU model initialization.
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
            max_num_seqs=32, max_num_batched_tokens=1024,
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
        if not bool(engine.vllm_config.cache_config.enable_prefix_caching):
            raise RuntimeError("native APC was not enabled")
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
                           run_id="measured", output_dir=output)
        timing["measurement_return_perf_s"] = time.perf_counter()
        drain = native_empty_contract(engine)
        dump(output / "native-drain.json", drain)
        if drain["status"] != "QUALIFIED":
            raise RuntimeError("measurement did not drain")
        dump(output / "memory-after.json", memory_snapshot(engine, torch))
        dump(output / "status.json", dict(summary,
            eos_qualification_status=resolved_eos["qualification_status"],
            scientific_scope="16 fixed GSM8K examples with eight-shot chat prompts; native Instruct workload qualification only, not a full GSM8K benchmark or strategy comparison"))
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

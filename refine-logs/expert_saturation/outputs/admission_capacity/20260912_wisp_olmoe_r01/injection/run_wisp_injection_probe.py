#!/usr/bin/env python3
"""One fresh-engine OLMoE prefill-injection cell; all timings are host observations."""
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
from run_wisp_olmoe_probe import PROMPTS, run_requests, worker_snapshot
from wisp_paging_trace import TRACE


class InvalidExperiment(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise InvalidExperiment(message)


def full_snapshot(worker, reset=False):
    result = worker_snapshot(worker, reset)
    module = sys.modules["wisp.integrations.vllm.fused_moe"]
    result["pager_execution_state"] = [dict(layer_idx=s.layer_idx,
        slot_to_expert=list(s.slot_to_expert), expert_to_slot=dict(s.expert_to_slot),
        lru_tick=list(s.lru_tick), lru_clock=s.lru_clock,
        cache_epoch="UNAVAILABLE: upstream state has no epoch field") for s in module._LAYER_STATES]
    return result


def scheduler_state(scheduler):
    return dict(running=[r.request_id for r in scheduler.running],
        waiting=[r.request_id for r in scheduler.waiting],
        total_kv_blocks=scheduler.kv_cache_manager.block_pool.num_gpu_blocks,
        free_kv_blocks=scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
        requests={rid: dict(computed=int(r.num_computed_tokens), prompt=int(r.num_prompt_tokens),
            num_preemptions=int(r.num_preemptions), status=r.status.name,
            kv_block_ids=scheduler.kv_cache_manager.get_block_ids(rid))
            for rid, r in scheduler.requests.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "workload", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--chunk", type=int, choices=(8, 32), required=True)
    p.add_argument("--new-prompt-length", type=int, choices=(8, 128), default=128)
    p.add_argument("--no-new", action="store_true")
    p.add_argument("--trace", action="store_true")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        handle.write('{}\n')
    result = dict(status="STARTING", args={k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()}, steps=[], requests={})
    def save():
        pending = args.out.with_name(args.out.name + ".pending")
        pending.write_text(json.dumps(result, indent=2) + "\n")
        pending.replace(args.out)
    def terminate(signum, frame):
        raise RuntimeError(f"Interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, terminate)
    save()
    try:
        raw = args.workload.read_bytes()
        workload = json.loads(raw)
        sources = workload["requests"] if isinstance(workload, dict) else workload
        require(len(sources) == 3 and len({r["request_id"] for r in sources}) == 3, "Need three unique requests")
        require([len(r["prompt_token_ids"]) for r in sources] == [32, 32, 128], "Expected prompt lengths 32/32/128")
        require(all(isinstance(r["document_id"], str) and r["document_id"] for r in sources), "Missing document IDs")
        for index, source in enumerate(sources):
            ids = source["prompt_token_ids"][:args.new_prompt_length] if index == 2 else source["prompt_token_ids"]
            result["requests"][source["request_id"]] = dict(source, prompt_token_ids=ids,
                max_tokens=8 if index == 2 else 16, output_token_ids=[], token_received_s=[], status="NOT_ARRIVED")
        model_config = json.loads((args.model / "config.json").read_text())
        require(model_config.get("model_type") == "olmoe" and model_config.get("num_hidden_layers") == 16, "Need OLMoE16")
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_USE_V1="1",
            VLLM_ENABLE_V1_MULTIPROCESSING="0", WISP_PLUGIN_DISABLE="0", WISP_MODE="paged",
            WISP_CAP_EXPERTS="16", WISP_PREFETCH="0", WISP_DYNAMIC="0")
        import torch
        import vllm
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import RequestOutputKind
        from wisp.integrations.vllm import install_wisp_moe
        require(vllm.__version__.split("+")[0] == "0.11.2", "Requires vLLM0.11.2")
        install_wisp_moe(mode="paged", cap_experts=16)
        config = dict(model=str(args.model.resolve()), dtype="bfloat16", seed=0,
            enforce_eager=True, max_model_len=256, max_num_seqs=3, max_num_batched_tokens=64,
            long_prefill_token_threshold=32, enable_chunked_prefill=True, enable_prefix_caching=False,
            async_scheduling=False, speculative_config=None, stream_interval=1, block_size=16,
            kv_cache_memory_bytes=512 * 1024**2, gpu_memory_utilization=0.75,
            tensor_parallel_size=1, trust_remote_code=False, disable_log_stats=True)
        result.update(status="LOADING", engine_config=config, model_config=model_config,
            runtime=dict(vllm=vllm.__version__, torch=torch.__version__, cuda=torch.version.cuda),
            workload_sha256=hashlib.sha256(raw).hexdigest(), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        save()
        llm = LLM(**config)
        engine = llm.llm_engine
        scheduler = engine.engine_core.engine_core.scheduler
        snapshot = lambda reset=False: engine.engine_core.collective_rpc(full_snapshot, args=(reset,))
        result["initialization"] = snapshot(True)
        require(all(len(w["pager_layers"]) == 16 and all(s["cap"] == 16 and s["master_pinned"] for s in w["pager_layers"])
                    and w["kv_allocated_tensor_bytes"] == 512 * 1024**2
                    for w in result["initialization"]), "Unexpected pager or KV allocation")
        require(engine.vllm_config.scheduler_config.async_scheduling is False and engine.vllm_config.speculative_config is None
                and engine.vllm_config.cache_config.enable_prefix_caching is False, "Unexpected async/spec/prefix execution")
        params = lambda count: SamplingParams(temperature=0, seed=0, max_tokens=count,
            min_tokens=count, ignore_eos=True, detokenize=False, output_kind=RequestOutputKind.CUMULATIVE)
        result["warmup"] = dict(name="warmup")
        TRACE.__init__()
        run_requests(engine, [llm.get_tokenizer().encode(s) for s in PROMPTS], params(8), result["warmup"], TRACE if args.trace else None)
        if args.trace:
            result["warmup"]["paging_trace"] = TRACE.export()
        result["measurement_initial"] = snapshot(True)
        TRACE.__init__()
        require(scheduler.scheduler_config.long_prefill_token_threshold == 32, "Pre-action threshold changed")
        result["status"] = "MEASURING"
        save()
        origin, wall_origin = time.perf_counter(), time.time()
        now = lambda: time.perf_counter() - origin
        old_ids, new_id = [r["request_id"] for r in sources[:2]], sources[2]["request_id"]
        def submit(rid):
            row = result["requests"][rid]
            row.update(arrival_s=now(), status="SUBMITTED")
            returned = engine.add_request(rid, {"prompt_token_ids": row["prompt_token_ids"]}, params(row["max_tokens"]), arrival_time=wall_origin + row["arrival_s"])
            require(returned is None and rid in scheduler.requests, "v0.11 request identity mismatch")
        for rid in old_ids:
            submit(rid)
        original = scheduler.schedule
        def schedule():
            call = result["steps"][-1]
            before = scheduler_state(scheduler)
            call["before"] = before
            out = original()
            after = scheduler_state(scheduler)
            call.update(after=after, scheduled=[], total_scheduled_tokens=int(out.total_num_scheduled_tokens))
            for rid, amount in out.num_scheduled_tokens.items():
                start = after["requests"][rid]["computed"] - amount
                prefill = min(amount, max(0, before["requests"][rid]["prompt"] - start))
                call["scheduled"].append(dict(request_id=rid, tokens=amount, start_computed=start,
                    prefill_tokens=prefill, decode_tokens=amount-prefill,
                    computed_adjustment=start-before["requests"][rid]["computed"]))
            preempted = [rid for rid, r in after["requests"].items() if r["num_preemptions"] > before["requests"][rid]["num_preemptions"]
                         or r["computed"] < before["requests"][rid]["computed"] or r["status"] == "PREEMPTED"]
            call["preempted_request_ids"] = preempted
            require(not preempted, "Preemption invalidates this injection cell")
            require(all(r["computed_adjustment"] == 0 for r in call["scheduled"]), "Computed-state reset/adjustment")
            for rid in old_ids:
                if rid in before["running"] and before["requests"][rid]["computed"] >= before["requests"][rid]["prompt"]:
                    require(out.num_scheduled_tokens.get(rid) == 1, "Old decode did not advance exactly once")
            if result.get("action", {}).get("engine_call") == call["index"]:
                expected = 2 if args.no_new else 2 + min(args.chunk, args.new_prompt_length)
                require(out.total_num_scheduled_tokens == expected, "First injection shape mismatch")
            return out
        scheduler.schedule = schedule
        while engine.has_unfinished_requests():
            if "action" not in result and all(len(result["requests"][rid]["output_token_ids"]) >= 4 for rid in old_ids):
                require(all(len(result["requests"][rid]["output_token_ids"]) == 4 for rid in old_ids), "Missed exact action boundary")
                started = now()
                action = result["action"] = dict(engine_call=len(result["steps"]), start_s=started,
                    before_scheduler=scheduler_state(scheduler),
                    old_output_tokens={rid: list(result["requests"][rid]["output_token_ids"]) for rid in old_ids}, threshold_before=32)
                action["snapshot_start_s"] = now()
                action["before_worker"] = snapshot()
                action["snapshot_end_s"] = now()
                scheduler.scheduler_config.long_prefill_token_threshold = args.chunk
                if not args.no_new:
                    submit(new_id)
                else:
                    result["requests"][new_id]["status"] = "NOT_INJECTED"
                action.update(end_s=now(), threshold_after=scheduler.scheduler_config.long_prefill_token_threshold, new_submitted=not args.no_new)
            call = dict(index=len(result["steps"]), output_request_ids=[])
            result["steps"].append(call)
            if args.trace:
                TRACE.begin_step(None, engine_call=call["index"])
            call["start_s"] = now()
            try:
                outputs = engine.step()
                received = now()
                call.update(return_s=received, returned=True, output_request_ids=[o.request_id for o in outputs])
            except BaseException as exc:
                call.update(return_s=now(), returned=False, error=str(exc))
                raise
            finally:
                if args.trace:
                    TRACE.end_step(call.get("error"))
            for output in outputs:
                row = result["requests"][output.request_id]
                require(len(output.outputs) == 1 and row["status"] != "COMPLETED", "Unexpected output")
                ids, previous = list(output.outputs[0].token_ids), row["output_token_ids"]
                require(ids[:len(previous)] == previous, "Cumulative tokens changed")
                row["token_received_s"].extend([received] * (len(ids)-len(previous)))
                row["output_token_ids"] = ids
                if output.finished:
                    row.update(status="COMPLETED", completion_s=received, finish_reason=output.outputs[0].finish_reason)
        result["wall_s"] = now()
        require("action" in result, "Action boundary never reached")
        require(all(result["requests"][rid]["status"] == "COMPLETED" and len(result["requests"][rid]["output_token_ids"]) == result["requests"][rid]["max_tokens"] for rid in old_ids + ([] if args.no_new else [new_id])), "Incomplete requests")
        if args.trace:
            result["paging_trace"] = TRACE.export()
        result.update(status="COMPLETED", final_worker=snapshot(), timing="Host receipt; includes inline observation and action snapshot costs", no_new_scope="Two-request diagnostic only; not same-task throughput" if args.no_new else None)
    except (Exception, KeyboardInterrupt) as exc:
        result.update(status="INVALID" if isinstance(exc, InvalidExperiment) else "FAILED", error=str(exc), traceback=traceback.format_exc())
        if args.trace:
            TRACE.end_step(str(exc))
            result["partial_trace"] = TRACE.export(resolve=False)
        save()
        return 1
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

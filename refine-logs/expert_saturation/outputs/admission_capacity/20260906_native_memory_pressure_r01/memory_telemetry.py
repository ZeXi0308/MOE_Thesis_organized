"""Small native KV observation and stop-before-preemption probe; no GPU sync."""
from __future__ import annotations

import time


class CapacityExhausted(RuntimeError):
    """Native scheduler requested preemption; its destructive method did not run."""


def _storage(tensor):
    storage = tensor.untyped_storage()
    return (str(tensor.device), storage.data_ptr()), int(storage.nbytes())


def memory_snapshot(engine, torch):
    worker = engine.engine_core.engine_core.model_executor.driver_worker.worker
    runner = worker.model_runner
    parameters, experts, caches = {}, {}, {}
    for name, parameter in runner.model.named_parameters(remove_duplicate=False):
        key, size = _storage(parameter)
        parameters[key] = size
        if name.rsplit(".", 1)[-1] in ("w13_weight", "w2_weight"):
            experts[key] = size
    for entry in runner.kv_caches:
        for tensor in entry if isinstance(entry, (tuple, list)) else (entry,):
            key, size = _storage(tensor)
            caches[key] = size
    device = runner.device
    return dict(parameter_storage_bytes=sum(parameters.values()),
        expert_w13_w2_storage_bytes=sum(experts.values()), kv_storage_bytes=sum(caches.values()),
        parameter_storage_count=len(parameters), kv_storage_count=len(caches),
        torch_allocated_bytes=int(torch.cuda.memory_allocated(device)),
        torch_reserved_bytes=int(torch.cuda.memory_reserved(device)),
        torch_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        torch_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        accounting="Expert storage is a subset of parameters; KV used blocks are a subset of KV storage. "
                   "Torch reserved includes allocated. These fields must not be summed.")


def capture_with_memory(engine, capture_fn, workload, config, **kwargs):
    scheduler = engine.engine_core.engine_core.scheduler
    manager = scheduler.kv_cache_manager
    pool = manager.block_pool
    trace, boundary, current = [], None, None
    saved = []

    def replace(obj, name, method):
        saved.append((obj, name, name in vars(obj), getattr(obj, name)))
        setattr(obj, name, method)

    def pool_state():
        total, free = int(pool.num_gpu_blocks), int(pool.get_num_free_blocks())
        return dict(total_blocks=total, usable_blocks=total - 1, free_blocks=free,
                    used_blocks=total - 1 - free)

    def request_state(request):
        groups = manager.get_blocks(request.request_id).blocks
        return dict(computed_tokens=int(request.num_computed_tokens),
            prompt_tokens=int(request.num_prompt_tokens),
            num_preemptions=int(request.num_preemptions),
            block_counts=[len(group) for group in groups])

    def state():
        return dict(pool=pool_state(), running_ids=[r.request_id for r in scheduler.running],
            waiting_count=scheduler.get_request_counts()[1],
            requests={rid: request_state(req) for rid, req in scheduler.requests.items()})

    original_schedule, original_allocate = scheduler.schedule, manager.allocate_slots

    def allocate(request, *args, **kw):
        before = pool_state()
        result = original_allocate(request, *args, **kw)
        if result is None and current is not None:
            current["allocation_failures"].append(dict(internal_request_id=request.request_id,
                requested_tokens=int(args[0] if args else kw["num_new_tokens"]),
                before=before, after=pool_state(), computed_tokens=int(request.num_computed_tokens)))
        return result

    def stop_preemption(request, timestamp):
        nonlocal boundary
        boundary = dict(victim_internal_request_id=request.request_id,
            attempted_step=len(trace), native_timestamp=timestamp,
            host_perf_counter_s=time.perf_counter(), victim_state=request_state(request),
            pool=pool_state(), original_preemption_called=False, model_execution_confirmed=False,
            hook_boundary="Caller already removed victim from running; see schedule-before snapshot.")
        raise CapacityExhausted("KV capacity requires native preemption; episode stopped before KV release")

    def schedule(*args, **kw):
        nonlocal current
        current = dict(attempted_step=len(trace), before=state(), after=None,
            host_start_perf_counter_s=time.perf_counter(), allocation_failures=[],
            schedule_completed=False, model_execution_confirmed=None,
            existing_decode_not_scheduled_ids=[])
        try:
            result = original_schedule(*args, **kw)
            before = current["before"]
            existing = [rid for rid in before["running_ids"]
                if before["requests"][rid]["computed_tokens"] >= before["requests"][rid]["prompt_tokens"]]
            current["existing_decode_not_scheduled_ids"] = [rid for rid in existing
                if result.num_scheduled_tokens.get(rid, 0) <= 0]
            current["schedule_completed"] = True
            return result
        except Exception as exc:
            current["error"] = f"{type(exc).__name__}: {exc}"
            current["model_execution_confirmed"] = False
            raise
        finally:
            current["after"] = state()
            current["host_end_perf_counter_s"] = time.perf_counter()
            trace.append(current)
            current = None

    try:
        replace(scheduler, "schedule", schedule)
        replace(manager, "allocate_slots", allocate)
        replace(scheduler, "_preempt_request", stop_preemption)
        raw = capture_fn(engine, workload, config, **kwargs)
    finally:
        for obj, name, had_instance, original in reversed(saved):
            if had_instance:
                setattr(obj, name, original)
            elif name in vars(obj):
                delattr(obj, name)
    raw.update(memory_trace=trace, capacity_boundary=boundary,
        block_count_semantics="Per-group block-table lengths may include null placeholders; global pool used excludes null.",
        strict_nonpreemptive_status="CAPACITY_BOUNDARY_STOP" if boundary is not None else (
            "COMPLETE_NO_PREEMPTION" if raw["status"] == "COMPLETE" else "INCOMPLETE_NO_PREEMPTION"),
        memory_trace_clock="Absolute perf_counter seconds; join native capture using attempted_step. "
            "Successful schedule alone does not confirm GPU execution. Failed attempts never executed.")
    return raw

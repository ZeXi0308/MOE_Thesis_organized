"""Small read-only helpers; host access follows the pinned A native_host_snapshot."""
import sys


def resource_snapshot(engine, runner, cpu_gib):
    scheduler = engine.engine_core.engine_core.scheduler
    pool, connector = scheduler.kv_cache_manager.block_pool, scheduler.connector
    expected = "OffloadingConnector" if cpu_gib else "NoneType"
    if type(connector).__name__ != expected:
        raise RuntimeError("effective KV connector differs from requested arm")
    unique = lambda ts: {(str(t.device), t.untyped_storage().data_ptr()):
                         t.untyped_storage().nbytes() for t in ts}
    out = dict(connector=expected, gpu_kv_unique_bytes=sum(unique(runner.kv_caches).values()),
        gpu_blocks=pool.num_gpu_blocks, free_gpu_blocks=pool.get_num_free_blocks(),
        cpu_kv_unique_bytes=0, cpu_capacity_blocks=0, cpu_gib_requested=cpu_gib)
    if connector is not None:
        cs = connector.connector_scheduler
        worker_connector = getattr(sys.modules.get("vllm.distributed.kv_transfer.kv_transfer_state"),
                                   "_KV_CONNECTOR_AGENT", None)
        if type(worker_connector).__name__ != expected or cs.config.num_workers != 1:
            raise RuntimeError("requires the pinned single in-process native offload worker")
        worker = worker_connector.connector_worker.worker
        tensors = list(worker._store_handler.dst_tensors) + list(worker._load_handler.src_tensors)
        if not tensors or any(t.device.type != "cpu" for t in tensors):
            raise RuntimeError("native CPU KV tensors missing or unexpected")
        manager, refs = cs.manager, [b.ref_cnt for b in cs.manager._policy.blocks.values()]
        out.update(cpu_kv_unique_bytes=sum(unique(tensors).values()),
            cpu_capacity_blocks=manager._num_blocks, cpu_valid_entries=sum(r >= 0 for r in refs),
            cpu_pending_store_entries=sum(r == -1 for r in refs),
            cpu_all_tensors_pinned=all(t.is_pinned() for t in tensors),
            pending_transfer_jobs=len(cs._jobs), pending_push_work=connector.has_pending_push_work())
        if out["cpu_kv_unique_bytes"] != cpu_gib * 1024**3:
            raise RuntimeError("actual CPU KV allocation differs from the requested whole-GiB budget")
    return out


def scheduled_overlap(raw):
    """Offline interval union: repeated scheduled positions, never a time saving."""
    seen, repeated, scheduled = {}, {}, 0
    for step in raw["scheduler_steps"]:
        for row in step["scheduled"]:
            rid, start, amount = row["request_id"], row["scheduled_start_computed"], row["scheduled_tokens"]
            end = start + amount
            if start < 0 or amount <= 0:
                raise ValueError("invalid scheduled token interval")
            intervals = seen.setdefault(rid, [])
            repeated[rid] = repeated.get(rid, 0) + sum(max(0, min(end, b)-max(start, a)) for a, b in intervals)
            merged = []
            for a, b in sorted(intervals + [(start, end)]):
                if merged and a <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
                else:
                    merged.append((a, b))
            seen[rid], scheduled = merged, scheduled + amount
    return dict(total_scheduled_tokens=scheduled, repeated_scheduled_positions=sum(repeated.values()),
        per_request_repeated_positions=repeated, capture_complete=raw["status"] == "COMPLETE",
        scope="Overlap of native scheduled token-position intervals; no GPU completion claim for an incomplete capture; not elapsed-time savings")

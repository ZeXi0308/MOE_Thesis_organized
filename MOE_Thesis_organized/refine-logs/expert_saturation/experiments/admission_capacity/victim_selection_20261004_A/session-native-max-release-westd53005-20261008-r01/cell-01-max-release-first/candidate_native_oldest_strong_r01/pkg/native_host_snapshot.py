"""On-demand, read-only native CPU KV snapshots for initialized vLLM 0.26."""
import os
from pathlib import Path
import sys
import time


def snapshot(engine, *, parent_cgroup="/sys/fs/cgroup"):
    """Call at selected boundaries; never create workers, tensors, or monitors."""
    out = dict(unix_s=time.time(), monotonic_s=time.perf_counter(), pid=os.getpid(),
        errors={}, cpu_kv=None, manager=None, pending=None, process_rss_bytes=None,
        parent_cgroup=dict(path=str(parent_cgroup), values={}),
        accounting="CPU KV storage, process RSS and shared cgroup charges overlap; do not add them. "
                   "The parent limit is shared, not an independent process or KV hard cap. Peaks retain prior process/cgroup history.",
        transfer_semantics="Pending metadata is not a CUDA completion/fence observation; no event queries.")

    def read(name, fn):
        try:
            return fn()
        except Exception as exc:
            out["errors"][name] = f"{type(exc).__name__}: {exc}"
            return None

    def rss(field="VmRSS"):
        row = next(s for s in Path("/proc/self/status").read_text().splitlines() if s.startswith(field + ":"))
        value, unit = row.split()[1:]
        if unit != "kB":
            raise ValueError("Unexpected VmRSS unit")
        return int(value) * 1024

    out["process_rss_bytes"] = read("process_rss", rss)
    out["process_peak_rss_bytes"] = read("process_peak_rss", lambda: rss("VmHWM"))
    out["cgroup_membership"] = read("cgroup_membership", lambda: Path("/proc/self/cgroup").read_text().strip())
    for name in ("memory.max", "memory.current", "memory.peak", "memory.swap.max", "memory.swap.current"):
        def value(name=name):
            text = (Path(parent_cgroup) / name).read_text().strip()
            return text if text == "max" else int(text)
        out["parent_cgroup"]["values"][name] = read(name, value)

    def scheduler_side():
        if getattr(sys.modules.get("vllm"), "__version__", None) != "0.26.0":
            raise ValueError("Requires already loaded vLLM 0.26.0")
        connector = engine.engine_core.engine_core.scheduler.connector
        if type(connector).__name__ != "OffloadingConnector":
            raise ValueError("Native OffloadingConnector required")
        cs = connector.connector_scheduler
        if cs.config.num_workers != 1:
            raise ValueError("Snapshot contract requires one native offload worker")
        return cs

    cs = read("scheduler", scheduler_side)

    def manager_state():
        manager = cs.manager
        if type(manager).__name__ != "CPUOffloadingManager" or type(manager._policy).__name__ != "LRUCachePolicy":
            raise ValueError("Requires the fixed native CPU manager with LRU policy")
        refs = [int(block.ref_cnt) for block in manager._policy.blocks.values()]
        if any(ref < -1 for ref in refs):
            raise ValueError("Unexpected native host block reference count")
        return dict(capacity_blocks=int(manager._num_blocks), entries=len(refs),
            valid_host_kv_entries=sum(ref >= 0 for ref in refs),
            pending_store_entries=sum(ref == -1 for ref in refs),
            entries_with_load_references=sum(ref > 0 for ref in refs),
            active_load_references=sum(ref for ref in refs if ref > 0),
            free_blocks=int(manager._num_blocks - manager._num_allocated_blocks + len(manager._free_list)),
            recorded_pending_store_blocks=int(manager._num_write_pending_blocks),
            semantics="Ready means ref_cnt >= 0; entries with pending stores are not valid cached KV.")

    if cs is not None:
        out["manager"] = read("manager", manager_state)
        out["pending"] = read("pending", lambda: dict(
            scheduler_store_jobs=sum(job.is_store for job in cs._jobs.values()),
            scheduler_load_jobs=sum(not job.is_store for job in cs._jobs.values()),
            pending_worker_acknowledgements=sum(job.pending_count for job in cs._jobs.values())))

    def worker_state():
        if cs is None:
            raise ValueError("Single-worker scheduler contract was not established")
        module = sys.modules.get("vllm.distributed.kv_transfer.kv_transfer_state")
        connector = getattr(module, "_KV_CONNECTOR_AGENT", None)
        if type(connector).__name__ != "OffloadingConnector":
            raise ValueError("Existing in-process worker connector is unavailable")
        wrapper = connector.connector_worker
        worker = wrapper.worker
        if type(worker).__name__ != "CPUOffloadingWorker":
            raise ValueError("Initialized CPUOffloadingWorker is unavailable")
        store, load = worker._store_handler, worker._load_handler
        tensors = list(store.dst_tensors) + list(load.src_tensors)
        if not store.dst_tensors or not load.src_tensors:
            raise ValueError("Initialized host KV tensors are absent")
        storage_rows = {}
        for tensor in tensors:
            if tensor.device.type != "cpu":
                raise ValueError("Expected CPU KV tensor")
            storage = tensor.untyped_storage()
            key = (int(storage.data_ptr()), int(storage.nbytes()))
            if key not in storage_rows:
                storage_rows[key] = dict(storage_address=key[0], bytes=key[1], is_pinned=bool(tensor.is_pinned()))
        total = sum(row["bytes"] for row in storage_rows.values())
        def block_bytes():
            capacity = out["manager"]["capacity_blocks"]
            if capacity <= 0 or total % capacity:
                raise ValueError("Host storage cannot be divided into capacity blocks")
            # Derive only for whole contiguous storage; mmap/sliced layouts stay unknown.
            for tensor in tensors:
                if (tensor.ndim != 2 or tensor.shape[0] != capacity or not tensor.is_contiguous()
                    or tensor.storage_offset() != 0 or tensor.numel() * tensor.element_size() != tensor.untyped_storage().nbytes()):
                    raise ValueError("Derived bytes require verified whole contiguous host tensors")
            return total // capacity
        per_block = read("host_block_layout", block_bytes)
        region = store._mmap_region
        return dict(unique_storage_bytes=total, bytes_per_host_block=per_block,
            derived_valid_host_kv_bytes=None if per_block is None else per_block * out["manager"]["valid_host_kv_entries"],
            derived_bytes_semantics="Ready entries times exact allocated bytes per host block; excludes pending stores, not additional allocation.",
            unique_storages=len(storage_rows), tensor_references=len(tensors),
            storages=list(storage_rows.values()), all_tensors_pinned=all(row["is_pinned"] for row in storage_rows.values()),
            mmap_region_is_pinned=None if region is None else bool(region.is_pinned),
            worker_store_pending_events=len(store._transfer_events), worker_load_pending_events=len(load._transfer_events),
            worker_unsubmitted_store_jobs=len(wrapper._unsubmitted_store_jobs), worker_load_jobs=len(wrapper._load_jobs),
            semantics="Existing CPU KV storage only; load/store aliases counted once. Excludes transfer buffers and allocator overhead.")

    out["cpu_kv"] = read("cpu_kv", worker_state)
    out["status"] = "COMPLETE" if not out["errors"] else "PARTIAL"
    return out

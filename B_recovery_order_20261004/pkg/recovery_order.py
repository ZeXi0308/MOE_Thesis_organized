"""Ready STORE ordering and host observations; no new CUDA event or synchronization."""
import time


def order_stores(entries, required, mode):
    """Fallback on unknown/overlapping CPU destinations; GPU sources are read-only."""
    if mode == 'native' or len(entries) < 2:
        return entries, None
    seen = set()
    for _, _, dst in entries:
        blocks = getattr(dst, 'block_ids', None)
        if blocks is None:
            return entries, 'unknown_destination'
        ids = set(map(int, blocks))
        if seen & ids:
            return entries, 'overlapping_destination'
        seen.update(ids)
    if mode == 'age':
        return sorted(entries, key=lambda e: e[0]), None
    if mode == 'flush_first':
        return sorted(entries, key=lambda e: e[0] not in required), None
    raise ValueError(mode)


def install(scheduler, mode):
    import sys
    if mode not in ('native', 'age', 'flush_first'):
        raise ValueError(mode)
    cs = scheduler.connector.connector_scheduler
    agent = sys.modules['vllm.distributed.kv_transfer.kv_transfer_state']._KV_CONNECTOR_AGENT
    cw = agent.connector_worker
    worker = cw.worker
    if type(worker).__name__ != 'CPUOffloadingWorker':
        raise RuntimeError('Requires native CPUOffloadingWorker')
    data = dict(mode=mode, events=[], clock='time.perf_counter host; GPU elapsed separate',
                semantics='GPU completion is only observed by native query/wait; no absolute GPU timestamp')
    undo, identities, recovering, last_lookup, allocation = [], {}, set(), {}, {}

    def emit(kind, **fields):
        data['events'].append(dict(kind=kind, host_perf_s=time.perf_counter(), **fields))

    def wrap(obj, name, factory):
        old, own = getattr(obj, name), name in vars(obj)
        setattr(obj, name, factory(old))
        undo.append((obj, name, old, own))

    def identity(jid):
        job = cs._jobs.get(jid)
        if job is not None:
            identities[jid] = dict(request=job.req_id, is_store=job.is_store)
        return identities.get(jid, dict(request=None, is_store=None))

    def build(old):
        def call(output):
            meta = old(output)
            for stores, jobs in ((True, meta.store_jobs), (False, meta.load_jobs)):
                for jid in jobs:
                    emit('job_created', job_id=jid, **identity(jid),
                         required_flush=jid in meta.jobs_to_flush)
            if meta.jobs_to_flush:
                emit('flush_dependency', required=sorted(meta.jobs_to_flush),
                     preempted=list(output.preempted_req_ids or []),
                     loads=[dict(job_id=j, request=e.req_id) for j, e in meta.load_jobs.items()])
            return meta
        return call

    def handle(old):
        def call(meta):
            # Build the exact native ready set. Original sees popped entries absent,
            # submits this list, clears it, then waits on the unchanged required IDs.
            entries = list(cw._unsubmitted_store_jobs)
            for jid in meta.jobs_to_flush:
                entry = meta.store_jobs.pop(jid, None)
                if entry is not None:
                    entries.append((jid, entry.src_spec, entry.dst_spec))
            legal_flush, flush_fallback = order_stores(entries, meta.jobs_to_flush, 'flush_first')
            result, fallback = (legal_flush, flush_fallback) if mode == 'flush_first' else order_stores(entries, meta.jobs_to_flush, mode)
            for jid, _, _ in entries:
                emit('ready', job_id=jid, **identity(jid), required_flush=jid in meta.jobs_to_flush)
            if entries:
                before, after = [e[0] for e in entries], [e[0] for e in result]
                emit('reorder', before=before, after=after, required=sorted(meta.jobs_to_flush),
                     changed=before != after, fallback=fallback,
                     legal_flush_order=[e[0] for e in legal_flush], flush_fallback=flush_fallback)
            cw._unsubmitted_store_jobs[:] = result
            return old(meta)
        return call

    def start(old):
        def call(meta):
            for jid, _, _ in cw._unsubmitted_store_jobs:
                emit('ready', job_id=jid, **identity(jid), required_flush=False)
            for jid in meta.load_jobs:
                emit('ready', job_id=jid, **identity(jid), required_flush=False)
            return old(meta)
        return call

    def submit(old):
        def call(jid, src, dst):
            emit('submit_begin', job_id=jid, **identity(jid))
            result = old(jid, src, dst)
            emit('submit_end', job_id=jid, **identity(jid), accepted=result)
            return result
        return call

    def finished(old):
        def call():
            results = old()
            for r in results:
                emit('job_completed', job_id=r.job_id, **identity(r.job_id),
                     bytes=r.transfer_size, gpu_elapsed_s=r.transfer_time)
            return results
        return call

    def wait(old):
        def call(jids):
            emit('wait_begin', required=sorted(jids))
            result = old(jids)
            emit('wait_end', required=sorted(jids))
            return result
        return call

    def ack(old):
        def call(output):
            meta = output.kv_connector_worker_meta
            jobs = list(meta.completed_jobs) if meta is not None else []
            for jid in jobs:
                identity(jid)
            result = old(output)
            for jid in jobs:
                if jid not in cs._jobs:
                    emit('ack_retired', job_id=jid, **identity(jid))
            return result
        return call

    def preempt(old):
        def call(req, *args, **kwargs):
            recovering.add(req.request_id)
            emit('preempt', request=req.request_id)
            result = old(req, *args, **kwargs)
            emit('logical_free_after_preempt', request=req.request_id)
            return result
        return call

    def lookup(old):
        def call(req, computed):
            result = old(req, computed)
            if req.request_id in recovering and last_lookup.get(req.request_id) != result:
                emit('lookup', request=req.request_id, result=list(result),
                     pending_jobs=sorted(cs._req_status[req.request_id].transfer_jobs))
                last_lookup[req.request_id] = result
            return result
        return call

    def allocate(old):
        def call(req, *args, **kwargs):
            result = old(req, *args, **kwargs)
            if req.request_id in recovering:
                ok = result is not None
                if allocation.get(req.request_id) != ok:
                    emit('allocation_ok' if ok else 'capacity_wait', request=req.request_id)
                    allocation[req.request_id] = ok
            return result
        return call

    def schedule(old):
        def call(*args, **kwargs):
            result = old(*args, **kwargs)
            for rid, count in result.num_scheduled_tokens.items():
                if rid in recovering:
                    emit('scheduled', request=rid, scheduled_tokens=count)
                    recovering.remove(rid)
                    allocation.pop(rid, None)
                    last_lookup.pop(rid, None)
            return result
        return call

    for obj, name, factory in [(cs, 'build_connector_meta', build),
            (cw, 'handle_preemptions', handle), (cw, 'start_kv_transfers', start), (worker, 'submit_store', submit),
            (worker, 'submit_load', submit), (worker, 'get_finished', finished),
            (worker, 'wait', wait), (cs, 'update_connector_output', ack),
            (scheduler, '_preempt_request', preempt), (scheduler, 'schedule', schedule),
            (cs, 'get_num_new_matched_tokens', lookup),
            (scheduler.kv_cache_manager, 'allocate_slots', allocate)]:
        wrap(obj, name, factory)

    def uninstall():
        for obj, name, old, own in reversed(undo):
            if own:
                setattr(obj, name, old)
            else:
                delattr(obj, name)
        return data
    return data, uninstall

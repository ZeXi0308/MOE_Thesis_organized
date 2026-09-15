"""Pinned OffloadingConnector observations; no scheduler policy changes."""
import time
import sys


def install(connector, *, diagnostic=False, host_snapshot=None):
    data = {'lookup': [], 'transfers': [], 'completed_jobs': [],
            'dispatch': [], 'host_snapshots': [], 'diagnostic': diagnostic,
            'detailed_logging': 'ENABLED' if diagnostic else 'DISABLED',
            'worker_wrapped': False}
    if connector is None:
        return data, lambda: None
    if type(connector).__name__ != 'OffloadingConnector':
        raise ValueError('unqualified connector')
    target = connector.connector_scheduler
    lookup, update = target.get_num_new_matched_tokens, target.update_connector_output
    worker = None
    worker_methods = {}

    def sample_host(event, **context):
        if not diagnostic or host_snapshot is None:
            return
        row = dict(event=event, host_perf_counter_s=time.perf_counter(), **context)
        try:
            row['actualhost'] = host_snapshot()
        except Exception as exc:
            row.update(actualhost=None, error=repr(exc))
        data['host_snapshots'].append(row)

    def job_identity(job_id, is_store):
        job = target._jobs.get(job_id)
        return dict(job_id=job_id, request=job.req_id if job is not None else None,
                    is_store=is_store)

    def wrap_submit(original, is_store):
        def submit(job_id, src, dst):
            identity = job_identity(job_id, is_store)
            sample_host('dispatch_before', **identity)
            row = dict(identity, before_perf_s=time.perf_counter(), accepted=None)
            try:
                result = original(job_id, src, dst)
                row['accepted'] = result
                return result
            except Exception as exc:
                row['error'] = repr(exc)
                raise
            finally:
                row['after_perf_s'] = time.perf_counter()
                data['dispatch'].append(row)
                sample_host('dispatch_after', **identity)
        return submit

    if diagnostic:
        module = sys.modules.get('vllm.distributed.kv_transfer.kv_transfer_state')
        agent = getattr(module, '_KV_CONNECTOR_AGENT', None)
        worker = getattr(getattr(agent, 'connector_worker', None), 'worker', None)
        if worker is None or type(worker).__name__ != 'CPUOffloadingWorker':
            raise RuntimeError('Existing CPUOffloadingWorker required for diagnostics')
        for name in ('submit_load', 'submit_store'):
            original = getattr(worker, name)
            if not callable(original):
                raise RuntimeError('Existing worker submit method is not callable')
            worker_methods[name] = (name in vars(worker), original)

    def observed_lookup(request, num_computed_tokens):
        before = time.perf_counter()
        result = lookup(request, num_computed_tokens)
        data['lookup'].append(dict(request=request.request_id, computed=num_computed_tokens,
                                   matched=result[0], asynchronous=result[1], seconds=time.perf_counter()-before))
        return result

    def observed_update(output):
        meta = output.kv_connector_worker_meta
        if meta is not None and meta.completed_jobs:
            completed = dict(time_s=time.perf_counter(), jobs=[
                dict(job_id=jid, count=count, request=target._jobs[jid].req_id,
                     is_store=target._jobs[jid].is_store)
                for jid,count in meta.completed_jobs.items() if jid in target._jobs])
            data['completed_jobs'].append(completed)
            sample_host('completed_jobs', jobs=completed['jobs'])
        if diagnostic and meta is not None and not meta.transfer_stats.is_empty():
            row = {}
            for name in ('load', 'store'):
                stats = getattr(meta.transfer_stats, name)
                row[name] = dict(bytes=stats.bytes, time=stats.time, sizes=list(stats.sizes))
            data['transfers'].append(row)
        return update(output)

    if diagnostic:
        target.get_num_new_matched_tokens = observed_lookup
        for name, (_, original) in worker_methods.items():
            setattr(worker, name, wrap_submit(original, name == 'submit_store'))
        data['worker_wrapped'] = True
    target.update_connector_output = observed_update

    def uninstall():
        if diagnostic:
            target.get_num_new_matched_tokens = lookup
            for name, (had_instance, original) in worker_methods.items():
                if had_instance:
                    setattr(worker, name, original)
                else:
                    delattr(worker, name)
        target.update_connector_output = update
    return data, uninstall


def drain(engine):
    connector = engine.engine_core.engine_core.scheduler.connector
    start = time.perf_counter()
    calls = 0
    while connector is not None and connector.has_pending_push_work():
        if time.perf_counter()-start > 30:
            raise RuntimeError('pending offload did not drain')
        outputs = engine.step()
        if outputs:
            raise RuntimeError('unexpected request outputs after capture completion')
        calls += 1
    return dict(calls=calls, seconds=time.perf_counter()-start)

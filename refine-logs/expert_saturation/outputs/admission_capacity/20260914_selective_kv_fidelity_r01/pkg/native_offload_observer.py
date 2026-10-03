"""Pinned OffloadingConnector observations; no scheduler policy changes."""
import time


def install(connector):
    data = {'lookup': [], 'transfers': [], 'completed_jobs': []}
    if connector is None:
        return data, lambda: None
    if type(connector).__name__ != 'OffloadingConnector':
        raise ValueError('unqualified connector')
    target = connector.connector_scheduler
    lookup, update = target.get_num_new_matched_tokens, target.update_connector_output

    def observed_lookup(request, num_computed_tokens):
        before = time.perf_counter()
        result = lookup(request, num_computed_tokens)
        data['lookup'].append(dict(request=request.request_id, computed=num_computed_tokens,
                                   matched=result[0], asynchronous=result[1], seconds=time.perf_counter()-before))
        return result

    def observed_update(output):
        meta = output.kv_connector_worker_meta
        if meta is not None and meta.completed_jobs:
            data['completed_jobs'].append(dict(time_s=time.perf_counter(), jobs=[
                dict(job_id=jid, count=count, request=target._jobs[jid].req_id,
                     is_store=target._jobs[jid].is_store)
                for jid,count in meta.completed_jobs.items() if jid in target._jobs]))
        if meta is not None and not meta.transfer_stats.is_empty():
            row = {}
            for name in ('load', 'store'):
                stats = getattr(meta.transfer_stats, name)
                row[name] = dict(bytes=stats.bytes, time=stats.time, sizes=list(stats.sizes))
            data['transfers'].append(row)
        return update(output)

    target.get_num_new_matched_tokens = observed_lookup
    target.update_connector_output = observed_update

    def uninstall():
        target.get_num_new_matched_tokens = lookup
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

"""Optional native completion-stat counters, installed after warmup drain/reset.

No policy changes, worker wrappers, event queries, per-transfer records, or host
snapshots. The caller marks capture end before post-request drain and uninstalls
after drain. Counts are attributed by metadata-observation time: a transfer can
start in capture and complete during drain. Summed native transfer durations are
copy work, potentially overlapping; they are never exposed wall-clock latency.
"""
from functools import wraps
from math import isfinite
import time


def _empty():
    return dict(native_update_calls=0,metadata_updates=0,completed_job_records=0,
                completed_worker_acknowledgements=0,
                completion_acks_without_transfer_stats=0,
                load=dict(bytes=0,transfer_records=0,reported_copy_time_sum=0.0),
                store=dict(bytes=0,transfer_records=0,reported_copy_time_sum=0.0))


def _add(destination,source):
    for name in ('native_update_calls','metadata_updates','completed_job_records',
                 'completed_worker_acknowledgements','completion_acks_without_transfer_stats'):
        destination[name]+=source[name]
    for kind in ('load','store'):
        for name in ('bytes','transfer_records','reported_copy_time_sum'):
            destination[kind][name]+=source[kind][name]


def _snapshot(output):
    row=_empty();row['native_update_calls']=1
    meta=output.kv_connector_worker_meta
    if meta is None:return row
    row['metadata_updates']=1
    row['completed_job_records']=len(meta.completed_jobs)
    row['completed_worker_acknowledgements']=sum(meta.completed_jobs.values())
    for kind in ('load','store'):
        native=getattr(meta.transfer_stats,kind)
        if (type(native.bytes) is not int or native.bytes<0
                or not isinstance(native.time,(float,int)) or not isfinite(native.time) or native.time<0):
            raise ValueError('Invalid native aggregate transfer counters')
        row[kind]=dict(bytes=native.bytes,transfer_records=len(native.sizes),
                       reported_copy_time_sum=native.time)
    row['completion_acks_without_transfer_stats']=max(0,
        row['completed_worker_acknowledgements']-row['load']['transfer_records']-row['store']['transfer_records'])
    return row


def install(connector):
    """Return (data, mark_capture_end, uninstall) for the pinned native backend.

    The native worker emits deltas and resets metadata after each build. The
    observer accumulates a delta only after the original update returns normally.
    Counter extraction failure marks the observation partial without changing
    the original update's behavior. Native exceptions propagate unchanged.
    """
    if connector is None or type(connector).__name__!='OffloadingConnector':
        raise ValueError('Native OffloadingConnector required')
    target=connector.connector_scheduler
    original=target.update_connector_output
    had_instance='update_connector_output' in vars(target)
    phase='capture';closed=False
    data=dict(status='INSTALLED',observer='NATIVE_COMPLETION_AGGREGATES_V1',
        installed_perf_s=time.perf_counter(),capture_end_perf_s=None,uninstalled_perf_s=None,
        capture=_empty(),post_capture_drain=_empty(),capture_plus_drain=None,
        observation_errors=0,last_observation_error=None,
        warmup_scope='Caller must install after warmup drain and successful connector reset.',
        volume_scope='Native completed-transfer deltas; capture/drain split uses metadata observation, not transfer start.',
        job_count_scope='completed_job_records counts metadata entries, not deduplicated job IDs; worker acknowledgements retain native multiplicity.',
        coverage_scope='Native transfer stats exist only when worker reports size and timing. Unmatched completion acknowledgements mark missing transfer stats rather than zero transfer.',
        time_scope='reported_copy_time_sum preserves native transfer_time units and is summed copy work; not exposed wall time. Copies may overlap or cross capture/drain boundaries.',
        overhead_scope='Observer code executes inside timed native updates; complete observer CPU overhead is not separately measured.',
        worker_wrapped=False,per_transfer_logs=False)

    @wraps(original)
    def observed_update(connector_output):
        snapshot=None
        try:snapshot=_snapshot(connector_output)
        except Exception as error:
            data['observation_errors']+=1
            data['last_observation_error']=type(error).__name__+': '+str(error)
        result=original(connector_output)
        if snapshot is not None:_add(data[phase],snapshot)
        return result

    target.update_connector_output=observed_update

    def mark_capture_end():
        nonlocal phase
        if closed:raise RuntimeError('Observer already uninstalled')
        if data['capture_end_perf_s'] is not None:raise RuntimeError('Capture end already marked')
        data['capture_end_perf_s']=time.perf_counter()
        phase='post_capture_drain'

    def uninstall():
        nonlocal closed
        if closed:return data
        if target.update_connector_output is not observed_update:
            raise RuntimeError('Aggregate observer hook ownership changed')
        if had_instance:target.update_connector_output=original
        else:delattr(target,'update_connector_output')
        closed=True;data['uninstalled_perf_s']=time.perf_counter()
        total=_empty();_add(total,data['capture']);_add(total,data['post_capture_drain'])
        data['capture_plus_drain']=total
        data['status']=('PARTIAL' if data['observation_errors'] else
                        'PARTIAL_TRANSFER_STATS' if total['completion_acks_without_transfer_stats'] else
                        'CAPTURE_END_UNMARKED' if data['capture_end_perf_s'] is None else 'COMPLETE')
        return data

    return data,mark_capture_end,uninstall

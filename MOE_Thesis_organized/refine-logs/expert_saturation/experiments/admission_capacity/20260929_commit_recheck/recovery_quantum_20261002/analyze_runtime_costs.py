#!/usr/bin/env python3
"""Extract bounded runtime costs without treating overlapping copy time as latency.

Reads archive/ (or output/) only. Refuses to overwrite an existing result. This
uses the current A runner's timing/observer contract: application warmup and
connector reset precede the measurement observer; observer uninstall follows
post-request drain. Missing instrumentation stays unavailable, never zero.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def difference(data, end, start):
    return data[end]-data[start] if number(data.get(end)) and number(data.get(start)) else None


def gpu_snapshot(data):
    if not data:return None
    if isinstance(data.get('device'),str):
        fields=next(csv.reader([data['device']]))
        return dict(raw_device=data['device'],compute_processes=data.get('compute_processes'),
                    gpu_name=fields[0].strip(),gpu_uuid=fields[1].strip(),
                    total_memory=fields[2].strip(),used_memory=fields[3].strip(),
                    scope='Boundary snapshot while engine process is alive; not a peak or time average.')
    return data


def host_snapshot(data):
    if not data:return None
    kv=data.get('cpu_kv') or {};manager=data.get('manager') or {}
    return dict(status=data.get('status'),monotonic_s=data.get('monotonic_s'),
        process_rss_bytes=data.get('process_rss_bytes'),
        process_peak_rss_bytes=data.get('process_peak_rss_bytes'),
        peak_scope='Process lifetime including initialization and warmup; not reset for measurement.',
        unique_cpu_kv_storage_bytes=kv.get('unique_storage_bytes'),
        bytes_per_host_block=kv.get('bytes_per_host_block'),
        ready_host_kv_bytes=kv.get('derived_valid_host_kv_bytes'),
        ready_host_kv_entries=manager.get('valid_host_kv_entries'),
        pending_store_entries=manager.get('pending_store_entries'),
        host_capacity_blocks=manager.get('capacity_blocks'),free_host_blocks=manager.get('free_blocks'),
        pending_metadata=data.get('pending'),
        parent_shared_cgroup=data.get('parent_cgroup'),errors=data.get('errors'),
        accounting=data.get('accounting'),
        occupancy_scope='Point-in-time ready cache occupancy; not cumulative transfer volume.')


def phase_logs(path):
    phase='ENGINE_SETUP';rows=[];transfer_rows=[];jit_rows=[]
    if not path.is_file():return dict(status='UNAVAILABLE')
    for lineno,line in enumerate(path.read_text(errors='replace').splitlines(),1):
        if 'PHASE APPLICATION_WARMUP_BEGIN' in line:phase='APPLICATION_WARMUP'
        elif 'PHASE APPLICATION_WARMUP_END' in line:phase='BETWEEN_WARMUP_AND_MEASUREMENT'
        elif 'PHASE MEASUREMENT_BEGIN' in line:phase='MEASUREMENT'
        elif 'PHASE MEASUREMENT_END' in line:phase='POST_MEASUREMENT'
        # Only explicit native timing summaries; never sum overlapping setup
        # subcomponents with engine_init elapsed time.
        if re.search(r'Model loading took|torch\.compile took|Graph capturing finished|'
                     r'init engine .* took|Initial profiling/warmup run took',line):
            rows.append(dict(line=lineno,phase=phase,text=line.strip()))
        if 'JIT compilation during inference' in line:
            jit_rows.append(dict(line=lineno,phase=phase,text=line.strip()))
        if re.search(r'(?:[Ss]tor(?:e|ing)|[Ll]oad(?:ing)?).*(?:[GMK]i?B/s|transfer time|[Tt]ransfer stats)',line):
            transfer_rows.append(dict(line=lineno,printed_phase=phase,text=line.strip()))
    return dict(status='READ',setup_subcomponent_reports=rows,jit_reports=jit_rows,
        printed_transfer_reports=transfer_rows,
        printed_transfer_scope='Print phase is known; accumulation-window start/end are not. No printed totals are attributed to measurement or subtracted as warmup.',
        setup_scope='Native setup subcomponents overlap engine initialization and each other; not added together.')


def offload_costs(data,timing):
    if data.get('observer')=='NATIVE_COMPLETION_AGGREGATES_V1':
        total=data.get('capture_plus_drain') or {}
        return dict(status=data.get('status'),source_status=data.get('status'),
            totals=total,capture_completion_observed_totals=data.get('capture'),
            post_capture_drain_completion_observed_totals=data.get('post_capture_drain'),
            observer_boundaries={k:data.get(k) for k in
                ('installed_perf_s','capture_end_perf_s','uninstalled_perf_s')},
            observation_errors=data.get('observation_errors'),
            volume_scope=data.get('volume_scope'),timing_scope=data.get('time_scope'),
            coverage_scope=data.get('coverage_scope'),job_count_scope=data.get('job_count_scope'),
            exposed_transfer_delay_s=None,
            measurement_partition_status='Completion-observation windows available; transfer start/end wall intervals were not recorded.')
    transfers=data.get('transfers');completed=data.get('completed_jobs');dispatch=data.get('dispatch')
    measured=isinstance(transfers,list) and data.get('status')!='NOT_MEASURED'
    totals={}
    for kind in ('load','store'):
        parts=[row[kind] for row in transfers or [] if isinstance(row.get(kind),dict)]
        totals[kind]=dict(
            bytes=sum(p.get('bytes',0) for p in parts) if measured else None,
            size_sample_count=sum(len(p.get('sizes',[])) for p in parts) if measured else None,
            reported_copy_time_sum=sum(p.get('time',0) for p in parts) if measured else None)
    # Worker metadata resets after each build; diagnostic transfer rows are
    # deltas. Their time field is accumulated worker transfer timing. No row
    # timestamp exists here, so measurement cannot be separated from drain.
    start=timing.get('measurement_start_perf_s');end=timing.get('measurement_return_perf_s')
    classified=Counter();job_ids={'load':set(),'store':set()}
    for batch in completed or []:
        t=batch.get('time_s')
        scope=('measurement' if number(t) and number(start) and number(end) and start<=t<=end
               else 'outside_measurement_or_unknown')
        for job in batch.get('jobs',[]):
            kind='store' if job.get('is_store') is True else 'load' if job.get('is_store') is False else 'unknown'
            classified[scope+'_'+kind+'_job_records']+=1
            if kind in job_ids:job_ids[kind].add(job.get('job_id'))
    return dict(status='OBSERVER_RECORDED' if measured else 'UNAVAILABLE',
        source_status=data.get('status'),source_scope=data.get('scope'),totals=totals,
        volume_scope='Observer lifetime: measurement plus post-request drain, excluding application warmup by current runner contract.' if measured else
                     'No worker transfer observer. Neither bytes nor transfer counts can be inferred from no preemptions or endpoint occupancy.',
        timing_scope='Reported copy-time sums are worker transfer work, possibly overlapping. They are not exposed request delay, GPU utilization, or an additive service-time bucket.',
        measurement_only_transfer_bytes=None,
        measurement_partition_status='UNAVAILABLE: transfer rows have no timestamps; use completed-job timestamps only for job-record classification.' if measured else 'UNAVAILABLE',
        dispatch_records=len(dispatch) if isinstance(dispatch,list) else None,
        unique_completed_job_ids={k:len(v) for k,v in job_ids.items()} if isinstance(completed,list) else None,
        completed_job_record_windows=dict(classified) if isinstance(completed,list) else None,
        exposed_transfer_delay_s=None)


def analyze_cell(cell,receipt):
    archive=cell/'archive'
    if not archive.is_dir():archive=cell/'output'
    timing=read(archive/'timing.json');raw=read(archive/'raw.json')
    policy=read(archive/'selective-store.json');config=read(archive/'config.json')
    pairs={
        'process_total':('process_end_perf_s','process_start_perf_s'),
        'before_engine_init':('engine_init_start_perf_s','process_start_perf_s'),
        'engine_init':('engine_init_end_perf_s','engine_init_start_perf_s'),
        'application_warmup':('warmup_end_perf_s','warmup_start_perf_s'),
        'after_warmup_before_measurement':('measurement_start_perf_s','warmup_end_perf_s'),
        'measurement_wrapper':('measurement_return_perf_s','measurement_start_perf_s'),
        'after_measurement_through_drain':('post_request_drain_end_perf_s','measurement_return_perf_s'),
        'shutdown':('process_end_perf_s','shutdown_start_perf_s')}
    durations={name:difference(timing,*pair) for name,pair in pairs.items()}
    hosts={phase:host_snapshot(read(archive/f'host-{phase}.json'))
           for phase in ('after-init','before','request-end','after')}
    memory={phase:read(archive/f'memory-{phase}.json') or None for phase in ('after-init','before','after')}
    before=hosts['before'] or {};after=hosts['request-end'] or {}
    ready_delta=(after['ready_host_kv_bytes']-before['ready_host_kv_bytes']
                 if number(after.get('ready_host_kv_bytes')) and number(before.get('ready_host_kv_bytes')) else None)
    timer=policy.get('lease_decision_cpu_s');duration=durations['measurement_wrapper']
    return dict(archive=str(archive),status=read(archive/'status.json'),
        resources={k:config.get(k) for k in ('model','requests','cap','fixed_kv_cache_memory_bytes','offload_gib','measurement_mode')},
        elapsed_host_intervals_s=durations,raw_timing_boundaries=timing,
        raw_request_observation_s=raw.get('observation_end_s'),
        raw_request_observation_origin_perf_s=raw.get('measurement_origin_perf_counter_s'),
        timing_accounting='process_total encloses all phases; raw request observation is nested inside measurement_wrapper. No phases or native setup subcomponents are blindly summed.',
        post_request_drain=read(archive/'post-request-drain.json') or None,
        warmup_drain=read(archive/'warmup-offload-drain.json') or None,
        transfers=offload_costs(read(archive/'offload-events.json'),timing),
        host_snapshots=hosts,gpu_memory_snapshots=memory,
        gpu_memory_peak_scope='Current runner resets torch peaks after application warmup. memory-after peaks cover reset through post-measurement/drain, including pre-measurement snapshots; not whole engine peak.',
        gpu_kv_live_occupancy_peak_blocks=None,
        gpu_kv_occupancy_scope='Unavailable without time-series block occupancy; allocated KV storage is not live occupied KV.',
        ready_host_cache_delta_bytes=ready_delta,
        ready_host_cache_delta_scope='Endpoint occupancy delta, not bytes stored/transferred, and not added to allocated CPU KV storage.',
        gpu_during_process_before_measurement=gpu_snapshot(raw.get('gpu_before',{})),
        gpu_during_process_after_measurement=gpu_snapshot(read(archive/'gpu-after.json')),
        controller_before_process=receipt.get('gpu_before'),controller_after_process=receipt.get('gpu_after'),
        controller_process_wall_s=receipt.get('elapsed_wall_s'),
        chooser=dict(host_elapsed_s=timer,
            fraction_of_measurement_wrapper=timer/duration if number(timer) and duration else None,
            scope='perf_counter elapsed in lease_decide, including snapshot construction and choosing; not thread CPU time or complete scheduler/policy overhead.',
            full_controller_cpu_s=None,full_controller_overhead_status='UNAVAILABLE',
            anchor_count=policy.get('oldest_anchor_count'),forced_commits=policy.get('oldest_forced_commits'),
            schedule_calls=policy.get('schedule_calls'),
            zero_timer_semantics='Zero with no anchors means chooser inactive; it does not establish zero cost for the installed policy.'),
        launch_log=phase_logs(cell/'launch.log'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():raise SystemExit('Refusing to overwrite existing result')
    receipt=read(args.session/'receipt.json');receipts={r.get('arm'):r for r in receipt.get('cells',[])}
    arms={cell.name[8:]:analyze_cell(cell,receipts.get(cell.name[8:],{}))
          for cell in sorted(args.session.glob('cell-[0-9][0-9]-*')) if cell.is_dir()}
    if not arms:raise SystemExit('No cell directories found')
    result=dict(status='RUNTIME_COST_EXTRACTION',session=str(args.session),
        session_wall_s=receipt.get('elapsed_wall_s'),session_status=receipt.get('status'),arms=arms,
        evidence_scope='Observed local archived artifacts only. Missing cost instrumentation remains unavailable; copy durations and storage subsets are not additive.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({arm:dict(measurement_s=row['elapsed_host_intervals_s']['measurement_wrapper'],
        init_s=row['elapsed_host_intervals_s']['engine_init'],warmup_s=row['elapsed_host_intervals_s']['application_warmup'],
        transfer_status=row['transfers']['status'],chooser_host_s=row['chooser']['host_elapsed_s'],
        ready_host_cache_delta_bytes=row['ready_host_cache_delta_bytes']) for arm,row in arms.items()},indent=2))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Bounded existing-trace inspection; no donor counterfactual or new measurements."""
import argparse
import bisect
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
TARGET = 'b-normal-0070571-long'


def read(path, sources):
    payload = path.read_bytes()
    sources[str(path.relative_to(BASE))] = hashlib.sha256(payload).hexdigest()
    return json.loads(payload)


def last_receipt(request, at):
    times = request['token_times_s']; count = bisect.bisect_left(times, at)
    return dict(last_client_output_s=times[count-1] if count else None,
                completed_output_tokens=count,
                client_output_age_s=at-times[count-1] if count else None)


def inspect_cell(directory):
    sources = {}; output = directory/'output'
    raw = read(output/'raw.json', sources)
    capacity = read(output/'capacity-handoff.json', sources)
    policy = read(output/'recovery-service-age.json', sources)
    selective = read(output/'selective-store.json', sources)
    recovery = read(output/'recovery-order.json', sources)
    source = read(output/'source-handoff.json', sources)
    request = next(row for row in raw['requests'] if row['request_id']==TARGET)
    rid = request['internal_request_id']; origin = raw['measurement_origin_perf_counter_s']
    mapping = raw['internal_to_source']; by_id = {row['internal_request_id']:row for row in raw['requests']}
    length, start, end, index = max((b-a,a,b,i) for i,(a,b) in enumerate(zip(request['token_times_s'],request['token_times_s'][1:])))
    allocations = []
    for i, event in enumerate(capacity['events']):
        if event.get('request')==rid and event['kind']=='allocate' and start<=event['begin_host_perf_s']-origin<=end:
            allocations.append(dict(event_index=i, begin_s=event['begin_host_perf_s']-origin,
                end_s=event['end_host_perf_s']-origin, record=event))
    failures = [event for event in allocations if event['record']['success'] is False]
    successes = [event for event in allocations if event['record']['success'] is True]
    snapshots = []
    for i, event in enumerate(policy['events']):
        at = event['host_perf_s']-origin
        target = next((row for row in event['candidates'] if row['request']==rid),None)
        if not start<=at<=end or target is None:continue
        context = target['context']
        snapshots.append(dict(event_index=i, observed_s=at,
            baseline_head=mapping.get(event['baseline_head']), selected_request=mapping.get(event['candidate_head']),
            target_waiting_position=event['waiting_before'].index(rid),
            target_was_baseline_head=event['baseline_head']==rid, policy_action=event['action'],
            running_count=event['running_count'], token_budget=event['token_budget'],
            free_gpu_blocks=event['free_gpu_blocks'], native_inflight_reserved_blocks=event['native_inflight_reserved_blocks'],
            target={key:target[key] for key in ('history_tokens','computed_tokens','held_gpu_blocks',
                'full_fit_extra_blocks','conservative_free_required','memory_fit','native_reservation_pass',
                'plain_unshared','computed_and_held_zero','eligible','reasons')},
            target_host_context={key:context.get(key) for key in ('known','ready','state','leading_ready_tokens',
                'pending_keys','registered_jobs','global_load_conflict_key_indices')},
            reconstructed_target_client_receipt=last_receipt(request,at)))
    victim = next(event for event in selective['victim_decisions'] if event['selected']==rid and start<=event['host_perf_counter_s']-origin<=end)
    at = victim['host_perf_counter_s']-origin
    running_rows = [dict(**row, source_request=mapping.get(row['request']),
        reconstructed_client_receipt=last_receipt(by_id[row['request']],at)) for row in victim['candidates']]
    handoff = [{key:value for key,value in event.items() if not isinstance(value,(dict,list))}
               for event in source['events'] if event.get('request')==rid or event.get('kind') in ('native_handoff','release')]
    for event in handoff:event['observed_s']=event['host_perf_s']-origin
    target_loads = [dict(event_index=i, observed_s=event['host_perf_s']-origin, record=event)
        for i,event in enumerate(recovery['events']) if event.get('request')==rid and event['kind']=='job_created'
        and event.get('is_store') is False and start<=event['host_perf_s']-origin<=end]
    preempts = [event for event in raw['preemption_events'] if event.get('internal_request_id')==rid
                and start<=event['method_entered_s']<=end]
    return dict(cell=directory.name, request=TARGET, internal_request_id=rid, sources_sha256=sources,
        clock='All *_s boundaries are original host observations relative to this raw origin; no GPU absolute timestamps.',
        measurement_origin_perf_counter_s=origin, request_status=request['status'], request_flow_s=request['completion_s']-request['arrival_s'],
        longest_gap=dict(start_s=start,end_s=end,duration_s=length,after_output_number=index+1,before_output_number=index+2),
        preemptions_inside_gap=preempts, allocation_observations=allocations,
        capacity_failure_span=dict(observed_failures=len(failures),observed_successes=len(successes),
            first_failure_begin_s=failures[0]['begin_s'],last_failure_end_s=failures[-1]['end_s'],
            first_success_begin_s=successes[0]['begin_s'],
            first_failure_return_to_success_entry_s=successes[0]['begin_s']-failures[0]['end_s'],
            no_target_allocation_between_last_failure_and_success_s=successes[0]['begin_s']-failures[-1]['end_s'],
            success_return_to_next_output_s=end-successes[0]['end_s'],
            semantics='Failed snapshots prove capacity shortfalls only at their observed calls. The interval to success also contains queue waiting; it is not continuous proven capacity insufficiency or a recoverable latency bound.'),
        policy_target_snapshots=snapshots, policy_snapshot_summary=dict(count=len(snapshots),
            memory_fit_count=sum(row['target']['memory_fit'] for row in snapshots),
            conservative_fit_eligible_count=sum(row['target']['eligible'] for row in snapshots),
            maximum_waiting_position=max(row['target_waiting_position'] for row in snapshots)),
        target_load_jobs_created_inside_gap=target_loads, source_handoff_observations=handoff,
        earlier_native_victim_snapshot=dict(observed_s=at,
            delay_to_first_fit_callback_s=snapshots[0]['observed_s']-at,
            header={key:value for key,value in victim.items() if key!='candidates'},
            candidate_count=len(running_rows),qualified_count=sum(row['qualified'] for row in running_rows),
            held_blocks_range=[min(row['held_blocks'] for row in running_rows),max(row['held_blocks'] for row in running_rows)],
            candidates=running_rows,
            semantics='This is the existing unprocessed RUNNING suffix at the earlier native victim decision, including the actual target victim. Its held pages cannot be combined with later waiting free/reserved values to certify an exchange. Qualified is the recorded native victim predicate, not a donor-host-coverage or exchange-safety certificate.'),
        observation_limits=dict(
            running_at_fit_callback='Only running_count is recorded; contemporaneous per-RUNNING membership/page ownership/inflight state is unavailable.',
            running_client_outputs='Raw token receipt times permit retrospective per-ID last receipt at a specified host boundary; that does not reconstruct contemporaneous RUNNING membership or ownership.',
            donor_host_coverage='No same-boundary per-RUNNING ready Host-prefix/ref/pending-transfer coverage is recorded. Waiting candidate context and STORE history are not current donor coverage.',
            continuous_capacity='No interpolation across unobserved states; later policy snapshots and allocator calls are kept separate.',
            exchange_counterfactual='Not constructed. Missing donor safety/source state and displaced-request service cost prevent declaring a legal beneficial swap.'),
        entry_observation='Existing fit callback before the native waiting peek records free/need/reserved and the waiting prefix; exact native allocation observer records its later arguments/result. Any exchange feasibility observation must be at one chosen current boundary, before side effects, and must preserve native preemption/STORE/LOAD ownership paths.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,default=BASE/'recovery_service_age/session-20261009-r01')
    parser.add_argument('--output',type=Path,default=ROOT/'opportunity-service-age-r01.json')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    cells=[args.session/name for name in ('cell-01-cap256-stall8','cell-02-cap256-stall8')]
    result=dict(scope='Two existing stall8 longest gaps for one target, descriptive CPU-only opportunity inspection.',
        source_session=str(args.session),target=TARGET,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cells=[inspect_cell(cell) for cell in cells],
        conclusion='Observed large head repeatedly fails capacity qualification; pure reordering of already-fitting tasks cannot admit it at those sampled states. A capacity-changing action is a distinct hypothesis, but existing traces do not certify any donor or net service benefit. No new experiment was run.')
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(args.output),cells=[dict(cell=cell['cell'],gap_s=cell['longest_gap']['duration_s'],
        capacity_failure_span=cell['capacity_failure_span'],policy_snapshot_summary=cell['policy_snapshot_summary']) for cell in result['cells']])))


if __name__=='__main__':main()

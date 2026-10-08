#!/usr/bin/env python3
"""Narrow within-run chain for the actual one-round yield and its waiting suffix."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def load_repeat():
    path = BASE/'recovery_repeat/analyze.py'
    spec = importlib.util.spec_from_file_location('yield_chain_repeat', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def analyze(session, metrics_path):
    metrics = read(metrics_path); rows = []
    inputs = {str(metrics_path): hashlib.sha256(metrics_path.read_bytes()).hexdigest()}
    repeat = load_repeat()
    for cell in metrics['cells']:
        actions = cell['recovery_start_yield_actions']
        for action in actions['rows']:
            if not action['actual_yield_executed']:
                continue
            directory = session/Path(cell['directory']).parent.name/'output'
            def source(name):
                path = directory/(name+'.json')
                inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                return read(path)
            raw, capacity, finalization = source('raw'), source('capacity-handoff'), source('completion-finalize')
            origin = raw['measurement_origin_perf_counter_s']
            event = action['raw_decision']; stamp = event['host_perf_s']; start = stamp-origin
            mapping = raw['internal_to_source']; deferred = event['deferred_request']
            suffix = event['waiting_order'][event['waiting_order'].index(deferred)+1:]
            long_suffix = next((rid for rid in suffix if mapping.get(rid,'').endswith('-long')), None)
            roles = [('DEFERRED_RECOMPUTE',deferred)]
            roles += [('REGISTERED_LOAD_BENEFICIARY',r['request']) for r in event['beneficiaries']]
            if long_suffix is not None: roles.append(('FIRST_LONG_IN_DEFERRED_SUFFIX',long_suffix))
            paths = []
            summaries = {r['request']:r for r in cell['run_summary']['per_request']}
            for role,rid in roles:
                path = repeat.request_evidence(rid,stamp,raw,cell,cell['recovery_allocations'])
                src = mapping.get(rid)
                request = next((r for r in raw['requests'] if r['request_id']==src),None)
                evidence = [e for e in capacity['events'] if e.get('request')==rid
                            and e.get('begin_host_perf_s',e.get('host_perf_s',0))>=stamp]
                allocations = [e for e in evidence if e['kind']=='allocate']
                schedules = [e for e in evidence if e['kind']=='resumed_schedule']
                jobs = [j for j in cell['jobs'] if j['request']==src and not j['is_store']
                        and (j.get('ack_retired_s') is not None and j['ack_retired_s']>=start)]
                record = next((r for r in event['beneficiaries'] if r['request']==rid),None)
                selected_jobs = [j for j in jobs if record and j['job_id'] in record['job_ids']]
                ack = max((j['ack_retired_s'] for j in selected_jobs),default=None)
                allocation = next((e for e in allocations if e['success'] and not e['arguments']['delay_cache_blocks']),None)
                path.update(role=role, first_allocation_record=next(iter(allocations),None),
                    first_resumed_schedule_record=next(iter(schedules),None),
                    load_jobs_after_or_spanning_decision=jobs, recorded_beneficiary_jobs=selected_jobs,
                    last_recorded_load_ack_s=ack,
                    ack_to_successful_non_async_allocation_s=allocation['begin_host_perf_s']-origin-ack
                        if allocation is not None and ack is not None else None,
                    completion_s=request.get('completion_s') if request else None,
                    whole_request_recovery_summary=summaries.get(src))
                paths.append(path)
            stop = max((p['next_output_after_decision_s'] for p in paths
                        if p['next_output_after_decision_s'] is not None),default=start)
            outputs = defaultdict(list)
            for output in raw['output_events']:
                if start <= output['received_s'] <= stop:
                    outputs[output['engine_call_index']].append(output)
            tracked = {mapping[rid] for _,rid in roles}
            phases = {e['iteration']:e for e in finalization['events']}
            if len(phases) != raw['engine_call_count'] or sorted(phases) != list(range(1,raw['engine_call_count']+1)):
                raise RuntimeError('Cannot establish zero-based engine call / one-based model iteration correspondence')
            calls = []
            for index, observations in sorted(outputs.items()):
                phase = phases[index+1]
                if phase['bookkeeping_return_perf_s']-origin > min(o['received_s'] for o in observations):
                    raise RuntimeError('Matched model bookkeeping is after client output')
                calls.append(dict(engine_call_index=index, model_iteration=index+1,
                    phase_host_s={key:value-origin for key,value in phase.items() if key.endswith('_perf_s')},
                    first_client_receive_s=min(o['received_s'] for o in observations),
                    last_client_receive_s=max(o['received_s'] for o in observations),
                    observed_output_requests=len({o['request_id'] for o in observations}),
                    observed_output_tokens=sum(o['chunk_size'] for o in observations),
                    tracked_outputs=[{k:o[k] for k in ('request_id','received_s','cumulative_tokens','chunk_size','finished')} for o in observations if o['request_id'] in tracked]))
            allocations = [e for e in capacity['events'] if e['kind']=='allocate'
                           and stamp <= e['begin_host_perf_s'] <= origin+stop]
            schedules = [e for e in capacity['events'] if e['kind']=='resumed_schedule'
                         and stamp <= e['host_perf_s'] <= origin+stop]
            reconstruction = dict(status='UNAVAILABLE', rows=[])
            if long_suffix is not None:
                long_path = next(p for p in paths if p['internal_request']==long_suffix)
                plan = long_path['first_resumed_schedule_record']
                if plan is not None:
                    admission = next((i for i in sorted(outputs) if phases[i+1]['enter_perf_s'] >= plan['host_perf_s']),None)
                    long_src = mapping[long_suffix]
                    if admission is not None and long_src not in {o['request_id'] for o in outputs[admission]}:
                        remaining = plan['after']['num_tokens']-plan['scheduled_tokens']
                        decoders = {o['request_id'] for o in outputs[admission] if not o['finished']}
                        valid = True
                        for i in sorted(index for index in outputs if index>admission):
                            new_tokens = min(remaining,1024-len(decoders))
                            remaining -= new_tokens
                            expected = decoders | ({long_src} if remaining==0 else set())
                            actual = {o['request_id'] for o in outputs[i]}
                            matched = expected==actual and all(o['chunk_size']==1 for o in outputs[i])
                            valid &= matched
                            reconstruction['rows'].append(dict(engine_call_index=i,model_iteration=i+1,
                                preceding_decoder_count=len(decoders),reconstructed_long_tokens=new_tokens,
                                reconstructed_long_remaining=remaining,
                                reconstructed_budget_after_running=1024-len(decoders)-new_tokens,
                                exact_output_request_set_matches=matched))
                            decoders -= {o['request_id'] for o in outputs[i] if o['finished']}
                            if remaining==0: break
                        reconstruction.update(status='SOURCE_CONSTRAINED_RECONSTRUCTION' if valid else 'UNVERIFIED',
                            request=long_src,initial_remaining_tokens=plan['after']['num_tokens']-plan['scheduled_tokens'],
                            note='Pinned running-first scheduler with quantum1024; infer decoder cohort from the admission-call outputs, remove observed finishes, and require exact following output ID sets. Per-step budget itself was not logged.')
            rows.append(dict(directory=str(directory),decision_s=start,raw_decision=event,
                next_schedule_entry_s=action['next_schedule_entry_s'],
                next_entry_note='Native schedule entry, not proof its waiting gate opened.',
                requests=paths, calls_until_last_tracked_first_output=calls,
                running_recompute_reconstruction=reconstruction,
                recovery_allocations_in_window=allocations, resumed_schedule_records_in_window=schedules,
                native_preemptions_in_window=[e for e in raw['preemption_events']
                    if start <= e['method_entered_s'] <= stop and e.get('original_preemption_returned') is True]))
    return dict(scope='CPU diagnosis of actual yield paths only; no matched-state cross-run counterfactual.',
        timing='All absolute host fields are perf_counter. Derived *_s use the original measurement origin. LOAD host completion, scheduler ACK, schedule plan, and client output remain distinct.',
        limits='No direct per-round waiting head, token-budget or running-cohort snapshot unless present in the action. Output counts alone do not establish token-budget exhaustion or GPU completion.',
        phase_join='request_measurement engine_call_index=call_count-1; completion_finalize iteration increments before each execute. Require one model iteration per engine call and match index+1.',
        action_count=len(rows),actions=rows,input_sha256=inputs,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--metrics',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result=analyze(args.session,args.metrics)
    with args.output.open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(args.output),actual_actions=result['action_count'])))


if __name__=='__main__': main()

"""Balanced natural-request timing: native_full minus selected, both pairs retained."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from statistics import mean

from saved_kv_analysis_base import performance

NAMES = ('block0-selected','block0-native_full','block1-native_full','block1-selected')
METRICS = ('full_wall_s','request_rate_s','output_token_rate_s','mean_completion_s',
    'mean_ttft_s','max_engine_return_gap_s','total_output_tokens',
    'post_request_drain_s','wall_plus_drain_s','output_rate_with_drain_s')


def sparse_segments(raw):
    rows = {r['request_id']:r for r in raw['requests']}
    groups, segments = {}, []
    for event in raw.get('preemption_events', []):
        if event['original_preemption_returned'] and event['engine_call_index'] < raw['engine_return_count']:
            groups.setdefault(event['request_id'], []).append(event)
    for rid, events in groups.items():
        row = rows[rid]
        for i,event in enumerate(events):
            nxt = events[i+1] if i+1 < len(events) else None
            start = event['last_returned_output_count']
            end = nxt['last_returned_output_count'] if nxt else len(row['token_times_s'])
            if not 0 <= start <= end <= len(row['token_times_s']):
                raise ValueError('Sparse preemption/output count alignment failed')
            times = row['token_times_s'][start:end]
            first = times[0] if times else None
            last = event['last_new_output_s']
            segments.append(dict(request_id=rid, preemption_index=i,
                preempt_entered_s=event['method_entered_s'], preempt_returned_s=event['method_returned_s'],
                native_vs_returned_output_count=event['native_output_count_before']-start,
                last_new_output_before_preempt_s=last, first_new_output_after_preempt_s=first,
                last_to_first_new_output_gap_s=first-last if first is not None and last is not None else None,
                returned_new_outputs=end-start,
                end='repreempted' if nxt else ('completed' if row['status']=='completed' else 'unfinished'),
                next_preempt_s=nxt['method_entered_s'] if nxt else None,
                completion_s=row.get('completion_s') if nxt is None else None))
    return dict(segments=segments, counts=dict(
        preemption_attempts=raw.get('preemption_attempt_count'),
        successful_preempt_methods=raw.get('actual_preemption_count'),
        successful_preempts_in_completed_calls=len(segments),
        zero_new_outputs_then_repreempted=sum(s['end']=='repreempted' and s['returned_new_outputs']==0 for s in segments),
        one_two_new_outputs_then_repreempted=sum(s['end']=='repreempted' and 1<=s['returned_new_outputs']<=2 for s in segments),
        longer_service_then_repreempted=sum(s['end']=='repreempted' and s['returned_new_outputs']>2 for s in segments),
        completed_after_preemption=sum(s['end']=='completed' for s in segments)),
        semantics='Actual returned outputs between successful preemptions. A1-2output segment proves brief delivered service; zero outputs does not prove costly recovery executed. No load-start/recompute/cause inference.')


def read_cell(folder):
    result = dict(cell=folder.name,status='UNRUN',comparable=False,errors=[],sources={})
    data = {}
    for name in ('status','raw','config','selective-store','post-request-drain','timing'):
        path = folder/(name+'.json')
        if path.exists():
            result['sources'][name] = dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            data[name] = json.loads(path.read_text())
    result['status'] = data.get('status',{}).get('status','UNRUN')
    result['config'] = data.get('config')
    result['host_snapshots'] = {p.name:json.loads(p.read_text()) for p in sorted(folder.glob('host-*.json'))}
    result['timing'] = data.get('timing')
    if 'raw' not in data:
        return result
    try:
        raw, config, policy = data['raw'],data['config'],data['selective-store']
        perf = performance(raw)
        if (config['requests']!=64 or perf['request_count']!=64 or config['ignore_eos'] is not False
                or config['min_tokens']!=0 or config['output_tokens']!=1024
                or config['measurement_mode']!='performance_sparse_preemptions'
                or raw['diagnostics']!='SPARSE_PREEMPTION_EVENTS'
                or policy['diagnostic'] is not False or policy['eligibility_snapshots'] or policy['gate_observations']
                or policy['store_scope']!=config['store_scope']
                or policy['native_calc_overridden']!=(config['store_scope']=='selected')):
            raise ValueError('Frozen workload/scope/measurement contract mismatch')
        if any(e['event'] in ('native_full_metadata','store_delta','metadata') for e in policy['events']):
            raise ValueError('Detailed job observer leaked into the timing arm')
        drain = data.get('post-request-drain',{}).get('seconds')
        wall_drained = perf['full_wall_s']+drain if drain is not None else None
        perf.update(post_request_drain_s=drain,wall_plus_drain_s=wall_drained,
            output_rate_with_drain_s=perf['total_output_tokens']/wall_drained if wall_drained else None,
            mean_output_tokens=mean(r['outputs'] for r in perf['requests']),
            min_output_tokens=min(r['outputs'] for r in perf['requests']),
            max_output_tokens=max(r['outputs'] for r in perf['requests']))
        result.update(requests=perf,sparse_recovery=sparse_segments(raw),
            applied_rotations=policy['applied_rotations'],policy_events=policy['events'],
            comparable=result['status']=='COMPLETE' and perf['complete'] and drain is not None)
        if result['comparable'] and raw['actual_preemption_count']<policy['applied_rotations']:
            raise ValueError('Sparse successful preempt count is below applied rotations')
    except Exception as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
        result['comparable']=False
    return result


def normalize_config(config):
    value=deepcopy(config)
    value.pop('store_scope')
    return value


def analyze(root):
    cells, pairs = {}, []
    for name in NAMES:
        try:
            cells[name]=read_cell(root/name)
        except Exception as exc:
            cells[name]=dict(status='ANALYSIS_FAILED',comparable=False,errors=[repr(exc)])
    for block in (0,1):
        a,b=(cells[f'block{block}-{scope}'] for scope in ('selected','native_full'))
        pair=dict(block=block,status='NOT_COMPARABLE')
        if a['comparable'] and b['comparable']:
            identity=lambda c:sorted((r['request_id'],r['document_id'],r['prompt_sha256'],r['arrival_s'],r['max_output']) for r in c['requests']['requests'])
            if normalize_config(a['config'])!=normalize_config(b['config']) or identity(a)!=identity(b):
                pair['reason']='Non-scope config or request/input/arrival/cap mismatch'
            else:
                x,y=a['requests'],b['requests']
                delta=lambda k:None if x[k] is None or y[k] is None else y[k]-x[k]
                pair.update(status='COMPLETE',full_minus_selected={k:delta(k) for k in METRICS},
                    full_relative_percent={k:100*(y[k]/x[k]-1) if x[k] and y[k] is not None else None for k in METRICS},
                    stop_counts=dict(selected=x['actual_stop_counts'],native_full=y['actual_stop_counts']))
                reference={r['request_id']:r for r in x['requests']}
                pair['request_deltas']=[]
                for row in y['requests']:
                    old=reference[row['request_id']]
                    diff=lambda k:row[k]-old[k] if row[k] is not None and old[k] is not None else None
                    pair['request_deltas'].append(dict(request_id=row['request_id'],
                        output_identical=row['output_sha256']==old['output_sha256'],
                        outputs_full_minus_selected=row['outputs']-old['outputs'],
                        completion_full_minus_selected_s=diff('completion_latency_s'),
                        gap_full_minus_selected_s=diff('max_engine_return_gap_s'),
                        ttft_full_minus_selected_s=diff('ttft_s'),
                        stop_selected=old['stop_reason'],stop_full=row['stop_reason']))
        pairs.append(pair)
    return dict(cells=cells,performance_comparisons=pairs,
        primary='Maximum and per-request engine-return generation gaps',
        necessary_costs=['Output/request throughput','Mean completion','TTFT','Completed requests/output quantity and stop distribution','Post-request native drain'],
        semantics=[
            'Four contemporaneous selected/full/full/selected cells; retain both pairs separately without choosing the better repeat.',
            'All64requests must complete for pair comparison. Natural output amounts may differ and are reported; latency differences are not equal-work speedup.',
            'Both arms use the same sparse preemption observer; its cost and all native policy/save/load costs remain measured.',
            'Detailed diagnostics and transfer observers are off. Job counts, recomputation and exact recovery starts are not measured here; qualification is reused from D/E.',
            'wall_plus_drain adds non-overlapping capture wall and native post-request drain only; intermediate host snapshots/serialization are excluded.',
            '0/1-token requests retain undefined gaps; tokens returned together have unresolved internal gaps, without interpolation.',
            'The1-2output short-service count is retained from the existing diagnosis; no new pause/SLO threshold is selected.',
            'Current EOS stop distribution is observed outcome, not an online feature. No actual future EOS or request-identity predictor is used.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with args.output.open('x') as handle:
        json.dump(analyze(args.results),handle,indent=2,ensure_ascii=False)
        handle.write('\n')

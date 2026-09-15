"""Independent natural-input cadence holdout; six lightweight cells and fixed cost budget."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from statistics import mean

from saved_kv_analysis_base import performance

NAMES = ('block0-native_full_native','block0-current','block0-eager','block1-eager','block1-current','block1-native_full_native')
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
        native = config['variant']=='native_full_native'
        expected_scope='native_full' if native else 'selected'
        if (config['requests']!=128 or perf['request_count']!=128 or config['ignore_eos'] is not False
                or config['min_tokens']!=0 or config['output_tokens']!=1024
                or config['measurement_mode']!='performance_sparse_preemptions'
                or raw['diagnostics']!='SPARSE_PREEMPTION_EVENTS'
                or policy['diagnostic'] is not False or policy['eligibility_snapshots'] or policy['gate_observations']
                or policy['store_scope']!=expected_scope or config['store_scope']!=expected_scope
                or policy['native_calc_overridden']!=(not native)
                or config['variant']!=folder.name.split('-',1)[1]
                or config['global_cooldown_steps']!={'current':20,'eager':0,'native_full_native':None}[config['variant']]
                or (not native and config['rotation_config']['min_steps_between_swaps']!=config['global_cooldown_steps'])
                or (native and (config['rotation_config'] is not None or policy['status']!='NOT_APPLICABLE'
                    or policy['applied_rotations'] is not None or policy['scheduler_schedule_overridden'] is not False))
                or policy['rotation_config']!=config['rotation_config']):
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
            applied_rotations=policy['applied_rotations'],policy_events=None if native else policy['events'],
            rotation_status='NOT_APPLICABLE' if native else 'MEASURED',
            comparable=result['status']=='COMPLETE' and perf['complete'] and drain is not None)
        if result['comparable'] and not native and raw['actual_preemption_count']<policy['applied_rotations']:
            raise ValueError('Sparse successful preempt count is below applied rotations')
    except Exception as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
        result['comparable']=False
    return result




def normalize_config(config):
    value=deepcopy(config)
    value.pop('variant')
    value.pop('global_cooldown_steps')
    value['rotation_config'].pop('min_steps_between_swaps')
    return value



def normalize_system_config(config):
    value=deepcopy(config)
    for key in ('variant','global_cooldown_steps','rotation_config','rotation_victim_order','store_scope','population_mode'):
        value.pop(key)
    return value


def compare(a,b,block,baseline,arm,system_reference=False):
    pair=dict(block=block,baseline=baseline,arm=arm,status='NOT_COMPARABLE',
        comparison_kind='SYSTEM_REFERENCE_NOT_SINGLE_FACTOR' if system_reference else 'COOLDOWN_ONLY')
    if not (a['comparable'] and b['comparable']):
        return pair
    identity=lambda c:sorted((r['request_id'],r['document_id'],r['prompt_sha256'],r['arrival_s'],r['max_output']) for r in c['requests']['requests'])
    normalize=normalize_system_config if system_reference else normalize_config
    if normalize(a['config'])!=normalize(b['config']) or identity(a)!=identity(b):
        pair['reason']='Non-treatment config or request/input/arrival/cap mismatch'
        return pair
    x,y=a['requests'],b['requests']
    delta=lambda k:None if x[k] is None or y[k] is None else y[k]-x[k]
    prefix='native_reference' if system_reference else 'eager'
    pair.update(status='COMPLETE',**{
        f'{prefix}_minus_{baseline}':{k:delta(k) for k in METRICS},
        f'{prefix}_relative_percent':{k:100*(y[k]/x[k]-1) if x[k] and y[k] is not None else None for k in METRICS}},
        stop_counts={baseline:x['actual_stop_counts'],arm:y['actual_stop_counts']})
    reference={r['request_id']:r for r in x['requests']}
    pair['request_deltas']=[]
    for row in y['requests']:
        old=reference[row['request_id']]
        diff=lambda k:row[k]-old[k] if row[k] is not None and old[k] is not None else None
        pair['request_deltas'].append(dict(request_id=row['request_id'],
            output_identical=row['output_sha256']==old['output_sha256'],
            outputs_arm_minus_baseline=row['outputs']-old['outputs'],
            completion_arm_minus_baseline_s=diff('completion_latency_s'),
            gap_arm_minus_baseline_s=diff('max_engine_return_gap_s'),
            ttft_arm_minus_baseline_s=diff('ttft_s'),
            stop_baseline=old['stop_reason'],stop_arm=row['stop_reason']))
    return pair


def transfer_budget(pairs):
    """Research cost budget chosen after G, frozen before any H outcomes."""
    checks=[]
    for pair in pairs:
        result=dict(block=pair['block'],status='NOT_ASSESSABLE',budget_pass=None,
            max_gap_decreases=None,tradeoff_pass=None)
        if pair['status']=='COMPLETE':
            value=pair['eager_relative_percent']
            rate,completion,gap=(value[k] for k in
                ('output_token_rate_s','mean_completion_s','max_engine_return_gap_s'))
            if all(v is not None for v in (rate,completion,gap)):
                result.update(status='ASSESSED',output_rate_cost_pass=rate>=-3.,
                    mean_completion_cost_pass=completion<=5.,
                    budget_pass=rate>=-3. and completion<=5.,max_gap_decreases=gap<0.,
                    tradeoff_pass=rate>=-3. and completion<=5. and gap<0.)
        checks.append(result)
    passed=sum(c['tradeoff_pass'] is True for c in checks)
    assessable=len(checks)==2 and all(c['status']=='ASSESSED' for c in checks)
    status=('NOT_ASSESSABLE' if not assessable else 'STABLE_TRADEOFF_CONFIRMED_IN_H' if passed==2
        else 'MIXED_UNCONFIRMED' if passed==1 else 'NOT_CONFIRMED_IN_H')
    return dict(status=status,pair_checks=checks,
        predeclared_limits=dict(max_output_token_rate_loss_percent=3.,max_mean_completion_increase_percent=5.),
        semantics='Both eager/current pairs must retain output tokens/s within-3%, mean completion within+5%, and lower maximum gap. G selected the mechanism and budgets; H verifies new inputs. These are research cost allowances, not SLOs, noise bounds or statistical significance. Preserve all request distributions and native system comparisons; no-action remains a legitimate domain result.')


def analyze(root):
    cells, pairs, references = {}, [], []
    for name in NAMES:
        try:
            cells[name]=read_cell(root/name)
        except Exception as exc:
            cells[name]=dict(status='ANALYSIS_FAILED',comparable=False,errors=[repr(exc)])
    for block in (0,1):
        current,eager,native=(cells[f'block{block}-{v}'] for v in ('current','eager','native_full_native'))
        pairs.append(compare(current,eager,block,'current','eager'))
        for variant,cell in (('current',current),('eager',eager)):
            references.append(compare(cell,native,block,variant,'native_full_native',True))
    return dict(qualification_reused=json.loads((Path(__file__).parent/'qualification_reuse.json').read_text()),
        transfer_budget=transfer_budget(pairs),cells=cells,
        performance_comparisons=pairs,system_reference_comparisons=references,
        primary='Maximum and per-request engine-return generation gaps',
        necessary_costs=['Output/request throughput','Mean completion','TTFT','Completed requests/output quantity and stop distribution','Post-request native drain','Host allocation/valid boundary state/lifecycle HWM'],
        semantics=[
            'Order: native/current/eager/eager/current/native; G qualification reused, no H diagnostic. Keep both pairs and every system-reference contrast.',
            'Current/eager use selected saving and differ only in global cooldown20/0. Native reference omits the rotation adapter and retains full native saving; its contrast is not cooldown attribution.',
            'All128requests must complete for comparison. Natural output amounts and EOS can differ; latency differences are not equal-work speedup.',
            'G qualified actual native store/load, productive return and sub20commit action. H reuses this path; no new diagnostic or internal-start measurement.',
            'Every timing cell has the same sparse preemption observer and no detailed eligibility or transfer observers. Job counts, recomputation and exact recovery starts are NOT_MEASURED.',
            'Native-reference rotation fields are NOT_APPLICABLE, not measured zeros; actual native preemptions remain sparsely observed.',
            'wall_plus_drain adds non-overlapping capture wall and post-request drain only; intermediate host snapshots/serialization are excluded.',
            '0/1-token requests retain undefined gaps; multi-token chunks are not interpolated. Short1-2output segments show brief delivery, not removable recovery cost.',
            'EOS termination after another request preempts does not prove recovery-time EOS; preserve per-request preemption/terminal scope.',
            'No actual future EOS, request identity rule or observed-outcome step choice enters the online policy.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.results)
    with args.output.open('x') as handle:
        json.dump(result,handle,indent=2,ensure_ascii=False)
        handle.write('\n')

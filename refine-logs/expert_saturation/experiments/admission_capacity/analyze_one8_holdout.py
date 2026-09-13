"""Full-request comparisons for fixed one8 versus declared simple strategies."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from analyze_document_state_abba import map_counters, preaction_calls
from analyze_next_chunk_branch import decisions as one8_decisions, jsha
from analyze_prefill_midpoint import prestate
from analyze_static_calibration import metric_row, static_decisions
from analyze_wisp_injection import hardware_summary, summarize, resources


def decisions(raw, strategy):
    if strategy == 'one8':
        return one8_decisions(raw, 8)
    if strategy.startswith('static'):
        return static_decisions(raw, int(strategy[6:]))
    issues = []
    for call in raw['steps'][4:]:
        d, rows = call.get('phase_prefill', {}), call['scheduled']
        ready = sorted(rid for rid in call['before']['running'] if
            any(t < call['start_s'] for t in raw['requests'][rid]['token_received_s']) and
            raw['requests'][rid]['completion_s'] >= call['start_s'])
        ok = d.get('mode') == 'phase8' and d.get('enabled') is True and d.get('status') == 'applied' and d.get('chosen_threshold') == (8 if ready else 0) and sorted(d.get('ready_decode_ids', [])) == ready and d.get('actual_scheduled') == rows
        for rid in ready:
            selected = [r for r in rows if r['request_id'] == rid]
            ok &= len(selected) == 1 and selected[0]['tokens'] == selected[0]['decode_tokens'] == 1 and selected[0]['prefill_tokens'] == 0
        if not ok:
            issues.append(call['index'])
    return dict(valid=not issues, issue_calls=issues)


def analyze(base):
    config = json.loads((base/'config.json').read_text())
    execution = json.loads((base/'execution.json').read_text())
    strategies = ['static8','static16','static32','phase8','one8']
    rng, expected = random.Random(config['design']['order_seed']), []
    for r in (0,1):
        block = [(doc,strategy) for doc in ('c10','c11') for strategy in strategies]
        rng.shuffle(block)
        expected += [f'r{r}-{doc}-{strategy}' for doc,strategy in block]
    out = dict(status=execution['status'], config=config, config_sha256=hashlib.sha256((base/'config.json').read_bytes()).hexdigest(),
        protocol_valid=[c['cell'] for c in config['cells']] == expected == config['design']['cell_order'],
        execution={k:execution.get(k) for k in ('status','pid','start_unix','end_unix','document_prebranch_sha256')},
        cells={}, consistency={}, one8_minus_baseline={}, observed_point_dominance={}, repeat_drift={},
        boundary='Two preselected new incoming documents, fixed old pair, n=2 per strategy/document. Point comparisons are descriptive, not significance or a noise bound. No online selector or global static optimum.')
    rows = {r['cell']:r for r in execution.get('cells',[])}
    invalid = not out['protocol_valid']
    for cell in config['cells']:
        name, strategy, doc = cell['cell'], cell['strategy'], cell['document_set']
        entry = out['cells'][name] = dict(cell_config=cell, run_status='UNRUN')
        path = base/name/'result.json'
        if not path.exists():
            continue
        data = path.read_bytes();raw = json.loads(data)
        entry.update(raw_sha256=hashlib.sha256(data).hexdigest(),run_status=raw.get('status'),error=raw.get('error'))
        if raw.get('status') != 'COMPLETED':
            invalid = True
            continue
        s = summarize(raw, rows[name]);d = decisions(raw,strategy);counter = map_counters(raw)
        measured,validation = raw.get('map_measurement',{}),raw.get('map_validation',{})
        checks = dict(complete=s['complete_expected_requests'],identity=s['identity_valid'] and s['preaction_old_prefix_valid'],
            timing=all(r['timing_valid'] for r in s['requests'].values()),first_shape=s['first_shape_valid'],
            preaction_calls=preaction_calls(raw)['valid'],no_preemption=not s['scheduler_violations'],decisions=d['valid'],
            common_preaction=prestate(raw) == config['preaction_sha256'],workload=raw['workload_sha256'] == config['workload_sources'][doc]['sha256'],
            first_counter=counter['valid'],map=validation.get('status') == 'PASS' and validation.get('layers') == 16 and measured.get('mode') == 'batched' and measured['totals']['scalar_device_assignments'] == 0)
        branch = raw.get('next_chunk_branch');metrics=metric_row(raw,s)
        hashes=dict(prestate=prestate(raw),workload=raw['workload_sha256'],resources=jsha(resources(raw)),output={rid:jsha(row['output_token_ids']) for rid,row in raw['requests'].items()})
        metrics['extra_capture_s'] = branch['capture_duration_s'] if branch else 0
        if strategy == 'one8':
            bc=map_counters(dict(first_action_map_counters=branch['map_counters'],action=dict(engine_call=5)))
            hashes['prebranch']=jsha(branch['before_state'])
            checks.update(one8_prebranch=hashes['prebranch'] == branch['prestate_sha256'] == execution['document_prebranch_sha256'][doc],
                one8_causal=branch['engine_call'] == 5 and branch['selected_chunk'] == branch['chosen_threshold'] == 8 and
                    branch['decision_boundary'] == 'before_native_schedule_and_KV_allocation' and
                    raw['steps'][4]['return_s'] <= branch['capture_start_s'] <= branch['capture_end_s'] <= raw['steps'][5]['start_s'],
                one8_counter=bc['valid'] and counter['after'] == bc['before'])
            entry['branch_counters']=bc
        else:
            checks['baseline_no_one8_patch'] = branch is None
        entry.update(analysis_status='VALID' if all(checks.values()) else 'INVALID_COMPARISON',checks=checks,metrics=metrics,hashes=hashes,
            first_counters=counter,preaction_calls=preaction_calls(raw),hardware=hardware_summary(base/'hardware.jsonl',raw,config['resources']['cpu_affinity']))
        invalid |= not all(checks.values())
    complete=execution['status'] == 'COMPLETED' and len([c for c in out['cells'].values() if 'metrics' in c]) == 20
    if complete:
        reuse=[r['reuse_record'] for r in execution['cells']]
        keys={(r['pid'],r['llm_id'],r['scheduler_id'],r['pool_id'],r['queue_id'],r['worker']['allocation_sha256']) for r in reuse}
        out['consistency']=dict(all_completed=True,same_resources=len({c['hashes']['resources'] for c in out['cells'].values()}) == 1,
            same_engine=len(keys) == 1 and len(reuse) == 20 and [r['invocation'] for r in reuse] == list(range(20)) and all(r['status'] == 'READY' and r['native_schedule_restored'] and r['threshold'] == 32 and r['worker']['allocation_unchanged'] and r['worker']['pager_empty'] for r in reuse),
            map_selftest=execution['map_selftest']['status'] == 'PASS' and execution['map_selftest']['cases'] == 5)
        invalid |= not all(out['consistency'].values())
        for doc in ('c10','c11'):
            for r in (0,1):
                one = out['cells'][f'r{r}-{doc}-one8']['metrics'];dominated=[]
                for strategy in strategies[:-1]:
                    ref=out['cells'][f'r{r}-{doc}-{strategy}']['metrics']
                    out['one8_minus_baseline'][f'r{r}-{doc}-vs-{strategy}']={k:one[k]-v for k,v in ref.items()}
                    axes=('wall_s','old_max_itl_s','new_ttft_s')
                    if all(ref[k] <= one[k] for k in axes) and any(ref[k] < one[k] for k in axes):
                        dominated.append(strategy)
                out['observed_point_dominance'][f'r{r}-{doc}']=dict(one8_dominated_by=dominated,axes=axes,statistical_inference=False)
            for strategy in strategies:
                a,b=[out['cells'][f'r{r}-{doc}-{strategy}']['metrics'] for r in (0,1)]
                out['repeat_drift'][f'{doc}-{strategy}']={k:b[k]-v for k,v in a.items()}
    out['status']=('INVALID_COMPARISON' if invalid else 'MEASUREMENT_ONLY') if complete else execution['status']
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',required=True,type=Path);p.add_argument('--out',required=True,type=Path);args=p.parse_args()
    encoded=json.dumps(analyze(args.results),indent=2,allow_nan=False)+'\n'
    if any(f'"{k}"' in encoded for k in ('prompt_token_ids','output_token_ids','token_received_s','requests','steps','before_state')):
        raise RuntimeError('unsafe raw export field')
    with args.out.open('x') as f:f.write(encoded)


if __name__ == '__main__':
    main()

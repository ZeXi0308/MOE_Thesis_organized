"""Recompute one-next-chunk action contrasts; export derived numbers/hashes only."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from analyze_document_state_abba import map_counters, preaction_calls
from analyze_prefill_midpoint import prestate
from analyze_static_calibration import metric_row
from analyze_wisp_injection import hardware_summary, resources, summarize


def jsha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def decisions(raw, action):
    issues = []
    for call in raw['steps'][4:]:
        d, rows = call.get('phase_prefill', {}), call['scheduled']
        cap = action if call['index'] == 5 else 16
        ready = sorted(rid for rid in call['before']['running'] if
            any(t < call['start_s'] for t in raw['requests'][rid]['token_received_s']) and
            raw['requests'][rid]['completion_s'] >= call['start_s'])
        ok = (d.get('enabled') is True and d.get('mode') == 'static16' and
              d.get('chosen_threshold') == cap and d.get('status') == 'applied' and
              d.get('actual_scheduled') == rows and sorted(d.get('ready_decode_ids', [])) == ready)
        for rid in ready:
            selected = [r for r in rows if r['request_id'] == rid]
            ok &= len(selected) == 1 and selected[0]['tokens'] == selected[0]['decode_tokens'] == 1 and selected[0]['prefill_tokens'] == 0
        if not ok:
            issues.append(call['index'])
    return dict(valid=not issues, issue_calls=issues)


def analyze(base):
    config = json.loads((base / 'config.json').read_text())
    execution = json.loads((base / 'execution.json').read_text())
    rng, expected = random.Random(config['design']['order_seed']), []
    for repeat in range(2):
        block = [(doc, cap) for doc in ('old', 'new') for cap in (8, 16, 32)]
        rng.shuffle(block)
        expected += [f'r{repeat}-{doc}-next{cap}' for doc, cap in block]
    declared = [c['cell'] for c in config['cells']]
    out = dict(status=execution['status'], config=config, config_sha256=hashlib.sha256((base/'config.json').read_bytes()).hexdigest(),
        protocol_valid=declared == expected == config['design']['cell_order'],
        execution={k:execution.get(k) for k in ('status','pid','start_unix','end_unix','document_prebranch_sha256')},
        cells={}, contrasts={}, interactions={}, repeat_drift={}, consistency={},
        boundary='Two randomized six-cell blocks; no statistical significance, population noise bound, KV tensor equality or SLO claim.')
    execution_rows = {r['cell']:r for r in execution.get('cells', [])}
    invalid = not out['protocol_valid']
    for cell in config['cells']:
        name, doc, cap = cell['cell'], cell['document_set'], cell['next_chunk']
        path = base/name/'result.json'
        entry = out['cells'][name] = dict(cell_config=cell, run_status='UNRUN')
        if not path.exists():
            continue
        data = path.read_bytes(); raw = json.loads(data)
        entry.update(raw_sha256=hashlib.sha256(data).hexdigest(), run_status=raw.get('status'))
        if raw.get('status') != 'COMPLETED':
            invalid = True
            continue
        summary = summarize(raw, execution_rows[name])
        branch, call = raw['next_chunk_branch'], raw['steps'][5]
        counter = map_counters(dict(first_action_map_counters=branch['map_counters'], action=dict(engine_call=5)))
        first_counter = map_counters(raw)
        d, state_hash = decisions(raw, cap), jsha(branch['before_state'])
        curve = config['frozen_prediction']['curve_s']
        expected_prediction = dict(static=curve[str(cap)], recent=branch['first_mixed_step_s']*curve[str(cap)]/curve['16'])
        first = raw['steps'][4]
        validation, measured = raw.get('map_validation', {}), raw.get('map_measurement', {})
        checks = dict(complete=summary['complete_expected_requests'], identity=summary['identity_valid'] and summary['preaction_old_prefix_valid'],
            first_shape=summary['first_shape_valid'] and summary['first_action_tokens'] == 18,
            timing=all(r['timing_valid'] for r in summary['requests'].values()),
            no_preemption=not summary['scheduler_violations'], preaction_calls=preaction_calls(raw)['valid'], decisions=d['valid'],
            branch_index=branch['engine_call'] == 5 and call['index'] == 5,
            actual_branch_shape=call['total_scheduled_tokens'] == cap+2 and sum(r['prefill_tokens'] for r in call['scheduled']) == cap,
            common_preaction=prestate(raw) == config['preaction_sha256'],
            prebranch=state_hash == branch['prestate_sha256'] == execution['document_prebranch_sha256'][doc],
            visible_feedback=branch['visible_feedback']['engine_call'] == 4 and branch['visible_feedback']['map_counters'] == raw['first_action_map_counters'] and
                branch['visible_feedback']['available_after_s'] == first['return_s'] and
                branch['first_mixed_step_s'] == first['return_s']-first['start_s'] and
                first['return_s'] <= branch['capture_start_s'] <= branch['capture_end_s'] <= call['start_s'],
            before_native=branch['decision_boundary'] == 'before_native_schedule_and_KV_allocation' and branch['chosen_threshold'] == cap,
            predictions=branch['predictions_s'] == expected_prediction and branch['frozen_curve_s'] == curve,
            workload=raw['workload_sha256'] == config['workload_sources'][doc]['sha256'],
            map=validation.get('status') == 'PASS' and validation.get('layers') == 16 and measured.get('mode') == 'batched' and measured['totals']['scalar_device_assignments'] == 0,
            counters=counter['valid'] and first_counter['valid'])
        metrics = metric_row(raw, summary)
        metrics.update(branch_step_s=call['return_s']-call['start_s'],
            branch_old_itl_s=max(raw['requests'][rid]['token_received_s'][5]-raw['requests'][rid]['token_received_s'][4] for rid in summary['old_ids']),
            capture_s=branch['capture_duration_s'], postbranch_wall_s=raw['wall_s']-branch['capture_start_s'])
        entry.update(analysis_status='VALID' if all(checks.values()) else 'INVALID_COMPARISON', checks=checks, metrics=metrics,
            hashes=dict(prestate=prestate(raw), prebranch=state_hash, workload=raw['workload_sha256'], resources=jsha(resources(raw)),
                output={rid:jsha(r['output_token_ids']) for rid,r in raw['requests'].items()}),
            visible_first_step_s=branch['first_mixed_step_s'], predictions_s=branch['predictions_s'],
            prediction_error_s={k:metrics['branch_step_s']-v for k,v in branch['predictions_s'].items()},
            first_counters=first_counter, branch_counters=counter, preaction_calls=preaction_calls(raw),
            hardware=hardware_summary(base/'hardware.jsonl', raw, config['resources']['cpu_affinity']))
        invalid |= not all(checks.values())
    complete = execution['status'] == 'COMPLETED' and len([c for c in out['cells'].values() if 'metrics' in c]) == 12
    if complete:
        reuse = [r['reuse_record'] for r in execution['cells']]
        keys = {(r['pid'],r['llm_id'],r['scheduler_id'],r['pool_id'],r['queue_id'],r['worker']['allocation_sha256']) for r in reuse}
        out['consistency'] = dict(all_completed=True, same_resources=len({r['hashes']['resources'] for r in out['cells'].values()}) == 1,
            same_engine=len(keys) == 1 and len(reuse) == 12 and [r['invocation'] for r in reuse] == list(range(12)) and
                all(r['status'] == 'READY' and r['native_schedule_restored'] and r['threshold'] == 32 and r['worker']['allocation_unchanged'] and r['worker']['pager_empty'] for r in reuse),
            map_selftest=execution['map_selftest']['status'] == 'PASS' and execution['map_selftest']['cases'] == 5,
            same_doc_branch=all(len({c['hashes']['prebranch'] for c in out['cells'].values() if c['cell_config']['document_set'] == doc}) == 1 for doc in ('old','new')))
        invalid |= not all(out['consistency'].values())
        for repeat in range(2):
            for doc in ('old','new'):
                baseline = out['cells'][f'r{repeat}-{doc}-next16']['metrics']
                for cap in (8,32):
                    metrics = out['cells'][f'r{repeat}-{doc}-next{cap}']['metrics']
                    out['contrasts'][f'r{repeat}-{doc}-next{cap}_minus16'] = {k:metrics[k]-v for k,v in baseline.items()}
            for cap in (8,32):
                a,b = [out['contrasts'][f'r{repeat}-{doc}-next{cap}_minus16'] for doc in ('old','new')]
                out['interactions'][f'r{repeat}-next{cap}'] = {k:b[k]-v for k,v in a.items()}
        for doc in ('old','new'):
            for cap in (8,16,32):
                a,b = [out['cells'][f'r{r}-{doc}-next{cap}']['metrics'] for r in range(2)]
                out['repeat_drift'][f'{doc}-next{cap}'] = {k:b[k]-v for k,v in a.items()}
    out['status'] = ('INVALID_COMPARISON' if invalid else 'MEASUREMENT_ONLY') if complete else execution['status']
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--results',required=True,type=Path); p.add_argument('--out',required=True,type=Path); args=p.parse_args()
    encoded=json.dumps(analyze(args.results),indent=2,allow_nan=False)+'\n'
    if any(f'"{k}"' in encoded for k in ('prompt_token_ids','output_token_ids','token_received_s','requests','steps','before_state')):
        raise RuntimeError('unsafe raw field in derived export')
    with args.out.open('x') as f: f.write(encoded)


if __name__ == '__main__':
    main()

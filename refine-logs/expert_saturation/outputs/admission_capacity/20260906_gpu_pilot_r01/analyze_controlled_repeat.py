"""Compare both retained ABBA runs; reconstruct only observable action frontiers."""
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parents[4] / 'refine-logs/expert_saturation/experiments/admission_capacity'))
from analyze_capacity import analyze


def read(path):
    return json.loads(path.read_text())


def frontier(raw):
    action = raw['actions'][1]
    at = action['applied_s']  # Loop boundary before prefill; requested_s may be mid-call.
    active, waiting = [], []
    for r in raw['requests']:
        if r['admission_s'] is not None and r['admission_s'] < at < r['completion_s']:
            emitted = [token for token, t in zip(r['output_token_ids'], r['token_times_s']) if t < at]
            if not emitted:
                raise ValueError('frontier intersects a prefill; no discrete active-state reconstruction')
            active.append(dict(request_id=r['request_id'], decode_step=len(emitted)-1,
                kv_length=r['prompt_tokens']+len(emitted)-1, output_prefix=emitted))
        elif r['arrival_s'] <= at and (r['admission_s'] is None or r['admission_s'] >= at):
            waiting.append(r['request_id'])
    if len(active) != action['actual_active']:
        raise ValueError('reconstructed active count differs from action record')
    return dict(active=active, waiting=waiting,
                completed_pressure=action['recent_pressure'])


def main():
    result = dict(evidence_type='CUSTOM_CONTINUOUS_RUNTIME', campaigns={},
        claim_ceiling='One fixed pulse and its repeat, not a U/C-selected policy or matched KV tensor state.',
        sample_unit='Same 16 requests reused; decode steps and repeats are not new documents.',
        all_runs_retained=True)
    workloads, sources = [], []
    for campaign in ('actions', 'actions-repeat'):
        lookup, pairs = {}, []
        for trial in ('a0_hold', 'b0_pulse', 'b1_pulse', 'a1_hold'):
            root = BASE / campaign / trial
            config, env = read(root/'config.json'), read(root/'environment.json')
            workloads.append(read(root/'workload.json'))
            sources.append(env['source_sha256'])
            checked = analyze(root)
            if not checked['complete_scan']:
                raise ValueError(f'{campaign}/{trial}: incomplete or invalid scan')
            for cell in checked['cells']:
                raw = read(root/f"cell-{cell['cell']:03d}.json")
                p, m = cell['plan'], cell['metrics']
                recorded = read(root/f"metrics-{cell['cell']:03d}.json")['metrics']
                if m != recorded:
                    raise ValueError('request metrics changed on raw recomputation')
                for s in raw['steps']:
                    active = {r['request_id'] for r in raw['requests']
                              if r['admission_s'] <= s['start_s'] < r['completion_s']}
                    if active != set(s['request_ids']):
                        raise ValueError('decode did not execute all active requests')
                lookup[trial, p['regime'], p['telemetry']] = (raw, m)
        for repeat in (0, 1):
            for regime in ('steady', 'bursty'):
                for on in (False, True):
                    a, am = lookup[f'a{repeat}_hold', regime, on]
                    b, bm = lookup[f'b{repeat}_pulse', regime, on]
                    af, bf = frontier(a), frontier(b)
                    pairs.append(dict(repeat=repeat, regime=regime, telemetry=on,
                        hold_goodput_rps=am['goodput_rps'], pulse_goodput_rps=bm['goodput_rps'],
                        relative_goodput_pct=100*(bm['goodput_rps']/am['goodput_rps']-1),
                        hold_passes=am['n_slo_pass'], pulse_passes=bm['n_slo_pass'],
                        discrete_frontier_equal=af['active']==bf['active'] and af['waiting']==bf['waiting'],
                        pressure_equal=af['completed_pressure']==bf['completed_pressure'] if on else None,
                        hold_frontier=af, pulse_frontier=bf,
                        pulse_actuation=[dict(target=x['target'], actual_at_application=x['actual_active'],
                            application_lag_s=x['applied_s']-x['requested_s'],
                            after_application_delay_s=x['effect_delay_s'],
                            total_delay_s=None if x['effective_s'] is None else x['effective_s']-x['requested_s'])
                            for x in b['actions'][1:]]))
        result['campaigns'][campaign] = dict(pairs=pairs, full_active_decode_verified=True)
    result['identical_workload_across_both_runs'] = all(x==workloads[0] for x in workloads)
    result['identical_executed_source_across_both_runs'] = all(x==sources[0] for x in sources)
    if not result['identical_workload_across_both_runs'] or not result['identical_executed_source_across_both_runs']:
        raise ValueError('controlled-repeat workload or executed source drift')
    result['alignment_limit'] = 'Discrete identity equality does not establish equal waiting ages, recent latency, KV tensors, RNG or hardware state.'
    with (BASE/'controlled_repeat.json').open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    for name, data in result['campaigns'].items():
        for p in data['pairs']:
            print(name, p['repeat'], p['regime'], p['telemetry'],
                  round(p['relative_goodput_pct'], 3), p['hold_passes'], p['pulse_passes'], p['discrete_frontier_equal'])


if __name__ == '__main__':
    main()

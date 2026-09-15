"""Descriptive ABBA action accounting; no offline policy or exact-KV-pair claim."""
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[4]
sys.path.insert(0, str(ROOT / 'refine-logs/expert_saturation/experiments/admission_capacity'))
from analyze_capacity import analyze


def signature(raw, before):
    # Observed execution identity only: equality does not prove bit-exact KV.
    steps = [s for s in raw['steps'] if s['token_emission_s'] < before]
    tokens = {r['request_id']: [token for token, t in zip(r['output_token_ids'], r['token_times_s'])
              if t < before] for r in raw['requests']}
    return dict(steps=[(s['request_ids'], s['decode_steps']) for s in steps], tokens=tokens)


def main():
    cells, results = {}, []
    for trial in ('a0_hold', 'b0_pulse', 'b1_pulse', 'a1_hold'):
        root = BASE / 'actions' / trial
        result = analyze(root)
        (root / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        for c in result['cells']:
            if c['state'] != 'COMPLETE':
                raise ValueError(f'{trial} cell {c["cell"]}: {c["state"]} {c["issues"]}')
            raw = json.loads((root / f'cell-{c["cell"]:03d}.json').read_text())
            for s in raw['steps']:
                active = {r['request_id'] for r in raw['requests']
                          if r['admission_s'] <= s['start_s'] < r['completion_s']}
                if active != set(s['request_ids']) or len(active) != s['ordinary']['actual_active']:
                    raise ValueError('recorded decode omitted or invented active requests')
            events = [dict(target=a['target'], actual_at_application=a['actual_active'],
                requested_s=a['requested_s'], application_lag_s=a['applied_s'] - a['requested_s'],
                effect_delay_s=a['effect_delay_s'], effective_s=a['effective_s'],
                total_delay_from_request_s=None if a['effective_s'] is None else a['effective_s'] - a['requested_s'],
                recent_pressure=a['recent_pressure']) for a in raw['actions'][1:]]
            m = c['metrics']
            row = dict(trial=trial, cell=c['cell'], regime=c['plan']['regime'], telemetry=c['plan']['telemetry'],
                goodput_rps=m['goodput_rps'], n_slo_pass=m['n_slo_pass'],
                ttft_p50_s=m['latency_s']['ttft']['p50'], tpot_p50_s=m['latency_s']['tpot']['p50'],
                actions=events, full_active_set_decode_verified=True)
            results.append(row)
            cells[(trial, c['plan']['regime'], c['plan']['telemetry'])] = (row, raw)
    pairs = []
    for repeat in (0, 1):
        for regime in ('steady', 'bursty'):
            for telemetry in (False, True):
                a, ar = cells[(f'a{repeat}_hold', regime, telemetry)]
                b, br = cells[(f'b{repeat}_pulse', regime, telemetry)]
                pairs.append(dict(repeat=repeat, regime=regime, telemetry=telemetry,
                    pulse_minus_hold_goodput_rps=b['goodput_rps'] - a['goodput_rps'],
                    pulse_relative_goodput=b['goodput_rps'] / a['goodput_rps'] - 1,
                    pulse_minus_hold_slo_count=b['n_slo_pass'] - a['n_slo_pass'],
                    observed_prefix_identity_equal=signature(ar, ar['actions'][1]['applied_s']) ==
                        signature(br, br['actions'][1]['applied_s']),
                    completed_pressure_before_action_equal=ar['actions'][1]['recent_pressure'] ==
                        br['actions'][1]['recent_pressure']))
    out = dict(verdict='EXPLORATORY_ABBA_END_TO_END_ACTION_COMPARISON',
        arm_order=['a0_hold', 'b0_pulse', 'b1_pulse', 'a1_hold'], cells=results, pairs=pairs,
        claim_ceiling='One fixed pulse versus hold6 in one reused cohort; no U/C-selected action, exact-KV pair, oracle, native-serving or general mechanism claim')
    (BASE / 'action_measurements.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(json.dumps(pairs, indent=2))


if __name__ == '__main__':
    main()

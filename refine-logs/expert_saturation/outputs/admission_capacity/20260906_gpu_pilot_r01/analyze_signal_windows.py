"""Describe existing action signal windows; never simulate a policy or future."""
import json
from pathlib import Path
import statistics

BASE = Path(__file__).resolve().parent


def mean(values):
    return statistics.mean(values)


def availability(step):
    # Reconstruct the recorded now() after telemetry, just before steps.append.
    # This omits tiny dictionary/append overhead; it is not a device timestamp.
    return step['start_s'] + step['iteration_s'] - step['ordinary']['prefill_s']


def point(step):
    if step is None:
        return None
    ordinary = step['ordinary']
    return dict(step=step['step'], start_s=step['start_s'], available_proxy_s=availability(step),
        width=ordinary['decode_requests'], request_ids=step['request_ids'],
        decode_steps=step['decode_steps'], kv_lengths=ordinary['kv_lengths'],
        waiting=ordinary['waiting_requests'], prefill=ordinary['prefill_requests_this_iteration'],
        U=mean(p['U'] for p in step['pressure']), C=mean(p['C'] for p in step['pressure']))


def window(raw, action):
    steps = [s for s in raw['steps'] if availability(s) <= action['applied_s']][-4:]
    if len(steps) != 4:
        raise ValueError('this diagnostic expects the four completed recorded steps')
    u = mean(p['U'] for s in steps for p in s['pressure'])
    c = mean(p['C'] for s in steps for p in s['pressure'])
    if abs(u - action['recent_pressure']['U_mean']) > 1e-9 or abs(c - action['recent_pressure']['C_mean']) > 1e-9:
        raise ValueError('reconstructed completed window differs from action record')
    return dict(steps=[point(s) for s in steps], U=u, C=c,
        widths=[s['ordinary']['decode_requests'] for s in steps],
        mean_age_at_application_s=mean(action['applied_s'] - availability(s) for s in steps),
        latest_age_at_application_s=action['applied_s'] - availability(steps[-1]))


def post_step(raw, when, next_action):
    return next((s for s in raw['steps'] if when <= s['start_s'] < next_action), None)


def main():
    output = BASE / 'signal_windows.json'
    if output.exists():
        raise FileExistsError('retain the previous analysis; use an addendum for corrections')
    events = []
    for family in ('actions', 'actions-repeat'):
        for pair in (0, 1):
            pulse_dir, hold_dir = BASE / family / f'b{pair}_pulse', BASE / family / f'a{pair}_hold'
            holds = {}
            for path in sorted(hold_dir.glob('cell-*.json')):
                raw = json.loads(path.read_text())
                if raw['plan']['telemetry']:
                    holds[raw['plan']['regime']] = (path, raw)
            for path in sorted(pulse_dir.glob('cell-*.json')):
                raw = json.loads(path.read_text())
                if not raw['plan']['telemetry']:
                    continue
                hold_path, hold = holds[raw['plan']['regime']]
                if raw['status'] != 'COMPLETE' or hold['status'] != 'COMPLETE':
                    raise ValueError('existing pulse/hold capture is incomplete')
                for index, action in enumerate(raw['actions'][1:], 1):
                    if action['effective_s'] is None or action['superseded_s'] is not None:
                        raise ValueError('action has no unambiguous observed effect')
                    before = window(raw, action)
                    next_action = raw['actions'][index + 1]['applied_s'] if index + 1 < len(raw['actions']) else float('inf')
                    post = point(post_step(raw, action['effective_s'], next_action))
                    control_action = hold['actions'][index]
                    control_before = window(hold, control_action)
                    control_horizon = control_action['applied_s'] + action['effect_delay_s']
                    control_next = hold['actions'][index + 1]['applied_s'] if index + 1 < len(hold['actions']) else float('inf')
                    control_post = point(post_step(hold, control_horizon, control_next))
                    latest, control_latest = before['steps'][-1], control_before['steps'][-1]
                    same_members = control_post is not None and control_latest['request_ids'] == control_post['request_ids']
                    events.append(dict(family=family, pair=pair, regime=raw['plan']['regime'],
                        pulse_raw=str(path.relative_to(BASE)), hold_raw=str(hold_path.relative_to(BASE)),
                        direction=action['direction'], target=action['target'], actual_at_application=action['actual_active'],
                        applied_s=action['applied_s'], effective_s=action['effective_s'], effect_delay_s=action['effect_delay_s'],
                        before=before, after_effect=post,
                        window_mean_age_at_effect_s=before['mean_age_at_application_s'] + action['effect_delay_s'],
                        rolling_minus_latest_U=before['U'] - latest['U'], rolling_minus_latest_C=before['C'] - latest['C'],
                        control_before=control_before, control_after_lag=control_post,
                        control_same_request_members=same_members,
                        control_latest_to_post_U=None if control_post is None else control_post['U'] - control_latest['U'],
                        control_latest_to_post_C=None if control_post is None else control_post['C'] - control_latest['C']))
    summary = {}
    for direction in ('up', 'down'):
        subset = [r for r in events if r['direction'] == direction]
        stable = [r for r in subset if r['control_same_request_members']]
        def span(field, rows=subset):
            values = [r[field] for r in rows if r[field] is not None]
            return [min(values), max(values)] if values else None
        summary[direction] = dict(events=len(subset), mixed_width_windows=sum(len(set(r['before']['widths'])) > 1 for r in subset),
            window_width_sequences=sorted({tuple(r['before']['widths']) for r in subset}),
            effect_delay_s=span('effect_delay_s'), window_mean_age_at_effect_s=span('window_mean_age_at_effect_s'),
            rolling_minus_latest_U=span('rolling_minus_latest_U'), rolling_minus_latest_C=span('rolling_minus_latest_C'),
            control_same_request_members=len(stable), control_same_members_delta_U=span('control_latest_to_post_U', stable),
            control_same_members_delta_C=span('control_latest_to_post_C', stable))
    result = dict(status='DESCRIPTIVE_EXISTING_GPU_DATA_ANALYSIS', events=events, summary=summary,
        limits=['Eight telemetry-ON pulse episodes plus eight actual hold episodes reuse 16 texts; events are not independent cohorts.',
            'Pressure availability is reconstructed after telemetry with tiny unrecorded append overhead; host emission is not availability.',
            'Control horizon borrows the measured pulse delay only for retrospective timing diagnosis, never online action selection.',
            'The control future was actually executed by hold6; it is not a replayed counterfactual pulse future.',
            'Matching request members does not match KV lengths, token contents, recent latency or exact pre-action states.',
            'Rolling versus latest and pre/post differences do not measure prediction value, decay rate, hardware traffic or causal residual.',
            'No new GPU run, predictor, dynamic Oracle, native result or method GO is produced.'])
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

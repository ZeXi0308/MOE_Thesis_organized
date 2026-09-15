"""Observed time partition at first selective action; no drift-corrected effect."""
import json
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / 'outputs/admission_capacity/20260914_selective_store_once_r01'


def analyze(base=BASE):
    arms = {}
    schedules = []
    arrivals = []
    for label in ('selective-off', 'selective-on'):
        root = base / 'readback/results' / label
        raw = json.loads((root / 'raw.json').read_text())
        action = json.loads((root / 'selective-store.json').read_text())
        selected = next(e for e in action['events'] if e['event'] == 'select')
        cutoff = selected['step']
        assert cutoff == 328
        call = next(c for c in raw['engine_steps'] if c['scheduler_step_start'] == cutoff)
        t = call['start_s']
        requests = raw['requests']
        assert len(requests) == 32 and all(q['status'] == 'completed' and q['completion_s'] > t for q in requests)
        arrivals.append({q['request_id']: q['arrival_s'] for q in requests})
        schedules.append([[(q['request_id'], q['scheduled_start_computed'], q['scheduled_tokens'], q['output_tokens_before'])
                           for q in s['scheduled']] for s in raw['scheduler_steps'][:cutoff]])
        mean = statistics.mean(q['completion_s'] - q['arrival_s'] for q in requests)
        pre = statistics.mean(t - q['arrival_s'] for q in requests)
        post = statistics.mean(q['completion_s'] - t for q in requests)
        assert abs(mean - pre - post) < 1e-10
        arms[label] = dict(cutoff_step=cutoff, cutoff_start_s=t, mean_completion_s=mean,
                           mean_arrival_to_cutoff_s=pre, mean_cutoff_to_completion_s=post,
                           wall_s=max(q['completion_s'] for q in requests) - min(q['arrival_s'] for q in requests),
                           cutoff_to_last_completion_s=max(q['completion_s'] for q in requests)-t)
    assert schedules[0] == schedules[1] and arrivals[0] == arrivals[1]
    delta = {k: arms['selective-on'][k] - arms['selective-off'][k]
             for k in arms['selective-on'] if k.endswith('_s')}
    assert abs(delta['mean_completion_s'] - delta['mean_arrival_to_cutoff_s'] - delta['mean_cutoff_to_completion_s']) < 1e-10
    result = dict(status='OBSERVED_TIME_PARTITION_ONLY', arms=arms, on_minus_off_s=delta,
                  pre_action_schedule_equal=True, arrivals_equal=True,
                  scope='One run per arm. Cutoff is engine call 328 start, before victim selection/store registration. '
                        'The partition is an identity, not a drift correction or causal estimate. '
                        'Post-cutoff trajectories and timing may both differ; no stable benefit or harm established.')
    (base / 'time_partition.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    print(json.dumps(analyze(), indent=2))

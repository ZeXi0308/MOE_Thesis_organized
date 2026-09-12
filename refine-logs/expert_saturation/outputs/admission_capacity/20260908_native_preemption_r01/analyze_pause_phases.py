"""Locate victim pauses relative to recomputation calls; not isolated GPU cost."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = []
for repeat in [0, 1]:
    raw = json.loads((ROOT / 'gpu_results' / f'repeat{repeat}-native32' / 'raw.json').read_text())
    for event in raw['preemption_events']:
        rid = event['victim_internal_request_id']
        source = raw['internal_to_source'][rid]
        request = next(r for r in raw['requests'] if r['request_id'] == source)
        times = request['token_times_s']
        index = max(range(len(times)-1), key=lambda i: times[i+1]-times[i])
        start, end = times[index:index+2]
        steps = [s['step'] for s in raw['scheduler_steps'] if any(
            r['internal_request_id'] == rid and r['recompute_tokens'] > 0 for r in s['scheduled'])]
        calls = [c for c in raw['engine_steps'] if any(
            c['scheduler_step_start'] <= s < c['scheduler_step_end'] for s in steps)]
        assert all(c['completed'] for c in calls)
        first = min(c['start_s'] for c in calls)
        last = max(c['returned_s'] for c in calls)
        assert start <= first <= last <= end
        rows.append(dict(repeat=repeat, request_id=source, gap_start_s=start, gap_end_s=end,
            gap_s=end-start, first_recompute_call_start_s=first, last_recompute_call_return_s=last,
            before_first_recompute_call_s=first-start, recompute_calls_span_s=last-first,
            after_last_recompute_call_s=end-last, recompute_step_ids=steps))
result = dict(scope='Victim host timeline. Recompute-call span includes concurrent requests and '
    'instrumentation; it is not isolated recomputation GPU time. These spans must not be added '
    'again to request latency or summed across overlapping victims.', rows=rows)
with (ROOT / 'analysis-pool-corrected' / 'pause_phases.json').open('x') as output:
    json.dump(result, output, indent=2)
    output.write('\n')
print(json.dumps(rows, indent=2))

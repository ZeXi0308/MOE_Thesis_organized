"""Describe actual post-schedule states; do not infer counterfactual savings."""
import argparse
from collections import Counter
import json
from pathlib import Path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
summary = json.loads((root/'analysis/summary.json').read_text())
records = []
for cell in summary['cells']:
    if cell['plan']['cohort'] != 'mixed' or cell['plan']['prefill_policy'] != 'per_request512':
        continue
    path = root/'readback/results'/cell['block']/Path(cell['source']).name
    raw = json.loads(path.read_text())
    counts, events = Counter(), []
    for event in cell['prefill_share_events']:
        if not event['threshold_limited_allocations']:
            continue
        step = raw['scheduler_steps'][event['step']]
        if any(x['preceding_threshold_limited_ids'] for x in event['short_first_admitted_after_long_prefill']):
            kind = 'clipped_then_short_first_admitted'
        elif event['prefill_requests'] != 1:
            kind = 'other_prefill_coparticipant'
        elif step['waiting_requests'] == 0:
            kind = 'no_waiter_after_schedule'
        elif step['actual_active'] >= step['target_cap']:
            kind = 'running_slots_full_with_waiters'
        else:
            kind = 'spare_slot_with_waiters_unexplained'
        counts[kind] += 1
        events.append(dict(step=step['step'], kind=kind, actual_active=step['actual_active'],
            waiting_after=step['waiting_requests'], tokens=step['total_scheduled_tokens'],
            unused_budget=event['unused_budget']))
    records.append(dict(block=cell['block'], counts=dict(counts), events=events))
result = dict(evidence_type='OBSERVED_POLICY_STATE_PARTITION_NOT_CAUSAL_SAVINGS',
    scope='Formal mixed per_request512 only; post-schedule state labels are descriptive', records=records)
with args.output.open('x') as handle:
    json.dump(result, handle, indent=2)
    handle.write('\n')

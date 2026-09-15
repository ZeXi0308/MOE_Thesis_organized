"""All eligible recorded prepares, with current state only for candidate scores."""
import argparse
import json
from collections import Counter
from pathlib import Path
from joint_growth_model import blocks, growth, swap_envelope

parser = argparse.ArgumentParser()
parser.add_argument('--selective', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
data = json.loads(args.selective.read_text())
snapshots = {s['step']: s for s in data['eligibility_snapshots']}
rows, excluded = [], []
for event in data['events']:
    if event.get('event') != 'prepare':
        continue
    s = snapshots[event['step']]
    reqs = s['requests']
    if any(reqs[r]['output'] <= 0 or reqs[r]['computed'] != reqs[r]['prompt']+reqs[r]['output']-1
           for r in s['running_ids']):
        excluded.append({'step': s['step'], 'reason': 'not all running pure decode'})
        continue
    cfg = s['tracker']['config']
    candidates = []
    for r in s['running_ids']:
        q = reqs[r]
        progress = min(1.0, max(0, q['computed']-q['prompt'])/max(1, q['max_output']))
        if (progress >= cfg['protect_progress_fraction']
            or s['tracker']['absence_count'].get(r, 0) >= cfg['max_absences_per_request']
            or s['step']-s['tracker']['resident_since'].get(r, -10**9) < cfg['min_residency_steps']):
            continue
        now = swap_envelope(s, r, event['target'], peer_tokens=0, target_outputs=1)
        if now['margin_blocks'] < 0:
            continue
        later = swap_envelope(s, r, event['target'])
        candidates.append({'victim': r, 'immediate': now, 'one_block_cycle': later})
    if not candidates:
        excluded.append({'step': s['step'], 'reason': 'no eligible immediate candidate'})
        continue
    best_release = max(c['immediate']['released_blocks'] for c in candidates)
    release_ties = [c for c in candidates if c['immediate']['released_blocks'] == best_release]
    best_horizon = max(c['one_block_cycle']['margin_blocks'] for c in candidates)
    actual = next((c for c in candidates if c['victim'] == event['victim']), None)
    rows.append(dict(step=s['step'], target=event['target'], actual_victim=event['victim'],
        actual_in_candidates=actual is not None, candidates=candidates,
        actual_margin=None if actual is None else actual['one_block_cycle']['margin_blocks'],
        best_horizon_margin=best_horizon,
        release_rule_regret_blocks=best_horizon-max(c['one_block_cycle']['margin_blocks'] for c in release_ties),
        all_release_ties_optimal=all(c['one_block_cycle']['margin_blocks']==best_horizon for c in release_ties)))

# Exact one-step conditional demand for the two published source-localized events.
# The event identity is used only to select evaluation rows, never as model input.
event_checks = []
for step in (903, 1161):
    s = snapshots[step]
    demands = {rid:growth(s['requests'][rid]['computed'], s['requests'][rid]['held_blocks'], 1)
               for rid in s['running_ids']}
    event_checks.append(dict(step=step, free=s['free_blocks'], one_step_joint_demand=sum(demands.values()),
                             deficit=max(0, sum(demands.values())-s['free_blocks'])))

result = dict(evidence='STRUCTURAL_CONDITIONAL_CURRENT_STATE_CANDIDATE_SCREEN',
    source=str(args.selective), question='Does one-block joint growth add ranking beyond maximum released blocks?',
    horizon={'peer_compute_tokens':16,'target_new_outputs':16,'reason':'one KV block, not fitted to outcomes'},
    restrictions=['exclusive blocks/no sharing; actual held blocks assumed releasable',
                  'hypothetical immediate swap, no prepare growth or async load delay',
                  'no new arrivals, no future EOS or completion credits',
                  'all peers advance together; no wall-time or request-utility ranking'],
    prepare_count=sum(e.get('event')=='prepare' for e in data['events']),
    evaluated=len(rows), excluded=excluded, event_checks=event_checks,
    release_rule_positive_regret_count=sum(r['release_rule_regret_blocks']>0 for r in rows),
    all_release_ties_optimal_count=sum(r['all_release_ties_optimal'] for r in rows),
    best_horizon_margin_signs=dict(Counter('negative' if r['best_horizon_margin']<0 else 'nonnegative' for r in rows)),
    rows=rows)
with args.output.open('x') as f:
    json.dump(result, f, indent=2); f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

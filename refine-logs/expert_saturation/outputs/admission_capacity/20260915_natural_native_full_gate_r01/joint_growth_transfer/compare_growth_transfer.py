"""Join existing prepare-state capacity scores to later observed service.

Outcomes never feed candidate scores. This checks a proposed interpretation of
the score, not the conditional block identity itself or a causal treatment.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--screen', type=Path, required=True)
parser.add_argument('--analysis', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
screen = json.loads(args.screen.read_text())
analysis = json.loads(args.analysis.read_text())
segments = analysis['diagnostic']['segments']
rows, unmatched = [], []
for p in screen['rows']:
    matches = [s for s in segments if s['internal_request_id'] == p['target']
               and s['preempt_step'] < p['step']
               and s['executed_steps'] and p['step'] <= s['executed_steps'][0]['step']]
    # A segment's next preemption closes its identity, avoiding later recoveries.
    matches = [s for s in matches if s.get('next_preempt_step') is None
               or p['step'] < s['next_preempt_step']]
    if len(matches) != 1 or not p['actual_in_candidates']:
        unmatched.append({'step': p['step'], 'matches': len(matches), 'actual_eligible': p['actual_in_candidates']})
        continue
    s = matches[0]
    actual = next(c for c in p['candidates'] if c['victim'] == p['actual_victim'])
    rows.append(dict(prepare_step=p['step'], target=p['target'],
        immediate_margin=actual['immediate']['margin_blocks'],
        growth_margin=actual['one_block_cycle']['margin_blocks'],
        best_growth_margin=p['best_horizon_margin'],
        useful_outputs=s['useful_outputs'], end=s['end'],
        short_repreempted=s['end']=='repreempted' and s['useful_outputs']<=2))

def contingency(field):
    counts = Counter()
    for r in rows:
        counts['negative' if r[field]<0 else 'nonnegative', 'short' if r['short_repreempted'] else 'not_short'] += 1
    return {a+'_'+b: counts[a,b] for a in ['negative','nonnegative'] for b in ['short','not_short']}

out = dict(source_screen=str(args.screen), source_analysis=str(args.analysis),
    joined=len(rows), unmatched=unmatched, rows=rows,
    immediate=contingency('immediate_margin'), growth=contingency('growth_margin'),
    semantics=['same observed policy, no execution of alternative victims',
        'negative conditional joint-capacity margin is not a prediction of <=2 target outputs',
        'this tests that tempting interpretation without training or threshold selection',
        'episodes differ in instrumentation and arrival-time state; no causal save-mode attribution',
        'no new independent samples, inferential accuracy or seconds-level performance claim'])
with args.output.open('x') as f:
    json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))

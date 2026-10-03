"""Keep all recorded prepares; observed next state only evaluates predictions."""
import argparse
import json
from pathlib import Path
from prepare_margin import predict
from joint_growth_model import blocks

p=argparse.ArgumentParser()
p.add_argument('--selective',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
d=json.loads(a.selective.read_text())
snaps={s['step']:s for s in d['eligibility_snapshots']}
rows=[]
for e in d['events']:
    if e.get('event')!='prepare':continue
    step=e['step'];s=snaps[step];v=e['victim'];t=e['target']
    pred=predict(s,v,t)
    row=dict(step=step,victim=v,target=t,prediction=pred)
    n=snaps.get(step+1)
    if n is None or v not in n['requests'] or t not in n['requests']:
        row['evaluation']='missing_next_state';rows.append(row);continue
    target=n['requests'][t]
    need=max(0,blocks(target['prompt']+target['output'])-target['held_blocks'])
    actual=n['free_blocks']+n['requests'][v]['held_blocks']-need
    changed=[r for r in s['running_ids'] if r not in n['requests'] or
             n['requests'][r]['computed']!=s['requests'][r]['computed']+1]
    row.update(observed_margin=actual,error_blocks=pred['next_margin']-actual,
        evaluation='matched' if pred['next_margin']==actual else 'mismatch',
        retained_running_set=set(s['running_ids'])==set(n['running_ids']),
        non_one_step_requests=changed,
        target_unchanged=s['requests'][t]==n['requests'][t],
        commit=[x.get('reason') for x in d['events'] if x.get('event')=='commit_check' and x['step']==step+1])
    rows.append(row)
out=dict(source=str(a.selective),evidence='ONE_STEP_CURRENT_STATE_PREDICTION_CHECK',rows=rows,
    prepares=len(rows),matches=sum(r['evaluation']=='matched' for r in rows),
    mismatches=sum(r['evaluation']=='mismatch' for r in rows),
    missing=sum(r['evaluation']=='missing_next_state' for r in rows),
    predicted_strict_decrease=sum(r['prediction']['peer_growth']>0 for r in rows),
    feasible_to_infeasible=sum(r['prediction']['immediate_margin']>=0 and r['prediction']['next_margin']<0 for r in rows),
    assumptions='Same running population, each one decode; target unchanged; exclusive blocks; no outside releases/admissions. No future EOS or notifications supplied to predictor.',
    boundary='Actual mandatory preparation step, not an executed optional delay intervention or utility ranking.')
with a.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))

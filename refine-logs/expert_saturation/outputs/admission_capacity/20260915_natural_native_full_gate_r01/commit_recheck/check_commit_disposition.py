"""Evaluate all original commit checks without outcome filtering or rerunning."""
import argparse
import json
from pathlib import Path
from commit_disposition import choose_commit_action
from joint_growth_model import growth, blocks

p=argparse.ArgumentParser()
p.add_argument('--selective',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
d=json.loads(a.selective.read_text());sn={s['step']:s for s in d['eligibility_snapshots']}
rows=[]
for e in d['events']:
    if e.get('event')!='commit_check':continue
    s=sn[e['step']];q=s['requests'].get(e['target'])
    if q is None:
        rows.append(dict(step=e['step'],status='target_missing',original_reason=e['reason']));continue
    need=max(0,blocks(q['prompt']+q['output'])-q['held_blocks'])
    decision=choose_commit_action(e['reason'],s['free_blocks'],need)
    row=dict(step=e['step'],status='evaluated',original_reason=e['reason'],decision=decision,
        free=s['free_blocks'],target_remaining_blocks=need,target=e['target'],victim=e['victim'])
    if decision=='RESUME_WITHOUT_VICTIM':
        peers=s['running_ids'];r=s['requests'];held=r[e['victim']]['held_blocks']
        pure=all(r[x]['output']>0 and r[x]['computed']==r[x]['prompt']+r[x]['output']-1 for x in peers)
        demand=sum(growth(r[x]['computed'],r[x]['held_blocks'],1) for x in peers)
        row.update(retained_victim_blocks=held,running_count=len(peers),all_running_pure_decode=pure,
                   current_peer_one_step_growth=demand,
                   margin_after_current_peer_decode=s['free_blocks']-demand-need,
                   pending_load_count=len(s['skipped_ids']),
                   conditional_all_peers_and_target_fit=pure and s['free_blocks']-demand>=need)
    rows.append(row)
out=dict(source=str(a.selective),scope='CPU decision candidate, no native action executed',
         checks=len(rows),changed=sum(r.get('decision')=='RESUME_WITHOUT_VICTIM' for r in rows),rows=rows,
         boundary='Existing READY checks retained; no future release/route/EOS/cost information. Allocation fit does not guarantee sustained service or net gain.')
with a.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({**{k:v for k,v in out.items() if k!='rows'},'changed_rows':[r for r in rows if r.get('decision')=='RESUME_WITHOUT_VICTIM']},indent=2))

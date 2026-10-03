"""Locate observed FCFS choices; do not simulate alternative completion times."""
from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parent
cells=[]
for path in sorted((root/'readback/results').glob('*/cell-*-raw.json')):
    raw=json.loads(path.read_text());seen=set();choices=[]
    for step in raw['scheduler_steps']:
        new=[x['request_id'] for x in step['scheduled'] if x['request_id'] not in seen]
        waiting=[r for r in raw['requests'] if r['request_id'] not in seen and r['engine_add_return_s']<=step['start_s']]
        waiting.sort(key=lambda r:r['arrival_s'])
        if new and waiting and waiting[0]['prompt_tokens']==2048 and any(r['prompt_tokens']==128 for r in waiting):
            choices.append({'step':step['step'],'start_s':step['start_s'],'newly_scheduled':new,'waiting_before_first_schedule':[{'request_id':r['request_id'],'arrival_s':r['arrival_s'],'engine_add_return_s':r['engine_add_return_s'],'prompt_tokens':r['prompt_tokens']} for r in waiting]})
        seen.update(x['request_id'] for x in step['scheduled'])
    cells.append({'block':path.parent.name,'plan':raw['plan'],'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'n_choice_events':len(choices),'events':choices})
with (root/'analysis/queue_order_opportunity.json').open('x') as f:json.dump({'semantics':'Observed already-submitted and never-scheduled requests at actual first-prefill admission. Prompt length is pre-action information. Event existence does not establish benefit, native reorder support, Oracle headroom, fairness, or counterfactual timing.','cells':cells},f,indent=2)
print(json.dumps([{'block':c['block'],'plan':c['plan'],'choice_events':c['n_choice_events']} for c in cells]))

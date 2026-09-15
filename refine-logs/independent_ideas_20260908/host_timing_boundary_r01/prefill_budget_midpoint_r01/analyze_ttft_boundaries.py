"""Same-host-clock TTFT boundaries, not exclusive request resource use."""
from pathlib import Path
import json,statistics,hashlib
root=Path(__file__).resolve().parent
output=[]
for path in sorted((root/'readback/results').glob('*/cell-*-raw.json')):
    raw=json.loads(path.read_text());first={}
    for step in raw['scheduler_steps']:
        for row in step['scheduled']:first.setdefault(row['request_id'],step['start_s'])
    rows=[]
    for r in raw['requests']:
        a,b,c,d=r['arrival_s'],r['engine_add_return_s'],first[r['request_id']],r['token_times_s'][0]
        assert a<=b<=c<=d
        parts={'arrival_to_add_return_s':b-a,'add_return_to_first_schedule_s':c-b,'first_schedule_to_first_token_s':d-c}
        assert abs(sum(parts.values())-(d-a))<1e-9
        rows.append({'request_id':r['request_id'],'prompt_tokens':r['prompt_tokens'],'ttft_s':d-a,**parts})
    groups={}
    for name,length in [('all',None),('short',128),('long',2048)]:
        rr=[r for r in rows if length is None or r['prompt_tokens']==length]
        groups[name]={k:statistics.mean(r[k] for r in rr) for k in ('ttft_s','arrival_to_add_return_s','add_return_to_first_schedule_s','first_schedule_to_first_token_s')}
    output.append({'block':path.parent.name,'plan':raw['plan'],'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'groups':groups,'requests':rows})
with (root/'analysis/ttft_boundaries.json').open('x') as f:json.dump({'semantics':'All host-clock wall intervals. First schedule uses scheduler wrapper entry stamp, not native queue metric. Waiting interval includes driver scheduling opportunities; no removable-cost or queue-policy counterfactual.', 'cells':output},f,indent=2)
for c in output:print(json.dumps({'block':c['block'],'plan':c['plan'],'all_mean_ms':{k:v*1000 for k,v in c['groups']['all'].items()}}))

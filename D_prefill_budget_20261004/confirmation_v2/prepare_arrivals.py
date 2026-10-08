"""Freeze three independent arrival realizations, without reading any results."""
import copy
import hashlib
import json
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'workload_high.json'
original=json.loads(SOURCE.read_text())
seeds=[2026100401,2026100402,2026100403]
receipt={'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
         'seeds':seeds,
         'arrival_rule':'Keep 24 short arrivals at 0 and 8 at 3. Within each long cohort (43/43/42), independently permute request IDs and draw sorted uniform arrivals in a window of (n-1)*0.01 seconds. Window starts are 0.6/2.5/5 plus independent uniform jitter of +/-0.15/0.25/0.5 seconds. Preserve prompts, output lengths, cohort membership and total counts.',
         'orders':[['fixed384','feedback','fixed304','decode_v2'],
                   ['decode_v2','fixed304','feedback','fixed384'],
                   ['fixed304','fixed384','decode_v2','feedback']],
         'warm_policies':['fixed304','fixed384','decode_v2','feedback'],
         'slo':{'ttft_s':4,'gap_s':0.1,'completion_s':20},
         'blocks':[]}
for i,seed in enumerate(seeds,1):
    rng=random.Random(seed)
    work=copy.deepcopy(original)
    starts=[]
    for lo,hi,base,jitter in [(0,43,.6,.15),(43,86,2.5,.25),(86,128,5.,.5)]:
        group=[r for r in work if r['request_id'].startswith('high-long-') and lo<=int(r['request_id'].rsplit('-',1)[1])<hi]
        rng.shuffle(group)
        start=base+rng.uniform(-jitter,jitter)
        starts.append(start)
        arrivals=sorted(start+rng.uniform(0,(hi-lo-1)*.01) for _ in group)
        for row,arrival in zip(group,arrivals):row['arrival_s']=round(arrival,9)
    path=ROOT/f'arrivals_{i:02d}.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(work,indent=2)+'\n')
    receipt['blocks'].append({'block':i,'seed':seed,'wave_starts':starts,'workload':path.name,
                             'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                             'prompt_tokens':sum(len(r['prompt_token_ids']) for r in work),
                             'output_tokens':sum(r['max_tokens'] for r in work)})
(ROOT/'design.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))

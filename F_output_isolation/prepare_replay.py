"""Freeze two existing, unscaled GPU host-output traces; no model execution."""
import argparse, hashlib, json
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parent
SOURCES={
 'medium':'D_prefill_budget_20261004/fixed-normal-r01/02_fixed2048/raw.json',
 'high':'D_prefill_budget_20261004/high-normal-r01/00_fixed2048/raw.json',
}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path);a=p.parse_args()
 out=ROOT/'inputs';out.mkdir(exist_ok=True)
 manifest={'schema':1,'heavy':'streaming top_logprobs=20','heavy_fraction':.25,
 'cpu_affinity':{'server':[2],'client':[4],'producer':[6]},
 'primary':'light token engine host-ready to complete parsed SSE visible, P99 ms',
 'rates':'Original host receipt times, no scaling; original prompt and generated IDs',
 'logprob_limit':'synthetic top20 values/alternatives; original actual generated token retained; GPU logprob cost NOT validated',
 'cpu_only':True,'sources':{}}
 for name,rel in SOURCES.items():
  p=(a.source_root/Path(rel).relative_to('D_prefill_budget_20261004')) if a.source_root else ROOT.parent/rel
  d=json.loads(p.read_text()); reqs=d['requests']
  heavy=set(sorted((r['request_id'] for r in reqs),key=lambda x:hashlib.sha256(x.encode()).hexdigest())[:len(reqs)//4])
  batches=defaultdict(list); output=[]
  for r in reqs:
   assert r['finished'] and len(r['output_token_ids'])==len(r['token_times_s'])==r['max_tokens']
   output.append({k:r[k] for k in ('request_id','arrival_s','prompt_token_ids','output_token_ids','token_times_s','finish_reason') } | {'heavy':r['request_id'] in heavy})
   for j,(tid,t) in enumerate(zip(r['output_token_ids'],r['token_times_s'])):
    batches[t].append({'request_id':r['request_id'],'token_id':tid,'position':j,'finished':j+1==len(r['output_token_ids'])})
  data={'name':name,'requests':output,'batches':[{'time_s':t,'outputs':v} for t,v in sorted(batches.items())],
  'source':rel,'source_sha256':sha(p),'elapsed_s':d['elapsed_s'],'semantics':'Host engine.step receipt after GPU step, detokenize=False; fixed output lengths, natural KV capacity'}
  dest=out/(name+'.json');dest.write_text(json.dumps(data,separators=(',',':'))+'\n')
  manifest['sources'][name]={'path':rel,'source_sha256':sha(p),'replay_sha256':sha(dest),'requests':len(reqs),'heavy_requests':len(heavy),'tokens':sum(len(r['output_token_ids']) for r in reqs),'batches':len(batches),'max_batch':max(map(len,batches.values())),'last_ready_s':max(batches),'average_tokens_s':sum(len(r['output_token_ids']) for r in reqs)/max(batches)}
 (ROOT/'frozen_workloads.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps(manifest['sources'],indent=2))
if __name__=='__main__':main()

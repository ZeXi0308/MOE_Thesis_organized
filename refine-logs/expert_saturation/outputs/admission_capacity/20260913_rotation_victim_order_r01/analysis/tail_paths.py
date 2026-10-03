import collections,json
from pathlib import Path
base=Path('/Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation/outputs/admission_capacity/20260913_rotation_victim_order_r01/execution/gpu_results')
out=[]
for cohort in ('cohort0','cohort1'):
 for block in (0,1):
  a=json.loads((base/f'{cohort}-block{block}-least_progress/raw.json').read_text()); b=json.loads((base/f'{cohort}-block{block}-most_output/raw.json').read_text())
  rows={}
  for role,r in [('A',a),('B',b)]:
   bytime={e['returned_s']:e for e in r['engine_steps']}
   ranks={x['request_id']:i+1 for i,x in enumerate(sorted(r['requests'],key=lambda x:(x['completion_s'],x['request_id'])))}
   for i,q in enumerate(r['requests']):
    rows.setdefault(q['request_id'],dict(request_id=q['request_id'],arrival_index=i))
    e=bytime[q['completion_s']]
    rows[q['request_id']][role]=dict(completion_s=q['completion_s'],completion_rank=ranks[q['request_id']],completion_step=e['scheduler_step_start'])
  for v in rows.values():v['B_minus_A_completion_ms']=(v['B']['completion_s']-v['A']['completion_s'])*1000
  widths={}
  for role,r in [('A',a),('B',b)]:
   bystep={e['scheduler_step_start']:e for e in r['engine_steps']}
   pure=[s for s in r['scheduler_steps'] if s['recompute_tokens']==0 and sum(x['prefill_tokens'] for x in s['scheduled'])==0]
   wc=collections.Counter(len(s['scheduled']) for s in pure)
   two=[s for s in pure if len(s['scheduled'])==2]
   widths[role]=dict(pure_decode_width_counts=dict(wc),pure_decode_width2_calls=len(two),
     width2_engine_duration_s=sum(bystep[s['step']]['returned_s']-bystep[s['step']]['start_s'] for s in two),
     width2_first_step=two[0]['step'],width2_last_step=two[-1]['step'],
     width2_first_output_counts={x['request_id']:x['output_tokens_before'] for x in two[0]['scheduled']},
     width2_request_sets=dict(collections.Counter('|'.join(sorted(x['request_id'] for x in s['scheduled'])) for s in two)))
  out.append(dict(cohort=cohort,block=block,requests=list(rows.values()),widths=widths))
p=Path('/private/tmp/victim-order-paths/tail_paths.json');p.write_text(json.dumps(out,indent=2)+'\n');print(p)
for q in out:
 print(q['cohort'],q['block'])
 for r in q['requests']:
  if r['arrival_index'] in [0,1,2,3,28,29,30,31]:print(r['arrival_index'],r['request_id'],round(r['B_minus_A_completion_ms'],3),r['A']['completion_rank'],r['B']['completion_rank'],r['A']['completion_step'],r['B']['completion_step'])
 print('width2 initial outputs',[(role, row['width2_first_output_counts']) for role,row in q['widths'].items()])

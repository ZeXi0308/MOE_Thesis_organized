"""Rebuild D input artifacts from archived, already tokenized public task prompts."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'C_research_artifacts/20261001'
qpath=SOURCE/'20261002_c_qmsum_inputs_v1/workload.json'
gpath=SOURCE/'20261001_c_instruct_gsm8k_inputs_v1/workload.json'
q=json.loads(qpath.read_text())['requests']
g=json.loads(gpath.read_text())['requests']
receipts={}
for name,offset in [('dev',0),('holdout',32)]:
    work=[]
    for i in range(12):
        item=g[i%len(g)]
        work.append(dict(request_id=f'{name}-background-{i}',arrival_s=0.0 if i<8 else 3.,
            prompt_token_ids=item['prompt_token_ids'],max_tokens=384,source='GSM8K',source_index=i%len(g)))
    for i in range(24):
        item=q[offset+i]
        work.append(dict(request_id=f'{name}-long-{i}',arrival_s=[.6,2.5,5.0][i//8]+.04*(i%8),
            prompt_token_ids=item['prompt_token_ids'],max_tokens=160,source='QMSum',source_index=offset+i))
    data=json.dumps(work).encode()
    (ROOT/f'workload_{name}.json').write_bytes(data)
    receipts[name]=dict(sha256=hashlib.sha256(data).hexdigest(),requests=len(work),
        prompt_tokens=sum(len(x['prompt_token_ids']) for x in work),output_tokens=sum(x['max_tokens'] for x in work))
(ROOT/'warmup.json').write_bytes((ROOT/'workload_dev.json').read_bytes())
(ROOT/'input_receipt.json').write_text(json.dumps(dict(
    source_sha256={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (qpath,gpath)},
    workloads=receipts,holdout_scope='24 QMSum prompts replaced; 12 short background prompts shared',
    output_contract='Actual autoregressive inference with fixed output lengths, ignore EOS; not a natural ending evaluation',
    warmup='Complete development trace repeated at every action level'),indent=2)+'\n')

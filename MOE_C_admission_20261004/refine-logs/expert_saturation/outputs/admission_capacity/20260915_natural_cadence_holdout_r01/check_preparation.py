"""Only H-specific input, unchanged-path, six-cell and cost-budget CPU checks."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
prior=root.parent/'20260915_natural_recovery_cadence_r01'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
config=read(root/'pkg/inputs/config.json');work=read(root/'pkg/inputs/workload.json')
inventory=read(root/'used_documents.json');used=inventory['train_articles']
assert config['workload_sha256']==hashlib.sha256(json.dumps(work,sort_keys=True).encode()).hexdigest()
assert config['exclusion_inventory_sha256']==sha(root/'used_documents.json')
assert len(used)==224 and len(work['source_requests'])==len(work['actual_prompt_token_ids'])==config['requests']==128
assert work['arrival_traces_s']['steady']==[i*.2 for i in range(128)]
assert abs(config['arrival_span_s']-25.4)<1e-12
assert config['ignore_eos'] is False and config['min_tokens']==0 and config['output_tokens']==1024
seen=set();token_seen=set()
for row,ids in zip(work['source_requests'],work['actual_prompt_token_ids']):
    doc_sha=hashlib.sha256(row['prompt'].encode()).hexdigest()
    tok_sha=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
    assert doc_sha==row['document_sha256']==row['prompt_sha256']
    assert tok_sha==row['prompt_token_ids_sha256']
    assert 256<=len(ids)==row['prompt_token_count']==row['original_document_token_count']<=3072
    assert doc_sha not in inventory['global_document_hashes'] and tok_sha not in inventory['global_prompt_token_hashes']
    assert doc_sha not in seen and tok_sha not in token_seen
    assert all(row['document_id']!=old['document_id'] and max(row['dataset_row_index'],old['dataset_row_index'])>=min(row['dataset_row_end_exclusive'],old['dataset_row_end_exclusive']) for old in used)
    seen.add(doc_sha);token_seen.add(tok_sha)
assert [r['dataset_row_index'] for r in work['source_requests']]==sorted(r['dataset_row_index'] for r in work['source_requests'])
manifest=read(prior/'manifest.json');unchanged=[]
for name,expected in manifest.items():
    if name.startswith('pkg/') and name not in ('pkg/run.sh','pkg/run_recovery_cadence.py') and not name.startswith('pkg/inputs/'):
        assert sha(root/name)==expected
        unchanged.append(name)
assert sha(root/'saved_kv_analysis_base.py')==manifest['saved_kv_analysis_base.py']
for name in ('prepare_inputs.py','analyze_cadence.py','controller.py','pkg/run_recovery_cadence.py'):
    ast.parse((root/name).read_text())
subprocess.run(['bash','-n',str(root/'pkg/run.sh')],check=True)
sys.path.insert(0,str(root))
import analyze_cadence as analyzer
assert len(analyzer.NAMES)==6 and all(n!='diagnostic-eager' for n in analyzer.NAMES)
for n in analyzer.NAMES:
    assert n in (root/'pkg/run.sh').read_text() and n in (root/'controller.py').read_text()
assert '--measurement-mode diagnostic' not in (root/'pkg/run.sh').read_text()
assert "choices=['performance']" in (root/'pkg/run_recovery_cadence.py').read_text()
def pair(block,rate,completion,gap):
    return dict(block=block,status='COMPLETE',eager_relative_percent=dict(
        output_token_rate_s=rate,mean_completion_s=completion,max_engine_return_gap_s=gap))
assert analyzer.transfer_budget([pair(0,-3,5,-1),pair(1,0,0,-1)])['status']=='STABLE_TRADEOFF_CONFIRMED_IN_H'
assert analyzer.transfer_budget([pair(0,-3.01,5,-1),pair(1,0,0,-1)])['status']=='MIXED_UNCONFIRMED'
assert analyzer.transfer_budget([pair(0,0,5.01,-1),pair(1,0,0,0)])['status']=='NOT_CONFIRMED_IN_H'
assert analyzer.transfer_budget([dict(block=0,status='UNRUN'),dict(block=1,status='UNRUN')])['status']=='NOT_ASSESSABLE'
print(json.dumps(dict(status='CPU_CHECKS_PASS_GPU_UNRUN',excluded_train_documents=len(used),
    new_documents=len(seen),measured_request_budget=128*len(analyzer.NAMES),unchanged_G_runtime_payloads=len(unchanged),
    model_eos_observer_checks='Reused G; not rerun',new_checks='H input identity/exclusion/arrival/count; shared bytes; six-cell CLI; budget inclusive boundaries')))

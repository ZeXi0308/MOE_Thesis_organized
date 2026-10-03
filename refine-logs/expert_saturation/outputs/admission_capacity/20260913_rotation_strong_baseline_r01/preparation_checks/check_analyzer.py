#!/usr/bin/env python3
"""CPU fixtures only: no GPU, old raw, or persisted measured result."""
from copy import deepcopy
import hashlib
import argparse
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
E = HERE.parents[3]/'experiments/admission_capacity'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=HERE/'analyzer_cpu_checks.json')
args = parser.parse_args()
assert not args.output.exists(), 'refusing to overwrite checks'
sys.path.insert(0, str(E))
import analyze_rotation_strong_baseline as module
checks = []
sha = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()
workload = dict(source_requests=[], actual_prompt_token_ids=[], arrival_traces_s=dict(steady=[round(i*.05, 9) for i in range(32)]))
for i in range(32):
    text = 'CPU fixture document '+str(i)
    text_sha = hashlib.sha256(text.encode()).hexdigest()
    ids = [i]*3072
    workload['source_requests'].append(dict(document_id=f'fixture-document-{i}', request_id=f'fixture-request-{i}',
        document_sha256=text_sha, prompt_sha256=text_sha, prompt=text, prompt_token_count=3072,
        dataset_row_index=1000+i*10, dataset_row_end_exclusive=1009+i*10,
        prompt_token_ids_sha256=hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()))
    workload['actual_prompt_token_ids'].append(ids)
config = dict(requests=32, prompt_tokens=3072, output_tokens=1024, arrival_gap_s=.05, workload_sha256=sha(workload))
manifest = dict(schema_version=4, experiment_id='20260913_rotation_strong_baseline_r01',
    cohorts=[dict(id='cohort2', input_root='cohorts/cohort2', workload_sha256=sha(workload))], cells=[])
for block, roles in ((0,module.ORDER),(1,tuple(reversed(module.ORDER)))):
    for role in roles:
        manifest['cells'].append(dict(label=f'cohort2-block{block}-{role}', cohort_id='cohort2', block=block, role=role,
            completion_policy='rotate' if role=='most_output' else 'native' if role=='native_aa' else role,
            cap=32, victim_order='most_output' if role=='most_output' else 'least_progress'))

receipt = dict(schema_version=1, status='INPUTS_PREPARED_GPU_UNRUN', cohort_id='cohort2', workload_sha256=sha(workload),
    prior_inputs=[dict(root=f'/cpu-fixture/prior{i}', workload_sha256=sha(i)) for i in range(3)],
    excluded_prior_requests=[dict(document_id=f'fixture-prior-{i}', document_sha256=sha(['document',i]),
        prompt_token_ids_sha256=sha(['tokens',i]), dataset_row_index=i*10, dataset_row_end_exclusive=i*10+9) for i in range(96)],
    checks={k:True for k in ('prior96_documents_and_tokens_reproduced','all128_document_hashes_unique',
        'all128_prompt_hashes_unique','source_intervals_disjoint','closed_article_boundaries')})
manifest['novelty_inputs']=dict(receipt_path='fresh_inputs_receipt.json', receipt_sha256='', excluded_prior_requests=96)

def reject(call):
    try:
        call()
    except (ValueError, KeyError, TypeError, IndexError):
        return
    raise AssertionError('invalid fixture accepted')

with tempfile.TemporaryDirectory(prefix='strong-baseline-cpu-') as temp:
    run = Path(temp)/'execution'
    frozen = run/'frozen'
    prepared = frozen/'cohorts/cohort2/inputs_preparation/prepared/long'
    prepared.mkdir(parents=True)
    (frozen/'metrics.py').write_text('# CPU fixture; never used for measurement\n')
    def write(m=manifest, c=config, w=workload, rehash=False, receipt_data=receipt):
        m,c,w,r=deepcopy((m,c,w,receipt_data))
        if rehash:
            m['cohorts'][0]['workload_sha256']=c['workload_sha256']=r['workload_sha256']=sha(w)
        receipt_path=frozen/'fresh_inputs_receipt.json'
        receipt_path.write_text(json.dumps(r))
        m['novelty_inputs']['receipt_sha256']=module.shared.digest(receipt_path)
        for p,x in ((frozen/'campaign.json',m),(prepared/'config.json',c),(prepared/'workload.json',w)):
            p.write_text(json.dumps(x))
        return m
    manifest=write()
    module.validate_manifest(manifest, frozen)
    checks.append('valid_fixed_eight_cell_manifest_and_receipt')
    for key,value in (('request_id','fixture-request-0'),('document_sha256',workload['source_requests'][0]['document_sha256'])):
        w=deepcopy(workload); w['source_requests'][1][key]=value
        m=write(w=w,rehash=True); reject(lambda:module.validate_manifest(m,frozen))
    checks.append('duplicate_identity_and_hash_rejected')
    for mutation in ('arrival','length','token_hash','input_hash'):
        w=deepcopy(workload)
        if mutation=='arrival': w['arrival_traces_s']['steady'][1]=.06
        if mutation=='length': w['actual_prompt_token_ids'][0].pop()
        if mutation=='token_hash': w['actual_prompt_token_ids'][0][0]=999
        if mutation=='input_hash': w['source_requests'][0]['prompt']='changed'
        m=write(w=w,rehash=mutation!='input_hash'); reject(lambda:module.validate_manifest(m,frozen))
    checks.append('arrivals_lengths_token_and_workload_hashes_rejected')
    for key,value in (('completion_policy','rotate'),('cap',29),('victim_order','most_output')):
        m=deepcopy(manifest); m['cells'][0][key]=value; write(m=m)
        reject(lambda:module.validate_manifest(m,frozen))
    m=deepcopy(manifest); m['cells'][0],m['cells'][1]=m['cells'][1],m['cells'][0]; write(m=m)
    reject(lambda:module.validate_manifest(m,frozen))
    checks.append('wrong_role_policy_cap_victim_and_order_rejected')
    write()
    with patch.object(module.outcome,'module',return_value=SimpleNamespace(distribution=lambda x: None)):
        result=module.analyze(run)
    all_pairs=lambda r:r['comparisons']+r['native_aa']+r['same_role_repeats']
    assert result['status']=='UNRUN' and len(result['cells'])==8
    assert [len(result[k]) for k in ('comparisons','native_aa','same_role_repeats')]==[6,2,4]
    assert all(p['status']=='UNRUN_OR_INCOMPLETE_CAMPAIGN' for p in all_pairs(result))
    assert all(p['baseline_role']=='native' and p['action_role']=='native_aa' for p in result['native_aa'])
    checks.append('all_unrun_preserves_six_primary_two_aa_four_repeats_without_numbers')
    def row(spec, complete=True):
        return dict(label=spec['label'],cohort_id='cohort2',block=spec['block'],role=spec['role'],
            status='COMPLETE' if complete else 'INCOMPLETE',full_episode_comparison_eligible=complete,
            engine_args={'fixture':True}, config=dict(completion_policy=spec['completion_policy'],rotation_victim_order=spec['victim_order'],cap=32))
    for kind in ('partial','common_config_drift'):
        def inspect(r,s,n,f,c):
            result=row(s,not(kind=='partial' and s['label']==manifest['cells'][-1]['label']))
            if kind=='common_config_drift' and s['label']==manifest['cells'][-1]['label']:
                result['config']['unexpected']=True
            return result
        with patch.object(module.outcome,'module',return_value=SimpleNamespace(distribution=lambda x: None)),patch.object(module,'inspect_cell',side_effect=inspect):
            result=module.analyze(run)
        assert result['status']==('INCOMPLETE_CAMPAIGN' if kind=='partial' else 'INVALID_EVIDENCE')
        assert not result['comparisons_eligible'] and not result['recovery_accounting']
        assert all(p['status']=='UNRUN_OR_INCOMPLETE_CAMPAIGN' and 'metric_changes' not in p for p in all_pairs(result))
        checks.append(kind+'_blocks_all_numeric_pairs')
    spec=next(s for s in manifest['cells'] if s['role']=='most_output')
    base=row(spec); base['preemption_accounting']=dict(forced_count=1,natural_count=1)
    decisions=[dict(step=0,status='APPLIED',applied_rotations_before=0,effective_victim_order='most_output',
        forced_preempted=[],preempted=['natural'],proposal=None),
        dict(step=1,status='APPLIED',applied_rotations_before=0,effective_victim_order='most_output',
        forced_preempted=['v'],preempted=['v'],proposal=dict(action='rotate',victim_id='v')),
        dict(step=2,status='APPLIED',applied_rotations_before=1,effective_victim_order='most_output',
        forced_preempted=[],preempted=[],proposal=None)]
    def inspect_fixture(base_row, trace):
        with patch.object(module.shared,'inspect_cell',return_value=deepcopy(base_row)),patch.object(module.outcome,'read',return_value=trace):
            return module.inspect_cell(run,spec,None,frozen,manifest['cohorts'][0])
    assert inspect_fixture(base,decisions)['transition_check']['successful_forced_count']==1
    for kind in ('mode','counter','native_count','victim_config'):
        trace=deepcopy(decisions); r=deepcopy(base)
        if kind=='mode': trace[2]['effective_victim_order']='least_progress'
        if kind=='counter': trace[1]['applied_rotations_before']=1
        if kind=='native_count': r['preemption_accounting']['forced_count']=2
        if kind=='victim_config': r['config']['rotation_victim_order']='least_progress'
        assert inspect_fixture(r,trace)['status']=='INVALID_EVIDENCE'
    checks.append('constant_most_output_and_native_forced_count_reconciliation')
    for role in ('native','native_aa','headroom'):
        s=next(s for s in manifest['cells'] if s['role']==role); r=row(s)
        r['preemption_accounting']=dict(forced_count=0,natural_count=0)
        with patch.object(module.shared,'inspect_cell',return_value=r),patch.object(module.first_swap,'transition_check',side_effect=AssertionError('nonrotation transition called')):
            assert module.inspect_cell(run,s,None,frozen,manifest['cohorts'][0])['full_episode_comparison_eligible']
    checks.append('native_and_headroom_reuse_existing_checks_without_rotation_transition')

    write()
    wrong=deepcopy(manifest); wrong['novelty_inputs']['receipt_sha256']='0'*64
    reject(lambda:module.validate_manifest(wrong,frozen))
    for kind in ('cohort','workload','checks','prior_count','excluded_count'):
        r=deepcopy(receipt)
        if kind=='cohort': r['cohort_id']='cohort0'
        if kind=='workload': r['workload_sha256']='0'*64
        if kind=='checks': r['checks']['source_intervals_disjoint']=False
        if kind=='prior_count': r['prior_inputs'].pop()
        if kind=='excluded_count': r['excluded_prior_requests'].pop()
        m=write(receipt_data=r); reject(lambda:module.validate_manifest(m,frozen))
    checks.append('receipt_hash_identity_checks_and_96_count_rejected')
    for key in ('document_id','document_sha256','prompt_token_ids_sha256','interval'):
        r=deepcopy(receipt)
        if key=='interval':
            for field in ('dataset_row_index','dataset_row_end_exclusive'):
                r['excluded_prior_requests'][0][field]=workload['source_requests'][0][field]
        else:
            r['excluded_prior_requests'][0][key]=workload['source_requests'][0][key]
        m=write(receipt_data=r); reject(lambda:module.validate_manifest(m,frozen))
    checks.append('new32_prior96_identity_hash_and_interval_overlap_rejected')
    actual_root=HERE.parent/'inputs'
    actual_config,actual_workload,actual_receipt=[json.loads((actual_root/n).read_text()) for n in ('config.json','workload.json','inputs_report.json')]
    m=deepcopy(manifest); m['cohorts'][0]['workload_sha256']=actual_config['workload_sha256']
    m=write(m=m,c=actual_config,w=actual_workload,receipt_data=actual_receipt)
    (frozen/'fresh_inputs_receipt.json').write_bytes((actual_root/'inputs_report.json').read_bytes())
    m['novelty_inputs']['receipt_sha256']=module.shared.digest(actual_root/'inputs_report.json')
    (frozen/'campaign.json').write_text(json.dumps(m))
    cohorts,inputs=module.validate_manifest(m,frozen)
    assert inputs['cohort2']['novelty_check']['selected_requests']==32
    checks.append('actual_new_cohort2_inputs_and_receipt_accepted_without_prior_reconstruction')
    actual_inputs={n:module.shared.digest(actual_root/n) for n in ('config.json','workload.json','inputs_report.json')}

result=dict(status='PASS',checks=checks,source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
    actual_inputs=actual_inputs, workload_sha256=actual_config['workload_sha256'],
    scope='Synthetic CPU input/metadata/decision fixtures plus read-only validation of actual newly prepared cohort2 inputs/receipt. No old raw, prior-input reconstruction, GPU rerun or measured output. Existing native/headroom/host/work/recovery helpers reused, not comprehensively retested.')
with args.output.open('x') as output:
    output.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

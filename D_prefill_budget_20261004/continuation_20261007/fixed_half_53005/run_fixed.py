"""One fixed512/fixed1024 group using the unchanged parent serving episode."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import run_confirmation as base
from cohort_trace_53005.run_shm import resource_check, own_output_log_bytes


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_output_path(output, design):
    exact = '/dev/shm/moe-d-prefill-continuation-20261007/fixedhalf53005_01'
    assert design['intended_output_path'] == exact and design['intended_output_name'] == 'fixedhalf53005_01'
    assert design['intended_log_path'] == exact+'.log'
    assert output.is_absolute() and str(output) == exact, 'Only the frozen absolute output is authorized'


def warm_saturation(raw, cap):
    assert raw['policy'] == 'fixed'+str(cap)
    completed = [s for s in raw['steps'] if s['end_s'] is not None]
    saturated = [s for s in completed if s['budget']==cap and s['prefill_tokens']==cap]
    return dict(policy=raw['policy'], cap=cap, completed_steps=len(completed),
        actual_P_equals_cap_steps=len(saturated),
        mixed_actual_P_equals_cap_steps=sum(s['decode_tokens']>0 for s in saturated),
        maximum_actual_P=max((s['prefill_tokens'] for s in completed),default=0))


def require_warm_saturation(coverage):
    assert set(coverage)=={'fixed512','fixed1024'}, 'Both frozen warm episodes must complete before formal measurements'
    assert all(r['actual_P_equals_cap_steps']>0 for r in coverage.values()), 'A warm cap had no actual saturated prefill execution; no formal measurements or extra warm retries permitted'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-lock', type=float, default=7200)
    parser.add_argument('--check-only', action='store_true', help='Verify execution inputs only; no live resource queries')
    args = parser.parse_args()
    design = json.loads((ROOT/'design.json').read_text())
    assert design['status_at_freeze']=='PRE_GPU_FROZEN'
    assert design['policies']==['fixed512','fixed1024','fixed1024','fixed512']
    assert design['warm_policies']==['fixed512','fixed1024']
    assert design['legal_caps']==[512,1024]
    assert design['primary_slo']==dict(ttft_s=4,gap_s=.1,completion_s=20)
    assert design['timeout_s']==180 and design['formal_attempt_limit']==4 and design['warm_attempt_limit']==2
    assert design['max_wait_lock_s']==7200 and 0<=args.wait_lock<=7200
    assert design['shared_lock']=='/root/autodl-tmp/moe-research-gpu.lock'
    assert design['root_start_free_bytes_required']==design['root_between_episode_free_bytes_required']==1610612736
    assert design['shm_free_bytes_required']==536870912 and design['own_output_log_bytes_limit']==524288000
    assert design['effective_host_headroom_bytes_required']==2147483648
    assert design['after_lock_idle_check']==dict(max_utilization_percent=5,
        max_memory_used_mib=16,consecutive_readings=2,sample_separation_s=.5,grace_s=5)
    validate_output_path(args.output,design)
    required = {'run_fixed.py','launch.sh','../cohort_trace_53005/run_shm.py',
        '../cohort_control/controller.py','../cohort_control/cohort_cost.py',
        '../niyama_component/run_component.py','../device_refresh_53005/run_refresh.py',
        '../run_confirmation.py','../prefill_policy.py','../runtime_bootstrap.py'}
    assert set(design['source_sha256'])==required, 'Frozen execution dependency set differs'
    for relative,digest in design['source_sha256'].items():
        assert sha(ROOT/relative)==digest,relative
    workload = (ROOT/design['workload']).resolve()
    assert sha(workload)==design['workload_sha256']
    work = json.loads(workload.read_text())
    contract = dict(request_count=len(work),prompt_tokens=sum(len(q['prompt_token_ids']) for q in work),
                    declared_output_tokens=sum(q['max_tokens'] for q in work))
    assert contract==design['input_totals']
    if args.check_only:
        print(json.dumps(dict(status='FROZEN_GPU_EXECUTION_INPUTS_VERIFIED_NO_GPU',
            resource_check='Not performed in check-only mode',design_sha256=sha(ROOT/'design.json'),
            input_totals=contract,episode='Unmodified run_confirmation.episode',
            cache_policy='No compiler/cache/TMPDIR environment changes in this entrypoint')))
        return 0
    assert not args.output.exists(), 'No overwrite or retry of an existing group'
    before_queue = resource_check(design,'before_queue')
    after_lock = None
    original_dump,original_episode = base.dump,base.episode
    expected = [('warm_'+p,p) for p in design['warm_policies']]
    expected += [(f'{i:02d}_{p}',p) for i,p in enumerate(design['policies'])]
    episode_count,warm_coverage = 0,{}
    def guarded_episode(engine,work,policy_name,path,target_ms):
        nonlocal episode_count
        assert episode_count<len(expected), 'No extra episode or automatic retry permitted'
        assert (path.name,policy_name)==expected[episode_count]
        if episode_count>=len(design['warm_policies']):
            require_warm_saturation(warm_coverage)
        check = resource_check(design,'before_episode')
        print('EPISODE_RESOURCE_CHECK '+json.dumps(dict(cell=path.name,**check)),flush=True)
        episode_count += 1
        result = original_episode(engine,work,policy_name,path,target_ms)
        if path.name.startswith('warm_'):
            # Read already-completed raw results outside the original episode clock.
            warm_coverage[policy_name] = warm_saturation(json.loads((path/'raw.json').read_text()),int(policy_name[5:]))
            protocol_path = path.parent/'protocol.json'
            protocol = json.loads(protocol_path.read_text())
            protocol['warm_actual_saturation'] = dict(warm_coverage)
            original_dump(protocol_path,protocol)
            print('WARM_ACTUAL_SATURATION '+json.dumps(warm_coverage[policy_name]),flush=True)
        return result
    def dump(path,value):
        nonlocal after_lock
        if path.name=='status.json' and value.get('status')=='INITIALIZING':
            try:
                after_lock = resource_check(design,'after_lock')
            except Exception as exc:
                original_dump(path,dict(status='RESOURCE_BLOCKED_NO_GPU_INITIALIZED',pid=os.getpid(),reason=repr(exc)))
                raise
        if path.name=='engine_args.json':
            assert value==design['engine_args'], 'Normal-capacity engine configuration changed'
        if path.name=='protocol.json':
            assert after_lock is not None
            assert value['policies']==design['policies'] and value['warm_policies']==design['warm_policies']
            assert value['slo']==design['primary_slo'] and value['workload_sha256']==design['workload_sha256']
            value = dict(value,experiment_kind='FIXED_HALF_BASELINE_CHECK',
                frozen_device_refresh_design=design,frozen_device_refresh_design_sha256=sha(ROOT/'design.json'),
                startup_resource_checks=dict(before_queue=before_queue,after_lock=after_lock),
                episode_scope='Exact parent episode, normal arrivals/output/full drain; resource and warm-coverage checks outside episode clock.',
                warm_actual_saturation_requirement='Each completed fixed warm must contain actual P equal to its cap before any formal episode; no extra warm loop.')
        original_dump(path,value)
    base.dump,base.episode = dump,guarded_episode
    sys.argv = [str(ROOT.parent/'run_confirmation.py'),'--output',str(args.output.resolve()),
        '--workload',str(workload),'--policies',','.join(design['policies']),
        '--warm-policies',','.join(design['warm_policies']),'--target-ms','24','--wait-lock',str(args.wait_lock)]
    try:
        result = base.main()
        if result==0:
            assert episode_count==len(expected)
            try:
                own_output_log_bytes(design)
            except Exception as exc:
                original_dump(args.output/'status.json',dict(status='RESOURCE_BUDGET_FAILED_AFTER_EPISODES',
                    pid=os.getpid(),completed_episodes=episode_count,reason=repr(exc)))
                raise
        return result
    finally:
        base.dump,base.episode = original_dump,original_episode


if __name__=='__main__':
    raise SystemExit(main())

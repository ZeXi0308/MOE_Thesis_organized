"""Bounded fixed/OA-style prefill-demand comparison; no new runtime actions."""
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
import niyama_component.run_component as original_component
from cohort_trace_53005.run_shm import resource_check, own_output_log_bytes
from prefill_demand.controller import DemandController


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derived_episode():
    previous = original_component.ComponentController
    original_component.ComponentController = DemandController
    try:
        return original_component.derived_episode()
    finally:
        original_component.ComponentController = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-lock', type=float, default=7200)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    design = json.loads((ROOT/'design.json').read_text())
    assert design['status_at_freeze']=='PRE_GPU_FROZEN'
    assert design['policies']==['fixed1024','prefill_demand','prefill_demand','fixed1024']
    assert design['warm_policies']==['fixed512','fixed1024','fixed2048','prefill_demand']
    assert design['legal_caps']==[512,1024,2048] and design['prediction_margin']==1.2
    assert design['primary_slo']==dict(ttft_s=4,gap_s=.1,completion_s=20)
    assert design['timeout_s']==180 and design['formal_attempt_limit']==design['warm_attempt_limit']==4
    assert design['max_wait_lock_s']==7200 and 0<=args.wait_lock<=7200
    exact = '/dev/shm/moe-d-prefill-continuation-20261007/demand53005_02'
    assert str(args.output)==design['intended_output_path']==exact
    assert design['intended_output_name']=='demand53005_02' and design['intended_log_path']==exact+'.log'
    assert design['shared_lock']=='/root/autodl-tmp/moe-research-gpu.lock'
    assert design['root_start_free_bytes_required']==design['root_between_episode_free_bytes_required']==1610612736
    assert design['shm_free_bytes_required']==536870912 and design['own_output_log_bytes_limit']==524288000
    assert design['effective_host_headroom_bytes_required']==2147483648
    assert design['after_lock_idle_check']==dict(max_utilization_percent=5,max_memory_used_mib=16,
        consecutive_readings=2,sample_separation_s=.5,grace_s=5)
    required = {'run_demand.py','launch.sh','../prefill_demand/controller.py','../cohort_trace_53005/run_shm.py',
        '../cohort_control/controller.py','../cohort_control/cohort_cost.py',
        '../niyama_component/run_component.py','../device_refresh_53005/run_refresh.py',
        '../run_confirmation.py','../prefill_policy.py','../runtime_bootstrap.py'}
    assert set(design['source_sha256'])==required
    for relative,digest in design['source_sha256'].items():
        assert sha(ROOT/relative)==digest,relative
    workload = (ROOT/design['workload']).resolve()
    assert sha(workload)==design['workload_sha256']
    work = json.loads(workload.read_text())
    contract = dict(request_count=len(work),prompt_tokens=sum(len(q['prompt_token_ids']) for q in work),
        declared_output_tokens=sum(q['max_tokens'] for q in work))
    assert contract==design['input_totals']
    calibration_path = ROOT/design['calibration_file']
    assert sha(calibration_path)==design['calibration_sha256']
    calibration = json.loads(calibration_path.read_text())
    assert not calibration['fit_errors'] and set(calibration['runtime_models'])=={'le512','gt512'}
    DemandController.design, DemandController.calibration = design, calibration
    component_episode, source = derived_episode()
    if args.check_only:
        print(json.dumps(dict(status='FROZEN_GPU_EXECUTION_INPUTS_VERIFIED_NO_GPU',
            design_sha256=sha(ROOT/'design.json'), input_totals=contract,
            episode='Existing component derived episode; controller class only replaced',
            resources='Not queried in check-only; original compiler/cache/TMPDIR paths unchanged')))
        return 0
    assert base.UUID==design['gpu_uuid'], 'Explicit authorized physical GPU required'
    assert not args.output.exists(), 'No overwrite or automatic retry'
    before_queue = resource_check(design,'before_queue')
    after_lock = None
    original_dump, original_episode = base.dump, base.episode
    expected = [('warm_'+p,p) for p in design['warm_policies']]
    expected += [(f'{i:02d}_{p}',p) for i,p in enumerate(design['policies'])]
    episode_count, coverage = 0, {}

    def guarded_episode(engine,work,policy,path,target_ms):
        nonlocal episode_count
        assert episode_count<len(expected) and (path.name,policy)==expected[episode_count]
        if episode_count>=4:
            assert set(coverage)=={'fixed512','fixed1024','fixed2048'}
            assert all(v['actual_P_equals_cap_steps']>0 for v in coverage.values()), 'Warm cap never saturated; no extra warm retry'
        check = resource_check(design,'before_episode')
        print('EPISODE_RESOURCE_CHECK '+json.dumps(dict(cell=path.name,**check)),flush=True)
        episode_count += 1
        result = component_episode(engine,work,policy,path,target_ms)
        if path.name.startswith('warm_fixed'):
            raw = json.loads((path/'raw.json').read_text())
            cap = int(policy[5:])
            saturated = [s for s in raw['steps'] if s['end_s'] is not None and s['prefill_tokens']==cap]
            coverage[policy] = dict(cap=cap,actual_P_equals_cap_steps=len(saturated),
                mixed_actual_P_equals_cap_steps=sum(s['decode_tokens']>0 for s in saturated))
            protocol_path = path.parent/'protocol.json'
            protocol = json.loads(protocol_path.read_text())
            protocol['warm_actual_saturation'] = dict(coverage)
            original_dump(protocol_path,protocol)
            print('WARM_ACTUAL_SATURATION '+json.dumps(coverage[policy]),flush=True)
        if path.name=='01_prefill_demand':
            raw = json.loads((path/'raw.json').read_text())
            interventions = sum(s['prefill_tokens']>1024 or
                (s['budget']<1024 and s['prefill_tokens']==s['budget'] and s['prefill_backlog_tokens_before']>s['budget'])
                for s in raw['steps'])
            print('FIRST_PAIR_ACTUAL_INTERVENTIONS '+str(interventions),flush=True)
            assert interventions>0, 'ZERO_ACTUAL_ACTION_STOP_AFTER_FIRST_PAIR; preserve completed raw, no reverse retry'
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
            value = dict(value,experiment_kind='OA_STYLE_PREFILL_DEMAND_BASELINE_DEVELOPMENT',
                frozen_component_design=design,frozen_component_design_sha256=sha(ROOT/'design.json'),
                startup_resource_checks=dict(before_queue=before_queue,after_lock=after_lock),
                warm_actual_saturation_requirement='All three fixed cap warms must actually saturate before formal.',
                episode_scope='Unchanged component episode; only controller class replaced. Guards outside full serving clock.')
            (path.parent/'derived_episode.py').write_text(source)
            original_dump(path.parent/'calibration.json',calibration)
        original_dump(path,value)

    base.dump, base.episode = dump, guarded_episode
    sys.argv = [str(ROOT.parent/'run_confirmation.py'),'--output',str(args.output),
        '--workload',str(workload),'--policies',','.join(design['policies']),
        '--warm-policies',','.join(design['warm_policies']),'--target-ms','24','--wait-lock',str(args.wait_lock)]
    try:
        result = base.main()
        if result==0:
            assert episode_count==8
            try:
                own_output_log_bytes(design)
            except Exception as exc:
                original_dump(args.output/'status.json',dict(status='RESOURCE_BUDGET_FAILED_AFTER_EPISODES',
                    pid=os.getpid(),completed_episodes=episode_count,reason=repr(exc)))
                raise
        return result
    finally:
        base.dump, base.episode = original_dump, original_episode


if __name__=='__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""r01 resource controller and four-cell order, with dynamically reconstructed guard wiring only."""
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
PARENT_SHA='3e1e4cd573a2eea685b9721a36dbe0ac3b731036116a2675dd1d000a155de8fd'
GUARD_SHA='fa1decbde3afbcbdf51d479b234cae832aeb25ad3084939d144d4267d2ac7ac3'
FROZEN_EXCHANGE_SHA='ce54ed804c757d936c945741b1f19b8e8906053836a03630d978e7e4d771ca97'


def load_parent():
    path=ROOT/'run_group.py'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=PARENT_SHA:raise RuntimeError('Frozen r01 controller changed')
    spec=importlib.util.spec_from_file_location('guarded_exchange_group_parent',path)
    parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
    return parent


exchange_action_evidence=load_parent().action_evidence


def action_evidence(data,mode):
    evidence=exchange_action_evidence(data,mode)
    guard=data.get('guard_adapter',{})
    checks=guard.get('checks')
    if (guard.get('status')!='UNINSTALLED' or guard.get('original_code_verified') is not True
            or guard.get('frozen_exchange_sha256')!='ce54ed804c757d936c945741b1f19b8e8906053836a03630d978e7e4d771ca97'
            or not isinstance(checks,list) or len(checks)!=evidence['actual_exchange_count']
            or any(c.get('allowed') is not True or c.get('reasons') or c.get('forced_count')!=1
                or c.get('donor')!=evidence['donor'] or c.get('result_preempted_req_ids')!=[evidence['donor']]
                for c in checks)):
        raise RuntimeError('Guarded return lacks exactly one matched authorization per actual exchange')
    runtime=guard.get('runtime_source_construction',{})
    if (runtime.get('installer_code_verified') is not True
            or runtime.get('namespace')!='normal_capacity_native_observation'
            or runtime.get('fixed_cap')!=256
            or runtime.get('normal_constructor_sha256')!='9f92617d44bbf8cda6992261129c35ff844a14a569dc8f259ed2bdedaa2388bb'
            or runtime.get('staged_runtime_source_sha256')!='52b5a7c9ead5ce2bfee84a0b0d51c1c7e73c2c9e177e84335e32d61544a60dc9'):
        raise RuntimeError('Guarded return runtime reconstruction provenance missing')
    evidence['runtime_source_construction']=runtime
    evidence['guarded_return_checks']=checks
    return evidence


def adapted_source():
    parent=load_parent();text,sources=parent.adapted_source()
    old=inspect.getsource(parent.action_evidence)
    new=old.replace('def action_evidence(','def exchange_action_evidence(',1)+'\n\n'+inspect.getsource(action_evidence)
    for before,after in ((old,new),("str(ROOT/'run_cell.py')","str(ROOT/'run_cell_runtime.py')")):
        if text.count(before)!=1:raise RuntimeError('Guarded controller boundary changed: '+before[:80])
        text=text.replace(before,after)
    return text,[*sources,ROOT/'run_group.py',ROOT/'run_cell_runtime.py',ROOT/'exchange_guarded_runtime.py',
        ROOT/'exchange_once.py',ROOT/'exchange_guarded.py',ROOT/'check_guard_cpu.py',
        ROOT/'check_guard_runtime_cpu.py',Path(__file__)]


def self_check():
    text,_=adapted_source();compile(text,'<guarded-controller>','exec')
    assert "str(ROOT/'run_cell_runtime.py')" in text and "str(ROOT/'run_cell.py')" not in text
    assert "['stall8', 'exchange_once', 'exchange_once', 'stall8']" in text
    assert "if i == 1 and evidence['actual_exchange_count'] == 0:" in text
    assert "raise RuntimeError('Cell failed; all raw retained, no automatic retry')" in text
    assert "child.wait(timeout=p['per_cell_seconds'])" in text
    baseline=dict(mode='stall8',status='UNINSTALLED',action_count=0,action_limit=8,events=[],bypassed_episodes=[])
    data=dict(mode='stall8',status='UNINSTALLED',action_count=0,max_protected_rounds=16,
        baseline_stall8=baseline,events=[],guard_adapter=dict(status='UNINSTALLED',
        original_code_verified=True,frozen_exchange_sha256=FROZEN_EXCHANGE_SHA,checks=[],
        runtime_source_construction=dict(installer_code_verified=True,
            namespace='normal_capacity_native_observation',fixed_cap=256,
            normal_constructor_sha256='9f92617d44bbf8cda6992261129c35ff844a14a569dc8f259ed2bdedaa2388bb',
            staged_runtime_source_sha256='52b5a7c9ead5ce2bfee84a0b0d51c1c7e73c2c9e177e84335e32d61544a60dc9')))
    assert action_evidence(data,'stall8')['actual_exchange_count']==0
    data['guard_adapter']['checks']=[dict(allowed=True)]
    try:action_evidence(data,'stall8')
    except RuntimeError:pass
    else:raise AssertionError('Foreign guard authorization accepted')
    print('PASS: guarded child, unchanged ABBA/450s/zero-action and failure stop; original whole-group resource main retained. CPU only.')


def main():
    if sys.argv[1:]==['--self-check']:self_check();return 0
    parent=load_parent();parent.adapted_source=adapted_source;parent.__file__=str(__file__)
    return parent.main()


if __name__=='__main__':raise SystemExit(main())

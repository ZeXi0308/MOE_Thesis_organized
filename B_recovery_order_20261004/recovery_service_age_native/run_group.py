#!/usr/bin/env python3
"""Frozen native/stall8 comparison with identical receipt observation."""
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PARENT = BASE/'recovery_service_age/run_group.py'
PARENT_SHA = 'c1e30f0eb306b1647e7e2b17302d9c78f9d5f2cca73be43fcb5f8dee8430e29d'


def load_parent():
    if hashlib.sha256(PARENT.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen service-age resource controller changed')
    spec = importlib.util.spec_from_file_location('native_service_age_group_parent', PARENT)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    return parent


def action_source():
    source = inspect.getsource(load_parent().action_evidence)
    old = "        reasons = []\n        if episode in episodes:"
    new = "        reasons = []\n        if mode == 'native': reasons.append('NATIVE_OBSERVATION_ONLY')\n        if episode in episodes:"
    if source.count(old) != 1: raise RuntimeError('Native action guard boundary changed')
    source = source.replace(old,new)
    old = "    expected = [dict(request=rid,num_preemptions=number) for rid,number in episodes]"
    new = "    if mode == 'native' and episodes: raise RuntimeError('Native arm executed a bypass')\n"+old
    if source.count(old) != 1: raise RuntimeError('Native action result boundary changed')
    return source.replace(old,new)


def action_evidence(data,mode):
    namespace = {}
    exec(compile(action_source(), '<native-service-age-evidence>', 'exec'),namespace)
    return namespace['action_evidence'](data,mode)


def adapted_source():
    parent=load_parent();text,sources=parent.adapted_source()
    def replace(old,new):
        nonlocal text
        if text.count(old)!=1:raise RuntimeError('Native controller boundary changed: '+old[:100])
        text=text.replace(old,new)
    replace(inspect.getsource(parent.action_evidence),action_source())
    replace("['age8', 'stall8', 'stall8', 'age8']","['native', 'stall8', 'stall8', 'native']")
    replace('Requires fixed cap256 age8/stall8/stall8/age8,450s per child,current GPU/110GiB host',
            'Requires fixed cap256 native/stall8/stall8/native,450s per child,current GPU/110GiB host')
    replace('B_RECOVERY_SERVICE_AGE=mode,','B_RECOVERY_SERVICE_AGE_NATIVE=mode,')
    replace("receipt['cells'][-1]['recovery_service_age_mode']","receipt['cells'][-1]['recovery_service_age_native_mode']")
    replace("cell/'output/recovery-service-age.json'","cell/'output/recovery-service-age-native.json'")
    replace("receipt['cells'][-1]['recovery_service_age_actions']","receipt['cells'][-1]['recovery_service_age_native_actions']")
    replace("if i == 1 and evidence['difference_decisions'] == 0:","if i == 1 and evidence['reorder_count'] == 0:")
    replace('First stall8 had no different executed queue mutation or executable age8 suppression; reverse cells not started; suggestions alone do not count',
            'First stall8 had no executed legal bypass from native queue; reverse cells not started; shadow suggestions alone do not count')
    return text,[*sources,PARENT,Path(__file__)]


def self_check():
    # Only the new observation-only guard and actual-bypass continuation test
    # changed; the already-tested shared-resource lifecycle is inherited.
    text,_=adapted_source();compile(text,'<native-service-age-group>','exec')
    assert "if i == 1 and evidence['reorder_count'] == 0:" in text
    assert 'B_RECOVERY_SERVICE_AGE_NATIVE=mode' in text
    data=dict(mode='native',status='UNINSTALLED',action_limit=8,action_count=0,
        bypassed_episodes=[],events=[dict(waiting_before=['h','a','s'],waiting_after=['h','a','s'],
        age8_suggestion='a',stall8_suggestion='s',age8_suggestion_num_preemptions=1,
        stall8_suggestion_num_preemptions=1,age8_would_execute=True,stall8_would_execute=True,
        candidate_head='s',candidates=[dict(request=r,eligible=True,num_preemptions=1) for r in ('a','s')],
        action_count_before=0,action_count_after=0,action='SHADOW_ONLY',
        action_skip_reasons=['NATIVE_OBSERVATION_ONLY'],final_head='h',queue_changed=False)])
    assert action_evidence(data,'native')['reorder_count']==0
    data['events'][0]['action_skip_reasons']=[]
    try:action_evidence(data,'native')
    except RuntimeError:pass
    else:raise AssertionError('Native observation without explicit guard accepted')
    print('PASS: native observation guard, no native mutation, assembled inherited controller and actual-bypass stop; no GPU.')


def main():
    if sys.argv[1:]==['--self-check']:self_check();return 0
    parent=load_parent();parent.adapted_source=adapted_source;parent.__file__=str(__file__)
    return parent.main()


if __name__=='__main__':raise SystemExit(main())

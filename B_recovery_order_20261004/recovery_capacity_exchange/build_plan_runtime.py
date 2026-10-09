#!/usr/bin/env python3
"""Create a new r03 plan; scientific inputs/settings are copied from frozen r01."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
PARENT_SHA='d4df63cd206a643157592ed6752bd214d6127326dba32a628453ab2f540c31d0'
R01_SHA='9f01c4f60214feac90c0d403226b7e58a4ad406c21955e1f7aa78ff9f811b019'
POLICY_PINS={
    'exchange_guarded_runtime.py':'fa1decbde3afbcbdf51d479b234cae832aeb25ad3084939d144d4267d2ac7ac3',
    'exchange_guarded.py':'6132bfacffcfc63d09e218213d3c6577fc749a7088eabe58d69d329da99dc689',
    'check_guard_runtime_cpu.py':'3bb6aa5fe5ac05bbe0b8b8fa9011eff2ea8166560b5492916e555c692aa0a906',
    'exchange_once.py':'ce54ed804c757d936c945741b1f19b8e8906053836a03630d978e7e4d771ca97',
    'check_guard_cpu.py':'586af77ddb32feef68d513dec9776cedb86a9fd350d7262bc5204d054bcef6a0',
}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def prepare_plan(session_dir=None):
    if sha(ROOT/'build_plan.py')!=PARENT_SHA or sha(ROOT/'plan-20261009-r01.json')!=R01_SHA:
        raise RuntimeError('Frozen failed-run plan or builder changed')
    for name,expected in POLICY_PINS.items():
        if sha(ROOT/name)!=expected:raise RuntimeError('Guarded policy dependency changed: '+name)
    child=load(ROOT/'run_cell_runtime.py','guarded_plan_child')
    if child.POLICY_SHA!=POLICY_PINS['exchange_guarded_runtime.py']:
        raise RuntimeError('Guarded child policy pin differs')
    parent=load(ROOT/'build_plan.py','guarded_plan_parent')
    original=json.loads((ROOT/'plan-20261009-r01.json').read_text())
    remote=original['source_dir']+'/recovery_capacity_exchange'
    plan=parent.prepare_plan(session_dir or remote+'/session-20261009-r03')
    for key in ('fixed','inputs_sha256','resource_checks','sequence','per_cell_seconds','stage_budget'):
        if plan[key]!=original[key]:raise RuntimeError('Guard correction changed frozen factor: '+key)
    group=load(ROOT/'run_group_runtime.py','guarded_plan_group')
    text,sources=group.adapted_source();compile(text,'<guarded-plan-controller>','exec')
    plan.update(policy_sha256=POLICY_PINS['exchange_guarded_runtime.py'],
        policy_pins={'recovery_capacity_exchange/'+name:value for name,value in POLICY_PINS.items()},
        source_parent_plan='plan-20261009-r01.json',source_parent_plan_sha256=R01_SHA,
        guard_interface_correction=dict(status='CPU_CHECKED_GPU_UNRUN',
            scope='Reconstruct the exact pinned normal-capacity dynamic staged installer and live schedule code before applying the unchanged narrow donor guard predicate. All original r01 target/donor selection, capacity arithmetic, Q1/16-entry bound, fixed workload, resource checks and stop rules remain unchanged.',
            failed_r01='Retained failed run: metadata built, then original guard raised before worker received SchedulerOutput; incomplete/failed requests remain in its own results.',
            failed_r02='Retained initialization failure: dynamic virtual filename caused source lookup failure before formal measurement, zero measured requests/actions. r03 is a new explicitly prepared sibling run; no automatic retries.',
            superseded_unrun_r03_plan_sha256='4d428778b288abf9fe223a5f924e787fcc4e5b587bfc7ce60ef4f7ee065b876a',
            cpu_fixture_path_correction='Optional B_GUARD_TEST_SCHEDULER_SOURCE selects the existing scheduler source for CPU checks; its original SHA remains mandatory. Prior r03 deployment stopped in this CPU fixture before GPU execution. Runtime policy/child/group unchanged.'),
        prepared_command=plan['python']+' -B '+remote+'/run_group_runtime.py --plan '+remote+'/plan-20261009-r03.json --wait-lock-seconds 1800')
    paths=[*sources,ROOT/'run_cell_runtime.py',ROOT/'run_group_runtime.py',Path(__file__),
        *(ROOT/name for name in POLICY_PINS)]
    plan['runner_sources_sha256'].update({str(path.relative_to(BASE)):sha(path) for path in set(paths)})
    return plan


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
    parser.add_argument('--session-dir');parser.add_argument('--self-check',action='store_true');args=parser.parse_args()
    plan=prepare_plan(args.session_dir)
    if args.self_check:
        assert plan['session_dir'].endswith('session-20261009-r03') or args.session_dir
        assert plan['per_cell_seconds']==450 and len(plan['sequence'])==4
        assert set(POLICY_PINS)=={Path(p).name for p in plan['policy_pins']}
        print('PASS: unchanged r01 science/resources/stop budget; guarded policy, frozen dependency, CPU test and actual new child/controller pins. CPU only.')
        return 0
    if args.output is None:parser.error('--output is required unless --self-check')
    if args.output.exists():raise FileExistsError(args.output)
    plan['prepared_command']=plan['prepared_command'].replace('/plan-20261009-r03.json ','/'+args.output.name+' ')
    with args.output.open('x') as stream:json.dump(plan,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(args.output)


if __name__=='__main__':raise SystemExit(main())

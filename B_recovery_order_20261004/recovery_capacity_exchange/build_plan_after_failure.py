#!/usr/bin/env python3
"""One controlled source-ID action probe, not an online target selector."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
PARENT=ROOT/'plan-20261009-r03.json'
PARENT_SHA='f17c36055383fecdc2e966db6b619c10bac3c0a8b547db2d6fb6c934dd992559'
TARGET='b-normal-0070571-long'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def prepare_plan(session_dir=None):
    if sha(PARENT)!=PARENT_SHA:raise RuntimeError('Frozen completed r03 plan changed')
    p=json.loads(PARENT.read_text())
    policy=ROOT/'exchange_after_failure.py'
    child=load(ROOT/'run_cell_after_failure.py','failure_probe_child')
    if child.POLICY_SHA is None or sha(policy)!=child.POLICY_SHA:
        raise RuntimeError('Controlled failure probe policy is not frozen')
    for rel,expected in p['inputs_sha256'].items():
        if sha(BASE/rel)!=expected:raise RuntimeError('Frozen workload changed: '+rel)
    group=load(ROOT/'run_group_after_failure.py','failure_probe_group')
    text,sources=group.adapted_source();compile(text,'<failure-probe-controller>','exec')
    remote=p['source_dir']+'/recovery_capacity_exchange'
    p.update(status='PREPARED_UNRUN',
        session_dir=session_dir or '/tmp/moe-b-recovery-after-failure-20261009-r01',
        source_parent_plan='plan-20261009-r03.json',source_parent_plan_sha256=PARENT_SHA,
        policy_sha256=child.POLICY_SHA,
        scope='Controlled action-value diagnostic on a preselected development request. Not an online selector, independent confirmation, natural-EOS experiment, or a matched-state causal estimate.',
        research_question='Can a fresh single-donor exchange after the known long request actually fails recovery allocation improve its full-request stall and the all-request tradeoff?',
        only_changed_factor='Relative to r03: target and trigger only. Wait for the first observed async full-history allocation failure of b-normal-0070571-long, then attempt once at the immediately following rotation_begin with current object/episode/receipt and fresh donor/T/R/G state. Original Q1,16-entry fail-open,one donor,stall8,copy paths and workload unchanged.',
        decision='No legal executed exchange: stop reverse cells and limit conclusion to this one boundary. Full-service harm/no valuable tradeoff: stop this long-target Q1 probe. Valuable tradeoff: only then design an ID-independent online selector and matched simple baseline; no threshold or Q sweep.',
        controlled_target=dict(source_request=TARGET, selection_basis='Largest long stall in prior development traces; chosen before this run. This privileged source-ID choice is a diagnostic and cannot enter any claimed deployable method.',trigger='First actual async allocation failure; fresh validation at immediately following begin; stale/changed evidence fails open.'),
        storage_placement=dict(reason='Authorized data filesystem below original floor; /tmp filesystem observed 4594266112B free at1791485763.0197115. Use identical private per-cell cache/output placement on /tmp, retain the original session-filesystem2.5GiB launch floor and public lock. No shared cache or other-line data removed.', scientific_comparison='Only the contemporaneous four cells are compared; r03 is not used as a performance baseline.'),
        prepared_command=p['python']+' -B '+remote+'/run_group_after_failure.py --plan '+remote+'/plan-after-failure-20261009-r01.json --wait-lock-seconds 1800')
    p.pop('guard_interface_correction',None)
    p['policy_pins']['recovery_capacity_exchange/exchange_after_failure.py']=child.POLICY_SHA
    p['fixed']['observations']='Identical frozen stall8 receipts, compact capture/passive GC, existing tail allocator observer plus controlled failure latch and fresh exchange shadow in both arms. No extra CUDA queries or allocation/lookup calls.'
    extra=[*sources,ROOT/'run_cell_after_failure.py',ROOT/'run_group_after_failure.py',policy,ROOT/'check_after_failure_cpu.py',Path(__file__)]
    p['runner_sources_sha256'].update({str(x.relative_to(BASE)):sha(x) for x in set(extra)})
    return p


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
    parser.add_argument('--session-dir');parser.add_argument('--self-check',action='store_true');args=parser.parse_args()
    p=prepare_plan(args.session_dir)
    if args.self_check:
        old=json.loads(PARENT.read_text())
        for key in ['inputs_sha256','resource_checks','sequence','per_cell_seconds','stage_budget','host_limit_bytes','gpu_uuid','fixed_gpu_kv_bytes','cap']:
            assert p[key]==old[key],key
        assert {k for k in p['fixed'] if p['fixed'][k]!=old['fixed'][k]}=={'observations'}
        assert p['controlled_target']['source_request']==TARGET
        print('PASS: only controlled target/trigger and private storage placement changed; resource, workload, Q1 and four-cell bounds retained. GPU unrun.')
        return
    if args.output is None:parser.error('--output required unless --self-check')
    if args.output.exists():raise FileExistsError(args.output)
    p['prepared_command']=p['prepared_command'].replace('plan-after-failure-20261009-r01.json',args.output.name)
    with args.output.open('x') as f:json.dump(p,f,ensure_ascii=False,indent=2);f.write('\n')
    print(args.output)


if __name__=='__main__':main()

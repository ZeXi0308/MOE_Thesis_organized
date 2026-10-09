#!/usr/bin/env python3
"""Freeze one maximum-release endpoint group over the completed failure probe."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PARENT = ROOT/'plan-after-failure-20261009-r01.json'
PARENT_SHA = '99396ca02c07e161fa7e8f005daa20aaf78e5be6cf5e227be1382a3e4b8a2099'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def prepare_plan():
    if sha(PARENT) != PARENT_SHA: raise RuntimeError('Completed failure-probe plan changed')
    p = json.loads(PARENT.read_text())
    child = load(ROOT/'run_cell_max_release.py', 'max_release_child_plan')
    policy = ROOT/'exchange_max_release.py'
    if sha(policy) != child.POLICY_SHA: raise RuntimeError('Maximum-release policy not frozen')
    group = load(ROOT/'run_group_max_release.py', 'max_release_group_plan')
    text, sources = group.adapted_source(); compile(text, '<max-release-group>', 'exec')
    for rel, expected in p['inputs_sha256'].items():
        if sha(BASE/rel) != expected: raise RuntimeError('Frozen workload changed: '+rel)
    p.update(status='PREPARED_UNRUN', session_dir='/tmp/moe-b-recovery-max-release-20261009-r01',
        source_parent_plan=PARENT.name, source_parent_plan_sha256=PARENT_SHA,
        policy_sha256=child.POLICY_SHA,
        research_question='Does ordinary maximum single-donor release create a worthwhile sustained recovery service tradeoff after the failed minimal exchange?',
        only_changed_factor='Within the same fresh legal donor set, choose maximum immediate release; tie by missing materialized Host pages then native tail. Controlled target, first-failure/next-begin opportunity, all gates, Q1/16-entry bounds and ordinary stall8 remain unchanged. No gate relaxation or repeated opportunities.',
        decision='A worthwhile full-service tradeoff supports ordinary extra capacity first, not a new predictor. Target benefit with donor/peer harm is a tradeoff, not net benefit. No benefit or renewed immediate re-preemption closes this endpoint in the single-donor/Q1 domain. First candidate zero action aborts reverse cells, remains opportunity-uncertain, and receives no automatic retry.',
        scope='Controlled development-ID action-value experiment comparing stall8 against an ordinary maximum-release endpoint. Not an optimal bound, ID-independent method, matched-state estimate, or independent confirmation.',
        storage_placement=dict(reason='Use the same /tmp filesystem as the contemporaneous failure-probe group. Read-only check1791488381.6594884 found3605139456B free; the unchanged2.5GiB floor is checked again under the public lock. Private per-cell empty caches, no other-line deletion.', scientific_comparison='Only contemporaneous arms are service baselines; prior runs diagnose mechanism and are not causal performance controls.'),
        prepared_command=p['python']+' -B '+p['source_dir']+'/recovery_capacity_exchange/run_group_max_release.py --plan '+p['source_dir']+'/recovery_capacity_exchange/plan-max-release-20261009-r01.json --wait-lock-seconds 1800')
    p['fixed']['victim'] = 'Native tail except at most one fresh maximum-immediate-release legal donor, tied by Host missing pages then native tail'
    p['fixed']['observations'] = 'Identical frozen failure-trigger/receipts/compact/GC/allocator observation; both arms record fresh minimal and maximum donor suggestions. No added GPU synchronization, allocations or lookups.'
    p['policy_pins']['recovery_capacity_exchange/exchange_max_release.py'] = child.POLICY_SHA
    sources += [ROOT/'run_cell_max_release.py', policy, ROOT/'check_max_release_cpu.py', Path(__file__)]
    p['runner_sources_sha256'].update({str(x.relative_to(BASE)):sha(x) for x in set(sources)})
    return p


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path)
    parser.add_argument('--self-check',action='store_true'); args=parser.parse_args()
    p=prepare_plan()
    if args.self_check:
        old=json.loads(PARENT.read_text())
        for key in ['inputs_sha256','resource_checks','sequence','per_cell_seconds','stage_budget','host_limit_bytes','gpu_uuid','fixed_gpu_kv_bytes','cap','controlled_target']:
            assert p[key]==old[key],key
        assert {k for k in p['fixed'] if p['fixed'][k]!=old['fixed'][k]}=={'victim','observations'}
        print('PASS: maximum donor only; original input, target/trigger, Q1, resource and group bounds retained. GPU unrun.')
        return
    if args.output is None: parser.error('--output required')
    with args.output.open('x') as f: json.dump(p,f,ensure_ascii=False,indent=2); f.write('\n')
    print(args.output)


if __name__=='__main__': main()

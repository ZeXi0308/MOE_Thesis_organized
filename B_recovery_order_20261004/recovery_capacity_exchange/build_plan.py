#!/usr/bin/env python3
"""Prepare one bounded capacity-exchange plan from the old development inputs."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PARENT = BASE/'recovery_service_age/plan-20261009-r01.json'
PARENT_SHA = 'a2a0c573cc84f4597273c1f5e49a4098ebbc3bed3a964f82f9a4c1a4bdac8c02'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_plan(session_dir=None):
    if sha(PARENT) != PARENT_SHA:
        raise RuntimeError('Frozen service-age parent plan changed')
    tree = ast.parse((ROOT/'run_cell.py').read_text())
    pins = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'POLICY_SHA' for t in n.targets)]
    if len(pins) != 1 or not isinstance(pins[0], str) or not re.fullmatch('[0-9a-f]{64}', pins[0]):
        raise RuntimeError('Capacity-exchange policy SHA is not frozen; no plan written')
    if sha(ROOT/'exchange_once.py') != pins[0]:
        raise RuntimeError('Capacity-exchange policy differs from the frozen child pin')
    p = json.loads(PARENT.read_text())
    for relative, expected in p['inputs_sha256'].items():
        if sha(BASE/relative) != expected:
            raise RuntimeError('Old development input changed: '+relative)
    input_dir = BASE/'recovery_start_gate/simple_cap/inputs/cap256'
    config = json.loads((input_dir/'config.json').read_text()); workload = json.loads((input_dir/'workload.json').read_text())
    canonical = hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest()
    if (canonical != '208e7c0786b326985a3f69f67600963dc2d9aaa81a6c750c4a338466d0529eb2'
            or canonical != config['workload_sha256'] or config['ignore_eos'] is not True
            or config['output_tokens'] != 1024 or config['requests'] != 256 or config['cap'] != 256):
        raise RuntimeError('Requires the old source-order development fixed1024 workload')
    spec = importlib.util.spec_from_file_location('exchange_group_plan_check', ROOT/'run_group.py')
    group = importlib.util.module_from_spec(spec); spec.loader.exec_module(group)
    text, sources = group.adapted_source(); compile(text, '<exchange-controller>', 'exec')
    remote = p['source_dir']+'/recovery_capacity_exchange'
    p.update(status='PREPARED_UNRUN', sequence=['stall8','exchange_once','exchange_once','stall8'],
        session_dir=session_dir or remote+'/session-20261009-r01',
        inputs_dir=p['source_dir']+'/recovery_start_gate/simple_cap/inputs',
        source_parent_plan='../recovery_service_age/plan-20261009-r01.json', source_parent_plan_sha256=PARENT_SHA,
        scope='One bounded exploratory capacity-exchange probe on the original development arrival order. Not new task data, independent confirmation, or a pure task-order intervention. No input chosen from the new native arrival-permutation outcomes.',
        research_question='When frozen stall8 has a non-fitting recovery, can one funded donor preemption plus finite target protection reduce all-request output stalls after accounting for donor and peers?',
        only_changed_factor='At most one extra native donor preemption with coupled target Q1 protection; protection fails open at16 rounds or earlier safety release. Both arms retain frozen stall8 and its total8 baseline bypass budget, but existing bypasses pause while the active target guard holds. Identical receipt observation and native transfer paths; ordinary donor preemption can add recomputation and Host save/load work, all counted.',
        decision='Report full-request maxgap, TTFT, completion, throughput, target/donor/peer costs and actual extra copy/recovery work. No action stops reverse cells; no full-service benefit or merely relocated waiting narrows this exact candidate, without pressure/budget scans.',
        stop_rule='After the first exchange_once, continue reverse cells only for one recorded successful native donor preemption; shadow decisions never qualify. Failed, timeout or inconsistent action cells remain recorded and abort without retries.',
        policy_sha256=pins[0])
    p['fixed'].update(victim='Native tail except at most one explicitly selected forced donor preemption in exchange_once',
        service_quantum='Native; the exchanged target has Q1 protection ending at first new actual client receipt, with fail-open at16 schedule entries or earlier safety release',
        observations='Identical frozen stall8 receipts, compact capture, passive GC and exchange eligibility shadow in both arms; all host control costs included',
        recovery_actions='Frozen stall8 retains original8 successful-bypass budget and episode history. During a funded target protection window its existing guard pauses bypasses; no budget reset. Extra capacity exchange count is0/1, separate from those8 actions.',
        transfers='Native STORE/LOAD submission, flush dependency, ACK, buffers, references and physical release fences unchanged; ordinary donor preemption may add recomputation and Host save/load work, all counted')
    p['primary_readout'] = 'All256 arrivals per arm: maxgap mean/P99/max; mandatory TTFT/flow/throughput/drain, failed/timeout/unfinished and output/copy/repreemption/control costs. Target, donor and peers reported with full-service denominators; historical joint SLO diagnostic only.'
    extra = [*sources, ROOT/'run_group.py', ROOT/'run_cell.py', ROOT/'exchange_once.py', ROOT/'check_cpu.py', ROOT/'analyze.py', Path(__file__)]
    p['runner_sources_sha256'].update({str(path.relative_to(BASE)): sha(path) for path in set(extra)})
    p['prepared_command'] = p['python']+' -B '+remote+'/run_group.py --plan '+remote+'/plan-20261009-r01.json --wait-lock-seconds 1800'
    return p


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--session-dir'); args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    plan = prepare_plan(args.session_dir)
    plan['prepared_command'] = plan['prepared_command'].replace('/plan-20261009-r01.json ', '/'+args.output.name+' ')
    with args.output.open('x') as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(args.output)


if __name__ == '__main__': main()

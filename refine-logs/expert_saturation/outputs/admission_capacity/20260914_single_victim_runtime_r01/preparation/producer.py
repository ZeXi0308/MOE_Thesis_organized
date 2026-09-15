#!/usr/bin/env python3
"""Prepare the one-event victim diagnostic from the unchanged funding package."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1] / 'outputs/admission_capacity'
SOURCE = OUTPUTS / '20260914_funding_filter_comparison_r01'
REMOTE = '/root/autodl-tmp/moe-single-victim-runtime-20260914-r01'
GPU = 'GPU-bd5e9bb9-f98b-db5b-cc1c-5857c39f0bdc'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def write(p, value):
    with p.open('x') as f:
        f.write(value)


def dump(p, value):
    write(p, json.dumps(value, indent=2, allow_nan=False) + '\n')


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'expected one source occurrence: {old!r}')
    return text.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=OUTPUTS / '20260914_single_victim_runtime_r01')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    checks, contract = read(out / 'CPU_CHECKS.json'), read(out / 'event_contract.json')
    if not checks['status'].startswith('CPU_QUALIFIED'):
        raise ValueError('event qualification must finish before preparation')
    if (checks['event_contract_sha256'] != sha(out / 'event_contract.json')
            or not all(sha(HERE / n) == h for n, h in checks['qualified_source_sha256'].items())):
        raise ValueError('qualified event source or contract changed')
    old = read(SOURCE / 'preparation/preparation.json')
    src = SOURCE / 'preparation/pkg'
    if not all(sha(src / n) == h for n, h in old['files_sha256'].items()):
        raise ValueError('original funding package changed')
    if ast.dump(ast.parse((src / 'absence_rotation.py').read_text())) != ast.dump(
            ast.parse((HERE / 'absence_rotation.py').read_text())):
        raise ValueError('base selector behavior differs from original funding')
    pkg = out / 'preparation/pkg'
    shutil.copytree(src, pkg)
    (pkg / 'SHA256SUMS').unlink()
    for name in ['absence_rotation.py', 'rotation_native.py', 'single_victim_event.py']:
        shutil.copyfile(HERE / name, pkg / name)
    shutil.copyfile(out / 'event_contract.json', pkg / 'event_contract.json')
    probe = (pkg / 'run_probe.py').read_text()
    probe = replace(probe, 'from absence_rotation import RotationConfig',
                    'from absence_rotation import RotationConfig\nfrom single_victim_event import SingleVictimEvent')
    probe = replace(probe, '    return dict(specs[name])',
        "    for value in specs.values():\n        value['event_enabled'] = False\n"
        "    specs['single_victim'] = dict(specs['least_feasible'], event_enabled=True)\n"
        '    return dict(specs[name])')
    probe = replace(probe, "choices=['most_output','least_progress','least_feasible']",
                    "choices=['most_output','least_feasible','single_victim']")
    probe = replace(probe,
        "\n        filter_victims_by_funding=spec['filter_victims_by_funding'])",
        "\n        filter_victims_by_funding=spec['filter_victims_by_funding'],\n"
        "        event_enabled=spec['event_enabled'], event_contract_sha256=hashlib.sha256((root/'event_contract.json').read_bytes()).hexdigest())")
    probe = replace(probe, "        obligations = None",
        "        obligations = None\n"
        "        def event_request_states():\n"
        "            owned = scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks\n"
        "            return {rid: dict(computed_tokens=r.num_computed_tokens,\n"
        "                prompt_tokens=r.num_prompt_tokens, output_tokens=r.num_output_tokens,\n"
        "                max_tokens=r.max_tokens, num_tokens=r.num_tokens,\n"
        "                owned_blocks=len(owned.get(rid, ()))) for rid, r in scheduler.requests.items()}\n"
        "        event = SingleVictimEvent(contract=read(root/'event_contract.json'),\n"
        "            enabled=spec['event_enabled'], request_state_provider=event_request_states)")
    probe = replace(probe,
        "                filter_victims_by_funding=spec['filter_victims_by_funding'])",
        "                filter_victims_by_funding=spec['filter_victims_by_funding'],\n"
        "                tracker_factory=event.factory)")
    probe = replace(probe, "            dump(out/decision_name, decisions)",
        "            dump(out/decision_name, decisions)\n"
        "            dump(out/'single-victim-event.json', event.snapshot())")
    probe = replace(probe, "'recovery_service_components.py', 'restore_obligation.py']",
        "'recovery_service_components.py', 'restore_obligation.py', 'single_victim_event.py', 'event_contract.json']")
    (pkg / 'run_probe.py').write_text(probe)
    variants = ('most_output', 'least_feasible', 'single_victim')
    cells = [dict(label=f'victim-block{b}-{v}', block=b, variant=v,
                  victim_order='most_output' if v == 'most_output' else 'least_progress',
                  filter_victims_by_funding=v != 'most_output', event_enabled=v == 'single_victim')
             for b, order in ((0, variants), (1, variants[::-1])) for v in order]
    campaign = dict(read(src / 'campaign.json'), experiment_id=out.name, cells=cells,
        question='Does the selected single victim change avoid the modeled later recovery loss in native execution?',
        treatment='One present-state-matched victim replacement; subsequent least_feasible unchanged.',
        event_contract_sha256=sha(out / 'event_contract.json'),
        selection='Exploratory candidate selected after all 27 structural branches; not an online policy or holdout.',
        primary='per-request maximum ITL distribution versus complete-episode throughput',
        causal_check='Match the fixed pre-action contract and paired prefix; verify the actual forced victim and subsequent independently executed trajectory.',
        unmatched_rule='Retain completed unmatched runs and all request metrics; label diagnostic UNMATCHED, never substitute or discard.',
        retention='All six cells in frozen order; no restart, favorable replacement, or post-outcome event changes.',
        interpretation_rule='No general method, significance, or timing gain follows from model step counts. Keep most_output as strong reference.',
        expected_gpu_uuid=GPU)
    (pkg / 'campaign.json').write_text(json.dumps(campaign, indent=2) + '\n')
    prefix = (src / 'run.sh').read_text().split('run funding-block0-most_output most_output\n')[0]
    if not prefix.endswith('}\n'):
        raise ValueError('source run order boundary differs')
    (pkg / 'run.sh').write_text(prefix + ''.join(
        f"run {c['label']} {c['variant']}\n" for c in cells)
        + 'echo "CAMPAIGN_FINISHED $(date -u +%FT%TZ)"\n')
    allowed = {'absence_rotation.py', 'rotation_native.py', 'run_probe.py', 'campaign.json', 'run.sh', 'SHA256SUMS'}
    unchanged = {n: h for n, h in old['files_sha256'].items() if n not in allowed}
    if not all(sha(pkg / n) == h for n, h in unchanged.items()):
        raise ValueError('unrelated frozen source or workload changed')
    for p in pkg.glob('*.py'):
        ast.parse(p.read_text())
    subprocess.run(['bash', '-n', str(pkg / 'run.sh')], check=True)
    smoke = "import run_probe as r;a=r.arm_spec('least_feasible');b=r.arm_spec('single_victim');assert {k for k in a if a[k]!=b[k]}=={'event_enabled'};assert b['event_enabled'];print('ARM_PASS')"
    subprocess.run([sys.executable, '-B', '-c', smoke], cwd=pkg, check=True)
    files = {str(p.relative_to(pkg)): sha(p) for p in sorted(pkg.rglob('*')) if p.is_file()}
    (pkg / 'SHA256SUMS').write_text(''.join(f'{h}  {n}\n' for n, h in files.items()))
    files['SHA256SUMS'] = sha(pkg / 'SHA256SUMS')
    archive = out / 'preparation/execution.tar.gz'
    with tarfile.open(archive, 'w:gz') as t:
        t.add(pkg, arcname='pkg')
    metadata = dict(status='CPU_PREPARED_GPU_UNRUN', cells=cells,
        source_archive_sha256=old['archive_sha256'], source_bundle=str(SOURCE),
        event_contract_sha256=sha(out / 'event_contract.json'), cpu_checks_sha256=sha(out / 'CPU_CHECKS.json'),
        producer_sha256=sha(Path(__file__)), files_sha256=files, unchanged_sha256=unchanged,
        selector_ast_matches_frozen_source=True,
        archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
        expected_runtime_sources=old['expected_runtime_sources'], expected_gpu_uuid=GPU)
    dump(out / 'preparation/preparation.json', metadata)
    shutil.copyfile(Path(__file__), out / 'preparation/producer.py')
    driver = (SOURCE / 'execute.py').read_text()
    driver = replace(driver, '/root/autodl-tmp/moe-funding-filter-comparison-20260914-r01', REMOTE)
    driver = replace(driver, '/private/tmp/moe-funding-01a0953f.sock', '/private/tmp/moe-funding-01a0953f-r02.sock')
    driver = replace(driver, 'GPU-70fa1c0a-77d4-c14a-9daf-7e685874eef9', GPU)
    driver = replace(driver, "        assert check['gpu']==GPU and not check['processes'], 'wrong GPU or busy; no launch'",
        "        assert check['gpu']==GPU, 'wrong GPU; no launch'\n"
        "        if mode == 'run':\n            assert not check['processes'], 'busy; no launch'")
    driver = driver.replace('frozen funding-filter comparison package', 'single-victim diagnostic package')
    ast.parse(driver)
    write(out / 'execute.py', driver)
    print(json.dumps(dict(status=metadata['status'], cells=len(cells), archive_sha256=metadata['archive_sha256'])))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Freeze one heterogeneous-context comparison of victim funding filtering."""
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
SOURCE_NAMES = ('absence_rotation.py', 'rotation_native.py')
REMOTE = '/root/autodl-tmp/moe-funding-filter-comparison-20260914-r01'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def write(path, value):
    with path.open('x') as stream:
        stream.write(value)


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'expected one replacement target: {old!r}')
    return text.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-bundle', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    source = args.source_bundle.resolve()
    src = source / 'preparation/pkg'
    old_meta = json.loads((source / 'preparation/preparation.json').read_text())
    if old_meta['status'] != 'CPU_PREPARED_GPU_UNRUN':
        raise ValueError('source package was not the frozen context preparation')
    if not all(sha(src / name) == digest for name, digest in old_meta['files_sha256'].items()):
        raise ValueError('source frozen package hash mismatch')
    old_campaign = json.loads((src / 'campaign.json').read_text())
    old_order = [f'context-block{block}-{variant}' for block, variants in
                 ((0, ('native', 'most_output', 'least_progress')),
                  (1, ('least_progress', 'most_output', 'native')))
                 for variant in variants]
    if [cell['label'] for cell in old_campaign['cells']] != old_order:
        raise ValueError('source campaign differs from executed context calibration')
    config = json.loads((src / 'inputs_preparation/prepared/heterogeneous/config.json').read_text())
    if (config['requests'], config['prompt_lengths'], config['output_tokens'],
            old_campaign['actual_kv_blocks_required']) != (
            32, [2560 if i % 2 == 0 else 3072 for i in range(32)], 1024, 6656):
        raise ValueError('heterogeneous input or physical pool changed')
    qualification_path = source / 'funding_filter_cpu_r01.json'
    qualification = json.loads(qualification_path.read_text())
    if qualification['status'] != 'CPU_QUALIFIED':
        raise ValueError('funding filter source was not CPU qualified')
    for name in SOURCE_NAMES:
        key = name.removesuffix('.py') + '_sha256'
        if qualification['implementation'][key] != sha(HERE / name):
            raise ValueError(f'{name} differs from its CPU qualification')

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    pkg = out / 'preparation/pkg'
    shutil.copytree(src, pkg)
    (pkg / 'SHA256SUMS').unlink()
    for name in SOURCE_NAMES:
        shutil.copy2(HERE / name, pkg / name)

    probe = (pkg / 'run_probe.py').read_text()
    probe = replace(probe,
        'One context-calibration native/most-output/least-progress episode.',
        'One context-calibration most/least/funding-filter component comparison.')
    probe = replace(probe,
        "    specs['least_progress'] = dict(specs['most_output'], victim_order='least_progress')\n"
        "    return dict(specs[name])",
        "    specs['least_progress'] = dict(specs['most_output'], victim_order='least_progress')\n"
        "    for value in specs.values():\n"
        "        value['filter_victims_by_funding'] = False\n"
        "    specs['least_feasible'] = dict(specs['least_progress'], filter_victims_by_funding=True)\n"
        "    return dict(specs[name])")
    probe = replace(probe,
        "choices=['native','most_output','least_progress']",
        "choices=['most_output','least_progress','least_feasible']")
    probe = replace(probe,
        "        preemption_mode='native_recompute', rotation_config=vars(RotationConfig()))",
        "        preemption_mode='native_recompute', rotation_config=vars(RotationConfig()),\n"
        "        filter_victims_by_funding=spec['filter_victims_by_funding'])")
    probe = replace(probe,
        "                rotation_config=RotationConfig(**config['rotation_config']), victim_order=spec['victim_order'])",
        "                rotation_config=RotationConfig(**config['rotation_config']), victim_order=spec['victim_order'],\n"
        "                filter_victims_by_funding=spec['filter_victims_by_funding'])")
    if probe.count("filter_victims_by_funding=spec['filter_victims_by_funding']") != 2:
        raise ValueError('runner must record and apply exactly one funding-filter flag')
    (pkg / 'run_probe.py').write_text(probe)

    variants = ('most_output', 'least_progress', 'least_feasible')
    cells = [dict(label=f'funding-block{block}-{variant}', block=block, variant=variant,
                  victim_order='most_output' if variant == 'most_output' else 'least_progress',
                  filter_victims_by_funding=variant == 'least_feasible')
             for block, order in ((0, variants), (1, variants[::-1])) for variant in order]
    campaign = dict(old_campaign,
        experiment_id='20260914_funding_filter_comparison_r01', cells=cells,
        scope=old_campaign['scope'] + ' Only the least_feasible arm adds present-state victim funding filtering.',
        question='Does filtering the existing least-progress victim order by current complete-history funding improve the observed request tradeoff?',
        primary='per-request maximum ITL distribution versus complete-episode request throughput',
        secondary=['mean and per-request completion', 'ITL distribution', 'recompute positions',
                   'actual restore waiting', 'selected and applied rotation counts'],
        treatment='least_feasible equals least_progress except filter_victims_by_funding=true',
        strongest_simple_baseline='most_output',
        interpretation_rule='If least_feasible improves only over least_progress and does not exceed most_output, retain it as a simple-baseline repair without independent contribution.',
        retention='retain all six cells in frozen order; no post-result tuning or cell replacement')
    (pkg / 'campaign.json').unlink()
    dump(pkg / 'campaign.json', campaign)

    old_run = (src / 'run.sh').read_text()
    marker = 'run context-block0-native native'
    if old_run.count(marker) != 1:
        raise ValueError('source run order marker changed')
    prefix = old_run.split(marker)[0]
    (pkg / 'run.sh').write_text(prefix + ''.join(
        f"run {cell['label']} {cell['variant']}\n" for cell in cells)
        + 'echo "CAMPAIGN_FINISHED $(date -u +%FT%TZ)"\n')

    for path in pkg.glob('*.py'):
        ast.parse(path.read_text())
    subprocess.run(['bash', '-n', str(pkg / 'run.sh')], check=True)
    smoke = ("import run_probe as r;"
        "a=r.arm_spec('most_output');b=r.arm_spec('least_progress');c=r.arm_spec('least_feasible');"
        "assert not a['filter_victims_by_funding'] and not b['filter_victims_by_funding'];"
        "assert c['filter_victims_by_funding'] and c['victim_order']==b['victim_order']=='least_progress';"
        "assert {k for k in c if c[k]!=b[k]}=={'filter_victims_by_funding'};print('ARM_PASS')")
    subprocess.run([sys.executable, '-B', '-c', smoke], cwd=pkg, check=True)
    tests = subprocess.run([sys.executable, '-B', '-m', 'unittest', '-q', 'test_absence_rotation.py'],
                           cwd=HERE, text=True, capture_output=True)
    if tests.returncode or 'Ran 27 tests' not in tests.stderr or 'OK' not in tests.stderr:
        raise ValueError(f'selector tests failed: {tests.stdout}{tests.stderr}')
    native = subprocess.run([sys.executable, '-B', 'verify_rotation_native.py'], cwd=HERE,
                            text=True, capture_output=True, check=True)
    if json.loads(native.stdout)['status'] != 'PASS':
        raise ValueError('native CPU allocator fixture failed')

    allowed = set(SOURCE_NAMES) | {'run_probe.py', 'campaign.json', 'run.sh', 'SHA256SUMS'}
    unchanged = {}
    for path in sorted(src.rglob('*')):
        if path.is_file() and str(path.relative_to(src)) not in allowed:
            relative = str(path.relative_to(src)); unchanged[relative] = sha(path)
            if sha(pkg / relative) != unchanged[relative]:
                raise ValueError(f'unexpected copied-file change: {relative}')
    input_hashes = {str(path.relative_to(pkg)): sha(path) for path in
                    sorted((pkg / 'inputs_preparation').rglob('*')) if path.is_file()}
    checks = dict(status='CPU_QUALIFIED_GPU_UNRUN', python_ast_pass=True,
        shell_syntax_pass=True, selector_tests=dict(status='PASS', count=27),
        native_allocator_fixture='PASS', arm_difference_only_filter_flag=True,
        runner_records_explicit_filter_flag=True, decision_flag_only_when_enabled=True,
        context=dict(requests=32, prompt_lengths=config['prompt_lengths'], output_tokens=1024,
                     usable_blocks=6656, fixed_kv_bytes=old_campaign['fixed_kv_bytes']),
        order=[cell['variant'] for cell in cells], input_sha256=input_hashes,
        source_funding_qualification_sha256=sha(qualification_path), unchanged_files_sha256=unchanged)
    dump(out / 'CPU_CHECKS.json', checks)

    files = {str(path.relative_to(pkg)): sha(path) for path in sorted(pkg.rglob('*')) if path.is_file()}
    (pkg / 'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name, digest in files.items()))
    files['SHA256SUMS'] = sha(pkg / 'SHA256SUMS')
    archive = out / 'preparation/execution.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(pkg, arcname='pkg')
    metadata = dict(status='CPU_PREPARED_GPU_UNRUN', source_bundle=str(source),
        source_package_archive_sha256=old_meta['archive_sha256'],
        source_funding_qualification_sha256=sha(qualification_path),
        producer_sha256=sha(Path(__file__)), files_sha256=files,
        archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
        cells=cells, expected_runtime_sources=old_meta['expected_runtime_sources'])
    dump(out / 'preparation/preparation.json', metadata)

    driver = (source / 'execute.py').read_text()
    driver = replace(driver, '/root/autodl-tmp/moe-context-victim-calibration-20260914-r01', REMOTE)
    driver = driver.replace('Stage/run the unchanged component package',
                            'Stage/run the frozen funding-filter comparison package')
    ast.parse(driver)
    write(out / 'execute.py', driver)
    report = f'''# Funding-filter component comparison\n\nStatus: `CPU_PREPARED_GPU_UNRUN`. Preparation is not a measured result.\n\nThis six-cell comparison reuses the frozen heterogeneous cohort, OLMoE/vLLM backend, 6,656-block pool, capture path, arrival trace, thresholds and native recovery semantics from the context calibration. No workload, pressure point or threshold was selected after observing this package.\n\nThe frozen order is `most_output / least_progress / least_feasible / least_feasible / least_progress / most_output`. `least_feasible` is the existing least-progress selector with one present-state component enabled: after the original target and victim guards, it removes victims for which `free_blocks + released_blocks < target_required_blocks`. The target, cooldown, progress, residency and absence guards are unchanged. `most_output` and `least_progress` explicitly set the flag false.\n\nThe primary view is the per-request maximum-ITL distribution against complete-episode request throughput. Secondary views are completion distribution, overall ITL distribution, recomputed positions, restore waiting and actual selected/applied rotations. Every complete or unfavorable cell is retained.\n\nIf filtering improves only over `least_progress` and does not exceed `most_output`, it is a simple-baseline repair and not an independent contribution. CPU qualification proves current-state selection and native fixture behavior only; it does not predict execution time, output equality or an alternate future.\n\nPackage archive SHA256: `{metadata['archive_sha256']}`. Remote path reserved by the copied execute entry: `{REMOTE}`. GPU execution remains `UNRUN`.\n'''
    write(out / 'REPORT.md', report)
    print(json.dumps(dict(status=metadata['status'], cells=len(cells),
                          archive_sha256=metadata['archive_sha256'], output_dir=str(out))))


if __name__ == '__main__':
    main()

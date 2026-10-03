#!/usr/bin/env python3
"""Prepare one default-off native-recovery execution comparison; never launch."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from prepare_single_victim_runtime import sha, read, dump, replace
from recovery_execution_model_adapter import load_simulator

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1] / 'outputs/admission_capacity'
SOURCE = OUTPUTS / '20260914_funding_filter_comparison_r01'
GPU = 'GPU-4015b79d-bed6-3a4b-9d2b-0c17be96d0e5'
VARIANTS = ('most_output', 'least_feasible', 'least_feasible_native_guard')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path,
                        default=OUTPUTS / '20260915_recovery_execution_share_r01')
    args = parser.parse_args(); bundle = args.bundle.resolve()
    prep, qualification = bundle / 'gpu_preparation', bundle / 'native_integration'
    if prep.exists():
        raise FileExistsError('refuse to replace an existing prepared package')
    fixture = read(qualification / 'native_fixture_cli.json')
    if fixture['status'] != 'PASS':
        raise ValueError('native schedule CPU qualification is not PASS')
    qualified_adapter = qualification / 'rotation_native.py'
    if (fixture['sha256'][str(qualified_adapter)] != sha(qualified_adapter)
            or sha(qualified_adapter) != sha(HERE / 'rotation_native.py')):
        raise ValueError('current native adapter differs from the CPU-qualified source')
    for name in ('verify_native_recovery_execution.py', 'recovery_execution_share.py'):
        if fixture['sha256'][str(HERE / name)] != sha(HERE / name):
            raise ValueError('CPU qualification dependency changed: ' + name)
    model = read(qualification / 'model.json')
    if model['status'] != 'CPU_RESOURCE_EXECUTION_COMPOSITION' or len(model['rows']) != 2:
        raise ValueError('resource-model qualification missing')
    model_source = qualification / 'resource_model_source.py'
    for row in model['rows']:
        _, current = load_simulator(model_source, funding_filter=row['source']['funding_filter'])
        if (not row['default_exact'] or row['source']['source_sha256'] != sha(model_source)
                or row['source']['adapted_ast_sha256'] != current['adapted_ast_sha256']):
            raise ValueError('qualified resource model or integration changed')
    original = read(SOURCE / 'preparation/preparation.json'); src = SOURCE / 'preparation/pkg'
    if not all(sha(src / name) == digest for name, digest in original['files_sha256'].items()):
        raise ValueError('original funding package changed')
    if ast.dump(ast.parse((src / 'absence_rotation.py').read_text())) != ast.dump(
            ast.parse((HERE / 'absence_rotation.py').read_text())):
        raise ValueError('upstream selector behavior changed')
    pkg = prep / 'pkg'; shutil.copytree(src, pkg); (pkg / 'SHA256SUMS').unlink()
    for name in ('absence_rotation.py', 'rotation_native.py'):
        shutil.copyfile(HERE / name, pkg / name)
    probe = (pkg / 'run_probe.py').read_text()
    probe = replace(probe, 'One context-calibration most/least/funding-filter component comparison.',
                    'One matched native-recovery execution comparison with a strong most baseline.')
    start, end = probe.index('def arm_spec(name):'), probe.index('\n\ndef main():')
    probe = probe[:start] + '''def arm_spec(name):
    if name not in ('most_output', 'least_feasible', 'least_feasible_native_guard'):
        raise ValueError('unsupported execution comparison arm')
    return dict(adapter='rotation', policy_family='strong_simple_rotation',
        completion_policy='rotate', victim_order='most_output' if name == 'most_output' else 'least_progress',
        boost=False, packing='native', complete_restores=False, reserve_ready_tokens=False,
        filter_victims_by_funding=name != 'most_output',
        protect_native_recovery=name == 'least_feasible_native_guard')
''' + probe[end:]
    probe = replace(probe, "choices=['most_output','least_progress','least_feasible']",
                    "choices=['most_output','least_feasible','least_feasible_native_guard']")
    probe = replace(probe, "\n        filter_victims_by_funding=spec['filter_victims_by_funding'])",
        "\n        filter_victims_by_funding=spec['filter_victims_by_funding'],\n"
        "        protect_native_recovery=spec['protect_native_recovery'],\n"
        "        primary_max_started_request_itl_s=3.0,\n"
        "        primary_objective='mean completion of all arrivals, conditional on complete episode and every started request max ITL <= 3.0 s')")
    probe = replace(probe, "                filter_victims_by_funding=spec['filter_victims_by_funding'])",
        "                filter_victims_by_funding=spec['filter_victims_by_funding'],\n"
        "                protect_native_recovery=spec['protect_native_recovery'])")
    # Preserve a returned partial/full raw before any cleanup can raise.
    start = probe.index("        try:\n            print('PHASE MEASUREMENT_BEGIN'")
    end = probe.index("        dump(out/'memory-after.json'", start)
    probe = probe[:start] + '''        raw = None
        try:
            print('PHASE MEASUREMENT_BEGIN', flush=True)
            raw = capture_with_memory(engine, capture_episode, workload, config,
                allow_preemption=True,
                regime='steady', arrival_scale=1.0, run_id='measured', max_seconds=120)
            print('PHASE MEASUREMENT_END', flush=True)
        finally:
            try:
                if raw is not None:
                    raw['rotation_exposed'] = any(d.get('forced_preempted') for d in decisions)
                    raw['protect_native_recovery'] = spec['protect_native_recovery']
                    raw['native_recovery_started'] = [dict(step=d['step'], schedule_status=d['status'],
                        **d['native_recovery_started']) for d in decisions if d.get('native_recovery_started')]
                    raw['calibration_role'] = 'retain every no-op, failure and unfavorable episode; no action-based substitution'
                    raw['gpu_before'] = before
                    dump(out/'raw.json', raw)
            finally:
                try:
                    dump(out/decision_name, decisions)
                finally:
                    uninstall()
''' + probe[end:]
    probe = replace(probe, 'def gpu_state():\n',
        "def gpu_state():\n"
        f"    expected = {GPU!r}\n"
        "    if os.environ.get('CUDA_VISIBLE_DEVICES') != expected:\n"
        "        raise RuntimeError('requires the frozen physical GPU UUID')\n"
        "    actual = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()\n"
        "    if actual != expected:\n"
        "        raise RuntimeError('physical GPU0 UUID differs')\n")
    probe = replace(probe, "return dict(compute_processes=processes, device=", 
        "return dict(selected_gpu_uuid=expected, cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], compute_processes=processes, device=")
    (pkg / 'run_probe.py').write_text(probe)
    cells = [dict(label=f'recovery-block{block}-{variant}', block=block, variant=variant,
                  victim_order='most_output' if variant == 'most_output' else 'least_progress',
                  filter_victims_by_funding=variant != 'most_output',
                  protect_native_recovery=variant == 'least_feasible_native_guard')
             for block, order in ((0, VARIANTS), (1, VARIANTS[::-1])) for variant in order]
    campaign = dict(read(src / 'campaign.json'), experiment_id=bundle.name, cells=cells,
        scope='Exploratory reuse of the same heterogeneous 32 documents; not fresh holdout or a victim search.',
        question='Does extending existing execution protection to an actually scheduled native recovery improve full requests under the fixed pause requirement?',
        treatment='least_feasible_native_guard differs from least_feasible only by protect_native_recovery=True; same ready-first sharing and upper-layer selection.',
        primary='Mean completion time of ALL arrivals, eligible only if every planned request completes and every started request has max engine-return ITL <= 3.0 s.',
        primary_max_started_request_itl_s=3.0,
        infeasible_rule='If neither arm is eligible, no feasible winner; if only one is eligible report feasibility only. Do not replace the primary with throughput.',
        secondary=['TTFT', 'mean TPOT', 'per-request maximum ITL and completion', 'episode wall and throughput',
                   'recompute', 'recovery waiting and first new output', 'improved and harmed requests'],
        no_action_rule='Retain complete no-op cells and report measured action counts; never tune or replace cells.',
        interpretation_rule='Most is the strong reference; component improvement alone is not a method or novelty claim.',
        retention='All six cells and failure logs remain in frozen order; no restart or favorable replacement.',
        expected_gpu_uuid=GPU, expected_physical_gpu_index=0,
        gpu_isolation='Whole-host process check and common flock; CUDA visible device fixed to physical GPU0 UUID.')
    (pkg / 'campaign.json').write_text(json.dumps(campaign, indent=2) + '\n')
    prefix = (src / 'run.sh').read_text().split('run funding-block0-most_output most_output\n')[0]
    if not prefix.endswith('}\n'):
        raise ValueError('frozen run-order boundary differs')
    prefix = replace(prefix, 'sha256sum -c SHA256SUMS\n',
        'sha256sum -c SHA256SUMS\n' + f'export CUDA_VISIBLE_DEVICES={GPU}\n'
        + '[ "$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)" = "$CUDA_VISIBLE_DEVICES" ] || exit 94\n')
    (pkg / 'run.sh').write_text(prefix + ''.join(f"run {c['label']} {c['variant']}\n" for c in cells)
        + 'echo "CAMPAIGN_FINISHED $(date -u +%FT%TZ)"\n')
    changed = {'absence_rotation.py', 'rotation_native.py', 'run_probe.py', 'campaign.json', 'run.sh', 'SHA256SUMS'}
    unchanged = {name: digest for name, digest in original['files_sha256'].items() if name not in changed}
    if not all(sha(pkg / name) == digest for name, digest in unchanged.items()):
        raise ValueError('unchanged workload/runtime inputs differ')
    for path in pkg.glob('*.py'): ast.parse(path.read_text())
    subprocess.run(['bash', '-n', str(pkg / 'run.sh')], check=True)
    smoke = "import run_probe as r; a=r.arm_spec('least_feasible'); b=r.arm_spec('least_feasible_native_guard'); assert {k for k in a if a[k]!=b[k]}=={'protect_native_recovery'}; assert b['protect_native_recovery'] and not r.arm_spec('most_output')['protect_native_recovery']; print('ARM_PASS')"
    subprocess.run([sys.executable, '-B', '-c', smoke], cwd=pkg, check=True)
    files = {str(path.relative_to(pkg)): sha(path) for path in sorted(pkg.rglob('*')) if path.is_file()}
    (pkg / 'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name, digest in files.items()))
    files['SHA256SUMS'] = sha(pkg / 'SHA256SUMS')
    archive = prep / 'execution.tar.gz'
    with tarfile.open(archive, 'w:gz') as handle: handle.add(pkg, arcname='pkg')
    metadata = dict(status='CPU_PREPARED_GPU_UNRUN', cells=cells, source_bundle=str(SOURCE),
        source_archive_sha256=original['archive_sha256'], files_sha256=files, unchanged_sha256=unchanged,
        archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
        expected_runtime_sources=original['expected_runtime_sources'], expected_gpu_uuid=GPU,
        expected_physical_gpu_index=0, primary_max_started_request_itl_s=3.0,
        cpu_qualification_sha256={name: sha(qualification / name) for name in
            ('native_fixture_cli.json', 'rotation_native.py', 'model.json', 'resource_model_source.py')},
        producer_sha256=sha(Path(__file__)), helper_sha256=sha(HERE / 'prepare_single_victim_runtime.py'),
        checks=dict(variant_difference='protect_native_recovery only', selector_ast_unchanged=True,
                    package_python_ast=True, bash_syntax=True, returned_raw_saved_before_cleanup=True),
        gpu_execution='UNRUN; no upload, driver or GPU launch performed by this producer')
    dump(prep / 'preparation.json', metadata)
    shutil.copyfile(Path(__file__), prep / 'producer.py')
    print(json.dumps(dict(status=metadata['status'], cells=len(cells), files=len(files),
                         archive_sha256=metadata['archive_sha256'])))


if __name__ == '__main__': main()

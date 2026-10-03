"""CPU checks for the frozen H128 performance package; never initializes CUDA."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
pkg = root / 'pkg'
parent = root.parent
env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}


def run(*args, expected=0):
    result = subprocess.run(args, env=env, text=True, capture_output=True)
    if result.returncode != expected:
        raise RuntimeError(f'{args}: expected exit {expected}, got {result.returncode}: {result.stderr}')


run(sys.executable, '-B', str(root / 'verify_package.py'))
run('bash', '-n', str(pkg / 'run.sh'))
for path in pkg.rglob('*.py'):
    compile(path.read_bytes(), str(path), 'exec')

for name in ('inputs/config.json', 'inputs/workload.json', 'inputs/INPUT_STATS.json', 'safe_static.py'):
    if (pkg / name).read_bytes() != (parent / 'candidate_h1' / 'pkg' / name).read_bytes():
        raise RuntimeError(f'H128 input or bound drift: {name}')
if (pkg / 'ltr_style_native.py').read_bytes() != (
        parent / 'candidate_g64_perf_guard_t200_r01' / 'pkg' / 'ltr_style_native.py').read_bytes():
    raise RuntimeError('Corrected in-flight guard source drift: ltr_style_native.py')
for name in ('ltr_style_selected.py', 'recovery_service_components.py'):
    if (pkg / name).read_bytes() != (parent / 'candidate_ltr_r02_env' / 'pkg' / name).read_bytes():
        raise RuntimeError(f'LTR policy drift: {name}')
for name in ('staged_store_rotation.py', 'absence_rotation.py'):
    if (pkg / name).read_bytes() != (parent / 'candidate_h1' / 'pkg' / name).read_bytes():
        raise RuntimeError(f'eager policy drift: {name}')

spec = importlib.util.spec_from_file_location('h128_perf_ltr_runner', pkg / 'run_ltr_style.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
config, workload = module.load_inputs(pkg / 'inputs')
lengths = [len(x) for x in workload['actual_prompt_token_ids']]
if ((len(lengths), min(lengths), max(lengths)) != (128, 460, 3064)
        or workload['arrival_traces_s']['steady'] != [i * .2 for i in range(128)]
        or config['model']['revision'] != '6d84c48581ece794365f2b8e9cfb043c68ade9c5'):
    raise RuntimeError('H128 workload identity drift')

out = '/tmp/h128-perf-invalid-cli-output'
run('bash', str(pkg / 'run.sh'), 'unknown', out, expected=64)
run('bash', str(pkg / 'run.sh'), 'ltr_t30_q10', out, expected=1)
env['HPERF_SELECTED_LTR_ARM'] = 'ltr_t200_q1'
run('bash', str(pkg / 'run.sh'), 'ltr_t30_q10', out, expected=64)
env.pop('HPERF_SELECTED_LTR_ARM')
run(sys.executable, '-B', str(pkg / 'run_recovery_cadence.py'), '--inputs', str(pkg / 'inputs'),
    '--warmup-inputs', str(pkg / 'warmups'), '--variant', 'current',
    '--measurement-mode', 'performance', '--output-dir', out, expected=2)
run(sys.executable, '-B', str(pkg / 'run_recovery_cadence.py'), '--inputs', str(pkg / 'inputs'),
    '--warmup-inputs', str(pkg / 'warmups'), '--variant', 'eager',
    '--measurement-mode', 'diagnostic', '--output-dir', out, expected=2)
run(sys.executable, '-B', str(pkg / 'run_ltr_style.py'), '--inputs', str(pkg / 'inputs'),
    '--warmup-inputs', str(pkg / 'warmups'), '--ltr-threshold', '201', '--ltr-quantum', '10',
    '--measurement-mode', 'performance', '--output-dir', out, expected=2)
run(sys.executable, '-B', str(pkg / 'run_ltr_style.py'), '--inputs', str(pkg / 'inputs'),
    '--warmup-inputs', str(pkg / 'warmups'), '--ltr-threshold', '30', '--ltr-quantum', '10',
    '--measurement-mode', 'diagnostic', '--output-dir', out, expected=2)
print(json.dumps({'status': 'PASS_CPU_ONLY', 'manifest_entries': len(json.loads((root / 'manifest.json').read_text())),
                  'requests': len(lengths), 'arms': 6, 'gpu_initialized': False}, sort_keys=True))

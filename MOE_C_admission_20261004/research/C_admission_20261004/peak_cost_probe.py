"""Bounded CPU-only comparison of exactly equivalent MC peak arithmetic.

Uses four saved first-eligible MC snapshots, not a reconstructed full serving
trajectory. Timing on this machine is not GPU-service or target-host evidence.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time


def load_function(source):
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == 'peak_envelope')
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<peak_envelope>', 'exec'), namespace)
    return namespace['peak_envelope']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-commit', default='df7816e74840ac72712f73fd578042ea33230203')
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.iterations <= 2000:
        parser.error('Output must be new; iterations must be between 1 and 2000.')
    root = Path(__file__).resolve().parent
    relative = 'research/C_admission_20261004/mc_budget.py'
    reference = subprocess.check_output(
        ['git', 'show', f'{args.reference_commit}:{relative}'], cwd=root, text=True)
    current = (root/'mc_budget.py').read_text()
    functions = {'reference': load_function(reference), 'optimized': load_function(current)}
    report = dict(status='RUNNING', evidence='CPU_FUNCTION_MICROBENCHMARK_NOT_SERVING',
        scope='Four saved first-eligible states only; no rejection-state sample or full-trajectory replay.',
        platform=platform.platform(), processor=platform.processor(), python=platform.python_version(),
        reference_commit=args.reference_commit,
        current_source_sha256=hashlib.sha256(current.encode()).hexdigest(),
        iterations=args.iterations, reverse=args.reverse, samples=[])
    paths = [root/'runs/westd-20261008'/group/cell/'admission.json'
             for group, cells in (
                 ('mc-budget-r01', ('probe-01-mcbudget', 'probe-02-mcbudget')),
                 ('mc-budget-ablation-r02', ('probe-01-mccapped', 'probe-02-mccapped')))
             for cell in cells]
    started = time.perf_counter()
    try:
        for path in paths:
            first = json.loads(path.read_text())['mc_budget']['first_eligible']
            inputs = (first['old_rows'], first['new_prompt_tokens'], first['new_max_tokens'])
            expected = first['envelope']
            assert expected is not None
            for fn in functions.values():
                assert fn(*inputs) == expected, 'Saved peak/checkpoints differ.'
                for _ in range(20):
                    fn(*inputs)
            sample = dict(source=str(path.relative_to(root)), requests=len(inputs[0]),
                checkpoints=len(expected['checkpoints']), exact_saved_result_match=True, timings=[])
            report['samples'].append(sample)
            for repeat in range(4):
                order = ('reference', 'optimized')
                if (repeat % 2 == 1) != args.reverse:
                    order = order[::-1]
                for name in order:
                    if time.perf_counter()-started > 30:
                        raise TimeoutError('CPU benchmark exceeded its 30-second cap.')
                    wall, cpu = time.perf_counter_ns(), time.process_time_ns()
                    for _ in range(args.iterations):
                        result = functions[name](*inputs)
                    cpu, wall = time.process_time_ns()-cpu, time.perf_counter_ns()-wall
                    assert result == expected
                    sample['timings'].append(dict(repeat=repeat, function=name,
                        wall_us_per_call=wall/args.iterations/1000,
                        cpu_us_per_call=cpu/args.iterations/1000))
            medians = {name: statistics.median(t['wall_us_per_call'] for t in sample['timings']
                       if t['function'] == name) for name in functions}
            sample.update(median_wall_us_per_call=medians,
                          median_speedup=medians['reference']/medians['optimized'])
        report['status'] = 'COMPLETE'
    except Exception as error:
        report.update(status='FAILED', error=repr(error))
    report['elapsed_s'] = time.perf_counter()-started
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(status=report['status'], elapsed_s=report['elapsed_s'],
        samples=[{k: v for k, v in sample.items() if k != 'timings'}
                 for sample in report['samples']])))
    return 0 if report['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())

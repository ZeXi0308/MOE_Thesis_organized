"""Six F/X/logical-X processes on one fresh cohort; unchanged compile setup and all costs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import driver_support as support

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def inventory(path):
    return {str(p.relative_to(path)): dict(bytes=p.stat().st_size,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(path.rglob('*')) if p.is_file()}


def check_inputs():
    hashes = read(ROOT / 'input_hashes.json')
    required = {'run_attempt.py', 'protocol.json', 'run_cells.json', 'driver_support.py',
                'sources.json', 'prepared/workload.json', 'allocations/uniform.json',
                'instrumentation/run_covered.py', 'instrumentation/compile_domain.py',
                'instrumentation/host_probe.py', 'instrumentation/logical_execution.py',
                'analyze_covered.py', 'finite_metrics.py', 'analyze_logical_performance.py'}
    if not required <= hashes.keys():
        raise RuntimeError('incomplete frozen input manifest')
    for name, expected in hashes.items():
        if support.digest(ROOT / name) != expected:
            raise RuntimeError('frozen input changed: ' + name)
    for name, expected in read(ROOT / 'sources.json').items():
        if support.digest(ROOT / 'source' / name) != expected:
            raise RuntimeError('frozen runtime changed: ' + name)
    return dict(files=len(hashes), manifest_sha256=support.digest(ROOT / 'input_hashes.json'))


def command(plan, protocol):
    cmd = support.command(plan, protocol)
    index = cmd.index(str(ROOT / 'source' / 'run_shared_pool_pager.py'))
    cmd[index] = str(ROOT / 'instrumentation' / 'run_covered.py')
    if plan['logical_alignment']:
        cmd.append('--logical-alignment')
    return cmd


def cell_environment(plan, protocol):
    base = ROOT / 'compiler_caches' / plan['label']
    return dict(support.environment(protocol), TRITON_CACHE_DIR=str(base / 'cache'),
                TRITON_DUMP_DIR=str(base / 'dump'), TRITON_OVERRIDE_DIR=str(base / 'override'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    protocol, plans = read(ROOT / 'protocol.json'), read(ROOT / 'run_cells.json')
    checked = check_inputs()
    if args.dry_run:
        print(json.dumps(dict(status='DRY_RUN_CPU_ONLY', checked=checked,
              lock=protocol['gpu_group_lock'],
              cells=[dict(**p, command=command(p, protocol), env=cell_environment(p, protocol)) for p in plans]), indent=2))
        return
    out = ROOT / 'results'
    out.mkdir(exist_ok=False)
    record = dict(status='CHECKING', started_unix_s=time.time(), cells=[],
                  gpu_group_lock=protocol['gpu_group_lock'])

    def save():
        temporary = out / 'execution.tmp'
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
        temporary.replace(out / 'execution.json')

    save()
    cell = None
    lock = None
    try:
        lock = open(protocol['gpu_group_lock'], 'a+')
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        record.update(lock_acquired_unix_s=time.time(), lock_owner_pid=os.getpid())
        lock.seek(0)
        lock.truncate()
        lock.write(json.dumps(dict(pid=os.getpid(), directory=str(ROOT), started=time.time())) + '\n')
        lock.flush()
        record['status'] = 'RUNNING'
        for plan in plans:
            env = cell_environment(plan, protocol)
            cell = dict(**plan, status='CHECKING', command=command(plan, protocol), env=env,
                        process_wall_s=None, returncode=None)
            record['cells'].append(cell)
            cell['bundle_check'] = check_inputs()
            cell['external_inputs'] = support.external_inputs(protocol)
            cache = Path(env['TRITON_CACHE_DIR'])
            for name in ('TRITON_CACHE_DIR', 'TRITON_DUMP_DIR', 'TRITON_OVERRIDE_DIR'):
                Path(env[name]).mkdir(parents=True, exist_ok=False)
            cell['cache_before'] = inventory(cache)
            if cell['cache_before']:
                raise RuntimeError('private cache must begin empty in every fresh engine')
            cell['gpu_before'] = support.gpu_snapshot()
            save()
            support.require_empty_gpu(cell['gpu_before'], protocol['expected_gpu_uuid'])
            cell.update(status='RUNNING', started_unix_s=time.time())
            save()
            # Only these declared Triton settings apply; no inherited force-compile or override.
            process_env = {k: v for k, v in os.environ.items() if not k.startswith('TRITON_')}
            process_env.update(env)
            with (out / (plan['label'] + '.log')).open('x') as log:
                cell['process_start_perf_ns'] = time.perf_counter_ns()
                try:
                    result = subprocess.run(cell['command'], stdout=log, stderr=subprocess.STDOUT,
                                            env=process_env)
                    cell['returncode'] = result.returncode
                finally:
                    cell['process_end_perf_ns'] = time.perf_counter_ns()
                    cell['process_wall_s'] = (cell['process_end_perf_ns'] - cell['process_start_perf_ns']) / 1e9
            cell.update(finished_unix_s=time.time(), cache_after=inventory(cache))
            save()
            if cell['returncode']:
                raise RuntimeError(plan['label'] + ' failed; no retry')
            terminal = read(out / plan['label'] / 'status.json')
            if (terminal['status'], terminal['requests_completed'], terminal['generated_tokens']) != ('COMPLETE', 16, 512):
                raise RuntimeError('incomplete cohort')
            if not cell['cache_after']:
                raise RuntimeError('no private Triton cache entries produced')
            cell.update(status='COMPLETE', terminal=terminal)
            setup = read(out / plan['label'] / 'compile_domain.json')
            observed = read(out / plan['label'] / 'jit_probe.json')
            cell['measurement_compiler_pipeline_events'] = [e['event_id'] for e in observed['events']
                if e['kind'] == 'compiler_call' and e['context'].get('phase') == 'measurement' and e.get('cache_hit') is False]
            cell['coverage_valid'] = (setup['status'] == 'COMPLETE_COMPILE_AND_LOAD_ONLY' and observed['status'] == 'COMPLETE'
                and not cell['measurement_compiler_pipeline_events'])
            save()
            if not cell['coverage_valid']:
                raise RuntimeError('compile-domain coverage unfulfilled; measured cell retained, no retry')
        record['status'] = 'COMPLETE'
    except BaseException as error:
        record.update(status='STOPPED', error=repr(error))
        if cell is not None and cell['status'] != 'COMPLETE':
            cell.update(status='STOPPED', error=repr(error))
        raise
    finally:
        record['finished_unix_s'] = time.time()
        save()
        if lock is not None:
            lock.close()


if __name__ == '__main__':
    main()

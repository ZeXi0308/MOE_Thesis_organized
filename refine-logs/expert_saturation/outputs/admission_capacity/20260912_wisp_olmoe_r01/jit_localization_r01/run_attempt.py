"""Two ordinary-U compiler-localization processes; whole-group advisory lock."""
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
                'instrumentation/run_jit_probe.py', 'analyze_jit.py', 'finite_metrics.py'}
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
    index = cmd.index(str(ROOT / 'source' / 'run_native_pager.py'))
    cmd[index] = str(ROOT / 'instrumentation' / 'run_jit_probe.py')
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    protocol, plans = read(ROOT / 'protocol.json'), read(ROOT / 'run_cells.json')
    checked = check_inputs()
    cache = ROOT / 'private_triton_cache'
    env = dict(support.environment(protocol), TRITON_CACHE_DIR=str(cache),
               TRITON_DUMP_DIR=str(ROOT / 'private_triton_dump'),
               TRITON_OVERRIDE_DIR=str(ROOT / 'private_triton_override'))
    if args.dry_run:
        print(json.dumps(dict(status='DRY_RUN_CPU_ONLY', checked=checked,
              lock=protocol['gpu_group_lock'], env=env,
              cells=[dict(**p, command=command(p, protocol)) for p in plans]), indent=2))
        return
    out = ROOT / 'results'
    out.mkdir(exist_ok=False)
    record = dict(status='CHECKING', started_unix_s=time.time(), cells=[],
                  gpu_group_lock=protocol['gpu_group_lock'], environment=env)

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
        cache.mkdir(exist_ok=False)
        (ROOT / 'private_triton_dump').mkdir(exist_ok=False)
        (ROOT / 'private_triton_override').mkdir(exist_ok=False)
        record['status'] = 'RUNNING'
        previous = {}
        for plan in plans:
            cell = dict(**plan, status='CHECKING', command=command(plan, protocol),
                        process_wall_s=None, returncode=None)
            record['cells'].append(cell)
            cell['bundle_check'] = check_inputs()
            cell['external_inputs'] = support.external_inputs(protocol)
            cell['cache_before'] = inventory(cache)
            if cell['cache_before'] != previous:
                raise RuntimeError('private cache changed outside prior recorded process')
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
            previous = cell['cache_after']
            save()
            if cell['returncode']:
                raise RuntimeError(plan['label'] + ' failed; no retry')
            terminal = read(out / plan['label'] / 'status.json')
            if (terminal['status'], terminal['requests_completed'], terminal['generated_tokens']) != ('COMPLETE', 16, 512):
                raise RuntimeError('incomplete cohort')
            if not previous:
                raise RuntimeError('no private Triton cache entries produced')
            cell.update(status='COMPLETE', terminal=terminal)
            save()
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

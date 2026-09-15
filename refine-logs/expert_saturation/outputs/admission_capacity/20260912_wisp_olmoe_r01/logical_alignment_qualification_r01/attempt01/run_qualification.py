"""One numerical qualification process, shared GPU lock, no retry or performance claim."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import driver_support as support

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def check_inputs():
    hashes = read(ROOT / 'input_hashes.json')
    required = {'run_qualification.py', 'driver_support.py', 'protocol.json',
                'prepared/workload.json', 'sources.json',
                'instrumentation/run_logical_alignment.py'}
    sources = read(ROOT / 'sources.json')
    required.update('source/' + n for n in sources)
    if len(sources) != 11 or not required <= hashes.keys():
        raise RuntimeError('incomplete frozen qualification inputs')
    for name, digest in hashes.items():
        if support.digest(ROOT / name) != digest:
            raise RuntimeError('frozen input changed: ' + name)
    for name, digest in sources.items():
        if support.digest(ROOT / 'source' / name) != digest:
            raise RuntimeError('original runtime changed: ' + name)
    return dict(files=len(hashes), manifest_sha256=support.digest(ROOT / 'input_hashes.json'))


def command(protocol):
    return ['taskset', '-c', '0-7', 'timeout', '-s', 'TERM', '900',
            protocol['python'], '-u', str(ROOT / 'instrumentation/run_logical_alignment.py'),
            '--output', str(ROOT / 'results/qualification'), '--prepared', str(ROOT / 'prepared'),
            '--model', protocol['model'], '--pool-execution', 'oneshot', '--expert-cap', '24',
            '--execution', 'expert', '--requests', '5', '--prompt-tokens', '128',
            '--output-tokens', '8', '--arrival-interval', '0', '--token-budget', '160',
            '--kv-bytes', '1073741824', '--prefill-limit', '32', '--group-retention', 'none',
            '--trace-retention', 'memory', '--max-seconds', '600']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    protocol = read(ROOT / 'protocol.json')
    checked, cmd, env = check_inputs(), command(protocol), support.environment(protocol)
    if args.dry_run:
        print(json.dumps(dict(status='CPU_ONLY_DRY_RUN', checked=checked, command=cmd, env=env)))
        return
    out = ROOT / 'results'
    out.mkdir(exist_ok=False)
    record = dict(status='CHECKING', started_unix_s=time.time(), command=cmd, env=env,
                  checked=checked, process_wall_s=None, returncode=None,
                  scope='Numerical qualification; all reference/negative/prefix work is charged, not performance.')

    def save():
        tmp = out / 'execution.tmp'
        tmp.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
        tmp.replace(out / 'execution.json')

    save()
    lock = None
    try:
        lock = open(protocol['gpu_group_lock'], 'a+')
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.seek(0); lock.truncate()
        lock.write(json.dumps(dict(pid=os.getpid(), directory=str(ROOT), started=time.time())) + '\n')
        lock.flush()
        record.update(lock_owner_pid=os.getpid(), external_inputs=support.external_inputs(protocol),
                      gpu_before=support.gpu_snapshot())
        save()
        support.require_empty_gpu(record['gpu_before'], protocol['expected_gpu_uuid'])
        record.update(status='RUNNING')
        save()
        with (out / 'qualification.log').open('x') as log:
            record['process_start_perf_ns'] = time.perf_counter_ns()
            try:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                        env=dict(os.environ, **env))
                record['returncode'] = result.returncode
            finally:
                record['process_end_perf_ns'] = time.perf_counter_ns()
                record['process_wall_s'] = (record['process_end_perf_ns'] - record['process_start_perf_ns']) / 1e9
        if record['returncode']:
            raise RuntimeError('qualification subprocess failed; retained without retry')
        terminal = read(out / 'qualification/status.json')
        record['terminal'] = terminal
        if (terminal['status'], terminal['requests_completed'], terminal['generated_tokens']) != ('COMPLETE', 5, 40):
            raise RuntimeError('incomplete native request cohort')
        qual = read(out / 'qualification/logical_alignment.json')
        record['qualification_status'] = qual['status']
        if qual['schema'] != 'logical_alignment_qualification_v1' or qual['status'] != 'QUALIFIED':
            raise RuntimeError('numerical/path coverage not qualified')
        record['gpu_after'] = support.gpu_snapshot()
        support.require_empty_gpu(record['gpu_after'], protocol['expected_gpu_uuid'])
        record['status'] = 'COMPLETE_NUMERIC_QUALIFICATION'
    except BaseException as error:
        record.update(status='STOPPED', error=repr(error))
        raise
    finally:
        record['finished_unix_s'] = time.time()
        save()
        if lock is not None:
            lock.close()


if __name__ == '__main__':
    main()

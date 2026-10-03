"""Eight frozen fresh-engine cells; stdlib dry-run, no retry or resource waiter."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_bundle(protocol):
    hashes = read(ROOT / 'input_hashes.json')
    required = {'run_attempt.py', 'protocol.json', 'run_cells.json', 'sources.json',
                'prepared/workload.json', 'input_provenance.json', 'COMMANDS.md',
                'allocations/uniform.json', 'allocations/selected.json'}
    sources = read(ROOT / 'sources.json')
    required.update('source/' + name for name in sources)
    required.update(protocol['required_analysis_files'])
    if len(sources) != 11 or not required <= hashes.keys():
        raise RuntimeError('incomplete frozen input/analysis manifest')
    for name, expected in hashes.items():
        if digest(ROOT / name) != expected:
            raise RuntimeError('frozen bundle changed: ' + name)
    for name, expected in sources.items():
        if digest(ROOT / 'source' / name) != expected:
            raise RuntimeError('frozen runtime changed: ' + name)
    return dict(files=len(hashes), manifest_sha256=digest(ROOT / 'input_hashes.json'))


def command(plan, protocol):
    mode = plan['mode']
    shared = mode in ('fullstage', 'oneshot')
    cmd = [protocol['python'], '-u', str(ROOT / 'source' /
           ('run_shared_pool_pager.py' if shared else 'run_native_pager.py')),
           '--output', str(ROOT / 'results' / plan['label']), '--prepared', str(ROOT / 'prepared'),
           '--model', protocol['model'], '--expert-cap', '24', '--execution', 'expert',
           '--requests', '16', '--prompt-tokens', '128', '--output-tokens', '32',
           '--arrival-interval', '0.25', '--token-budget', '160', '--kv-bytes', '1073741824',
           '--prefill-limit', '32', '--group-retention', 'none', '--trace-retention', 'memory',
           '--max-seconds', str(protocol['limits']['capture_max_seconds'])]
    cmd += ['--pool-execution', mode] if shared else [
        '--layer-caps', str(ROOT / 'allocations' / (mode + '.json'))]
    return ['taskset', '-c', '0-7', 'timeout', '-s', 'TERM',
            str(protocol['limits']['subprocess_timeout_seconds']), *cmd]


def environment(protocol):
    return dict(OMP_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
                PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='0',
                PYTHONPATH=protocol['wisp_src'] + ':' + str(ROOT / 'source'))


def external_inputs(protocol):
    actual = {name: digest(Path(protocol['wisp_src']) / name)
              for name in protocol['expected_external_wisp']}
    if actual != protocol['expected_external_wisp']:
        raise RuntimeError('external WiSP source changed')
    packages = {name: importlib.metadata.version(name) for name in protocol['expected_packages']}
    if packages != protocol['expected_packages']:
        raise RuntimeError('package environment changed: ' + str(packages))
    if Path(sys.executable).absolute() != Path(protocol['python']).absolute():
        raise RuntimeError('execute with the frozen vLLM Python path')
    if not (Path(protocol['model']) / 'model.safetensors.index.json').is_file():
        raise RuntimeError('pinned local model snapshot missing')
    return dict(wisp_sha256=actual, packages=packages, python=sys.version,
                executable=sys.executable, model=protocol['model'])


def gpu_snapshot():
    def query(fields, flag):
        r = subprocess.run(['nvidia-smi', flag + '=' + fields, '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=20)
        return dict(returncode=r.returncode, stdout=r.stdout.strip(), stderr=r.stderr.strip())
    return dict(unix_s=time.time(), gpu=query('index,uuid,name,memory.used', '--query-gpu'),
                processes=query('gpu_uuid,pid,used_gpu_memory', '--query-compute-apps'))


def require_empty_gpu(snapshot, expected_uuid):
    if any(snapshot[k]['returncode'] for k in ('gpu', 'processes')):
        raise RuntimeError('ABORT: GPU query failed')
    devices = [line.split(',') for line in snapshot['gpu']['stdout'].splitlines()]
    if len(devices) != 1 or devices[0][0].strip() != '0' or devices[0][1].strip() != expected_uuid:
        raise RuntimeError('ABORT: expected single GPU UUID/index absent')
    if snapshot['processes']['stdout']:
        raise RuntimeError('ABORT: GPU busy; no process killed and no waiting/retry')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    protocol, plans = read(ROOT / 'protocol.json'), read(ROOT / 'run_cells.json')
    if args.dry_run:
        checked = check_bundle(protocol)
        print(json.dumps(dict(status='DRY_RUN_CPU_ONLY', checked=checked,
            external_checks='UNRUN: dry-run performs no GPU query or external runtime import/check',
            cells=[dict(**p, command=command(p, protocol), env=environment(protocol)) for p in plans]), indent=2))
        return
    out = ROOT / 'results'
    out.mkdir(exist_ok=False)
    record = dict(status='RUNNING', started_unix_s=time.time(), cells=[],
                  expected_gpu_uuid=protocol['expected_gpu_uuid'], process_scope=
                  'Each subprocess wall includes startup, one warmup, finite cohort, tracing/IO, finalize and shutdown.')
    def save():
        temporary = out / 'execution.tmp'
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
        temporary.replace(out / 'execution.json')
    save()
    cell = None
    try:
        for plan in plans:
            cell = dict(**plan, status='CHECKING', command=command(plan, protocol),
                        env=environment(protocol), process_wall_s=None, returncode=None)
            record['cells'].append(cell)
            save()
            cell['bundle_check'] = check_bundle(protocol)
            cell['external_inputs'] = external_inputs(protocol)
            cell['gpu_before'] = gpu_snapshot()
            save()
            require_empty_gpu(cell['gpu_before'], protocol['expected_gpu_uuid'])
            cell.update(status='RUNNING', started_unix_s=time.time())
            save()
            with (out / (plan['label'] + '.log')).open('x') as log:
                cell['process_start_perf_ns'] = time.perf_counter_ns()
                try:
                    result = subprocess.run(cell['command'], stdout=log, stderr=subprocess.STDOUT,
                                            env=dict(os.environ, **cell['env']))
                    cell['returncode'] = result.returncode
                finally:
                    cell['process_end_perf_ns'] = time.perf_counter_ns()
                    cell['process_wall_s'] = (cell['process_end_perf_ns'] - cell['process_start_perf_ns']) / 1e9
            cell['finished_unix_s'] = time.time()
            save()
            if cell['returncode']:
                raise RuntimeError(plan['label'] + ' failed; original attempt retained, no retry')
            terminal = read(out / plan['label'] / 'status.json')
            if (terminal['status'], terminal['requests_completed'], terminal['generated_tokens']) != ('COMPLETE', 16, 512):
                raise RuntimeError(plan['label'] + ' incomplete cohort')
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


if __name__ == '__main__':
    main()

"""Drive this prepared two-block campaign once; stop for review on any ambiguity."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/Users/leandrozhao/Desktop/、++++++++/refine-logs/independent_ideas_20260910/waiting_bypass_limit_r01')
HELPER = Path('/tmp/moe-01a07d4b-newhost-remote.py')
OPS = ROOT / 'operations'

def emit(stage, **state):
    print(json.dumps(dict(stage=stage, **state), ensure_ascii=False), flush=True)

def stop(reason, **state):
    emit('STOP_REQUIRES_ROOT_REVIEW', reason=reason, **state)
    raise SystemExit(1)

def run(stage, command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        stop(stage, exit_code=result.returncode, stdout=result.stdout[-1500:], stderr=result.stderr[-1500:])
    return result.stdout

def remote(name):
    return run(name, [sys.executable, str(HELPER), 'exec', str(OPS / name)])

def wait_ready(block):
    while True:
        state = json.loads(remote('readiness.py'))
        if state.get('terminal_preparation_errors'):
            stop('terminal_preparation_errors', errors=state['terminal_preparation_errors'])
        names = ('forward', 'reverse') if block == 'forward' else ('reverse',)
        existing = {name: state['artifacts_by_block'][name] for name in names if state['artifacts_by_block'][name]}
        if existing:
            stop('existing_remote_artifacts', artifacts=existing)
        ready = (state['model_ready'] and state['environment_ready']
                 and state['gpu_query_exit'] == 0 and not state['gpu_processes'].strip())
        emit('READY_CHECK', block=block, ready=ready, model_ready=state['model_ready'],
             environment_ready=state['environment_ready'], gpu_query_exit=state['gpu_query_exit'],
             gpu_empty=not state['gpu_processes'].strip(), time_utc=state['time_utc'],
             preparation_jobs=state['preparation_jobs'],
             downloaded_weight_bytes=sum(p['bytes'] for p in state['partial_downloads'])
                                    +sum(p['bytes'] or 0 for p in state['shards']))
        if ready:
            return
        time.sleep(45)

def wait_exit(block):
    while True:
        state = json.loads(remote('poll.py'))[block]
        process = state.get(f'results/{block}-process.json')
        supervisor_alive = state.get(f'{block}-supervisor.json_process_alive')
        process_alive = state.get(f'results/{block}-process.json_process_alive', False)
        code = process.get('exit_code') if isinstance(process, dict) else None
        emit('POLL', block=block, exit_code=code, supervisor_alive=supervisor_alive, process_alive=process_alive)
        if code is not None and supervisor_alive is False and not process_alive:
            return code
        if supervisor_alive is False and not process_alive and code is None:
            stop('supervisor_ended_without_recorded_exit', block=block)
        time.sleep(30)

try:
    paths = [ROOT / 'analysis']
    for block in ('forward', 'reverse'):
        paths.extend([ROOT / f'ARCHIVE-{block}.json', ROOT / f'readback-{block}.tar.gz',
                      ROOT / f'TRANSFER-{block}.json', ROOT / 'readback/results' / block])
    if any(path.exists() for path in paths):
        stop('existing_local_artifacts', paths=[str(path) for path in paths if path.exists()])
    for block in ('forward', 'reverse'):
        wait_ready(block)
        emit('LAUNCH', block=block, supervisor=json.loads(remote(f'launch-{block}.py')))
        code = wait_exit(block)
        emit('ARCHIVE', block=block, process_exit_code=code)
        archive_text = remote(f'archive-{block}.py')
        with (ROOT / f'ARCHIVE-{block}.json').open('x') as file:
            file.write(archive_text)
        archive = json.loads(archive_text)
        destination = ROOT / f'readback-{block}.tar.gz'
        if destination.exists():
            stop('download_destination_exists', path=str(destination))
        run('DOWNLOAD', [sys.executable, str(HELPER), 'download', archive['archive'], str(destination)])
        emit('DOWNLOAD_COMPLETE', block=block, bytes=destination.stat().st_size)
        run('VERIFY', [sys.executable, str(ROOT / 'verify_readback.py'), block])
        transfer = json.loads((ROOT / f'TRANSFER-{block}.json').read_text())
        complete = (code == 0 and archive['process']['exit_code'] == 0
                    and archive['status']['status'] == 'COMPLETE'
                    and transfer['process_exit_code'] == 0 and transfer['all_raw_complete'] is True
                    and transfer['measured_episodes'] == 6 and transfer['warmup_episodes'] == 6)
        emit('READBACK_VERIFIED', block=block, complete=complete, transfer=transfer)
        if not complete:
            stop('block_incomplete_after_retaining_readback', block=block)
    if (ROOT / 'analysis').exists():
        stop('analysis_destination_exists')
    emit('ANALYZE')
    run('ANALYZE', [sys.executable, str(ROOT / 'analyze_results.py'),
                    '--results-dir', str(ROOT / 'readback/results'), '--output-dir', str(ROOT / 'analysis')])
    emit('TWO_BLOCKS_COMPLETE_PENDING_REVIEW', measured_episodes=12, warmup_episodes=12, analysis=str(ROOT / 'analysis'))
except Exception as error:
    stop('driver_error_no_automatic_restart', error=f'{type(error).__name__}: {error}')

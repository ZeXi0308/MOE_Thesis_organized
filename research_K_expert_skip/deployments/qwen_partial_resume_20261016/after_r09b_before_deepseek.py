#!/usr/bin/env python3
"""CPU bridge: successful r09b -> lossless r07 storage -> Qwen -> DS handoff.

The original stopped handoff receipt is never overwritten. Only this bridge's
CPU helper process groups can be signalled; submitted DeepSeek jobs are untouched.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
ORIGINAL_SHA = '5c8baea5b2bb227dfe60179edcfecb57dddbc9898ac08384840704c688ac7dd7'
OLD_PID = 64317
STEM = 'bridge_r09b_qwen_deepseek'


def load_original(stage):
    path = stage / 'after_r09b.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != ORIGINAL_SHA:
        raise ValueError('Original DeepSeek handoff source SHA changed')
    spec = importlib.util.spec_from_file_location('original_ds_handoff', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stop_qwen(child):
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-tmp/research_K_expert_skip'))
    parser.add_argument('--python', default='/root/miniconda3/bin/python')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan-only', action='store_true')
    mode.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    stage = root / 'deployments/deepseek_matrix_20261016'
    m = load_original(stage)
    command = [args.python, str(HERE/'resume_qwen.py'), '--execute', '--root', str(root),
               '--destination', str(root/'models/qwen3_bf16_ad44e777'),
               '--fixed-placement', str(HERE/'fixed_placement.json'), '--c-reservation', 'append',
               '--max-workers', '2', '--chunk-mib', '64', '--retries', '2']
    archive = root/'runs/olmoe_bf16_matrix_r07/archived_performance.tgz'
    compact_commands = [[args.python, str(HERE/'compact_r07.py'), mode, '--root', str(root),
                         '--archive', str(archive)] for mode in ('--prepare', '--compact')]
    sources = {str(p): m.sha(p) for p in (Path(__file__), stage/'after_r09b.py',
                HERE/'resume_qwen.py', HERE/'fixed_placement.json',
                HERE/'compact_r07.py')}
    if args.plan_only:
        print(json.dumps({'state': 'PLAN_ONLY', 'source_sha256': sources, 'download_command': command,
            'compact_commands': compact_commands, 'compact_total_timeout_seconds': 300,
            'download_timeout_seconds': 3600, 'download_environment': {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2'},
            'prerequisites': ['original PID64317 exited AND original receipt STOPPED',
                'original DS CPU handoff.lock acquired; no DS targets/live payload',
                'r09b successful terminal SHA plus actual controller/worker exit', 'no other active K run'],
            'next': 'original handoff(root, dsstage, python, save); download failure/timeout does not cancel DS',
            'signals': 'Only owned Qwen process group; never signal DS', 'gpu_or_remote_action': False}, indent=2))
        return 0
    with (stage/'handoff.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        destination = HERE/(STEM+'.json')
        state = {'pid': os.getpid(), 'created_unix': time.time(), 'source_sha256': sources, 'root': str(root)}
        with destination.open('x') as handle:
            json.dump({**state, 'state': 'STARTING'}, handle)
        with (HERE/(STEM+'.events.jsonl')).open('x') as events:
            def save(value, **fields):
                state.update(state=value, observed_unix=time.time(), **fields)
                temporary = destination.with_suffix('.tmp')
                temporary.write_text(json.dumps(state, indent=2)+'\n')
                temporary.replace(destination)
                line = json.dumps(state)
                events.write(line+'\n'); events.flush()
                print(line, flush=True)
            child = None
            def interrupted(signum, frame):
                raise InterruptedError(f'Bridge signal {signum}; only own CPU helper is stopped')
            signal.signal(signal.SIGTERM, interrupted)
            signal.signal(signal.SIGINT, interrupted)
            try:
                old = m.read_json(stage/'handoff_r09b.json')
                process = m.process(OLD_PID)
                if old.get('pid') != OLD_PID or old.get('state') != 'STOPPED' or (process and process['state'] not in ('Z', 'X')):
                    raise RuntimeError('Original CPU handoff must be STOPPED and PID64317 actually exited')
                save('CHECKING_PREDECESSOR', original_stopped_receipt_sha256=m.sha(stage/'handoff_r09b.json'))
                m.available(root, m.SUCCESSORS, allowed_active=m.PREDECESSOR)
                m.verify_payload(stage, root, False)
                receipt = m.wait_success(root/'runs'/m.PREDECESSOR, m.PREDECESSOR_STAGES, save)
                m.available(root, m.SUCCESSORS)
                compact = {'started_unix': time.time(), 'status': 'FAILED'}
                save('COMPACTING_R07', predecessor=receipt, compact_commands=compact_commands)
                try:
                    deadline = time.monotonic() + 300
                    with (HERE/(STEM+'.compact.log')).open('x') as log:
                        for argv in compact_commands[1:] if archive.exists() else compact_commands:
                            child = subprocess.Popen(argv, cwd=root, stdout=log,
                                                     stderr=subprocess.STDOUT, start_new_session=True)
                            compact['pid'] = child.pid
                            try:
                                compact['exit_code'] = child.wait(timeout=max(.1, deadline-time.monotonic()))
                                compact['status'] = 'COMPLETE' if child.returncode == 0 else 'FAILED'
                            except subprocess.TimeoutExpired:
                                compact['status'] = 'TIMEOUT'
                                stop_qwen(child)
                                compact['exit_code'] = child.returncode
                            if compact['status'] != 'COMPLETE':
                                break
                except OSError as error:
                    compact['error'] = f'{type(error).__name__}: {error}'
                finally:
                    if child is not None and child.poll() is None:
                        stop_qwen(child)
                    child = None
                    compact['ended_unix'] = time.time()
                    save('COMPACTION_FINISHED', compaction_result=compact)
                m.available(root, m.SUCCESSORS)
                save('QWEN_DOWNLOADING', predecessor=receipt, download_command=command, download_timeout_seconds=3600)
                result = {'started_unix': time.time(), 'status': 'FAILED'}
                try:
                    with (HERE/(STEM+'.download.log')).open('x') as log:
                        child = subprocess.Popen(command, cwd=root, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2'},
                                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        result['pid'] = child.pid
                        save('QWEN_DOWNLOADING', download_pid=child.pid)
                        try:
                            result['exit_code'] = child.wait(timeout=3600)
                        except subprocess.TimeoutExpired:
                            result['status'] = 'TIMEOUT'
                            stop_qwen(child)
                            result['exit_code'] = child.returncode
                    for path in (root/'runs').glob(f'qwen_partial_resume_*_{child.pid}/source.json'):
                        result['receipt'] = {'path': str(path), 'sha256': m.sha(path), 'status': m.read_json(path).get('status')}
                    if result['status'] != 'TIMEOUT' and result.get('exit_code') == 0 and result.get('receipt', {}).get('status') == 'READY':
                        result['status'] = 'COMPLETE'
                except (OSError, ValueError) as error:
                    result['error'] = f'{type(error).__name__}: {error}'
                finally:
                    if child is not None and child.poll() is None:
                        stop_qwen(child)
                    child = None
                    result['ended_unix'] = time.time()
                    save('QWEN_DOWNLOAD_FINISHED', download_result=result)
                m.available(root, m.SUCCESSORS)
                m.handoff(root, stage, args.python, save)
            except BaseException as error:
                if child is not None:
                    stop_qwen(child)
                save('STOPPED', error=f'{type(error).__name__}: {error}')
                raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

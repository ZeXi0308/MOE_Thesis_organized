#!/usr/bin/env python3
"""LOCAL one-shot continuation: verified calibration -> export -> forward -> serving.

Uses the existing SSH master; never retries/reuses a submitted GPU job name.
Whole terminal groups are copied and verified by sync_terminal_run.py before
the next submission. Large checkpoint/artifact copies remain separate jobs.
Run only once after deploying the referenced remote consumers.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='runs/integration_advance_r01')
    p.add_argument('--initial-wait-seconds', type=int, default=7200)
    p.add_argument('--calibration-run', default='olmoe_awq_integration_r01')
    p.add_argument('--lock-wait-seconds', type=int, default=21600)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = (root / a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    remote = '/root/autodl-tmp/research_C_format_sched'
    python = '/root/miniconda3/bin/python'
    cuda_libs = '/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib'
    ssh = ['ssh', '-S', '/tmp/c-format-ssh.sock', '-p', '50328', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'root@connect.westb.seetacloud.com']
    stages = [('olmoe_integration_export_r01', 1800, [python, 'configs/build_integration_configs.py', '--execute-export']),
              ('integration_group_r01', 1800, [python, 'code/run_full_integration_group.py', '--stage', 'forward', '--output', 'runs/integration_group_r01/group']),
              ('minimal_serving_group_r01', 1200, [python, 'code/run_full_integration_group.py', '--stage', 'serving', '--output', 'runs/minimal_serving_group_r01/group'])]
    state = {'state': 'WAITING_FOR_VERIFIED_CALIBRATION', 'started_unix_s': time.time(), 'stages': []}
    def save():
        temporary = out / 'state.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(out / 'state.json')
    def read(path):
        return json.loads(path.read_text()) if path.is_file() else {}
    def verified(name):
        receipt = read(root / 'runs' / (name + '.local_sync.json'))
        status = read(root / 'runs' / name / 'status.json')
        if receipt.get('state') in ('VERIFICATION_FAILED', 'WAIT_TIMEOUT') or status.get('state') == 'FAILED':
            raise RuntimeError(f'{name}: calibration/job or local verification failed')
        return receipt.get('state') == 'VERIFIED_LOCAL' and status.get('state') == 'COMPLETE'
    save()
    try:
        deadline = time.monotonic() + a.initial_wait_seconds
        while not verified(a.calibration_run):
            if time.monotonic() >= deadline:
                raise TimeoutError('calibration did not reach verified local COMPLETE before deadline')
            time.sleep(min(30, max(0, deadline - time.monotonic())))
        for name, budget, consumer in stages:
            status_path = f'runs/{name}/status.json'
            command = [python, 'code/run_locked.py', '--name', name, '--wait-seconds', str(a.lock_wait_seconds), '--timeout', str(budget), '--', *consumer, '--lock-status', status_path]
            remote_command = 'cd ' + shlex.quote(remote) + ' && ' + shlex.join(['env', 'LD_LIBRARY_PATH=' + cuda_libs, *command])
            stage = {'name': name, 'state': 'SUBMITTING', 'command': command, 'started_unix_s': time.time()}
            state['stages'].append(stage)
            state['state'] = name
            save()
            with (out / (name + '.stdout.log')).open('w') as log:
                result = subprocess.run([*ssh, remote_command], cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=a.lock_wait_seconds + budget + 120)
            stage.update(ssh_exit_code=result.returncode, state='SYNCING_TERMINAL_GROUP')
            save()
            # Failed remote groups are backed up too. A disconnected SSH never
            # causes a resubmission; this only waits for the original terminal.
            with (out / (name + '.sync.log')).open('w') as log:
                copied = subprocess.run([sys.executable, str(root / 'code/sync_terminal_run.py'), '--name', name, '--wait-seconds', str(a.lock_wait_seconds + budget + 120)], cwd=root, stdout=log, stderr=subprocess.STDOUT)
            stage['sync_exit_code'] = copied.returncode
            if result.returncode or copied.returncode or not verified(name):
                raise RuntimeError(f'{name}: stop after remote/backup failure; inspect retained stdout and original status before any manual continuation')
            stage.update(state='VERIFIED_LOCAL_COMPLETE', ended_unix_s=time.time())
            save()
        state.update(state='COMPLETE', ended_unix_s=time.time())
        save()
    except BaseException as error:
        state.update(state='FAILED', error=f'{type(error).__name__}: {error}', ended_unix_s=time.time())
        save()
        raise


if __name__ == '__main__':
    main()

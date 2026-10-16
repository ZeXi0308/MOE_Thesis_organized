#!/usr/bin/env python3
"""Submit the bounded FP8 matrix once, after r08b completed and exited."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path('/root/autodl-tmp/research_K_expert_skip')
STAGE = ROOT / 'deployments/olmoe_fp8_matrix_r09_20261016'
PREDECESSOR = ROOT / 'runs/task_quality_pilot_r08b'
NAME = 'olmoe_fp8_matrix_r09'
STATE = {'pid': os.getpid(), 'created_unix': time.time(),
         'predecessor': str(PREDECESSOR), 'next_run': NAME}


def save(state, **fields):
    STATE.update(state=state, observed_unix=time.time(), **fields)
    temporary = STAGE / 'handoff_r02.tmp'
    temporary.write_text(json.dumps(STATE, indent=2) + '\n')
    temporary.replace(STAGE / 'handoff_r02.json')
    print(json.dumps(STATE), flush=True)


def alive(pid):
    path = Path(f'/proc/{pid}/cmdline')
    return path.exists() and bool(path.read_bytes())


try:
    deadline = time.monotonic() + 28800
    save('WAITING_FOR_PREDECESSOR')
    while True:
        prior = json.loads((PREDECESSOR / 'status.json').read_text())
        state = prior['state']
        if state not in ('PREPARING', 'WAITING', 'RUNNING', 'COMPLETE'):
            raise RuntimeError(f'Predecessor ended as {state}; inspect it before any next submission')
        if state == 'COMPLETE' and (PREDECESSOR / 'SHA256.json').is_file():
            if not any(alive(prior.get(k)) for k in ('controller_pid', 'worker_pid')):
                break
        elif state in ('PREPARING', 'WAITING', 'RUNNING') and not alive(prior.get('controller_pid')):
            raise RuntimeError('Predecessor controller is absent; no automatic replacement')
        if time.monotonic() >= deadline:
            raise TimeoutError('Observation deadline reached; existing processes were left unchanged')
        time.sleep(30)
    if (ROOT / 'runs' / NAME).exists():
        raise RuntimeError('Successor directory already exists; refusing duplicate submission')
    for path in (ROOT / 'runs').glob('*/status.json'):
        entry = json.loads(path.read_text())
        if entry.get('state') in ('PREPARING', 'WAITING', 'RUNNING'):
            raise RuntimeError(f'Another K run is active: {path}')
    manifest = json.loads((STAGE / 'deployment_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] not in ('code', 'configs'):
            raise ValueError('Invalid deployment path')
        source = STAGE / relative
        if source.is_symlink() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Deployment content mismatch: {name}')
        target = ROOT / relative
        if target.is_symlink():
            raise ValueError(f'Refusing symlink deployment target: {target}')
        previous = manifest['previous_files'][name]
        actual = hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None
        if actual != previous:
            raise RuntimeError(f'Live source changed since staging: {name}; reconcile before deployment')
    save('DEPLOYING', predecessor_ended_unix=prior['ended_unix'],
         predecessor_lock_held_s=prior['lock_held_s'])
    for name in manifest['files']:
        source, target = STAGE / name, ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.r09.tmp')
        shutil.copy2(source, temporary)
        temporary.replace(target)
    command = [
        '/root/miniconda3/bin/python', 'code/run_locked.py',
        '--name', NAME, '--wait-seconds', '21600', '--timeout', '7200', '--',
        '/root/miniconda3/bin/python', 'code/run_olmoe_fp8_r09.py',
        '--root', str(ROOT), '--output', f'runs/{NAME}', '--budget-seconds', '7170',
    ]
    environment = os.environ.copy()
    environment.pop('VLLM_TUNED_CONFIG_FOLDER', None)
    environment.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', VLLM_ENABLE_V1_MULTIPROCESSING='0')
    with (STAGE / 'r09_controller_r02.log').open('x') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                 stdout=log, stderr=subprocess.STDOUT)
        save('SUCCESSOR_SUBMITTED', command=command, controller_pid=child.pid)
        rc = child.wait()
    save('SUCCESSOR_EXITED', returncode=rc)
except BaseException as error:
    save('STOPPED', error=f'{type(error).__name__}: {error}')
    raise

"""One remote six-cell lifecycle recorder, adapted from the funding wrapper.

Run only after the shared queue releases the host. run.sh owns the common flock
and exits immediately when busy. Never relaunch this directory automatically.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PYTHON = '/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python'
GPU = 'GPU-4015b79d-bed6-3a4b-9d2b-0c17be96d0e5'
ARCHIVE_SHA256 = 'ec04a11134f6f27edc529deda4d8bb662e9c7b461938766aa4c9d4fd2fb62d9a'


def main():
    root = Path(__file__).resolve().parent
    cells = json.loads((root / 'pkg/campaign.json').read_text())['cells']
    if len(cells) != 6 or not Path(PYTHON).is_file():
        raise RuntimeError('requires the prepared six-cell package and existing Python')
    if any((root / name).exists() for name in ('results', 'group-status.json', 'campaign.log')):
        raise FileExistsError('refuse existing execution artifacts')
    with (root / 'launch-once').open('x') as handle:
        handle.write(str(os.getpid()) + '\n')
    # An SSH transport closing must not abandon the recorder while run.sh runs.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    state = dict(status='STARTING', pid=os.getpid(), started_unix_s=time.time(),
                 shell_pid=None, returncode=None, expected_gpu_uuid=GPU,
                 archive_sha256=ARCHIVE_SHA256, python=PYTHON)

    def save():
        temporary = root / 'group-status.json.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(root / 'group-status.json')

    def cell_states():
        rows = []
        for cell in cells:
            path = root / 'results' / cell['label'] / 'status.json'
            try:
                terminal = json.loads(path.read_text()) if path.exists() else dict(status='UNRUN')
            except (OSError, ValueError) as error:
                terminal = dict(status='UNREADABLE', error=f'{type(error).__name__}: {error}')
            rows.append(dict(label=cell['label'], terminal=terminal))
        return rows

    process = None
    save()
    try:
        env = dict(os.environ, HF_HOME='/root/autodl-tmp/hf-cache', HF_HUB_OFFLINE='1',
                   VLLM_USE_FLASHINFER_SAMPLER='0', CUDA_VISIBLE_DEVICES=GPU)
        with (root / 'campaign.log').open('x') as log:
            process = subprocess.Popen(['bash', 'pkg/run.sh', PYTHON], cwd=root, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            state.update(status='RUNNING', shell_pid=process.pid)
            save()
            state['returncode'] = process.wait()
        state['cells'] = cell_states()
        complete = state['returncode'] == 0 and all(
            row['terminal']['status'] == 'COMPLETE' for row in state['cells'])
        state['status'] = ('COMPLETE' if complete else
                           'BLOCKED_RESOURCE_BUSY' if state['returncode'] in (90, 93) else 'STOPPED')
    except BaseException as error:
        state.update(status='UNKNOWN_RUNNING' if process and process.poll() is None else 'STOPPED',
                     error=f'{type(error).__name__}: {error}')
        raise
    finally:
        state['cells'] = cell_states()
        if process is not None:
            state['returncode'] = process.poll()
        if state['status'] != 'UNKNOWN_RUNNING':
            state['finished_unix_s'] = time.time()
        state['updated_unix_s'] = time.time()
        save()
    return 0 if state['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    sys.exit(main())

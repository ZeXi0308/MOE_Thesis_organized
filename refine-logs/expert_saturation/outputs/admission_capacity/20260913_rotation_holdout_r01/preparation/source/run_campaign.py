"""Expose the frozen order and launch one fresh cell before local readback."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

CAMPAIGN = json.loads((Path(__file__).resolve().parent / 'campaign.json').read_text())
CELL_BY_LABEL = {cell['label']: cell for cell in CAMPAIGN['cells']}

def run_cell(label):
    cell = CELL_BY_LABEL[label]
    domain, budget = 'long', 0.9
    root = Path(__file__).resolve().parent
    out = root / 'results'
    out.mkdir(exist_ok=True)
    metadata = out / f'{label}-execution.json'
    if (out / label).exists():
        raise FileExistsError(f'cell results already exist: {out / label}')
    command = [sys.executable, '-u', 'run_probe.py', '--domain', domain, '--cap', str(cell['cap']), '--cohort', cell['cohort_id'],
               '--gpu-memory-utilization', str(budget), '--reservation-policy', 'full', '--completion-policy', cell['completion_policy'],
               '--output-dir', str(out / label)]
    env = dict(os.environ, HF_HUB_CACHE='/root/autodl-tmp/hf-cache/hub', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='0', VLLM_ENABLE_V1_MULTIPROCESSING='0',
        VLLM_USE_FLASHINFER_SAMPLER='0')
    state = dict(label=label, campaign_cell=cell, status='STARTING', launcher_pid=os.getpid(), child_pid=None,
        command=command, cwd=str(root), started_unix_s=time.time(), returncode=None,
        launcher_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                        for name in ['run_campaign.py', 'run_one_cell.py']})
    with metadata.open('x') as stream:
        json.dump(state, stream, indent=2)

    def save():
        temp = metadata.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(metadata)

    print(shlex.join(command), flush=True)
    try:
        with (out / f'{label}.stdout.log').open('x') as stdout, \
             (out / f'{label}.stderr.log').open('x') as stderr:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=stdout, stderr=stderr)
            state.update(status='RUNNING', child_pid=process.pid)
            save()
            code = process.wait()
        state.update(status='EXITED', returncode=code)
    except BaseException as error:
        state.update(status='LAUNCH_OR_WAIT_FAILED', error=repr(error))
        raise
    finally:
        state['finished_unix_s'] = time.time()
        save()
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--list', action='store_true', help='print the frozen twenty-cell order')
    mode.add_argument('--cell', choices=list(CELL_BY_LABEL), help='run only this cell, then stop for readback')
    args = parser.parse_args()
    if args.list:
        print(json.dumps(list(CELL_BY_LABEL), indent=2))
        return 0
    return run_cell(args.cell)


if __name__ == '__main__':
    raise SystemExit(main())

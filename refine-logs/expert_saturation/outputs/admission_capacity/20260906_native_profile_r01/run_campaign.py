"""Four fresh processes for an off/on/on/off attribution diagnostic."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path.cwd()
results = root / 'results'
results.mkdir(exist_ok=False)
state = dict(status='STARTED', queue_pid=os.getpid(), completed=[])


def save():
    temp = results / 'queue_status.tmp'
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(results / 'queue_status.json')


save()
for label, mode in [('off0', 'off'), ('on0', 'on'), ('on1', 'on'), ('off1', 'off')]:
    env = dict(os.environ, HF_HUB_CACHE='/root/autodl-tmp/hf-cache/hub', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='0', VLLM_ENABLE_V1_MULTIPROCESSING='0',
        VLLM_USE_FLASHINFER_SAMPLER='0', PROBE_MODE=mode)
    cmd = [sys.executable, '-u', 'run_native_capacity.py', '--prepared-dir', 'original',
        '--output-dir', str(results / label), '--cap', '8', '--engine-max-seqs', '8',
        '--arrival-scales', '0.02', '--repeats', '1', '--ttft-slo-s', '0.20',
        '--tpot-slo-s', '0.009', '--warmup-condition-rounds', '1']
    with (results / f'{label}.console.log').open('x') as log:
        child = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        state.update(status='RUNNING', label=label, child_pid=child.pid, started_unix_s=time.time())
        save()
        code = child.wait()
    state.update(child_exit_code=code, finished_unix_s=time.time())
    if code:
        state['status'] = 'STOPPED_AFTER_CHILD_FAILURE'
        save()
        sys.exit(code)
    state['completed'].append(label)
state['status'] = 'COMPLETE'
save()

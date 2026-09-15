"""Run the declared tiny injection matrix, stopping on its first failed cell."""
from pathlib import Path
import json
import os
import subprocess
import time

base = Path(__file__).resolve().parent
root = base.parent
protocol = json.loads((base / 'protocol.json').read_text())
state = dict(status='RUNNING', cells=[], start_unix=time.time())
state_path = base / 'execution.json'
model = '/root/autodl-tmp/hf-cache/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5'
def save():
    state_path.write_text(json.dumps(state, indent=2) + '\n')
try:
    for cell in protocol['cells']:
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True).strip()
        if apps:
            raise RuntimeError('GPU occupied before cell: ' + apps)
        out = base / cell['cell']
        out.mkdir(exist_ok=False)
        cmd = [str(root / 'venv/bin/python'), str(base / 'run_wisp_injection_probe.py'), '--model', model,
               '--workload', str(base / 'workload.json'), '--out', str(out / 'result.json'),
               '--chunk', str(cell['chunk']), '--new-prompt-length', str(cell['new_prompt_length']), '--trace']
        if cell.get('no_new'):
            cmd.append('--no-new')
        row = dict(cell, command=cmd, start_unix=time.time(), status='RUNNING')
        state['cells'].append(row)
        save()
        env = dict(os.environ, OMP_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
                   PYTHONPATH=str(root / 'source-traced/src') + ':' + str(base))
        with (out / 'run.log').open('wb') as log:
            code = subprocess.run(['timeout', '-s', 'TERM', '600'] + cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        result = json.loads((out / 'result.json').read_text()) if (out / 'result.json').exists() else {}
        row.update(exit_code=code, end_unix=time.time(), status=result.get('status', 'NO_RESULT'))
        save()
        if code or row['status'] != 'COMPLETED':
            raise RuntimeError('Cell failed: ' + cell['cell'])
    state['status'] = 'COMPLETED'
except Exception as exc:
    state.update(status='STOPPED', error=repr(exc))
finally:
    state['end_unix'] = time.time()
    save()

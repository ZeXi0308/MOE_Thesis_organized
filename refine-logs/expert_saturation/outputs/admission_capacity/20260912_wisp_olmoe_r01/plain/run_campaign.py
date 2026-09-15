"""Untraced repeated injection with external, timestamped NVML observations."""
from pathlib import Path
import json
import os
import subprocess
import time
import pynvml as nv

base = Path(__file__).resolve().parent
root = base.parent
model = '/root/autodl-tmp/hf-cache/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5'
state = dict(status='RUNNING', cells=[], start_unix=time.time())
def save():
    (base / 'execution.json').write_text(json.dumps(state, indent=2) + '\n')
nv.nvmlInit()
gpu = nv.nvmlDeviceGetHandleByIndex(0)
def observe():
    return dict(unix_s=time.time(), sm_clock_mhz=nv.nvmlDeviceGetClockInfo(gpu, nv.NVML_CLOCK_SM),
        memory_clock_mhz=nv.nvmlDeviceGetClockInfo(gpu, nv.NVML_CLOCK_MEM),
        temperature_c=nv.nvmlDeviceGetTemperature(gpu, nv.NVML_TEMPERATURE_GPU),
        power_mw=nv.nvmlDeviceGetPowerUsage(gpu), gpu_util_pct=nv.nvmlDeviceGetUtilizationRates(gpu).gpu,
        compute_pids=[p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(gpu)],
        cgroup_cpu_stat=Path('/sys/fs/cgroup/cpu.stat').read_text())
try:
    for cell in json.loads((base / 'protocol.json').read_text())['cells']:
        if nv.nvmlDeviceGetComputeRunningProcesses(gpu):
            raise RuntimeError('GPU occupied before cell')
        out = base / cell['cell']
        out.mkdir(exist_ok=False)
        cmd = [str(root / 'venv/bin/python'), str(base / 'run_wisp_injection_probe.py'), '--model', model,
               '--workload', str(base / 'workload.json'), '--out', str(out / 'result.json'),
               '--chunk', str(cell['chunk']), '--new-prompt-length', str(cell['new_prompt_length'])]
        row = dict(cell, command=cmd, start_unix=time.time(), status='RUNNING')
        state['cells'].append(row)
        save()
        env = dict(os.environ, OMP_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
                   PYTHONPATH=str(root / 'source/src') + ':' + str(base))
        with (out / 'run.log').open('wb') as log, (out / 'hardware.jsonl').open('x') as telemetry:
            process = subprocess.Popen(['timeout', '-s', 'TERM', '600'] + cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                telemetry.write(json.dumps(observe()) + '\n')
                telemetry.flush()
                time.sleep(0.1)
            code = process.returncode
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
    nv.nvmlShutdown()

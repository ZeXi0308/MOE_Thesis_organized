"""Wait for this attempt's resources, then run the retained driver and analyzer once."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
BASE_PYTHON = '/root/miniconda3/bin/python'
PYTHON = '/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python'
REMOTE = '/root/autodl-tmp/moe-kv-budget-20260910-r01'
SSH = ['ssh', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=10', '-o',
       'NumberOfPasswordPrompts=1', '-o', 'StrictHostKeyChecking=yes', '-p', '11155',
       'root@connect.weste.seetacloud.com']
VERSIONS = {'torch': '2.11.0', 'vllm': '0.26.0', 'transformers': '5.15.1'}
DOWNLOADER = {'pid': 2025, 'start_ticks': '483712808', 'boot_id': 'ec337ff8-dddc-45c8-b47c-f63bfd4737e7'}
GPU_UUID = 'GPU-e4f548ee-bb0c-664b-b2ce-3a006789e189'
SNAPSHOT = '/root/autodl-tmp/hf-cache/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5'
SHARDS = {
    'model-00001-of-00003.safetensors': [4997744872, '5e3cff7e367794685c241169072c940d200918617d5e2813f1c387dff52d845e'],
    'model-00002-of-00003.safetensors': [4997235176, '15ef5c730ee3cfed7199498788cd2faf337203fc74b529625e7502cdd759f4a7'],
    'model-00003-of-00003.safetensors': [3843741912, 'a9abac4ac1b55c9adabac721a02fa39971f103eea9a65c310972b1246de76e04']}
QUERY = r'''
import hashlib,json,pathlib,subprocess,time
version_code="import importlib.metadata as m,json; print(json.dumps({d.metadata['Name'].lower():d.version for d in m.distributions() if d.metadata['Name'].lower() in ['torch','vllm','transformers']}))"
p=subprocess.run([PYTHON,'-c',version_code],capture_output=True,text=True,timeout=20)
versions=json.loads(p.stdout) if p.returncode==0 else {}
versions_ready=all(versions.get(k,'').split('+')[0]==v for k,v in VERSIONS.items())
setup_path=pathlib.Path('/root/autodl-tmp/moe-kv-budget-20260910-r01/setup-runtime-mirror-state.json')
try:setup=json.loads(setup_path.read_text())
except (FileNotFoundError,json.JSONDecodeError):setup={'status':'PENDING'}
runtime_ready=versions_ready and setup.get('status')=='EXITED' and setup.get('returncode')==0
proc=pathlib.Path('/proc')/str(DOWNLOADER['pid']); identity=dict(pid=DOWNLOADER['pid'],exists=proc.exists())
try:
 stat=(proc/'stat').read_text().split(') ',1)[1].split()
 identity.update(start_ticks=stat[19],state=stat[0],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip())
except FileNotFoundError:identity['exists']=False
same=identity['exists'] and identity.get('state')!='Z' and all(identity.get(k)==v for k,v in DOWNLOADER.items())
shards={name:dict(expected_bytes=spec[0],expected_sha256=spec[1],bytes=(pathlib.Path(SNAPSHOT)/name).stat().st_size if (pathlib.Path(SNAPSHOT)/name).exists() else None) for name,spec in SHARDS.items()}
sizes_ready=all(s['bytes']==s['expected_bytes'] for s in shards.values())
if VERIFY and sizes_ready:
 for name,s in shards.items():
  digest=hashlib.sha256()
  with (pathlib.Path(SNAPSHOT)/name).open('rb') as stream:
   for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
  s['sha256']=digest.hexdigest()
gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
print(json.dumps(dict(observed_unix_s=time.time(),versions=versions,versions_ready=versions_ready,setup=setup,runtime_ready=runtime_ready,downloader=identity,downloader_same=same,shards=shards,sizes_ready=sizes_ready,gpu_uuid=gpu,compute_processes=processes,ready=runtime_ready and sizes_ready and gpu==GPU_UUID and not processes)))
'''


def remote(verify=False):
    values = dict(PYTHON=PYTHON, VERSIONS=VERSIONS, DOWNLOADER=DOWNLOADER,
                  GPU_UUID=GPU_UUID, SNAPSHOT=SNAPSHOT, SHARDS=SHARDS, VERIFY=verify)
    code = '\n'.join(f'{key}={value!r}' for key, value in values.items()) + '\n' + QUERY
    command = [*SSH, shlex.join([BASE_PYTHON, '-c', code])]
    return json.loads(subprocess.check_output(command, text=True, timeout=600 if verify else 45))


def main():
    state_path = HERE / 'await-readiness-state-20260910.json'
    state = dict(status='WAITING', local_pid=os.getpid(), child_pid=None,
                 started_unix_s=time.time(), expected_downloader=DOWNLOADER, remote_root=REMOTE)
    with state_path.open('x') as stream: json.dump(state, stream)

    def save(**changes):
        state.update(changes, updated_unix_s=time.time())
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n'); temporary.replace(state_path)

    def child(script, label):
        with (HERE / f'await-{label}-20260910.stdout.log').open('x') as out, \
             (HERE / f'await-{label}-20260910.stderr.log').open('x') as err:
            process = subprocess.Popen([sys.executable, '-u', script], cwd=HERE, stdout=out, stderr=err)
            save(status=label.upper(), child_pid=process.pid, child_script=script)
            code = process.wait()
        save(child_pid=None, **{f'{label}_pid': process.pid, f'{label}_returncode': code})
        if code: raise RuntimeError(f'{label} exited {code}; no restart')

    try:
        while True:
            try:
                observation = remote()
                save(last_observation=observation)
                if observation['setup'].get('status')=='EXITED' and observation['setup'].get('returncode')!=0:
                    raise RuntimeError('Runtime setup failed; preserve logs before any repair')
                if observation['gpu_uuid'] != GPU_UUID: raise RuntimeError('GPU identity changed')
                if not observation['sizes_ready'] and not observation['downloader_same']:
                    raise RuntimeError('Original downloader absent/replaced with missing shards')
                if observation['ready']:
                    observation = remote(verify=True)
                    if not all(s.get('sha256') == s['expected_sha256'] for s in observation['shards'].values()):
                        raise RuntimeError('Pinned shard SHA256 mismatch')
                    with (HERE / 'await-readiness-qualification-20260910.json').open('x') as stream:
                        json.dump(observation, stream, indent=2)
                    if not observation['ready']: raise RuntimeError('Resource readiness changed during hash verification')
                    save(status='QUALIFIED', last_observation=observation)
                    break
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
                if isinstance(error, subprocess.CalledProcessError) and error.returncode != 255: raise
                observation = dict(event='SSH_RETRY_SAME_HANDLES', error_type=type(error).__name__)
                save(last_connection_error=observation)
            with (HERE / 'await-readiness-observations-20260910.jsonl').open('a') as stream:
                stream.write(json.dumps(observation) + '\n')
            print(json.dumps(observation), flush=True)
            time.sleep(45)
        child('execute_and_readback_20260910.py', 'driver')
        execution = json.loads((HERE / 'execution-state-20260910.json').read_text())
        if execution['status'] != 'COMPLETE_LOCAL_READBACK': raise RuntimeError('Driver did not confirm complete readback')
        child('analyze_kv_budget.py', 'analysis')
        save(status='COMPLETE', finished_unix_s=time.time())
    except BaseException as error:
        save(status='STOPPED_FOR_DIAGNOSIS', error=repr(error))
        raise


if __name__ == '__main__': main()

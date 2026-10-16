#!/usr/bin/env python3
"""Serial cells under the established public GPU flock; no automatic retries."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--wait-lock-seconds',type=int,default=1800)
    args=p.parse_args();plan=json.loads(args.plan.read_text())
    session=Path(plan['session']);session.mkdir(parents=True,exist_ok=False)
    receipt=dict(status='WAITING_LOCK',pid=os.getpid(),start_unix_s=time.time(),cells=[],
                 plan=plan,plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest())
    write(session/'receipt.json',receipt)
    lock=open('/root/autodl-tmp/moe-research-gpu.lock','a+')
    acquired=False
    try:
        deadline=time.monotonic()+args.wait_lock_seconds
        while True:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True;break
            except BlockingIOError:
                if time.monotonic()>=deadline:raise TimeoutError('Public GPU lock wait expired')
                time.sleep(2)
        receipt.update(status='RUNNING',acquired_unix_s=time.time(),lock_identity=[os.fstat(lock.fileno()).st_dev,os.fstat(lock.fileno()).st_ino])
        write(session/'receipt.json',receipt)
        for i,cell in enumerate(plan['cells']):
            free=shutil.disk_usage('/root/autodl-tmp').free
            if free<10*2**30:raise RuntimeError('Data disk below 10 GiB; no cell started')
            processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
            if processes:raise RuntimeError('GPU occupied despite lock: '+processes)
            out=session/f'cell-{i:02d}-{cell["policy"]}'
            cache=session/'cache'/str(i);cache.mkdir(parents=True)
            env={k:v for k,v in os.environ.items() if not k.startswith(('A_','B_','H1_','VLLM_','HF_','HUGGINGFACE_','TRANSFORMERS_'))}
            for key in ('PYTHONPATH','PYTHONHOME','PYTHONINSPECT','PYTHONUSERBASE','VIRTUAL_ENV'):
                env.pop(key,None)
            env.update(PYTHONPATH=plan['runtime_overlay'],LD_LIBRARY_PATH=plan['library_path'],
                     HF_HOME=plan['hf_home'],PYTHONDONTWRITEBYTECODE='1',
                     HF_HUB_CACHE=plan['hf_home']+'/hub',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                     VLLM_USE_FLASHINFER_SAMPLER='0',PYTHONNOUSERSITE='1',
                     TMPDIR=str(cache),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),
                     TRITON_CACHE_DIR=str(cache/'triton'),VLLM_CACHE_ROOT=str(cache/'vllm'))
            cmd=[sys.executable,'-B',str(Path(__file__).with_name('run_cell.py')),
                 '--inputs',plan['inputs'],'--output',str(out)]
            for k,v in cell.items():cmd += ['--'+k.replace('_','-'),str(v)]
            row=dict(index=i,command=cmd,start_unix_s=time.time(),disk_free=free)
            receipt['cells'].append(row);write(session/'receipt.json',receipt)
            with (session/f'cell-{i:02d}.log').open('w') as log:
                child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                row['pid']=child.pid;write(session/'receipt.json',receipt)
                try:code=child.wait(timeout=plan.get('per_cell_seconds',450))
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    code=124
            row.update(exit_code=code,end_unix_s=time.time());write(session/'receipt.json',receipt)
            if code:raise RuntimeError('Cell failed; remainder unrun')
        receipt['status']='COMPLETE'
    except BaseException as e:
        receipt.update(status='STOPPED',error=repr(e))
    finally:
        receipt['end_unix_s']=time.time();write(session/'receipt.json',receipt)
        if acquired:fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()
    archive=session.with_suffix('.raw.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for path in session.rglob('*'):
            if path.is_file() and 'cache' not in path.relative_to(session).parts:
                tar.add(path,arcname=str(Path(session.name)/path.relative_to(session)))
    write(session/'archive.json',dict(path=str(archive),bytes=archive.stat().st_size,
          sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))
    print(json.dumps(receipt))
    return 0 if receipt['status']=='COMPLETE' else 1


if __name__=='__main__':raise SystemExit(main())

"""Bounded serialized GPU campaign, retaining every failure and cumulative usage."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil


def write(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--run',required=True)
    p.add_argument('--gammas',type=int,nargs='+')
    p.add_argument('--wait-lock-seconds',type=int,default=1800)
    p.add_argument('--batches',type=int,nargs='+')
    p.add_argument('--output-tokens',type=int)
    p.add_argument('--repeats',type=int)
    p.add_argument('--skip-trace',action='store_true')
    args=p.parse_args()
    plan=json.loads(Path(args.config).read_text())
    if not Path(plan['workload']).is_file():
        raise FileNotFoundError(plan['workload'])
    out=Path('runs')/args.run
    out.mkdir(parents=True,exist_ok=False)
    shutil.copytree('code',out/'source',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(args.config,out/'plan.json')
    state={'status':'waiting_for_lock','pid':os.getpid(),'started_unix_s':time.time(),
           'args':vars(args),'cells':[]}
    write(out/'status.json',state)
    lock=open('/root/autodl-tmp/moe-research-gpu.lock','a')
    deadline=time.monotonic()+args.wait_lock_seconds
    while True:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic()>=deadline:
                state['status']='lock_wait_expired';write(out/'status.json',state);return
            time.sleep(10)
    acquired=time.monotonic()
    state.update(status='running',acquired_unix_s=time.time())
    write(out/'status.json',state)
    previous=sum(json.loads(p.read_text()).get('gpu_lock_seconds',0) for p in Path('runs').glob('*/status.json') if p!=out/'status.json')
    allowance=min(plan['group_gpu_budget_seconds'],plan['total_gpu_budget_seconds']-previous)
    try:
        for gamma in (args.gammas or plan['gammas']):
            remain=allowance-(time.monotonic()-acquired)
            if remain<=0:
                state['status']='budget_exhausted';break
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu,power.draw','--format=csv,noheader'],text=True).strip()
            if apps:
                raise RuntimeError('GPU has active compute processes despite lock: '+apps)
            cell={'gamma':gamma,'start_unix_s':time.time(),'gpu_before':gpu}
            state['cells'].append(cell);write(out/'status.json',state)
            command=[sys.executable,'-u',str(out/'source'/'run_matrix.py'),'--config',str(out/'plan.json'),
                '--gamma',str(gamma),'--output',str(out/f'gamma{gamma}')]
            for flag in ['output_tokens','repeats']:
                if getattr(args,flag) is not None:
                    command.extend(['--'+flag.replace('_','-'),str(getattr(args,flag))])
            if args.batches:
                command.extend(['--batches',*map(str,args.batches)])
            if args.skip_trace:
                command.append('--skip-trace')
            env=dict(os.environ,MOE_S_GPU_LOCKED='1',PYTHONUNBUFFERED='1')
            cuda_lib='/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib'
            env['LD_LIBRARY_PATH']=cuda_lib+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
            with (out/f'gamma{gamma}.log').open('w') as log:
                proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                cell['pid']=proc.pid;write(out/'status.json',state)
                try:
                    cell['returncode']=proc.wait(timeout=min(remain,3600))
                except subprocess.TimeoutExpired:
                    # This process group belongs only to this campaign.
                    import signal
                    os.killpg(proc.pid,signal.SIGTERM)
                    try: proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    cell['returncode']=124
            cell['end_unix_s']=time.time()
            state['gpu_lock_seconds']=time.monotonic()-acquired
            write(out/'status.json',state)
            if cell['returncode']!=0:
                state['status']='needs_implementation_fix';break
        else:
            state['status']='complete'
    except BaseException as exc:
        state.update(status='error',error=repr(exc));raise
    finally:
        state['gpu_lock_seconds']=time.monotonic()-acquired
        state['end_unix_s']=time.time();write(out/'status.json',state)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__': main()

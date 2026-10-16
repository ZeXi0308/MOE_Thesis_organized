"""Run the route-driven layer calibration within the same GPU budget and lock."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from lock_wait import acquire_exclusive


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--routes',nargs='+',required=True)
    p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('--wait-lock-seconds',type=int,default=600)
    p.add_argument('--layer',type=int,default=0)
    p.add_argument('--batches',default='1,2,4,8,16,32,64')
    p.add_argument('--samples-per-cell',type=int,default=1)
    p.add_argument('--include-token-extremes',action='store_true')
    p.add_argument('--profile',action='store_true')
    p.add_argument('--execution-mode',choices=['eager','cudagraph'],default='eager')
    args=p.parse_args()
    out=Path('runs')/args.run
    out.mkdir(parents=True,exist_ok=False)
    shutil.copytree('code',out/'source',ignore=shutil.ignore_patterns('__pycache__'))
    state={'status':'waiting_for_lock','pid':os.getpid(),'args':vars(args),'started_unix_s':time.time()}
    def save():
        temp=out/'status.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(out/'status.json')
    save()
    with open('/root/autodl-tmp/moe-research-gpu.lock','a') as lock:
        if not acquire_exclusive(lock,args.wait_lock_seconds):
            state['status']='lock_wait_expired';save();return
        start=time.monotonic()
        state.update(status='running',acquired_unix_s=time.time())
        proc=None
        try:
            used=sum(json.loads(path.read_text()).get('gpu_lock_seconds',0)
                for path in Path('runs').glob('*/status.json') if path!=out/'status.json')
            budget=min(args.timeout,72000-used)
            if budget<=0:
                state['status']='budget_exhausted';return
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if apps:
                raise RuntimeError('GPU occupied despite lock: '+apps)
            env=dict(os.environ,LD_LIBRARY_PATH='/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib'+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else ''))
            cmd=[sys.executable,'-u',str(out/'source/bench_moe_routes.py'),'--model',args.model,
                '--routes',*args.routes,'--layer',str(args.layer),'--samples-per-cell',str(args.samples_per_cell),
                '--batches',args.batches,'--repeats','5','--execution-mode',args.execution_mode,
                '--out',str(out/'measurements.jsonl')]
            if args.profile:cmd.append('--profile')
            if args.include_token_extremes:cmd.append('--include-token-extremes')
            state['command']=cmd;save()
            with (out/'driver.log').open('w') as log:
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                state['child_pid']=proc.pid;save()
                try:
                    code=proc.wait(timeout=budget)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    code=124
            state.update(status='complete' if code==0 else 'needs_implementation_fix',returncode=code)
        except BaseException as exc:
            state.update(status='error',error=repr(exc));raise
        finally:
            state.update(gpu_lock_seconds=time.monotonic()-start,end_unix_s=time.time());save()
            fcntl.flock(lock,fcntl.LOCK_UN)


if __name__=='__main__':main()

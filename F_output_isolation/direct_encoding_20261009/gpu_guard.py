"""Hold the shared research GPU lock around one bounded child process."""
import argparse, fcntl, json, os, subprocess, sys, time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--record',type=Path,required=True)
    p.add_argument('--seconds',type=int,required=True); p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args(); command=a.command[1:] if a.command[:1]==['--'] else a.command
    assert command and not a.record.exists()
    lock=open('/root/autodl-tmp/moe-research-gpu.lock','a+')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('LOCK_BUSY_NO_GPU_INITIALIZED',flush=True); return 75
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip()
    if occupied: raise RuntimeError('GPU already occupied: '+occupied)
    ids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().splitlines()
    assert len(ids)==1,ids
    env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=ids[0],OMP_NUM_THREADS='8',
        OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_USE_FLASHINFER_SAMPLER='0')
    record={'started_unix':time.time(),'guard_pid':os.getpid(),'gpu_uuid':ids[0],
            'lock':'/root/autodl-tmp/moe-research-gpu.lock','command':command,'max_seconds':a.seconds}
    a.record.parent.mkdir(parents=True,exist_ok=True)
    try:
        child=subprocess.Popen(command,env=env); record['child_pid']=child.pid
        a.record.write_text(json.dumps(record,indent=2)+'\n')
        try: record['exit_code']=child.wait(timeout=a.seconds)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            record.update(exit_code=124,timed_out=True)
    finally:
        record['ended_unix']=time.time()
        a.record.write_text(json.dumps(record,indent=2)+'\n')
    return record['exit_code']

if __name__=='__main__':sys.exit(main())

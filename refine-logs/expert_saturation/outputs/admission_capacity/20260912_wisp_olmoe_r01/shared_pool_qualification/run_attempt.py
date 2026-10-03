"""One retained kernel-interface qualification; no retry."""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent;out=root/'results';out.mkdir(exist_ok=False)
record=dict(status='RUNNING',started_unix_s=time.time(),cells=[])
def save():(out/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
cmd=[sys.executable,'-u',str(root/'source/qualify_shared_expert_pool.py'),'--output',str(out/'interface'),'--prepared',str(root/'prepared'),'--model','/root/autodl-tmp/hf-cache/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5']
cell=dict(label='interface',status='RUNNING',command=cmd,started_unix_s=time.time(),cpu_affinity=list(range(8)));record['cells'].append(cell);save()
try:
 cell['process_start_perf_ns']=time.perf_counter_ns()
 with (out/'interface.log').open('x') as log:
  res=subprocess.run(['taskset','-c','0-7','timeout','-s','TERM','600']+cmd,env=dict(os.environ,OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',PYTHONPATH='/root/autodl-tmp/wisp-pager-smoke-20260912/source/src:'+str(root/'source')),stdout=log,stderr=subprocess.STDOUT)
 cell['process_end_perf_ns']=time.perf_counter_ns();cell['process_wall_s']=(cell['process_end_perf_ns']-cell['process_start_perf_ns'])/1e9;cell.update(returncode=res.returncode,finished_unix_s=time.time(),status='COMPLETE' if res.returncode==0 else 'FAILED')
 record['status']='COMPLETE' if res.returncode==0 else 'STOPPED'
 if res.returncode:raise RuntimeError('qualification failed; inspect retained artifacts')
except BaseException as exc:record.update(status='STOPPED',error=f'{type(exc).__name__}: {exc}');raise
finally:record['finished_unix_s']=time.time();save()

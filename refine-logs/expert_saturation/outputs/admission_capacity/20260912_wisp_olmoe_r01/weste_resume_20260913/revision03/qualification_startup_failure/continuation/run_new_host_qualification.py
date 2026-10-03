"""New-host fullstage/oneshot lifecycle qualification; terminal failure retained without retry."""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
out=root/'results';out.mkdir(exist_ok=False)
record=dict(status='RUNNING',started_unix_s=time.time(),cells=[])
def save():
    (out/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
save()
try:
    for mode in ('fullstage','oneshot'):
        label=mode+('20' if mode=='fullstage' else '21')
        cmd=[sys.executable,'-u',str(root/'source/run_shared_pool_pager.py'),'--pool-execution',mode,'--pool-qualification',
            '--output',str(out/label),'--prepared',str(root/'prepared'),
            '--model','/root/autodl-tmp/hf-cache/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5',
            '--expert-cap','24','--execution','expert','--verify-kernel','--requests','4','--prompt-tokens','64','--output-tokens','8',
            '--token-budget','160','--kv-bytes','1073741824','--max-seconds','900']
        cell=dict(label=label,status='RUNNING',command=cmd,started_unix_s=time.time(),cpu_affinity=list(range(8)))
        record['cells'].append(cell);save();cell['process_start_perf_ns']=time.perf_counter_ns()
        with (out/(label+'.log')).open('x') as f:
            result=subprocess.run(['taskset','-c','0-7','timeout','-s','TERM','1000',*cmd],stdout=f,stderr=subprocess.STDOUT,
                env=dict(os.environ,OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',
                         PYTHONPATH='/root/autodl-tmp/wisp-pager-smoke-20260912/source/src:'+str(root/'source')))
        cell.update(process_end_perf_ns=time.perf_counter_ns(),returncode=result.returncode,finished_unix_s=time.time(),
                    status='COMPLETE' if result.returncode==0 else 'FAILED')
        cell['process_wall_s']=(cell['process_end_perf_ns']-cell['process_start_perf_ns'])/1e9;save()
        if result.returncode:raise RuntimeError(label+' failed; no automatic retry')
    record['status']='COMPLETE'
except BaseException as exc:
    record.update(status='STOPPED',error=repr(exc));raise
finally:
    record['finished_unix_s']=time.time();save()

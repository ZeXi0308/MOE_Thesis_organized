import hashlib,json,os,signal,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent
protocol=json.loads((p/'protocol.json').read_text())
for name,sha in protocol['package_files'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==sha,name
installed=Path('/root/autodl-tmp/expert-saturation/vllm-0.26/lib/python3.12/site-packages/vllm')
for name,sha in protocol['installed_source_sha256'].items():assert hashlib.sha256((installed/name).read_bytes()).hexdigest()==sha,name
out=Path('/root/autodl-tmp/qwen3-localized-static-v026-r01');out.mkdir(exist_ok=False)
shards=out/'shard_workspace';shards.mkdir()
python='/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python'
env=dict(os.environ,OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',WISP_PLUGIN_DISABLE='1',WISP_PREFETCH='0',WISP_DYNAMIC='0',PYTHONPATH='/root/autodl-tmp/wisp-runtime-0112-r01/source/src:'+str(p))
launch=dict(status='RUNNING',start_unix_s=time.time(),results=str(out),qualification_only=False, startup_numerical_localization=True)
def save(): (p/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
save()
def read_memory():
    c=Path('/sys/fs/cgroup')
    stat={k:int(v) for k,v in (line.split() for line in (c/'memory.stat').read_text().splitlines())}
    events={k:int(v) for k,v in (line.split() for line in (c/'memory.events').read_text().splitlines())}
    return dict(current=int((c/'memory.current').read_text()),limit=int((c/'memory.max').read_text()),stat=stat,events=events)
def launchers():
    found=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:argv=(entry/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        if len(argv)>1 and b'python' in Path(argv[0].decode(errors='replace')).name.encode() and any(Path(x.decode(errors='replace')).name.startswith(('run_native_pager','run_wisp','run_session','run_expert_union')) and x.endswith(b'.py') for x in argv[1:]):found.append(int(entry.name))
    return found
def descendant(pid):
    for _ in range(32):
        if pid==process.pid:return True
        if pid<=1:return False
        try:
            status=Path('/proc',str(pid),'status').read_text()
            pid=int(next(line.split()[1] for line in status.splitlines() if line.startswith('PPid:')))
        except (OSError,StopIteration,ValueError):return False
    return False
import pynvml as nv
nv.nvmlInit();gpu=nv.nvmlDeviceGetHandleByIndex(0);exit_code=0;process=None
try:
    with (p/'preflight.jsonl').open('x') as log:
        idle=0
        for tick in range(121):
            query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
            if query.returncode:raise RuntimeError('GPU query failed: '+query.stderr)
            active=launchers();idle=idle+1 if not query.stdout.strip() and not active else 0
            state=dict(unix_s=time.time(),gpu=query.stdout.strip(),launchers=active,consecutive_idle=idle)
            log.write(json.dumps(state)+'\n');log.flush()
            if tick%6==0 or idle==3:print(json.dumps(dict(preflight=state)),flush=True)
            if idle==3:break
            time.sleep(5)
        else:raise RuntimeError('GPU remained occupied; no initialization attempted')
    memory_start=read_memory();disk=os.statvfs(out);disk_available=disk.f_bavail*disk.f_frsize
    assert memory_start['limit']==98784247808 and memory_start['stat']['anon']<8*2**30
    assert disk_available>6*2**30
    launch.update(memory_start=memory_start,disk_available_before=disk_available)
    command=['taskset','-c','0-7',python,'-u',str(p/'run_native_pager.py'),'--output',str(out/'comparison'),'--prepared',str(p/'prepared'),'--model',str(p/'model_metadata'),'--qwen-static-prefill','--qualification-prepared',str(p/'qualification_prepared'),'--expert-cap','48','--requests','4','--token-budget','64','--kv-bytes','536870912','--prompt-tokens','64','--output-tokens','8','--arrival-interval','0.05','--prefill-limit','32','--trace-retention','episode','--execution','expert','--max-seconds','600']
    for flag,value in [('manifest',p/'qwen3.manifest.json'),('index',p/'qwen3.index.json'),('workspace',shards),('receipt',out/'loader_receipt.jsonl')]:command+=['--loader-'+flag,str(value)]
    launch.update(command=command,child_start_unix_s=time.time());save()
    with (p/'run.log').open('x') as log,(p/'hardware.jsonl').open('x') as hw:
        process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        launch['child_pid']=process.pid;save();print(json.dumps(dict(child_pid=process.pid,command=command)),flush=True)
        tick=0
        while process.poll() is None:
            memory=read_memory();info=nv.nvmlDeviceGetMemoryInfo(gpu)
            procs=[dict(pid=x.pid,used_bytes=x.usedGpuMemory) for x in nv.nvmlDeviceGetComputeRunningProcesses(gpu)]
            foreign=[x['pid'] for x in procs if not descendant(x['pid'])]
            row=dict(unix_s=time.time(),processes=procs,foreign=foreign,used_bytes=info.used,temperature_c=nv.nvmlDeviceGetTemperature(gpu,nv.NVML_TEMPERATURE_GPU),power_mw=nv.nvmlDeviceGetPowerUsage(gpu),memory=memory)
            hw.write(json.dumps(row)+'\n');hw.flush()
            if foreign:raise RuntimeError('Foreign GPU process arrived; stopping only our process group')
            if memory['stat']['anon']>84*2**30:raise RuntimeError('Host anonymous memory exceeds84GiB guard within92GiB cgroup')
            if any(memory['events'].get(k,0)>memory_start['events'].get(k,0) for k in ('oom','oom_kill')):raise RuntimeError('Host OOM event increased')
            if time.time()-launch['child_start_unix_s']>7200:raise RuntimeError('7200s complete loading/localization/comparison process limit exceeded')
            if tick%40==0:
                receipt=out/'loader_receipt.jsonl'
                recent=receipt.read_text().splitlines()[-1:] if receipt.exists() else []
                current=[dict(name=x.name,bytes=x.stat().st_size) for x in shards.rglob('*.safetensors')]
                print(json.dumps(dict(progress_s=time.time()-launch['child_start_unix_s'],gpu_used_bytes=info.used,host_current_bytes=memory['current'],host_anon_bytes=memory['stat']['anon'],loader_latest=recent,temporary_shards=current)),flush=True)
            tick+=1;time.sleep(.5)
    exit_code=process.returncode
except BaseException as exc:
    if process is not None and process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    launch['error']=f'{type(exc).__name__}: {exc}';exit_code=1
finally:
    if process is not None:launch.update(child_exit_code=process.poll(),child_end_unix_s=time.time())
    nv.nvmlShutdown();launch.update(status='EXITED',exit_code=exit_code,end_unix_s=time.time())
    launch['parent_wall_s']=launch['end_unix_s']-launch['start_unix_s'];save();print(json.dumps(launch),flush=True)
raise SystemExit(exit_code)

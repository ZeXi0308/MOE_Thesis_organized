from pathlib import Path
import datetime,json,subprocess
root=Path('/root/autodl-tmp')
py=root/'expert-saturation/vllm-0.26/bin/python'
code='''import importlib.metadata as m,json
out={}
for name in ('torch','vllm','transformers','huggingface_hub'):
 try: out[name]=m.version(name)
 except m.PackageNotFoundError: out[name]=None
print(json.dumps(out))
'''
p=subprocess.run([str(py),'-c',code],capture_output=True,text=True)
packages=json.loads(p.stdout) if p.returncode==0 else {'probe_error':p.stderr}
cache=root/'moe-model-cache-01a07d4b/hub/models--allenai--OLMoE-1B-7B-0924'
snapshot=cache/'snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5'
index=snapshot/'model.safetensors.index.json'
weights=json.loads(index.read_text()) if index.exists() else {}
names=sorted(set(weights.get('weight_map',{}).values()))
shards=[{'name':n,'exists':(snapshot/n).exists(),'bytes':(snapshot/n).stat().st_size if (snapshot/n).exists() else None} for n in names]
model_root=root/'moe-model-cache-01a07d4b'
partial=[{'name':p.name,'bytes':p.stat().st_size} for p in (model_root/'downloads').glob('*.part')]
verification_path=model_root/'verification.json'
verification=json.loads(verification_path.read_text()) if verification_path.exists() else {}
jobs={}
for name,pid,ticks in (('model',5085,'484136788'),('pip',2497,'483773493')):
    proc=Path('/proc',str(pid))
    try:
        stat=(proc/'stat').read_text().split()
        jobs[name]={'pid':pid,'start_ticks':stat[21],'state':stat[2],
                    'live':stat[21]==ticks and stat[2]!='Z' and bool((proc/'cmdline').read_bytes())}
    except FileNotFoundError:
        jobs[name]={'pid':pid,'live':False,'handle_missing':True}
campaign=root/'moe-waiting-bypass-01a07d4b-20260910-r01'
artifacts={block:[str(p.relative_to(campaign)) if p.is_relative_to(campaign) else str(p)
                  for p in (campaign/f'{block}-supervisor.json',campaign/'results'/block,
                            campaign/f'{block}-archive.json',root/f'moe-waiting-bypass-01a07d4b-{block}.tar.gz')
                  if p.exists()] for block in ('forward','reverse')}
packages_ready=packages.get('vllm')=='0.26.0' and all(packages.get(n) for n in ('torch','transformers','huggingface_hub'))
terminal=[]
if not jobs['model']['live'] and verification.get('status')!='COMPLETE_VERIFIED':
    terminal.append('model download process ended without complete verification')
if not jobs['pip']['live'] and not packages_ready:
    terminal.append('known installation process ended without required packages')
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True)
result={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'packages':packages,
        'tensor_payload_bytes':weights.get('metadata',{}).get('total_size'),
        'expected_weight_file_bytes':4997744872+4997235176+3843741912,
        'shards':shards,'partial_downloads':partial,
        'model_ready':verification.get('status')=='COMPLETE_VERIFIED' and bool(names) and all(s['exists'] for s in shards) and not jobs['model']['live'],
        'environment_ready':packages_ready and not jobs['pip']['live'],
        'preparation_jobs':jobs,'terminal_preparation_errors':terminal,
        'model_verification_status':verification.get('status'),'artifacts_by_block':artifacts,
        'gpu_processes':gpu.stdout,'gpu_query_exit':gpu.returncode,
        'own_campaign_launched':(root/'moe-waiting-bypass-01a07d4b-20260910-r01/forward-supervisor.json').exists()}
print(json.dumps(result,indent=2))

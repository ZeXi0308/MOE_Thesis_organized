import hashlib,importlib.metadata as md,inspect,json,os,shutil,subprocess,sys,time
from pathlib import Path
stage=Path('/root/autodl-tmp/qwen3-localized-static-v026-launch-r01')
protocol=json.loads((stage/'protocol.json').read_text())
import torch,vllm,safetensors,pynvml
r={'unix_s':time.time(),'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'cuda_build':torch.version.cuda,'vllm':vllm.__version__,'packages':{d.metadata['Name']:d.version for d in md.distributions() if d.metadata.get('Name')}}
installed=Path(vllm.__file__).parent
r['installed_sources']={name:hashlib.sha256((installed/name).read_bytes()).hexdigest() for name in protocol['installed_source_sha256']}
r['installed_sources_match']=r['installed_sources']==protocol['installed_source_sha256']
r['torch_memory_sha256']=hashlib.sha256(Path(inspect.getfile(torch.cuda.memory)).read_bytes()).hexdigest()
sys.path.insert(0,'/root/autodl-tmp/wisp-runtime-0112-r01/source/src');os.environ['WISP_PLUGIN_DISABLE']='1'
import wisp.integrations.vllm.fused_moe as fm
r['wisp_fused_moe_sha256']=hashlib.sha256(Path(fm.__file__).read_bytes()).hexdigest()
r['gpu_processes']=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,timeout=15).stdout
r['disk']={str(q):shutil.disk_usage(q)._asdict() for q in [Path('/root'),Path('/root/autodl-tmp')]}
r['status']='PASS' if (r['installed_sources_match'] and r['torch_memory_sha256']=='71a36cb635c0a076bf8cda0ae08c112d7b0e15d9d2fb21a0dee373e2967c4700' and r['wisp_fused_moe_sha256']=='5572d4f05593a5a9fc4adaa14421cc596886a435b6f5f51f476a1dae6e521857' and r['torch']=='2.11.0+cu130' and r['cuda_build']=='13.0' and r['vllm']=='0.26.0') else 'FAIL'
r['cpu_model'] = next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),None)
r['affinity']=sorted(os.sched_getaffinity(0))
print(json.dumps(r))

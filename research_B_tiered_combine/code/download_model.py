"""Fetch a public, pinned model into this project's directory; no GPU use."""
import argparse,fnmatch,json,pathlib,shutil,os
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
from huggingface_hub import HfApi,snapshot_download
p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--name',required=True)
p.add_argument('--endpoint',default='https://hf-mirror.com');p.add_argument('--max-gib',type=float,default=12)
a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[1]
if not a.name.replace('_','').replace('-','').isalnum():raise ValueError('invalid local name')
out=root/'runs'/f'model_source_{a.name}';out.mkdir(parents=True,exist_ok=False)
api=HfApi(endpoint=a.endpoint,token=False)
info=api.model_info(a.repo,files_metadata=True)
patterns=['*.safetensors','*.json','*.model','*.txt','README.md','LICENSE*','*.tiktoken']
files=[s for s in info.siblings if any(fnmatch.fnmatch(s.rfilename,g) for g in patterns)]
if not any(s.rfilename.endswith('.safetensors') for s in files):raise ValueError('No safetensors weights; inspect repo before choosing fallback')
if any(s.size is None for s in files):raise ValueError('Missing file sizes')
size=sum(s.size for s in files)
manifest={'repo':a.repo,'revision':info.sha,'endpoint':a.endpoint,'files':[{'path':s.rfilename,'size':s.size,'blob_id':s.blob_id} for s in files],'bytes':size,'status':'DOWNLOADING'}
(out/'source.json').write_text(json.dumps(manifest,indent=2))
if size>a.max_gib*2**30 or shutil.disk_usage(root).free<size+2*2**30:raise RuntimeError('Download exceeds size/free-disk limit')
print(a.repo,info.sha,size,flush=True)
dest=root/'models'/a.name
a_path=snapshot_download(a.repo,revision=info.sha,endpoint=a.endpoint,token=False,
                         local_dir=dest,allow_patterns=[s.rfilename for s in files],max_workers=2)
missing=[s.rfilename for s in files if not (dest/s.rfilename).is_file() or (dest/s.rfilename).stat().st_size!=s.size]
if missing:raise RuntimeError(f'Incomplete files: {missing}')
for name in ['config.json','tokenizer_config.json','generation_config.json']:
    if (dest/name).exists():shutil.copy2(dest/name,out/name)
manifest['status']='COMPLETE';manifest['local_dir']=str(dest)
(out/'source.json').write_text(json.dumps(manifest,indent=2));print('COMPLETE',flush=True)

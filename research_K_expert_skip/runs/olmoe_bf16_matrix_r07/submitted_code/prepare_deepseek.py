"""Pinned public DeepSeek-V2-Lite download, CPU/network only, space reserved."""
import argparse,fnmatch,hashlib,json,os,shutil,time,fcntl
from pathlib import Path
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
from huggingface_hub import HfApi,snapshot_download


def allocated(path):
    total=0;seen=set()
    for f in path.rglob('*'):
        if f.is_file():
            s=f.stat();key=(s.st_dev,s.st_ino)
            if key not in seen:total+=s.st_blocks*512;seen.add(key)
    return total


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/root/autodl-tmp/research_K_expert_skip')
    p.add_argument('--endpoint',default='https://hf-mirror.com');p.add_argument('--name',default='deepseek_source_r01')
    a=p.parse_args();root=Path(a.root);out=root/'runs'/a.name;out.mkdir(parents=True,exist_ok=False)
    lock=(root/'deepseek_download.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest={'status':'PREPARING','scope':'public model download; no GPU','pid':os.getpid(),'start_unix':time.time()}
    def save():(out/'source.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    try:
        api=HfApi(endpoint=a.endpoint,token=False)
        repo='deepseek-ai/DeepSeek-V2-Lite';info=api.model_info(repo,files_metadata=True)
        patterns=['*.safetensors','*.json','*.model','*.txt','*.jinja','README.md','LICENSE*']
        files=[f for f in info.siblings if any(fnmatch.fnmatch(f.rfilename,x) for x in patterns)]
        if any(f.size is None for f in files):raise ValueError('missing file sizes')
        needed=sum(f.size for f in files)
        if not 20*2**30<needed<40*2**30:raise ValueError('unexpected checkpoint size')
        reservations=[]
        shared=Path('/root/autodl-tmp')
        croot=shared/'research_C_format_sched/checkpoints'
        for directory in ['qwen3_bf16_ad44e777','qwen3_awq_c3b14f75','qwen3_nvfp4_2538ded2']:
            d=croot/directory
            expected=json.loads((d/'model.safetensors.index.json').read_text())['metadata']['total_size']+30*2**20
            present=allocated(d)
            reservations.append({'name':directory,'expected_bytes':expected,'allocated_bytes':present,'remaining_bytes':max(0,expected-present)})
        broot=shared/'research_B_tiered_combine'
        for src in (broot/'runs').glob('model_source_*/source.json'):
            s=json.loads(src.read_text());name=src.parent.name.removeprefix('model_source_')
            present=allocated(broot/'models'/name)
            reservations.append({'name':name,'expected_bytes':s['bytes'],'allocated_bytes':present,'remaining_bytes':max(0,s['bytes']-present)})
        asource=json.loads((shared/'research_A_horizon/models/qwen3-8b-source.json').read_text())
        ai=api.model_info(asource['id'],revision=asource['sha'],files_metadata=True)
        expected=sum(f.size for f in ai.siblings if f.size is not None and any(fnmatch.fnmatch(f.rfilename,x) for x in patterns))
        present=allocated(shared/'research_A_horizon/models/hf')
        reservations.append({'name':'Qwen3-8B','expected_bytes':expected,'allocated_bytes':present,'remaining_bytes':max(0,expected-present)})
        outstanding=sum(x['remaining_bytes'] for x in reservations);free=shutil.disk_usage(root).free
        manifest.update(repo=repo,revision=info.sha,endpoint=a.endpoint,bytes=needed,
            files=[{'path':f.rfilename,'size':f.size,'lfs_sha256':f.lfs.sha256 if f.lfs else None} for f in files],
            other_download_reservations=reservations,free_bytes=free,remaining_shared_bytes=outstanding,reserve_bytes=20*2**30)
        save()
        if free-needed-outstanding<20*2**30:raise RuntimeError('Insufficient space after reserving other active downloads and 20GiB free')
        dest=root/'models'/('deepseek_v2_lite_'+info.sha[:8]);manifest.update(status='DOWNLOADING',local_dir=str(dest));save()
        snapshot_download(repo,revision=info.sha,endpoint=a.endpoint,token=False,local_dir=dest,
                          allow_patterns=[f.rfilename for f in files],max_workers=1)
        for f in files:
            path=dest/f.rfilename
            if path.stat().st_size!=f.size:raise RuntimeError('Size mismatch '+f.rfilename)
            if f.lfs:
                h=hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(8*2**20),b''):h.update(chunk)
                if h.hexdigest()!=f.lfs.sha256:raise RuntimeError('LFS hash mismatch '+f.rfilename)
        manifest.update(status='COMPLETE',config=json.loads((dest/'config.json').read_text()),ended_unix=time.time());save()
        print(json.dumps({k:manifest[k] for k in ['status','repo','revision','local_dir','bytes']}),flush=True)
    except BaseException as e:
        manifest.update(status='FAILED',error=repr(e),ended_unix=time.time());save();raise


if __name__=='__main__':main()

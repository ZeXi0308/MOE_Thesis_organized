#!/usr/bin/env python3
"""Download the exact existing A base model into a private HF cache; no GPU use."""
from pathlib import Path
import concurrent.futures, hashlib, json, subprocess, time
BASE=Path('/root/autodl-tmp/moe-a-20261002')
REV='6d84c48581ece794365f2b8e9cfb043c68ade9c5'
DEST=BASE/'hf/hub/models--allenai--OLMoE-1B-7B-0924/snapshots'/REV
FILES={
'config.json':(None,'3643aa880d2f1c9b418156269ae791c73e5612d6b6b6fde0724d927cf89b6335'),
'generation_config.json':(None,'d77272ffaa7e62a904e8e130bb25ab11585bd4a5026e388d6d682e4b82892ce2'),
'model.safetensors.index.json':(None,'0e2e1e0d8d357ac7af817cff28410c3dbad398f060c517a433e4076b2aae5579'),
'special_tokens_map.json':(None,'b77491e270c6fcc5b2ecf22370f7318a6a18d3cabea09ba7bab92e9bf12656c2'),
'tokenizer.json':(None,'a094266ac6c4982efba277bc251349a5a6d6ad37efb39a2a90f53d8be2a40a40'),
'tokenizer_config.json':(None,'78a839c7851f14f9fb30e664c2b46166dc0628f2900679e5ec160656f702edff'),
'model-00001-of-00003.safetensors':(4997744872,'5e3cff7e367794685c241169072c940d200918617d5e2813f1c387dff52d845e'),
'model-00002-of-00003.safetensors':(4997235176,'15ef5c730ee3cfed7199498788cd2faf337203fc74b529625e7502cdd759f4a7'),
'model-00003-of-00003.safetensors':(3843741912,'a9abac4ac1b55c9adabac721a02fa39971f103eea9a65c310972b1246de76e04')}
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def one(item):
 name,(size,sha)=item;p=DEST/name
 if p.exists():assert digest(p)==sha;return name
 part=p.with_suffix(p.suffix+'.part')
 url=f'https://hf-mirror.com/allenai/OLMoE-1B-7B-0924/resolve/{REV}/{name}'
 print('DOWNLOAD',name,flush=True)
 subprocess.run(['curl','-fL','--retry','2','--connect-timeout','20','--max-time','7200','--silent','--show-error','-C','-','-o',str(part),url],check=True)
 assert size is None or part.stat().st_size==size,name
 assert digest(part)==sha,name
 part.rename(p);print('VERIFIED',name,p.stat().st_size,flush=True);return name
DEST.mkdir(parents=True,exist_ok=True)
started=time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:done=list(pool.map(one,FILES.items()))
(BASE/'model-download-receipt.json').write_text(json.dumps(dict(status='COMPLETE',model='allenai/OLMoE-1B-7B-0924',revision=REV,files=done,started=started,finished=time.time()),indent=2)+'\n')

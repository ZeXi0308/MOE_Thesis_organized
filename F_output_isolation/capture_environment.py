"""Record the installed output path and CPU context without querying a GPU."""
import hashlib, importlib.metadata, json, os, platform, shutil, subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
v=Path(importlib.metadata.distribution('vllm').locate_file('vllm'))
paths=['_version.py','envs.py','v1/engine/async_llm.py','v1/engine/output_processor.py','v1/engine/logprobs.py','v1/engine/detokenizer.py','entrypoints/openai/api_server.py','entrypoints/serve/utils/server_utils.py','utils/gc_utils.py','entrypoints/openai/chat_completion/serving.py','entrypoints/openai/chat_completion/protocol.py','entrypoints/openai/engine/protocol.py']
receipt={'versions':{m:importlib.metadata.version(m) for m in ['vllm','torch','transformers','tokenizers','pydantic','fastapi','uvicorn','uvloop','aiohttp']},'python':platform.python_version(),'platform':platform.platform(),'cpu_topology':subprocess.check_output(['lscpu'],text=True),'affinity':sorted(os.sched_getaffinity(0)),'installed_source':{},'prototype':{},'gpu_used':False,'note':'One pinned frontend CPU; separate client and producer cores. Shared host, not exclusive CPU reservation. No GPU access or model loading.'}
for rel in paths:
 p=v/rel;dest=root/'native_sources'/rel;dest.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(p,dest)
 receipt['installed_source'][rel]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
for p in root.glob('*.py'):receipt['prototype'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
(root/'environment.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt['versions']))

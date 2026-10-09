"""Run the two frozen CPU workloads once, preserving code identity and stdout."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
if __name__ == '__main__':
    for load in ('high','medium'):
        assert not (HERE/load).exists(), 'Refusing to overwrite '+str(HERE/load)
    identity = {}
    paths = [HERE/x for x in ('gate.py','server.py','run.py','semantic_check.py','protocol.md','execute.py')]
    paths += [ROOT/x for x in ('native_adapter.py','bootstrap.py','inputs/high.json','inputs/medium.json')]
    for path in paths:
        identity[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (HERE/'execution_hashes.json').write_text(json.dumps(identity,indent=2)+'\n')
    old = json.loads((ROOT/'review_20261008/runtime_identity.json').read_text())
    runtime = {}
    installed = Path(sys.prefix)/'lib/python3.12/site-packages/vllm'
    for name,entry in old['installed_output_sources_match_original'].items():
        value = hashlib.sha256((installed/name).read_bytes()).hexdigest()
        assert value == entry['current'], 'Installed source changed: '+name
        runtime[name] = value
    (HERE/'runtime_identity.json').write_text(json.dumps({'installed_sources_sha256':runtime,
        'versions':{m:importlib.metadata.version(m) for m in
            ('vllm','torch','transformers','tokenizers','pydantic','fastapi','uvicorn','aiohttp')}},indent=2)+'\n')
    records = []
    with (HERE/'execution.log').open('w') as log:
        for load in ('high','medium'):
            command = [sys.executable,str(HERE/'run.py'),'--trace',str(ROOT/'inputs'/f'{load}.json'),
                       '--output',str(HERE/load)]
            record = {'load':load,'command':command,'started_unix':time.time()}
            records.append(record)
            process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            record['pid'] = process.pid
            (HERE/'execution.json').write_text(json.dumps(records,indent=2)+'\n')
            for line in process.stdout:
                log.write(line); log.flush(); print(line,end='',flush=True)
            record.update(exit_code=process.wait(),ended_unix=time.time())
            (HERE/'execution.json').write_text(json.dumps(records,indent=2)+'\n')
            if record['exit_code']:
                raise SystemExit(record['exit_code'])

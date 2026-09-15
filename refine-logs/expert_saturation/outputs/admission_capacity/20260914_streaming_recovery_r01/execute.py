"""Stage/run one frozen open-EOS diagnostic group; never retry an existing run."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

BASE = Path(__file__).resolve().parent
REMOTE = '/root/autodl-tmp/moe-streaming-recovery-20260914-r01'
PYTHON = '/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python'
HOST, PORT = 'root@connect.westc.seetacloud.com', '53036'
CONTROL = '/private/tmp/moe-logical-westc-53036-r08.sock'
GPU = 'GPU-70fa1c0a-77d4-c14a-9daf-7e685874eef9'
SSH = ['ssh', '-p', PORT, '-S', CONTROL, '-o', 'BatchMode=yes', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4', HOST]
SCP = ['scp', '-P', PORT, '-o', 'ControlPath='+CONTROL, '-o', 'BatchMode=yes']
WRAPPER = '''import json,os,pathlib,subprocess,sys,time
r=pathlib.Path.cwd()
with (r/'launch-once').open('x') as f:f.write(str(os.getpid()))
s=dict(status='STARTING',pid=os.getpid(),started_unix_s=time.time())
def save():
 p=r/'group-status.json.tmp';p.write_text(json.dumps(s,indent=2)+'\\n');p.replace(r/'group-status.json')
save()
try:
 with (r/'campaign.log').open('x') as log:
  p=subprocess.Popen(['bash','pkg/run.sh',sys.executable],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  s.update(status='RUNNING',shell_pid=p.pid);save();rc=p.wait()
 cells=json.loads((r/'pkg/campaign.json').read_text())['cells']
 s['cells']=[dict(label=c['label'],terminal=json.loads((r/'results'/c['label']/'status.json').read_text()) if (r/'results'/c['label']/'status.json').exists() else dict(status='UNRUN')) for c in cells]
 s.update(status='COMPLETE' if rc==0 and all(c['terminal']['status']=='COMPLETE' for c in s['cells']) else 'STOPPED',returncode=rc)
except BaseException as exc:
 s.update(status='STOPPED',error=f'{type(exc).__name__}: {exc}');raise
finally:
 s['finished_unix_s']=time.time();save()
sys.exit(0 if s['status']=='COMPLETE' else 1)
'''


def query(code):
    return json.loads(subprocess.check_output(SSH+[shlex.join([PYTHON, '-c', code])], text=True, timeout=30))


def main():
    mode = sys.argv[1]
    assert mode in ('stage', 'run')
    prep, out = BASE/'preparation', BASE/'execution'
    metadata = json.loads((prep/'preparation.json').read_text())
    data = (prep/'execution.tar.gz').read_bytes()
    assert hashlib.sha256(data).hexdigest() == metadata['archive_sha256']
    assert all(hashlib.sha256((prep/'pkg'/n).read_bytes()).hexdigest() == h for n,h in metadata['files_sha256'].items())
    if mode == 'stage':
        out.mkdir()
        state = dict(status='PREFLIGHT', archive_sha256=metadata['archive_sha256'], remote_dir=REMOTE, host=HOST,
                     started_unix_s=time.time(), driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    else:
        state = json.loads((out/'execution.json').read_text())
        assert state['status'] == 'STAGED' and state['archive_sha256'] == metadata['archive_sha256']
        state['run_driver_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def save():
        p=out/'execution.json.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(out/'execution.json')
    lock=out/'.driver.lock'
    with lock.open('x') as f:f.write(mode)
    try:
        check = query(f'''import hashlib,json,pathlib,subprocess,torch,vllm,transformers
root=pathlib.Path(vllm.__file__).parent
print(json.dumps(dict(gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip(),processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),torch=torch.__version__,vllm=vllm.__version__,transformers=transformers.__version__,sources={{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in {list(metadata['expected_runtime_sources'])!r}}})))''')
        assert check['gpu']==GPU and not check['processes'], 'wrong GPU or busy; no launch'
        assert (check['torch'],check['vllm'],check['transformers'])==('2.11.0+cu130','0.26.0','5.15.1')
        assert check['sources']==metadata['expected_runtime_sources'], 'runtime sources changed'
        state['preflight_'+mode]=check;save()
        if mode=='stage':
            subprocess.run(SSH+[shlex.join(['mkdir',REMOTE])],check=True)
            subprocess.run(SCP+[str(prep/'execution.tar.gz'),HOST+':'+REMOTE+'/execution.tar.gz'],check=True)
            got=query(f"import hashlib,json,pathlib;print(json.dumps(hashlib.sha256((pathlib.Path({REMOTE!r})/'execution.tar.gz').read_bytes()).hexdigest()))")
            assert got==metadata['archive_sha256']
            subprocess.run(SSH+[shlex.join(['tar','-xzf',REMOTE+'/execution.tar.gz','-C',REMOTE])],check=True)
            state['status']='STAGED';return
        verified=query(f'''import hashlib,json,pathlib
r=pathlib.Path({REMOTE!r})
print(json.dumps(dict(files={{n:hashlib.sha256((r/'pkg'/n).read_bytes()).hexdigest() for n in {list(metadata['files_sha256'])!r}}},untouched=not (r/'launch-once').exists() and not (r/'results').exists())))''')
        assert verified['files']==metadata['files_sha256'] and verified['untouched'], 'source changed or remote already started'
        payload=base64.b64encode(WRAPPER.encode()).decode()
        got=query(f"import base64,hashlib,json,pathlib;p=pathlib.Path({REMOTE!r})/'execute_group.py';b=base64.b64decode({payload!r});p.open('xb').write(b);print(json.dumps(hashlib.sha256(b).hexdigest()))")
        assert got==hashlib.sha256(WRAPPER.encode()).hexdigest()
        state.update(status='RUNNING',wrapper_sha256=got);save()
        with (out/'ssh.log').open('x') as log:
            rc=subprocess.run(SSH+['cd '+shlex.quote(REMOTE)+' && '+shlex.join([PYTHON,'-u','execute_group.py'])],stdout=log,stderr=subprocess.STDOUT).returncode
        state['ssh_returncode']=rc
        if rc==255:
            state['status']='UNKNOWN_REMOTE';raise RuntimeError('SSH interrupted: inspect existing detached group, never relaunch automatically')
        receipt=query(f'''import hashlib,json,pathlib,tarfile
r=pathlib.Path({REMOTE!r});p=r/'readback.tar.gz';assert not p.exists()
with tarfile.open(p,'w:gz') as t:
 for name in ['pkg','results','campaign.log','group-status.json','execute_group.py']:
  if (r/name).exists():t.add(r/name,arcname=name)
print(json.dumps(dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),group=json.loads((r/'group-status.json').read_text()))))''')
        archive=out/'readback.tar.gz';subprocess.run(SCP+[HOST+':'+REMOTE+'/readback.tar.gz',str(archive)],check=True)
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256'], 'readback hash mismatch'
        with tarfile.open(archive) as t:t.extractall(out/'readback',filter='data')
        state.update(status=receipt['group']['status'],readback_sha256=receipt['sha256'],group=receipt['group'])
        print(json.dumps(dict(status=state['status'],cells=receipt['group'].get('cells',[]))),flush=True)
    except BaseException as exc:
        if state['status']!='UNKNOWN_REMOTE':state['status']='STOPPED'
        state['error']=f'{type(exc).__name__}: {exc}';raise
    finally:
        state['updated_unix_s']=time.time();save();lock.unlink()


if __name__=='__main__':main()

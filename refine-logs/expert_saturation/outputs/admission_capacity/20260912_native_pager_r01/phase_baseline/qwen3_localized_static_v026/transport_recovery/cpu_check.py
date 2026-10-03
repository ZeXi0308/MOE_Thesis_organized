import hashlib,json,os,subprocess,sys,tempfile,time,venv
from pathlib import Path
wrapper=Path('refine-logs/expert_saturation/outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/qwen3_localized_static_v026/transport_recovery/detached_launch.py').resolve()
root=Path(tempfile.mkdtemp(prefix='detached-transport-cpu-')); package=root/'package'; package.mkdir(); receipts=root/'receipts'; receipts.mkdir(); envdir=root/'venv'; venv.EnvBuilder(with_pip=False,symlinks=True).create(envdir); venv_python=envdir/'bin/python'
entry=package/'run_remote.py'
entry.write_text('''import json,os,subprocess,sys,time
from pathlib import Path
p=Path(__file__).resolve().parent
print("FAKE_PARENT_STARTED",flush=True)
code="import json,os,sys,time;from pathlib import Path;time.sleep(.7);Path('child_state.json').write_text(json.dumps(dict(status='FINISHED',pid=os.getpid(),end_unix_s=time.time(),python=sys.executable,prefix=sys.prefix)));print('FAKE_CHILD_FINISHED',flush=True)"
child=subprocess.Popen([sys.executable,'-u','-c',code],cwd=p)
(p/'child_started.json').write_text(json.dumps(dict(pid=child.pid,parent_pid=os.getpid(),session_id=os.getsid(0))))
assert child.wait()==0
print("FAKE_PARENT_FINISHED",flush=True)
''')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=package/'protocol.json'; protocol.write_text(json.dumps(dict(package_files={'run_remote.py':sha(entry)})))
cmd=[sys.executable,str(wrapper),'--package-dir',str(package),'--python',str(venv_python),'--entry-sha256',sha(entry),'--protocol-sha256',sha(protocol),'--receipt-root',str(receipts)]
caller=root/'caller.py'; caller.write_text('import subprocess,os\nsubprocess.run('+repr(cmd)+',check=True)\nos._exit(0)\n')
with (root/'caller.log').open('w') as log:
 result=subprocess.run([sys.executable,str(caller)],stdout=log,stderr=subprocess.STDOUT,timeout=5)
caller_exited=time.time(); assert result.returncode==0
for _ in range(100):
 dirs=list(receipts.iterdir())
 if dirs and (dirs[0]/'receipt.jsonl').exists():
  events=[json.loads(x) for x in (dirs[0]/'receipt.jsonl').read_text().splitlines()]
  if events and events[-1]['status']=='CHILD_EXITED':break
 time.sleep(.05)
else:raise AssertionError('detached parent did not finish')
d=dirs[0]; log=(d/'monitor.log').read_text(); state=json.loads((package/'child_state.json').read_text())
assert state['end_unix_s']>caller_exited
assert all(x in log for x in ('FAKE_PARENT_STARTED','FAKE_CHILD_FINISHED','FAKE_PARENT_FINISHED'))
assert events[-1]['exit_code']==0 and events[0]['monitor_session_id']==events[0]['monitor_pid']
context=json.loads((d/'context.json').read_text()); assert context['python']==str(venv_python); assert Path(state['prefix']).resolve()==envdir.resolve(); assert Path(context['python']).is_symlink()
assert sha(entry)==json.loads(protocol.read_text())['package_files']['run_remote.py']
print(json.dumps(dict(status='CPU_PASS',caller_exited_before_child_finish=True,monitor_owns_new_session=True,logged_parent_and_child_finish=True,exit_code=0,entry_unchanged=True,venv_symlink_prefix_preserved=True,requested_interpreter=str(venv_python),receipt_dir=str(d),fake_child_state=str(package/'child_state.json'),wrapper_sha256=sha(wrapper)),indent=2))

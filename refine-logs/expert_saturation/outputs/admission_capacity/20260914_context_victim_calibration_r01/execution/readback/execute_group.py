import json,os,pathlib,subprocess,sys,time
r=pathlib.Path.cwd()
with (r/'launch-once').open('x') as f:f.write(str(os.getpid()))
s=dict(status='STARTING',pid=os.getpid(),started_unix_s=time.time())
def save():
 p=r/'group-status.json.tmp';p.write_text(json.dumps(s,indent=2)+'\n');p.replace(r/'group-status.json')
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

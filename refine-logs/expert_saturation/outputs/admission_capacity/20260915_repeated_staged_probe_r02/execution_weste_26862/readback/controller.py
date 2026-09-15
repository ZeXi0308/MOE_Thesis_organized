import json,os,pathlib,subprocess,time
r=pathlib.Path(__file__).resolve().parent
with (r/'launch-once').open('x') as f:f.write(str(os.getpid()))
s=dict(status='STARTING',pid=os.getpid(),started=time.time())
def save():
 p=r/'group-status.json.tmp';p.write_text(json.dumps(s,indent=2));p.replace(r/'group-status.json')
save()
try:
 with (r/'campaign.log').open('x') as log:
  child=subprocess.Popen(['bash','pkg/run.sh'],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  s.update(status='RUNNING',shell_pid=child.pid);save();rc=child.wait()
 cells={}
 for arm in ('off','on'):
  p=r/'results'/('save-'+arm)/'status.json'
  cells[arm]=json.loads(p.read_text()) if p.exists() else dict(status='UNRUN')
 s.update(returncode=rc,cells=cells,status='COMPLETE' if rc==0 and all(c['status']=='COMPLETE' for c in cells.values()) else 'STOPPED')
except BaseException as exc:
 s.update(status='STOPPED',error=repr(exc));raise
finally:
 s['finished']=time.time();save()

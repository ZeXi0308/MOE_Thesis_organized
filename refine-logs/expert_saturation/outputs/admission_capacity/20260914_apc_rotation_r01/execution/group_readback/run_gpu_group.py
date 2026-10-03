import fcntl,hashlib,json,os,pathlib,subprocess,sys,tarfile,time
root=pathlib.Path.cwd()
record=dict(status="STARTING",pid=os.getpid(),started_unix_s=time.time(),cells=[])
def save():
 p=root/"group-execution.json.tmp";p.write_text(json.dumps(record,indent=2)+"\n");p.replace(root/"group-execution.json")
with (root/"group-once.lock").open("x") as once:
 once.write(str(os.getpid()))
with pathlib.Path("/root/autodl-tmp/moe-research-gpu.lock").open("a+") as lock:
 try:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  record["shared_lock_acquired_unix_s"]=time.time();save()
  assert not (root/"results").exists() or not any((root/"results").iterdir()),"existing results; no rerun"
  with tarfile.open(root/"execution.tar.gz") as archive:
   assert all((root/m.name).read_bytes()==archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()),"source changed"
  cells=json.loads((root/"campaign.json").read_text())["cells"]
  record["status"]="RUNNING";save()
  for cell in cells:
   gpu=subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True).strip()
   assert gpu==sys.argv[1],"GPU UUID changed"
   processes=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
   assert not processes,"GPU busy; abort group"
   row=dict(label=cell["label"],status="RUNNING",started_unix_s=time.time(),gpu_uuid=gpu,gpu_processes=processes)
   record["cells"].append(row);save()
   rc=subprocess.run([sys.executable,"-u","run_one_cell.py",cell["label"]],pass_fds=(lock.fileno(),),start_new_session=True).returncode
   terminal_path=root/"results"/cell["label"]/"status.json"
   row.update(returncode=rc,finished_unix_s=time.time(),terminal=json.loads(terminal_path.read_text()) if terminal_path.exists() else {})
   row["status"]="COMPLETE" if rc==0 and row["terminal"].get("status")=="COMPLETE" else "INCOMPLETE";save()
   assert row["status"]=="COMPLETE","cell failed; preserve and stop"
  record["status"]="COMPLETE"
 except BaseException as exc:
  record.update(status="STOPPED",error=f"{type(exc).__name__}: {exc}");raise
 finally:
  record["finished_unix_s"]=time.time();save()

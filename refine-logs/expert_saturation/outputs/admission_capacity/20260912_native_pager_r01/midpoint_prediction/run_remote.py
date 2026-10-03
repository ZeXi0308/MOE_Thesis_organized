import pathlib,json,subprocess,os,time,sys,hashlib,shutil
root=pathlib.Path("/root/autodl-tmp/wisp-runtime-0112-r01")
source=root/"injection";out=root/"model-calibration-mid16-r01";out.mkdir(exist_ok=False)
state={"status":"RUNNING","started_unix_s":time.time(),"cells":[],"source_sha256":{}}
for n in ["run_wisp_injection_probe.py","run_wisp_olmoe_probe.py","wisp_paging_trace.py","workload.json"]:
 state["source_sha256"][n]=hashlib.sha256((source/n).read_bytes()).hexdigest()
def save():(out/"execution.json").write_text(json.dumps(state,indent=2)+"\n")
save()
try:
 for rep in [0,1]:
  apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader"],text=True).strip()
  if apps:raise RuntimeError("GPU occupied before midpoint cell: "+apps)
  cell=out/("r"+str(rep)+"-long16");cell.mkdir(exist_ok=False)
  cmd=[str(root/"venv/bin/python"),str(source/"run_wisp_injection_probe.py"),"--model","/root/autodl-tmp/hf-cache/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5","--workload",str(source/"workload.json"),"--out",str(cell/"result.json"),"--chunk","16","--new-prompt-length","128","--trace"]
  row={"label":cell.name,"command":cmd,"status":"RUNNING","started_unix_s":time.time()};state["cells"].append(row);save()
  env=dict(os.environ,OMP_NUM_THREADS="8",TOKENIZERS_PARALLELISM="false",PYTHONUNBUFFERED="1",PYTHONPATH=str(root/"source-traced/src")+":"+str(source))
  with (cell/"run.log").open("x") as log:code=subprocess.run(["timeout","-s","TERM","600"]+cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
  row.update(returncode=code,status=json.loads((cell/"result.json").read_text()).get("status") if (cell/"result.json").exists() else "NO_RESULT",finished_unix_s=time.time());save()
  if code or row["status"]!="COMPLETED":raise RuntimeError("midpoint cell failed")
 state["status"]="COMPLETE"
except BaseException as e:state.update(status="STOPPED",error=repr(e));raise
finally:state["finished_unix_s"]=time.time();save();print(json.dumps(state),flush=True)


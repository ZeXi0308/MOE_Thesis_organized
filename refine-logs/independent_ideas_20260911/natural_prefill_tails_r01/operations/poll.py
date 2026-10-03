from pathlib import Path
import json,subprocess,os
root=Path('/root/autodl-tmp/moe-natural-prefill-01a07d4b-20260911-r01')
out={}
for block in ('forward','reverse'):
 b={}
 for name in (f'{block}-supervisor.json',f'results/{block}-process.json',f'results/{block}/status.json'):
  p=root/name
  if p.exists():
   try: b[name]=json.loads(p.read_text())
   except ValueError:b[name]='incomplete JSON during write'
 for key in (f'{block}-supervisor.json',f'results/{block}-process.json'):
  state=b.get(key)
  if isinstance(state,dict) and state.get('pid'):
   p=Path('/proc',str(state['pid']),'stat')
   try:
    stat=p.read_text().split();b[key+'_process_alive']=stat[2]!='Z' and ('proc_start_ticks' not in state or state['proc_start_ticks']==stat[21])
   except FileNotFoundError:b[key+'_process_alive']=False
 folder=root/'results'/block
 b['raw_files']=[p.name for p in folder.glob('*-raw.json')]
 for name in ('stderr.log','stdout.log','failure.txt'):
  p=folder/name
  if p.exists():b[name+'_tail']=p.read_text(errors='replace')[-750:]
 out[block]=b
out['gpu']=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout
print(json.dumps(out,indent=2))

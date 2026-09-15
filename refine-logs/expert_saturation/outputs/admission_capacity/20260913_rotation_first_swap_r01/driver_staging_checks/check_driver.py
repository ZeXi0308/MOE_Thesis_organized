#!/usr/bin/env python3
"""Mocked orchestration only: no SSH, SCP, GPU, or new F cell evidence."""
import contextlib, hashlib, importlib.util, io, json, pathlib, shlex, subprocess, sys, tarfile, tempfile
from types import SimpleNamespace
from unittest.mock import patch
HERE=pathlib.Path(__file__).resolve().parent
E=HERE.parents[3]/'experiments/admission_capacity'
spec=importlib.util.spec_from_file_location('driver',E/'run_frozen_kv_remote.py');driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
T=pathlib.Path(tempfile.mkdtemp(prefix='mocked-',dir=HERE));source=T/'source';source.mkdir()
labels=[f'mock-cell-{i}' for i in range(6)]
def archive(entries):
 stream=io.BytesIO()
 with tarfile.open(fileobj=stream,mode='w:gz') as tar:
  for name,value in entries.items():
   value=value.encode();item=tarfile.TarInfo(name);item.size=len(value);tar.addfile(item,io.BytesIO(value))
 return stream.getvalue()
(source/'execution.tar.gz').write_bytes(archive({'campaign.json':json.dumps({'cells':[{'label':x} for x in labels]}),'run_one_cell.py':'# MOCK ONLY\n','inputs_preparation/README':'MOCK ONLY\n'}))
sha=lambda b:hashlib.sha256(b).hexdigest()
class Remote:
 def __init__(self):self.busy='5127';self.uploads=0;self.cells=[];self.queries=[];self.archive=None;self.bad_sha=False;self.source_matches=True;self.activity=False;self.shards={}
 def check_output(self,cmd,**kw):
  assert cmd[0]=='ssh';code=shlex.split(cmd[-1])[-1]
  if 'import torch,vllm,transformers' in code:
   self.queries.append('preflight');result=dict(torch='2.11.0+cu130',vllm='0.26.0',transformers='5.15.1',gpu='GPU-mock RTX5090',processes=self.busy,shards=[])
  elif 'source_matches' in code:
   self.queries.append('staged');result=dict(sha256='bad' if self.bad_sha else sha(self.archive),source_matches=self.source_matches,results_empty=not(self.activity or self.cells))
  else:
   self.queries.append('hash');name=next((x for x in labels if x+'.tar.gz' in code),None);result=sha(self.shards[name] if name else self.archive)
  return json.dumps(result)
 def run(self,cmd,**kw):
  assert cmd[0] in ('ssh','scp')
  if cmd[0]=='scp':
   if ':' in cmd[-1]:self.uploads+=1;self.archive=pathlib.Path(cmd[-2]).read_bytes()
   else:
    label=next(x for x in labels if x+'.tar.gz' in cmd[-2]);pathlib.Path(cmd[-1]).write_bytes(self.shards[label])
  elif 'run_one_cell.py' in cmd[-1]:
   label=shlex.split(cmd[-1])[-1];assert label not in self.cells;self.cells.append(label)
   self.shards[label]=archive({label+'/status.json':json.dumps(dict(status='COMPLETE',evidence_type='MOCK_ONLY'))})
  return SimpleNamespace(returncode=0)

def invoke(out,remote,mode=None,changes=None):
 args=dict(source=str(source),output=str(out),host='root@mock',port='1',control_path='/mock/socket',gpu_uuid='GPU-mock',remote_dir='/mock/staged')
 args.update(changes or {});argv=['driver']
 for key,value in args.items():argv+=['--'+key.replace('_','-'),value]
 argv+=['--labels',*labels]
 if mode:argv+=['--'+mode]
 with patch.object(sys,'argv',argv),patch.object(driver.subprocess,'check_output',side_effect=remote.check_output),patch.object(driver.subprocess,'run',side_effect=remote.run),contextlib.redirect_stdout(io.StringIO()):driver.main()

def rejected(fn):
 try:fn()
 except (AssertionError,RuntimeError,FileNotFoundError):return
 raise AssertionError('expected rejection')
def state(out):return json.loads((out/'execution.json').read_text())
checks=[];r=Remote();out=T/'staged'
invoke(out,r,'stage-only');assert state(out)['status']=='STAGED' and state(out)['cells']==[] and r.uploads==1 and not r.cells
assert json.loads((out/'preflight.json').read_text())['processes']=='5127'
checks.append('busy_stage_uploads_once_and_executes_zero_cells')
rejected(lambda:invoke(out,r,'resume-staged'));assert state(out)['status']=='STAGED' and not state(out)['cells'] and not r.cells and r.uploads==1 and r.queries.count('preflight')==2 and r.queries.count('staged')==1
checks.append('busy_resume_rechecks_preflight_remote_package_and_preserves_STAGED')
for key,value in [('host','root@other'),('port','2'),('gpu_uuid','GPU-other'),('remote_dir','/mock/other'),('source',str(T/'other-source'))]:
 old=len(r.queries);rejected(lambda key=key,value=value:invoke(out,r,'resume-staged',{key:value}));assert len(r.queries)==old and state(out)['status']=='STAGED'
checks.append('source_host_port_GPU_remote_dir_identity_rejected_before_remote_calls')
labels.reverse();rejected(lambda:invoke(out,r,'resume-staged'));labels.reverse();checks.append('label_order_mismatch_rejected')
for path in (source/'execution.tar.gz',out/'execution.tar.gz',out/'frozen/run_one_cell.py'):
 old=path.read_bytes();path.write_bytes(old+b'changed');rejected(lambda:invoke(out,r,'resume-staged'));path.write_bytes(old);assert state(out)['status']=='STAGED'
checks.append('source_archive_local_archive_local_frozen_tampering_rejected')
r.busy='';r.bad_sha=True;rejected(lambda:invoke(out,r,'resume-staged'));r.bad_sha=False
r.source_matches=False;rejected(lambda:invoke(out,r,'resume-staged'));r.source_matches=True
checks.append('remote_archive_or_extracted_source_mismatch_rejected')
invoke(out,r,'resume-staged');assert state(out)['status']=='COMPLETE' and len(state(out)['cells'])==6 and r.cells==labels and r.uploads==1
checks.append('valid_resume_runs_frozen_six_once_without_second_upload')
rejected(lambda:invoke(out,r,'resume-staged'));assert r.cells==labels and r.uploads==1
checks.append('completed_or_started_campaign_cannot_resume')
for kind in ('remote','local'):
 other=T/(kind+'-activity');rr=Remote();invoke(other,rr,'stage-only');rr.busy=''
 if kind=='remote':rr.activity=True
 else:(other/'gpu_results/mock-existing').write_text('MOCK activity')
 rejected(lambda:invoke(other,rr,'resume-staged'));assert state(other)['status']=='STOPPED_'+kind.upper()+'_ACTIVITY' and not rr.cells
checks.append('existing_remote_or_local_cell_activity_permanently_stops_resume')
normal=T/'default';rn=Remote();rn.busy='';invoke(normal,rn);assert state(normal)['status']=='COMPLETE' and rn.uploads==1 and rn.cells==labels
checks.append('default_upload_then_execute_path_unchanged')
busy=T/'default-busy';rb=Remote();rejected(lambda:invoke(busy,rb));assert state(busy)['status']=='STOPPED' and rb.uploads==0 and not rb.cells
checks.append('default_busy_GPU_still_blocks_upload_and_execution')
attempts=list(T.glob('*/attempts/*/attempt.json'))
assert all(json.loads(p.read_text()).get('driver_source_sha256')==sha((E/'run_frozen_kv_remote.py').read_bytes()) for p in attempts)
assert any(json.loads(p.read_text())['status']=='REJECTED_OR_FAILED' for p in attempts)
assert not list(T.glob('*/.driver.lock'))
checks.append('all_attempts_keep_driver_hash_status_failure_and_release_lock')
report=dict(status='PASS_MOCKED_ORCHESTRATION_ONLY',checks=checks,fixture_dir=str(T),real_remote_commands=0,new_F_GPU_cells=0,driver_sha256=sha((E/'run_frozen_kv_remote.py').read_bytes()))
with (HERE/'checks.json').open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
print(json.dumps(report,indent=2))

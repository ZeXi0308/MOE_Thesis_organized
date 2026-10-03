"""Check native mutation sites and unscheduled-suffix victim safety without GPU."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parent/'candidate_native_residency_victim_r01/pkg'
sys.path.insert(0,str(ROOT))
from rotation_native import patched_schedule_tree
source=(ROOT.parents[1]/'liveness_pinned_sources_20260930/scheduler.py').read_text()
patched=patched_schedule_tree(source); compile(patched,'<patched native>','exec')
text=ast.unparse(patched)
assert text.count('self._rotation_pick_victim(req_index)')==1
assert text.count('self._rotation_admit_running(request)')==1
assert 'preempted_req = self.running.pop()' not in text
# Execute the actual nested selector body with explicit physical-state substitutes.
tree=ast.parse((ROOT/'staged_store_rotation.py').read_text())
install=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='install')
fn=next(n for n in install.body if isinstance(n,ast.FunctionDef) and n.name=='pick_victim')
def request(rid,arrival,outputs,blocks):
 return NS(request_id=rid,arrival_time=arrival,num_output_tokens=outputs,num_preemptions=1,
           state=NS(pure_decode=True,status='RUNNING',blocks=tuple(range(blocks))))
rows=[request('already_scheduled',100,1000,1),request('current',1,60,100),
      request('arrived_later',9,40,10),request('fresh_restore',3,1,20)]
def pick(rule,unknown=False):
 import time
 residence={r.request_id:(1,0) for r in rows}
 if unknown:residence.pop('arrived_later')
 env=dict(scheduler=NS(running=rows),view=lambda r:r.state,residence=residence,
          victim_rule=rule,data={'victim_decisions':[]},step=12,time=time,
          pool=NS(get_num_free_blocks=lambda:0))
 exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual picker>','exec'),env)
 selected=env['pick_victim'](1)
 assert selected>=1 and 'already_scheduled' in residence
 assert rows[selected].request_id not in residence
 return selected,env['data']['victim_decisions'][0]
assert pick('tail')[0]==3
assert pick('arrival')[0]==2
assert pick('service_density')[0]==2
assert pick('service_density',True)[0]==3
rows[2].state.pure_decode=False
assert pick('arrival')[0]==3
assert pick('service_density')[0]==3
for p in ROOT.glob('*.py'):ast.parse(p.read_text())
print('PASS: pinned AST sites; suffix exclusion; native/arrival/density selection; unknown/mixed fallback')

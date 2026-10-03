"""Exercise the three score choices using the actual frozen selector body."""
from pathlib import Path
import ast,time
from types import SimpleNamespace as N
p=Path('candidate_native_residency_ablation_r01/pkg/staged_store_rotation.py')
install=next(x for x in ast.parse(p.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='install')
fn=next(x for x in install.body if isinstance(x,ast.FunctionDef) and x.name=='pick_victim')
def r(name,outputs,pages):return N(request_id=name,num_output_tokens=outputs,num_preemptions=1,arrival_time=0,state=N(pure_decode=True,status='RUNNING',blocks=tuple(range(pages))))
rows=[r('scheduled',10000,1),r('large_epoch',60,100),r('dense_epoch',40,10),r('recent_restore',101,20)]
for rule,expected in [('service_density',2),('residence_outputs',1),('lifetime_density',3)]:
 for unknown in (False,True):
  residence={x.request_id:(1,100 if x.request_id=='recent_restore' else 0) for x in rows}
  if unknown:residence.pop('dense_epoch')
  env=dict(scheduler=N(running=rows),view=lambda x:x.state,residence=residence,victim_rule=rule,data={'victim_decisions':[]},step=0,time=time,pool=N(get_num_free_blocks=lambda:0))
  exec(compile(ast.Module(body=[fn],type_ignores=[]),'<frozen selector>','exec'),env)
  assert env['pick_victim'](1)==(3 if unknown else expected)
print('PASS: reset and normalization each create intended action difference; unknown suffix returns tail')

"""Exercise the actual new continuation guard and its pinned native exit site."""
import ast
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS

ROOT=Path(__file__).resolve().parent
PKG=ROOT/'candidate_native_self_preempt_continue_r01/pkg'
sys.path.insert(0,str(PKG))
from rotation_native import patched_schedule_tree

source=(ROOT/'liveness_pinned_sources_20260930/scheduler.py').read_text()
patched=patched_schedule_tree(source)
compile(patched,'<pinned native with guarded continuation>','exec')
text=ast.unparse(patched)
assert text.count('self._rotation_continue_after_self_preempt(request, req_index)')==1
assert text.count('if new_blocks is None:')==2
tree=ast.parse((PKG/'staged_store_rotation.py').read_text())
install=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='install')
functions=[]
for name in ('continue_after_self_preempt','hold'):
    fn=next(n for n in install.body if isinstance(n,ast.FunctionDef) and n.name==name)
    fn.body=[ast.Global(names=n.names) if isinstance(n,ast.Nonlocal) else n for n in fn.body]
    functions.append(fn)
module=ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[]))

def exercise(mode='on',reserved=0,protected=None,phase=None,unknown=False):
    current=NS(request_id='current',status=NS(name='PREEMPTED'))
    prefix=NS(request_id='already_scheduled')
    next_request=NS(request_id='original_successor')
    later=NS(request_id='later')
    data={'victim_decisions':[dict(step=17,selected='current',failed_request='current',
                                  fallback_unknown=unknown)],'self_preempt_continuations':[]}
    disposition='DIRECT_READY' if reserved==0 else 'KEEP_NATIVE_RESERVATION'
    env=dict(scheduler=NS(running=[prefix,next_request,later]),step=17,data=data,
             protected=protected,phase=phase,continue_mode=mode,time=time,
             pending_continuation_visit=None,
             _native_reservation_disposition=lambda scheduler,disposition0:(disposition,reserved))
    exec(compile(module,'<actual continuation hook>','exec'),env)
    applied=env['continue_after_self_preempt'](current,1)
    if applied:
        assert env['hold'](next_request) is False
        assert data['self_preempt_continuations'][0]['next_running_visited']=='original_successor'
        assert env['pending_continuation_visit'] is None
    assert data['self_preempt_continuations'][0]['suffix_ids']==['original_successor','later']
    # No later running request is the original tail-only termination case.
    assert env['continue_after_self_preempt'](current,3) is False
    return applied

assert exercise()
assert not exercise(mode='off')
assert not exercise(reserved=1)
assert not exercise(reserved=None)
assert not exercise(protected=NS(request_id='protected'))
assert not exercise(phase='ordinary_backfill')
assert not exercise(unknown=True)
print('PASS: one pinned running exit changed; original successor visited; off/unknown/reservation/protection/phase/tail exits preserved')

"""One-decision CPU checks, reusing the frozen native waiting AST fixture."""
import ast
import copy
import hashlib
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace as NS

import fit_once

B = Path(__file__).resolve().parents[1]
fixture_path = B/'recovery_queue/check_cpu.py'
assert hashlib.sha256(fixture_path.read_bytes()).hexdigest() == '702e80971cbff2c193980dd29d82aa48d38584634b73bd96db681ddcb6000506'
sys.path.insert(0, str(fixture_path.parent))
frozen = runpy.run_path(str(fixture_path))
fixture, fragment, full_tree = frozen['fixture'], frozen['fragment'], frozen['full_tree']


class RemoveFit(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == fit_once.HELPERS.CALLBACK:
            return None
        return self.generic_visit(node)


instrumented = fit_once._instrument(full_tree)
assert ast.dump(RemoveFit().visit(copy.deepcopy(instrumented)), include_attributes=True) == ast.dump(full_tree, include_attributes=True)


def remove_request(self, request):
    self.remove_calls = getattr(self, 'remove_calls', 0)+1
    self.remove(request)


def prepend_request(self, request):
    self.prepend_calls = getattr(self, 'prepend_calls', 0)+1
    self.insert(0, request)


frozen['Queue'].remove_request = remove_request
frozen['Queue'].prepend_request = prepend_request


def setup(mode):
    s, unused, manager, single, cs = fixture()
    head, target = s.waiting
    head.num_tokens = 160  # Need 10 full-history blocks, while free=8; target needs 4.
    wrapper = s.schedule
    slot = fit_once.HELPERS._native_slot(wrapper, s); native = slot.cell_contents
    data, undo = fit_once._attach(s, manager, single, cs, mode, fragment)
    return s, manager, single, cs, data, undo, wrapper, slot, native, head, target


for mode in ('native', 'fit_once'):
    s,manager,single,cs,data,undo,wrapper,slot,native,head,target=setup(mode)
    blocks = dict(cs.manager._policy.blocks)
    result=s.schedule(8,[])
    assert result == ('head' if mode=='native' else 'anchor')
    assert s.schedule is wrapper and len(data['events'])==1
    e=data['events'][0]
    assert e['baseline_head']=='head' and e['candidate_head']=='anchor'
    assert e['queue_changed'] == (mode=='fit_once')
    assert data['action_count'] == (1 if mode=='fit_once' else 0)
    assert e['waiting_before']==['head','anchor']
    assert e['waiting_after']==(['head','anchor'] if mode=='native' else ['anchor','head'])
    assert getattr(s.waiting,'remove_calls',0) == (1 if mode=='fit_once' else 0)
    assert getattr(s.waiting,'prepend_calls',0) == (1 if mode=='fit_once' else 0)
    assert cs.manager._policy.blocks==blocks and all(b.ref_cnt==0 for b in blocks.values())
    assert not single.req_to_blocks and manager.block_pool.free_block_queue.num_free_blocks==8
    # Reset only the test queue to make the same opportunity recur; policy must not act twice.
    s.waiting[:]=[head,target]
    assert s.schedule(8,[])=='head' and len(data['events'])==1
    assert data['action_count'] == (1 if mode=='fit_once' else 0)
    undo(); undo()
    assert s.schedule is wrapper and slot.cell_contents is native and s.schedule(8,[])=='head'

for barrier in ('new_request', 'different_priority'):
    s,manager,single,cs,data,undo,wrapper,slot,native,head,target=setup('fit_once')
    middle=frozen['request']('barrier',0)
    if barrier=='new_request': middle.status=NS(name='WAITING')
    else: middle.priority=1
    s.waiting.insert(1,middle)
    assert s.schedule(8,[])=='head' and not data['events'] and data['outcome']=='NO_OPPORTUNITY'
    assert [r.request_id for r in s.waiting]==['head','barrier','anchor']
    undo()

for reason in ('head_fits','candidate_held','candidate_computed','pending_host','pending_global','unknown_global'):
    s,manager,single,cs,data,undo,wrapper,slot,native,head,target=setup('fit_once')
    if reason=='head_fits': head.num_tokens=64
    if reason=='candidate_held': single.req_to_blocks['anchor']=[NS(is_null=False,ref_cnt=1)]
    if reason=='candidate_computed': target.num_computed_tokens=1
    if reason=='pending_host': cs.manager._policy.blocks[0]=NS(ref_cnt=-1,is_ready=False)
    if reason=='pending_global': cs._chunks_being_loaded={0}
    if reason=='unknown_global': del cs._chunks_being_loaded
    assert s.schedule(8,[])=='head' and not data['events']
    undo()

# Earliest arrival, then ID, only inside the allowed prefix; no original queue scan mutation.
s,manager,single,cs,data,undo,wrapper,slot,native,head,target=setup('fit_once')
other=frozen['request']('earlier',0)
cs._req_status['earlier']=NS(req=other,transfer_jobs=set(),group_states=[NS(offload_keys=list(range(4)))])
s.waiting.append(other)
assert s.schedule(8,[])=='earlier' and data['events'][0]['candidate_head']=='earlier'
undo()

# The original native peek still executes once after the callback and raises unchanged.
s,manager,single,cs,data,undo,wrapper,slot,native,head,target=setup('fit_once')
s.waiting.fail=True
try: s.schedule(8,[])
except ValueError as error: assert str(error)=='native queue exception'
else: raise AssertionError('Native exception swallowed')
assert data['action_count']==1 and data['events'][0]['queue_changed']
undo()
print('PASS: one native AST callback only; shadow/native head unchanged; candidate head changed exactly once; no new-request/priority crossing; ownership/Host refs untouched; fit/source guards; native exceptions and uninstall')

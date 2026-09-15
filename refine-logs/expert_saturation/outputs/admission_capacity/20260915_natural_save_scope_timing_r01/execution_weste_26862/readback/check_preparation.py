"""Necessary sparse timing/EOS boundary checks; reuse existing fixtures."""
import ast
from copy import deepcopy
import importlib
from pathlib import Path
import subprocess
import sys
import unittest

root=Path(__file__).resolve().parent
repo=root.parents[4]
sys.path.insert(0,str(root/'pkg'))
sys.path.append(str(repo/'refine-logs/expert_saturation/experiments/admission_capacity'))
from analyze_scope_timing import sparse_segments, normalize_config

for path in root.rglob('*.py'):
    ast.parse(path.read_text(),filename=str(path))
subprocess.run(['bash','-n',str(root/'pkg/run.sh')],check=True)
subprocess.run([sys.executable,str(root/'pkg/run_save_scope_timing.py'),'--help'],check=True,stdout=subprocess.DEVNULL)
prior=root.parent/'20260915_natural_native_full_gate_r01'
for folder in ('inputs','warmups'):
    for path in (root/'pkg'/folder).rglob('*.json'):
        assert path.read_bytes()==(prior/'pkg'/path.relative_to(root/'pkg')).read_bytes()
for name in ('absence_rotation.py','rotation_native.py','staged_save_contract.py','native_capture.py'):
    assert (root/'pkg'/name).read_bytes()==(prior/'pkg'/name).read_bytes()
def functions(path):
    tree=ast.parse(path.read_text())
    install=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='install')
    return {n.name:ast.dump(n,include_attributes=False) for n in install.body if isinstance(n,ast.FunctionDef)}
old,new=(functions(p/'pkg/staged_store_rotation.py') for p in (prior,root))
for name in ('begin','hold','limited','uninstall'):assert old[name]==new[name]
runner=ast.parse((root/'pkg/run_save_scope_timing.py').read_text())
calls=[n for n in ast.walk(runner) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='measure_episode']
assert len(calls)==1 and any(k.arg=='record_preemptions' and k.value.value is True for k in calls[0].keywords)
tests=importlib.import_module('test_request_measurement')
names=('test_external_policy_cost_and_hook_identity_survive_natural_eos',
    'test_sparse_preempt_records_last_returned_output_and_restores_hook',
    'test_sparse_preempt_failure_retains_partial_result_and_restores_hook')
suite=unittest.TestSuite(tests.RequestMeasurementTest(n) for n in names)
assert unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful()

event=lambda count,at,call:dict(request_id='r',internal_request_id='r',
    original_preemption_returned=True,engine_call_index=call,last_returned_output_count=count,
    last_new_output_s=[.1,.9,1.4][count-1],native_output_count_before=count,
    method_entered_s=at,method_returned_s=at+.01)
raw=dict(requests=[dict(request_id='r',token_times_s=[.1,.9,1.4],status='completed',completion_s=1.5)],
    preemption_events=[event(1,.2,0),event(2,1.1,1)],engine_return_count=2,
    preemption_attempt_count=2,actual_preemption_count=2)
value=sparse_segments(raw)
assert value['counts']['one_two_new_outputs_then_repreempted']==1
assert value['segments'][0]['first_new_output_after_preempt_s']==.9
other=deepcopy(raw)
other['preemption_events']=[event(3,1.45,1)]
last=sparse_segments(other)['segments'][0]
assert last['end']=='completed' and last['returned_new_outputs']==0
assert last['first_new_output_after_preempt_s'] is None
bad=deepcopy(raw);bad['preemption_events'][1]['last_returned_output_count']=4
try:sparse_segments(bad)
except ValueError:pass
else:raise AssertionError('Out-of-range sparse counts must fail')
assert normalize_config(dict(store_scope='selected',cap=32))==normalize_config(dict(store_scope='native_full',cap=32))
assert normalize_config(dict(store_scope='selected',cap=32))!=normalize_config(dict(store_scope='native_full',cap=31))
print('PASS: unchanged workload/common policy;3reused sparse/EOS fixtures; short-service/terminal output accounting and comparison config. GPU UNRUN.')

"""Only new scope branching and native KV-range evidence; EOS checks are reused."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

root = Path(__file__).resolve().parent
repo = root.parents[4]
sys.path.insert(0, str(root/'pkg'))
import staged_store_rotation as adapter
from native_full_store_evidence import inspect_native_full_jobs
from native_store_delta import inspect_store_delta

for p in root.rglob('*.py'):
    ast.parse(p.read_text(), filename=str(p))
subprocess.run(['bash', '-n', str(root/'pkg/run.sh')], check=True)
prior = root.parent/'20260915_natural_saved_recovery_gate_r01'
for folder in ('inputs', 'warmups'):
    for p in (root/'pkg'/folder).rglob('*.json'):
        assert p.read_bytes() == (prior/'pkg'/p.relative_to(root/'pkg')).read_bytes()
for name in ('native_capture.py', 'absence_rotation.py', 'rotation_native.py', 'staged_save_contract.py'):
    assert (root/'pkg'/name).read_bytes() == (prior/'pkg'/name).read_bytes()

# Execute the pinned native pure range functions, without a runtime or GPU.
source_path = repo/'refine-logs/expert_saturation/outputs/admission_capacity/20260914_kv_roundtrip_feasibility_r01/native_offload_source.json'
source = json.loads(source_path.read_text())['distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py']
expected = json.loads((root/'pkg/runtime_source_hashes.json').read_text())['distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py']
assert hashlib.sha256(source.encode()).hexdigest() == expected
nodes = {n.name:n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef)}
tree = ast.parse('from __future__ import annotations\nclass NativeCalc:\n pass\nclass NativeState:\n pass')
tree.body[1].body = [deepcopy(nodes['_calc_num_offloadable_tokens'])]
tree.body[2].body = [deepcopy(nodes['storable_chunks'])]
env = {}
exec(compile(ast.fix_missing_locations(tree), '<pinned-native-range>', 'exec'), env)
cs = env['NativeCalc']()
group_config = NS(tokens_per_chunk=16, is_eagle_group=False)
cs.config = NS(blocks_per_chunk=1, kv_group_configs=[group_config], offload_prompt_only=False)
state = env['NativeState']()
state.req = NS(request_id='r', num_prompt_tokens=16, num_tokens=33,
    num_computed_tokens=32, is_finished=lambda:False)
state.max_offload_tokens = None
state.transfer_jobs = {7}
state.group_states = [NS(block_ids=[101,102], offload_keys=['k0','k1'])]
cs._req_status = {'r':state}
cs._jobs = {7:NS(req_id='r',is_store=True,keys={'k1'},non_sliding_window_block_ids=[102])}
meta = NS(store_jobs={7:NS(req_id='r',src_spec=NS(block_ids=[102]))},jobs_to_flush={7})
output = NS(kv_connector_metadata=meta,num_scheduled_tokens={'r':1})
plan = NS(victim=NS(request_id='r'), saved_tokens=16, source_blocks=(101,))
job = inspect_native_full_jobs(output,cs,plan)['store_jobs'][0]
assert job['beyond_selected_prefix_blocks'] == 1 and job['chunks_containing_decode_positions'] == 1
assert inspect_native_full_jobs(output,cs)['store_jobs'][0]['outside_selected_preparation'] is True
try:
    inspect_store_delta(plan,meta,state,cs._jobs)
except ValueError as error:
    assert 'outside prepared complete prefix' in str(error)
else:
    raise AssertionError('Fixture must distinguish selected and native-full ranges')
state.req.num_computed_tokens = 31
try:
    inspect_native_full_jobs(output,cs)
except ValueError as error:
    assert 'native scheduled/finished range' in str(error)
else:
    raise AssertionError('Scheduled tokens must not be added twice')
state.req.is_finished = lambda:True
state.req.num_tokens = 32
output.num_scheduled_tokens = {}
assert inspect_native_full_jobs(output,cs)['store_jobs'][0]['finished_at_metadata'] is True
cs._jobs[7].keys = {'wrong'}
try:
    inspect_native_full_jobs(output,cs)
except ValueError as error:
    assert 'source-to-key mapping' in str(error)
else:
    raise AssertionError('Mismatched registered key must be rejected')

# Actual scope-install/uninstall closures: full never sets an instance override.
install = next(n for n in ast.parse((root/'pkg/staged_store_rotation.py').read_text()).body
    if isinstance(n,ast.FunctionDef) and n.name=='install')
start = next(i for i,n in enumerate(install.body) if isinstance(n,ast.Assign)
    and any(isinstance(t,ast.Name) and t.id=='step' for t in n.targets))
factory = ast.parse('def fixture(scheduler,cs,store_scope):\n expected_requests=32\n diagnostic=False\n global_cooldown_steps=20\n population_mode="open"\n open_population=True\n save=True\n oldcalc=cs._calc_num_offloadable_tokens\n hadcalc=False\n owned={}\n pool=None\n native=None').body[0]
factory.body += deepcopy(install.body[start:])
space = dict(vars(adapter))
exec(compile(ast.fix_missing_locations(ast.Module(body=[factory],type_ignores=[])),
    '<actual-scope-closures>', 'exec'), space)
for scope in ('native_full','selected'):
    scheduler = NS(requests={})
    original = cs._calc_num_offloadable_tokens.__func__
    data, uninstall = space['fixture'](scheduler,cs,scope)
    assert ('_calc_num_offloadable_tokens' in vars(cs)) == (scope=='selected')
    assert data['native_calc_overridden'] == (scope=='selected')
    uninstall()
    assert '_calc_num_offloadable_tokens' not in vars(cs)
    assert cs._calc_num_offloadable_tokens.__func__ is original
print('PASS: unchanged D inputs/EOS/policy; original native full range, no double counting, source/key checks; scope install/restoration. No EOS rerun or GPU.')

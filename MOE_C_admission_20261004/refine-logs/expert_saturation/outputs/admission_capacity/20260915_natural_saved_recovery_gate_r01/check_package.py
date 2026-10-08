"""Bounded CPU checks of actual open closures, EOS capture and frozen inputs."""
import ast
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest

root = Path(__file__).resolve().parent
repo = root.parents[4]
sys.path.insert(0, str(root/'pkg'))
import staged_store_rotation as adapter
from staged_save_contract import RequestState, prepare
from run_streaming_recovery import load_inputs, KV_BYTES

for path in root.rglob('*.py'):
    ast.parse(path.read_text(), filename=str(path))
subprocess.run(['bash', '-n', str(root/'pkg/run.sh')], check=True)
subprocess.run([sys.executable, str(root/'pkg/run_streaming_recovery.py'), '--help'],
    check=True, stdout=subprocess.DEVNULL)
cfg, workload = load_inputs(root/'pkg/inputs')
prior = repo/'refine-logs/expert_saturation/outputs/admission_capacity/20260914_streaming_recovery_r01/preparation/pkg'
old = json.loads((prior/'inputs/workload.json').read_text())
assert workload['source_requests'] == old['source_requests']
assert workload['actual_prompt_token_ids'] == old['actual_prompt_token_ids']
assert workload['arrival_traces_s']['steady'] == [i*.2 for i in range(64)]
assert cfg['ignore_eos'] is False and cfg['min_tokens'] == 0 and cfg['output_tokens'] == 1024
assert KV_BYTES == 8592031744 and cfg['target_usable_kv_blocks'] == 4096 and cfg['offload_gib'] == 16

# Reuse the actual adapter closures while replacing allocation/transfers with empty CPU metadata.
tree = ast.parse((root/'pkg/staged_store_rotation.py').read_text())
install = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'install')
start = next(i for i,n in enumerate(install.body) if isinstance(n,ast.Assign)
    and any(isinstance(t,ast.Name) and t.id=='step' for t in n.targets))
factory = ast.parse('def fixture(scheduler,native,owned,pool,cs):\n expected_requests=32\n oldcalc=cs._calc_num_offloadable_tokens\n hadcalc=False\n save=True\n diagnostic=True\n global_cooldown_steps=20\n open_population=True\n population_mode="open"\n').body[0]
factory.body += deepcopy(install.body[start:])
env = dict(vars(adapter))
exec(compile(ast.fix_missing_locations(ast.Module(body=[factory],type_ignores=[])),
    '<actual-open-staged-closures>', 'exec'), env)
scheduler = NS(requests={},running=[],waiting=[],skipped_waiting=[])
owned, blocks = {}, {}
pool = NS(blocks=blocks,get_num_free_blocks=lambda:100)
cs = NS(_calc_num_offloadable_tokens=lambda r,n:n,_jobs={},_req_status={})
def native():
    scheduler._rotation_begin([], 0.0)
    return NS(num_scheduled_tokens={},preempted_req_ids=[],
        scheduled_cached_reqs=NS(resumed_req_ids=[]),
        kv_connector_metadata=NS(store_jobs={},load_jobs={},jobs_to_flush=set()))
data, uninstall = env['fixture'](scheduler,native,owned,pool,cs)
scheduler.schedule()
assert data['gate_observations'][-1]['gate'] == 'no_preempted_waiter'
late = NS(request_id='late',num_computed_tokens=0,num_prompt_tokens=32,
    num_output_tokens=0,max_tokens=1024,status=NS(name='RUNNING'))
scheduler.requests['late']=late
scheduler.running.append(late)
owned['late']=[]
scheduler.schedule()
assert data['gate_observations'][-1]['gate'] == 'mixed_prefill'
assert data['gate_observations'][-1]['legacy_closed_activation_condition_met'] is False
closures = dict(zip(scheduler.schedule.__code__.co_freevars,scheduler.schedule.__closure__))
terminal = NS(request_id='terminal',num_output_tokens=3,status=NS(name='FINISHED_STOPPED'),is_finished=lambda:True)
closures['protected'].cell_contents=terminal
closures['output_start'].cell_contents=3
scheduler.schedule()
assert data['events'][-1]['event'] == 'target_terminal' and data['events'][-1]['new_output_tokens'] == 0
# A newly mixed running population cancels an otherwise valid next-boundary commit.
victim = NS(request_id='v',num_computed_tokens=31,num_prompt_tokens=16,
    num_output_tokens=16,max_tokens=1024,status=NS(name='RUNNING'))
target = NS(request_id='t',num_computed_tokens=0,num_prompt_tokens=32,
    num_output_tokens=1,max_tokens=1024,status=NS(name='PREEMPTED'))
for req in (victim,target):scheduler.requests[req.request_id]=req
scheduler.running.append(victim);scheduler.waiting.append(target)
owned['v']=[NS(block_id=i,is_null=False) for i in (1,2)]
owned['t']=[]
blocks.update({b.block_id:b for b in owned['v']})
state = lambda r:RequestState(r.request_id,r.num_computed_tokens,r.num_prompt_tokens,
    r.num_output_tokens,r.max_tokens,r.status.name,tuple(b.block_id for b in owned[r.request_id]))
step=closures['step'].cell_contents
closures['plan'].cell_contents=prepare(step-1,state(victim),state(target),100)
victim.num_computed_tokens+=1;victim.num_output_tokens+=1
scheduler.schedule()
assert [e for e in data['events'] if e['event']=='commit_check'][-1]['reason'] == 'CANCEL_OPEN_MIXED_PREFILL_OR_RECOVERY'
assert data['applied_rotations'] == 0
uninstall()

# Existing three EOS/late-arrival/output-boundary tests, using this package's capture.
sys.path.append(str(repo/'refine-logs/expert_saturation/experiments/admission_capacity'))
tests = importlib.import_module('test_native_capture_eos')
capture = importlib.import_module('native_capture')
tests.capture_module, tests.capture_episode = capture, capture.capture_episode
outcome = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(tests.EosCaptureTest))
assert outcome.wasSuccessful()
print('PASS: frozen documents/0.2s/budgets; syntax/CLI; actual open lifecycle/cancel closures; three reused EOS checks. GPU UNRUN.')

"""Actual child -> normal dynamic installer -> staged guard, without GPU/model.

Only the model-native schedule invocation is a double. The complete child source
construction, install_policy dispatch, dynamic staged installer, its closure and
return wrapper are real pinned code, including the non-file co_filename.
"""
import ast
import copy
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

import exchange_guarded_runtime as runtime

B = Path(__file__).resolve().parents[1]
CHILD = B/'recovery_capacity_exchange/run_cell_guarded.py'
assert hashlib.sha256(CHILD.read_bytes()).hexdigest() == '7c895e603b9b824c24f17acc751905035c61b37a12fde53b17e1df9429e8ddb9'
spec = importlib.util.spec_from_file_location('actual_failed_guard_child', CHILD)
child = importlib.util.module_from_spec(spec); spec.loader.exec_module(child)
captured = {}


class Captured(Exception): pass


def capture(text):
    compile(text, '<complete-actual-child>', 'exec')
    assert text.count('install_rotation = normal_install_rotation') == 1
    assert text.count('from exchange_guarded import install') == 1
    frame = inspect.currentframe().f_back
    while frame is not None:
        if Path(frame.f_globals.get('__file__', '')).resolve() == B/'normal_capacity/run_cell.py':
            captured['normal'] = frame.f_globals
            break
        frame = frame.f_back
    assert frame is not None, 'Did not traverse actual normal-capacity main'
    captured['source'] = text
    raise Captured


with patch.dict(os.environ, B_RECOVERY_CAPACITY_EXCHANGE='exchange_once'):
    try: child.main(capture)
    except Captured: pass
    else: raise AssertionError('Reached model initialization')
assert captured['normal']['normal_install_rotation'].__code__ == runtime.NORMAL.normal_install_rotation.__code__
captured['normal']['FIXED_CAP'] = 256
tree = ast.parse(captured['source'])
dispatch = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'install_policy')
dispatch_ns = {}
exec(compile(ast.Module(body=[dispatch], type_ignores=[]), '<actual-child-dispatch>', 'exec'), dispatch_ns)
SCHEDULER = Path(os.environ.get('B_GUARD_TEST_SCHEDULER_SOURCE',
    str(B.parent/'MOE_Thesis_organized/refine-logs/expert_saturation/experiments/admission_capacity/20260929_commit_recheck/liveness_pinned_sources_20260930/scheduler.py')))
assert hashlib.sha256(SCHEDULER.read_bytes()).hexdigest() == '2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941'
sys.path.insert(0, str(B/'pkg'))
SchedulerOutput = object  # Native schedule return annotation; no vLLM import.


class Scheduler:
    def schedule(self): raise AssertionError('Model-native schedule must not execute')


class CPUConnectorScheduler:
    def _calc_num_offloadable_tokens(self, *args): raise AssertionError('No STORE hook expected')


def fixture(*, mode='exchange_once', changed=None, native_error=False):
    cs = CPUConnectorScheduler()
    cs.config = NS(offload_prompt_only=False, blocks_per_chunk=1, num_workers=1)
    cs._req_status = {}
    s = Scheduler()
    s.max_num_running_reqs = 256
    s.requests = {}; s.running = []; s.waiting = []; s.skipped_waiting = []
    s.num_lookahead_tokens = s.num_spec_tokens = 0
    s.dcp_world_size = s.pcp_world_size = 1
    s.use_v2_model_runner = s.defer_block_free = s.is_encoder_decoder = False
    s.ec_connector = None; s.policy = NS(name='FCFS'); s.scheduler_reserve_full_isl = True
    single = type('FullAttentionManager', (), {})(); single.req_to_blocks = {}
    coordinator = type('KVCacheCoordinatorNoPrefixCache', (), {})()
    coordinator.single_type_managers = [single]
    s.kv_cache_manager = NS(enable_caching=False, num_kv_cache_groups=1, use_eagle=False,
        watermark_blocks=0, coordinator=coordinator, block_pool=NS())
    s.connector = type('OffloadingConnector', (), {})(); s.connector.connector_scheduler = cs
    original_getsourcefile = inspect.getsourcefile
    def sourcefile(obj):
        return str(SCHEDULER) if obj is Scheduler else original_getsourcefile(obj)
    # Execute the very installer selected in the actual child, with its normal
    # namespace, imports, cap transformation and real native AST construction.
    with patch.object(inspect, 'getsourcefile', sourcefile):
        data, undo_staged = dispatch_ns['install_policy'](s,
            NS(scheduler_config=NS(async_scheduling=False), speculative_config=None),
            16, 'native_full_ordinary_only', False, 0,
            captured['normal']['normal_install_rotation'],
            ordinary_backfill=False, oldest_admission_mode='native', oldest_repeat=True)
    staged = s.schedule
    assert staged.__globals__['__name__'] == 'normal_capacity_native_observation'
    assert staged.__code__.co_filename.endswith('[normal_capacity]')
    assert inspect.getsourcefile(staged) is None  # Exact r02 failure condition.
    assert not Path(staged.__code__.co_filename).exists()
    cells = dict(zip(staged.__code__.co_freevars, staged.__closure__))
    # Confirm that install really built the pinned rotation-native method before
    # substituting only its model execution boundary with a deterministic double.
    native = cells['native'].cell_contents
    assert native.__self__ is s and native.__func__.__code__.co_filename == str(SCHEDULER)
    donor = NS(request_id='donor', status=NS(name='PREEMPTED'), num_preemptions=2)
    s.requests['donor'] = donor; cs._req_status['donor'] = NS(req=donor)
    result = NS(num_scheduled_tokens={}, preempted_req_ids={'donor'},
        scheduled_cached_reqs=NS(resumed_req_ids=set()),
        kv_connector_metadata=NS(store_jobs={}, load_jobs={71:object()}, jobs_to_flush={17}))
    exchange = dict(mode=mode, action_count=0, events=[])
    calls = []
    def fake_native():
        calls.append('native_metadata_constructed')
        if native_error: raise ValueError('unchanged native exception')
        s._rotation_forced_count = 1
        exchange.update(action_count=1, events=[dict(kind='action',
            host_perf_s=runtime.GUARD.time.perf_counter(), donor='donor',
            native_preempt_called=True, forced_count=1, donor_num_preemptions=2,
            pending_store_jobs=[17])])
        if changed: changed(s, exchange, result)
        return result
    cells['native'].cell_contents = fake_native
    def outer_factory(previous):
        def outer(): return previous()
        return outer
    s.schedule = outer_factory(staged)
    return NS(s=s, staged=staged, data=data, result=result, exchange=exchange,
        cells=cells, calls=calls, undo_staged=undo_staged)


f = fixture()
try: runtime.GUARD._attach_guard(f.s, f.exchange)
except RuntimeError as error: assert str(error) == 'Missing staged source'
else: raise AssertionError('Old failed adapter unexpectedly accepted runtime path')
try: f.s.schedule()
except RuntimeError as error: assert str(error) == runtime.GUARD.ERROR
else: raise AssertionError('Original native-only guard missing')
assert f.calls == ['native_metadata_constructed']
f.undo_staged()

f = fixture(); outer = f.s.schedule
receipt, undo = runtime._attach_guard(f.s, f.exchange)
assert f.s.schedule() is f.result and receipt['checks'][0]['allowed']
assert f.s._rotation_forced_count == 1 and f.cells['allow_forced_rotations'].cell_contents is False
assert f.data['applied_rotations'] == 0 and receipt['runtime_source_construction']['fixed_cap'] == 256
assert f.data['status'] == 'EXECUTING' and receipt['original_code_verified']
undo(); undo(); assert f.s.schedule is outer and outer.__closure__[0].cell_contents is f.staged
f.undo_staged(); assert 'schedule' not in vars(f.s)

def no_exchange(s, exchange, result):
    s._rotation_forced_count = 0
    exchange.update(action_count=0, events=[])
    result.preempted_req_ids = set()

f = fixture(mode='stall8', changed=no_exchange)
receipt, undo = runtime._attach_guard(f.s, f.exchange)
assert f.s.schedule() is f.result and not receipt['checks']
assert f.exchange['action_count'] == 0 and f.s._rotation_forced_count == 0
undo(); f.undo_staged()

# The dynamically rebuilt source differs at exactly the original predicate.
original = ast.parse(runtime.NORMAL.adapted_rotation_source())
modified = runtime.GUARD._instrument(original)
restored = copy.deepcopy(modified)
hits = [n for n in ast.walk(restored) if isinstance(n, ast.If)
        and runtime.GUARD.CALLBACK in ast.unparse(n.test)]
assert len(hits) == 1; hits[0].test = hits[0].test.values[0]
assert ast.dump(original, include_attributes=True) == ast.dump(restored, include_attributes=True)

for mode, changed in (
    ('stall8', None),
    ('exchange_once', lambda s,e,r:setattr(r, 'preempted_req_ids', {'other'})),
    ('exchange_once', lambda s,e,r:setattr(s, '_rotation_forced_count', 2)),
    ('exchange_once', lambda s,e,r:setattr(r.kv_connector_metadata, 'jobs_to_flush', set())),
):
    f = fixture(mode=mode, changed=changed); receipt, undo = runtime._attach_guard(f.s, f.exchange)
    try: f.s.schedule()
    except RuntimeError as error: assert str(error) == runtime.GUARD.ERROR
    else: raise AssertionError('Broadened native-only guard')
    assert not receipt['checks'][0]['allowed']; undo(); f.undo_staged()

for mutate in (
    lambda f:f.staged.__globals__.update(NORMAL_FIXED_CAP=224),
    lambda f:f.staged.__globals__.update(__name__='unknown_dynamic_wrapper'),
    lambda f:setattr(f.staged, '__code__', f.staged.__code__.replace(co_firstlineno=1)),
):
    f = fixture(); mutate(f); original_outer = f.s.schedule
    try: runtime._attach_guard(f.s, f.exchange)
    except RuntimeError: pass
    else: raise AssertionError('Unknown runtime code/namespace accepted')
    assert f.s.schedule is original_outer; f.undo_staged()

f = fixture(native_error=True); receipt, undo = runtime._attach_guard(f.s, f.exchange)
try: f.s.schedule()
except ValueError as error: assert str(error) == 'unchanged native exception'
else: raise AssertionError('Native exception swallowed')
assert not receipt['checks']; undo(); f.undo_staged()
print('PASS: actual guarded child dispatch + dynamic normal installer reproduces None source; exact runtime code/installer verified; only original donor predicate extended; metadata identity/counters preserved; unknown code rejects; native errors and teardown preserved. CPU only; GPU_UNRUN.')

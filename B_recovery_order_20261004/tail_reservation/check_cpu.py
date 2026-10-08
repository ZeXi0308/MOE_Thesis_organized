"""CPU-only native-AST checks; no vLLM import, model, CUDA, or GPU probe."""
import ast
from collections import defaultdict
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace as NS

import reserve_tail

B = Path(__file__).resolve().parents[1]
FROZEN = B.parent / 'MOE_Thesis_organized/refine-logs/expert_saturation/experiments/admission_capacity/20260929_commit_recheck/liveness_pinned_sources_20260930'
SOURCES = {
    'Scheduler': FROZEN / 'scheduler.py',
    'KVCacheManager': FROZEN / 'kv_cache_manager.py',
    'KVCacheCoordinatorNoPrefixCache': B / 'native_sources/kv_cache_coordinator_ack_path_20261007.py',
    'FullAttentionManager': B / 'native_sources/single_type_kv_cache_manager_ack_path_20261007.py',
    'BlockPool': B / 'native_sources/block_pool_ack_path_20261007.py',
}
OFFLOAD = B / 'native_sources/offloading_scheduler_20261007.py'
if os.environ.get('B_NATIVE_VLLM_ROOT'):
    installed = Path(os.environ['B_NATIVE_VLLM_ROOT'])
    SOURCES = {name: installed / relative for name, relative in {
        'Scheduler': 'v1/core/sched/scheduler.py',
        'KVCacheManager': 'v1/core/kv_cache_manager.py',
        'KVCacheCoordinatorNoPrefixCache': 'v1/core/kv_cache_coordinator.py',
        'FullAttentionManager': 'v1/core/single_type_kv_cache_manager.py',
        'BlockPool': 'v1/core/block_pool.py',
    }.items()}
    OFFLOAD = installed / 'distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py'
for name, path in SOURCES.items():
    assert hashlib.sha256(path.read_bytes()).hexdigest() == reserve_tail.PINS[name], name
assert hashlib.sha256(OFFLOAD.read_bytes()).hexdigest() == '89ac26a80fbc29b9bcaa5a0daba88fb6309247f9bb053e48d2d61efa7d9d66f1'


class KVCacheBlocks:
    def __init__(self, blocks):
        self.blocks = blocks


class GPULoadStoreSpec:
    def __init__(self, block_ids, **kwargs):
        self.block_ids = block_ids
        vars(self).update(kwargs)


ENV = dict(cdiv=lambda n, d: (n + d - 1) // d,
    KVCacheBlocks=KVCacheBlocks, CrossAttentionManager=type('CrossAttentionManager', (), {}),
    RequestStatus=NS(WAITING='waiting', PREEMPTED='preempted'),
    GPULoadStoreSpec=GPULoadStoreSpec, TransferJob=NS, TransferJobStatus=NS,
    OffloadPolicy=NS(BLOCK_LEVEL='block'), ScheduleEndContext=NS,
    OffloadingConnectorMetadata=NS)


def native_class(path, name, methods):
    """Execute the selected original method ASTs, without rewriting their bodies."""
    source = next(x for x in ast.parse(path.read_text()).body
                  if isinstance(x, ast.ClassDef) and x.name == name)
    selected = [x for x in source.body if isinstance(x, ast.FunctionDef) and x.name in methods]
    assert {x.name for x in selected} == set(methods)
    cls = ast.ClassDef(name=name, bases=[], keywords=[], body=selected, decorator_list=[])
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
    namespace = dict(ENV)
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace[name]


Manager = native_class(SOURCES['KVCacheManager'], 'KVCacheManager',
    ['allocate_slots', 'create_kv_cache_blocks', 'get_blocks'])
Single = native_class(SOURCES['FullAttentionManager'], 'SingleTypeKVCacheManager',
    ['get_num_blocks_to_allocate', '_get_num_evictable_blocks', '_has_partial_local_hit',
     'add_local_computed_blocks', 'allocate_external_computed_blocks', 'allocate_new_blocks',
     'get_num_skipped_tokens'])
Coordinator = native_class(SOURCES['KVCacheCoordinatorNoPrefixCache'], 'KVCacheCoordinator',
    ['get_num_blocks_to_allocate', 'allocate_new_computed_blocks', 'allocate_new_blocks', 'get_blocks'])
Scheduler = native_class(SOURCES['Scheduler'], 'Scheduler',
    ['_request_remaining_blocks', '_inflight_prefill_reserved_blocks'])
Offloader = native_class(OFFLOAD, 'OffloadingConnectorScheduler',
    ['update_state_after_alloc', 'build_connector_meta'])


class Pool:
    def __init__(self, free):
        self.free_block_queue = NS(num_free_blocks=free)
        self.allocated = []
        self.calls = []

    def get_num_free_blocks(self):
        return self.free_block_queue.num_free_blocks

    def get_new_blocks(self, count):
        assert 0 <= count <= self.get_num_free_blocks()
        self.calls.append(count)
        blocks = [NS(block_id=len(self.allocated) + i + 1, ref_cnt=1,
                     is_null=False, block_hash=None) for i in range(count)]
        self.allocated.extend(blocks)
        self.free_block_queue.num_free_blocks -= count
        return blocks


def fixture(free=300):
    pool, single, coord, manager = Pool(free), Single(), Coordinator(), Manager()
    vars(single).update(block_size=16, block_pool=pool, req_to_blocks=defaultdict(list),
        num_cached_block={}, _partial_hit_reqs={}, _max_admission_blocks_per_request=None,
        enable_caching=False, _record_new_block_ids=False, _null_block=None)
    vars(coord).update(single_type_managers=[single], cache_calls=[], skipped_calls=[])
    coord.remove_skipped_blocks = lambda *a, **k: coord.skipped_calls.append((a, k))
    coord.cache_blocks = lambda *a: coord.cache_calls.append(a)
    vars(manager).update(coordinator=coord, block_pool=pool, max_model_len=4096,
        enable_caching=False, watermark_blocks=0, empty_kv_cache_blocks=KVCacheBlocks(([],)))
    req = NS(request_id='r', num_preemptions=1, num_tokens=206 * 16, num_computed_tokens=0,
             num_prompt_tokens=3072, num_in_flight_tokens=0, has_encoder_inputs=False,
             status='preempted')
    return manager, single, req


KW = dict(num_external_computed_tokens=108 * 16, delay_cache_blocks=True,
          full_sequence_must_fit=True)


def run(mode, free=300, **kwargs):
    manager, single, req = fixture(free)
    data, uninstall = reserve_tail._install(manager, single, mode)
    result = manager.allocate_slots(req, 0, **dict(KW, **kwargs))
    return manager, single, req, data, uninstall, result


for mode, held, tail in [('native', 108, 0), ('full_tail', 206, 98)]:
    manager, single, req, data, uninstall, result = run(mode)
    assert len(manager.get_blocks('r').blocks[0]) == held
    assert len(result.blocks[0]) == tail  # Native return excludes external prefix allocations.
    assert manager.block_pool.calls == ([108, 98] if tail else [108])
    assert all(b.ref_cnt == 1 for b in manager.get_blocks('r').blocks[0])
    assert not manager.coordinator.cache_calls
    assert req.num_computed_tokens == 0 and req.num_tokens == 3296
    assert data['events'][0]['original_calls'] == 1
    assert data['events'][0]['after'] == dict(held_blocks=held, free_blocks=300 - held)
    # Original scheduler registers the async request; its remaining reservation is not duplicated.
    req.num_computed_tokens = 108 * 16
    sched = Scheduler()
    vars(sched).update(kv_cache_manager=manager, max_model_len=4096, _inflight_prefills=[req])
    assert sched._request_remaining_blocks(req) == 206 - held
    assert sched._inflight_prefill_reserved_blocks() == 206 - held
    uninstall()
    assert 'allocate_slots' not in vars(manager)
    uninstall()  # Idempotent.

# Both full-fit and the native second reservation check remain active.
for mode in ('native', 'full_tail'):
    manager, single, req, data, uninstall, result = run(mode, free=205)
    assert result is None and not single.req_to_blocks and not manager.block_pool.calls
    assert not manager.coordinator.skipped_calls
    assert data['events'][0]['outcome'] == 'allocation_failed'
    uninstall()
assert run('native', free=250, reserved_blocks=60)[-1] is not None
manager, single, req, data, uninstall, result = run('full_tail', free=250, reserved_blocks=60)
assert result is None and not manager.block_pool.calls
assert data['events'][0]['effective']['reserved_blocks'] == 60
uninstall()

# Unsupported/ordinary calls, no tail, exact-once return identity, and exception restoration.
for label, request_changes, args in [
    ('ordinary', {}, dict(num_new_tokens=16, num_external_computed_tokens=0)),
    ('new_request', dict(num_preemptions=0), dict(num_new_tokens=0)),
    ('zero_external', {}, dict(num_new_tokens=0, num_external_computed_tokens=0)),
    ('no_tail', dict(num_tokens=108 * 16), dict(num_new_tokens=0)),
    ('no_delay', {}, dict(num_new_tokens=0, delay_cache_blocks=False)),
]:
    manager, single, req = fixture()
    vars(req).update(request_changes)
    calls, sentinel = [], KVCacheBlocks(([],))
    def spy(*a, **k):
        calls.append((a, k))
        return sentinel
    manager.allocate_slots = spy
    data, uninstall = reserve_tail._install(manager, single, 'full_tail')
    kwargs = dict(KW, **args)
    requested = kwargs.pop('num_new_tokens')
    assert manager.allocate_slots(req, requested, **kwargs) is sentinel
    assert len(calls) == 1 and calls[0][0][1] == requested, label
    assert all(not event['changed'] for event in data['events'])
    uninstall()
    assert manager.allocate_slots is spy

manager, single, req = fixture()
calls, error = [], RuntimeError('native error')
def fail(*a, **k):
    calls.append((a, k))
    raise error
manager.allocate_slots = fail
data, uninstall = reserve_tail._install(manager, single, 'full_tail')
try:
    manager.allocate_slots(req, 0, **KW)
    raise AssertionError('exception swallowed')
except RuntimeError as caught:
    assert caught is error
assert len(calls) == 1 and data['events'][0]['outcome'] == 'exception'
uninstall()
assert manager.allocate_slots is fail

# Actual native connector AST sees all allocations; LOAD slices only the matched prefix.
manager, single, req, data, uninstall, result = run('full_tail')
offloader = Offloader()
load_keys = []
vars(offloader).update(config=NS(kv_group_configs=[NS(tokens_per_block=16, tokens_per_chunk=16,
    sliding_window_size_in_chunks=None)], blocks_per_chunk=1, num_workers=1),
    _req_status={'r': NS(num_locally_computed_tokens=0, req_context=NS(),
        group_states=[NS(offload_keys=list(range(206)), next_stored_chunk_idx=0)],
        offloading_context=NS(policy='block'), transfer_jobs=set())},
    _current_batch_allocated_block_ids=set(), _current_batch_load_jobs={},
    _current_batch_jobs_to_flush=set(), _chunks_being_loaded=None, _jobs={},
    _block_id_to_pending_jobs={200: {999}},
    manager=NS(prepare_load=lambda keys, context: load_keys.extend(keys), on_schedule_end=lambda context: None))
offloader._generate_job_id = lambda: 1
offloader._update_req_states = lambda output: None
offloader._build_store_jobs = lambda output: {}
offloader.update_state_after_alloc(req, manager.get_blocks('r'), 108 * 16)
assert offloader._current_batch_allocated_block_ids == set(range(1, 207))
assert load_keys == list(range(108))
assert offloader._current_batch_load_jobs[1].dst_spec.block_ids == list(range(1, 109))
meta = offloader.build_connector_meta(NS(scheduled_new_reqs=[], preempted_req_ids=[]))
assert meta.jobs_to_flush == {999}  # Reused tail address remains protected by native STORE fence.
uninstall()
print('PASS: pinned native allocator/coordinator/single-manager/scheduler/connector AST; CPU only, GPU_UNRUN')

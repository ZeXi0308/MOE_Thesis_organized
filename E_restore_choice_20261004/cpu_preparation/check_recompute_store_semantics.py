"""Bounded CPU function-level check; no vLLM engine, tensors, or GPU.

Run: python3 -B cpu_preparation/check_recompute_store_semantics.py

Compile selected original AST definitions from the saved native source, without
rewriting their bodies. Fixtures supply request/cache objects, block allocation,
and synchronous transfer completion. This checks STORE cursor semantics for one
full-attention/block-level case, not runtime performance or the observed cause
of a particular run's cache holes.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
import importlib.util
from itertools import chain, islice
import json
import logging
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace as NS


ROOT = Path(__file__).resolve().parents[1]
SCHED = ROOT / 'native_sources/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py'
MANAGER = ROOT / 'native_sources/v1/kv_offload/cpu/manager.py'


class LookupResult(Enum):
    HIT = 1
    MISS = 2
    HIT_PENDING = 3
    RETRY = 4


@dataclass
class Block:
    block_id: int
    ref_cnt: int = -1

    @property
    def is_ready(self):
        return self.ref_cnt >= 0


class CacheFixture:
    """Resident keys are explicit; no eviction/threshold policy is simulated."""
    def __init__(self):
        self.blocks = {i: Block(i, 0) for i in range(2)}
        self.touches = []

    def get(self, key):
        return self.blocks.get(key)

    def touch(self, keys, context):
        self.touches.append(list(keys))

    def insert(self, key, block):
        self.blocks[key] = block

    def mark_non_evictable(self, key):
        assert key in self.blocks

    def mark_evictable(self, key):
        assert key in self.blocks

    def evict(self, *args):
        raise AssertionError('Fixture must not exercise eviction')


def spec(block_ids, **kwargs):
    return NS(block_ids=list(block_ids), **kwargs)


def load_native_definitions():
    module = ModuleType('_e_native_store_function_check')
    sys.modules[module.__name__] = module
    ns = module.__dict__
    ns.update(dataclass=dataclass, field=field, chain=chain, islice=islice,
              time=time, logger=logging.getLogger(__name__),
              cdiv=lambda n, d: (n+d-1)//d, override=lambda fn: fn,
              LookupResult=LookupResult, OffloadPolicy=NS(BLOCK_LEVEL='block'),
              make_offload_key=lambda h, group: h, BlockStatus=Block,
              CPULoadStoreSpec=spec, GPULoadStoreSpec=spec,
              PrepareStoreOutput=NS, TransferJob=NS,
              _ConnectorMetricName=NS(LOOKUP_SYNC_DELAY='sync',
                                     LOOKUP_ASYNC_DELAY='async',
                                     ALLOCATION_FAILURE='allocation'))
    selections = {
        SCHED: {
            'TransferJobStatus': None, 'RequestGroupState': None,
            'RequestOffloadState': None,
            'OffloadingConnectorScheduler': {
                '_generate_job_id', '_calc_num_offloadable_tokens',
                '_maybe_cleanup_finished_req', '_maybe_observe_lookup_async_delay',
                '_maximal_prefix_lookup', '_touch', '_lookup',
                'get_num_new_matched_tokens', 'update_state_after_alloc',
                '_build_store_jobs',
            },
        },
        MANAGER: {'CPUOffloadingManager': {
            '_get_num_free_blocks', '_allocate_blocks', '_free_block',
            '_get_load_store_spec', 'lookup', 'prepare_load', 'touch',
            'complete_load', 'prepare_store', 'complete_store',
        }},
    }
    references = {}
    for path, classes in selections.items():
        tree = ast.parse(path.read_text())
        selected = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in classes:
                methods = classes[node.name]
                references[node.name] = {'source': str(path.relative_to(ROOT)),
                                         'line': node.lineno}
                if methods is not None:
                    node.bases = []
                    node.body = [n for n in node.body
                                 if isinstance(n, ast.FunctionDef) and n.name in methods]
                    assert {n.name for n in node.body} == methods
                    references[node.name]['methods'] = {n.name: n.lineno for n in node.body}
                selected.append(node)
        assert {n.name for n in selected} == set(classes)
        compiled = ast.Module(body=[ast.ImportFrom(module='__future__',
                               names=[ast.alias(name='annotations')], level=0),
                               *selected], type_ignores=[])
        exec(compile(ast.fix_missing_locations(compiled), str(path), 'exec'), ns)
    return ns, references


def run_branch(ns, mode, selector_type):
    cfg_group = NS(tokens_per_block=16, tokens_per_chunk=16, hashes_per_chunk=1,
                   group_idx=0, is_eagle_group=False, sliding_window_size_in_chunks=None,
                   alignment_chunk_count=None)
    cfg = NS(kv_group_configs=[cfg_group], blocks_per_chunk=1,
             num_workers=1, offload_prompt_only=False)
    req = NS(request_id='fixture', num_tokens=128, num_prompt_tokens=112,
             num_output_tokens=16, num_computed_tokens=0, num_preemptions=1,
             all_token_ids=list(range(128)), block_hashes=list(range(8)),
             skip_reading_prefix_cache=(mode == 'native_skip'),
             kv_transfer_params=None, sampling_params=object(), status='PREEMPTED',
             stop_reason=None, is_finished=lambda: False)
    state = ns['RequestOffloadState'](cfg, req, NS(req_id='fixture'), NS(policy='block'))
    gs = state.group_states[0]
    gs.offload_keys = list(range(8))
    # Seven chunks were stored previously; chunks 2..6 are now absent. The
    # retained STORE high-water mark therefore exceeds the ready prefix.
    gs.next_stored_chunk_idx = 7
    gs.block_ids = list(range(1, 9))
    manager = ns['CPUOffloadingManager']()
    manager.__dict__.update(_policy=CacheFixture(), counts=None, events=None,
        _num_blocks=16, _num_allocated_blocks=2, _free_list=[],
        _num_evictable_cache_blocks=2, _num_write_pending_blocks=0,
        allocation_sizes_in_current_batch=[], stores_skipped_in_current_batch=0)
    scheduler = ns['OffloadingConnectorScheduler']()
    scheduler.__dict__.update(config=cfg, manager=manager, _req_status={'fixture': state},
        _sliding_window_groups=(), _lookup_groups=(0,), _chunks_being_loaded=set(),
        _jobs={}, _job_counter=0, _current_batch_load_jobs={},
        _current_batch_allocated_block_ids=set(), _block_id_to_pending_jobs={},
        _current_batch_jobs_to_flush=set(),
        _connector_stats=NS(observe_histogram=lambda *args: None,
                            increase_counter=lambda *args: None),
        _events_tracker=NS(record_store=lambda *args: None))
    if mode == 'forced_recompute':
        # Invoke the real adapter lookup. Its unused headroom observation is
        # supplied as a fixture; fixed_recompute never branches on that value.
        selector = selector_type.__new__(selector_type)
        selector.__dict__.update(enabled=True, cs=scheduler, policy='recompute',
            target_spec=None, last_step_s=0.0, events=[], pending={},
            now=time.monotonic, _lookup=scheduler.get_num_new_matched_tokens,
            headroom_feature=lambda *args: {'supported': True},
            scheduler=NS(kv_cache_manager=NS(block_pool=NS(get_num_free_blocks=lambda: 32),
                                             watermark_blocks=0),
                         _inflight_prefill_reserved_blocks=lambda: 0,
                         running=[], waiting=[req], current_step=0))
        lookup = selector.lookup(req, 0)
        assert selector.events[0]['action'] == 'recompute'
    else:
        lookup = scheduler.get_num_new_matched_tokens(req, 0)
    after_lookup = dict(num_hit_chunks=gs.num_hit_chunks,
                        next_stored_chunk_idx=gs.next_stored_chunk_idx,
                        block_ids=list(gs.block_ids),
                        touch_calls=[list(keys) for keys in manager._policy.touches])
    external = lookup[0]
    gpu_blocks = NS(blocks=([NS(block_id=i+1, is_null=False, block_hash=None)
                            for i in range((external+15)//16)],))
    scheduler.update_state_after_alloc(req, gpu_blocks, external)
    after_alloc_cursor = gs.next_stored_chunk_idx
    # Fixture acknowledges the native LOAD immediately; no real transfer or
    # worker completion path is exercised.
    for jid in list(state.transfer_jobs):
        job = scheduler._jobs.pop(jid)
        assert not job.is_store
        manager.complete_load(job.keys, state.req_context)
        state.transfer_jobs.remove(jid)
    scheduler._chunks_being_loaded.clear()
    previous = external
    stores = []
    for end in (64, 128):
        # Supply the GPU blocks and compute progress a legal native scheduler
        # would deliver. The actual store-builder body decides the STORE keys.
        needed = (end+15)//16
        state.update_block_id_groups((list(range(len(gs.block_ids)+1, needed+1)),))
        req.num_computed_tokens = previous
        output = NS(num_scheduled_tokens={'fixture': end-previous}, finished_req_ids=set())
        jobs = scheduler._build_store_jobs(output)
        keys = sorted(k for jid in jobs for k in scheduler._jobs[jid].keys)
        stores.append(dict(computed_end=end, submitted_keys=keys,
                           next_stored_chunk_idx=gs.next_stored_chunk_idx,
                           job_specs=[dict(request=job.req_id,
                                           src_spec=vars(job.src_spec),
                                           dst_spec=vars(job.dst_spec),
                                           pending_count=scheduler._jobs[jid].pending_count,
                                           is_store=scheduler._jobs[jid].is_store)
                                      for jid, job in jobs.items()]))
        for jid in jobs:
            job = scheduler._jobs.pop(jid)
            manager.complete_store(job.keys, state.req_context)
            state.transfer_jobs.remove(jid)
        previous = end
    return dict(lookup=list(lookup), after_lookup=after_lookup,
                after_alloc_cursor=after_alloc_cursor, stores=stores,
                final_ready_keys=sorted(k for k, v in manager._policy.blocks.items() if v.is_ready))


def main():
    ns, refs = load_native_definitions()
    spec_ = importlib.util.spec_from_file_location('_e_selector_for_store_check', ROOT/'selector.py')
    module = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(module)
    result = {m: run_branch(ns, m, module.RestoreSelector)
              for m in ('native_skip', 'forced_recompute', 'host')}
    assert result['native_skip'] == result['forced_recompute']
    assert result['native_skip']['after_alloc_cursor'] == 7
    assert result['native_skip']['stores'][0]['submitted_keys'] == []
    assert result['native_skip']['stores'][1]['submitted_keys'] == [7]
    assert result['native_skip']['final_ready_keys'] == [0, 1, 7]
    assert result['host']['after_alloc_cursor'] == 2
    assert result['host']['stores'][0]['submitted_keys'] == [2, 3]
    assert result['host']['stores'][1]['submitted_keys'] == [4, 5, 6, 7]
    assert result['host']['final_ready_keys'] == list(range(8))
    print(json.dumps(dict(status='PASS_FUNCTION_LEVEL', branches=result,
        interpretation='Forced recompute and native skip-read share STORE behavior in this fixture. Host rewinds the native STORE cursor and fills old holes; both recompute paths retain the old cursor but store newly reached chunks.',
        limitations='Original function bodies with explicit CPU fixtures; not a full-engine, CUDA, cache-eviction-policy, timing, or observed-run causality test. STORE/load completion is acknowledged synchronously. One full-attention group, block policy, APC disabled, store threshold disabled.',
        sources=refs, selector_lookup_line=module.RestoreSelector.lookup.__code__.co_firstlineno),
        indent=2))


if __name__ == '__main__':
    main()

"""One focused CPU probe: original native functions + explicit object fixtures.

No vLLM engine/worker/GPU. Reuses the existing AST extraction of native lookup,
allocation, store builder and CPU manager; the real selector commits the target.
"""
from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_recompute_store_semantics import load_native_definitions, CacheFixture, Block
from selector import RestoreSelector, token_hash
from native_store_replay import NativeStoreReplay
from host_history_observer import HostHistoryObserver


def fixture(ns):
    group_config = NS(tokens_per_block=16, tokens_per_chunk=16, hashes_per_chunk=1,
                      group_idx=0, is_eagle_group=False, sliding_window_size_in_chunks=None,
                      alignment_chunk_count=None)
    config = NS(kv_group_configs=[group_config], blocks_per_chunk=1,
                num_workers=1, offload_prompt_only=False)
    req = NS(request_id='fixture', num_tokens=128, num_prompt_tokens=112,
             num_output_tokens=16, num_computed_tokens=0, num_preemptions=1,
             all_token_ids=list(range(128)), block_hashes=list(range(8)),
             skip_reading_prefix_cache=False, kv_transfer_params=None,
             sampling_params=object(), status='PREEMPTED', stop_reason=None,
             is_finished=lambda: False)
    state = ns['RequestOffloadState'](config, req, NS(req_id='fixture'), NS(policy='block'))
    group = state.group_states[0]
    group.offload_keys, group.next_stored_chunk_idx = list(range(8)), 7
    cache = CacheFixture()
    # Retain non-contiguous suffix chunks. Replaying [2,8) must not copy all
    # six candidates: original prepare_store should select only holes 2/5/7.
    cache.blocks = {key: Block(i, 0) for i, key in enumerate([0, 1, 3, 4, 6])}
    manager = ns['CPUOffloadingManager']()
    manager.__dict__.update(_policy=cache, counts=None, events=None,
        _num_blocks=16, _num_allocated_blocks=5, _free_list=[],
        _num_evictable_cache_blocks=5, _num_write_pending_blocks=0,
        allocation_sizes_in_current_batch=[], stores_skipped_in_current_batch=0)
    cs = ns['OffloadingConnectorScheduler']()
    cs.__dict__.update(config=config, manager=manager, _req_status={'fixture': state},
        _sliding_window_groups=(), _lookup_groups=(0,), _chunks_being_loaded=set(),
        _jobs={}, _job_counter=0, _current_batch_load_jobs={},
        _current_batch_allocated_block_ids=set(), _block_id_to_pending_jobs={},
        _current_batch_jobs_to_flush=set(), update_connector_output=lambda output: None,
        _connector_stats=NS(observe_histogram=lambda *args: None, increase_counter=lambda *args: None),
        _events_tracker=NS(record_store=lambda *args: None))
    selector = RestoreSelector.__new__(RestoreSelector)
    target = dict(external_id='measured/E246', num_preemptions=1, known_tokens=128,
                  generated_tokens=16, prefix_sha256=token_hash(req.all_token_ids))
    selector.__dict__.update(cs=cs, enabled=True, policy='recompute', threshold=0,
        target_spec={'schema': 'E.one_event_target.v1', 'target': target},
        target_commit_count=0, target_effective_action=None,
        target_identity_lookups=0, target_state_lookups=0, target_eligible_lookups=0,
        target_mismatch_counts=Counter(), external_ids={'fixture': 'measured/E246'},
        events=[], commits=[], pending={}, last_step_s=0, now=lambda: 0.01,
        _lookup=cs.get_num_new_matched_tokens, _alloc=cs.update_state_after_alloc,
        headroom_feature=lambda *args: {'supported': True},
        scheduler=NS(kv_cache_manager=NS(block_pool=NS(get_num_free_blocks=lambda: 32), watermark_blocks=0),
                     _inflight_prefill_reserved_blocks=lambda: 0,
                     running=[], waiting=[req], current_step=416))
    assert selector.lookup(req, 0) == (0, False)
    assert selector.pending['fixture']['target_selected']
    assert selector.pending['fixture']['host_hit_tokens'] == 32
    cs.update_state_after_alloc = selector.allocated
    blocks = NS(blocks=([NS(block_id=i+1, is_null=False, block_hash=None) for i in range(4)],))
    return selector, cs, state, req, blocks


def rebuild(cs, state):
    previous, stored = 0, []
    group, req = state.group_states[0], state.req
    for end in (64, 128):
        state.update_block_id_groups((list(range(len(group.block_ids)+1, end//16+1)),))
        req.num_computed_tokens = previous
        jobs = cs._build_store_jobs(NS(num_scheduled_tokens={'fixture': end-previous}, finished_req_ids=set()))
        keys = sorted(key for jid in jobs for key in cs._jobs[jid].keys)
        stored.append(keys)
        for jid in jobs:
            job = cs._jobs.pop(jid)
            cs.manager.complete_store(job.keys, state.req_context)
            state.transfer_jobs.remove(jid)
        previous = end
    return stored


def main():
    ns, _ = load_native_definitions()
    results = {}
    for enabled in (False, True):
        selector, cs, state, req, blocks = fixture(ns)
        original = cs.update_state_after_alloc
        touches = len(cs.manager._policy.touches)
        pending_keys = set(cs.manager._policy.blocks)
        with NativeStoreReplay(selector, enabled=enabled) as replay:
            with HostHistoryObserver(cs, selector.now, selector, enabled=True) as history:
                replay.service_step(416); history.service_step(416)
                assert cs.update_state_after_alloc(req, blocks, 0) is None
                assert len(cs.manager._policy.touches) == touches
                assert set(cs.manager._policy.blocks) == pending_keys
                assert selector.target_commit_count == 1
                assert replay.executed_count == int(enabled)
                allocation = history.records[0]
                assert allocation['before']['groups'][0]['next_stored_chunk_idx'] == 7
                assert allocation['after']['groups'][0]['next_stored_chunk_idx'] == (2 if enabled else 7)
                # A repeated native callback cannot apply the probe twice.
                assert cs.update_state_after_alloc(req, blocks, 0) is None
                assert replay.executed_count == int(enabled)
                if enabled:
                    assert replay.records[0]['status'] == 'executed'
                    assert replay.records[0]['request_state_preserved'] is True
                    assert replay.records[0]['known_prefix_matches_record'] is True
                    assert replay.records[1]['reason'] == 'already_executed'
                else:
                    assert not replay.records
                results[enabled] = rebuild(cs, state)
        assert cs.update_state_after_alloc == original
        assert not replay.errors and not history.errors
    assert results == {False: [[], [7]], True: [[2], [5, 7]]}

    for case, reason in [('non_target', 'not_target'),
                         ('fallback', 'fallback_or_not_jointly_legal'),
                         ('unsupported', 'unsupported_offload_policy'),
                         ('host', 'target_action_not_recompute'),
                         ('no_commit', 'target_commit_not_observed')]:
        selector, cs, state, req, blocks = fixture(ns)
        event = selector.pending['fixture']
        external = 0
        sentinel = object()
        if case == 'non_target': event['target_selected'] = False
        if case == 'fallback': event.update(target_selected=False, eligible=False,
                                             joint_capacity=False, action='native', fallback='native')
        if case == 'unsupported': state.offloading_context.policy = 'request_level'
        if case == 'host': event['action'], external = 'host', 32
        if case == 'no_commit': cs.update_state_after_alloc = lambda *args, **kwargs: sentinel
        with NativeStoreReplay(selector, enabled=True) as replay:
            value = cs.update_state_after_alloc(req, blocks, external)
        assert value is (sentinel if case == 'no_commit' else None)
        assert replay.executed_count == 0 and not replay.errors
        assert replay.records[0]['reason'] == reason
        assert state.group_states[0].next_stored_chunk_idx == (2 if case == 'host' else 7)

    selector, cs, state, req, blocks = fixture(ns)
    marker = RuntimeError('native allocation sentinel')
    def fail(*args, **kwargs): raise marker
    selector._alloc = fail
    original = cs.update_state_after_alloc
    try:
        with NativeStoreReplay(selector, enabled=True) as replay:
            cs.update_state_after_alloc(req, blocks, 0)
    except RuntimeError as exc:
        assert exc is marker
    else:
        raise AssertionError('Native exception was swallowed')
    assert cs.update_state_after_alloc == original
    assert replay.executed_count == 0 and not selector.commits and not replay.errors
    assert replay.records[0]['status'] == 'native_raised'
    assert state.group_states[0].next_stored_chunk_idx == 7
    print('PASS_FUNCTION_LEVEL: original native builder/manager store only holes; '
          'single committed replay; outer history sees rewind; disabled/non-target/'
          'fallback/Host/unsupported/uncommitted unchanged; native return/exception preserved')
    print('Actual fixture STORE keys: off=[[],[7]], on=[[2],[5,7]]; '
          'not an engine/GPU test or observed-run causality proof.')


if __name__ == '__main__':
    main()

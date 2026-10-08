"""Fixed restoration headroom baseline v1; CPU prepared, GPU-UNRUN.
Outer observe_native_pressure, inner restore_fixed_margin, then common.generate.
Only PREEMPTED admission changes. Native allocation/victim selection is intact.
This is a known-style admission baseline, NOT a protected execution quantum:
no persistent reservation, no promise against later preemption or new arrivals.
Candidate full-known-sequence+1 matches runway; margin is fixed at 32/48/64 blocks.
No running-request progress tracking or per-resident allocation queries are made.
No predicted length, future EOS, or remaining-output estimate is read.
"""
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import time
from native_pressure_observer_v1 import ALLOCATION_PARAMETERS


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def name(status):
    return getattr(status, 'name', str(status))


def fixed_gate(m, values, margin_blocks):
    """Same candidate query as runway; one native query, no resident traversal."""
    r = values['request']
    computed = r.num_computed_tokens + values['num_new_computed_tokens']
    blocks = values['new_computed_blocks']
    blocks = m.empty_kv_cache_blocks.blocks if blocks is None else blocks.blocks
    target = min(max(r.num_tokens, computed + values['num_new_tokens']) + 1,
                 m.max_model_len)
    candidate_need = m.coordinator.get_num_blocks_to_allocate(
        request_id=r.request_id, num_tokens=target, new_computed_blocks=blocks,
        num_encoder_tokens=0, total_computed_tokens=computed,
        num_local_computed_tokens=computed, num_tokens_main_model=target,
        apply_admission_cap=False)
    require(type(candidate_need) is int and candidate_need >= 0, 'invalid candidate-block query')
    watermark = m.watermark_blocks if values['has_scheduled_reqs'] else 0
    free = m.block_pool.get_num_free_blocks()
    required = margin_blocks + candidate_need + watermark + values['reserved_blocks']
    return dict(free_blocks=free, fixed_margin_blocks=margin_blocks,
        candidate_known_tokens=r.num_tokens, candidate_target_tokens=target,
        candidate_full_plus_one_blocks=candidate_need,
        candidate_cached_tokens=values['num_new_computed_tokens'], watermark=watermark,
        original_reserved_blocks=values['reserved_blocks'], required_blocks=required,
        policy_deferred=required > free)


@contextmanager
def restore_fixed_margin(engine, output_path, margin_blocks):
    require(type(margin_blocks) is int and margin_blocks in (32, 48, 64),
            'fixed margin must be exactly 32, 48, or 64 physical blocks')
    s = engine.engine_core.engine_core.scheduler
    m = s.kv_cache_manager
    managers = m.coordinator.single_type_managers
    require(len(managers) == 1 and type(managers[0]).__name__ == 'FullAttentionManager'
        and type(managers[0].kv_cache_spec).__name__ == 'FullAttentionSpec',
        'prototype requires one native FullAttention group')
    require(not s.scheduler_config.async_scheduling and not s.use_eagle
        and s.num_lookahead_tokens == 0 and s.num_sampled_tokens_per_step == 1
        and s.connector is None and not s.is_encoder_decoder,
        'prototype requires synchronous decoder-only non-speculative local serving')
    require(s.scheduler_reserve_full_isl and not s.running and not s.waiting
        and not s.skipped_waiting, 'install after cold drain with full-ISL enabled')
    original = m.allocate_slots
    # Observer's bound wrapper has *args/**kwargs: inspect the native class API.
    native_signature = inspect.signature(type(m).allocate_slots)
    require(tuple(native_signature.parameters)[1:] == ALLOCATION_PARAMETERS,
            'native allocation signature differs')
    signature = native_signature.replace(parameters=list(native_signature.parameters.values())[1:])
    decisions = []
    calls, extra_wall = 0, 0.0
    origin = time.perf_counter()
    report = dict(schema='c-restore-fixed-margin-policy-v1', status='INCOMPLETE',
        scope='Fixed-margin restoration admission baseline; no dynamic resident signal, persistent reservation, or protected-run guarantee.',
        fixed_margin_blocks=margin_blocks,
        native_signature=str(signature), decisions=decisions,
        cost_scope='Policy hook wall excludes native/observer call; parent phase includes receipt serialization.')
    def allocate(*args, **kwargs):
        nonlocal calls, extra_wall
        started, native_wall, event = time.perf_counter(), 0.0, None
        calls += 1
        try:
            r = args[0] if args else kwargs['request']
            if name(r.status) == 'PREEMPTED':
                bound = signature.bind(*args, **kwargs); bound.apply_defaults()
                v = bound.arguments
                require(v['full_sequence_must_fit'] and not any(v[k] for k in
                    ('num_lookahead_tokens', 'num_external_computed_tokens',
                     'delay_cache_blocks', 'num_encoder_tokens')),
                    'unsupported restoration allocation arguments')
                event = dict(decision=len(decisions), scheduler_step=s.current_step,
                    host_s=time.perf_counter() - origin, request_id=r.request_id,
                    **fixed_gate(m, v, margin_blocks))
                decisions.append(event)
                if event['policy_deferred']:
                    event['native_called'] = False
                    return None  # Outside observer: do not label this native KV None.
                event['native_called'] = True
            # RUNNING/WAITING: direct forwarding, no binding or progress tracking.
            begin = time.perf_counter()
            try:
                result = original(*args, **kwargs)  # Arguments remain byte-for-byte unchanged.
            finally:
                native_wall = time.perf_counter() - begin
            if event is not None:
                event['native_returned_none'] = result is None
            return result
        except BaseException as exc:
            if event is not None:
                event['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            extra_wall += time.perf_counter() - started - native_wall
    with Path(output_path).open('x', encoding='utf-8') as stream:
        try:
            m.allocate_slots = allocate
            yield report
            report['status'] = 'COMPLETE'
        except BaseException as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            m.allocate_slots = original
            report.update(allocation_hook_calls=calls, decision_count=len(decisions),
                policy_deferred_count=sum(d['policy_deferred'] for d in decisions),
                policy_hook_wall_s=extra_wall, hooks_restored=True,
                observation_end_s=time.perf_counter() - origin)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')


def self_test():
    """Boundary/APC, unchanged forwarding, and restoration checks; no GPU."""
    import tempfile
    from types import SimpleNamespace as NS
    class FullAttentionSpec: pass
    class FullAttentionManager:
        def __init__(self): self.kv_cache_spec = FullAttentionSpec()
    class Coordinator:
        def __init__(self):
            self.single_type_managers, self.queries = [FullAttentionManager()], []
        def get_num_blocks_to_allocate(self, **v):
            self.queries.append(v)
            hits = v['new_computed_blocks'][0]
            return max((v['num_tokens']+15)//16-len(hits), 0)+sum(b.ref_cnt == 0 for b in hits)
    class Manager:
        def allocate_slots(self, request, num_new_tokens, num_new_computed_tokens=0,
                new_computed_blocks=None, num_lookahead_tokens=0,
                num_external_computed_tokens=0, delay_cache_blocks=False,
                num_encoder_tokens=0, full_sequence_must_fit=False,
                reserved_blocks=0, has_scheduled_reqs=True):
            return self.token
    m = Manager(); m.coordinator = Coordinator(); m.max_model_len = 4096
    m.empty_kv_cache_blocks = NS(blocks=((),)); m.watermark_blocks = 0
    m.block_pool = NS(free=33); m.block_pool.get_num_free_blocks = lambda: m.block_pool.free
    m.token = object()
    r = NS(request_id='restored', status='PREEMPTED', num_tokens=16, num_computed_tokens=0)
    v = dict(request=r, num_new_tokens=16, num_new_computed_tokens=0,
             new_computed_blocks=None, has_scheduled_reqs=True, reserved_blocks=0)
    assert fixed_gate(m, v, 32)['policy_deferred']  # Needs 32 + ceil(17/16) = 34.
    m.block_pool.free = 34
    assert not fixed_gate(m, v, 32)['policy_deferred']  # Exact-fit boundary is allowed.
    r.num_tokens = 48; v['num_new_tokens'] = 1; v['num_new_computed_tokens'] = 16
    hit = NS(ref_cnt=0); v['new_computed_blocks'] = NS(blocks=([hit],))
    assert fixed_gate(m, v, 32)['candidate_full_plus_one_blocks'] == 4
    hit.ref_cnt = 1
    assert fixed_gate(m, v, 32)['candidate_full_plus_one_blocks'] == 3
    assert m.block_pool.free == 34  # Queries do not allocate/free/retain anything.
    s = NS(kv_cache_manager=m, scheduler_config=NS(async_scheduling=False),
        use_eagle=False, num_lookahead_tokens=0, num_sampled_tokens_per_step=1,
        connector=None, is_encoder_decoder=False, scheduler_reserve_full_isl=True,
        running=[], waiting=[], skipped_waiting=[], current_step=7)
    engine = NS(engine_core=NS(engine_core=NS(scheduler=s)))
    forwarded = []
    native = m.allocate_slots
    def observer(*args, **kwargs):
        forwarded.append((args, kwargs.copy())); return native(*args, **kwargs)
    m.allocate_slots = observer
    original_bind = inspect.Signature.bind
    with tempfile.TemporaryDirectory(prefix='fixed-margin-check-') as tmp:
        with restore_fixed_margin(engine, Path(tmp)/'receipt.json', 32) as report:
            def forbid_bind(*a, **k): raise AssertionError('hotpath signature binding')
            inspect.Signature.bind = forbid_bind
            try:
                running = NS(status='RUNNING')  # Deliberately has no progress attributes.
                waiting = NS(status='WAITING')
                assert m.allocate_slots(running, 1) is m.token
                assert m.allocate_slots(request=waiting, num_new_tokens=8) is m.token
            finally:
                inspect.Signature.bind = original_bind
            queries_before = len(m.coordinator.queries); m.block_pool.free = 100
            kwargs = dict(num_new_computed_tokens=16, new_computed_blocks=v['new_computed_blocks'],
                          full_sequence_must_fit=True, reserved_blocks=2)
            assert m.allocate_slots(r, 1, **kwargs) is m.token
            assert forwarded[-1][0] == (r, 1) and forwarded[-1][1] == kwargs
            assert forwarded[-1][1]['new_computed_blocks'] is kwargs['new_computed_blocks']
            assert len(m.coordinator.queries) == queries_before+1
            m.block_pool.free = 0; before = len(forwarded)
            assert m.allocate_slots(r, 1, **kwargs) is None and len(forwarded) == before
            assert report['decisions'][-1]['native_called'] is False
        assert m.allocate_slots is observer and report['hooks_restored']
    return dict(status='PASS', checks=2, gpu_used=False)


if __name__ == '__main__':
    print(json.dumps(self_test()))

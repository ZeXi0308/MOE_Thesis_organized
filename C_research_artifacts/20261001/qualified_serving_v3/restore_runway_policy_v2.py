"""One-step physical-headroom restoration gate v2; CPU prototype, GPU-UNRUN.
Outer observe_native_pressure, inner restore_runway, then common.generate.
Only PREEMPTED admission changes. Native allocation/victim selection is intact.
This is a known-style admission baseline, NOT a protected execution quantum:
no persistent reservation, no promise against later preemption or new arrivals.
For partial prefills, reserve the entire already-known sequence plus one token.
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


def runway(s, m, values, progress):
    """Read-only native physical-allocation queries for the pinned full-attention case."""
    co, r = m.coordinator, values['request']
    empty = m.empty_kv_cache_blocks.blocks
    def need(request, target, blocks, computed):
        return co.get_num_blocks_to_allocate(request_id=request.request_id,
            num_tokens=target, new_computed_blocks=blocks, num_encoder_tokens=0,
            total_computed_tokens=computed, num_local_computed_tokens=computed,
            num_tokens_main_model=target, apply_admission_cap=False)
    running_need, partial, contributing = 0, 0, 0
    for resident in s.running:
        require(name(resident.status) == 'RUNNING', 'non-running resident')
        post = progress.get(resident.request_id, (None, resident.num_computed_tokens))
        computed = post[1] if post[0] == s.current_step else resident.num_computed_tokens
        partial += computed < resident.num_tokens
        target = min(max(computed, resident.num_tokens) + 1, m.max_model_len)
        n = need(resident, target, empty, computed)
        require(type(n) is int and n >= 0, 'invalid native running-block query')
        running_need += n
        contributing += n > 0
    computed = r.num_computed_tokens + values['num_new_computed_tokens']
    blocks = values['new_computed_blocks']
    blocks = empty if blocks is None else blocks.blocks
    target = min(max(r.num_tokens, computed + values['num_new_tokens']) + 1,
                 m.max_model_len)
    candidate_need = need(r, target, blocks, computed)
    require(type(candidate_need) is int and candidate_need >= 0, 'invalid candidate-block query')
    watermark = m.watermark_blocks if values['has_scheduled_reqs'] else 0
    free = m.block_pool.get_num_free_blocks()
    required = running_need + candidate_need + watermark + values['reserved_blocks']
    return dict(free_blocks=free, running_next_blocks=running_need,
        running_count=len(s.running), running_contributors=contributing,
        partial_running_count=partial, candidate_known_tokens=r.num_tokens,
        candidate_target_tokens=target, candidate_full_plus_one_blocks=candidate_need,
        candidate_cached_tokens=values['num_new_computed_tokens'], watermark=watermark,
        original_reserved_blocks=values['reserved_blocks'], required_blocks=required,
        policy_deferred=required > free)


@contextmanager
def restore_runway(engine, output_path):
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
    progress, decisions = {}, []
    calls, extra_wall = 0, 0.0
    origin = time.perf_counter()
    report = dict(schema='c-restore-runway-policy-v2', status='INCOMPLETE',
        scope='One-step physical-headroom admission baseline; no persistent reservation or protected-run guarantee.',
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
                new_tokens, cached_tokens = v['num_new_tokens'], v['num_new_computed_tokens']
                require(v['full_sequence_must_fit'] and not any(v[k] for k in
                    ('num_lookahead_tokens', 'num_external_computed_tokens',
                     'delay_cache_blocks', 'num_encoder_tokens')),
                    'unsupported restoration allocation arguments')
                event = dict(decision=len(decisions), scheduler_step=s.current_step,
                    host_s=time.perf_counter() - origin, request_id=r.request_id,
                    **runway(s, m, v, progress))
                decisions.append(event)
                if event['policy_deferred']:
                    event['native_called'] = False
                    return None  # Outside observer: do not label this native KV None.
                event['native_called'] = True
            else:
                # Native RUNNING/WAITING fast path: no signature binding.
                # Both positional and keyword forms preserve original forwarding.
                new_tokens = args[1] if len(args) > 1 else kwargs['num_new_tokens']
                cached_tokens = args[2] if len(args) > 2 else kwargs.get('num_new_computed_tokens', 0)
            begin = time.perf_counter()
            try:
                result = original(*args, **kwargs)  # Arguments remain byte-for-byte unchanged.
            finally:
                native_wall = time.perf_counter() - begin
            if event is not None:
                event['native_returned_none'] = result is None
            if result is not None:
                post = r.num_computed_tokens + cached_tokens + new_tokens
                progress[r.request_id] = (s.current_step, post)
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
    """Two CPU checks of physical headroom and partial-prefill/APC accounting."""
    from types import SimpleNamespace as NS
    class Coordinator:
        def __init__(self): self.held = {'running': 2}
        def get_num_blocks_to_allocate(self, request_id, num_tokens, new_computed_blocks,
                num_encoder_tokens, total_computed_tokens, num_local_computed_tokens,
                num_tokens_main_model, apply_admission_cap=False):
            hits = new_computed_blocks[0]
            return max((num_tokens + 15)//16 - self.held.get(request_id, 0) - len(hits), 0) + sum(b.ref_cnt == 0 for b in hits)
    co, pool = Coordinator(), NS(free=2)
    pool.get_num_free_blocks = lambda: pool.free
    m = NS(coordinator=co, block_pool=pool, max_model_len=4096,
           empty_kv_cache_blocks=NS(blocks=((),)), watermark_blocks=0)
    resident = NS(request_id='running', status='RUNNING', num_tokens=32, num_computed_tokens=31)
    s = NS(running=[resident], current_step=7)
    request = NS(request_id='restored', num_tokens=16, num_computed_tokens=0)
    v = dict(request=request, num_new_tokens=16, num_new_computed_tokens=0,
        new_computed_blocks=None, has_scheduled_reqs=True, reserved_blocks=0)
    progress = {'running': (7, 32)}
    before = dict(co.held)
    a = runway(s, m, v, progress)
    assert a['running_next_blocks'] == 1 and a['candidate_full_plus_one_blocks'] == 2
    assert a['policy_deferred']  # Current chunk fits, but next-round physical headroom does not.
    pool.free = 3
    assert not runway(s, m, v, progress)['policy_deferred']
    assert co.held == before and pool.free == 3  # Read-only, including allow case.
    request.num_tokens = 48; v['num_new_tokens'] = 1; v['num_new_computed_tokens'] = 16
    v['new_computed_blocks'] = NS(blocks=([NS(ref_cnt=0)],))
    b = runway(s, m, v, progress)
    assert b['candidate_full_plus_one_blocks'] == 4 and b['required_blocks'] == 5
    assert b['policy_deferred']  # Evictable APC hit consumes free capacity; chunk-only reserve fails.
    v['new_computed_blocks'].blocks[0][0].ref_cnt = 1
    assert runway(s, m, v, progress)['candidate_full_plus_one_blocks'] == 3
    assert co.held == before and pool.free == 3
    return dict(status='PASS', checks=2, gpu_used=False)


if __name__ == '__main__':
    print(json.dumps(self_test()))

#!/usr/bin/env python3
"""Small CPU fixtures using the real fixed-margin context and fake native objects."""
from collections import deque
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS

sys.dont_write_bytecode = True
from restore_cost_order_policy_v1 import restore_cost_order


class FCFSRequestQueue(deque):
    def peek_request(self): return self[0]
    def remove_request(self, request): self.remove(request)
    def prepend_request(self, request): self.appendleft(request)


class FullAttentionSpec: pass
class FullAttentionManager:
    def __init__(self): self.kv_cache_spec = FullAttentionSpec()


class Coordinator:
    def __init__(self):
        self.single_type_managers = [FullAttentionManager()]
        self.requests, self.hit_queries, self.physical_queries = {}, [], []
    def find_longest_cache_hit(self, hashes, max_length):
        r = self.requests[hashes[0]]
        assert max_length == r.num_tokens - 1
        self.hit_queries.append(r.request_id)
        return ((),), r.hit, 0
    def get_num_blocks_to_allocate(self, **kwargs):
        self.physical_queries.append(kwargs.copy())
        r = self.requests[kwargs['request_id']]
        assert kwargs['num_tokens'] == min(r.num_tokens+1, 4096)
        assert kwargs['apply_admission_cap'] is False
        return r.need


class Manager:
    def __init__(self, free=100):
        self.coordinator = Coordinator(); self.max_model_len = 4096
        self.enable_caching, self.watermark_blocks = True, 0
        self.empty_kv_cache_blocks = NS(blocks=((),))
        self.block_pool = NS(free=free)
        self.block_pool.get_num_free_blocks = lambda: self.block_pool.free
        self.token, self.outcome, self.forwarded = object(), 'success', []
    def create_kv_cache_blocks(self, blocks): return NS(blocks=blocks)
    def allocate_slots(self, request, num_new_tokens, num_new_computed_tokens=0,
            new_computed_blocks=None, num_lookahead_tokens=0,
            num_external_computed_tokens=0, delay_cache_blocks=False,
            num_encoder_tokens=0, full_sequence_must_fit=False,
            reserved_blocks=0, has_scheduled_reqs=True):
        if self.outcome == 'error': raise RuntimeError('native fixture error')
        if self.outcome == 'none': return None
        self.block_pool.free -= request.need
        return self.token


def setup(free=100):
    m = Manager(free)
    s = NS(kv_cache_manager=m, policy='fcfs', waiting=FCFSRequestQueue(),
        running=[], skipped_waiting=[], lora_config=None, connector=None,
        scheduler_config=NS(async_scheduling=False, enable_chunked_prefill=True,
                            long_prefill_token_threshold=0),
        num_waiting_for_streaming_input=0, use_eagle=False, num_lookahead_tokens=0,
        num_sampled_tokens_per_step=1, is_encoder_decoder=False,
        scheduler_reserve_full_isl=True, current_step=32)
    s._select_waiting_queue_for_scheduling = lambda: s.waiting
    native = m.allocate_slots
    def observer(*args, **kwargs):
        m.forwarded.append((args, kwargs.copy()))
        return native(*args, **kwargs)
    m.allocate_slots = observer
    engine = NS(engine_core=NS(engine_core=NS(scheduler=s)))
    return engine, s, m, s._select_waiting_queue_for_scheduling, observer


def request(m, rid, *, tokens=32, hit=0, need=2, status='PREEMPTED'):
    r = NS(request_id=rid, status=status, num_tokens=tokens, num_computed_tokens=0,
        has_encoder_inputs=False, skip_reading_prefix_cache=False,
        block_hashes=(rid,), hit=hit, need=need)
    m.coordinator.requests[rid] = r
    return r


def allocate(m, r, keyword=False):
    blocks = m.create_kv_cache_blocks(((),))
    kwargs = dict(num_new_computed_tokens=r.hit, new_computed_blocks=blocks,
        full_sequence_must_fit=True, reserved_blocks=0, has_scheduled_reqs=False)
    if keyword:
        kwargs.update(request=r, num_new_tokens=r.num_tokens-r.hit)
        result = m.allocate_slots(**kwargs)
        expected_args = ()
    else:
        expected_args = (r, r.num_tokens-r.hit)
        result = m.allocate_slots(*expected_args, **kwargs)
    if m.forwarded:
        args, actual = m.forwarded[-1]
        assert args == expected_args and actual == kwargs
        assert (args[0] if args else actual['request']) is r
        assert actual['new_computed_blocks'] is blocks
    return result


def self_test():
    passed = []
    with tempfile.TemporaryDirectory(prefix='restore-cost-order-check-') as tmp:
        root = Path(tmp)
        def path(label):
            directory = root/label; directory.mkdir(); return directory/'restore-cost-order-policy.json'
        # Real fixed gate, stable cost tie, identity forwarding, no fresh crossing or peek mutation.
        e,s,m,old_select,old_allocate = setup()
        h=request(m,'head'); b=request(m,'b',hit=16); c=request(m,'c',hit=16)
        fresh=request(m,'fresh',status='WAITING',tokens=1,need=1)
        hidden=request(m,'behind-fresh',tokens=1,need=1)
        original=[h,b,c,fresh,hidden]
        with restore_cost_order(e,path('min'), 'min-recompute') as report:
            s.waiting.extend(original)
            assert s.waiting.peek_request() is h and not report['decisions']
            assert not m.coordinator.hit_queries and list(s.waiting)==original
            s.current_step += 1; assert s._select_waiting_queue_for_scheduling() is s.waiting
            assert list(s.waiting)==[b,h,c,fresh,hidden] and m.block_pool.free==100
            d=report['decisions'][0]
            assert d['selected_id']=='b' and d['reorders_fit_head_for_lower_cost']
            assert d['prefix_request_ids']==['head','b','c'] and d['queried_candidates']==3
            assert m.coordinator.hit_queries==['head','b','c']
            assert allocate(m,b,keyword=True) is m.token
            a=report['allocation_receipts'][0]
            assert a['native_called'] and a['native_succeeded'] and a['native_returned_none'] is False
            assert (a['free_blocks_before'],a['free_blocks_after'])==(100,98)
            assert a['scheduler_step']-(report['initial_scheduler_step']+1)==0
        assert s._select_waiting_queue_for_scheduling is old_select and m.allocate_slots is old_allocate
        assert report['hooks_restored']; passed.append('tie_identity_peek_and_fresh_boundary')
        # First-fit short-circuits on head even when a later request has smaller cost.
        e,s,m,_,_ = setup(); h=request(m,'head'); low=request(m,'low',hit=16)
        with restore_cost_order(e,path('first-head'),'first-fit') as report:
            s.waiting.extend([h,low]); s._select_waiting_queue_for_scheduling()
            assert list(s.waiting)==[h,low] and m.coordinator.hit_queries==['head']
            assert report['decisions'][0]['action']=='KEEP_RESTORE_HEAD'
            assert allocate(m,h) is m.token
        passed.append('first_fit_short_circuit_head_unchanged')
        # Native None and policy-gate None are distinct; both restore exact original ordering.
        for outcome in ('none','policy-none','error'):
            e,s,m,old_select,old_allocate = setup(50)
            h=request(m,'head',need=3); fit=request(m,'fit'); later=request(m,'later',hit=16,need=1)
            original=[h,fit,later]; target=path(outcome)
            try:
                with restore_cost_order(e,target,'first-fit') as report:
                    s.waiting.extend(original); s._select_waiting_queue_for_scheduling()
                    assert s.waiting.peek_request() is fit and m.coordinator.hit_queries==['head','fit']
                    assert report['decisions'][0]['bypasses_unfit_head']
                    if outcome=='policy-none': m.block_pool.free=49
                    else: m.outcome=outcome
                    result=allocate(m,fit)
                    assert result is None and list(s.waiting)==original
            except RuntimeError as exc:
                assert outcome=='error' and str(exc)=='native fixture error'
            else: assert outcome!='error'
            assert list(s.waiting)==original and report['decisions'][0]['queue_rollback']
            assert s._select_waiting_queue_for_scheduling is old_select and m.allocate_slots is old_allocate
            a=report['allocation_receipts'][0]
            assert a['native_called'] is (outcome!='policy-none')
            assert a['native_succeeded'] is (False if outcome=='none' else None)
            assert len(m.forwarded)==(0 if outcome=='policy-none' else 1)
            assert report['hooks_restored']
        passed.append('native_none_policy_none_exception_exact_rollback')
        # No fit, lone restore, fresh head, and16-prefix boundary preserve the queue.
        for case in ('none-fit','single','fresh-head','limit16'):
            e,s,m,_,_=setup(50)
            if case=='none-fit': rows=[request(m,'a',need=3),request(m,'b',need=4)]
            elif case=='single': rows=[request(m,'a'),request(m,'fresh',status='WAITING')]
            elif case=='fresh-head': rows=[request(m,'fresh',status='WAITING'),request(m,'a')]
            else: rows=[request(m,str(i)) for i in range(16)]+[request(m,'unseen-cheapest',tokens=1,need=1)]
            with restore_cost_order(e,path(case),'min-recompute') as report:
                s.waiting.extend(rows); s._select_waiting_queue_for_scheduling()
                assert list(s.waiting)==rows
                if case in ('single','fresh-head'):
                    assert not report['decisions'] and not m.coordinator.hit_queries
                elif case=='none-fit':
                    assert report['decisions'][0]['action']=='NO_RESTORE_CANDIDATE_FITS'
                    assert allocate(m,rows[0]) is None and not m.forwarded
                else:
                    assert len(m.coordinator.hit_queries)==16 and 'unseen-cheapest' not in m.coordinator.hit_queries
                    assert report['decisions'][0]['prefix_boundary']=='LIMIT16'
                    assert allocate(m,rows[0]) is m.token
        passed.append('none_fit_single_fresh_head_limit16')
        # Interrupted after selection but before allocation must roll back and restore hooks.
        e,s,m,old_select,old_allocate=setup(); h=request(m,'h'); low=request(m,'low',hit=16)
        try:
            with restore_cost_order(e,path('abandoned'),'min-recompute') as report:
                s.waiting.extend([h,low]); s._select_waiting_queue_for_scheduling()
                raise InterruptedError('fixture interruption')
        except InterruptedError: pass
        assert list(s.waiting)==[h,low] and report['decisions'][0]['queue_rollback']
        assert s._select_waiting_queue_for_scheduling is old_select and m.allocate_slots is old_allocate
        assert report['hooks_restored']; passed.append('interrupted_pending_selection_cleanup')
    return dict(status='PASS', checks=passed, gpu_used=False)


if __name__ == '__main__':
    print(json.dumps(self_test()))

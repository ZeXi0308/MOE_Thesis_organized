"""Bounded CPU checks of candidate identity, rollback, and observer separation."""
from collections import deque
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
from restore_new_bypass_policy_v1 import restore_new_bypass

class FullAttentionSpec: pass
class FullAttentionManager:
    def __init__(self): self.kv_cache_spec = FullAttentionSpec()
class FCFSRequestQueue(deque):
    def peek_request(self): return self[0]
    def remove_request(self, r): self.remove(r)
    def prepend_request(self, r): self.appendleft(r)
class Coordinator:
    def __init__(self):
        self.single_type_managers = [FullAttentionManager()]
        self.lookups = []
    def find_longest_cache_hit(self, hashes, maximum):
        self.lookups.append((hashes, maximum))
        return ((),), 0, 0
    def get_num_blocks_to_allocate(self, **kw):
        return (kw['num_tokens']+15)//16
class Manager:
    def __init__(self):
        self.coordinator = Coordinator()
        self.max_model_len, self.watermark_blocks = 4096, 0
        self.enable_caching = True
        self.empty_kv_cache_blocks = NS(blocks=((),))
        self.block_pool = NS(free=58)
        self.block_pool.get_num_free_blocks = lambda: self.block_pool.free
        self.calls, self.fail_next = [], False
    def create_kv_cache_blocks(self, blocks): return NS(blocks=blocks)
    def get_computed_blocks(self, request): raise AssertionError('stats-mutating lookup used')
    def allocate_slots(self, request, num_new_tokens, num_new_computed_tokens=0,
            new_computed_blocks=None, num_lookahead_tokens=0,
            num_external_computed_tokens=0, delay_cache_blocks=False,
            num_encoder_tokens=0, full_sequence_must_fit=False,
            reserved_blocks=0, has_scheduled_reqs=True):
        self.calls.append(request.request_id)
        return None if self.fail_next else object()

def request(rid, status, tokens):
    return NS(request_id=rid, status=status, num_tokens=tokens,
              num_computed_tokens=0, has_encoder_inputs=False,
              skip_reading_prefix_cache=False, block_hashes=(rid,))

def fixture():
    m = Manager()
    s = NS(kv_cache_manager=m, policy='fcfs', waiting=FCFSRequestQueue(),
        skipped_waiting=FCFSRequestQueue(), running=[], lora_config=None,
        connector=None, num_waiting_for_streaming_input=0, use_eagle=False,
        num_lookahead_tokens=0, num_sampled_tokens_per_step=1,
        is_encoder_decoder=False, scheduler_reserve_full_isl=True, current_step=4,
        scheduler_config=NS(async_scheduling=False,enable_chunked_prefill=True,
                            long_prefill_token_threshold=0))
    s._select_waiting_queue_for_scheduling = lambda: s.skipped_waiting or s.waiting or None
    return NS(engine_core=NS(engine_core=NS(scheduler=s))), s, m

def main():
    with tempfile.TemporaryDirectory() as td:
        for case in ('admit', 'rollback', 'head_fits', 'none_fits'):
            e,s,m = fixture()
            old_select, old_allocate = s._select_waiting_queue_for_scheduling, m.allocate_slots
            path=Path(td)/case;path.mkdir()
            with restore_new_bypass(e, path/'policy.json',48):
                head=request('restore','PREEMPTED',160)
                other=request('other_restore','PREEMPTED',176)
                fresh=request('new','WAITING',64)
                s.waiting.extend((head,other,fresh));s.running.append(request('run','RUNNING',70))
                original=list(s.waiting)
                assert s.waiting.peek_request() is head and not m.coordinator.lookups
                if case=='head_fits':m.block_pool.free=60
                if case=='none_fits':m.block_pool.free=50
                q=s._select_waiting_queue_for_scheduling()
                if case in ('admit','rollback'):
                    assert q is s.waiting and list(q)==[fresh,head,other]
                    m.fail_next=case=='rollback'
                    result=m.allocate_slots(fresh,64,full_sequence_must_fit=True)
                    assert m.calls==['new']
                    if case=='rollback':assert result is None and list(q)==original
                    else:assert result is not None and q.popleft() is fresh and list(q)==[head,other]
                else:
                    assert list(q)==original and not m.calls
                    if case=='head_fits':assert len(m.coordinator.lookups)==1
                s.running.clear();s.waiting.clear()
            assert s._select_waiting_queue_for_scheduling is old_select
            assert m.allocate_slots==old_allocate
            result=json.loads((path/'policy.json').read_text())
            assert result['status']=='COMPLETE'
            assert result['applied_reorders']==int(case in ('admit','rollback'))
            assert result['native_successful_bypass_allocations']==int(case=='admit')
    print('PASS4: native admission identity; failed-allocation rollback; feasible-head no bypass; no-fit unchanged; observer peek inert; hooks restored')

if __name__=='__main__': main()

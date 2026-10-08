"""Small CPU mock checks for one-event gating; not native/GPU correctness evidence."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
from selector import RestoreSelector, token_hash, validate_target_spec


class CPUOffloadingManager:
    pass


class Status:
    def __init__(self):
        self.reset_calls = []

    def update_num_hit_chunks(self, tokens):
        self.reset_calls.append(tokens)


def request(rid='internal-target'):
    return NS(request_id=rid, num_preemptions=1, num_tokens=8,
              num_prompt_tokens=4, num_output_tokens=4, all_token_ids=list(range(8)),
              sampling_params=object(), status='PREEMPTED', stop_reason=None)


SPEC = dict(schema='E.one_event_target.v1', target=dict(external_id='measured/E249',
    num_preemptions=1, known_tokens=8, generated_tokens=4, prefix_sha256=token_hash(range(8))))


def fixture(policy='recompute', spec=None, threshold=0):
    state = NS(free=100, native=(4, True), allocations=[])
    cs = NS(config=NS(kv_group_configs=[NS(tokens_per_block=4, sliding_window_size_in_chunks=None)]),
            manager=CPUOffloadingManager(), _jobs={}, _req_status={})
    cs.get_num_new_matched_tokens = lambda req, local: state.native
    def alloc(req, blocks, external):
        state.allocations.append((req.request_id, external))
        return 'native_alloc'
    cs.update_state_after_alloc = alloc
    cs.update_connector_output = lambda output: None
    scheduler = NS(connector=NS(connector_scheduler=cs), scheduler_reserve_full_isl=True,
        kv_cache_manager=NS(enable_caching=False, watermark_blocks=0,
                            block_pool=NS(get_num_free_blocks=lambda: state.free)),
        _inflight_prefill_reserved_blocks=lambda: 0, running=[], waiting=[],
        _preempt_request=lambda req, timestamp: None)
    selector = RestoreSelector(scheduler, policy, threshold, target_spec=copy.deepcopy(spec))
    def bind(req, external='measured/E249'):
        cs._req_status[req.request_id] = Status()
        if spec is not None:
            selector.register_request(req.request_id, external)
        return req
    return selector, state, bind


def commit(selector, req, result):
    return selector.allocated(req, NS(blocks=[[0, 1]]), result[0] or 0)


def main():
    passed = []
    # Existing full-policy host, recompute and length decisions stay unchanged.
    for policy, threshold, expected in [('host',0,(4,True)), ('recompute',0,(0,False)),
                                         ('length',8,(0,False)), ('length',7,(4,True))]:
        s, state, bind = fixture(policy, threshold=threshold)
        q = bind(request())
        for _ in range(2):
            result = s.lookup(q,0)
            assert result == expected
            assert commit(s,q,result) == 'native_alloc'
        assert s.target_commit_count == 0 and len(s.commits) == 2
    passed.append('default full-policy host/recompute/length and repeated recompute unchanged')
    s, state, bind = fixture(spec=SPEC)
    q = bind(request())
    other = bind(request('internal-other'),'measured/E248')
    assert s.lookup(other,0) == state.native
    assert s.events[-1]['target_reason'] == 'other_request_native_host'
    before = (tuple(q.all_token_ids),q.sampling_params,q.num_output_tokens,q.status,q.stop_reason)
    assert s.lookup(q,0) == (0,False)
    assert s.target_commit_count == 0, 'lookup is not an allocation commit'
    # Retry after no successful allocation may still select the same legal event.
    result = s.lookup(q,0)
    assert result == (0,False) and s.target_commit_count == 0
    commit(s,q,result)
    assert s.target_commit_count == 1 and s.target_summary()['target_reached']
    assert s.commits[-1]['target_committed'] and s.commits[-1]['actual_action'] == 'recompute'
    assert before == (tuple(q.all_token_ids),q.sampling_params,q.num_output_tokens,q.status,q.stop_reason)
    for req in (q,other):
        result=s.lookup(req,0)
        assert result == state.native
        commit(s,req,result)
    assert s.target_commit_count == 1
    assert sum(c.get('target_committed',False) for c in s.commits) == 1
    assert [c['actual_action'] for c in s.commits] == ['recompute','host','host']
    passed.append('one successful target recompute only; precommit retry, then same target and peers stay Host')
    s.clear()
    assert not s.external_ids and not s.events and not s.commits and s.target_commit_count == 0
    q=bind(request()); s.enabled=False
    assert s.lookup(q,0) == state.native and not s.events
    commit(s,q,state.native)
    assert s.target_commit_count == 0 and s.target_summary()['target_result'] == 'DISABLED'
    passed.append('clear resets per-episode latch and identity map; disabled warmup cannot intervene')
    s, state, bind=fixture('host',SPEC);q=bind(request())
    result=s.lookup(q,0);assert result==state.native
    commit(s,q,result)
    assert s.target_summary()['target_effective_action']=='host' and s.target_commit_count==1
    passed.append('Host control records one matched legal target while retaining native Host action')
    for reason in ['num_preemptions_mismatch','known_tokens_mismatch','generated_tokens_mismatch',
                   'prefix_sha256_mismatch','native_host_not_ready','joint_capacity_unavailable']:
        s,state,bind=fixture(spec=SPEC);q=bind(request())
        if reason=='num_preemptions_mismatch':q.num_preemptions=2
        elif reason=='known_tokens_mismatch':q.num_tokens=9;q.num_prompt_tokens=5;q.all_token_ids.append(9)
        elif reason=='generated_tokens_mismatch':q.num_output_tokens=3;q.num_prompt_tokens=5
        elif reason=='prefix_sha256_mismatch':q.all_token_ids[-1]=99
        elif reason=='native_host_not_ready':state.native=(None,False)
        else:state.free=1
        assert s.lookup(q,0)==state.native
        assert not s.events[-1]['target_selected'] and reason in s.events[-1]['target_reason']
        assert not s.cs._req_status[q.request_id].reset_calls and s.target_commit_count==0
        assert not s.target_summary()['target_comparison_eligible']
    s,state,bind=fixture(spec=SPEC);q=bind(request());state.native=(0,False)
    result=s.lookup(q,0);assert result==state.native;commit(s,q,result)
    assert s.target_commit_count==0 and s.events[-1]['fallback']=='host_miss_native_recompute'
    passed.append('identity/state/native-ready/joint-capacity mismatch preserves native result and fallback')
    root=Path(__file__).resolve().parent
    actual=json.loads((root/'one_event_target_spec.json').read_text());validate_target_spec(actual)
    receipt=dict(status='CPU_MOCK_PASS_GPU_UNRUN',checks=passed,
        limitation='Mock gating only: does not prove real allocation, output quality, timing, or GPU correctness.',
        file_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in
                     ['selector.py','run_cell.py','run_group.py','one_event_target_spec.json','check_one_event.py']})
    (root/'one_event_cpu_check.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()

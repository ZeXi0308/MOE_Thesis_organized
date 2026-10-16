"""Three requested CPU gates, using pinned native code and a real KNOWN trace."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import horizon_sched as hz
from check_cpu import growth_scenario

ROOT=Path(__file__).resolve().parents[2]
PIN=ROOT/'MOE_Thesis_organized/refine-logs/expert_saturation/experiments/admission_capacity/20260929_commit_recheck/liveness_pinned_sources_20260930/scheduler.py'


def check_victim():
    source=PIN.read_text()
    assert hashlib.sha256(source.encode()).hexdigest()==hz.PINS['Scheduler']
    tree=hz.instrument(source)
    compile(tree,'<whole-pinned-schedule>','exec')
    branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
                and ast.unparse(n.test)=='self.policy == SchedulingPolicy.PRIORITY')
    # Execute the ACTUAL transformed FCFS branch, including native refunds.
    code=compile(ast.fix_missing_locations(ast.Module(body=branch.orelse,type_ignores=[])),
                 '<native-victim-and-rollback>','exec')
    for selected,leased in ((0,{'b':40,'c':40}),(2,{'a':40,'b':40,'c':40}),(2,{'c':32})):
        requests=[NS(request_id=x) for x in ('a','b','c')]
        s=NS(running=list(requests),kv_cache_manager=NS(coordinator=NS(single_type_managers=[NS()])),connector=None)
        adapter=hz.Adapter(s,hz.Config());adapter.step=32;adapter.leases=leased
        ns=dict(_hz=adapter,self=s,req_index=1,scheduled_running_reqs=[requests[0]],
                token_budget=7,num_scheduled_tokens={'a':5},req_to_new_blocks={'a':'blocks'},
                scheduled_spec_decode_tokens={'a':[1]},scheduled_encoder_inputs={},encoder_compute_budget=0)
        exec(code,ns)
        assert ns['preempted_req'] is requests[selected]
        if selected==0:
            assert ns['token_budget']==12 and ns['req_index']==0
            assert not ns['scheduled_running_reqs'] and not ns['num_scheduled_tokens']
            assert not ns['req_to_new_blocks'] and not ns['scheduled_spec_decode_tokens']
        else:
            assert ns['token_budget']==7 and ns['req_index']==1
        adapter.uninstall();adapter.uninstall()
    # Lease expiry is driven by expiry buckets, including a renewed lease
    # and immediate removal of obsolete renewal/completion entries.
    adapter=hz.Adapter(s,hz.Config())
    adapter._lease('a',2);adapter._lease('b',2)
    adapter.begin()
    adapter._lease('a',3)
    assert adapter.lease_expirations=={2:{'b'},4:{'a'}}
    adapter._unlease('b')
    adapter._unlease('b')
    assert adapter.lease_expirations=={4:{'a'}}
    adapter.begin()
    assert adapter.leases=={'a':4}
    assert 2 not in adapter.lease_expirations
    adapter.begin();assert adapter.leases=={'a':4}
    adapter.begin();assert not adapter.leases and not adapter.lease_expirations
    adapter._lease('a',32);adapter._clear()
    assert not adapter.leases and not adapter.lease_expirations
    s.requests={'a':NS(is_finished=lambda:False),'b':NS(is_finished=lambda:True)}
    adapter.receipt('a',1.);adapter.receipt('b',1.);adapter.receipt('removed',1.)
    assert adapter.receipts=={'a':1.}
    return dict(native_branch_executed=True,prefix_refund=True,
                all_leased_native_fallback=True,expiry=True,
                expiry_bucket_renewal_and_removal=True)


def check_known():
    path=ROOT/'B_recovery_order_20261004/recovery_start_gate/session-20261008-r02/cell-00-cap256-native/output/recovery-start-gate.json'
    snap=json.loads(path.read_text())['progress_observation']['snapshots'][0]
    assert snap['status']=='KNOWN'
    columns=snap['columns']
    rows=[dict(zip(columns,r)) for r in snap['requests']]
    target=dict(zip(columns,snap['target']))
    single=NS(req_to_blocks={},_partial_hit_reqs=set())
    requests={}
    scheduled={}
    for row in [*rows,target]:
        rid=row['request']
        r=NS(request_id=rid,num_tokens=row['history_tokens'],num_computed_tokens=row['computed_tokens'],
             num_in_flight_tokens=row['num_in_flight_tokens'],has_encoder_inputs=False,
             num_output_placeholders=0,spec_token_ids=[],num_preemptions=1 if rid==target['request'] else 0)
        requests[rid]=r
        single.req_to_blocks[rid]=[NS(is_null=False,ref_cnt=1) for _ in range(row['held_gpu_blocks'])]
        scheduled[rid]=row['scheduled_tokens_this_step']
    m=NS(coordinator=NS(single_type_managers=[single]),max_model_len=snap['max_model_len'],
         block_pool=NS(get_num_free_blocks=lambda:snap['free_gpu_blocks']))
    s=NS(kv_cache_manager=m,connector=None,requests=requests,
         running=[requests[r] for r in snap['running_ids']],
         _inflight_prefills=[requests[r] for r in snap['inflight_ids']],
         _inflight_prefill_reserved_blocks=lambda:snap['native_reserved_blocks'])
    from capacity import known_capacity
    a=hz.Adapter(s,hz.Config(H_a=32,H_r=32));a.begin()
    expected=known_capacity(snap,32)
    actual=a.reject(requests[target['request']],scheduled,0,0,0,NS(blocks=[[]]))
    assert a.ledger['status']=='KNOWN',a.ledger
    assert a.data['steps'][-1]['required']==expected['required_free_blocks']
    assert actual==(not expected['capacity_fit'])
    # UNKNOWN must preserve native choice.
    s._inflight_prefill_reserved_blocks=lambda:snap['native_reserved_blocks']+1
    a.begin();assert a.reject(requests[target['request']],scheduled,0,0,0,NS(blocks=[[]])) is False
    assert a.ledger['status']=='UNKNOWN'
    # Native async LOAD membership transition: allocated() runs before the
    # request enters _inflight_prefills, then the next step rebuilds the ledger.
    # Both paths must retain the same future growth, with no double charging
    # when ACK moves the owner into running.
    r=NS(request_id='load',num_tokens=16,num_computed_tokens=0,
         num_in_flight_tokens=0,has_encoder_inputs=False,num_output_placeholders=0,
         spec_token_ids=[],num_preemptions=1)
    single=NS(req_to_blocks={'load':[NS(is_null=False,ref_cnt=1)]},_partial_hit_reqs=set())
    m=NS(coordinator=NS(single_type_managers=[single]),max_model_len=4096)
    s=NS(kv_cache_manager=m,connector=None,requests={'load':r},running=[],
         _inflight_prefills=[],_inflight_prefill_reserved_blocks=lambda:0)
    transition=hz.Adapter(s,hz.Config(H_a=32,H_r=32))
    transition._build({});transition.allocated(r,True)
    incremental=dict(transition.ledger['demand'])
    s._inflight_prefills=[r];r.num_computed_tokens=16
    transition._build({})
    assert transition.ledger['demand']==incremental=={32:2}
    s.running=[r]
    transition._build({})
    assert transition.ledger['demand']==incremental
    # A complete, unchanged LOAD reentry keeps the cached demand and updates
    # membership once. Compare the fast path to a full post-transition rebuild.
    s.running=[]
    transition._build({})
    cached=transition.ledger
    transition.allocated(r,False)
    assert transition.ledger is cached and cached['n']==1
    expected_reentry={k:cached[k] for k in ('demand','R','I','n','native_reserved')}
    s.running=[r]
    transition._build({})
    assert {k:transition.ledger[k] for k in expected_reentry}==expected_reentry
    # Changed allocation cannot reuse the old demand, even with the same id.
    s.running=[]
    transition._build({})
    single.req_to_blocks['load'].append(NS(is_null=False,ref_cnt=1))
    transition.allocated(r,False)
    assert transition.ledger is None
    # A retry callback or asynchronous allocation must not count a second
    # running owner using the unchanged-history shortcut.
    single.req_to_blocks['load'].pop()
    transition._build({})
    transition.allocated(r,False)
    transition.allocated(r,False)
    assert transition.ledger is None
    transition._build({})
    transition.allocated(r,True)
    assert transition.ledger is None
    # New requests can also LOAD a shared prefix without a prior preemption.
    # Their inflight ownership must not be charged again on synchronous entry.
    r.num_preemptions=0
    transition.pending.clear()
    transition._build({})
    transition.allocated(r,False)
    assert transition.ledger['demand']=={32:2},transition.ledger
    assert transition.ledger['n']==1
    expected_new={k:transition.ledger[k] for k in ('demand','R','I','n','native_reserved')}
    s.running=[r]
    reference=hz.Adapter(s,hz.Config(H_a=32,H_r=32))
    reference._build({})
    assert {k:reference.ledger[k] for k in expected_new}==expected_new
    s.running=[]
    transition.allocated(r,False)
    assert transition.ledger is None
    # The measured new LOAD lookups matched only 32 tokens. Such partial
    # history must invalidate after allocating more pages, not add a second
    # copy of the old owner's outstanding history to the ledger.
    r.num_tokens=512;r.num_computed_tokens=32
    single.req_to_blocks['load']=[NS(is_null=False,ref_cnt=1) for _ in range(2)]
    s._inflight_prefill_reserved_blocks=lambda:30
    transition._build({})
    assert transition.ledger['demand']=={32:32}
    single.req_to_blocks['load'] += [NS(is_null=False,ref_cnt=1) for _ in range(30)]
    transition.allocated(r,False)
    assert transition.ledger is None
    s.running=[r];s._inflight_prefill_reserved_blocks=lambda:0
    transition._build({'load':480})
    assert transition.ledger['demand']=={32:2} and transition.ledger['n']==1
    s.running=[];r.num_tokens=16;r.num_computed_tokens=16
    single.req_to_blocks['load']=[NS(is_null=False,ref_cnt=1)]
    r.num_preemptions=1
    # Exercise the installed ACK wrapper: native consumes the failure flag,
    # so invalidation must use the state observed before calling native.
    s.failed_recving_kv_req_ids={'load'}
    s._update_waiting_for_remote_kv=lambda request:s.failed_recving_kv_req_ids.discard(request.request_id)
    s._preempt_request=lambda request:None
    s._free_request=lambda request:None
    s.reset_prefix_cache=lambda:True
    s.reset_connector_cache=lambda:True
    transition.attach(None,schedule_hook=False)
    transition._build({})
    s._update_waiting_for_remote_kv(r)
    assert not s.failed_recving_kv_req_ids and transition.ledger is None
    transition._build({})
    cached=transition.ledger
    s._update_waiting_for_remote_kv(r)
    assert transition.ledger is cached
    transition.uninstall()
    return dict(real_snapshot=str(path.relative_to(ROOT)),
                snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                required=expected['required_free_blocks'],matches_reference=True,unknown_native_fallback=True,
                inflight_growth_survives_rebuild=True,ack_membership_not_double_charged=True,
                unchanged_reentry_matches_full_rebuild=True,changed_reentry_invalidates=True,
                new_load_entry_not_double_charged=True,
                partial_new_load_reentry_invalidates=True,
                failed_ack_invalidates_before_reentry=True)


if __name__=='__main__':
    print(json.dumps(dict(growth=growth_scenario(),victim=check_victim(),known=check_known(),
        scope='pinned code branch and recorded native snapshot; full engine validation follows on GPU'),indent=2))

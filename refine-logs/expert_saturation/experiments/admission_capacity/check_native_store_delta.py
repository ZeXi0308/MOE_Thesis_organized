"""CPU interface fixtures for repeated native-store metadata; no host allocator."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS
from native_store_delta import inspect_store_delta


def check():
    plan=NS(victim=NS(request_id='r'),source_blocks=(11,12,13,14))
    state=NS(req=NS(request_id='r'),group_states=[NS(block_ids=[11,12,13,14],offload_keys=['a','b','c','d'])],transfer_jobs={7})
    def case(blocks,keys):
        return NS(store_jobs={7:NS(req_id='r',src_spec=NS(block_ids=blocks))}),{7:NS(req_id='r',is_store=True,keys=keys)}
    rows={}
    for name,blocks,keys in [('full',[11,12,13,14],['a','b','c','d']),('suffix',[13,14],['c','d']),('sparse_missing_keys',[12,14],['b','d'])]:
        meta,jobs=case(blocks,keys);before=deepcopy(vars(state.group_states[0]))
        rows[name]=inspect_store_delta(plan,meta,state,jobs)
        assert vars(state.group_states[0])==before
    rows['no_new_job']=inspect_store_delta(plan,NS(store_jobs={}),state,{})
    assert rows['no_new_job']['status']=='NO_NEW_STORE' and rows['no_new_job']['full_prefix_residency']=='UNKNOWN'
    for name,blocks,keys in [('wrong_keys',[13,14],['a','b']),('unknown_block',[99],['d']),('duplicate',[14,14],['d']),('wrong_order',[14,13],['c','d'])]:
        meta,jobs=case(blocks,keys)
        try:inspect_store_delta(plan,meta,state,jobs)
        except ValueError as e:rows[name]=dict(rejected=str(e))
        else:raise AssertionError(name)
    return dict(status='CPU_INTERFACE_FIXTURES_PASS',cases=rows,scope='Synthetic job/key/block objects only. No native builder, eviction, allocation, transfer or repeated GPU policy verified.')

if __name__=='__main__':print(json.dumps(check(),indent=2))

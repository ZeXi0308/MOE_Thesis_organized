"""Focused CPU regression for continuation-allocation links from real records.

Metadata projections of H records 0/1/2 and R records 5/6/7 from
/private/tmp/e_history_link_diagnostic.json, SHA256:
9fb324874a64196c640588e1896e35e4f9bb0cea67aa7d5be4d4a48949ae6b00.
Only fields used by allocation_link are retained; this checks analysis logic,
not the native engine, GPU correctness, or any performance hypothesis.
"""
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze_host_history import allocation_link


def fixture(request_id,preemptions,root_index,times,event):
    state=dict(request_id=request_id,preemptions=preemptions)
    root=dict(record_index=root_index,operation='allocation',status='returned',time_s=times[0],
              before=dict(state),after=dict(state),external_tokens=1168,lookup=dict(event=event))
    continuation=dict(record_index=root_index+1,operation='allocation',status='returned',time_s=times[1],
                      before=dict(state),after=dict(state),external_tokens=0,lookup=None)
    link=dict(record_index=root_index+1,time_s=times[1],preemptions=preemptions,external_tokens=0,lookup=None)
    builder=dict(record_index=root_index+2,operation='store_builder',status='returned',time_s=times[2],
                 before=[dict(state,last_allocation=link)])
    return [root,continuation,builder]


def check(records):
    return allocation_link(records,records[0],records[2],records[2]['before'][0])


def main():
    host=fixture('measured/E246-8435c26c',1,0,
                 (24.813267916440964,24.88443037122488,24.88518873602152),0)
    recompute_then_host=fixture('measured/E246-856dbded',2,5,
                 (25.401326168328524,25.475265473127365,25.47616147994995),1)
    for records in (host,recompute_then_host):
        result=check(records)
        assert result['valid'] and result['intermediate_allocation_count']==1,result

    missing=deepcopy(host);del missing[2]['before'][0]['last_allocation']
    stale=deepcopy(host);root=stale[0]
    stale[2]['before'][0]['last_allocation']=dict(record_index=root['record_index'],time_s=root['time_s'],
        preemptions=root['before']['preemptions'],external_tokens=root['external_tokens'],lookup=root['lookup'])
    wrong_preemption=deepcopy(host)
    wrong_preemption[1]['before']['preemptions']=wrong_preemption[1]['after']['preemptions']=2
    wrong_request=deepcopy(host);wrong_request[1]['after']['request_id']='different-request'
    for records in (missing,stale,wrong_preemption,wrong_request):
        assert not check(records)['valid'],check(records)
    print('PASS CPU analysis regression: 2 real metadata chains; 4 invalid links rejected. No GPU execution.')


if __name__=='__main__':
    main()

"""Three CPU arithmetic checks; native adapter validation is still pending."""
import importlib.util
import json
from pathlib import Path
from capacity import ceildiv, fluid_growth, phase_growth, known_capacity, victim_index

ROOT = Path(__file__).resolve().parents[2]


def growth_scenario():
    # The requested 250-request/32-step scenario, all 16 page phases.
    for phase in range(16):
        lengths = [320 + phase] * 250
        held = [ceildiv(n, 16) for n in lengths]
        residues = [0] * 16
        residues[phase] = 250
        free = phase_growth(residues, 32)
        assert free == fluid_growth(250, 32) == 500
        for h in range(1, 33):
            needed = [ceildiv(n+h,16) for n in lengths]
            growth = sum(a-b for a,b in zip(needed, held))
            assert growth <= free
            free -= growth
            held = needed
        assert free == 0
    # Expose, rather than hide, the assumptions needed for the paper lemma.
    assert fluid_growth(250, 1) == 16
    assert phase_growth([250] + [0]*15, 1) == 250
    assert phase_growth([251] + [0]*15, 32) == 502
    return dict(n=250, H=32, phases_checked=16, reserved=500,
                one_step_fluid=16, one_step_exact=250,
                with_recovered_target=502)


def victim_scenario():
    assert victim_index(['a','b','c'], {'b':40,'c':40}, 32) == (0,2,False)
    assert victim_index(['a','b','c'], {'a':40,'b':40,'c':40}, 32) == (2,3,True)
    assert victim_index(['a','b','c'], {'c':32}, 32) == (2,0,False)
    return dict(skip=True, all_leased_fallback=True, expiry=True,
                native_rollback_tested=False)


def known_scenario():
    path=ROOT/'B_recovery_order_20261004/recovery_start_gate/progress_capacity.py'
    spec=importlib.util.spec_from_file_location('existing_capacity',path)
    old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    columns=['request','status','history_tokens','held_gpu_blocks','computed_tokens',
             'num_in_flight_tokens','scheduled_tokens_this_step','is_running',
             'is_inflight','registry_identity','plain_unshared_blocks']
    rows=[['a','RUNNING',512,32,512,0,0,True,False,True,True],
          ['b','RUNNING',1024,32,512,0,0,True,True,True,True],
          ['c','WAITING_FOR_REMOTE_KVS',512,16,256,0,0,False,True,True,True]]
    target=['t','PREEMPTED',1024,0,0,0,0,False,False,True,True]
    s=dict(status='KNOWN',columns=columns,requests=rows,target=target,
           running_ids=['a','b'],inflight_ids=['b','c'],block_size=16,
           max_model_len=4096,free_gpu_blocks=200,native_reserved_blocks=48)
    for h in (0,1,16,32,64):
        a=known_capacity(s,h)
        b=old.estimate(target=dict(zip(columns,target)),
            requests=[dict(zip(columns,r)) for r in rows],running_ids=s['running_ids'],
            inflight_ids=s['inflight_ids'],native_reserved_blocks=48,
            free_gpu_blocks=200,block_size=16,max_model_len=4096,
            growth_tokens=h,target_growth_tokens=h,qualified=True)
        assert a['status']==b['status']=='KNOWN'
        for key in ('inflight_remaining_blocks','target_total_blocks'):
            assert a[key]==b[key], (h,key,a,b)
        # The native reference reserves c's history only. HORIZON additionally
        # covers c entering decode within the window: ceil(H/16) more pages.
        extra=ceildiv(h,16)
        assert a['required_free_blocks']==b['required_free_blocks']+extra
        assert a['owner_increment_blocks']==b['running_increment_blocks']+extra
        assert a['capacity_fit']==(a['required_free_blocks']<=200)
    assert known_capacity(dict(s,native_reserved_blocks=49),32)['status']=='UNKNOWN'
    assert known_capacity(dict(s,status='UNKNOWN'),32)['capacity_fit'] is None
    return dict(matches_native_history_plus_inflight_growth=True, overlap_counted_once=True,
                real_runtime_snapshot_checked=False)


if __name__=='__main__':
    print(json.dumps(dict(scope='CPU model only; native hooks not installed',
        growth=growth_scenario(),victim=victim_scenario(),known=known_scenario()),indent=2))

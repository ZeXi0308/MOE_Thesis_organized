#!/usr/bin/env python3
"""CPU component challenge; no GPU/native imports, no raw modifications."""
import argparse, hashlib, importlib, itertools, json, random, sys
from pathlib import Path
from types import SimpleNamespace as NS

B, CAP = 16, 4096
ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/private/tmp/moe-research-c-20260929-v2/refine-logs/expert_saturation/experiments/admission_capacity/20260929_c_baseline_delivery')
EXPECTED_SHA256 = {
    'C_NATIVE_RETIREMENT_ADMISSION_V1.py': '261b8f2697042f75130203019e8722c424cca3e71d82ac309c78f29ec1bc8eb6',
    'C_NATIVE_MAX_BOUND_ADMISSION.py': 'b6a601d1edc05804bee70d7f6ecdb2f00e1dd017b2280163df2db88250fb50ee',
    'C_NATIVE_RETIREMENT_FRESH_CELL_V1.py': '90c0065e8f004f98c636c1453caacaf678033144b96e9e74a708943e8b13703b',
}
for source_name, expected_hash in EXPECTED_SHA256.items():
    actual_hash = hashlib.sha256((SOURCE/source_name).read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise SystemExit(f'SOURCE_MISMATCH: {source_name}: expected {expected_hash}, got {actual_hash}')
# Do not let imports create __pycache__ in the inspected worktree.
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE))
M0 = importlib.import_module('C_NATIVE_RETIREMENT_ADMISSION_V1')


def request(p, o, rem, name='r'):
    return NS(request_id=name, num_prompt_tokens=p, num_output_tokens=o,
              max_tokens=o+rem, num_computed_tokens=p+o-1,
              num_tokens_with_spec=p+o, num_output_placeholders=0,
              num_in_flight_tokens=0, spec_token_ids=[], has_encoder_inputs=False,
              next_decode_eligible_step=0, status='WAITING')


def packing_peak(rows, exact=False):
    # Independently enumerate per-round capacity; no endpoint shortcut.
    horizon = max((r.max_tokens-r.num_output_tokens for r in rows), default=0)
    occupancy = [0]*(horizon+1)
    for r in rows:
        rem = r.max_tokens-r.num_output_tokens
        start = r.num_prompt_tokens+r.num_output_tokens-(1 if exact else 0)
        for t in range(rem+1):
            occupancy[t] += (start+t+B-1)//B
    return max(occupancy, default=0)


def prefix(peak, nrunning, waiting):
    answer=[]
    for r in waiting:
        cost=(r.num_prompt_tokens+r.max_tokens+B-1)//B
        if nrunning+len(answer)>=32 or peak+cost>CAP:
            break
        answer.append(r.request_id)
        peak+=cost
    return answer


def actual_m0_action(rows, waiting):
    # Execute actual adapter method with a minimal non-GPU native schedule stub.
    # Stub returns its allowed prefix; it does not validate native allocations.
    s=NS(running=list(rows), waiting=list(waiting), skipped_waiting=[],
         num_waiting_for_streaming_input=0, requests={r.request_id:r for r in rows},
         current_step=0, max_num_running_reqs=32,
         kv_cache_manager=NS(block_pool=NS(get_num_free_blocks=lambda:CAP)))
    g=M0.RetirementAdmission.__new__(M0.RetirementAdmission)
    g.scheduler=s;g.RequestStatus=NS(WAITING='WAITING')
    g.active={r.request_id:M0.blocks(r) for r in rows}
    g.protected=set();g.expected_outputs={};g.violations=[];g.events=[];g.steps=[]
    for name in ['schedule_calls','hold_calls','reserved_blocks_peak','preemptions',
                 'eligible_calls','incremental_admissions','envelope_peak_blocks',
                 'physical_blocks_peak','decision_seconds']:
        setattr(g,name,0)
    def native_stub():
        count=s.max_num_running_reqs-len(s.running)
        s.running.extend(s.waiting[:count])
        return NS(preempted_req_ids=[],num_scheduled_tokens={r.request_id:1 for r in rows})
    g.original_schedule=native_stub
    g.schedule()
    return [e['request_id'] for e in g.events if e['event']=='admit']


def generated():
    grid=[request(p,o,r) for p,o,r in itertools.product([1,14,15,16,17,31],[1,2,15],[1,2,16])]
    rng=random.Random(20261001)
    stats=dict(exhaustive_states=0,random_states=0,peak_mismatches=0,action_mismatches=0,
               m0_above_exact_states=0,max_extra_blocks=0)
    examples=[]
    def test(rows,which):
        rows=[request(r.num_prompt_tokens,r.num_output_tokens,r.max_tokens-r.num_output_tokens,f'r{i}') for i,r in enumerate(rows)]
        waiting=[request(p,0,m,f'w{i}') for i,(p,m) in enumerate([(3072,1024),(256,256),(1,1)])]
        a=M0.decode_envelope(rows);b=packing_peak(rows);e=packing_peak(rows,True)
        stats[which]+=1;stats['peak_mismatches']+=a!=b
        stats['m0_above_exact_states']+=a>e;stats['max_extra_blocks']=max(stats['max_extra_blocks'],a-e)
        aa=actual_m0_action(rows,waiting);bb=prefix(b,len(rows),waiting)
        stats['action_mismatches']+=aa!=bb
        if (a!=b or aa!=bb) and len(examples)<3:
            examples.append(dict(rows=[vars(r) for r in rows],m0=a,packing=b,m0_action=aa,packing_action=bb))
    for n in range(3):
        for rows in itertools.product(grid,repeat=n):test(rows,'exhaustive_states')
    while stats['random_states']<1000:
        n=rng.randrange(0,32)
        rows=[request(rng.randint(1,3072),rng.randint(1,1023),1) for _ in range(n)]
        for r in rows:r.max_tokens=1024
        if M0.decode_envelope(rows)<=CAP:test(rows,'random_states')
    stats['counterexamples']=examples
    return stats


def inspect_cell(path, trace):
    raw=json.loads((path/'raw.json').read_text())
    gate=json.loads((path/'retirement-envelope.json').read_text())
    reqs={r['request_id']:r for r in raw['requests']}
    source_by_internal={r['internal_request_id']:r['request_id'] for r in raw['requests']}
    first={}
    for i,s in enumerate(raw['scheduler_steps']):
        for r in s['scheduled']:first.setdefault(r['request_id'],i)
    queue=sorted(reqs.values(),key=lambda r:r['engine_add_return_s'])
    assert len(raw['scheduler_steps'])==len(gate['steps'])==len(raw['memory_trace'])
    stats=dict(total_calls=len(gate['steps']),all_decode_calls=0,peak_mismatches=0,
               prefix_count_mismatches=0,queue_alignment_failures=0,
               unrecoverable_resident_states=0,mixed_waiting_calls=0,
               mixed_capacity_fit_calls=0,mixed_token_room_calls=0,
               hol_capacity_blocked_calls=0,smaller_follower_fits_calls=0,
               m0_above_exact_calls=0,max_extra_blocks=0)
    mixed_heads=set();hol_heads=set();follower_heads=set()
    for i,(s,g,mem) in enumerate(zip(raw['scheduler_steps'],gate['steps'],raw['memory_trace'])):
        internal=mem['before']['running_ids']
        scheduled={r['request_id']:r for r in s['scheduled']}
        resident_ids=[source_by_internal[x] for x in internal]
        if not all(x in scheduled for x in resident_ids):
            stats['unrecoverable_resident_states']+=1;continue
        residents=[]
        for rid in resident_ids:
            row=scheduled[rid];q=reqs[rid]
            r=request(q['prompt_tokens'],row['output_tokens_before'],q['max_output_tokens']-row['output_tokens_before'],rid)
            r.num_computed_tokens=row['computed_before']
            residents.append(r)
        waiting_raw=[r for r in queue if r['engine_add_return_s']<=s['start_s'] and first[r['request_id']]>=i]
        if len(waiting_raw)!=g['waiting_before'] or len(residents)!=g['running_before']:
            stats['queue_alignment_failures']+=1;continue
        waiting=[request(r['prompt_tokens'],0,r['max_output_tokens'],r['request_id']) for r in waiting_raw]
        D=[r for r in residents if M0.decode_ready(r)]
        P=[r for r in residents if not M0.decode_ready(r)]
        record=dict(cell=path.parent.name,call=i+1,running=len(residents),waiting=len(waiting),all_decode=g['eligible_all_decode'])
        if g['eligible_all_decode']:
            assert not P
            stats['all_decode_calls']+=1
            peak=packing_peak(D);exact=packing_peak(D,True)
            got=prefix(peak,len(D),waiting)
            stats['peak_mismatches']+=peak!=g['conditional_decode_peak']
            stats['prefix_count_mismatches']+=len(got)!=g['selected']
            stats['m0_above_exact_calls']+=peak>exact
            stats['max_extra_blocks']=max(stats['max_extra_blocks'],peak-exact)
            record.update(packing_peak=peak,m0_peak=g['conditional_decode_peak'],packing_selected=got,m0_selected=g['selected'])
            if waiting and len(D)<32 and not got:
                stats['hol_capacity_blocked_calls']+=1;hol_heads.add(waiting[0].request_id)
                if any(peak+M0.blocks(r)<=CAP for r in waiting[1:]):
                    stats['smaller_follower_fits_calls']+=1;follower_heads.add(waiting[0].request_id)
        elif waiting:
            stats['mixed_waiting_calls']+=1
            mixed=packing_peak(D)+sum(M0.blocks(r) for r in P)
            fits=len(residents)<32 and mixed+M0.blocks(waiting[0])<=CAP
            record.update(mixed_peak=mixed,mixed_head_fits=fits,protected=g['protected_decode_requests'],pure_decode=len(D),prefill=len(P))
            if fits:
                stats['mixed_capacity_fit_calls']+=1;mixed_heads.add(waiting[0].request_id)
                stats['mixed_token_room_calls']+=len(D)+1<=1024
        trace.write(json.dumps(record,sort_keys=True)+'\n')
    stats.update(mixed_distinct_heads=len(mixed_heads),hol_distinct_heads=len(hol_heads),
                 smaller_follower_distinct_heads=len(follower_heads),
                 scope='Observed policy states; no action-conditioned replay; all-state properties only where recovered.')
    return stats


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    sources={n:hashlib.sha256((SOURCE/n).read_bytes()).hexdigest() for n in ['C_NATIVE_RETIREMENT_ADMISSION_V1.py','C_NATIVE_MAX_BOUND_ADMISSION.py','C_NATIVE_RETIREMENT_FRESH_CELL_V1.py']}
    result=dict(evidence='STRUCTURAL_CPU_COMPONENT_CHALLENGE',source_sha256=sources,generated=generated(),observed={})
    trace_path=args.output.with_name('state_comparison.jsonl')
    with trace_path.open('x') as trace:
        for label in ['retirement_1','retirement_2']:
            result['observed'][label]=inspect_cell(ROOT/'c-native-retirement-fresh-v1'/label/'native_retirement',trace)
    result['verdict']='SAME_DOMAIN_ACTION_EQUIVALENCE' if not any(result['generated'][k] for k in ['peak_mismatches','action_mismatches']) and not any(s[k] for s in result['observed'].values() for k in ['peak_mismatches','prefix_count_mismatches','queue_alignment_failures','unrecoverable_resident_states']) else 'REVIEW_MISMATCH_OR_COVERAGE'
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

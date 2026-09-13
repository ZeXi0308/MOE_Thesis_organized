"""CPU symbol validation for the ordinary20 full-stage baseline, without GPU execution."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import shlex
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-dir',type=Path,default=Path('refine-logs/expert_saturation/experiments/admission_capacity'))
parser.add_argument('--out',type=Path,required=True,help='New output file; never overwrite')
args=parser.parse_args()
if args.out.exists(): parser.error('--out must name a new file')
if not __debug__: raise RuntimeError('Assertions must be enabled')
source_dir=args.source_dir.resolve(); sys.dont_write_bytecode=True
sys.path.insert(0,str(source_dir))
from full_stage_plan import plan_full_stage
from analyze_layer_budget import LRU
from wisp_expert_groups import partition_experts
sources=[source_dir/n for n in ('full_stage_plan.py','analyze_layer_budget.py','wisp_expert_groups.py')]+[Path(__file__).resolve()]
def hashes(): return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
source_hashes=hashes(); rng=random.Random(20260913384)
states=[LRU(20) for _ in range(16)]; pool=[('poison',i) for i in range(384)]
counts=dict(calls=0,empty_entry=0,partial_entry=0,full_entry=0,empty_active=0,all64_active=0,
            hit_stage_d2d=0,h2d=0,writeback_d2d=0,max_stage_active=0,
            skip_hit_stage_failures=0,skip_writeback_failures=0)
sequence=[(layer,active) for layer in range(16)
          for active in ([],[layer],list(range(64)),list(range(64)),[],list(range(19)))]
sequence += [(rng.randrange(16),rng.sample(range(64),rng.randrange(65))) for _ in range(8000)]
for layer,active in sequence:
    state=states[layer]; snapshot=state.snapshot()
    if counts['calls']%2: snapshot['expert_to_slot']={int(e):s for e,s in snapshot['expert_to_slot'].items()}
    saved=copy.deepcopy(snapshot); entry=dict(state.mapping); active_set=set(active); base=layer*20
    # Arbitrary previous layer/stage contents cannot supply a current cache hit.
    pool[320:]=[('stale',counts['calls'],e) for e in range(64)]
    before=list(pool); plan=plan_full_stage(active,snapshot,layer)
    assert snapshot==saved
    assert plan['stage_slice']==[320,384] and plan['global_num_experts']==64 and plan['expert_map'] is None
    assert plan['copy_order']==['hit_stage','h2d','writeback'] and plan['kernel_after_copies']
    groups=partition_experts(active,entry,20); loaded=[]; evicted=[]
    for group in groups:
        missing,victims=state.ensure(group); loaded+=missing; evicted+=victims
    assert plan['final_state']==state.snapshot() and plan['canonical_groups']==groups
    assert (plan['canonical_group_count'],plan['canonical_miss'],plan['canonical_evict'])==(len(groups),len(loaded),len(evicted))
    assert plan['canonical_loaded_experts']==loaded and plan['canonical_evicted_experts']==evicted
    assert plan['entry_evicted_experts']==sorted(entry.keys()-state.mapping.keys())
    assert plan['entry_evict']==len(entry.keys()-state.mapping.keys())
    assert [c['expert'] for c in plan['hit_stage']]==sorted(active_set & entry.keys())
    assert [c['expert'] for c in plan['h2d']]==sorted(active_set-entry.keys())
    assert [c['expert'] for c in plan['writeback']]==sorted(state.mapping.keys()-entry.keys())
    assert all(state.mapping[e]==entry[e] for e in entry.keys() & state.mapping.keys())
    skipped_hit=list(pool)
    for c in plan['h2d']: skipped_hit[c['dst']]=(layer,c['expert'])
    counts['skip_hit_stage_failures'] += any(skipped_hit[320+e]!=(layer,e) for e in active)
    for c in plan['hit_stage']:
        assert pool[c['src']]==(layer,c['expert']) and c['dst']==320+c['expert']
        pool[c['dst']]=pool[c['src']]
    for c in plan['h2d']:
        assert c['dst']==320+c['expert']; pool[c['dst']]=(layer,c['expert'])
    assert pool[:320]==before[:320] and all(pool[320+e]==(layer,e) for e in active)
    stage_before_writeback=list(pool[320:])
    counts['skip_writeback_failures'] += any(pool[base+s]!=(layer,e) for e,s in state.mapping.items())
    for c in plan['writeback']:
        assert c['src']==320+c['expert'] and pool[c['src']]==(layer,c['expert'])
        pool[c['dst']]=pool[c['src']]
    assert pool[320:]==stage_before_writeback and all(pool[320+e]==(layer,e) for e in active)
    assert all(pool[base+s]==(layer,e) for e,s in state.mapping.items())
    assert pool[:base]==before[:base] and pool[base+20:320]==before[base+20:320]
    counts['calls']+=1; counts['empty_active']+=not active; counts['all64_active']+=len(active)==64
    counts['empty_entry' if not entry else 'full_entry' if len(entry)==20 else 'partial_entry']+=1
    counts['hit_stage_d2d']+=len(plan['hit_stage']); counts['h2d']+=len(plan['h2d'])
    counts['writeback_d2d']+=len(plan['writeback']); counts['max_stage_active']=max(counts['max_stage_active'],len(active_set))
# Sparse private slots and tied LRU clocks exercise ordinary slot selection.
for width in [0,1,7,19,20]:
    state=LRU(20)
    for slot,expert in zip(rng.sample(range(20),width),rng.sample(range(64),width)):
        state.slots[slot]=expert; state.mapping[expert]=slot; state.ticks[slot]=4
    state.clock=4
    for size in [0,1,20,21,44,64]: plan_full_stage(rng.sample(range(64),size),state.snapshot(),15)
invalid=[]
def reject(active,snapshot,layer):
    try: plan_full_stage(active,snapshot,layer)
    except (ValueError,KeyError,TypeError): invalid.append(True)
    else: raise AssertionError('Malformed input accepted')
blank=LRU(20).snapshot()
for active,layer in [([-1],0),([64],0),([True],0),([1.0],0),([0],16),([0],-1),([0],True)]: reject(active,blank,layer)
bad=copy.deepcopy(blank); bad['expert_to_slot']={'0':0,0:0}; reject([],bad,0)
bad=copy.deepcopy(blank); bad['slot_to_expert'][:2]=[0,0]; bad['expert_to_slot']={'0':1}; reject([],bad,0)
bad=copy.deepcopy(blank); bad['lru_tick'][0]=1; reject([],bad,0)
bad=copy.deepcopy(blank); bad['expert_map_device'][0]=0; reject([],bad,0)
assert counts['skip_hit_stage_failures']>0 and counts['skip_writeback_failures']>0 and counts['max_stage_active']==64
assert hashes()==source_hashes,'Source changed during validation'
command='PYTHONDONTWRITEBYTECODE=1 '+shlex.join([sys.executable,str(Path(__file__).resolve()),'--source-dir',str(source_dir),'--out','NEW_FILE'])
result=dict(validation='PASS',evidence_type='STRUCTURAL_CPU_SYMBOL_MODEL',seed=20260913384,**counts,
            sparse_tie_state_calls=30,invalid_inputs_rejected=len(invalid),source_sha256=source_hashes,
            reproducible_command=command,scope='Known full-stage baseline planner only. Private20 LRU exact on actual active IDs; no previous stage reuse, numerical GPU result, performance or novelty claim.')
with args.out.open('x') as f: json.dump(result,f,indent=2); f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','reproducible_command','scope')},indent=2))

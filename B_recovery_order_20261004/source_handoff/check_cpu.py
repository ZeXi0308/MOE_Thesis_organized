"""Bounded native Host-reference lifecycle checks; no GPU or vLLM import."""
import ast
from collections import OrderedDict, defaultdict
import ctypes
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS
import time

import source_handoff as policy

ROOT = Path(__file__).resolve().parents[1]/'native_sources'
PATHS = dict(OffloadingConnectorScheduler='offloading_scheduler_20261007.py',
    CPUOffloadingManager='cpu_offloading_manager_20261007.py', LRUCachePolicy='cpu_offloading_lru_20261007.py',
    OffloadingManager='kv_offload_base_20261007.py', CPULoadStoreSpec='cpu_common_20261007.py',
    BlockStatus='cpu_policy_base_20261007.py')
for name, filename in PATHS.items():
    assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest() == policy.PINS[name]
ENV = dict(ctypes=ctypes, OrderedDict=OrderedDict, override=lambda f:f,
    np=NS(array=lambda x, dtype:list(x), int64='int64'), MEDIUM_CPU='CPU',
    LookupResult=NS(HIT='hit', MISS='miss', HIT_PENDING='pending', RETRY='retry'),
    PrepareStoreOutput=NS, OffloadingEvent=NS, time=time,
    cdiv=lambda n,d:(n+d-1)//d, logger=NS(debug=lambda *a:None),
    _ConnectorMetricName=NS(LOOKUP_SYNC_DELAY='lookup'),
    TransferJob=NS, TransferJobStatus=NS, OffloadPolicy=NS(BLOCK_LEVEL='block'))


def native(path, name, methods=None, bases=()):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==name)
    if methods is not None:
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
        assert {n.name for n in node.body} == set(methods)
    node.bases = [ast.parse(b, mode='eval').body for b in bases]
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),ENV)
    return ENV[name]


Block = native(ROOT/PATHS['BlockStatus'],'BlockStatus',bases=['ctypes.Structure'])
native(ROOT/PATHS['OffloadingManager'],'BlockIDsLoadStoreSpec',['__init__'])
native(ROOT/PATHS['CPULoadStoreSpec'],'CPULoadStoreSpec',bases=['BlockIDsLoadStoreSpec'])
native(ROOT/PATHS['OffloadingManager'],'GPULoadStoreSpec',['__init__'],bases=['BlockIDsLoadStoreSpec'])
LRU = native(ROOT/PATHS['LRUCachePolicy'],'LRUCachePolicy')
ENV['_CACHE_POLICIES']={'lru':LRU}
Host = native(ROOT/PATHS['CPUOffloadingManager'],'CPUOffloadingManager',
    ['__init__','_get_num_free_blocks','_allocate_blocks','_free_block','_get_load_store_spec',
     'lookup','prepare_load','complete_load','touch','prepare_store','complete_store','reset_cache'])
Base = native(ROOT/PATHS['OffloadingManager'],'OffloadingManager',['on_request_finished'])
Host.on_request_finished=Base.on_request_finished
CS = native(ROOT/PATHS['OffloadingConnectorScheduler'],'OffloadingConnectorScheduler',
    ['get_num_new_matched_tokens','_maximal_prefix_lookup','_touch','update_state_after_alloc','request_finished','reset_cache'])


class GPU:
    def __init__(self):
        self.calls=[];self.result=None;self.error=None;self.free=10;self.max_model_len=4096
        self.block_pool=NS(get_num_free_blocks=lambda:self.free)
    def allocate_slots(self,*a,**k):
        self.calls.append((a,k))
        if self.error:raise self.error
        return self.result


def fixture(mode='early_pin', keys=None):
    host=Host(80)
    host.prepare_store(list(range(80)),NS());host.complete_store(list(range(80)),NS())
    req=NS(request_id='first',status=NS(name='PREEMPTED'),num_preemptions=1,
        num_tokens=2048,num_computed_tokens=0,num_in_flight_tokens=0,
        has_encoder_inputs=False,is_finished=lambda:False,skip_reading_prefix_cache=False)
    group=NS(offload_keys=list(range(64)) if keys is None else keys,block_ids=[])
    state=NS(req=req,req_context=NS(id='first'),transfer_jobs=set(),group_states=[group],
        num_locally_computed_tokens=0,deferred_lookup_start_time=None,
        update_offload_keys=lambda:None,update_num_hit_chunks=lambda tokens:None,
        offloading_context=NS(policy='block'))
    cs=CS();cs.manager=host
    cs.config=NS(kv_group_configs=[NS(tokens_per_block=16,tokens_per_chunk=16,
        sliding_window_size_in_chunks=None)],blocks_per_chunk=1,num_workers=1)
    cs._req_status={'first':state};cs._jobs={};cs._chunks_being_loaded=None
    cs._current_batch_allocated_block_ids=set();cs._current_batch_load_jobs={}
    cs._current_batch_jobs_to_flush=set();cs._block_id_to_pending_jobs={};cs._job_counter=1
    cs._generate_job_id=lambda:1
    cs._connector_stats=NS(observe_histogram=lambda *a:None)
    cs._events_tracker=NS(reset=lambda:None)
    cs._maybe_observe_lookup_async_delay=lambda state:None
    cs._lookup=lambda state:16*cs._maximal_prefix_lookup(state.group_states[0].offload_keys,state.req_context)
    gpu=GPU();single=NS(block_size=16,req_to_blocks=defaultdict(list))
    data,uninstall=policy._install(gpu,single,cs,mode)
    return host,req,state,cs,gpu,single,data,uninstall


KW=dict(num_external_computed_tokens=1024,delay_cache_blocks=True,full_sequence_must_fit=True)
def select(f):
    host,req,state,cs,gpu,single,data,uninstall=f
    assert cs.get_num_new_matched_tokens(req,0)==(1024,True)
    assert gpu.allocate_slots(req,0,**KW) is None
    assert len(gpu.calls)==1 and gpu.calls[0][0]==(req,0)
    assert gpu.calls[0][1]['num_external_computed_tokens']==1024
    return f


# Native lookup/touch can lose a prefix under a fixed Host budget; early refs survive the same stores.
for mode in ('native','early_pin'):
    f=select(fixture(mode));host,req,state,cs,gpu,single,data,uninstall=f
    assert data['selected_request']=='first'
    assert all(host._policy.get(k).ref_cnt==(mode=='early_pin') for k in range(64))
    for start in range(80,160,16):
        stored=host.prepare_store(list(range(start,start+16)),NS());assert stored is not None
        host.complete_store(stored.keys_to_store,NS())
    matched=cs.get_num_new_matched_tokens(req,0)
    assert matched==((1024,True) if mode=='early_pin' else (0,False))
    assert host._num_allocated_blocks==80 and host._get_num_free_blocks()==0
    gpu.allocate_slots(req,0,num_external_computed_tokens=matched[0],delay_cache_blocks=True,full_sequence_must_fit=True)
    rows=[e for e in data['events'] if e['kind']=='allocation']
    assert rows and rows[-1]['before'][0]['cached']==(mode=='early_pin')
    if mode=='early_pin':
        # Pressure exceeding remaining evictable blocks fails through the original store path.
        assert host.prepare_store(list(range(200,217)),NS()) is None
    uninstall();assert data['remaining_early_refs']==0

# Native LOAD gets a second reference before early release; original ACK releases only the remaining one.
f=select(fixture());host,req,state,cs,gpu,single,data,uninstall=f
blocks=NS(blocks=([NS(block_id=i+1,is_null=False,block_hash=None) for i in range(64)],))
sentinel=NS();gpu.result=sentinel
assert gpu.allocate_slots(req,0,**KW) is sentinel
assert cs.update_state_after_alloc(req,blocks,1024) is None
handoff=next(e for e in data['events'] if e['kind']=='native_handoff')
assert all(r['ref_cnt']==2 for r in handoff['before_early_release'])
assert all(host._policy.get(k).ref_cnt==1 for k in range(64))
assert state.transfer_jobs=={1} and cs._jobs[1].keys==set(range(64))
assert cs._current_batch_load_jobs[1].dst_spec.block_ids==list(range(1,65))
host.complete_load(cs._jobs[1].keys,state.req_context)
assert all(host._policy.get(k).ref_cnt==0 for k in range(64))
assert host._num_evictable_cache_blocks==80
# One request per run: a later qualifying request cannot acquire refs.
other=NS(**vars(req));other.request_id='second';gpu.result=None
gpu.allocate_slots(other,0,**KW)
assert data['selected_request']=='first' and sum(e['kind']=='early_pin' for e in data['events'])==1
uninstall();uninstall();assert 'allocate_slots' not in vars(gpu)

# Early-only references leave before native finish/abort/reset; uninstall also releases once.
for end in ('finish','reset','uninstall'):
    f=select(fixture());host,req,state,cs,gpu,single,data,uninstall=f
    if end=='finish':
        req.is_finished=lambda:True
        assert cs.request_finished(req)==(False,None)
    elif end=='reset':cs.reset_cache()
    else:uninstall()
    assert all(b.ref_cnt==0 for b in host._policy.blocks.values())
    assert not host._policy.blocks if end=='reset' else host._num_evictable_cache_blocks==80
    uninstall();assert data['remaining_early_refs']==0

# The first invalid ready-set is logged and skipped; never pin duplicate/missing/not-ready keys.
for variant in ('duplicate','missing','pending'):
    keys=list(range(64));keys[-1]=0 if variant=='duplicate' else keys[-1]
    f=fixture(keys=keys);host,req,state,cs,gpu,single,data,uninstall=f
    if variant=='missing':host._policy.remove(63);host._num_evictable_cache_blocks-=1
    if variant=='pending':host._policy.mark_non_evictable(63);host._num_evictable_cache_blocks-=1;host._policy.get(63).ref_cnt=-1
    gpu.allocate_slots(req,0,**KW)
    assert not any(e['kind']=='early_pin' for e in data['events'])
    assert data['events'][0]['valid'] is False
    assert all(b.ref_cnt<=0 for b in host._policy.blocks.values())
    uninstall()

# Native descriptor construction failure occurs after all increments: owned refs must be rolled back.
f=fixture();host,req,state,cs,gpu,single,data,uninstall=f
original_array=ENV['np'].array
ENV['np'].array=lambda *a,**k:(_ for _ in ()).throw(MemoryError('descriptor'))
try:
    gpu.allocate_slots(req,0,**KW)
    raise AssertionError('prepare error hidden')
except MemoryError:pass
finally:ENV['np'].array=original_array
assert all(host._policy.get(k).ref_cnt==0 for k in range(64)) and host._num_evictable_cache_blocks==80
uninstall()

# A native prepare failure part-way through the keys is rolled back only for acquired refs.
f=fixture();host,req,state,cs,gpu,single,data,uninstall=f
original_prepare=host.prepare_load
def partial(keys,ctx):original_prepare(keys[:3],ctx);raise RuntimeError('partial prepare')
host.prepare_load=partial
try:gpu.allocate_slots(req,0,**KW);raise AssertionError('partial failure hidden')
except RuntimeError as error:assert str(error)=='partial prepare'
assert all(host._policy.get(k).ref_cnt==0 for k in range(64))
uninstall()

# Native complete_load can also fail part-way: a retry must not decrement completed keys again.
f=select(fixture());host,req,state,cs,gpu,single,data,uninstall=f
original_complete=host.complete_load
def partial_complete(keys,ctx):original_complete(keys[:3],ctx);raise RuntimeError('partial complete')
host.complete_load=partial_complete
try:uninstall();raise AssertionError('release failure hidden')
except RuntimeError as error:assert str(error)=='partial complete'
assert all(host._policy.get(k).ref_cnt==0 for k in range(3))
assert all(host._policy.get(k).ref_cnt==1 for k in range(3,64))
host.complete_load=original_complete;uninstall()
assert all(host._policy.get(k).ref_cnt==0 for k in range(64))

# Allocation exceptions propagate and release early refs; normal/short/success calls remain untouched.
f=select(fixture());host,req,state,cs,gpu,single,data,uninstall=f;gpu.error=RuntimeError('allocate')
try:gpu.allocate_slots(req,0,**KW);raise AssertionError('allocation error hidden')
except RuntimeError:pass
assert all(host._policy.get(k).ref_cnt==0 for k in range(64));uninstall()
for variant in ('new','short','success','computed','held','no_fullfit','no_gap'):
    f=fixture();host,req,state,cs,gpu,single,data,uninstall=f;kwargs=dict(KW)
    if variant=='new':req.num_preemptions=0
    elif variant=='short':kwargs['num_external_computed_tokens']=1008
    elif variant=='success':gpu.result=NS()
    elif variant=='computed':req.num_computed_tokens=16
    elif variant=='held':single.req_to_blocks['first']=[NS()]
    elif variant=='no_fullfit':kwargs['full_sequence_must_fit']=False
    else:gpu.free=128
    gpu.allocate_slots(req,0,**kwargs)
    assert data['selected_request'] is None and len(gpu.calls)==1
    uninstall()
print('PASS: native Host lookup/eviction/read refs/LOAD handoff/finish/reset AST; one selection; partial prepare rollback; CPU only, GPU_UNRUN')

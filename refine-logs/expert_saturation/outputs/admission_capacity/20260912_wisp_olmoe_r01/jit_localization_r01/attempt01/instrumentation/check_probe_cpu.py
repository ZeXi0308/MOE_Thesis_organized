"""CPU-only regression of real wrappers: cache classification, attribution, failure, restore."""
import json, sys
from types import SimpleNamespace as N
from run_jit_probe import Probe, plain
prior = []
listener = lambda **kw: prior.append(kw['cache_hit'])
triton = N(knobs=N(compilation=N(listener=listener)))
class Kernel:
    def __init__(self, src): self.src, self.hash, self.module = src, 'key', None
    def _init_handles(self): self.module = 1
def compile_original(src):
    if src.name == 'failure': raise ValueError('deliberate')
    triton.knobs.compilation.listener(src=src, cache_hit=src.hit, metadata={'hash':'key'},
        times=N(ir_initialization=2, lowering_stages=[('cubin',3)], store_results=1))
    return Kernel(src)
compiler = N(compile=compile_original, CompiledKernel=Kernel)
triton.compile = compile_original
sys.modules['triton.compiler'] = N(compile=compile_original)
layer=N(layer_name='layer0'); runtime=N(context={'phase':'measurement','step_id':3},next_call_id=7,
    records=[],layers={id(layer):{'state':N(cap_experts=24)}})
src=N(name='kernel',fn=N(arg_names=['M']),constants={(0,):96},signature={'M':'constexpr'},hit=False)
def original_apply(*args):
    runtime.records.append({'call_id':runtime.next_call_id,'groups':[{}]}); runtime.next_call_id+=1
    kernel=triton.compile(src); kernel._init_handles(); return 19
groups=N(_apply=original_apply); probe=Probe(N(_runtime=runtime)); probe.install(triton,compiler,groups)
for hit in (False,True):
    src.hit=hit; assert groups._apply(runtime,None,layer,N(shape=(96,2048)),None,N(shape=(96,8)))==19
src.name='failure'
try: groups._apply(runtime,None,layer,N(shape=(96,2048)),None,N(shape=(96,8)))
except ValueError: pass
else: raise AssertionError('failure swallowed')
for event in probe.events:
    assert event['end_perf_ns']>=event['start_perf_ns'] and event['context']['call_id'] in (7,8,9)
    if event['kind']!='pager_apply':
        parent=next(p for p in probe.events if p['event_id']==event['context']['apply_event_id'])
        assert parent['start_perf_ns']<=event['start_perf_ns']<=event['end_perf_ns']<=parent['end_perf_ns']
compiles=[e for e in probe.events if e['kind']=='compiler_call']
assert [e['context']['call_id'] for e in compiles]==[7,8,9]
assert [e['result_kind'] for e in compiles]==['compiler_pipeline','disk_cache_load','unclassified']
assert compiles[-1]['status']=='failed' and compiles[-1]['cache_hit'] is None and prior==[False,True]
probe.close(); assert groups._apply is original_apply and compiler.compile is compile_original
assert triton.knobs.compilation.listener is listener and probe.local.apply is None
json.dumps(plain(float('inf')),allow_nan=False)
print('PASS: compile/cache classification; nested call attribution; failure preserved; hooks restored')

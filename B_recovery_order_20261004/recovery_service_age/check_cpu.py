"""Bounded ordering and real compact-measurement receipt timing checks; no GPU."""
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import service_age as policy

B=Path(__file__).resolve().parents[1]
path=B/'recovery_repeat/check_cpu.py';source=path.read_text()
assert hashlib.sha256(source.encode()).hexdigest()=='a914433d0f977b57f2ac36226dce7d2daeaffa85d514086e4ad393c706ca6bee'
sys.path.insert(0,str(path.parent));defs={'__file__':str(path)}
cut='\nfor mode, limit in repeat_fit.LIMITS.items():\n'
assert source.count(cut)==1
exec(compile(source.split(cut)[0],str(path),'exec'),defs)
fixture=defs['frozen']


def setup(mode,receipts):
    s,_,manager,single,cs=fixture['fixture']();head,anchor=s.waiting;head.num_tokens=160
    other=fixture['request']('other',3)
    cs._req_status['other']=NS(req=other,transfer_jobs=set(),group_states=cs._req_status['anchor'].group_states)
    s.waiting.append(other)
    data,undo=policy._attach(s,manager,single,cs,mode,fixture['fragment'],receipts)
    return s,head,anchor,other,data,undo


for mode in policy.LIMITS:
    receipts={'anchor':5.,'other':1.}
    s,head,anchor,other,data,undo=setup(mode,receipts)
    selected=anchor if mode=='age8' else other
    assert s.schedule(8,[])==selected.request_id
    row=data['events'][-1]
    assert row['age8_suggestion']=='anchor' and row['stall8_suggestion']=='other'
    assert row['age8_would_execute'] and row['stall8_would_execute']
    assert not row['receipt_fallback_reasons'] and data['action_count']==1
    assert row['stall8_fallback_reason'] is None
    s.waiting[:]=[head,anchor,other]
    assert s.schedule(8,[])=='head' and data['action_count']==1
    assert data['events'][-1]['action_skip_reasons']==['EPISODE_ALREADY_BYPASSED']
    if mode=='stall8':assert data['events'][-1]['age8_would_execute'] and not data['events'][-1]['stall8_would_execute']
    # Per-episode and global8 guards remain after selection, without fall-through.
    for episode in range(2,10):
        selected.num_preemptions=episode;s.waiting[:]=[head,anchor,other]
        assert s.schedule(8,[])==(selected.request_id if episode<=8 else 'head')
    assert data['action_count']==len(data['bypassed_episodes'])==8
    assert data['events'][-1]['action_skip_reasons']==['BUDGET_EXHAUSTED']
    undo();undo();assert data['status']=='UNINSTALLED'

for missing,reason in ((None,'MISSING_RECEIPT'),(float('nan'),'NONFINITE_RECEIPT'),(1e300,'NOT_STRICTLY_PAST_RECEIPT')):
    for mode in policy.LIMITS:
        receipts={'anchor':5.}
        if missing is not None:receipts['other']=missing
        s,head,anchor,other,data,undo=setup(mode,receipts)
        assert s.schedule(8,[])=='anchor'
        row=data['events'][-1]
        assert row['receipt_fallback_reasons']==[reason]
        assert row['stall8_fallback_reason']=='UNKNOWN_CANDIDATE_RECEIPT'
        assert row['age8_suggestion']==row['stall8_suggestion']=='anchor'
        undo()

# Equal last receipts retain exact arrival/id tie behavior; mutation error does
# not mark a successful bypass. Neither path mutates receipt ownership.
s,head,anchor,other,data,undo=setup('stall8',{'anchor':1.,'other':1.})
s.waiting.fail_mutation=True
try:s.schedule(8,[])
except RuntimeError as error:assert str(error)=='native mutation exception'
else:raise AssertionError('Native mutation error swallowed')
assert data['action_count']==0 and data['events'][-1]['candidate_head']=='anchor';undo()

# Reuse deterministic fake vLLM/clock definitions only, not their old test suite.
path=B/'output_event_compact/check_cpu.py';source=path.read_text()
assert hashlib.sha256(source.encode()).hexdigest()=='528b03ce1f7dc64c70d393fd11bc584f01eb4405683ac25a1813d8bd40803ac1'
sys.path.insert(0,str(path.parent));capture_defs={'__file__':str(path)}
assert source.count("\nfake_vllm = ModuleType('vllm')")==1
exec(compile(source.split("\nfake_vllm = ModuleType('vllm')")[0],str(path),'exec'),capture_defs)
compact=capture_defs['compact'];Engine=capture_defs['Engine']
fake=ModuleType('vllm');sampling=ModuleType('vllm.sampling_params')
fake.SamplingParams=lambda **kw:NS(**kw);sampling.RequestOutputKind=NS(CUMULATIVE='cumulative')
saved={name:sys.modules.get(name) for name in ('vllm','vllm.sampling_params')}
sys.modules.update({'vllm':fake,'vllm.sampling_params':sampling})
scripts=[
    [[(0,[10],False,None)],[(0,[10],False,None)],[(0,[10,11,12],True,'length')]],
    [[(0,[10],False,None)],[(0,[99,11],False,None)]],
    [[(0,[10],False,None)],[(0,[10,11,12,13],True,'length')]],
    [[(0,[10],False,None)],RuntimeError('native engine failure')],
]
try:
    for script in scripts:
        baseline,clock0,_,_=capture_defs['run']('compact',script)
        receipts={};calls=[];fn=policy.make_capture(compact,receipts)
        class Observe(Engine):
            def step(self):
                calls.append(dict(receipts))
                return super().step()
        capture_defs['Engine']=Observe
        old=compact._CAPTURES['compact'];compact._CAPTURES['compact']=fn
        try:raw,clock1,engine,_=capture_defs['run']('compact',script)
        finally:compact._CAPTURES['compact']=old;capture_defs['Engine']=Engine
        assert raw==baseline and clock0.calls==clock1.calls
        assert calls[0]=={}
        valid=[e for e in raw['output_events'] if e['prefix_valid'] and e['chunk_size']>0]
        expected=raw['measurement_origin_perf_counter_s']+valid[-1]['received_s']
        assert receipts=={'internal:cpu/r0':expected}
        for call,observed in enumerate(calls):
            past=[e for e in valid if e['engine_call_index']<call]
            assert observed==({'internal:cpu/r0':raw['measurement_origin_perf_counter_s']+past[-1]['received_s']} if past else {})
finally:
    for name,module in saved.items():
        if module is None:sys.modules.pop(name,None)
        else:sys.modules[name]=module

# Compile the complete actual child adaptation and stop before native execution.
path=Path(__file__).with_name('run_cell.py')
spec=importlib.util.spec_from_file_location('checked_service_child',path)
child=importlib.util.module_from_spec(spec);spec.loader.exec_module(child)
class Captured(Exception):pass
def capture(text):
    compile(text,'<service-age-complete-child>','exec')
    assert text.count("install_recovery_service_age(scheduler, os.environ['B_RECOVERY_SERVICE_AGE'], last_receipts=last_receipts)")==1
    assert 'from repeat_unique import' not in text and 'from repeat_fit import' not in text
    assert "measure_episode = make_capture(compact_capture, last_receipts)" in text
    assert text.count('install_gc_probe()')==1
    assert 'ignore_eos=True, min_tokens=0,' in text and 'output_tokens=16, output_tokens_by_request={}' in text
    assert text.index('uninstall_gc())')<text.index('uninstall_fit())')<text.index('uninstall_source())')
    raise Captured
import os
old_env=dict(os.environ)
try:
    for mode in policy.LIMITS:
        os.environ['B_RECOVERY_SERVICE_AGE']=mode
        try:child.main(capture)
        except Captured:pass
        else:raise AssertionError('Missing pre-execution capture')
finally:os.environ.clear();os.environ.update(old_env)
print('PASS: legal age/stall/ties/fallback, select-before-episode and budget8; successful counts/errors; real compact validated positive receipts only, previous-call visibility, equal clocks/raw; full fixed1024/GC child composition. CPU only.')

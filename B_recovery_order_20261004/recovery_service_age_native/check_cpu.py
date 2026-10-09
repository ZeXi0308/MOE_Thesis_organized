"""Only new native-guard and frozen-stall equivalence risks; existing CPU fixture."""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import native_service_age as policy

B=Path(__file__).resolve().parents[1]
path=B/'recovery_service_age/check_cpu.py';source=path.read_text()
assert hashlib.sha256(source.encode()).hexdigest()=='cd4fc5a6e58b2371336e52ecf7dd71c0c8f20b9fbc4f58d3c4702d75453cd17f'
cut='\nfor mode in policy.LIMITS:\n';assert source.count(cut)==1
sys.path.insert(0,str(path.parent));defs={'__file__':str(path)}
exec(compile(source.split(cut)[0],str(path),'exec'),defs)
fixture=defs['fixture']


def setup(module,mode,receipts):
    s,_,manager,single,cs=fixture['fixture']();head,anchor=s.waiting;head.num_tokens=160
    other=fixture['request']('other',3);s.waiting.append(other)
    cs._req_status['other']=NS(req=other,transfer_jobs=set(),group_states=cs._req_status['anchor'].group_states)
    wrapper=s.schedule
    slot=policy.FROZEN.REPEAT.FROZEN.HELPERS._native_slot(wrapper,s);native=slot.cell_contents
    data,undo=module._attach(s,manager,single,cs,mode,fixture['fragment'],receipts)
    return s,head,anchor,other,data,undo,wrapper,slot,native


# Fixed host clock permits complete per-event equality, not just equal targets.
clock=NS(perf_counter=lambda:100.)
namespaces=[policy._decision.__globals__,policy.FROZEN._decision.__globals__,
            policy.FROZEN.REPEAT.FROZEN._decision.__globals__]
saved=[namespace['time'] for namespace in namespaces]
try:
    for namespace in namespaces:namespace['time']=clock
    receipts=[{'anchor':5.,'other':1.},{'anchor':5.,'other':1.}]
    arms=[setup(policy.FROZEN,'stall8',receipts[0]),setup(policy,'stall8',receipts[1])]
    for episode in (1,1,2,3,4,5,6,7,8,9):
        results=[]
        for s,head,anchor,other,data,undo,*_ in arms:
            other.num_preemptions=episode;s.waiting[:]=[head,anchor,other]
            results.append(s.schedule(8,[]))
        assert results[0]==results[1]
        assert arms[0][4]['events']==arms[1][4]['events']
        assert arms[0][4]['bypassed_episodes']==arms[1][4]['bypassed_episodes']
    for receipt,arm in zip(receipts,arms):
        del receipt['other'];s,head,anchor,other,*_=arm;s.waiting[:]=[head,anchor,other];s.schedule(8,[])
    assert arms[0][4]['events']==arms[1][4]['events']
    assert arms[0][4]['action_count']==arms[1][4]['action_count']==8
    for arm in arms:arm[5]()

    s,head,anchor,other,data,undo,wrapper,slot,native=setup(policy,'native',{'anchor':5.,'other':1.})
    for episode in range(1,11):
        other.num_preemptions=episode;s.waiting[:]=[head,anchor,other]
        before=list(s.waiting)
        assert s.schedule(8,[])=='head' and list(s.waiting)==before
        event=data['events'][-1]
        assert event['candidate_head']=='other' and event['final_head']=='head'
        assert event['action']=='SHADOW_ONLY' and not event['queue_changed']
        assert event['action_skip_reasons']==['NATIVE_OBSERVATION_ONLY']
        assert event['age8_would_execute'] and event['stall8_would_execute']
        assert data['action_count']==0 and data['bypassed_episodes']==[]
    assert not getattr(s.waiting,'remove_calls',0) and not getattr(s.waiting,'prepend_calls',0)
    s.waiting.fail=True
    try:s.schedule(8,[])
    except ValueError as error:assert str(error)=='native queue exception'
    else:raise AssertionError('Native queue exception swallowed')
    assert data['action_count']==0
    undo();undo();assert s.schedule is wrapper and slot.cell_contents is native
finally:
    for namespace,previous in zip(namespaces,saved):namespace['time']=previous

# No new measurement code: direct identity reuse of the frozen receipt capture.
assert policy.make_capture is policy.FROZEN.make_capture
path=B/'output_event_compact/compact_capture.py'
spec=importlib.util.spec_from_file_location('native_compact_check',path)
compact=importlib.util.module_from_spec(spec);spec.loader.exec_module(compact)
assert policy.make_capture(compact,{}).service_age_capture_sha256=='0d71af580baa93c26d16516202e3321a2168654550e3fc7a6b668e4fcf5f3f21'
path=Path(__file__).with_name('run_cell.py')
spec=importlib.util.spec_from_file_location('native_service_child_check',path)
child=importlib.util.module_from_spec(spec);spec.loader.exec_module(child)
class Captured(Exception):pass
def capture(text):
    compile(text,'<native-service-age-complete-child>','exec')
    assert text.count('from native_service_age import install as install_recovery_service_age')==1
    assert text.count("install_recovery_service_age(scheduler, os.environ['B_RECOVERY_SERVICE_AGE_NATIVE'], last_receipts=last_receipts)")==1
    assert 'from service_age import' not in text and 'from repeat_fit import' not in text
    assert text.count('measure_episode = make_capture(compact_capture, last_receipts)')==1
    assert "out/'recovery-service-age-native.json'" in text
    assert "config['B_recovery_service_age_native']" in text
    raise Captured
old_env=dict(os.environ)
try:
    for mode in policy.LIMITS:
        os.environ['B_RECOVERY_SERVICE_AGE_NATIVE']=mode
        try:child.main(capture)
        except Captured:pass
        else:raise AssertionError('Did not stop before native execution')
finally:os.environ.clear();os.environ.update(old_env)
print('PASS: native observes without mutation/counts, explicit guard, original exception/uninstall; stall8 complete events equal frozen; receipt source unchanged; native/stall child composition. CPU only.')

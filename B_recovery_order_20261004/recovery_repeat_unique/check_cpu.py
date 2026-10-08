"""Only per-ID selection/budget risks; execute the existing native CPU fragment."""
import hashlib
from pathlib import Path
import sys

import repeat_unique as policy

B = Path(__file__).resolve().parents[1]
fixture_path = B/'recovery_repeat/check_cpu.py'
source = fixture_path.read_text()
assert hashlib.sha256(source.encode()).hexdigest() == 'a914433d0f977b57f2ac36226dce7d2daeaffa85d514086e4ad393c706ca6bee'
cut = '\nfor mode, limit in repeat_fit.LIMITS.items():\n'
assert source.count(cut) == 1
sys.path.insert(0, str(fixture_path.parent))
definitions = {'__file__': str(fixture_path)}
exec(compile(source.split(cut)[0], str(fixture_path), 'exec'), definitions)
frozen = definitions['frozen']


def setup(mode):
    s, _, manager, single, cs = frozen['fixture']()
    head, target = s.waiting
    head.num_tokens = 160
    wrapper = s.schedule
    slot = policy.REPEAT.FROZEN.HELPERS._native_slot(wrapper, s)
    native = slot.cell_contents
    data, undo = policy._attach(s, manager, single, cs, mode, frozen['fragment'])
    return s, head, target, data, undo, wrapper, slot, native, manager, single, cs


def add(s, cs, rid, arrival=3):
    request = frozen['request'](rid, arrival)
    original = cs._req_status['anchor']
    cs._req_status[rid] = definitions['NS'](req=request, transfer_jobs=set(), group_states=original.group_states)
    return request


# Repeat8 retains the exact old decisions, including skipping an already-used
# episode instead of falling through to another request. Unique8 changes IDs only.
for mode in policy.LIMITS:
    s, head, target, data, undo, wrapper, slot, native, manager, single, cs = setup(mode)
    extra = add(s, cs, 'second')
    old_data, old_call = policy.REPEAT._decision(s, manager, single, cs, 'repeat8')
    for episode in (1, 1, 2):
        target.num_preemptions = episode
        s.waiting[:] = [head, target, extra]
        if mode == 'repeat8':
            old_call(s, 8, [], s.waiting)
            expected_order = [r.request_id for r in s.waiting]
            s.waiting[:] = [head, target, extra]
        result = s.schedule(8, [])
        event = data['events'][-1]
        assert event['repeat8_suggestion'] == 'anchor'
        assert event['repeat8_suggestion_num_preemptions'] == episode
        assert event['repeat8_would_execute'] == [True,False,True][len(data['events'])-1]
        assert [r['request'] for r in event['candidates'] if r['eligible']] == ['anchor', 'second']
        if mode == 'repeat8':
            assert [r.request_id for r in s.waiting] == expected_order
            assert event['action'] == old_data['events'][-1]['action']
            assert event['action_count_after'] == old_data['action_count']
        else:
            assert result == ['anchor', 'second', 'head'][len(data['events'])-1]
    assert data['action_count'] == 2
    assert data['used_request_ids'] == (['anchor'] if mode == 'repeat8' else ['anchor','second'])
    if mode == 'unique8':
        assert event['unique8_suggestion'] is None and event['candidate_head'] is None
        assert event['action_skip_reasons'] == ['NO_UNUSED_REQUEST_IN_LEGAL_SET']
    undo();undo()
    assert s.schedule is wrapper and slot.cell_contents is native and data['status']=='UNINSTALLED'

# All successful mutations, and only those mutations, consume the same budget8.
for mode in policy.LIMITS:
    s, head, target, data, undo, _, _, _, _, _, cs = setup(mode)
    for i in range(10):
        request = add(s, cs, 'rid'+str(i), i)
        s.waiting[:] = [head, request]
        assert s.schedule(8, []) == (request.request_id if i < 8 else 'head')
    assert data['action_count'] == len(data['used_request_ids']) == len(data['bypassed_episodes']) == 8
    assert data['shadow_count'] == 10 and data['events'][-1]['action_skip_reasons'] == ['BUDGET_EXHAUSTED']
    assert data['events'][-1]['repeat8_would_execute'] is False
    assert s.waiting.remove_calls == s.waiting.prepend_calls == 8
    undo()

# Once the first legal ID was used, selection cannot jump either frozen boundary
# to find an unused ID, nor admit a physically non-fitting member of the prefix.
for barrier in ('priority','new_waiter','capacity'):
    s, head, target, data, undo, _, _, _, _, _, cs = setup('unique8')
    assert s.schedule(8, []) == 'anchor'
    extra = add(s, cs, 'second')
    if barrier == 'priority': extra.priority=1
    elif barrier == 'new_waiter': extra.status.name='WAITING'
    else: extra.num_tokens=160
    s.waiting[:] = [head,target,extra]
    assert s.schedule(8, []) == 'head' and data['action_count']==1
    assert data['events'][-1]['unique8_suggestion'] is None
    assert [r.request_id for r in s.waiting]==['head','anchor','second']
    undo()

for failure in ('mutation','peek'):
    s, head, target, data, undo, _, _, _, _, _, _ = setup('unique8')
    if failure=='mutation':s.waiting.fail_mutation=True
    else:s.waiting.fail=True
    try:s.schedule(8, [])
    except (RuntimeError,ValueError) as error:
        assert str(error)==('native mutation exception' if failure=='mutation' else 'native queue exception')
    else:raise AssertionError('Original exception swallowed')
    assert data['action_count']==(0 if failure=='mutation' else 1)
    assert data['used_request_ids']==([] if failure=='mutation' else ['anchor'])
    undo()

print('PASS: repeat8 preserved; unique IDs versus episodes; next unused legal choice; total budget8; no priority/new-WAITING/fit crossing; native errors and uninstall retained. CPU only.')

"""Bounded new-budget/episode tests using the existing real native decision fixture."""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import repeat_fit

B = Path(__file__).resolve().parents[1]
fixture_path = B/'recovery_queue/check_cpu.py'
fixture_source = fixture_path.read_text()
assert hashlib.sha256(fixture_path.read_bytes()).hexdigest() == '702e80971cbff2c193980dd29d82aa48d38584634b73bd96db681ddcb6000506'
# Load fixture definitions, not the previous observer's test cases.
cut = '\ns,source,manager,single,cs = fixture()\n'
assert fixture_source.count(cut) == 1
sys.path.insert(0, str(fixture_path.parent))
frozen = {'__file__': str(fixture_path)}
exec(compile(fixture_source.split(cut)[0], str(fixture_path), 'exec'), frozen)


def remove_request(self, request):
    self.remove_calls = getattr(self, 'remove_calls', 0)+1
    self.remove(request)


def prepend_request(self, request):
    self.prepend_calls = getattr(self, 'prepend_calls', 0)+1
    if getattr(self, 'fail_mutation', False):
        raise RuntimeError('native mutation exception')
    self.insert(0, request)


frozen['Queue'].remove_request = remove_request
frozen['Queue'].prepend_request = prepend_request


def setup(mode):
    s, _, manager, single, cs = frozen['fixture']()
    head, target = s.waiting
    head.num_tokens = 160
    wrapper = s.schedule
    slot = repeat_fit.FROZEN.HELPERS._native_slot(wrapper, s)
    native = slot.cell_contents
    data, undo = repeat_fit._attach(s, manager, single, cs, mode, frozen['fragment'])
    return s, head, target, data, undo, wrapper, slot, native


for mode, limit in repeat_fit.LIMITS.items():
    s, head, target, data, undo, wrapper, slot, native = setup(mode)
    for episode in range(1, 11):
        target.num_preemptions = episode
        s.waiting[:] = [head, target]
        assert s.schedule(8, []) == ('anchor' if episode <= limit else 'head')
        event = data['events'][-1]
        assert event['candidate_num_preemptions'] == episode
        assert event['queue_changed'] == (episode <= limit)
        assert event['action_count_after'] == min(episode, limit)
        if episode > limit:
            assert event['action_skip_reasons'] == ['BUDGET_EXHAUSTED']
        # The same episode cannot consume a second action; both modes still observe.
        s.waiting[:] = [head, target]
        assert s.schedule(8, []) == 'head'
        expected = (['EPISODE_ALREADY_BYPASSED'] if episode <= limit else [])
        if episode >= limit:
            expected += ['BUDGET_EXHAUSTED']
        assert data['events'][-1]['action_skip_reasons'] == expected
    assert data['action_count'] == limit and len(data['bypassed_episodes']) == limit
    assert data['shadow_count'] == 20 and len(data['events']) == 20
    assert data['callback_calls'] == 20
    assert s.waiting.remove_calls == s.waiting.prepend_calls == limit
    assert len({(e['request'], e['num_preemptions']) for e in data['bypassed_episodes']}) == limit
    # An unqualified step increments counters without adding a large snapshot.
    head.num_tokens = 64
    assert s.schedule(8, []) == 'head' and len(data['events']) == 20
    assert data['skip_counts']['HEAD_FULL_HISTORY_FITS'] == 1
    undo(); undo()
    assert data['status'] == 'UNINSTALLED' and s.schedule is wrapper
    assert slot.cell_contents is native and s.schedule(8, []) == 'head'

for mode in repeat_fit.LIMITS:
    for barrier in ('different_priority', 'new_request'):
        s, head, target, data, undo, _, _, _ = setup(mode)
        middle = frozen['request']('barrier', 0)
        if barrier == 'different_priority':
            middle.priority = 1
        else:
            middle.status = NS(name='WAITING')
        s.waiting.insert(1, middle)
        assert s.schedule(8, []) == 'head'
        assert not data['events'] and data['action_count'] == 0
        assert [r.request_id for r in s.waiting] == ['head', 'barrier', 'anchor']
        undo()

for error_kind in ('peek', 'mutation'):
    s, head, target, data, undo, _, _, _ = setup('repeat8')
    if error_kind == 'peek':
        s.waiting.fail = True
    else:
        s.waiting.fail_mutation = True
    try:
        s.schedule(8, [])
    except (ValueError, RuntimeError) as error:
        assert str(error) == ('native queue exception' if error_kind == 'peek' else 'native mutation exception')
    else:
        raise AssertionError('Original exception swallowed')
    assert data['action_count'] == (1 if error_kind == 'peek' else 0)
    assert data['events'][-1]['action'] == ('REORDERED' if error_kind == 'peek' else 'ERROR')
    undo()

print('PASS: budgets exactly 1/8; per-preemption at most one bypass; shadows continue after budget; no priority/new-request crossing; native errors and uninstall preserved')

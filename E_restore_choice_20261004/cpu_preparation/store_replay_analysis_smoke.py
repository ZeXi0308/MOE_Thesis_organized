"""Focused CPU check: a wrong event or zero executed rewind must not qualify."""
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_one_event import audit_store_replay

commit = dict(event=0, request_id='measured/E246-test', preemptions=1,
              decision_s=24.0, allocation_s=24.001,
              known_tokens=3165, generated_tokens=125, prefix_sha256='known-only',
              host_hit_tokens=1168, action='recompute', actual_action='recompute',
              external_tokens=0, target_committed=True, eligible=True, joint_capacity=True)
record = {k: commit[k] for k in ('event', 'request_id', 'decision_s', 'preemptions', 'known_tokens',
                                'generated_tokens', 'prefix_sha256', 'host_hit_tokens', 'action')}
record.update(status='executed', executed=True, requested=True, target_selected=True,
              eligible=True, joint_capacity=True, target_commit_observed=True,
              request_state_preserved=True, known_prefix_matches_record=True,
              native_external_tokens=0, fallback='none', cursor_before=197,
              native_allocation_s=24.001, phase='service', step_index=416,
              cursor_after=73, ready_prefix_chunk_index=73, tokens_per_chunk=16)
record.update({k + '_after': commit[k] for k in
               ('known_tokens', 'generated_tokens', 'prefix_sha256')})
off = dict(enabled=False, records=[], errors=[], executed_count=0)
raw = dict(store_replay=dict(enabled=True, records=[record], errors=[], executed_count=1))
# The module also retains later non-target allocation observations.
raw['store_replay']['records'].append(dict(event=1, request_id=commit['request_id'],
    status='skipped', requested=False, executed=False, target_selected=False,
    target_commit_observed=False, reason='already_executed'))
warmup = dict(store_replay=off)
config = dict(policy='recompute', store_replay_enabled=True, store_replay_modes='off,on')

assert audit_store_replay(config, raw, warmup, [commit])['valid']
wrong = deepcopy(raw)
wrong['store_replay']['records'][0]['event'] = 1
assert 'replay_target_event_or_history_mismatch' in audit_store_replay(
    config, wrong, warmup, [commit])['failures']
zero = deepcopy(raw)
zero['store_replay']['executed_count'] = 0
zero['store_replay']['records'][0].update(status='skipped', executed=False, reason='no_op')
assert 'enabled_replay_requires_exactly_one_executed_record' in audit_store_replay(
    config, zero, warmup, [commit])['failures']
assert audit_store_replay(dict(config, store_replay_enabled=False),
                          dict(store_replay=off), warmup, [commit])['valid']
assert audit_store_replay({}, {}, None, [commit])['legacy_off']
assert audit_store_replay({}, {}, None, [commit])['valid']
print('PASS: executed replay/off/legacy accepted; wrong-event and zero-execution rejected')

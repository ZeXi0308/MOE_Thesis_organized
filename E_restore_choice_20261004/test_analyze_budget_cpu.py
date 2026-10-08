"""Synthetic CPU contract checks; these fixtures contain no performance evidence."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_budget import replay_decisions, validate_budget_episode


def fixture():
    budget = dict(step=0, checked=True, status='CHECKED', pause_state='UNPAUSED',
        initial_budget_tokens=4096, remaining_budget_tokens=0,
        allocated_compute_tokens=4096, native_scheduled_tokens_total=4096,
        positive_allocation_requests=2, allocation_calls=2,
        failed_allocations=0, zero_token_allocations=0, start_s=1.1, end_s=1.8)
    # All fields are invented contract examples, not measured service data.
    decisions = []
    for index, (rid, event, known, remaining, action) in enumerate([
            ('a', 0, 1793, 2048, 'recompute'),
            ('b', 0, 1792, 1792, 'recompute'),
            ('a', 1, 1792, 1791, 'host'),
            ('b', 1, 1793, 1000, 'host')]):
        decisions.append(dict(request_id=rid, event=event, eligible=True,
            known_tokens=known, remaining_budget_tokens=remaining, budget_step=0,
            action=action, decision_s=1.2 + index * .1))
    return dict(target_selection=dict(budget_observation=dict(
        signal='native_remaining_scheduled_token_budget', checked_steps=1,
        error_steps=0, steps=[budget])),
        scheduler_steps=[dict(time_s=1.85, scheduled=[
            dict(request_id='a', count=2048), dict(request_id='b', count=2048)])],
        steps=[dict(start_s=1.0, end_s=2.0)], all_complete_s=2.2,
        decisions=decisions, commits=[dict(row, actual_action=row['action']) for row in decisions])


def observation(raw):
    return raw['target_selection']['budget_observation']


def with_drain():
    raw = fixture()
    drain = dict(observation(raw)['steps'][0], step=1, start_s=2.3, end_s=2.4,
        remaining_budget_tokens=4096, allocated_compute_tokens=0,
        native_scheduled_tokens_total=0, positive_allocation_requests=0, allocation_calls=0)
    observation(raw)['steps'].append(drain)
    observation(raw)['checked_steps'] = 2
    raw['scheduler_steps'].append(dict(time_s=2.45, scheduled=[]))
    return raw


class BudgetAnalysisTests(unittest.TestCase):
    def assert_invalid(self, result, reason=None):
        self.assertEqual(result['status'], 'INVALID')
        self.assertTrue(result['errors'])
        if reason:
            self.assertTrue(any(reason in error for error in result['errors']), result['errors'])

    def test_missing_observation_is_unrun(self):
        for raw in ({}, {'target_selection': {}}):
            self.assertEqual(validate_budget_episode(raw)['status'], 'UNRUN')
            self.assertEqual(replay_decisions(raw, 'budget')['status'], 'UNRUN')

    def test_present_malformed_observation_is_invalid(self):
        for malformed in ([], 'bad', {'steps': 'bad'}, {'steps': [None]}):
            with self.subTest(malformed=malformed):
                raw = fixture()
                raw['target_selection']['budget_observation'] = malformed
                self.assert_invalid(validate_budget_episode(raw))
                self.assert_invalid(replay_decisions(raw, 'budget'))

    def test_valid_episode_and_zero_compute_drain(self):
        result = validate_budget_episode(fixture())
        self.assertEqual(result['status'], 'VALID', result['errors'])
        self.assertEqual(result['scheduled_native_tokens'], 4096)
        self.assertEqual(result['main_step_schedule_counts'], [1])
        drained = validate_budget_episode(with_drain())
        self.assertEqual(drained['status'], 'VALID', drained['errors'])
        self.assertEqual(drained['drain_schedule_steps'], 1)
        self.assertEqual(drained['scheduled_native_tokens'], 4096)

    def test_counters_indices_status_and_signal_are_checked(self):
        for field, value in [('signal', 'unknown'), ('checked_steps', 0), ('error_steps', 1)]:
            with self.subTest(field=field):
                raw = fixture()
                observation(raw)[field] = value
                self.assert_invalid(validate_budget_episode(raw))
        for field, value in [('step', 1), ('checked', False), ('status', 'ERROR'),
                             ('pause_state', 'UNKNOWN')]:
            with self.subTest(field=field):
                raw = fixture()
                observation(raw)['steps'][0][field] = value
                self.assert_invalid(validate_budget_episode(raw))

    def test_native_and_allocation_accounting_rejects_corruption(self):
        for field, value in [('remaining_budget_tokens', 1), ('allocated_compute_tokens', 4095),
                ('native_scheduled_tokens_total', 4095), ('positive_allocation_requests', 1),
                ('allocation_calls', 3), ('initial_budget_tokens', True),
                ('failed_allocations', -1), ('zero_token_allocations', 0.5)]:
            with self.subTest(field=field):
                raw = fixture()
                observation(raw)['steps'][0][field] = value
                self.assert_invalid(validate_budget_episode(raw))
        raw = fixture()
        raw['scheduler_steps'][0]['scheduled'][1]['request_id'] = 'a'
        self.assert_invalid(validate_budget_episode(raw), 'duplicate_native_request')
        raw = fixture()
        raw['scheduler_steps'].clear()
        self.assert_invalid(validate_budget_episode(raw), 'budget_and_schedule_count_differ')

    def test_paused_all_requires_zero_budget(self):
        raw = fixture()
        observation(raw)['steps'][0]['pause_state'] = 'PAUSED_ALL'
        self.assert_invalid(validate_budget_episode(raw), 'paused_all_nonzero_budget')

    def test_schedule_timing_must_fit_one_main_step(self):
        for key, value in [('start_s', .9), ('end_s', 1.9), ('end_s', float('nan'))]:
            with self.subTest(key=key, value=value):
                raw = fixture()
                observation(raw)['steps'][0][key] = value
                self.assert_invalid(validate_budget_episode(raw))
        raw = fixture()
        raw['steps'].append(dict(raw['steps'][0]))
        self.assert_invalid(validate_budget_episode(raw), 'overlapping_main_step_intervals')
        raw = fixture()
        raw['steps'].append(dict(start_s=3.0, end_s=4.0))
        self.assert_invalid(validate_budget_episode(raw), 'main_step_without_exactly_one_schedule')

    def test_drain_cannot_precede_completion_or_compute(self):
        raw = with_drain()
        raw['all_complete_s'] = 2.35
        self.assert_invalid(validate_budget_episode(raw), 'outside_main_steps_before_drain')
        raw = with_drain()
        raw['scheduler_steps'][1]['scheduled'] = [dict(request_id='late', count=1)]
        observation(raw)['steps'][1].update(remaining_budget_tokens=4095,
            allocated_compute_tokens=1, native_scheduled_tokens_total=1,
            positive_allocation_requests=1, allocation_calls=1)
        self.assert_invalid(validate_budget_episode(raw), 'drain_scheduled_compute')

    def test_replay_boundaries_and_both_disagreement_directions(self):
        result = replay_decisions(fixture(), 'budget')
        self.assertEqual(result['status'], 'VALID', result['errors'])
        self.assertEqual([(x['budget_prediction'], x['length_prediction']) for x in result['events']],
            [('recompute', 'host'), ('recompute', 'recompute'), ('host', 'recompute'), ('host', 'host')])
        for key in ('event_summary', 'commit_summary'):
            self.assertEqual(result[key]['eligible_records'], 4)
            self.assertEqual(result[key]['budget_vs_length_disagreements'], 2)

    def test_length_policy_verifies_actual_action(self):
        raw = fixture()
        for row in raw['decisions'] + raw['commits']:
            row['action'] = 'recompute' if row['known_tokens'] <= 1792 else 'host'
            if 'actual_action' in row:
                row['actual_action'] = row['action']
        self.assertEqual(replay_decisions(raw, 'length', 1792)['status'], 'VALID')
        self.assert_invalid(replay_decisions(raw, 'length', 1791), 'actual_length_action_mismatch')

    def test_actual_budget_action_and_committed_action_are_checked(self):
        raw = fixture()
        raw['decisions'][0]['action'] = 'host'
        self.assert_invalid(replay_decisions(raw, 'budget'), 'actual_budget_action_mismatch')
        raw = fixture()
        raw['commits'][0]['actual_action'] = 'host'
        self.assert_invalid(replay_decisions(raw, 'budget'), 'selected_vs_committed_action_mismatch')

    def test_ineligible_fallback_has_no_replayed_prediction(self):
        raw = fixture()
        raw['decisions'].append(dict(request_id='miss', event=9, eligible=False, action='native'))
        raw['commits'].append(dict(raw['decisions'][-1], actual_action='recompute'))
        result = replay_decisions(raw, 'budget')
        self.assertEqual(result['status'], 'VALID', result['errors'])
        self.assertEqual(result['event_summary']['eligible_records'], 4)
        self.assertEqual(result['commit_summary']['eligible_records'], 4)

    def test_invalid_decision_signals_are_rejected(self):
        for field, value in [('known_tokens', 0), ('known_tokens', True),
                ('remaining_budget_tokens', -1), ('remaining_budget_tokens', float('nan')),
                ('budget_step', 5), ('budget_step', None), ('decision_s', 5.0),
                ('remaining_budget_tokens', 4097)]:
            with self.subTest(field=field, value=value):
                raw = fixture()
                raw['decisions'][0][field] = value
                self.assert_invalid(replay_decisions(raw, 'budget'))
        raw = fixture()
        raw['decisions'][1]['remaining_budget_tokens'] = 2049
        self.assert_invalid(replay_decisions(raw, 'budget'), 'remaining_budget_increased')

    def test_event_keys_join_on_request_and_event_and_reject_bad_links(self):
        self.assertEqual(replay_decisions(fixture(), 'budget')['status'], 'VALID')
        for collection, reason in [('decisions', 'duplicate_decision_key'),
                                   ('commits', 'duplicate_commit_key')]:
            raw = fixture()
            raw[collection].append(dict(raw[collection][0]))
            self.assert_invalid(replay_decisions(raw, 'budget'), reason)
        raw = fixture()
        raw['commits'][0]['request_id'] = 'orphan'
        self.assert_invalid(replay_decisions(raw, 'budget'), 'orphan_commit')
        raw = fixture()
        raw['commits'][0]['known_tokens'] += 1
        self.assert_invalid(replay_decisions(raw, 'budget'), 'commit_decision_state_mismatch')

    def test_future_output_fields_cannot_change_replayed_actions(self):
        raw = fixture()
        before = deepcopy(raw)
        baseline = replay_decisions(raw, 'budget')
        self.assertEqual(raw, before, 'analysis must not mutate its input')
        raw.update(requests=[dict(output_token_ids=[999] * 100, max_output_tokens=10000,
                                  completion_s=999.0, token_times_s=[998.0] * 100)],
                   all_complete_s=1000.0, future_output_length=100000)
        self.assertEqual(replay_decisions(raw, 'budget'), baseline)


if __name__ == '__main__':
    unittest.main(verbosity=2)

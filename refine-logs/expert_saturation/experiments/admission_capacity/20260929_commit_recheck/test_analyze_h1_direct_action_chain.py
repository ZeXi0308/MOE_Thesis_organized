"""CPU evidence-boundary tests; these fixtures are not runtime qualification."""
import unittest
from analyze_h1_direct_action_chain import analyze


def fixture():
    raw = {'status': 'COMPLETE', 'error': None, 'measurement_origin_perf_counter_s': 100,
           'requests': [
               {'internal_request_id': 'target', 'request_id': 'source-target', 'status': 'completed',
                'token_times_s': [0.5, 2.0, 3.0], 'completion_s': 3.0, 'stop_reason': 'stop'},
               {'internal_request_id': 'victim', 'request_id': 'source-victim', 'status': 'completed'}],
           'scheduler_steps': [{'step': 7, 'start_s': 1.0, 'scheduled': [
               {'internal_request_id': 'target', 'scheduled_tokens': 16}], 'preempted_request_ids': []}]}
    data = {'status': 'DRAINED', 'commit_recheck': True, 'direct_commits': 1,
            'native_reservation_gate': dict(checked=1, zero=1, positive_keep=0, unknown_keep=0),
            'events': [
                {'event': 'commit_recheck', 'step': 7, 'target': 'target', 'planned_victim': 'victim',
                 'reason': 'DIRECT_READY', 'base_reason': 'DIRECT_READY', 'native_inflight_reserved_blocks': 0},
                {'event': 'direct_commit', 'step': 7, 'target': 'target', 'native_admission': 'SCHEDULED_TOKENS',
                 'scheduled_tokens': 16, 'load_job_ids': []}]}
    return raw, data, {'completed_jobs': []}


class EvidenceTests(unittest.TestCase):
    def test_native_compute_and_later_output_chain(self):
        result = analyze(*fixture())
        self.assertEqual(result['status'], 'OBSERVED_DIRECT_ACTION_CHAIN')
        self.assertEqual(result['chains'][0]['first_observed_output_after_ready_s'], 2.0)

    def test_async_requires_matching_completed_target_load(self):
        raw, data, offload = fixture()
        event = data['events'][-1]
        event.update(native_admission='ASYNC_LOAD_ADMITTED', scheduled_tokens=0, load_job_ids=[12])
        raw['scheduler_steps'][0]['scheduled'] = []
        offload['completed_jobs'] = [{'time_s': 101.5, 'jobs': [
            {'job_id': 12, 'request': 'target', 'is_store': False}]}]
        self.assertEqual(analyze(raw, data, offload)['chains'][0]['native_ready_s'], 1.5)
        offload['completed_jobs'][0]['jobs'][0]['request'] = 'unrelated'
        with self.assertRaisesRegex(ValueError, 'load did not complete'):
            analyze(raw, data, offload)

    def test_admission_with_only_earlier_output_does_not_qualify(self):
        raw, data, offload = fixture()
        raw['requests'][0]['token_times_s'] = [0.5]
        with self.assertRaisesRegex(ValueError, 'subsequent new token'):
            analyze(raw, data, offload)

    def test_victim_preempted_in_source_namespace_is_rejected(self):
        raw, data, offload = fixture()
        raw['scheduler_steps'][0]['preempted_request_ids'] = ['source-victim']
        with self.assertRaisesRegex(ValueError, 'planned victim'):
            analyze(raw, data, offload)

    def test_zero_direct_actions_is_a_stop(self):
        raw, data, offload = fixture()
        data['events'] = []
        data['direct_commits'] = 0
        data['native_reservation_gate'] = dict(checked=0, zero=0, positive_keep=0, unknown_keep=0)
        self.assertEqual(analyze(raw, data, offload)['status'], 'NO_ACTION')


if __name__ == '__main__':
    unittest.main()

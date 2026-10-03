import copy
import unittest

from partition import partition


def request(rid, times, status='completed', completion=5, arrival=0):
    return dict(request_id=rid, token_times_s=times, output_token_ids=list(range(len(times))),
                status=status, completion_s=completion, arrival_s=arrival)


def event(rid, count, last, entered, call=1, success=True):
    return dict(request_id=rid, last_returned_output_count=count, last_new_output_s=last,
        method_entered_s=entered, method_returned_s=entered+.01 if success else None,
        original_preemption_returned=success, engine_call_index=call)


def raw(rows, events, status='COMPLETE', returned=4):
    return dict(diagnostics='SPARSE_PREEMPTION_EVENTS', requests=rows,
        preemption_events=events, observation_end_s=5, engine_return_count=returned, status=status)


class PartitionTest(unittest.TestCase):
    def test_two_preemptions_one_gap_and_unresolved_chunk(self):
        data = raw([request('a', [1, 2, 2, 4])], [event('a', 3, 2, 2.5), event('a', 3, 2, 3)])
        result = partition(data)
        r = result['requests'][0]
        self.assertEqual(r['gap_with_preemption_s'], 2)
        self.assertEqual(r['successful_preemptions_in_generation_gaps'], 2)
        self.assertEqual(r['marked_gap_count'], 1)
        self.assertEqual(r['other_generation_gap_s'], 1)
        self.assertEqual(r['pre_output_wait_s'], 1)
        self.assertEqual(r['completed_tail_s'], 1)
        self.assertEqual(r['observed_request_s'], 5)
        bad = copy.deepcopy(data)
        bad['preemption_events'][0]['last_new_output_s'] = 1
        with self.assertRaisesRegex(ValueError, 'last output differs'):
            partition(bad)

    def test_first_output_failure_censored_tail_and_future_request(self):
        rows = [request('a', [1], 'failed', None), request('b', [], 'failed', None),
                request('c', [], 'unfinished', None, arrival=7)]
        events = [event('a', 1, 1, 2, call=2), event('b', 0, None, 2, call=2, success=False)]
        result = partition(raw(rows, events, status='INCOMPLETE', returned=2))
        a, b, c = result['requests']
        self.assertFalse(result['comparable_complete_service'])
        self.assertEqual(a['censored_tail_s'], 4)
        self.assertEqual(a['tail_successful_event_indices'], [0])
        self.assertEqual(a['preemption_events'][0]['kind'], 'method_returned_call_not_returned')
        self.assertEqual(b['pre_output_wait_s'], 5)
        self.assertFalse(b['first_output_observed'])
        self.assertEqual(b['pre_output_successful_event_indices'], [])
        self.assertEqual(b['preemption_events'][0]['kind'], 'method_failed')
        self.assertEqual(c['observed_request_s'], 0)
        self.assertEqual(result['counts']['planned'], 3)
        self.assertEqual(result['counts']['arrived'], 2)

    def test_pre_output_preemption_and_zero_output_completion(self):
        result = partition(raw([request('a', [3, 4]), request('b', [])],
            [event('a', 0, None, 1), event('b', 0, None, 2)]))
        a, b = result['requests']
        self.assertEqual(a['pre_output_wait_s'], 3)
        self.assertEqual(a['pre_output_successful_event_indices'], [0])
        self.assertEqual(a['marked_gap_count'], 0)
        self.assertTrue(b['completed_without_output'])
        self.assertFalse(b['first_output_observed'])
        self.assertEqual(b['pre_output_wait_s'], 5)
        self.assertEqual(b['pre_output_successful_event_indices'], [1])


if __name__ == '__main__':
    unittest.main()

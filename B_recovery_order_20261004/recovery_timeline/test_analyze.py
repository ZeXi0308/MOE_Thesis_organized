"""Small synthetic checks for ambiguous recovery evidence; no trace files read."""
import unittest

from recovery_timeline.analyze import analyze_documents, interval


RID = 'internal-a'


def documents():
    return {
        'raw.json': {
            'status': 'COMPLETE', 'measurement_origin_perf_counter_s': 100.,
            'observation_end_s': 12., 'internal_to_source': {RID: 'source-a'},
            'requests': [{'request_id': 'source-a', 'internal_request_id': RID,
                          'token_times_s': [1., 10.], 'status': 'completed'}],
            'preemption_events': [],
        },
        'recovery-order.json': {'clock': 'time.perf_counter host',
            'events': [{'kind': 'preempt', 'request': RID, 'host_perf_s': 102.}]},
        'capacity-handoff.json': {'clock': 'host time.perf_counter', 'events': [],
            'scope': 'Synthetic recorded calls only; no continuous capacity observation'},
        'source-handoff.json': {'clock': 'time.perf_counter host', 'events': []},
    }


def load(job_id, start=103., request=RID):
    # These are separate host observations, not an assumed GPU duration.
    return [dict(kind=kind, job_id=job_id, request=request, is_store=False,
                 host_perf_s=start+offset)
            for kind, offset in (
                ('job_created', 0.), ('ready', .1), ('submit_begin', .2),
                ('submit_end', .3), ('job_completed', .8), ('ack_retired', .9))]


def analyze(docs):
    return analyze_documents(docs, run_id='fixture')


class RecoveryTimelineTests(unittest.TestCase):
    def assert_unknown(self, result):
        self.assertEqual(result['status'], 'unknown')
        self.assertIsNone(result['duration_s'])
        self.assertEqual(result['causal_attribution'], 'unknown')

    def test_multiple_load_jobs_for_one_request_are_all_retained(self):
        docs = documents()
        docs['recovery-order.json']['events'] += load(11, 103.) + load(12, 103.4)
        result = analyze(docs)
        self.assertEqual(result['summary']['jobs'], 2)
        self.assertEqual(result['summary']['multi_load_episodes'], 1)
        self.assertEqual(set(result['episodes'][0]['load_job_ids']), {11, 12})
        for job in result['jobs']:
            self.assertEqual(job['association'], 'unique')
            self.assertEqual(job['episode_association'], 'unique')
            self.assertEqual(len(job['event_ids']), 6)
            self.assertEqual(job['occurrence_counts']['ready'], 1)
            self.assertEqual(job['gpu_completion_time']['status'], 'unknown')
        self.assertEqual(len([e for e in result['events'] if e['job_id'] is not None]), 12)

    def test_repeated_ready_is_not_collapsed_to_a_unique_interval(self):
        docs = documents()
        rows = load(11)
        rows.insert(2, dict(rows[1], host_perf_s=103.15))
        docs['recovery-order.json']['events'] += rows
        result = analyze(docs)
        job = result['jobs'][0]
        self.assertEqual(job['occurrence_counts']['ready'], 2)
        self.assertEqual(len(job['event_ids']), 7)
        self.assertEqual(result['source_event_counts']['recovery-order.json']['ready'], 2)
        for phase in ('job_created_to_ready', 'ready_to_submit_begin', 'preempt_to_ready'):
            self.assert_unknown(job['phases'][phase])
            self.assertEqual(job['phases'][phase]['reason'], 'ambiguous_repeated_endpoint')

    def test_missing_ack_remains_unknown(self):
        docs = documents()
        docs['recovery-order.json']['events'] += load(11)[:-1]
        docs['recovery-order.json']['events'].append(
            dict(kind='scheduled', request=RID, host_perf_s=105., scheduled_tokens=1))
        job = analyze(docs)['jobs'][0]
        self.assertEqual(job['occurrence_counts']['ack_retired'], 0)
        for phase in ('job_completed_to_ack_retired', 'ack_to_first_schedule'):
            self.assert_unknown(job['phases'][phase])
            self.assertEqual(job['phases'][phase]['reason'], 'missing_endpoint')

    def test_different_clock_domains_do_not_produce_elapsed_time(self):
        left = dict(event_id='left', clock_domain='clock-a', time_s=1.)
        right = dict(event_id='right', clock_domain='clock-b', time_s=2.)
        elapsed = interval([left], [right])
        self.assert_unknown(elapsed)
        self.assertEqual(elapsed['reason'], 'unaligned_clock')
        docs = documents()
        rows = load(11)
        rows[1]['clock_domain'] = 'different-host'
        docs['recovery-order.json']['events'] += rows
        job = analyze(docs)['jobs'][0]
        self.assert_unknown(job['phases']['job_created_to_ready'])
        self.assert_unknown(job['phases']['ready_to_submit_begin'])
        self.assert_unknown(job['phases']['preempt_to_ready'])

    def test_conflicting_request_identity_cannot_certify_one_job_chain(self):
        docs = documents()
        rows = load(11)
        rows[1]['request'] = 'internal-b'
        docs['recovery-order.json']['events'] += rows
        result = analyze(docs)
        job = result['jobs'][0]
        self.assertEqual(job['association'], 'unknown')
        self.assertEqual(job['episode_association'], 'unknown')
        self.assertIsNone(job['request_id'])
        self.assertEqual(set(job['candidate_requests']), {RID, 'internal-b'})
        self.assertEqual(result['episodes'][0]['load_job_ids'], [])
        self.assertEqual(result['episodes'][0]['ambiguous_load_job_ids'], [11])
        # Same job ID with different request owners cannot certify a lifecycle
        # duration by pairing those owners' observations.
        self.assert_unknown(job['phases']['job_created_to_ready'])
        self.assert_unknown(job['phases']['ready_to_submit_begin'])

    def test_no_load_still_links_preemption_and_next_host_output(self):
        docs = documents()
        docs['raw.json']['preemption_events'] = [dict(request_id='source-a',
            internal_request_id=RID, method_entered_s=2., method_returned_s=2.01)]
        docs['source-handoff.json'].update(selected_request=RID)
        docs['source-handoff.json']['events'] = [dict(kind='native_handoff',
            host_perf_s=104., external_tokens=16, load_jobs=[])]
        result = analyze(docs)
        self.assertEqual(result['summary']['episodes'], 1)  # Raw hook corroborates, not duplicates.
        self.assertEqual(result['summary']['jobs'], 0)
        self.assertEqual(result['summary']['zero_observed_load_episodes'], 1)
        episode = result['episodes'][0]
        self.assertEqual(episode['output_association'], 'observed')
        self.assertEqual(episode['load_job_ids'], [])
        self.assertEqual(episode['source_handoffs'][0]['load_jobs'], [])
        self.assertEqual(episode['phases']['preempt_to_next_output']['duration_s'], 8.)
        self.assertEqual(episode['phases']['previous_to_next_output']['duration_s'], 9.)
        self.assert_unknown(episode['phases']['preempt_to_first_lookup'])
        self.assert_unknown(episode['phases']['preempt_to_first_schedule'])

    def test_overlapping_preemptions_leave_job_episode_ownership_unknown(self):
        docs = documents()
        docs['recovery-order.json']['events'].append(
            dict(kind='preempt', request=RID, host_perf_s=102.5))
        docs['recovery-order.json']['events'] += load(11, 103.)
        result = analyze(docs)
        self.assertEqual(result['summary']['episodes'], 2)
        job = result['jobs'][0]
        self.assertEqual(job['association'], 'unique')
        self.assertEqual(job['episode_association'], 'unknown')
        self.assertEqual(len(job['episode_candidates']), 2)
        for episode in result['episodes']:
            self.assertEqual(episode['load_job_ids'], [])
            self.assertEqual(episode['ambiguous_load_job_ids'], [11])
            self.assertEqual(len(episode['overlapping_episode_ids']), 1)

    def test_all_capacity_attempts_and_unattributed_retry_gaps_are_preserved(self):
        docs = documents()
        for attempt, begin, end, success, shortfall in (
                (1, 102.4, 102.5, False, 3),
                (1, 103., 103.1, False, 2),  # Repeated attempt label is not an event ID.
                (2, 104., 104.2, True, 0)):
            docs['capacity-handoff.json']['events'].append(dict(kind='allocate',
                request=RID, attempt=attempt, begin_host_perf_s=begin, end_host_perf_s=end,
                success=success, descriptor=dict(exact=True, shortfall_blocks=shortfall)))
        result = analyze(docs)
        allocations = result['episodes'][0]['allocations']
        self.assertEqual((allocations['attempts'], allocations['failed'], allocations['successful']), (3, 2, 1))
        self.assertEqual(len(set(allocations['event_ids'])), 3)
        self.assertEqual(result['source_event_counts']['capacity-handoff.json']['allocate'], 3)
        self.assertEqual(len(allocations['retry_gaps']), 2)
        for gap, duration, shortfall in zip(allocations['retry_gaps'], (.5, .9), (3, 2)):
            self.assertEqual(gap['status'], 'observed')
            self.assertAlmostEqual(gap['duration_s'], duration)
            self.assertEqual(gap['failed_snapshot_shortfall_blocks'], shortfall)
            self.assertEqual(gap['continuous_capacity_shortage'], 'unknown')
            self.assertEqual(gap['causal_attribution'], 'unknown')
            self.assertEqual(gap['reason'], 'no_recorded_allocator_retry_between_these_observations')


if __name__ == '__main__':
    unittest.main()

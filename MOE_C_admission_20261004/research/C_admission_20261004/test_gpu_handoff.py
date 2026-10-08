"""Bounded GPU handoff checks with simulated queries and clocks only."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run as runner


class Clock:
    def __init__(self):
        self.now = 100.
        self.sleeps = []

    def monotonic(self):
        return self.now

    def time(self):
        return 1000. + self.now - 100.

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class GPUHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.clock = Clock()
        self.device = runner.UUID + ', NVIDIA RTX PRO 6000, 97887 MiB, 100 MiB\n'
        self.own_process = str(os.getpid()) + ', python, 100 MiB\n'
        self.queries = []
        for field in ('monotonic', 'time', 'sleep'):
            mocked = patch.object(runner.time, field, side_effect=getattr(self.clock, field))
            mocked.start()
            self.addCleanup(mocked.stop)

    def query(self, response, latency=0.):
        """Build one fake subprocess call; verify its remaining deadline."""
        def perform(args, **kwargs):
            self.assertEqual(args[0], 'nvidia-smi')
            self.assertEqual(args[-1], '--format=csv,noheader')
            self.assertTrue(kwargs['text'])
            self.assertAlmostEqual(kwargs['timeout'], 115. - self.clock.now)
            self.assertGreater(kwargs['timeout'], 0.)
            self.queries.append((args[1], kwargs['timeout']))
            self.clock.now += latency
            return response
        return perform

    def report(self, status, elapsed):
        report = json.loads((self.out / 'gpu-handoff.json').read_text())
        self.assertEqual(report['status'], status)
        self.assertEqual(report['expected_uuid'], runner.UUID)
        self.assertEqual(report['pid'], os.getpid())
        self.assertAlmostEqual(report['elapsed_s'], elapsed)
        self.assertAlmostEqual(report['finished_unix'] - report['started_unix'], elapsed)
        for row in report['polls']:
            self.assertGreaterEqual(row['start_elapsed_s'], 0.)
            self.assertGreaterEqual(row['duration_s'], 0.)
            self.assertLessEqual(row['start_elapsed_s'] + row['duration_s'], elapsed + 1e-9)
        return report

    def test_busy_own_pid_then_empty_releases_and_records_both_polls(self):
        calls = iter(self.query(value, .1) for value in
                     (self.device, self.own_process, self.device, ''))
        with patch.object(runner.subprocess, 'check_output',
                          side_effect=lambda *a, **kw: next(calls)(*a, **kw)) as query:
            state = runner.wait_idle_gpu(self.out)
        self.assertEqual(state['processes'], '')
        self.assertEqual(query.call_count, 4)
        self.assertEqual(self.clock.sleeps, [.5])
        report = self.report('IDLE', .9)
        self.assertEqual([row['status'] for row in report['polls']], ['BUSY', 'IDLE'])
        self.assertIn(str(os.getpid()), report['polls'][0]['state']['processes'])
        for row in report['polls']:
            self.assertAlmostEqual(row['duration_s'], .2)
        self.assertAlmostEqual(report['polls'][1]['start_elapsed_s'], .7)
        self.assertEqual([field for field, _ in self.queries],
                         ['--query-gpu=uuid,name,memory.total,memory.used',
                          '--query-compute-apps=pid,process_name,used_memory'] * 2)

    def test_persistent_busy_refuses_at_15_seconds_with_all_polls(self):
        def busy(args, **kwargs):
            value = self.device if args[1].startswith('--query-gpu=') else self.own_process
            return self.query(value)(args, **kwargs)
        with patch.object(runner.subprocess, 'check_output', side_effect=busy) as query:
            with self.assertRaisesRegex(TimeoutError, 'handoff deadline'):
                runner.wait_idle_gpu(self.out)
        self.assertEqual(query.call_count, 60)
        self.assertEqual(self.clock.sleeps, [.5] * 30)
        report = self.report('FAILED', 15.)
        self.assertIn('TimeoutError', report['error'])
        self.assertEqual(len(report['polls']), 30)
        self.assertEqual([row['start_elapsed_s'] for row in report['polls']],
                         [.5 * i for i in range(30)])
        self.assertTrue(all(row['status'] == 'BUSY' and row['duration_s'] == 0.
                            for row in report['polls']))

    def test_wrong_uuid_fails_immediately_without_compute_query(self):
        wrong = 'GPU-wrong-device, Other GPU, 97887 MiB, 100 MiB\n'
        with patch.object(runner.subprocess, 'check_output', side_effect=self.query(wrong, .1)) as query:
            with self.assertRaises(runner.GPUIdentityMismatch):
                runner.wait_idle_gpu(self.out)
        self.assertEqual(query.call_count, 1)
        self.assertTrue(self.queries[0][0].startswith('--query-gpu='))
        self.assertEqual(self.clock.sleeps, [])
        report = self.report('FAILED', .1)
        self.assertEqual(len(report['polls']), 1)
        row = report['polls'][0]
        self.assertEqual(row['status'], 'ERROR')
        self.assertEqual(row['state']['device'], wrong.strip())
        self.assertIsNone(row['state']['processes'])
        self.assertIn('GPUIdentityMismatch', report['error'])

    def test_query_timeout_propagates_and_retains_error_poll(self):
        failure = subprocess.TimeoutExpired('nvidia-smi', 15.)
        def query_timeout(args, **kwargs):
            self.query('', kwargs['timeout'])(args, **kwargs)
            raise failure
        with patch.object(runner.subprocess, 'check_output', side_effect=query_timeout) as query:
            with self.assertRaises(subprocess.TimeoutExpired) as caught:
                runner.wait_idle_gpu(self.out)
        self.assertIs(caught.exception, failure)
        self.assertEqual(query.call_count, 1)
        self.assertEqual(self.clock.sleeps, [])
        report = self.report('FAILED', 15.)
        self.assertEqual(len(report['polls']), 1)
        row = report['polls'][0]
        self.assertEqual(row['status'], 'ERROR')
        self.assertIsNone(row['state'])
        self.assertAlmostEqual(row['duration_s'], 15.)
        self.assertIn('TimeoutExpired', row['error'])
        self.assertEqual(report['error'], row['error'])


if __name__ == '__main__':
    unittest.main()

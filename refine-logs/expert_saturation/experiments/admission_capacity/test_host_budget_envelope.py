"""File-fixture and bootstrap-order checks, NOT Linux enforcement tests."""
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import host_budget_envelope as env


@contextmanager
def delegated_fixture(root, *, delegated=True):
    parent = (root/'delegated').resolve()
    parent.mkdir()
    (parent/'cgroup.subtree_control').write_text('memory' if delegated else 'cpu')
    (parent/'cgroup.procs').write_text('999999')  # Must remain untouched.
    original_mkdir = Path.mkdir

    def mkdir(path, *args, **kwargs):
        original_mkdir(path, *args, **kwargs)
        if path.parent == parent:
            for key, value in {'memory.max': 'max', 'memory.swap.max': 'max',
                               'memory.current': '0', 'memory.peak': '0',
                               'memory.events': 'max 0\noom 0\noom_kill 0',
                               'memory.stat': 'anon 0\nfile 0', 'memory.swap.current': '0',
                               'cgroup.events': 'populated 0\nfrozen 0', 'cgroup.procs': ''}.items():
                (path/key).write_text(value)
    with mock.patch.object(env, 'is_cgroup2', return_value=True), \
            mock.patch.object(Path, 'mkdir', autospec=True, side_effect=mkdir):
        yield parent


class HostBudgetEnvelopeTests(unittest.TestCase):
    def test_budget_readback_and_owned_path_do_not_mutate_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with delegated_fixture(root) as parent:
                receipt = {}
                group = env.create_owned_group(parent, 4096, 0, receipt)
                self.assertEqual(receipt['limit_readback'], {'memory.max': '4096', 'memory.swap.max': '0'})
                self.assertEqual(group.parent, parent)
                self.assertTrue(group.name.startswith('moe-host-'))
                self.assertEqual((parent/'cgroup.procs').read_text(), '999999')
                self.assertEqual((parent/'cgroup.subtree_control').read_text(), 'memory')

    def test_no_delegation_refuses_launch_and_keeps_failure_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with delegated_fixture(root, delegated=False) as parent, mock.patch.object(env.subprocess, 'Popen') as launch:
                result = env.run_budgeted(['must-not-run'], parent=parent, budget_bytes=4096,
                                          swap_bytes=0, output=root/'out')
            launch.assert_not_called()
            self.assertEqual(result['status'], 'NOT_ENFORCED')
            self.assertFalse(result['launcher_started'])
            self.assertEqual(json.loads((root/'out/terminal.json').read_text())['status'], 'NOT_ENFORCED')
            self.assertEqual(list(parent.glob('moe-host-*')), [])

    def test_limit_mismatch_refuses_launch_and_retains_requested_and_actual(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original_read = env.read
            def read(path):
                return '8192' if path.name == 'memory.max' else original_read(path)
            with delegated_fixture(root) as parent, mock.patch.object(env, 'read', side_effect=read), \
                    mock.patch.object(env.subprocess, 'Popen') as launch:
                result = env.run_budgeted(['must-not-run'], parent=parent, budget_bytes=4096,
                                          swap_bytes=0, output=root/'out')
            launch.assert_not_called()
            self.assertEqual(result['status'], 'NOT_ENFORCED')
            self.assertEqual(result['budget_bytes'], 4096)
            self.assertEqual(result['limit_readback']['memory.max'], '8192')
            self.assertIn('readback mismatch', result['error'])

    def test_launch_failure_is_recorded_after_verified_limits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with delegated_fixture(root) as parent, mock.patch.object(env.subprocess, 'Popen', side_effect=OSError('fixture launch error')):
                result = env.run_budgeted(['must-not-run'], parent=parent, budget_bytes=4096,
                                          swap_bytes=0, output=root/'out')
            self.assertEqual(result['status'], 'START_FAILED')
            self.assertIn('fixture launch error', result['error'])
            self.assertFalse(result['launcher_started'])
            self.assertEqual(json.loads((root/'out/terminal.json').read_text())['status'], 'START_FAILED')

    def test_bootstrap_joins_and_records_membership_before_exec_or_descendant_spawn(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); group = root/'group'; group.mkdir()
            (group/'cgroup.procs').write_text('')
            membership = root/'membership.json'; result = root/'target.json'
            target = """import json,os,subprocess,sys
from pathlib import Path
group,membership,result=map(Path,sys.argv[1:])
receipt=json.loads(membership.read_text())
assert receipt['membership_verified'] and receipt['pid']==os.getpid()
assert str(os.getpid()) in (group/'cgroup.procs').read_text().split()
child=subprocess.check_output([sys.executable,'-c','import os;print(os.getppid())'],text=True)
result.write_text(json.dumps(dict(target_pid=os.getpid(),child_parent=int(child))))
"""
            command = [sys.executable, '-c', env.BOOTSTRAP, str(group), str(membership),
                       sys.executable, '-c', target, str(group), str(membership), str(result)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            row = json.loads(result.read_text())
            self.assertEqual(row['target_pid'], row['child_parent'])
            self.assertEqual(json.loads(membership.read_text())['pid'], row['target_pid'])

    def test_failed_membership_write_does_not_exec_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'cgroup.procs').mkdir()  # Fixture write failure.
            target = root/'must-not-exist'; membership = root/'membership.json'
            completed = subprocess.run([sys.executable, '-c', env.BOOTSTRAP, str(root), str(membership),
                                        sys.executable, '-c', 'from pathlib import Path;import sys;Path(sys.argv[1]).touch()', str(target)],
                                       capture_output=True, timeout=10)
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(target.exists()); self.assertFalse(membership.exists())

    def test_populated_is_distinct_from_frozen(self):
        self.assertFalse(env.populated('populated 0\nfrozen 1'))
        self.assertTrue(env.populated('populated 1\nfrozen 0'))
        with self.assertRaises(KeyError):
            env.populated('frozen 0')

    def test_failed_command_retains_streamed_samples_and_terminal_returncode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            failed = mock.Mock(returncode=7, pid=999998)
            failed.poll.return_value = 7
            with delegated_fixture(root) as parent, mock.patch.object(env.subprocess, 'Popen', return_value=failed):
                result = env.run_budgeted(['fixture-failure'], parent=parent, budget_bytes=4096,
                                          swap_bytes=0, output=root/'out')
            self.assertEqual(result['status'], 'COMMAND_FAILED')
            self.assertEqual(result['returncode'], 7)
            samples = [json.loads(s) for s in (root/'out/samples.jsonl').read_text().splitlines()]
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]['memory.current'], '0')
            self.assertIn('sampled_rss_sum_bytes', samples[0])
            self.assertEqual(json.loads((root/'out/terminal.json').read_text())['returncode'], 7)


if __name__ == '__main__':
    unittest.main()

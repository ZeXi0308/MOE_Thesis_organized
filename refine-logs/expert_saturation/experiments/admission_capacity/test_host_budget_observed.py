"""Read-only launcher scope and failure-retention tests; no cgroup enforcement."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

import host_budget_observed as observed


def cgroup_fixture(root, limit='4096'):
    group = root/'cgroup'
    group.mkdir()
    values = {
        'memory.max': limit, 'memory.current': '100', 'memory.peak': '200',
        'memory.events': 'oom 0\noom_kill 0', 'memory.swap.max': '0',
        'memory.swap.current': '0', 'memory.swap.events': 'fail 0'}
    for name, value in values.items():
        (group/name).write_text(value)
    return group


def proc_entry(root, pid, ppid, starttime, rss_kib):
    folder = root/str(pid)
    folder.mkdir(parents=True)
    fields = ['S', str(ppid)] + ['0']*17 + [str(starttime)]
    (folder/'stat').write_text(f"{pid} (fixture process) "+' '.join(fields))
    (folder/'status').write_text(f"VmRSS:\t{rss_kib} kB\nRssAnon:\t{rss_kib} kB\n")


class FailedChild:
    pid = 100
    returncode = 7

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


class HostBudgetObservedTests(unittest.TestCase):
    def test_parent_limit_mismatch_refuses_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent = cgroup_fixture(root, '8192')
            proc = root/'proc'
            (proc/'self').mkdir(parents=True)
            (proc/'self/cgroup').write_text('0::/')
            with mock.patch.object(observed.subprocess, 'Popen') as launch:
                for index, interval in enumerate((0.0, -1.0, float('nan'),
                                                  float('inf'), float('-inf'))):
                    with self.subTest(interval=interval), self.assertRaises(ValueError):
                        observed.run_observed(['must-not-run'], parent=parent,
                            expected_parent_memory_max=4096, output=root/f'invalid-{index}',
                            declared_host_kv_offload_bytes=0, interval_s=interval, proc_root=proc)
                result = observed.run_observed(['must-not-run'], parent=parent,
                    expected_parent_memory_max=4096, output=root/'out',
                    declared_host_kv_offload_bytes=0, proc_root=proc)
            launch.assert_not_called()
            self.assertEqual(result['status'], 'REFUSED_PARENT_LIMIT_MISMATCH')
            self.assertEqual(result['observed_parent_memory_max'], '8192')
            self.assertFalse(result['launched'])
            self.assertEqual(json.loads((root/'out/terminal.json').read_text())['status'],
                             'REFUSED_PARENT_LIMIT_MISMATCH')

    def test_failed_child_keeps_sample_and_returncode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent = cgroup_fixture(root)
            proc = root/'proc'
            (proc/'self').mkdir(parents=True)
            (proc/'self/cgroup').write_text('0::/')
            proc_entry(proc, 100, 1, 1000, 10)
            proc_entry(proc, 101, 100, 1001, 20)
            with mock.patch.object(observed.subprocess, 'Popen', return_value=FailedChild()):
                result = observed.run_observed(['fails'], parent=parent,
                    expected_parent_memory_max=4096, output=root/'out',
                    declared_host_kv_offload_bytes=0, interval_s=.01, proc_root=proc)
            self.assertEqual(result['status'], 'CHILD_FAILED_OBSERVED_ONLY')
            self.assertEqual(result['returncode'], 7)
            self.assertEqual(result['samples_written'], 1)
            sample = json.loads((root/'out/samples.jsonl').read_text())
            self.assertEqual([row['pid'] for row in sample['members']], [100, 101])
            self.assertEqual(sample['sampled_process_tree_rss_sum_bytes'], 30*1024)
            self.assertFalse(result['child_running_at_monitor_exit'])

    def test_monitor_error_is_saved_then_waits_for_owned_child(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent = cgroup_fixture(root)
            proc = root/'proc'
            (proc/'self').mkdir(parents=True)
            (proc/'self/cgroup').write_text('0::/')
            real_popen = observed.subprocess.Popen
            real_save = observed.save
            children, saved_statuses = [], []

            def launch(*args, **kwargs):
                child = real_popen(*args, **kwargs)
                children.append(child)
                return child

            def record_save(out, receipt):
                saved_statuses.append(receipt['status'])
                return real_save(out, receipt)

            command = [sys.executable, '-c',
                       'import sys,time; time.sleep(0.2); sys.exit(9)']
            started = time.monotonic()
            with mock.patch.object(observed.subprocess, 'Popen', side_effect=launch), \
                    mock.patch.object(observed, 'sample',
                                      side_effect=RuntimeError('injected monitor failure')), \
                    mock.patch.object(observed, 'save', side_effect=record_save):
                result = observed.run_observed(command, parent=parent,
                    expected_parent_memory_max=4096, output=root/'out',
                    declared_host_kv_offload_bytes=0, interval_s=.01, proc_root=proc)
            elapsed = time.monotonic()-started
            terminal = json.loads((root/'out/terminal.json').read_text())
            self.assertGreaterEqual(elapsed, .15)
            self.assertEqual(children[0].poll(), 9)
            self.assertIn('MONITOR_FAILED_WAITING_FOR_CHILD', saved_statuses)
            self.assertEqual(result['status'], 'MONITOR_FAILED_CHILD_TERMINAL')
            self.assertEqual(result['error'], 'RuntimeError: injected monitor failure')
            self.assertEqual(result['returncode'], 9)
            self.assertFalse(result['child_running_at_monitor_exit'])
            self.assertEqual(terminal['error'], result['error'])
            self.assertEqual(terminal['status'], result['status'])

    def test_rss_scope_is_root_and_ppid_descendants(self):
        with tempfile.TemporaryDirectory() as temp:
            proc = Path(temp)
            proc_entry(proc, 100, 1, 1000, 10)
            proc_entry(proc, 101, 100, 1001, 20)
            proc_entry(proc, 102, 101, 1002, 30)
            proc_entry(proc, 200, 1, 2000, 99)
            members, errors, rss = observed.process_tree_snapshot(100, 1000, proc)
            self.assertEqual([row['pid'] for row in members], [100, 101, 102])
            self.assertEqual([row['starttime_ticks'] for row in members], [1000, 1001, 1002])
            self.assertEqual(rss, 60*1024)
            self.assertEqual(errors, [])


if __name__ == '__main__':
    unittest.main()

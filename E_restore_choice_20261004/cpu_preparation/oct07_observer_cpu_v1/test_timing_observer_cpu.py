"""Synthetic Linux accounting contracts; no GPU command or remote access."""
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from timing_observer import (
    parse_proc_stat, parse_proc_status, parse_cpu_stat, parse_cpu_pressure,
    resolve_cgroup, HostCpuProbe,
)


def proc_stat(comm='decode worker (background) tail)', pid=123):
    # Values are indexed by documented /proc/PID/stat field number, not by
    # the implementation's split offsets. Distinct neighboring values expose
    # off-by-one parsing errors after the parenthesized command field.
    fields = {number: str(1000 + number) for number in range(4, 53)}
    fields.update({4: '7', 14: '12345', 15: '678', 20: '4', 22: '54321', 39: '6'})
    return f'{pid} ({comm}) R ' + ' '.join(fields[n] for n in range(4, 53)) + '\n'


class PureParserTests(unittest.TestCase):
    def test_proc_stat_command_parentheses_do_not_shift_cpu_fields(self):
        for comm in ('python', 'decode worker (background) tail)', '(()) spaced name'):
            with self.subTest(comm=comm):
                result = parse_proc_stat(proc_stat(comm))
                self.assertEqual(result['pid'], 123)
                self.assertEqual(result['comm'], comm)
                self.assertEqual(result['state'], 'R')
                self.assertEqual(result['utime_ticks'], 12345)
                self.assertEqual(result['stime_ticks'], 678)
                self.assertEqual(result['num_threads'], 4)
                self.assertEqual(result['processor'], 6)
                self.assertIs(type(result['utime_ticks']), int)

    def test_proc_status_context_switch_counts_remain_integer_counts(self):
        result = parse_proc_status('Name:\tworker\nVmRSS:\t8192 kB\n'
            'voluntary_ctxt_switches:\t123\nnonvoluntary_ctxt_switches:\t456\n')
        self.assertEqual(result['voluntary_ctxt_switches'], 123)
        self.assertEqual(result['nonvoluntary_ctxt_switches'], 456)
        self.assertIs(type(result['voluntary_ctxt_switches']), int)
        self.assertIs(type(result['nonvoluntary_ctxt_switches']), int)

    def test_cgroup_cpu_stat_retains_units_and_large_integer_precision(self):
        result = parse_cpu_stat('usage_usec 9007199254740993\nuser_usec 5000\n'
            'system_usec 6000\nnr_periods 400\nnr_throttled 8\nthrottled_usec 123456\n')
        self.assertEqual(result, dict(usage_usec=9007199254740993, user_usec=5000,
            system_usec=6000, nr_periods=400, nr_throttled=8, throttled_usec=123456))
        self.assertTrue(all(type(value) is int for value in result.values()))

    def test_pressure_averages_and_total_preserve_distinct_units(self):
        result = parse_cpu_pressure('some avg10=0.25 avg60=1.50 avg300=2.75 total=123456789\n'
            'full avg10=0.00 avg60=0.10 avg300=0.20 total=987654\n')
        self.assertEqual(result['some'], dict(avg10=0.25, avg60=1.5, avg300=2.75, total=123456789))
        self.assertEqual(result['full'], dict(avg10=0.0, avg60=0.1, avg300=0.2, total=987654))
        self.assertIs(type(result['some']['avg10']), float)
        self.assertIs(type(result['some']['total']), int)

    def test_cgroup_v2_maps_membership_relative_to_mount_root(self):
        result = resolve_cgroup('0::/tenant/jobs/worker\n',
            '31 22 0:28 /tenant /sys/fs/cgroup rw,nosuid - cgroup2 cgroup rw\n')
        self.assertEqual(result['member_path'], '/tenant/jobs/worker')
        self.assertEqual(result['version'], 'v2')
        self.assertEqual(result['mount_root'], '/tenant')
        self.assertEqual(result['mount_point'], '/sys/fs/cgroup')
        self.assertEqual(str(result['directory']), '/sys/fs/cgroup/jobs/worker')

    def test_explicit_v1_cpu_controller_owns_cpu_accounting_in_hybrid(self):
        member = '4:cpu,cpuacct:/batch/team\n'
        mount = '40 25 0:34 /batch /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu,cpuacct\n'
        v1 = resolve_cgroup(member, mount)
        self.assertEqual(v1['version'], 'v1')
        self.assertEqual(str(v1['directory']), '/sys/fs/cgroup/cpu/team')
        both = resolve_cgroup(member + '0::/unified/team\n', mount +
            '41 25 0:35 /unified /sys/fs/cgroup/unified rw - cgroup2 cgroup rw\n')
        self.assertEqual(str(both['directory']), '/sys/fs/cgroup/cpu/team')
        self.assertEqual(both['version'], 'v1')

    def test_empty_or_missing_required_accounting_fields_fail_explicitly(self):
        for parser, text in [(parse_cpu_stat, ''), (parse_cpu_pressure, ' \n'),
                (parse_proc_status, 'Name:\tworker\nvoluntary_ctxt_switches:\t123\n')]:
            with self.subTest(parser=parser.__name__):
                with self.assertRaises(ValueError):
                    parser(text)

    def test_mountinfo_escaped_spaces_are_decoded_before_mapping(self):
        result = resolve_cgroup('0::/tenant name/jobs\n',
            '31 22 0:28 /tenant\\040name /sys/fs/cgroup/my\\040mount rw - cgroup2 cgroup rw\n')
        self.assertEqual(result['mount_root'], '/tenant name')
        self.assertEqual(result['mount_point'], '/sys/fs/cgroup/my mount')
        self.assertEqual(str(result['directory']), '/sys/fs/cgroup/my mount/jobs')

    def test_cgroup_mapping_rejects_traversal_and_root_prefix_collision(self):
        for member, root in [('/tenant/../escape', '/tenant'),
                             ('/tenantB/team', '/tenant'),
                             ('/other/team', '/tenant'),
                             ('/tenant/team', '/tenant/..')]:
            with self.subTest(member=member, root=root):
                with self.assertRaises(ValueError):
                    resolve_cgroup(f'0::{member}\n',
                        f'31 22 0:28 {root} /sys/fs/cgroup rw - cgroup2 cgroup rw\n')


class HostCpuProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='observer_cpu_test_')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.proc, self.sys = self.base / 'proc', self.base / 'sys'
        self.pid = 123
        self.parent = self.proc / str(self.pid)
        self.task = self.parent / 'task' / str(self.pid)
        self.mount = self.sys / 'fs' / 'cgroup'
        self.group = self.mount / 'job'
        self.frequency = self.sys / 'devices' / 'system' / 'cpu' / 'cpu6' / 'cpufreq'
        self.gpu_call_guard = patch('timing_observer.subprocess.run',
                                   side_effect=AssertionError('CPU probe launched subprocess'))
        self.gpu_call_guard.start()
        self.addCleanup(self.gpu_call_guard.stop)

    @staticmethod
    def write(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def fixture(self):
        self.write(self.task / 'stat', proc_stat())
        self.write(self.task / 'status', 'Name:\tworker\nCpus_allowed_list:\t0-15\n'
                   'voluntary_ctxt_switches:\t123\nnonvoluntary_ctxt_switches:\t456\n')
        self.write(self.parent / 'cgroup', '0::/tenant/job\n')
        self.write(self.proc / 'self' / 'mountinfo',
                   f'31 22 0:28 /tenant {self.mount} rw - cgroup2 cgroup rw\n')
        for namespace, target in [('cgroup', 'cgroup:[123]'), ('mnt', 'mnt:[456]')]:
            for owner in (self.proc / 'self', self.parent):
                link = owner / 'ns' / namespace
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(target)
        self.write(self.group / 'cpu.stat', 'usage_usec 12000\nnr_throttled 2\nthrottled_usec 300\n')
        self.write(self.group / 'cpu.pressure',
                   'some avg10=0.25 avg60=0.50 avg300=0.75 total=1234\n'
                   'full avg10=0.00 avg60=0.10 avg300=0.20 total=40\n')
        self.write(self.frequency / 'scaling_cur_freq', '3400000\n')
        self.write(self.frequency / 'cpuinfo_cur_freq', '2800000\n')
        return HostCpuProbe(self.pid, self.proc, self.sys)

    def test_complete_fixture_keeps_tick_microsecond_and_khz_units(self):
        probe = self.fixture()
        result = probe.sample()
        self.assertEqual(result['schema'], 'E.passive_host_cpu.v1')
        self.assertEqual((result['parent_pid'], result['driver_tid']), (123, 123))
        self.assertEqual(result['driver_task_stat']['values']['utime_ticks'], 12345)
        self.assertIsNone(result['driver_task_stat']['error'])
        self.assertEqual(result['driver_task_status']['values']['cpus_allowed_list'], '0-15')
        self.assertEqual(result['cgroup']['cpu_stat']['values']['usage_usec'], 12000)
        self.assertEqual(result['cgroup']['cpu_pressure']['values']['some']['total'], 1234)
        frequency = result['current_cpu_frequency']
        self.assertEqual((frequency['cpu'], frequency['khz']), (6, 3400000))
        self.assertEqual(frequency['source_path'], str(self.frequency / 'scaling_cur_freq'))
        self.assertEqual(frequency['attempts'], [])
        self.assertGreater(result['clock_ticks_per_second'], 0)
        self.assertGreaterEqual(result['probe_wall_s'], 0)
        self.assertTrue(result['cgroup']['provenance']['mapping_cached_at_start'])
        json.dumps(result, allow_nan=False)

    def test_frequency_fallback_and_unavailable_never_use_another_cpu(self):
        probe = self.fixture()
        (self.frequency / 'scaling_cur_freq').unlink()
        result = probe.sample()['current_cpu_frequency']
        self.assertEqual(result['khz'], 2800000)
        self.assertEqual(result['source_path'], str(self.frequency / 'cpuinfo_cur_freq'))
        self.assertEqual(len(result['attempts']), 1)
        (self.frequency / 'cpuinfo_cur_freq').unlink()
        self.write(self.sys / 'devices/system/cpu/cpu0/cpufreq/scaling_cur_freq', '9999999\n')
        result = probe.sample()['current_cpu_frequency']
        self.assertIsNone(result['khz'])
        self.assertIsNone(result['source_path'])
        self.assertTrue(result['error'])
        self.assertEqual(len(result['attempts']), 2)
        self.assertTrue(all('/cpu6/' in item['path'] for item in result['attempts']))

    def test_missing_proc_and_sys_roots_return_errors_with_null_values(self):
        result = HostCpuProbe(self.pid, self.proc, self.sys).sample()
        for name in ('driver_task_stat', 'driver_task_status'):
            self.assertIsNone(result[name]['values'])
            self.assertTrue(result[name]['error'])
        self.assertIsNone(result['current_cpu_frequency']['khz'])
        self.assertTrue(result['current_cpu_frequency']['error'])
        self.assertIsNone(result['cgroup']['cpu_stat']['values'])
        self.assertTrue(result['cgroup']['cpu_stat']['error'])
        json.dumps(result, allow_nan=False)

    def test_changed_membership_suppresses_cached_and_new_cgroup_paths(self):
        probe = self.fixture()
        self.write(self.parent / 'cgroup', '0::/tenant/new-job\n')
        self.write(self.mount / 'new-job' / 'cpu.stat', 'usage_usec 99999\n')
        result = probe.sample()
        self.assertEqual(result['cgroup']['current_membership'], '0::/tenant/new-job\n')
        self.assertEqual(result['cgroup']['provenance']['initial_membership'], '0::/tenant/job\n')
        for name in ('cpu_stat', 'cpu_pressure'):
            self.assertIsNone(result['cgroup'][name]['values'])
            self.assertIsNone(result['cgroup'][name]['path'])
            self.assertIn('membership changed', result['cgroup'][name]['error'])
        self.assertIsNone(result['driver_task_stat']['error'])

    def test_namespace_mismatch_or_unreadable_namespace_suppresses_cgroup(self):
        for kind in ('different', 'missing'):
            with self.subTest(kind=kind):
                probe = self.fixture()
                link = self.parent / 'ns' / 'mnt'
                link.unlink()
                if kind == 'different':
                    link.symlink_to('mnt:[999]')
                result = probe.sample()
                self.assertIsNone(result['cgroup']['cpu_stat']['values'])
                self.assertIn('namespace', result['cgroup']['cpu_stat']['error'])
                # Reset only synthetic state before the next fixture construction.
                for owner in (self.parent, self.proc / 'self'):
                    for name in ('cgroup', 'mnt'):
                        (owner / 'ns' / name).unlink(missing_ok=True)

    def test_read_only_local_probe_is_json_serializable_when_proc_is_unavailable(self):
        result = HostCpuProbe(os.getpid()).sample()
        self.assertEqual(result['parent_pid'], os.getpid())
        self.assertGreaterEqual(result['probe_wall_s'], 0)
        for name in ('driver_task_stat', 'driver_task_status'):
            self.assertIn('values', result[name])
            self.assertIn('error', result[name])
            self.assertTrue(result[name]['values'] is not None or result[name]['error'])
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main(verbosity=2)

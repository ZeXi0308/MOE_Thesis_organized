"""Bounded synthetic contracts; no generated fixture is GPU evidence."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import analyze_cpu_tail as audit
from test_analyze_telemetry_control_cpu import fixture as telemetry_fixture, write


def sample(time_s, counter, cpu=6):
    namespaces = {k:dict(value=k+'[1]', error=None) for k in ('cgroup', 'mnt')}
    parsed = lambda values:dict(values=values, error=None)
    return dict(monotonic_s=time_s-.2, gpu=dict(returncode=0), compute_processes=dict(returncode=0),
        host_cpu=dict(schema='E.passive_host_cpu.v1', monotonic_s=time_s, probe_wall_s=.001,
            parent_pid=123, driver_tid=123, clock_ticks_per_second=100,
            driver_task_stat=parsed(dict(pid=123, starttime_ticks=1, processor=cpu,
                                        utime_ticks=counter, stime_ticks=counter)),
            driver_task_status=parsed(dict(voluntary_ctxt_switches=counter, nonvoluntary_ctxt_switches=counter)),
            current_cpu_frequency=dict(cpu=cpu, khz=3400000, source_path=f'/sys/cpu{cpu}/scaling_cur_freq', error=None),
            cgroup=dict(current_membership='0::/job\n', parent_namespaces=namespaces,
                provenance=dict(error=None, mapping=dict(version='v2', directory='/sys/fs/cgroup/job'),
                                initial_membership='0::/job\n', observer_namespaces=namespaces),
                cpu_stat=parsed(dict(usage_usec=counter, nr_periods=counter, nr_throttled=counter, throttled_usec=counter)),
                cpu_pressure=parsed({k:dict(total=counter, avg10=.1) for k in ('some', 'full')}))))


def fixture(group):
    parent = telemetry_fixture(group)
    for name in ('02_host', '03_host'):
        shutil.rmtree(parent/name)
    write(parent/'runtime_source_hashes.json', {'/frozen/timing_observer.py':audit.OBSERVER_SHA})
    for name in ('00_host', '01_host'):
        cell = parent/name
        config = json.loads((cell/'config.json').read_text())
        config.update(policies='host,host', gpu_telemetry_enabled=True, gpu_telemetry_modes='on,on')
        write(cell/'config.json', config)
        (cell/'timing_observer.jsonl').write_text('\n'.join(json.dumps(s) for s in
            [sample(101.9, 10), sample(102.9, 15, 7)])+'\n')
    return parent


class CpuTailTests(unittest.TestCase):
    def test_complete_repeat_alignment_uses_CPU_timestamp_and_preserves_units(self):
        with tempfile.TemporaryDirectory() as folder:
            group = Path(folder); fixture(group)
            result = audit.analyze_group(group)
            self.assertEqual(result['status'], 'VALIDATED_COMPLETE', result['errors'])
            self.assertTrue(result['matched_work'])
            cpu = result['cells'][0]['CPU']
            self.assertEqual(cpu['status'], 'AVAILABLE')
            self.assertEqual([s['aligned_step'] for s in cpu['samples']], [0, 1])
            self.assertEqual(cpu['tail']['observed_last_cpu_changes']['sum'], 1)
            self.assertEqual(cpu['tail']['counter_increments']['throttled_usec']['sum'], 5)
            self.assertEqual(cpu['tail']['counter_increments']['pressure_some_total_usec']['sum'], 5)
            self.assertEqual(cpu['tail']['frequency_khz']['mean'], 3400000)

    def test_missing_samples_and_resets_are_not_filled_or_bridged(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = fixture(Path(folder)); cell=parent/'00_host'
            rows = [sample(101.9, 10), sample(102.2, 8), sample(102.5, 13), sample(102.9, 18)]
            rows[2]['host_cpu']['cgroup']['cpu_stat'] = dict(values=None, error='missing file')
            (cell/'timing_observer.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            raw = json.loads((cell/'raw.json').read_text()); _, times=audit.step_observations(raw)
            report = audit.cpu_diagnostic(cell/'timing_observer.jsonl', raw, times)
            counter = report['tail']['counter_increments']['throttled_usec']
            self.assertEqual(report['status'], 'PARTIAL')
            self.assertIsNone(counter['sum'])
            self.assertEqual(counter['valid_intervals'], 0)
            self.assertIn('counter_decreased_or_reset:throttled_usec', report['adjacent_sample_intervals'][0]['errors'])
            self.assertEqual(report['samples'][2]['host_cpu_record']['cgroup']['cpu_stat']['error'], 'missing file')

    def test_identity_frequency_and_cgroup_changes_are_independent_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            parent=fixture(Path(folder)); cell=parent/'00_host'
            rows=[sample(101.9, 10),sample(102.9, 15)]
            rows[1]['host_cpu']['driver_task_stat']['values']['starttime_ticks']=2
            rows[1]['host_cpu']['current_cpu_frequency']['cpu']=99
            rows[1]['host_cpu']['cgroup']['current_membership']='0::/different\n'
            (cell/'timing_observer.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            raw=json.loads((cell/'raw.json').read_text()); _, times=audit.step_observations(raw)
            report=audit.cpu_diagnostic(cell/'timing_observer.jsonl',raw,times)
            self.assertEqual(report['status'], 'PARTIAL')
            self.assertIsNone(report['samples'][1]['frequency_khz'])
            self.assertIsNone(report['tail']['observed_last_cpu_changes']['sum'])
            self.assertTrue(all(v['sum'] is None for v in report['tail']['counter_increments'].values()))

    def test_tail_deltas_require_two_inside_endpoints_and_gaps_remain_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            parent=fixture(Path(folder)); cell=parent/'00_host'
            rows=[sample(101.7,1), sample(101.9,2), sample(103.8,3)]
            (cell/'timing_observer.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            raw=json.loads((cell/'raw.json').read_text()); _, times=audit.step_observations(raw)
            report=audit.cpu_diagnostic(cell/'timing_observer.jsonl',raw,times)
            self.assertEqual(report['tail']['samples'],1)
            self.assertEqual(report['tail']['adjacent_intervals'],0)
            self.assertIsNone(report['tail']['counter_increments']['throttled_usec']['sum'])
            self.assertTrue(report['adjacent_sample_intervals'][1]['sampling_gap_over_1_5s'])

    def test_unrun_source_identity_and_work_comparability_are_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            group=Path(folder)
            self.assertEqual(audit.analyze_group(group)['status'],'UNRUN')
            parent=fixture(group)
            p=parent/'01_host/raw.json'; raw=json.loads(p.read_text()); raw['requests'][0]['output_token_ids'][0]=99;write(p,raw)
            result=audit.analyze_group(group)
            self.assertEqual(result['status'],'VALIDATED_COMPLETE')
            self.assertFalse(result['matched_work'])
            write(parent/'runtime_source_hashes.json',{'/frozen/timing_observer.py':'b'*64})
            self.assertEqual(audit.analyze_group(group)['status'],'INVALID')

    def test_frequency_loss_alone_is_partial_and_only_all_CPU_domains_missing_is_unavailable(self):
        for all_missing in (False, True):
            with self.subTest(all_missing=all_missing), tempfile.TemporaryDirectory() as folder:
                group=Path(folder); parent=fixture(group); cell=parent/'00_host'
                rows=[sample(101.9,10),sample(102.9,15)]
                for row in rows:
                    h=row['host_cpu']
                    h['current_cpu_frequency'].update(khz=None,error='sysfs unavailable')
                    if all_missing:
                        for key in ('driver_task_stat','driver_task_status'):
                            h[key]=dict(values=None,error='proc unavailable')
                        for key in ('cpu_stat','cpu_pressure'):
                            h['cgroup'][key]=dict(values=None,error='cgroup unavailable')
                (cell/'timing_observer.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
                result=audit.analyze_group(group)
                self.assertEqual(result['status'],'VALIDATED_COMPLETE')
                self.assertTrue(result['matched_work'])
                cpu=result['cells'][0]['CPU']
                self.assertEqual(cpu['status'],'UNAVAILABLE' if all_missing else 'PARTIAL')
                self.assertEqual(cpu['valid_samples_by_field']['frequency_khz'],0)
                self.assertEqual(cpu['valid_samples_by_field']['throttled_usec'],0 if all_missing else 2)
                self.assertEqual(cpu['valid_samples_by_field']['voluntary_ctxt_switches'],0 if all_missing else 2)
                self.assertEqual(any(cpu['valid_samples_by_field'].values()),not all_missing)


if __name__ == '__main__':
    unittest.main()

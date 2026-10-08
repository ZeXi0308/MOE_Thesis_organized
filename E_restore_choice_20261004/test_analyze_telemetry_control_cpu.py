"""Synthetic contract checks only; no fixture represents a GPU experiment."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import analyze_telemetry_control as audit


def write(path, value):
    path.write_text(json.dumps(value))


def fixture(group):
    parent = group / '00_same_engine'
    parent.mkdir()
    for p in (group, parent):
        write(p / 'status.json', dict(status='COMPLETE'))
    script = '/frozen/timing_observer.py'
    write(parent / 'runtime_source_hashes.json', {script: audit.OLD_TELEMETRY_SHA})
    for i, enabled in enumerate(audit.EXPECTED_MODES):
        cell = parent / f'{i:02d}_host'
        cell.mkdir()
        rid = f'internal-random-{i}'
        config = dict(out=str(cell), policy='host', policies='host,host,host,host', threshold=1792,
                      gpu_telemetry_enabled=enabled, gpu_telemetry_modes='on,off,off,on',
                      telemetry_script=script, timing_observer=True, target_spec=None, requests=1)
        raw = dict(requests=[dict(request_id=rid, external_id='measured/E000', arrival_s=0.,
                prompt_tokens=1, max_output_tokens=2, output_token_ids=[4, 5], token_times_s=[2., 3.],
                completed=True, completion_s=3., finish_reason='stop', stop_reason=None)],
            steps=[dict(start_s=end-.2, end_s=end, process_cpu_s=.15, driver_thread_cpu_s=.12) for end in (2., 3.)],
            scheduler_steps=[dict(time_s=end-.1, free_blocks_after_schedule=99, scheduled=[dict(
                request_id=rid, count=1, start_computed=j, end_computed=j+1, known_tokens=j+1, generated_tokens=j)])
                for j, end in enumerate((2., 3.))], decisions=[], commits=[], preemptions=[], transfers=[],
            all_complete_s=3., service_and_drain_s=3., clock_alignment=dict(episode_origin_monotonic_s=100.))
        for filename, data in [('raw.json', raw), ('config.json', config), ('status.json', dict(status='COMPLETE')),
                               ('inputs.json', [dict(prompt_token_ids=[1], output_tokens=2, arrival_s=0.)]),
                               ('engine_args.json', dict(max_num_seqs=256)),
                               ('resources.json', dict(gpu_blocks=100)), ('warmup.json', dict(requests=[]))]:
            write(cell / filename, data)
        row = dict(monotonic_s=101., unix_s=1., observer_pid=1, query_wall_s=.1,
                   gpu=dict(returncode=0, stdout='GPU-example, 40', stderr=''),
                   compute_processes=dict(returncode=0, stdout='123, python', stderr=''))
        (cell / 'timing_observer.jsonl').write_text(json.dumps(row)+'\n' if enabled else '')
    return parent


class TelemetryControlTests(unittest.TestCase):
    def test_complete_control_normalizes_IDs_and_excludes_only_factor(self):
        with tempfile.TemporaryDirectory() as folder:
            group = Path(folder)
            fixture(group)
            result = audit.analyze_group(group)
            self.assertEqual(result['status'], 'VALIDATED_COMPLETE', result['errors'])
            self.assertTrue(result['matched_work'])
            self.assertEqual(len(result['pairs']), 6)
            self.assertTrue(all(p['normalized_schedule']['equal'] for p in result['pairs']))
            # Existing base fingerprint stays unchanged and observes the known toggle difference.
            pair = result['base_analysis']['greedy_token_consistency'][0]
            self.assertFalse(pair['matching_metadata']['fixed_config'])
            self.assertAlmostEqual(result['cells'][0]['timing']['all_steps']['wall_minus_driver_cpu_s']['sum'], .16)

    def test_empty_group_is_unrun_and_parent_failure_wins(self):
        with tempfile.TemporaryDirectory() as folder:
            group = Path(folder)
            self.assertEqual(audit.analyze_group(group)['status'], 'UNRUN')
            parent = fixture(group)
            write(parent / 'status.json', dict(status='FAILED'))
            result = audit.analyze_group(group)
            self.assertEqual(result['status'], 'INVALID')
            self.assertFalse(result['matched_work'])

    def test_logs_order_and_cpu_timing_cannot_be_silently_missing(self):
        cases = ['off_nonempty', 'on_empty', 'off_missing', 'wrong_order', 'CPU_missing', 'wrong_script']
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as folder:
                group = Path(folder)
                parent = fixture(group)
                if case == 'off_nonempty':
                    (parent/'01_host/timing_observer.jsonl').write_text('{}\n')
                elif case == 'on_empty':
                    (parent/'00_host/timing_observer.jsonl').write_text('')
                elif case == 'off_missing':
                    (parent/'01_host/timing_observer.jsonl').unlink()
                elif case == 'CPU_missing':
                    p = parent/'01_host/raw.json'
                    raw = json.loads(p.read_text()); raw['steps'][0].pop('driver_thread_cpu_s'); write(p, raw)
                elif case == 'wrong_order':
                    p = parent/'01_host/config.json'
                    config = json.loads(p.read_text()); config['gpu_telemetry_enabled'] = True; write(p, config)
                else:
                    write(parent/'runtime_source_hashes.json', {'/frozen/timing_observer.py': 'a'*64})
                result = audit.analyze_group(group)
                self.assertEqual(result['status'], 'INVALID', result)
                self.assertFalse(result['matched_work'])

    def test_work_changes_remain_descriptive_not_equal_work_attribution(self):
        for kind in ('output', 'schedule', 'fixed_config'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                group = Path(folder)
                parent = fixture(group)
                p = parent/'01_host'/('config.json' if kind == 'fixed_config' else 'raw.json')
                data = json.loads(p.read_text())
                if kind == 'output':
                    data['requests'][0]['output_token_ids'][0] = 99
                elif kind == 'schedule':
                    data['scheduler_steps'][0]['scheduled'][0]['known_tokens'] = 2
                else:
                    data['different_resource_setting'] = 1
                write(p, data)
                result = audit.analyze_group(group)
                self.assertEqual(result['status'], 'VALIDATED_COMPLETE', result['errors'])
                self.assertFalse(result['matched_work'])
                self.assertEqual(len(result['base_analysis']['cells']), 4)

    def test_tail_uses_request_count_and_existing_fixed_buckets(self):
        rows = [dict(step=i, scheduled_requests=n, scheduled_tokens=1,
                     wall_s=.1, process_cpu_s=.1, driver_thread_cpu_s=.1, wall_minus_driver_cpu_s=0.)
                for i, n in enumerate((64, 2, 33, 32, 8, 1))]
        result = audit.timing_summary(rows)
        self.assertEqual(result['small_batch_tail_start_step'], 3)
        self.assertEqual(result['small_batch_tail']['steps'], 3)
        self.assertEqual(result['scheduled_request_buckets']['33-128']['steps'], 2)

    def test_legal_empty_drains_are_reported_without_changing_matched_compute(self):
        with tempfile.TemporaryDirectory() as folder:
            group = Path(folder)
            parent = fixture(group)
            for i, count in enumerate((0, 1, 2, 3)):
                p = parent/f'{i:02d}_host/raw.json'
                raw = json.loads(p.read_text())
                old_trace, old_steps = audit.step_observations(raw)
                if count:
                    raw['service_and_drain_s'] = 3.5
                    raw['scheduler_steps'].extend(dict(time_s=3.5-.1*j, scheduled=[],
                        free_blocks_after_schedule=100) for j in reversed(range(count)))
                # Existing callers still unpack two values; drain never becomes a service step.
                trace, steps = audit.step_observations(raw)
                self.assertEqual((trace, steps), (old_trace, old_steps))
                write(p, raw)
            result = audit.analyze_group(group)
            self.assertEqual(result['status'], 'VALIDATED_COMPLETE', result['errors'])
            self.assertTrue(result['matched_work'])
            self.assertEqual([c['drain']['empty_schedule_count'] for c in result['cells']], [0, 1, 2, 3])
            self.assertTrue(all(c['timing']['all_steps']['steps'] == 2 for c in result['cells']))
            self.assertTrue(all(c['scheduled_native_tokens'] == 2 for c in result['base_analysis']['cells']))
            pair = result['pairs'][0]
            self.assertTrue(pair['normalized_schedule']['equal'])
            self.assertFalse(pair['drain_comparison']['empty_schedule_boundaries']['equal'])
            self.assertEqual(pair['drain_comparison']['right_empty_schedule_count'], 1)
            self.assertEqual(pair['drain_comparison']['right_minus_left_drain_wall_s'], .5)
            self.assertEqual(pair['right_minus_left_makespan_s'], 0.)

    def test_drain_exception_does_not_hide_outside_compute_missing_or_duplicate_service(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = fixture(Path(folder))
            original = json.loads((parent/'00_host/raw.json').read_text())
            original['service_and_drain_s'] = 3.5
            cases = ['before_service', 'between_steps', 'at_complete', 'before_complete',
                     'past_drain', 'drain_compute', 'missing_service', 'duplicate_service', 'bad_boundary']
            for case in cases:
                with self.subTest(case=case):
                    raw = deepcopy(original)
                    extra = dict(time_s=3.2, scheduled=[], free_blocks_after_schedule=100)
                    if case == 'before_service':
                        extra['time_s'] = 1.
                    elif case == 'between_steps':
                        extra['time_s'] = 2.5
                    elif case == 'at_complete':
                        extra['time_s'] = raw['all_complete_s']
                    elif case == 'before_complete':
                        raw['all_complete_s'], extra['time_s'] = 3.2, 3.1
                    elif case == 'past_drain':
                        extra['time_s'] = 3.5001
                    elif case == 'drain_compute':
                        extra['scheduled'] = deepcopy(raw['scheduler_steps'][-1]['scheduled'])
                    elif case == 'missing_service':
                        raw['scheduler_steps'].pop(0)
                    elif case == 'duplicate_service':
                        extra['time_s'] = 1.95
                    else:
                        raw['all_complete_s'] = 2.9
                    raw['scheduler_steps'].append(extra)
                    with self.assertRaises(ValueError):
                        audit.step_observations(raw)


if __name__ == '__main__':
    unittest.main()

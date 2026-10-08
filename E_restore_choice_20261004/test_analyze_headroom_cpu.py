"""Small synthetic schema checks; fixtures are not measured performance."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze_headroom as audit


def fixture():
    row = dict(request_id='background', known_tokens=32, computed_tokens=31)
    h = dict(schema='E.headroom.v1', block_tokens=16, current_slack_blocks=4,
        supported=True, background_next_decode_growth_blocks=1,
        target_next_decode_growth_blocks=1, margin_blocks=2,
        running_requests=[dict(row, is_prefill_chunk=False, next_decode_eligible_step=1299)],
        background_requests=[row], unsupported_background_requests=[], unsupported_background_reasons={})
    event = dict(request_id='target', event=0, eligible=True, joint_capacity=True,
        headroom=h, headroom_supported=True, scheduler_step=1300,
        known_tokens=16, free_blocks=5, full_required_blocks=1, reserved_blocks=0,
        watermark_blocks=0, running=1, host_hit_tokens=16, decision_s=1.2,
        action='recompute', policy='headroom', fallback='none',
        policy_reason='headroom_nonnegative_margin')
    raw = dict(decisions=[event], commits=[],
        steps=[dict(start_s=1., end_s=2.), dict(start_s=2.1, end_s=3.)],
        scheduler_steps=[dict(time_s=1.8, scheduled=[dict(request_id='background', count=1,
            start_computed=31, end_computed=32, known_tokens=32)]), dict(time_s=2.8, scheduled=[])],
        requests=[dict(request_id=rid, external_id=rid, prompt_tokens=n,
                       token_times_s=[2., 3.], output_token_ids=[100, 101])
                  for rid, n in [('target', 16), ('background', 32)]])
    sync_commit(raw)
    return raw


def sync_commit(raw):
    raw['commits'] = [dict(deepcopy(e), allocation_s=e['decision_s'] + .1,
        actual_action=e['action'], external_tokens=16 if e['action'] == 'host' else 0)
        for e in raw['decisions']]


def validate(raw):
    return audit.validate_episode(raw, 'headroom', 1792, 4096, 16)


def profile_fixture(group):
    """Synthetic zero-event artifacts, not a completed performance experiment."""
    name = 'multinews320_headroom8'
    profile = dict(audit.PROFILES[name])
    frozen = [dict(prompt_token_ids=[i], output_tokens=1, arrival_s=0.) for i in range(320)]
    profile['input_path'] = group / 'frozen.json'
    profile['input_path'].write_text(json.dumps(dict(source_requests=frozen)))
    profile['input_sha256'] = hashlib.sha256(profile['input_path'].read_bytes()).hexdigest()
    hashes = {suffix: 'a' * 64 for suffix in audit.RUNTIME_SOURCE_SUFFIXES}
    receipt = dict(schema='E.headroom_source_receipt.v1', profile=name,
                   input_sha256=profile['input_sha256'], expected_order=profile['order'],
                   config=dict(requests=320, max_num_seqs=320, threshold=1792), runtime_source_hashes=hashes)
    receipt_path = group / 'receipt.json'
    receipt_path.write_text(json.dumps(receipt))
    parent = group / '00_same_engine'
    parent.mkdir()
    for p in (parent, group):
        (p / 'status.json').write_text(json.dumps(dict(status='COMPLETE')))
    (parent / 'runtime_source_hashes.json').write_text(json.dumps(
        dict(hashes, **{'/frozen_workload.json': profile['input_sha256']})))
    cells = []
    for i, policy in enumerate(profile['order']):
        cell = parent / f'{i:02d}_{policy}'
        cell.mkdir()
        config = dict(requests=320, max_num_seqs=320, threshold=1792, policy=policy,
                      policies=','.join(profile['order']), natural=True, timing_observer=True,
                      target_spec=None, workload='/frozen_workload.json', kv_bytes=None,
                      host_gib=32, batch_tokens=2048)
        raw = dict(requests=[dict(request_id=f'internal{j}', external_id=f'measured/E{j:03d}',
                         prompt_tokens=1, max_output_tokens=1, arrival_s=0., token_times_s=[], output_token_ids=[])
                         for j in range(320)], decisions=[], commits=[], steps=[], scheduler_steps=[])
        for filename, data in [('raw.json', raw), ('config.json', config), ('inputs.json', frozen),
                               ('status.json', dict(status='COMPLETE')),
                               ('engine_args.json', dict(max_num_seqs=320, max_model_len=4096,
                                   gpu_memory_utilization=.9, max_num_batched_tokens=2048,
                                   kv_cache_memory_bytes=None, kv_offloading_size=32,
                                   enable_prefix_caching=False, async_scheduling=False)),
                               ('resources.json', dict(group_config=[dict(tokens_per_block=16)]))]:
            (cell / filename).write_text(json.dumps(data))
        cells.append(dict(cell=str(cell.relative_to(group)), policy=policy, threshold=1792,
                          raw_path=str(cell / 'raw.json'), committed_event_diagnostics=[]))
    return name, profile, receipt_path, dict(cells=cells, analysis_errors=[], invalid_cells=[], incomplete_cells=[])


class HeadroomAnalysisTests(unittest.TestCase):
    def invalid(self, raw, fragment):
        result = validate(raw)
        self.assertEqual(result['status'], 'INVALID', result)
        self.assertTrue(any(fragment in e for e in result['errors']), result['errors'])

    def test_old_missing_snapshot_is_unrun(self):
        self.assertEqual(validate({})['status'], 'UNRUN')
        r = fixture()
        for e in r['decisions'] + r['commits']:
            e.pop('headroom')
        self.assertEqual(validate(r)['status'], 'UNRUN')

    def test_global_step_is_not_local_index_and_input_is_unchanged(self):
        r = fixture()
        before = deepcopy(r)
        result = validate(r)
        self.assertEqual(result['status'], 'VALID', result['errors'])
        self.assertEqual(result['events'][0]['local_step'], 0)
        self.assertEqual(result['events'][0]['scheduler_step'], 1300)
        self.assertEqual(r, before)

    def test_margin_membership_native_progress_and_timing_corruption(self):
        mutations = [
            (lambda r: r['decisions'][0]['headroom'].update(margin_blocks=3), 'margin_mismatch'),
            (lambda r: r['decisions'][0]['headroom']['background_requests'].clear(), 'membership'),
            (lambda r: r['scheduler_steps'][0]['scheduled'][0].update(start_computed=30), 'native_mismatch'),
            (lambda r: r['scheduler_steps'][0].update(time_s=1.1), 'decision_after_schedule'),
            (lambda r: r['decisions'][0].update(decision_s=.9), 'decision_outside'),
        ]
        for mutate, error in mutations:
            with self.subTest(error=error):
                r = fixture()
                mutate(r)
                sync_commit(r)
                self.invalid(r, error)

    def test_unsupported_is_eligible_Host_not_fallback_and_margin_is_null(self):
        r = fixture()
        e = r['decisions'][0]
        row = dict(request_id='background', known_tokens=32, computed_tokens=30)
        e['headroom'].update(supported=False, margin_blocks=None, background_next_decode_growth_blocks=0,
            running_requests=[dict(row, is_prefill_chunk=True, next_decode_eligible_step=1299)],
            background_requests=[], unsupported_background_requests=[row],
            unsupported_background_reasons={'background': ['not_one_token_ready', 'prefill_chunk']})
        e.update(headroom_supported=False, action='host', policy_reason='headroom_background_unsupported')
        sync_commit(r)
        result = validate(r)
        self.assertEqual(result['status'], 'VALID', result['errors'])
        self.assertEqual(result['commit_summary']['unsupported'], 1)
        self.assertTrue(result['commits'][0]['predictions_disagree'])
        e['headroom']['margin_blocks'] = 0
        sync_commit(r)
        self.invalid(r, 'margin_mismatch')

    def test_cadence_prefill_and_model_flags_cannot_be_silently_ignored(self):
        for changes in [dict(next_decode_eligible_step=1301), dict(is_prefill_chunk=True),
                        dict(known_tokens=4096, computed_tokens=4095)]:
            r = fixture()
            r['decisions'][0]['headroom']['running_requests'][0].update(changes)
            sync_commit(r)
            self.invalid(r, 'ALL_scope_mismatch')

    def test_commit_link_and_actual_action_are_required(self):
        r = fixture()
        r['commits'][0]['event'] = 99
        self.invalid(r, 'orphan_commit')
        r = fixture()
        r['commits'][0]['actual_action'] = 'host'
        self.invalid(r, 'actual_action_mismatch')
        r = fixture()
        r['commits'].append(deepcopy(r['commits'][0]))
        self.invalid(r, 'duplicate_commit')

    def test_shared_output_keeps_choices_but_deduplicates_following_gap(self):
        raw = fixture()
        records = [dict(request_id='target', event=i, decision_to_next_output_s=.8-i*.1,
            next_output_s=2., next_output_token_index=0, censored=False, censor_reason=None,
            commits_sharing_next_output=2, later_commit_before_next_output=i == 0) for i in range(2)]
        result = audit.recovery_progress(raw, dict(committed_event_diagnostics=records),
            dict(commits=[dict(request_id='target', event=i) for i in range(2)]))
        self.assertEqual(result['choice_to_next_output_s']['count'], 2)
        self.assertEqual(result['subsequent_gap_s_unique_outputs']['count'], 1)
        self.assertEqual(result['subsequent_gap_s_unique_outputs']['mean'], 1.)
        self.assertEqual(result['shared_next_output_commits'], 2)

    def test_following_gap_includes_later_ineligible_native_fallback(self):
        raw = fixture()
        raw['requests'][0]['token_times_s'] = [2., 47.5]
        raw['commits'].append(dict(request_id='target', event=1, eligible=False,
            allocation_s=47.4, actual_action='recompute', fallback='host_miss_native_recompute'))
        records = [dict(request_id='target', event=0, decision_to_next_output_s=.8,
            next_output_s=2., next_output_token_index=0, censored=False, censor_reason=None,
            commits_sharing_next_output=1, later_commit_before_next_output=False),
            dict(request_id='target', event=1, next_output_s=47.5, next_output_token_index=1)]
        result = audit.recovery_progress(raw, dict(committed_event_diagnostics=records),
            dict(commits=[dict(request_id='target', event=0)]))
        self.assertEqual(result['records'][0]['subsequent_token_gap_s'], 45.5)
        self.assertEqual(result['subsequent_gap_s_unique_outputs']['count'], 1)

    def test_failed_group_without_observations_is_failure_not_unrun(self):
        with tempfile.TemporaryDirectory() as folder:
            group = Path(folder)
            status = group / 'status.json'
            base = dict(cells=[], analysis_errors=[], invalid_cells=[], incomplete_cells=[])
            with patch.object(audit.base, 'analyze_group', return_value=base):
                status.write_text(json.dumps(dict(status='FAILED')))
                failed = audit.analyze_headroom_group(group)
                self.assertEqual(failed['status'], 'INVALID')
                self.assertFalse(failed['validation_passed'])
                status.write_text(json.dumps(dict(status='COMPLETE')))
                self.assertEqual(audit.analyze_headroom_group(group)['status'], 'UNRUN')

    def test_parent_failure_cannot_validate_complete_cells(self):
        with tempfile.TemporaryDirectory() as folder:
            group = Path(folder)
            parent = group / 'same_engine'
            parent.mkdir()
            (group / 'status.json').write_text(json.dumps(dict(status='COMPLETE')))
            (parent / 'status.json').write_text(json.dumps(dict(status='FAILED')))
            cells = []
            for i, policy in enumerate(audit.EXPECTED_ARMS):
                cell = parent / str(i)
                cell.mkdir()
                raw = cell / 'raw.json'
                raw.write_text('{}')
                cells.append(dict(cell=f'same_engine/{i}', policy=policy, threshold=1792, raw_path=str(raw)))
            base = dict(cells=cells, analysis_errors=[], invalid_cells=[], incomplete_cells=[])
            with patch.object(audit.base, 'analyze_group', return_value=base), \
                 patch.object(audit, 'validate_episode', return_value=dict(status='VALID', errors=[], events=[], commits=[])), \
                 patch.object(audit, 'recovery_progress', return_value={}):
                result = audit.analyze_headroom_group(group)
            self.assertEqual(result['status'], 'INVALID')
            self.assertFalse(result['validation_passed'])
            self.assertEqual(len(result['base_analysis']['cells']), 6)

    def test_explicit_profile_complete_zero_events_is_unexercised(self):
        with tempfile.TemporaryDirectory() as folder:
            group = Path(folder)
            name, profile, receipt, service = profile_fixture(group)
            with patch.dict(audit.PROFILES, {name: profile}), patch.object(audit.base, 'analyze_group', return_value=service):
                result = audit.analyze_headroom_group(group, profile=name, source_receipt=receipt)
                self.assertEqual(result['status'], 'VALIDATED_NO_ELIGIBLE_EVENTS', result)
                self.assertTrue(result['validation_passed'])
                self.assertFalse(result['headroom_mechanism_exercised'])
                self.assertEqual(len(result['cells']), 8)
                self.assertEqual(audit.analyze_headroom_group(group)['status'], 'UNRUN')
                # Even perfectly shaped zero-event data cannot authorize its own source receipt.
                self.assertEqual(audit.analyze_headroom_group(group, profile=name)['status'], 'INVALID')
                service['cells'] = service['cells'][:-1]
                self.assertEqual(audit.analyze_headroom_group(group, profile=name, source_receipt=receipt)['status'], 'INCOMPLETE')

    def test_profile_binding_rejects_config_input_runtime_or_group_corruption(self):
        def change(path, mutate):
            data = json.loads(path.read_text())
            mutate(data)
            path.write_text(json.dumps(data))
        cases = [
            ('maxseq', '00_same_engine/00_host/config.json', lambda x: x.update(max_num_seqs=256)),
            ('requests', '00_same_engine/00_host/config.json', lambda x: x.update(requests=256)),
            ('controlled_cap', '00_same_engine/00_host/config.json', lambda x: x.update(kv_bytes=1024)),
            ('quantum', '00_same_engine/00_host/engine_args.json', lambda x: x.update(max_num_batched_tokens=1024)),
            ('GPU_budget', '00_same_engine/00_host/engine_args.json', lambda x: x.update(gpu_memory_utilization=.8)),
            ('APC', '00_same_engine/00_host/engine_args.json', lambda x: x.update(enable_prefix_caching=True)),
            ('async', '00_same_engine/00_host/engine_args.json', lambda x: x.update(async_scheduling=True)),
            ('input', '00_same_engine/00_host/inputs.json', lambda x: x[0].update(prompt_token_ids=[999])),
            ('source', '00_same_engine/runtime_source_hashes.json', lambda x: x.update({'/selector.py': 'b' * 64})),
            ('receipt', 'receipt.json', lambda x: x['runtime_source_hashes'].pop('/timing_observer.py')),
            ('failed', 'status.json', lambda x: x.update(status='FAILED')),
            ('missing_snapshot', '00_same_engine/00_host/raw.json', lambda x: x['decisions'].append(
                dict(request_id='internal0', event=0, eligible=True, headroom=None, headroom_supported=None))),
        ]
        for name, path, mutate in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as folder:
                group = Path(folder)
                profile_name, profile, receipt, service = profile_fixture(group)
                change(group / path, mutate)
                with patch.dict(audit.PROFILES, {profile_name: profile}), patch.object(audit.base, 'analyze_group', return_value=service):
                    result = audit.analyze_headroom_group(group, profile=profile_name, source_receipt=receipt)
                self.assertEqual(result['status'], 'INVALID')
                self.assertFalse(result['validation_passed'])


if __name__ == '__main__':
    unittest.main()

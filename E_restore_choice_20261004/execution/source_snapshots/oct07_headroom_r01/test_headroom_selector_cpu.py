"""CPU-only headroom adapter contracts; no scheduler replica or GPU imports."""
from copy import deepcopy
from enum import Enum
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from selector import RestoreSelector


class Policy(Enum):
    FCFS = 'fcfs'
    PRIORITY = 'priority'


class CPUOffloadingManager:
    pass


class NoFutureSampling:
    def __getattr__(self, name):
        raise AssertionError('selector read sampling/future field: ' + name)


class HitStatus:
    def __init__(self):
        self.updates = []

    def update_num_hit_chunks(self, tokens):
        self.updates.append(tokens)


class Connector:
    def __init__(self):
        self.config = NS(kv_group_configs=[NS(tokens_per_block=16,
                        sliding_window_size_in_chunks=None)])
        self.manager = CPUOffloadingManager()
        self._jobs, self._req_status = {}, {}
        self.native_hit = (16, True)

    def get_num_new_matched_tokens(self, request, local):
        self._req_status.setdefault(request.request_id, HitStatus())
        return self.native_hit

    def update_state_after_alloc(self, *args):
        return 'allocated'

    def update_connector_output(self, output):
        return output


class Scheduler:
    def __init__(self):
        self.connector = NS(connector_scheduler=Connector())
        self.free = 1000
        self.reserved = 3
        self.kv_cache_manager = NS(enable_caching=False, watermark_blocks=2,
            block_pool=NS(get_num_free_blocks=lambda: self.free))
        self.scheduler_reserve_full_isl = True
        self.policy = Policy.FCFS
        self.scheduler_config = NS(async_scheduling=False, long_prefill_token_threshold=0)
        self.parallel_config = NS(pipeline_parallel_size=1,
            decode_context_parallel_size=1, prefill_context_parallel_size=1,
            data_parallel_size=1)
        self.vllm_config = NS(speculative_config=None, parallel_config=self.parallel_config)
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        self.num_sampled_tokens_per_step = 1
        self.dynamic_sd_lookup = None
        self.use_eagle = self.is_encoder_decoder = self.has_mamba_layers = False
        self.need_mamba_block_aligned_split = self.use_pp = False
        self.max_num_encoder_input_tokens = 0
        self.ec_connector = self.lora_config = None
        self.dcp_world_size = self.pcp_world_size = 1
        self.block_size = 16
        self.max_model_len = 4096
        self.current_step = 37
        self.running, self.waiting = [], []

    def _inflight_prefill_reserved_blocks(self):
        return self.reserved

    def _preempt_request(self, request, timestamp):
        return None


def request(rid='target', known=706, computed=0, preemptions=1):
    return NS(request_id=rid, num_tokens=known, num_computed_tokens=computed,
        num_preemptions=preemptions, num_prompt_tokens=known - 1,
        num_output_tokens=1, all_token_ids=list(range(known)),
        sampling_params=NoFutureSampling(), status='PREEMPTED', stop_reason=None,
        has_encoder_inputs=False, lora_request=None, spec_token_ids=[],
        num_output_placeholders=0, num_tokens_with_spec=known,
        num_in_flight_tokens=0, next_decode_eligible_step=0, is_prefill_chunk=False)


class HeadroomSelectorTests(unittest.TestCase):
    def make(self, known=706, slack=7, bg=10, policy='headroom', threshold=1792):
        scheduler = Scheduler()
        scheduler.running = [request('bg' + str(i), known=32, computed=31,
                                     preemptions=0) for i in range(bg)]
        target = request(known=known)
        watermark = scheduler.kv_cache_manager.watermark_blocks if scheduler.running else 0
        scheduler.free = (known + 15) // 16 + scheduler.reserved + watermark + slack
        selector = RestoreSelector(scheduler, policy, threshold=threshold)
        return scheduler, selector, target

    def choose(self, scheduler, selector, target):
        value = scheduler.connector.connector_scheduler.get_num_new_matched_tokens(target, 0)
        return value, selector.events[-1]

    def test_requested_low_and_high_slack_examples(self):
        for known, slack, expected, margin in [(706, 7, 'host', -3),
                                               (719, 41, 'recompute', 31)]:
            with self.subTest(known=known):
                scheduler, selector, target = self.make(known=known, slack=slack)
                result, event = self.choose(scheduler, selector, target)
                feature = event['headroom']
                self.assertEqual(event['action'], expected)
                self.assertEqual(result, (0, False) if expected == 'recompute' else (16, True))
                self.assertEqual(feature['schema'], 'E.headroom.v1')
                self.assertEqual(feature['block_tokens'], 16)
                self.assertEqual(feature['current_slack_blocks'], slack)
                self.assertEqual(feature['background_next_decode_growth_blocks'], 10)
                self.assertEqual(feature['target_next_decode_growth_blocks'], 0)
                self.assertEqual(feature['margin_blocks'], margin)
                self.assertEqual(event['scheduler_step'], 37)

    def test_zero_margin_and_length_threshold_are_inclusive(self):
        for known, slack, expected in [(1792, 4, 'recompute'), (1793, 4, 'host')]:
            with self.subTest(known=known):
                scheduler, selector, target = self.make(known=known, slack=slack, bg=3)
                _, event = self.choose(scheduler, selector, target)
                self.assertEqual(event['action'], expected)
                self.assertEqual(event['headroom']['margin_blocks'], 0 if known == 1792 else 1)

    def test_target_block_boundary_requires_one_growth_block(self):
        for known, expected_growth, expected_action in [(15, 0, 'recompute'), (16, 1, 'host')]:
            with self.subTest(known=known):
                scheduler, selector, target = self.make(known=known, slack=0, bg=0)
                _, event = self.choose(scheduler, selector, target)
                self.assertEqual(event['headroom']['target_next_decode_growth_blocks'], expected_growth)
                self.assertEqual(event['headroom']['margin_blocks'], -expected_growth)
                self.assertEqual(event['action'], expected_action)

    def test_only_boundary_decode_backgrounds_contribute_current_progress(self):
        scheduler, selector, target = self.make(bg=1, slack=7)
        boundary = scheduler.running[0]
        scheduler.running.append(request('nonboundary', 33, 32, 0))
        before = [(r.request_id, r.num_tokens, r.num_computed_tokens) for r in scheduler.running]
        _, event = self.choose(scheduler, selector, target)
        self.assertEqual(event['headroom']['background_next_decode_growth_blocks'], 1)
        self.assertTrue(event['headroom']['supported'])
        self.assertTrue(event['headroom_supported'])
        self.assertEqual(event['headroom']['background_requests'],
                         [dict(request_id=boundary.request_id, known_tokens=32, computed_tokens=31)])
        self.assertEqual([(r.request_id, r.num_tokens, r.num_computed_tokens)
                          for r in scheduler.running], before)
        self.assertEqual(target.num_computed_tokens, 0)

    def test_unsupported_background_cadence_chooses_host_only_for_headroom(self):
        features = []
        for policy, action in [('host', 'host'), ('recompute', 'recompute'),
                               ('length', 'recompute'), ('headroom', 'host')]:
            with self.subTest(policy=policy):
                scheduler, selector, target = self.make(bg=1, slack=41, policy=policy)
                scheduler.running.extend([request('prefill', 64, 20, 0),
                                          request('alreadycomputed', 32, 32, 0)])
                _, event = self.choose(scheduler, selector, target)
                self.assertTrue(event['eligible'])
                self.assertEqual(event['fallback'], 'none')
                self.assertEqual(event['action'], action)
                self.assertFalse(event['headroom_supported'])
                self.assertFalse(event['headroom']['supported'])
                self.assertIsNone(event['headroom']['margin_blocks'])
                self.assertEqual([{k: row[k] for k in ('request_id', 'known_tokens', 'computed_tokens')}
                                  for row in event['headroom']['unsupported_background_requests']], [
                    dict(request_id='prefill', known_tokens=64, computed_tokens=20),
                    dict(request_id='alreadycomputed', known_tokens=32, computed_tokens=32)])
                if policy == 'headroom':
                    self.assertEqual(event['policy_reason'], 'headroom_background_unsupported')
                features.append(event['headroom'])
        self.assertTrue(all(feature == features[0] for feature in features))

    def test_deferred_decode_or_model_length_limit_is_unsupported(self):
        for attr, value in [('next_decode_eligible_step', 38), ('num_tokens', 4096)]:
            with self.subTest(attr=attr):
                scheduler, selector, target = self.make(bg=1, slack=41)
                background = scheduler.running[0]
                setattr(background, attr, value)
                if attr == 'num_tokens':
                    background.num_computed_tokens = value - 1
                    background.num_tokens_with_spec = value
                _, event = self.choose(scheduler, selector, target)
                self.assertFalse(event['headroom_supported'])
                self.assertEqual(event['action'], 'host')
                self.assertEqual(event['policy_reason'], 'headroom_background_unsupported')

    def test_running_prefill_chunk_is_unsupported_but_waiting_target_flag_is_stale(self):
        scheduler, selector, target = self.make(bg=1, slack=41)
        target.is_prefill_chunk = True
        _, event = self.choose(scheduler, selector, target)
        self.assertTrue(event['headroom_supported'])
        self.assertEqual(event['action'], 'recompute')
        scheduler.running[0].is_prefill_chunk = True
        _, event = self.choose(scheduler, selector, target)
        self.assertFalse(event['headroom_supported'])
        self.assertEqual(event['action'], 'host')
        self.assertEqual(event['policy_reason'], 'headroom_background_unsupported')
        self.assertEqual(event['headroom']['unsupported_background_reasons'], {'bg0': ['prefill_chunk']})
        self.assertTrue(event['headroom']['running_requests'][0]['is_prefill_chunk'])

    def test_fixed_policy_actions_preserved_with_equal_feature_observation(self):
        features = []
        for policy, action in [('host', 'host'), ('recompute', 'recompute'),
                               ('length', 'recompute'), ('headroom', 'host')]:
            scheduler, selector, target = self.make(policy=policy)
            _, event = self.choose(scheduler, selector, target)
            self.assertEqual(event['action'], action)
            features.append(event['headroom'])
        self.assertTrue(all(feature == features[0] for feature in features))

    def test_ineligible_paths_keep_native_fallback_without_scanning(self):
        class NeverScan(list):
            def __iter__(self):
                raise AssertionError('ineligible lookup scanned running requests')
        for native, slack, reason in [((None, False), 7, 'pending_native'),
                ((0, False), 7, 'host_miss_native_recompute'),
                ((16, True), -1, 'full_capacity_not_jointly_available_native')]:
            for policy in ('host', 'recompute', 'length', 'headroom'):
                with self.subTest(native=native, policy=policy):
                    scheduler, selector, target = self.make(policy=policy, slack=slack)
                    scheduler.connector.connector_scheduler.native_hit = native
                    scheduler.running = NeverScan(scheduler.running)
                    result, event = self.choose(scheduler, selector, target)
                    self.assertIs(result, native)
                    self.assertEqual(event['fallback'], reason)
                    self.assertIsNone(event['headroom'])
                    self.assertIsNone(event['headroom_supported'])

    def test_commit_feature_is_a_snapshot_of_lookup_state(self):
        scheduler, selector, target = self.make()
        _, event = self.choose(scheduler, selector, target)
        saved = deepcopy(event['headroom'])
        value = scheduler.connector.connector_scheduler.update_state_after_alloc(
            target, NS(blocks=[[1, 2]]), 16)
        self.assertEqual(value, 'allocated')
        scheduler.running[0].num_computed_tokens = 999
        scheduler.running[0].request_id = 'mutated'
        scheduler.running.clear()
        scheduler.free += 100
        scheduler.current_step += 1
        self.choose(scheduler, selector, request('later', 719))
        self.assertEqual(event['headroom'], saved)
        self.assertEqual(selector.commits[0]['headroom'], saved)
        self.assertEqual(selector.commits[0]['scheduler_step'], 37)
        commit = selector.commits[0]
        selector.clear()
        self.assertEqual(event['headroom'], saved)
        self.assertEqual(commit['headroom'], saved)

    def test_disabled_and_never_preempted_requests_stay_native(self):
        scheduler, selector, target = self.make()
        target.num_preemptions = 0
        self.assertEqual(selector.lookup(target, 0), (16, True))
        target.num_preemptions = 1
        selector.enabled = False
        self.assertEqual(selector.lookup(target, 0), (16, True))
        self.assertEqual(selector.events, [])

    def test_retired_budget_policy_is_rejected(self):
        with self.assertRaises((AssertionError, ValueError)):
            RestoreSelector(Scheduler(), 'budget')

    def test_frozen_one_event_target_remains_limited_to_host_or_recompute(self):
        target_spec = dict(schema='E.one_event_target.v1', target=dict(
            external_id='frozen/E001', num_preemptions=1, known_tokens=706,
            generated_tokens=1, prefix_sha256='a' * 64))
        for policy in ('host', 'recompute'):
            with self.subTest(policy=policy):
                selector = RestoreSelector(Scheduler(), policy, target_spec=target_spec)
                self.assertEqual(selector.target_spec, target_spec)
        for policy in ('length', 'headroom', 'budget'):
            with self.subTest(policy=policy):
                with self.assertRaises(AssertionError):
                    RestoreSelector(Scheduler(), policy, threshold=1792,
                                    target_spec=target_spec)

    def test_unsupported_execution_modes_fail_before_choice(self):
        for attr, value in [('policy', Policy.PRIORITY), ('num_spec_tokens', 1),
                ('use_pp', True), ('dcp_world_size', 2), ('pcp_world_size', 2),
                ('is_encoder_decoder', True), ('has_mamba_layers', True)]:
            with self.subTest(attr=attr):
                scheduler = Scheduler()
                setattr(scheduler, attr, value)
                with self.assertRaises(AssertionError):
                    RestoreSelector(scheduler, 'headroom', threshold=1792)
        scheduler = Scheduler()
        scheduler.parallel_config.data_parallel_size = 2
        with self.assertRaises(AssertionError):
            RestoreSelector(scheduler, 'headroom', threshold=1792)

    def test_unsupported_request_state_rejected_for_target_and_background(self):
        for which in ('target', 'background'):
            for attr, value in [('has_encoder_inputs', True), ('lora_request', object()),
                    ('spec_token_ids', [42]), ('num_output_placeholders', 1),
                    ('num_in_flight_tokens', 1)]:
                with self.subTest(which=which, attr=attr):
                    scheduler, selector, target = self.make(bg=1)
                    changed = target if which == 'target' else scheduler.running[0]
                    setattr(changed, attr, value)
                    with self.assertRaises(AssertionError):
                        self.choose(scheduler, selector, target)


if __name__ == '__main__':
    unittest.main(verbosity=2)

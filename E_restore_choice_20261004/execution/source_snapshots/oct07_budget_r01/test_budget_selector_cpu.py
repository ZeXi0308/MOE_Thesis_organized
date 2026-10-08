"""CPU-only contract tests for the thin native-scheduler budget observer.

These callbacks exercise the adapter boundary; they do not emulate vLLM's
scheduling algorithm.  Run with Python's unittest discovery or this file.
No vLLM, Torch, model, remote host, or GPU is needed.
"""
import enum
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from selector import RestoreSelector


class Policy(enum.Enum):
    FCFS = "fcfs"
    PRIORITY = "priority"


class Pause(enum.Enum):
    UNPAUSED = 0
    PAUSED_NEW = 1
    PAUSED_ALL = 2


class CPUOffloadingManager:
    pass


class HitStatus:
    def __init__(self):
        self.updates = []

    def update_num_hit_chunks(self, value):
        self.updates.append(value)


class MockConnector:
    def __init__(self):
        self.config = NS(kv_group_configs=[NS(
            tokens_per_block=4, sliding_window_size_in_chunks=None)])
        self.manager = CPUOffloadingManager()
        self._jobs = {}
        self._req_status = {}
        self.native_hit = (4, True)
        self.lookup_calls = []

    def get_num_new_matched_tokens(self, request, local_tokens):
        self.lookup_calls.append((request, local_tokens))
        self._req_status.setdefault(request.request_id, HitStatus())
        return self.native_hit

    def update_state_after_alloc(self, *args, **kwargs):
        return None

    def update_connector_output(self, *args, **kwargs):
        return None


class MockCacheManager:
    def __init__(self):
        self.enable_caching = False
        self.watermark_blocks = 0
        self.free_blocks = 100
        self.block_pool = NS(get_num_free_blocks=lambda: self.free_blocks)
        self.outcomes = []
        self.calls = []

    def allocate_slots(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.outcomes.pop(0) if self.outcomes else object()


class MockScheduler:
    def __init__(self, quantum=8):
        self.connector = NS(connector_scheduler=MockConnector())
        self.kv_cache_manager = MockCacheManager()
        self.scheduler_reserve_full_isl = True
        self.policy = Policy.FCFS
        self.max_num_scheduled_tokens = quantum
        self.scheduler_config = NS(long_prefill_token_threshold=0,
                                   async_scheduling=False)
        self.vllm_config = NS(speculative_config=None)
        self.num_spec_tokens = 0
        self.num_sampled_tokens_per_step = 1
        self.num_lookahead_tokens = 0
        self.dynamic_sd_lookup = None
        self.use_eagle = False
        self.is_encoder_decoder = False
        self.max_num_encoder_input_tokens = 0
        self.ec_connector = None
        self.lora_config = None
        self.has_mamba_layers = False
        self.need_mamba_block_aligned_split = False
        self._pause_state = Pause.UNPAUSED
        self.running = []
        self.waiting = []
        self.schedule_calls = []
        self.preempt_calls = []
        self.native_schedule = lambda *args, **kwargs: output()

    def _inflight_prefill_reserved_blocks(self):
        return 0

    def schedule(self, *args, **kwargs):
        self.schedule_calls.append((args, kwargs))
        return self.native_schedule(*args, **kwargs)

    def _preempt_request(self, request, timestamp):
        self.preempt_calls.append((request, timestamp))
        return "native-preempt-result"


def request(rid="restore", known=5, preemptions=1):
    return NS(request_id=rid, num_tokens=known, num_preemptions=preemptions,
              num_prompt_tokens=known - 1, num_output_tokens=1,
              num_computed_tokens=0, all_token_ids=list(range(known)),
              sampling_params=object(), status="PREEMPTED", stop_reason=None,
              has_encoder_inputs=False, lora_request=None)


def output(tokens=None):
    tokens = {} if tokens is None else tokens
    return NS(num_scheduled_tokens=tokens,
              total_num_scheduled_tokens=sum(tokens.values()))


class BudgetSelectorTests(unittest.TestCase):
    def make(self, policy="budget", quantum=8, threshold=5):
        scheduler = MockScheduler(quantum)
        selector = RestoreSelector(scheduler, policy, threshold=threshold)
        return scheduler, selector

    def test_success_failed_and_zero_load_allocations(self):
        scheduler, selector = self.make()
        cache = scheduler.kv_cache_manager
        running, failed, load = request("running"), request("failed"), request("load")
        successful = object()
        cache.outcomes = [successful, None, False]

        def native():
            self.assertIs(cache.allocate_slots(running, 3), successful)
            self.assertEqual(selector._budget_context["remaining"], 5)
            self.assertIsNone(cache.allocate_slots(failed, 2))
            self.assertEqual(selector._budget_context["remaining"], 5)
            # False is a successful non-None result, even for a zero-token LOAD.
            self.assertIs(cache.allocate_slots(load, 0), False)
            self.assertEqual(selector._budget_context["positive_allocations"], {"running": 3})
            return output({"running": 3})

        scheduler.native_schedule = native
        scheduler.schedule()
        self.assertIsNone(selector._budget_context)
        self.assertEqual(len(selector.budget_steps), 1)
        self.assertIsInstance(selector.budget_summary(), dict)

    def test_false_result_still_debits_positive_allocation(self):
        scheduler, selector = self.make()
        scheduler.kv_cache_manager.outcomes = [False]

        def native():
            self.assertIs(scheduler.kv_cache_manager.allocate_slots(request("r"), 2), False)
            self.assertEqual(selector._budget_context["remaining"], 6)
            return output({"r": 2})

        scheduler.native_schedule = native
        scheduler.schedule()

    def test_budget_decision_uses_prior_compute_and_inclusive_boundary(self):
        for known, expected in [(5, "recompute"), (6, "host")]:
            with self.subTest(known=known):
                scheduler, selector = self.make()
                target = request(known=known)

                def native():
                    scheduler.kv_cache_manager.allocate_slots(request("running"), 3)
                    result = scheduler.connector.connector_scheduler.get_num_new_matched_tokens(target, 0)
                    self.assertEqual(result, (0, False) if expected == "recompute" else (4, True))
                    self.assertEqual(selector.events[-1]["action"], expected)
                    self.assertEqual(selector.events[-1]["remaining_budget_tokens"], 5)
                    self.assertIsInstance(selector.events[-1]["budget_step"], int)
                    return output({"running": 3})

                scheduler.native_schedule = native
                scheduler.schedule()

    def test_failed_allocation_does_not_change_later_decision(self):
        scheduler, selector = self.make(quantum=5)
        scheduler.kv_cache_manager.outcomes = [None]

        def native():
            scheduler.kv_cache_manager.allocate_slots(request("failed"), 4)
            result = scheduler.connector.connector_scheduler.get_num_new_matched_tokens(request(known=5), 0)
            self.assertEqual(result, (0, False))
            return output()

        scheduler.native_schedule = native
        scheduler.schedule()

    def test_fixed_policies_keep_their_decisions(self):
        for policy, known, expected in [
                ("host", 4, "host"), ("recompute", 7, "recompute"),
                ("length", 5, "recompute"), ("length", 6, "host")]:
            with self.subTest(policy=policy, known=known):
                scheduler, selector = self.make(policy=policy, quantum=1, threshold=5)

                def native():
                    scheduler.connector.connector_scheduler.get_num_new_matched_tokens(request(known=known), 0)
                    self.assertEqual(selector.events[-1]["action"], expected)
                    return output()

                scheduler.native_schedule = native
                scheduler.schedule()

    def test_ineligible_lookup_retains_identical_native_fallback(self):
        for native_hit, free, fallback in [
                ((None, False), 100, "pending_native"),
                ((0, False), 100, "host_miss_native_recompute"),
                ((4, True), 0, "full_capacity_not_jointly_available_native")]:
            for policy in ("host", "recompute", "length", "budget"):
                with self.subTest(hit=native_hit, free=free, policy=policy):
                    scheduler, selector = self.make(policy=policy)
                    connector = scheduler.connector.connector_scheduler
                    connector.native_hit = native_hit
                    scheduler.kv_cache_manager.free_blocks = free

                    def native():
                        self.assertIs(connector.get_num_new_matched_tokens(request(), 0), native_hit)
                        self.assertEqual(selector.events[-1]["fallback"], fallback)
                        self.assertFalse(selector.events[-1]["eligible"])
                        return output()

                    scheduler.native_schedule = native
                    scheduler.schedule()

    def test_output_mismatch_rejected_even_when_totals_match(self):
        for reported in ({"other": 3}, {"r": 2}, {}):
            with self.subTest(reported=reported):
                scheduler, selector = self.make()

                def native():
                    scheduler.kv_cache_manager.allocate_slots(request("r"), 3)
                    return output(reported)

                scheduler.native_schedule = native
                with self.assertRaises(AssertionError):
                    scheduler.schedule()
                self.assertIsNone(selector._budget_context)

    def test_exception_clears_context_and_next_schedule_can_run(self):
        scheduler, selector = self.make()

        def native():
            scheduler.kv_cache_manager.allocate_slots(request("r"), 3)
            raise RuntimeError("native failure")

        scheduler.native_schedule = native
        with self.assertRaisesRegex(RuntimeError, "native failure"):
            scheduler.schedule()
        self.assertIsNone(selector._budget_context)
        scheduler.native_schedule = lambda: output()
        scheduler.schedule()
        self.assertIsNone(selector._budget_context)

    def test_inconsistent_native_total_is_rejected(self):
        scheduler, selector = self.make()

        def native():
            scheduler.kv_cache_manager.allocate_slots(request("r"), 2)
            result = output({"r": 2})
            result.total_num_scheduled_tokens = 3
            return result

        scheduler.native_schedule = native
        with self.assertRaises(AssertionError):
            scheduler.schedule()
        self.assertIsNone(selector._budget_context)
        self.assertEqual(selector.budget_steps[-1]["status"], "ERROR")

    def test_summary_snapshot_survives_mutation_and_clear(self):
        scheduler, selector = self.make()
        scheduler.schedule()
        saved = selector.target_summary()["budget_observation"]
        self.assertEqual(saved["checked_steps"], 1)
        self.assertEqual(saved["error_steps"], 0)
        selector.budget_steps[0]["status"] = "changed-after-snapshot"
        self.assertEqual(saved["steps"][0]["status"], "CHECKED")
        selector.clear()
        self.assertEqual(selector.budget_steps, [])
        self.assertEqual(len(saved["steps"]), 1)
        self.assertEqual(saved["steps"][0]["initial_budget_tokens"], 8)
        self.assertEqual(saved["steps"][0]["remaining_budget_tokens"], 8)
        self.assertEqual(selector.budget_summary()["checked_steps"], 0)

    def test_error_summary_snapshot_is_deeply_independent(self):
        scheduler, selector = self.make()

        def native():
            scheduler.kv_cache_manager.allocate_slots(request("r"), 2)
            return output({"wrong-id": 2})

        scheduler.native_schedule = native
        with self.assertRaises(AssertionError):
            scheduler.schedule()
        saved = selector.budget_summary()
        self.assertEqual(saved["error_steps"], 1)
        selector.budget_steps[0]["observed_allocations"]["r"] = 100
        selector.budget_steps[0]["native_scheduled_tokens"]["wrong-id"] = 100
        self.assertEqual(saved["steps"][0]["observed_allocations"], {"r": 2})
        self.assertEqual(saved["steps"][0]["native_scheduled_tokens"], {"wrong-id": 2})

    def test_reentrant_schedule_rejected_and_context_cleared(self):
        scheduler, selector = self.make()
        scheduler.native_schedule = lambda: scheduler.schedule()
        with self.assertRaises(AssertionError):
            scheduler.schedule()
        self.assertIsNone(selector._budget_context)

    def test_recovery_lookup_requires_active_schedule(self):
        scheduler, selector = self.make()
        with self.assertRaises(AssertionError):
            scheduler.connector.connector_scheduler.get_num_new_matched_tokens(request(), 0)
        self.assertIsNone(selector._budget_context)

    def test_disabled_and_new_request_lookup_remain_native(self):
        scheduler, selector = self.make()
        connector = scheduler.connector.connector_scheduler
        self.assertEqual(connector.get_num_new_matched_tokens(request(preemptions=0), 0), (4, True))
        selector.enabled = False
        self.assertEqual(connector.get_num_new_matched_tokens(request(), 0), (4, True))
        self.assertEqual(selector.events, [])

    def test_fcfs_preempt_cannot_refund_a_charged_request(self):
        scheduler, selector = self.make()
        target = request("r")

        def native():
            scheduler.kv_cache_manager.allocate_slots(target, 2)
            scheduler._preempt_request(target, 1.25)
            return output({"r": 2})

        scheduler.native_schedule = native
        with self.assertRaises(AssertionError):
            scheduler.schedule()
        self.assertEqual(scheduler.preempt_calls, [])
        self.assertIsNone(selector._budget_context)

    def test_uncharged_preempt_is_forwarded(self):
        scheduler, selector = self.make()
        target = request("victim")

        def native():
            self.assertEqual(scheduler._preempt_request(target, 1.25), "native-preempt-result")
            return output()

        scheduler.native_schedule = native
        scheduler.schedule()
        self.assertEqual(scheduler.preempt_calls, [(target, 1.25)])

    def test_pause_state_sets_correct_initial_budget(self):
        for pause, expected in [(Pause.UNPAUSED, 8), (Pause.PAUSED_NEW, 8), (Pause.PAUSED_ALL, 0)]:
            with self.subTest(pause=pause):
                scheduler, selector = self.make()
                scheduler._pause_state = pause

                def native():
                    self.assertEqual(selector._budget_context["initial"], expected)
                    self.assertEqual(selector._budget_context["remaining"], expected)
                    return output()

                scheduler.native_schedule = native
                scheduler.schedule()

    def test_unknown_pause_state_rejected(self):
        scheduler, selector = self.make()
        scheduler._pause_state = NS(name="UNKNOWN")
        with self.assertRaises(AssertionError):
            scheduler.schedule()
        self.assertIsNone(selector._budget_context)

    def test_callbacks_preserve_args_kwargs_and_native_result(self):
        scheduler, selector = self.make()
        target = request("r")
        native_output = output({"r": 2})
        marker = object()

        def native(*args, **kwargs):
            self.assertEqual(args, (marker,))
            self.assertEqual(kwargs, {"flag": marker})
            scheduler.kv_cache_manager.allocate_slots(request=target, num_new_tokens=2,
                                                      num_lookahead_tokens=0, marker=marker)
            return native_output

        scheduler.native_schedule = native
        self.assertIs(scheduler.schedule(marker, flag=marker), native_output)
        self.assertEqual(scheduler.kv_cache_manager.calls, [
            ((), {"request": target, "num_new_tokens": 2,
                  "num_lookahead_tokens": 0, "marker": marker})])

    def test_unsupported_constructor_modes_rejected(self):
        modes = [
            ("policy", Policy.PRIORITY), ("num_spec_tokens", 1),
            ("num_lookahead_tokens", 1), ("dynamic_sd_lookup", [1]),
            ("use_eagle", True), ("is_encoder_decoder", True),
            ("max_num_encoder_input_tokens", 1), ("ec_connector", object()),
            ("lora_config", object()), ("has_mamba_layers", True),
            ("num_sampled_tokens_per_step", 0), ("need_mamba_block_aligned_split", True),
            ("max_num_scheduled_tokens", 0), ("max_num_scheduled_tokens", -1),
        ]
        for attr, value in modes:
            with self.subTest(attr=attr, value=value):
                scheduler = MockScheduler()
                setattr(scheduler, attr, value)
                with self.assertRaises(AssertionError):
                    RestoreSelector(scheduler, "budget")
        for target, attr, value in [
                ("vllm_config", "speculative_config", object()),
                ("scheduler_config", "long_prefill_token_threshold", 4),
                ("scheduler_config", "async_scheduling", True),
                ("kv_cache_manager", "enable_caching", True)]:
            with self.subTest(target=target, attr=attr):
                scheduler = MockScheduler()
                setattr(getattr(scheduler, target), attr, value)
                with self.assertRaises(AssertionError):
                    RestoreSelector(scheduler, "budget")

    def test_constructor_rejection_leaves_native_hooks_intact(self):
        scheduler = MockScheduler()
        scheduler.policy = Policy.PRIORITY
        connector = scheduler.connector.connector_scheduler
        originals = [(scheduler, "schedule"), (scheduler, "_preempt_request"),
                     (scheduler.kv_cache_manager, "allocate_slots"),
                     (connector, "get_num_new_matched_tokens"),
                     (connector, "update_state_after_alloc"),
                     (connector, "update_connector_output")]
        saved = [getattr(obj, name) for obj, name in originals]
        with self.assertRaises(AssertionError):
            RestoreSelector(scheduler, "budget")
        self.assertEqual(saved, [getattr(obj, name) for obj, name in originals])

    def test_scope_change_is_rejected_before_native_schedule(self):
        scheduler, selector = self.make()
        scheduler.num_lookahead_tokens = 1
        with self.assertRaises(AssertionError):
            scheduler.schedule()
        self.assertEqual(scheduler.schedule_calls, [])
        self.assertIsNone(selector._budget_context)


if __name__ == "__main__":
    unittest.main(verbosity=2)

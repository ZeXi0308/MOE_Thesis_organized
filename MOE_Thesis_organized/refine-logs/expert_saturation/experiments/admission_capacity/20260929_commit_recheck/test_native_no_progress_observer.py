"""CPU fixtures for the bounded post-LTR diagnostic observer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("native_no_progress_observer.py")
spec = importlib.util.spec_from_file_location("native_no_progress_observer", MODULE_PATH)
observer = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(observer)


class FakePool:
    def __init__(self) -> None:
        self.free = 4

    def get_num_free_blocks(self) -> int:
        return self.free


class FakeManager:
    def __init__(self) -> None:
        self.block_pool = FakePool()
        self.coordinator = FakeCoordinator()
        self.calls = []
        self.return_value = None
        self.probe_blocks = False

    def allocate_slots(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.probe_blocks:
            self.coordinator.get_num_blocks_to_allocate(
                request_id=args[0].request_id, num_tokens=args[1],
                apply_admission_cap=True)
            self.coordinator.get_num_blocks_to_allocate(
                request_id=args[0].request_id, num_tokens=args[1])
        return self.return_value


class FakeCoordinator:
    def __init__(self) -> None:
        self.calls = []

    def get_num_blocks_to_allocate(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return 12 if kwargs.get("apply_admission_cap") else 3


class OffloadingConnector:
    def __init__(self) -> None:
        self.connector_scheduler = NS(_jobs={}, _req_status={})
        self.calls = []
        self.return_value = (0, False)
        self.pending_push = False

    def get_num_new_matched_tokens(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.return_value

    def has_pending_push_work(self) -> bool:
        return self.pending_push


class FakeRequest:
    def __init__(self, rid: str) -> None:
        self.request_id = rid
        self.status = NS(name="PREEMPTED")
        self.num_computed_tokens = 10
        self.num_prompt_tokens = 8
        self.num_output_tokens = 2
        self.num_tokens = 11
        self.max_tokens = 100
        self.num_in_flight_tokens = 0


class FakeScheduler:
    def __init__(self) -> None:
        self.kv_cache_manager = FakeManager()
        self.connector = OffloadingConnector()
        self.target = FakeRequest("target")
        self.requests = {"target": self.target}
        self.running = [self.target]
        self.waiting = []
        self.skipped_waiting = []
        self._inflight_prefills = []
        self._rotation_target = "target"
        self._rotation_begin = lambda *args, **kwargs: None
        self._rotation_hold = lambda req: req is self.target
        self.schedule = lambda *args, **kwargs: NS(num_scheduled_tokens={})

    def _inflight_prefill_reserved_blocks(self) -> int:
        return 7

    def _request_remaining_blocks(self, request: FakeRequest) -> int:
        return 3


class ObserverTest(unittest.TestCase):
    def test_exact_forwarding_and_instance_restoration(self):
        scheduler = FakeScheduler()
        manager = scheduler.kv_cache_manager
        connector = scheduler.connector
        native_result = object()
        lookup_result = (17, True)
        schedule_result = NS(num_scheduled_tokens={"target": 3})
        manager.return_value = native_result
        manager.probe_blocks = True
        connector.return_value = lookup_result
        original_schedule = scheduler.schedule
        original_begin = scheduler._rotation_begin
        original_hold = scheduler._rotation_hold
        scheduler.schedule = lambda *args, **kwargs: schedule_result
        original_schedule = scheduler.schedule
        data, uninstall = observer.install(scheduler)

        req = scheduler.target
        result = manager.allocate_slots(req, 4, reserved_blocks=7,
                                        full_sequence_must_fit=True,
                                        new_computed_blocks=None)
        self.assertIs(result, native_result)
        self.assertEqual(manager.calls, [((req, 4), {
            "reserved_blocks": 7, "full_sequence_must_fit": True,
            "new_computed_blocks": None})])
        self.assertIs(connector.get_num_new_matched_tokens(req, 10), lookup_result)
        self.assertEqual(connector.calls, [((req, 10), {})])
        self.assertIsNone(scheduler._rotation_begin([], 123.0))
        self.assertTrue(scheduler._rotation_hold(req))
        self.assertIs(scheduler.schedule("arg", named="value"), schedule_result)
        self.assertEqual(data["recent_allocations"][0]["actual_kwargs"]["reserved_blocks"], 7)
        self.assertEqual(data["recent_allocations"][0]["inflight_prefill_reserved_blocks_before"], 7)
        requirements = data["recent_block_requirements"]
        self.assertEqual([row["required_blocks_returned"] for row in requirements], [12, 3])
        self.assertEqual([row["allocation_call"] for row in requirements], [1, 1])
        self.assertEqual(requirements[0]["actual_kwargs"]["apply_admission_cap"], True)
        self.assertEqual(len(manager.coordinator.calls), 2)
        self.assertEqual(data["recent_lookups"][0]["matched_tokens"], 17)
        self.assertEqual(data["recent_holds"][0]["held"], True)

        returned = uninstall()
        self.assertIs(returned, data)
        self.assertEqual(data["status"], "UNINSTALLED_WITHOUT_SNAPSHOT")
        self.assertNotIn("allocate_slots", vars(manager))
        self.assertNotIn("get_num_blocks_to_allocate", vars(manager.coordinator))
        self.assertNotIn("get_num_new_matched_tokens", vars(connector))
        self.assertIs(scheduler.schedule, original_schedule)
        self.assertIs(scheduler._rotation_begin, original_begin)
        self.assertIs(scheduler._rotation_hold, original_hold)
        json.dumps(data, allow_nan=False)

    def test_bounded_snapshot_after_same_target_empty_without_pending_jobs(self):
        scheduler = FakeScheduler()
        req = scheduler.target
        scheduler.kv_cache_manager.probe_blocks = True
        calls = []

        def native_schedule(*args, **kwargs):
            calls.append((args, kwargs))
            scheduler._rotation_begin([], 0.0)
            scheduler._rotation_hold(req)
            scheduler.connector.get_num_new_matched_tokens(req, 10)
            scheduler.kv_cache_manager.allocate_slots(
                req, 0, num_external_computed_tokens=80,
                delay_cache_blocks=True, full_sequence_must_fit=True,
                reserved_blocks=7, has_scheduled_reqs=True)
            return NS(num_scheduled_tokens={})

        scheduler.schedule = native_schedule
        data, uninstall = observer.install(scheduler)
        clock = iter(i * 0.05 for i in range(100))
        try:
            with patch.object(observer.time, "perf_counter", side_effect=lambda: next(clock)):
                with self.assertRaises(observer.NoProgressSnapshot) as caught:
                    for _ in range(observer.EMPTY_SCHEDULE_CALLS):
                        scheduler.schedule()
            self.assertEqual(len(calls), observer.EMPTY_SCHEDULE_CALLS)
            self.assertEqual(data["status"], "SNAPSHOT_CAPTURED")
            snapshot = caught.exception.snapshot
            self.assertIs(snapshot, data["snapshot"])
            self.assertEqual(snapshot["candidate_request_id"], "target")
            self.assertGreaterEqual(snapshot["empty_duration_seconds"], 1.0)
            self.assertEqual(snapshot["state"]["free_blocks"], 4)
            self.assertEqual(snapshot["state"]["inflight_prefill_reserved_blocks"], 7)
            self.assertEqual(snapshot["state"]["pending"]["scheduler_job_count"], 0)
            self.assertEqual(snapshot["state"]["pending"]["transfer_job_count"], 0)
            self.assertTrue(snapshot["state"]["running"]["requests"][0]["held_this_schedule"])
            refusal = snapshot["recent_allocations"][-1]
            self.assertTrue(refusal["returned_none"])
            self.assertEqual(refusal["actual_kwargs"]["reserved_blocks"], 7)
            self.assertEqual(refusal["actual_kwargs"]["num_external_computed_tokens"], 80)
            self.assertEqual(snapshot["recent_block_requirements"][-2]["required_blocks_returned"], 12)
            self.assertEqual(snapshot["recent_block_requirements"][-1]["required_blocks_returned"], 3)
            self.assertEqual(snapshot["recent_lookups"][-1]["matched_tokens"], 0)
            json.dumps(data, allow_nan=False)
        finally:
            uninstall()

    def test_pending_native_job_suppresses_trigger_and_resets_streak(self):
        scheduler = FakeScheduler()
        data, uninstall = observer.install(scheduler)
        clock = iter(i * 0.1 for i in range(200))
        cs = scheduler.connector.connector_scheduler
        cs._jobs["job1"] = NS(req_id="target", is_store=False, pending_count=1)
        cs._req_status["target"] = NS(transfer_jobs={"job1"})
        try:
            with patch.object(observer.time, "perf_counter", side_effect=lambda: next(clock)):
                for _ in range(40):
                    scheduler.schedule()
                self.assertEqual(data["consecutive_empty_schedules"], 0)
                cs._jobs.clear()
                cs._req_status["target"].transfer_jobs.clear()
                for _ in range(10):
                    scheduler.schedule()
                self.assertEqual(data["consecutive_empty_schedules"], 10)
                scheduler.connector.pending_push = True
                scheduler.schedule()
                self.assertEqual(data["consecutive_empty_schedules"], 0)
            self.assertEqual(data["status"], "OBSERVING")
        finally:
            uninstall()

    def test_positive_schedule_resets_and_event_rings_stay_bounded(self):
        scheduler = FakeScheduler()
        result = NS(num_scheduled_tokens={})
        scheduler.schedule = lambda: result
        data, uninstall = observer.install(scheduler)
        clock = iter(i * 0.1 for i in range(500))
        try:
            with patch.object(observer.time, "perf_counter", side_effect=lambda: next(clock)):
                for _ in range(20):
                    scheduler.schedule()
                result.num_scheduled_tokens = {"target": 1}
                scheduler.schedule()
                self.assertEqual(data["consecutive_empty_schedules"], 0)
                result.num_scheduled_tokens = {}
                for _ in range(20):
                    scheduler.schedule()
                self.assertEqual(data["consecutive_empty_schedules"], 20)
                for _ in range(80):
                    scheduler.kv_cache_manager.allocate_slots(scheduler.target, 1, reserved_blocks=2)
            self.assertEqual(len(data["recent_allocations"]), observer.MAX_EVENTS)
            self.assertLessEqual(len(data["recent_schedules"]), observer.MAX_EVENTS)
            self.assertEqual(data["status"], "OBSERVING")
        finally:
            uninstall()

    def test_outside_coordinator_reads_do_not_evict_allocation_requirements(self):
        scheduler = FakeScheduler()
        manager = scheduler.kv_cache_manager
        manager.probe_blocks = True
        data, uninstall = observer.install(scheduler)
        try:
            manager.allocate_slots(scheduler.target, 1, reserved_blocks=2)
            self.assertEqual(len(data["recent_block_requirements"]), 2)
            for _ in range(observer.MAX_EVENTS + 100):
                self.assertEqual(manager.coordinator.get_num_blocks_to_allocate(
                    request_id="target", num_tokens=1), 3)
            self.assertEqual(data["outside_block_requirement_calls"], observer.MAX_EVENTS + 100)
            self.assertEqual(data["block_requirement_calls"], 2)
            self.assertEqual([row["allocation_call"] for row in data["recent_block_requirements"]], [1, 1])
            self.assertEqual([row["required_blocks_returned"] for row in data["recent_block_requirements"]], [12, 3])
        finally:
            uninstall()


if __name__ == "__main__":
    unittest.main()

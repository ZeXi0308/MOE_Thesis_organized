"""CPU-only checks for opt-in native async observation and natural tail drain."""
import sys
import types
import unittest
from unittest.mock import patch

import request_measurement as measurement


class FakeClock:
    def __init__(self):
        self.elapsed = 0.0

    def perf_counter(self):
        return 1000.0 + self.elapsed

    def time(self):
        return 1700000000.0 + self.elapsed

    def sleep(self, seconds):
        self.elapsed += seconds

    def advance_step(self):
        self.elapsed += 1.0


class NoHasWork:
    @property
    def has_work(self):
        raise AssertionError("drain must inspect explicit backend state, not has_work")


class FakeConnector(NoHasWork):
    def __init__(self, engine):
        self.engine = engine

    def has_pending_push_work(self):
        return self.engine.backend_flags()[3]


class FakeScheduler(NoHasWork):
    def __init__(self, engine):
        self.engine = engine
        self.connector = FakeConnector(engine)

    def has_requests(self):
        return self.engine.active or self.engine.backend_flags()[1]

    @property
    def deferred_frees(self):
        # Immutable views ensure the observer cannot free resources itself.
        return ("deferred-request",) if self.engine.backend_flags()[2] else ()


class FakeCore(NoHasWork):
    def __init__(self, engine):
        self.engine = engine
        self.scheduler = FakeScheduler(engine)

    @property
    def batch_queue(self):
        return ("batch",) if self.engine.backend_flags()[0] else ()


class FakeEngine(NoHasWork):
    def __init__(self, clock, *, async_scheduling=True, initial_empty=True,
                 tail="staged", expose_backend=True):
        self.clock = clock
        self.initial_empty = initial_empty
        self.tail = tail
        self.expose_backend = expose_backend
        self.active = False
        self.frontend_done = False
        self.steps = 0
        self.submissions = []
        self.core = FakeCore(self)
        self.vllm_config = types.SimpleNamespace(
            scheduler_config=types.SimpleNamespace(
                async_scheduling=async_scheduling, stream_interval=1))

    @property
    def engine_core(self):
        if not self.expose_backend:
            raise AssertionError("default synchronous measurement touched backend internals")
        return types.SimpleNamespace(engine_core=self.core)

    def backend_flags(self):
        """Pending batch, scheduler requests, deferred frees, connector work."""
        if not self.frontend_done:
            return False, False, False, False
        if self.tail == "deferred_forever":
            return False, False, True, False
        if self.tail == "staged":
            # Frontend completes on step 2; one backend condition then clears
            # per natural engine.step, finishing with connector work on step 6.
            return self.steps < 3, self.steps < 4, self.steps < 5, self.steps < 6
        return False, False, False, False

    def add_request(self, rid, prompt, params, arrival_time):
        self.rid, self.params, self.active = rid, params, True
        self.submissions.append((rid, arrival_time))
        return "internal/" + rid

    def has_unfinished_requests(self):
        return self.active

    def step(self):
        self.steps += 1
        if self.steps > 10:
            raise AssertionError("unexpected excess drain calls")
        self.clock.advance_step()
        if self.frontend_done or (self.initial_empty and self.steps == 1):
            return []
        self.active = False
        self.frontend_done = True
        completion = types.SimpleNamespace(token_ids=[31], finish_reason="length")
        return [types.SimpleNamespace(request_id=self.rid, finished=True,
                                      outputs=[completion])]


class AsyncMeasurementTest(unittest.TestCase):
    def run_episode(self, engine, **kwargs):
        fake = types.ModuleType("vllm")
        fake.SamplingParams = lambda **values: types.SimpleNamespace(**values)
        sampling = types.ModuleType("vllm.sampling_params")
        sampling.RequestOutputKind = types.SimpleNamespace(CUMULATIVE="cumulative")
        workload = dict(source_requests=[dict(request_id="a", document_id="a")],
            actual_prompt_token_ids=[[11, 12]], arrival_traces_s={"steady": [0.0]})
        with patch.dict(sys.modules, {"vllm": fake, "vllm.sampling_params": sampling}), \
                patch.object(measurement, "time", engine.clock):
            return measurement.measure_episode(engine, workload,
                dict(output_tokens=1), "steady", 1.0, "test", **kwargs)

    def assert_single_progress_event(self, raw, *, completed_s, call_index):
        row = raw["requests"][0]
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["output_token_ids"], [31])
        self.assertEqual(row["token_times_s"], [completed_s])
        self.assertEqual(row["completion_s"], completed_s)
        self.assertEqual(len(raw["output_events"]), 1)
        event = raw["output_events"][0]
        self.assertEqual(event["engine_call_index"], call_index)
        self.assertEqual(event["received_s"], completed_s)
        self.assertEqual(event["new_token_ids"], [31])
        self.assertEqual(event["chunk_size"], 1)
        self.assertTrue(event["finished"])

    def test_frontend_completion_keeps_stepping_until_all_backend_work_drains(self):
        clock = FakeClock()
        engine = FakeEngine(clock)
        raw = self.run_episode(engine, allow_async=True, max_seconds=10)
        self.assertEqual(raw["status"], "COMPLETE")
        self.assertIsNone(raw["error"])
        self.assertEqual(engine.steps, 6)
        self.assertEqual(raw["engine_call_count"], 6)
        self.assertEqual(raw["engine_return_count"], 6)
        self.assertEqual(raw["observation_end_s"], 6.0)
        self.assertEqual(engine.backend_flags(), (False, False, False, False))
        self.assertEqual(engine.submissions, [("test/a", 1700000000.0)])
        observed = raw["async_observation"]
        self.assertTrue(observed["async_scheduling"])
        self.assertEqual(observed["frontend_complete_s"], 2.0)
        self.assertTrue(observed["backend_drained"])
        self.assertIsInstance(observed["final_backend_state"], dict)
        final = observed["final_backend_state"]
        self.assertEqual(final["batch_queue_depth"], 0)
        self.assertFalse(final["scheduler_has_requests"])
        self.assertEqual(final["deferred_free_entries"], 0)
        self.assertFalse(final["connector_pending_push_work"])
        self.assertEqual(observed["tail_step_calls"], 4)
        self.assertEqual(observed["tail_elapsed_s"], 4.0)
        self.assertEqual(observed["empty_output_step_calls"], 5)
        # Neither the empty initial async step nor empty tail returns are tokens.
        self.assert_single_progress_event(raw, completed_s=2.0, call_index=1)

    def test_only_deferred_frees_at_deadline_is_incomplete_even_if_request_completed(self):
        clock = FakeClock()
        engine = FakeEngine(clock, tail="deferred_forever")
        raw = self.run_episode(engine, allow_async=True, max_seconds=4)
        self.assertEqual(raw["status"], "INCOMPLETE")
        self.assertEqual(raw["error"], "runtime_limit")
        self.assertEqual(engine.steps, 4)
        self.assertEqual(raw["observation_end_s"], 4.0)
        self.assertEqual(engine.backend_flags(), (False, False, True, False))
        observed = raw["async_observation"]
        self.assertTrue(observed["async_scheduling"])
        self.assertEqual(observed["frontend_complete_s"], 2.0)
        self.assertFalse(observed["backend_drained"])
        self.assertIsInstance(observed["final_backend_state"], dict)
        self.assertEqual(observed["final_backend_state"]["deferred_free_entries"], 1)
        self.assertEqual(observed["tail_step_calls"], 2)
        self.assertEqual(observed["tail_elapsed_s"], 2.0)
        self.assertEqual(observed["empty_output_step_calls"], 3)
        self.assert_single_progress_event(raw, completed_s=2.0, call_index=1)

    def test_final_blocking_step_over_deadline_is_not_complete_even_when_drained(self):
        clock = FakeClock()
        engine = FakeEngine(clock, initial_empty=False, tail="none")
        raw = self.run_episode(engine, allow_async=True, max_seconds=0.5)
        self.assertEqual(raw["status"], "INCOMPLETE")
        self.assertEqual(raw["error"], "runtime_limit")
        self.assertEqual(raw["observation_end_s"], 1.0)
        self.assertTrue(raw["async_observation"]["backend_drained"])
        self.assertEqual(raw["async_observation"]["runtime_overrun_s"], 0.5)
        self.assert_single_progress_event(raw, completed_s=1.0, call_index=0)

    def test_sync_default_keeps_legacy_output_and_does_not_inspect_backend(self):
        clock = FakeClock()
        engine = FakeEngine(clock, async_scheduling=False, initial_empty=False,
                            expose_backend=False)
        raw = self.run_episode(engine)
        self.assertEqual(raw["status"], "COMPLETE")
        self.assertEqual(engine.steps, 1)
        self.assertEqual(raw["observation_end_s"], 1.0)
        self.assertNotIn("async_observation", raw)
        self.assert_single_progress_event(raw, completed_s=1.0, call_index=0)

    def test_native_async_requires_explicit_opt_in(self):
        clock = FakeClock()
        engine = FakeEngine(clock, expose_backend=False)
        raw = self.run_episode(engine)
        self.assertEqual(raw["status"], "INCOMPLETE")
        self.assertIn("ValueError", raw["error"])
        self.assertIn("async_scheduling", raw["error"])
        self.assertEqual(engine.steps, 0)
        self.assertEqual(engine.submissions, [])
        self.assertNotIn("async_observation", raw)
        self.assertEqual(raw["output_events"], [])


if __name__ == "__main__":
    unittest.main()

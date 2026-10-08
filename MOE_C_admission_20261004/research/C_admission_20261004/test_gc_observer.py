"""CPU checks for optional GC observation and callback lifetime."""
import gc
import sys
import time
import types
import unittest
from unittest.mock import patch

import request_measurement as measurement


class Engine:
    def __init__(self, *, fail_step=False):
        self.fail_step = fail_step
        self.active = False
        self.collections = []
        self.vllm_config = types.SimpleNamespace(
            scheduler_config=types.SimpleNamespace(
                async_scheduling=False, stream_interval=1))

    def collect(self, phase):
        callbacks = tuple(gc.callbacks)
        start = time.perf_counter()
        gc.collect(2)
        end = time.perf_counter()
        self.collections.append((phase, start, end, callbacks))

    def add_request(self, rid, prompt, params, arrival_time):
        self.rid, self.params, self.active = rid, params, True
        self.collect("ingress")
        return "internal/" + rid

    def has_unfinished_requests(self):
        return self.active

    def step(self):
        self.collect("engine_step")
        if self.fail_step:
            raise RuntimeError("worker failure after GC")
        self.active = False
        engine = self

        class Output:
            request_id, finished = engine.rid, True
            collected = False

            @property
            def outputs(self):
                if not self.collected:
                    self.collected = True
                    engine.collect("output_recording")
                return [types.SimpleNamespace(token_ids=[31], finish_reason="stop")]

        return [Output()]


class GCObserverTest(unittest.TestCase):
    def setUp(self):
        self.original_callbacks = list(gc.callbacks)
        self.enabled = gc.isenabled()
        self.thresholds = gc.get_threshold()
        self.sentinel_events = []

        def sentinel(phase, info):
            self.sentinel_events.append((phase, info["generation"]))

        self.sentinel = sentinel
        gc.callbacks.append(sentinel)
        self.expected_callbacks = list(gc.callbacks)

    def tearDown(self):
        try:
            self.assert_restored()
        finally:
            # Keep even a failing check from contaminating later tests.
            gc.callbacks[:] = self.original_callbacks

    def assert_restored(self):
        self.assertEqual(gc.callbacks, self.expected_callbacks)
        self.assertEqual(gc.isenabled(), self.enabled)
        self.assertEqual(gc.get_threshold(), self.thresholds)

    def run_episode(self, engine, **kwargs):
        fake = types.ModuleType("vllm")
        fake.SamplingParams = lambda **values: types.SimpleNamespace(**values)
        sampling = types.ModuleType("vllm.sampling_params")
        sampling.RequestOutputKind = types.SimpleNamespace(CUMULATIVE="cumulative")
        workload = dict(source_requests=[dict(request_id="a", document_id="a")],
            actual_prompt_token_ids=[[11, 12]], arrival_traces_s={"steady": [0.]})
        with patch.dict(sys.modules, {"vllm": fake, "vllm.sampling_params": sampling}):
            return measurement.measure_episode(engine, workload,
                dict(output_tokens=2, min_tokens=0, ignore_eos=False),
                "steady", 1., "test", **kwargs)

    def assert_observation(self, raw, engine):
        observed = raw["gc_observation"]
        self.assertEqual(observed["enabled_at_install"], self.enabled)
        self.assertEqual(tuple(observed["thresholds_at_install"]), self.thresholds)
        self.assertEqual(observed["origin_perf_s"],
            raw["measurement_origin_perf_counter_s"])
        self.assertEqual(observed["error_count"], 0)
        self.assertGreaterEqual(observed["callback_count"], 2 * len(engine.collections))
        self.assertGreaterEqual(observed["callback_wall_s"], 0.)
        self.assertGreaterEqual(observed["callback_cpu_s"], 0.)
        for phase, start, end, callbacks in engine.collections:
            with self.subTest(phase=phase):
                # Match explicit collections by their real clock interval; automatic
                # collections are allowed and need not have a fixed event count.
                events = [e for e in observed["events"] if e["generation"] == 2
                    and e["start_perf_s"] is not None and e["end_perf_s"] is not None
                    and start <= e["start_perf_s"] <= e["end_perf_s"] <= end]
                self.assertTrue(events, (phase, observed["events"]))
                for event in events:
                    self.assertEqual(event["phase"], phase)
                    self.assertEqual(event["stop_phase"], phase)
                    self.assertIsInstance(event["collected"], int)
                    self.assertIsInstance(event["uncollectable"], int)
                    self.assertGreaterEqual(event["collected"], 0)
                    self.assertGreaterEqual(event["uncollectable"], 0)
                self.assertIn(self.sentinel, callbacks)
        self.assert_restored()

    def test_natural_eos_records_three_phases_and_restores_callbacks(self):
        engine = Engine()
        raw = self.run_episode(engine, record_gc=True)
        self.assertEqual(raw["status"], "COMPLETE")
        self.assertEqual(raw["requests"][0]["output_token_ids"], [31])
        self.assertEqual(raw["requests"][0]["stop_reason"], "stop")
        self.assertFalse(engine.params.ignore_eos)
        self.assertEqual([c[0] for c in engine.collections],
            ["ingress", "engine_step", "output_recording"])
        self.assert_observation(raw, engine)

    def test_engine_exception_retains_gc_events_and_restores_callbacks(self):
        engine = Engine(fail_step=True)
        raw = self.run_episode(engine, record_gc=True)
        self.assertEqual(raw["status"], "INCOMPLETE")
        self.assertEqual(raw["error"], "RuntimeError: worker failure after GC")
        self.assertEqual(raw["requests"][0]["status"], "failed")
        self.assertEqual([c[0] for c in engine.collections], ["ingress", "engine_step"])
        self.assert_observation(raw, engine)

    def test_default_disabled_keeps_existing_callbacks_throughout(self):
        engine = Engine()
        raw = self.run_episode(engine)
        self.assertEqual(raw["status"], "COMPLETE")
        self.assertNotIn("gc_observation", raw)
        for _, _, _, callbacks in engine.collections:
            self.assertEqual(callbacks, tuple(self.expected_callbacks))
        self.assertGreaterEqual(self.sentinel_events.count(("start", 2)), 3)
        self.assertGreaterEqual(self.sentinel_events.count(("stop", 2)), 3)
        self.assert_restored()


if __name__ == "__main__":
    unittest.main()

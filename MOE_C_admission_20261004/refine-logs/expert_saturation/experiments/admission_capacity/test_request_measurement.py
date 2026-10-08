"""Behavior checks for request timing and complete-cohort accounting; CPU only."""
import sys
import types
import unittest
from unittest.mock import patch

import request_measurement as measurement
from metrics import summarize_episode_requests


class Clock:
    def __init__(self):
        self.value = 0.0

    def perf_counter(self):
        return self.value

    def time(self):
        return 1000.0 + self.value

    def sleep(self, seconds):
        self.value += seconds


def output(rid, tokens, finished=False, reason=None):
    return types.SimpleNamespace(request_id="test/" + rid, finished=finished,
        outputs=[types.SimpleNamespace(token_ids=tokens, finish_reason=reason)])


class Engine:
    def __init__(self, clock, actions):
        self.clock, self.actions, self.active, self.params = clock, iter(actions), set(), {}
        self.vllm_config = types.SimpleNamespace(scheduler_config=types.SimpleNamespace(
            async_scheduling=False, stream_interval=1))
        self.policy_calls = 0
        self.scheduler = types.SimpleNamespace(schedule=self.policy_step)

    def policy_step(self):
        self.policy_calls += 1
        self.clock.sleep(0.1)

    def add_request(self, rid, prompt, params, arrival_time):
        self.active.add(rid)
        self.params[rid] = params
        return "internal/" + rid

    def has_unfinished_requests(self):
        return bool(self.active)

    def step(self):
        self.scheduler.schedule()
        duration, values = next(self.actions)
        self.clock.sleep(duration)
        if isinstance(values, Exception):
            raise values
        for value in values:
            if value.finished:
                self.active.remove(value.request_id)
        return values


def workload(arrivals):
    return dict(source_requests=[dict(request_id=rid, document_id=rid) for rid, _ in arrivals],
        actual_prompt_token_ids=[[11, 12] for _ in arrivals],
        arrival_traces_s={"steady": [arrival for _, arrival in arrivals]})


class RequestMeasurementTest(unittest.TestCase):
    def run_episode(self, engine, arrivals, **kwargs):
        fake = types.ModuleType("vllm")
        fake.SamplingParams = lambda **params: types.SimpleNamespace(**params)
        sampling = types.ModuleType("vllm.sampling_params")
        sampling.RequestOutputKind = types.SimpleNamespace(CUMULATIVE="cumulative")
        config = kwargs.pop("config", dict(output_tokens=2))
        with patch.dict(sys.modules, {"vllm": fake, "vllm.sampling_params": sampling}), \
                patch.object(measurement, "time", engine.clock):
            return measurement.measure_episode(engine, workload(arrivals), config,
                "steady", 1.0, "test", **kwargs)

    def test_internal_work_and_duplicate_output_do_not_reset_output_wait(self):
        clock = Clock()

        class SlowToRead:
            request_id, finished = "test/a", False

            @property
            def outputs(self):
                clock.sleep(0.4)  # Host processing after engine return must not shift receipt.
                return [types.SimpleNamespace(token_ids=[31], finish_reason=None)]

        engine = Engine(clock, [(0.9, []), (0.9, [SlowToRead()]),
            (0.9, [output("a", [31])]), (0.9, [output("a", [31, 32], True, "length")])])
        result = self.run_episode(engine, [("a", 0.0)])
        row = result["requests"][0]
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual((result["engine_call_count"], result["engine_return_count"]), (4, 4))
        self.assertEqual(row["output_token_ids"], [31, 32])
        self.assertAlmostEqual(row["token_times_s"][0], 2.0)
        self.assertAlmostEqual(row["token_times_s"][1], 4.8)
        self.assertEqual([event["chunk_size"] for event in result["output_events"]], [1, 0, 1])
        metrics = summarize_episode_requests(result["requests"], observation_end_s=result["observation_end_s"],
            ttft_slo_s=10, tpot_slo_s=10)
        self.assertAlmostEqual(metrics["per_request"][0]["itl_s"][0], 2.8)

    def test_abort_retains_partial_outputs_arrived_and_future_denominators(self):
        for failure in (RuntimeError("worker failure"), None):
            with self.subTest(failure=failure):
                clock = Clock()
                engine = Engine(clock, [(0.9, [output("a", [31])]), (0.9, failure or [])])
                result = self.run_episode(engine, [("a", 0.0), ("b", 0.5), ("c", 5.0)], max_seconds=1.5)
                rows = {row["request_id"]: row for row in result["requests"]}
                self.assertEqual(result["status"], "INCOMPLETE")
                self.assertEqual(result["engine_call_count"], 2)
                self.assertEqual(result["engine_return_count"], 1 if failure else 2)
                self.assertEqual(rows["a"]["output_token_ids"], [31])
                self.assertEqual(rows["a"]["token_times_s"], [1.0])
                self.assertEqual(rows["a"]["status"], "failed" if failure else "unfinished")
                self.assertIsNone(rows["c"]["admission_s"])
                self.assertFalse(rows["c"]["arrived_at_observation_end"])
                metrics = summarize_episode_requests(result["requests"], observation_end_s=result["observation_end_s"],
                    ttft_slo_s=10, tpot_slo_s=10)
                self.assertEqual((metrics["n_planned"], metrics["n_arrived"], metrics["n_not_yet_arrived"]), (3, 2, 1))
                self.assertEqual((metrics["n_completed"], metrics["goodput_rps"]), (0, 0))

    def test_external_policy_cost_and_hook_identity_survive_natural_eos(self):
        clock = Clock()
        engine = Engine(clock, [(0.9, [output("a", [31], True, "stop")])])
        installed = dict(vars(engine.scheduler))
        result = self.run_episode(engine, [("a", 0.0)],
            config=dict(output_tokens=100, min_tokens=0, ignore_eos=False))
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(engine.policy_calls, 1)
        self.assertEqual(vars(engine.scheduler), installed)
        self.assertIs(engine.scheduler.schedule, installed["schedule"])
        self.assertEqual(result["requests"][0]["completion_s"], 1.0)
        self.assertEqual(engine.params["test/a"].min_tokens, 0)
        self.assertFalse(engine.params["test/a"].ignore_eos)
        self.assertEqual(result["diagnostics"], "DIAGNOSTICS_DISABLED")
        for key in ("actual_preemption_count", "recomputed_tokens", "recovery_count"):
            self.assertIsNone(result[key])
        for key in ("memory_trace", "scheduler_steps", "preemption_events"):
            self.assertNotIn(key, result)

    def preempting_engine(self, clock, actions, *, fail=False, instance_hook=True):
        class Preempting(Engine):
            def policy_step(self):
                super().policy_step()
                if self.policy_calls == 2:
                    request = types.SimpleNamespace(request_id="internal/test/a", num_output_tokens=99)
                    self.preempt_result = self.scheduler._preempt_request(request, timestamp=clock.perf_counter())

        engine = Preempting(clock, actions)
        calls = []

        class Scheduler:
            def _preempt_request(self, request, timestamp):
                calls.append((request.request_id, timestamp))
                if fail:
                    raise RuntimeError("preempt failed")
                return "prior hook result"

        engine.scheduler = Scheduler()
        engine.scheduler.schedule = engine.policy_step
        if instance_hook:
            original = engine.scheduler._preempt_request
            engine.scheduler._preempt_request = lambda request, **kwargs: original(request, **kwargs)
        engine.engine_core = types.SimpleNamespace(engine_core=types.SimpleNamespace(scheduler=engine.scheduler))
        return engine, calls

    def test_sparse_preempt_records_last_returned_output_and_restores_hook(self):
        for instance_hook in (False, True):
            with self.subTest(instance_hook=instance_hook):
                clock = Clock()
                engine, calls = self.preempting_engine(clock,
                    [(0.9, [output("a", [31])]), (0.9, [output("a", [31])]),
                     (0.9, [output("a", [31, 32], True, "length")])], instance_hook=instance_hook)
                installed = dict(vars(engine.scheduler))
                result = self.run_episode(engine, [("a", 0)], record_preemptions=True)
                self.assertEqual(result["status"], "COMPLETE")
                self.assertEqual(result["actual_preemption_count"], 1)
                self.assertEqual(len(calls), 1)
                self.assertEqual(engine.preempt_result, "prior hook result")
                event, = result["preemption_events"]
                self.assertEqual(event["engine_call_index"], 1)
                self.assertEqual(event["last_returned_output_count"], 1)
                self.assertEqual(event["last_new_output_s"], 1.0)
                self.assertEqual(event["native_output_count_before"], 99)
                self.assertTrue(event["original_preemption_returned"])
                self.assertEqual(result["requests"][0]["token_times_s"], [1.0, 3.0])
                self.assertEqual(vars(engine.scheduler), installed)
                self.assertEqual(result["diagnostics"], "SPARSE_PREEMPTION_EVENTS")

    def test_sparse_preempt_failure_retains_partial_result_and_restores_hook(self):
        clock = Clock()
        engine, calls = self.preempting_engine(clock,
            [(0.9, [output("a", [31])]), (0.9, [])], fail=True)
        installed = dict(vars(engine.scheduler))
        result = self.run_episode(engine, [("a", 0)], record_preemptions=True)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["error"], "RuntimeError: preempt failed")
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["preemption_attempt_count"], 1)
        self.assertEqual(result["actual_preemption_count"], 0)
        event, = result["preemption_events"]
        self.assertFalse(event["original_preemption_returned"])
        self.assertIsNone(event["method_returned_s"])
        self.assertEqual(event["error"], result["error"])
        self.assertEqual(result["requests"][0]["token_times_s"], [1.0])
        self.assertEqual(result["engine_return_count"], 1)
        self.assertEqual(vars(engine.scheduler), installed)


if __name__ == "__main__":
    unittest.main()

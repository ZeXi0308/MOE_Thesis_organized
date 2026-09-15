"""Only frozen-plan, drained-setter, and empty-output/guard checks; no GPU."""
from copy import deepcopy
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import run_prefill_budget as runner
from native_capture import capture_episode


class Scheduler:
    def __init__(self):
        self.requests, self.running, self.calls = {}, [], 0
        self.max_num_scheduled_tokens, self.max_num_running_reqs = 1024, 8
        self.fault = None

    def get_request_counts(self):
        return len(self.running), len(self.requests) - len(self.running)

    def schedule(self):
        self.calls += 1
        req = self.requests["native"]
        self.running = [req]
        amount = 2 if self.calls < 3 else 1
        if self.calls == 3 and self.fault == "skip":
            return NS(num_scheduled_tokens={}, preempted_req_ids=set())
        req.num_computed_tokens += amount
        return NS(num_scheduled_tokens={"native": amount},
                  preempted_req_ids={"native"} if self.calls == 3 and self.fault == "preempt" else set())


class Engine:
    def __init__(self):
        self.scheduler = Scheduler()
        self.engine_core = NS(engine_core=NS(scheduler=self.scheduler))
        self.vllm_config = NS(scheduler_config=NS(max_num_seqs=8, max_num_batched_tokens=1024,
                                                async_scheduling=False, stream_interval=1))

    def has_unfinished_requests(self):
        return bool(self.scheduler.requests)

    def add_request(self, external_id, prompt, params, **kwargs):
        self.external_id = external_id
        self.scheduler.requests["native"] = NS(request_id="native", num_prompt_tokens=4, num_computed_tokens=0)
        return "native"

    def step(self):
        self.scheduler.schedule()
        if self.scheduler.calls == 1:
            return []
        finished = self.scheduler.calls == 3
        if finished:
            self.scheduler.requests.clear()
            self.scheduler.running.clear()
        return [NS(request_id=self.external_id, finished=finished, metrics=None,
                   outputs=[NS(token_ids=[10, 11] if finished else [10], finish_reason="length" if finished else None)])]


class RunnerTest(unittest.TestCase):
    def test_budget_only_changes_on_complete_drain(self):
        engine = Engine()
        frozen = deepcopy(engine.vllm_config)
        self.assertEqual(runner.set_empty_budget(engine, 256)["previous_budget"], 1024)
        self.assertEqual(engine.vllm_config, frozen)
        for reason in ("active", "waiting", "retained", "flag"):
            with self.subTest(reason=reason):
                with patch.object(engine, "has_unfinished_requests", return_value=reason == "flag"), \
                     patch.object(engine.scheduler, "get_request_counts", return_value=(1, 0) if reason == "active" else (0, 1) if reason == "waiting" else (0, 0)):
                    engine.scheduler.requests = {"retained": NS()} if reason == "retained" else {}
                    with self.assertRaises(ValueError):
                        runner.set_empty_budget(engine, 1024)
                    self.assertEqual(engine.scheduler.max_num_scheduled_tokens, 256)

    def test_common_warmups_and_exact_reverse_measured_plan(self):
        _, forward = runner.prepare("forward")
        _, reverse = runner.prepare("reverse")
        self.assertEqual(forward["warmups"], reverse["warmups"])
        self.assertEqual(forward["plans"], list(reversed(reverse["plans"])))
        self.assertEqual(len(forward["plans"]), 4)
        self.assertEqual(forward["engine_args"], reverse["engine_args"])

    def test_empty_outputs_recorded_and_decode_faults_retained(self):
        modules = {"vllm": NS(SamplingParams=lambda **kw: NS(**kw)),
                   "vllm.sampling_params": NS(RequestOutputKind=NS(CUMULATIVE="cumulative"))}
        workload = dict(source_requests=[dict(request_id="r", document_id="d")],
                        actual_prompt_token_ids=[[1, 2, 3, 4]], arrival_traces_s={"steady": [0]})
        for fault in (None, "skip", "preempt"):
            engine = Engine()
            engine.scheduler.fault = fault
            with self.subTest(fault=fault), patch.dict(sys.modules, modules):
                raw = capture_episode(engine, workload, dict(cap=8, output_tokens=2), regime="steady", arrival_scale=1, run_id="test")
                self.assertEqual(raw["engine_step_returns"][0]["output_count"], 0)
                self.assertEqual(len(raw["scheduler_steps"]), 3)
                self.assertEqual(raw["status"], "COMPLETE" if fault is None else "INCOMPLETE")
                self.assertEqual(len(raw["engine_step_returns"]), 3 if fault is None else 2)
                self.assertEqual(raw["scheduler_steps"][-1]["existing_decode_request_ids"], ["r"])


if __name__ == "__main__":
    unittest.main()

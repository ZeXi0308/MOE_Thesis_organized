"""Native capture risks exercised without importing vLLM or allocating a GPU."""
from copy import deepcopy
import hashlib
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from native_capture import capture_episode, set_empty_admission_cap


class Clock:
    value = 0.0

    def perf_counter(self):
        return self.value

    def time(self):
        return 1000.0 + self.value

    def sleep(self, seconds):
        self.value += seconds


class Scheduler:
    def __init__(self, engine):
        self.engine, self.requests, self.running = engine, {}, set()
        self.max_num_running_reqs = 8

    def get_request_counts(self):
        return len(self.running), len(self.requests) - len(self.running)

    def schedule(self, throttle_prefills=False):
        self.engine.forwarded.append(throttle_prefills)
        source, amount, _, _ = self.engine.frames[self.engine.index]
        rid = self.engine.internal[source]
        self.running = {rid}
        self.requests[rid].num_computed_tokens += amount  # Actual vLLM return semantics.
        return NS(num_scheduled_tokens={rid: amount}, preempted_req_ids=set())


class Engine:
    def __init__(self, clock, frames):
        self.clock, self.frames, self.index = clock, frames, 0
        self.internal, self.external, self.adds, self.forwarded = {}, {}, [], []
        self.scheduler = Scheduler(self)
        self.engine_core = NS(engine_core=NS(scheduler=self.scheduler))
        self.vllm_config = NS(scheduler_config=NS(async_scheduling=False, stream_interval=1, max_num_seqs=8))
        self.metrics = NS(arrival_time=1000.0, num_generation_tokens=0)
        self.first_step_submissions = None

    def add_request(self, external_id, prompt, params, *, arrival_time):
        source = external_id.rsplit("/", 1)[1]
        rid = external_id + "-internal"
        self.external[source], self.internal[source] = external_id, rid
        self.scheduler.requests[rid] = NS(num_computed_tokens=0, num_prompt_tokens=len(prompt["prompt_token_ids"]))
        self.adds.append((external_id, params, arrival_time))
        self.clock.sleep(0.01)
        return rid

    def has_unfinished_requests(self):
        return bool(self.scheduler.requests)

    def step(self):
        if self.first_step_submissions is None:
            self.first_step_submissions = len(self.adds)
        self.scheduler.schedule(False)
        source, _, tokens, finished = self.frames[self.index]
        self.index += 1
        self.clock.sleep(0.25)
        self.metrics.num_generation_tokens = len(tokens)  # Mutable across outputs.
        if finished:
            del self.scheduler.requests[self.internal[source]]
            self.scheduler.running.discard(self.internal[source])
        return [NS(request_id=self.external[source], finished=finished, metrics=self.metrics,
                   outputs=[NS(token_ids=tokens, finish_reason="length" if finished else None)])]


def workload(arrivals):
    return dict(source_requests=[dict(request_id=f"r{i}", document_id=f"doc{i}") for i in range(len(arrivals))],
                actual_prompt_token_ids=[[1, 2] for _ in arrivals], arrival_traces_s={"steady": arrivals})


def capture(engine, arrivals, *, count=2, scale=1.0):
    fake_modules = {"vllm": NS(SamplingParams=lambda **kw: NS(**kw)),
                    "vllm.sampling_params": NS(RequestOutputKind=NS(CUMULATIVE="cumulative"))}
    with patch.dict(sys.modules, fake_modules), patch("native_capture.time", engine.clock):
        return capture_episode(engine, workload(arrivals), dict(cap=1, output_tokens=count),
                               regime="steady", arrival_scale=scale, run_id="episode")


class NativeCaptureTest(unittest.TestCase):
    def test_empty_cap_changes_preserve_engine_and_graph_configuration(self):
        engine = Engine(Clock(), [])
        engine.vllm_config.compilation_config = NS(cudagraph_capture_sizes=[1, 2, 4, 8, 16])
        engine.worker = NS(max_num_reqs=8, captured_graphs=(1, 2, 4, 8))
        before_config, before_worker = deepcopy(engine.vllm_config), deepcopy(engine.worker)
        for cap, previous in ((6, 8), (8, 6)):
            with self.subTest(cap=cap):
                self.assertEqual(set_empty_admission_cap(engine, cap),
                    dict(engine_max_num_seqs=8, previous_admission_cap=previous, admission_cap=cap))
                self.assertEqual(engine.scheduler.max_num_running_reqs, cap)
                self.assertEqual(engine.vllm_config, before_config)
                self.assertEqual(engine.worker, before_worker)

    def test_cap_rejects_nonempty_engine_and_invalid_bounds_without_mutation(self):
        cases = [(cap, False, (0, 0), {}) for cap in (0, -1, 9, True, 6.0, "6", None)]
        cases += [(6, True, (0, 0), {}), (6, False, (1, 0), {}),
                  (6, False, (0, 1), {}), (6, False, (0, 0), {"retained": NS()})]
        for cap, unfinished, counts, requests in cases:
            with self.subTest(cap=cap, unfinished=unfinished, counts=counts, requests=bool(requests)):
                engine = Engine(Clock(), [])
                engine.scheduler.requests = requests
                before_config = deepcopy(engine.vllm_config)
                with patch.object(engine, "has_unfinished_requests", return_value=unfinished), \
                     patch.object(engine.scheduler, "get_request_counts", return_value=counts):
                    with self.assertRaises(ValueError):
                        set_empty_admission_cap(engine, cap)
                self.assertEqual(engine.scheduler.max_num_running_reqs, 8)
                self.assertEqual(engine.vllm_config, before_config)
                self.assertIs(engine.scheduler.requests, requests)

    def test_due_submission_internal_ids_and_post_schedule_phase_counts(self):
        engine = Engine(Clock(), [("r0", 2, [10], False), ("r0", 1, [10, 11], True),
                                  ("r1", 2, [20], False), ("r1", 1, [20, 21], True)])
        raw = capture(engine, [0, 0.02], scale=0)
        self.assertEqual(raw["status"], "COMPLETE", raw["error"])
        self.assertEqual(engine.first_step_submissions, 2)  # cap=1 does not restrict submission.
        self.assertEqual([r["arrival_s"] for r in raw["requests"]], [0, 0])
        self.assertEqual([entry[2] for entry in engine.adds], [1000, 1000])
        self.assertEqual(engine.forwarded, [False] * 4)
        self.assertEqual([(s["scheduled"][0]["prefill_tokens"], s["scheduled"][0]["decode_tokens"])
                          for s in raw["scheduler_steps"]], [(2, 0), (0, 1), (2, 0), (0, 1)])
        first = raw["scheduler_steps"][0]
        self.assertEqual((first["actual_active"], first["waiting_requests"]), (1, 1))
        self.assertEqual(first["scheduled"][0]["request_id"], "r0")
        self.assertEqual(first["scheduled"][0]["computed_before"], 0)
        self.assertEqual(first["scheduled"][0]["computed_after"], 2)
        self.assertEqual(raw["internal_to_source"]["episode/r0-internal"], "r0")
        self.assertGreater(raw["requests"][0]["engine_add_return_s"], raw["requests"][0]["admission_s"])
        self.assertEqual(raw["requests"][0]["prompt_token_ids_sha256"], hashlib.sha256(b"[1,2]").hexdigest())
        self.assertEqual(vars(engine.adds[0][1]), dict(n=1, temperature=0.0, max_tokens=2, min_tokens=2,
                         ignore_eos=True, detokenize=False, output_kind="cumulative"))
        self.assertNotIn("schedule", vars(engine.scheduler))

    def test_cumulative_chunk_times_and_metric_snapshots(self):
        engine = Engine(Clock(), [("r0", 2, [7], False), ("r0", 2, [7, 8, 9], True)])
        raw = capture(engine, [0.01], count=3, scale=2)
        row = raw["requests"][0]
        self.assertEqual(raw["status"], "COMPLETE", raw["error"])
        self.assertEqual(row["arrival_s"], 0.02)
        self.assertEqual(engine.adds[0][2], 1000.02)
        self.assertEqual(row["output_token_ids"], [7, 8, 9])
        self.assertEqual(row["token_times_s"][1], row["token_times_s"][2])
        self.assertGreater(row["token_times_s"][1], row["token_times_s"][0])
        self.assertEqual([e["new_token_ids"] for e in raw["output_events"]], [[7], [8, 9]])
        self.assertEqual(raw["output_events"][0]["native_metrics"]["num_generation_tokens"], 1)
        self.assertEqual(raw["host_chunk_diagnostics"]["multi_token_chunks"], 1)
        self.assertFalse(raw["host_chunk_diagnostics"]["token_level_itl_resolved"])
        self.assertFalse(raw["host_chunk_diagnostics"]["interpolated"])

    def test_prefix_failure_retains_raw_unsubmitted_requests_and_restores_wrapper(self):
        engine = Engine(Clock(), [("r0", 2, [10], False), ("r0", 1, [99, 11], True)])
        original = engine.scheduler.schedule
        engine.scheduler.schedule = original  # Also cover an existing instance override.
        raw = capture(engine, [0, 50])
        self.assertEqual(raw["status"], "INCOMPLETE")
        self.assertIn("cumulative token prefix", raw["error"])
        self.assertEqual([r["status"] for r in raw["requests"]], ["failed", "unfinished"])
        self.assertEqual(raw["requests"][0]["output_token_ids"], [10])
        self.assertIsNone(raw["requests"][1]["admission_s"])
        self.assertEqual(raw["output_events"][-1]["cumulative_token_ids"], [99, 11])
        self.assertFalse(raw["output_events"][-1]["prefix_valid"])
        self.assertIs(engine.scheduler.schedule, original)


if __name__ == "__main__":
    unittest.main()

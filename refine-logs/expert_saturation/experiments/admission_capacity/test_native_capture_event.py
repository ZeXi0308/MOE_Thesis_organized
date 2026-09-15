"""CPU event-contract checks; these do not validate vLLM or GPU execution."""
import sys
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from native_capture import capture_episode, event_release_time, output_limits


class EventCaptureTest(unittest.TestCase):
    def test_exact_output_boundary_and_receipt_time(self):
        event = dict(after_output_tokens={"a": 4, "b": 4})
        rows = {rid: dict(output_token_ids=[1] * n, token_times_s=list(range(1, n + 1)))
                for rid, n in (("a", 3), ("b", 4))}
        self.assertIsNone(event_release_time(rows, event))
        rows["a"] = dict(output_token_ids=[1] * 4, token_times_s=[1, 2, 3, 4.5])
        self.assertEqual(event_release_time(rows, event), 4.5)
        rows["a"]["output_token_ids"].append(1)
        with self.assertRaises(ValueError):
            event_release_time(rows, event)

    def test_per_request_output_limits_and_legacy_default(self):
        sources = [dict(request_id=rid) for rid in ("a", "b", "new")]
        self.assertEqual(output_limits(sources, dict(output_tokens=8)), dict(a=8, b=8, new=8))
        self.assertEqual(output_limits(sources, dict(output_tokens=8,
            output_tokens_by_request=dict(a=16, b=16))), dict(a=16, b=16, new=8))

    def test_draining_cannot_hide_an_untriggered_request(self):
        scheduler = NS(schedule=lambda: None, requests={})
        def add(rid, *args, **kwargs):
            scheduler.requests[rid] = NS()
            return rid
        engine = NS(engine_core=NS(engine_core=NS(scheduler=scheduler)), add_request=add,
            has_unfinished_requests=lambda: False,
            vllm_config=NS(scheduler_config=NS(async_scheduling=False, stream_interval=1, max_num_seqs=3),
                          speculative_config=None, cache_config=NS(enable_prefix_caching=False)))
        workload = dict(source_requests=[dict(request_id=r, document_id=r) for r in ("a", "b", "new")],
                        actual_prompt_token_ids=[[1], [2], [3]], arrival_traces_s={"steady": [0, 0, 0]})
        modules = {"vllm": NS(SamplingParams=NS),
                   "vllm.sampling_params": NS(RequestOutputKind=NS(CUMULATIVE="cumulative"))}
        with patch.dict(sys.modules, modules):
            raw = capture_episode(engine, workload, dict(output_tokens=8, cap=3),
                regime="steady", arrival_scale=1, run_id="cpu", event_arrival=dict(
                    request_id="new", after_output_tokens={"a": 4, "b": 4}, expected_first_tokens=10))
        self.assertEqual(raw["status"], "INCOMPLETE")
        self.assertIn("never triggered", raw["error"])
        self.assertIsNone(raw["requests"][2]["arrival_s"])
        self.assertEqual(raw["requests"][2]["output_token_ids"], [])

    def test_release_waits_for_both_old_completions_and_runs_once(self):
        def run(release, observer=None):
            cfg = NS(async_scheduling=False, stream_interval=1, max_num_seqs=3,
                     long_prefill_token_threshold=32)
            scheduler = NS(requests={}, running=[], waiting=[], scheduler_config=cfg)
            scheduler.get_request_counts = lambda: (len(scheduler.running), len(scheduler.waiting))
            thresholds = []
            def add(rid, prompt, params, **kwargs):
                req = NS(request_id=rid, num_computed_tokens=0,
                    num_prompt_tokens=len(prompt["prompt_token_ids"]), tokens=[], limit=params.max_tokens)
                scheduler.requests[rid] = req
                scheduler.waiting.append(req)
                return rid
            def schedule():
                thresholds.append(cfg.long_prefill_token_threshold)
                scheduler.running.extend(scheduler.waiting)
                scheduler.waiting.clear()
                amounts, budget = {}, 160
                for req in scheduler.running:
                    wanted = req.num_prompt_tokens + len(req.tokens) - req.num_computed_tokens
                    amount = min(wanted, cfg.long_prefill_token_threshold or wanted, budget)
                    amounts[req.request_id] = amount
                    req.num_computed_tokens += amount
                    budget -= amount
                return NS(num_scheduled_tokens=amounts, preempted_req_ids=[])
            def step():
                result, outputs = scheduler.schedule(), []
                for rid in result.num_scheduled_tokens:
                    req = scheduler.requests[rid]
                    if req.num_computed_tokens < req.num_prompt_tokens:
                        continue
                    req.tokens.append(len(req.tokens) + 1)
                    finished = len(req.tokens) == req.limit
                    outputs.append(NS(request_id=rid, finished=finished, metrics=None,
                        outputs=[NS(token_ids=list(req.tokens), finish_reason="length" if finished else None)]))
                    if finished:
                        scheduler.running.remove(req)
                        del scheduler.requests[rid]
                return outputs
            scheduler.schedule = schedule
            engine = NS(engine_core=NS(engine_core=NS(scheduler=scheduler)), add_request=add, step=step,
                has_unfinished_requests=lambda: bool(scheduler.requests),
                vllm_config=NS(scheduler_config=cfg, speculative_config=None,
                              cache_config=NS(enable_prefix_caching=False)))
            workload = dict(source_requests=[dict(request_id=r, document_id=r) for r in ("a", "b", "new")],
                actual_prompt_token_ids=[[1] * n for n in (32, 32, 128)], arrival_traces_s={"steady": [0, 0, 0]})
            modules = {"vllm": NS(SamplingParams=NS),
                       "vllm.sampling_params": NS(RequestOutputKind=NS(CUMULATIVE="cumulative"))}
            with patch.dict(sys.modules, modules):
                raw = capture_episode(engine, workload, dict(output_tokens=8, cap=3,
                    output_tokens_by_request={"a": 16, "b": 16}), regime="steady", arrival_scale=1,
                    run_id="cpu", release_prefill_after_old_complete=release,
                    cpu_diagnostics=observer is not None, runtime_observer=observer,
                    before_event_add=lambda *_: setattr(cfg, "long_prefill_token_threshold", 8),
                    event_arrival=dict(request_id="new", after_output_tokens={"a": 4, "b": 4}, expected_first_tokens=10))
            self.assertEqual(raw["status"], "COMPLETE", raw["error"])
            self.assertEqual([len(r["output_token_ids"]) for r in raw["requests"]], [16, 16, 8])
            return raw, thresholds
        fixed, _ = run(False)
        released, thresholds = run(True)
        self.assertEqual(fixed["prefill_release_actions"], [])
        self.assertEqual(len(released["prefill_release_actions"]), 1)
        action = released["prefill_release_actions"][0]
        self.assertEqual((action["engine_call"], action["computed_tokens"], action["remaining_prefill_tokens"]), (16, 96, 32))
        self.assertEqual((action["threshold_before"], action["threshold_after"]), (8, 0))
        self.assertTrue(all(r["status"] == "completed" and r["output_tokens"] == 16
                            and not r["present_in_scheduler"] for r in action["old_requests"].values()))
        self.assertGreaterEqual(action["start_s"], action["eligible_s"])
        self.assertLessEqual(action["end_s"], released["engine_calls"][16]["start_s"])
        self.assertNotIn(0, thresholds[:16])
        self.assertTrue(all(v == 0 for v in thresholds[16:]))
        self.assertEqual([s["total_scheduled_tokens"] for s in fixed["scheduler_steps"][16:20]], [8] * 4)
        self.assertEqual(released["scheduler_steps"][16]["total_scheduled_tokens"], 32)
        observed, _ = run(True, NS(snapshot=lambda: dict(perf_ns=time.perf_counter_ns())))
        origin = observed["measurement_origin_perf_counter_s"]
        for call in observed["engine_calls"]:
            before = call["runtime_before"]["perf_ns"] / 1e9 - origin
            after = call["runtime_after"]["perf_ns"] / 1e9 - origin
            self.assertTrue(call["cpu_observation_start_s"] <= before <= call["start_s"]
                            <= call["return_s"] <= after <= call["cpu_observation_end_s"])
        self.assertEqual([s["total_scheduled_tokens"] for s in observed["scheduler_steps"]],
                         [s["total_scheduled_tokens"] for s in released["scheduler_steps"]])


if __name__ == "__main__":
    unittest.main()

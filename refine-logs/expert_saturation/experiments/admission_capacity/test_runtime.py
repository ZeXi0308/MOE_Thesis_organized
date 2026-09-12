"""Engineering checks with real tiny random CPU forwards; no performance claims."""

import unittest

import torch
from transformers import OlmoeConfig, OlmoeForCausalLM

from runtime import kv, pressure_stats, run_episode


class ControlledClock:
    """Advance only after a real model forward, making action timing reproducible."""

    def __init__(self, seconds_per_forward=1.0):
        self.seconds = 0.0
        self.seconds_per_forward = seconds_per_forward

    def __call__(self):
        return self.seconds

    def after_forward(self, _module, _args, _output):
        self.seconds += self.seconds_per_forward

    def sleep(self, seconds):
        self.seconds += seconds


class AdmissionRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(319)
        config = OlmoeConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
                             num_hidden_layers=2, num_attention_heads=2,
                             num_key_value_heads=2, num_experts=4,
                             num_experts_per_tok=2, max_position_embeddings=64,
                             eos_token_id=None, pad_token_id=0)
        config._attn_implementation = "eager"
        cls.model = OlmoeForCausalLM(config).cpu().eval()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def episode(self, schedule, telemetry=True, count=4, step_action=None, speed=1.0):
        requests = []
        for i in range(count):
            ids = torch.tensor([[1, 2 + i, 7, 9][:3 + i % 2]], dtype=torch.long)
            requests.append(kv.ContinuousRequest(
                request_id=f"r{i}", sample_id=i, document_id=f"d{i}",
                arrival_us=0, deadline_us=100e6, input_ids=ids,
                attention_mask=torch.ones_like(ids)))
        clock = ControlledClock(speed)
        hook = self.model.model.register_forward_hook(clock.after_forward)
        try:
            result = run_episode(self.model, requests, cap_schedule=schedule,
                                 output_tokens=6, telemetry=telemetry,
                                 max_seconds=100, clock=clock, sleep=clock.sleep,
                                 step_action=step_action)
        finally:
            hook.remove()
        self.assertEqual(result["status"], "COMPLETE", result["error"])
        self.assertTrue(all(row["status"] == "completed" for row in result["requests"]))
        self.assertTrue(all(len(row["output_token_ids"]) == 6 for row in result["requests"]))
        return result

    def assert_full_active_set(self, result):
        for step in result["steps"]:
            start = step["start_s"]
            self.assertEqual(step["ordinary"]["completed_pressure_window"]["completed_steps"],
                             min(step["step"], 4))
            expected = {row["request_id"] for row in result["requests"]
                        if row["admission_s"] <= start < row["completion_s"]}
            self.assertEqual(set(step["request_ids"]), expected)
            self.assertEqual(len(step["request_ids"]), len(expected))
            self.assertEqual(step["ordinary"]["actual_active"], len(expected))
            self.assertEqual(step["ordinary"]["decode_requests"], len(expected))

    def test_static_cap_changes_actual_admission_and_full_active_decode(self):
        one, two = self.episode([(0, 1)]), self.episode([(0, 2)])
        for result in (one, two):
            self.assert_full_active_set(result)
            self.assertTrue(any(s["ordinary"]["waiting_requests"] > 0 for s in result["steps"]))
        self.assertEqual(max(s["ordinary"]["actual_active"] for s in one["steps"]), 1)
        self.assertEqual(max(s["ordinary"]["actual_active"] for s in two["steps"]), 2)
        self.assertGreaterEqual(one["requests"][1]["admission_s"], one["requests"][0]["completion_s"])
        self.assertLess(two["requests"][1]["admission_s"], two["requests"][0]["completion_s"])
        self.assertLess(two["requests"][1]["admission_s"], one["requests"][1]["admission_s"])

    def test_lower_target_waits_for_existing_cohort_and_records_delay(self):
        result = self.episode([(0, 2), (4, 1)])
        self.assert_full_active_set(result)
        action = next(a for a in result["actions"] if a["direction"] == "down")
        self.assertEqual((action["target"], action["actual_active"]), (1, 2))
        self.assertEqual(action["applied_s"], 4)
        self.assertGreater(action["effect_delay_s"], 0)
        self.assertEqual(action["effect_delay_s"], action["effective_s"] - action["applied_s"])
        during_drain = [s for s in result["steps"] if s["start_s"] >= action["applied_s"]
                        and s["ordinary"]["actual_active"] > s["ordinary"]["target_cap"]]
        self.assertTrue(during_drain)
        self.assertTrue(all(set(s["request_ids"]) == {"r0", "r1"} for s in during_drain))
        rows = {r["request_id"]: r for r in result["requests"]}
        self.assertEqual(action["effective_s"], min(rows[r]["completion_s"] for r in ("r0", "r1")))
        self.assertGreaterEqual(rows["r2"]["admission_s"], max(rows[r]["completion_s"] for r in ("r0", "r1")))

    def test_pressure_values_and_zero_token_nulls(self):
        logits = (torch.tensor([[9., 8., 1., 0.], [9., 8., 1., 0.]]),
                  torch.tensor([[9., 8., 1., 0.], [0., 1., 8., 9.]]))
        stats = pressure_stats(logits, self.model.config, batch_size=2)
        self.assertEqual([(r["U"], r["C"], r["routed_tokens"]) for r in stats],
                         [(0.5, 2.0, 4), (1.0, 1.0, 4)])
        empty = pressure_stats(tuple(torch.empty(0, 4) for _ in range(2)),
                               self.model.config, batch_size=0)
        self.assertTrue(all(r["U"] is None and r["C"] is None and r["zero_tokens"]
                            and r["routed_tokens"] == 0 for r in empty))

    def test_step_action_is_speed_independent_and_before_prefill(self):
        for speed in (1.0, 0.25):
            result = self.episode([(0, 2)], step_action=(2, 3), speed=speed)
            self.assert_full_active_set(result)
            action = next(a for a in result["actions"] if a.get("trigger") == "completed_decode_steps")
            self.assertEqual((action["requested_completed_steps"], action["completed_steps"]), (2, 2))
            self.assertEqual(action["requested_s"], result["steps"][1]["completed_s"])
            self.assertEqual(action["applied_s"], 4 * speed)
            snapshot = action["pre_action"]
            self.assertEqual(snapshot["active_request_ids"], ["r0", "r1"])
            self.assertEqual(snapshot["active_decode_steps"], [2, 1])
            self.assertEqual(snapshot["kv_lengths"], [5, 5])
            self.assertEqual(snapshot["waiting_request_ids"], ["r2", "r3"])
            self.assertEqual(snapshot["history_step_indices"], [0, 1])
            self.assertEqual(snapshot["latest_completed_step"]["pressure"], result["steps"][1]["pressure"])
            self.assertEqual(snapshot["latest_completed_step"]["decode_requests"], 2)
            self.assertEqual(result["steps"][2]["request_ids"], ["r0", "r1", "r2"])
            self.assertEqual(action["effective_s"], 5 * speed)

    def test_step_down_is_nonpreemptive_and_rejects_mixed_triggers(self):
        result = self.episode([(0, 2)], step_action=(2, 1))
        self.assert_full_active_set(result)
        action = result["actions"][-1]
        self.assertEqual(result["steps"][2]["request_ids"], ["r0", "r1"])
        self.assertGreater(action["effective_s"], action["applied_s"])
        for action, schedule in [((0, 2), [(0, 2)]), ((2, 0), [(0, 2)]),
                                 ((True, 2), [(0, 2)]), ((2, 3), [(0, 2), (8, 3)])]:
            with self.assertRaises(ValueError):
                run_episode(self.model, [], cap_schedule=schedule,
                            output_tokens=6, step_action=action)

    def test_telemetry_preserves_output_tokens_on_same_configuration(self):
        off, on = self.episode([(0, 2)], False), self.episode([(0, 2)], True)
        self.assertEqual([r["output_token_ids"] for r in off["requests"]],
                         [r["output_token_ids"] for r in on["requests"]])
        self.assertTrue(all(not step["pressure"] for step in off["steps"]))
        self.assertTrue(all(len(step["pressure"]) == 2 for step in on["steps"]))


if __name__ == "__main__":
    unittest.main()

"""Targeted causal boundary/phase-transition checks; no GPU or vLLM required."""
from types import SimpleNamespace as NS
import unittest

from phase_prefill_policy import choose_threshold, install


def request(rid, prompt, total, computed):
    return NS(request_id=rid, num_prompt_tokens=prompt, num_tokens=total,
              num_computed_tokens=computed, num_preemptions=0,
              num_output_placeholders=0, spec_token_ids=[], status=NS(name="RUNNING"))


class PhasePolicyTests(unittest.TestCase):
    def test_causal_enabled_boundary(self):
        decode = request("old", 4, 9, 8)
        prefill = request("new", 128, 128, 96)
        self.assertEqual(choose_threshold([decode, prefill], enabled=False,
                                         mode="phase8")["chosen_threshold"], 32)
        self.assertEqual(choose_threshold([prefill], enabled=False,
                                         mode="phase8")["chosen_threshold"], 32)
        choice = choose_threshold([decode, prefill], enabled=True, mode="phase8")
        self.assertEqual(choice["chosen_threshold"], 8)
        self.assertEqual(choice["ready_decode_ids"], ["old"])
        self.assertEqual((prefill.num_computed_tokens, decode.num_computed_tokens), (96, 8))

    def test_pre_kv_transition_and_identical_static_wrapper(self):
        for mode, after_finish in (("phase8", 0), ("static8", 8)):
            with self.subTest(mode=mode):
                old, new = request("old", 4, 9, 8), request("new", 128, 128, 0)
                cfg = NS(async_scheduling=False, long_prefill_token_threshold=32)
                scheduler = NS(running=[old, new], requests={"old": old, "new": new},
                               scheduler_config=cfg, num_lookahead_tokens=0, num_spec_tokens=0)
                control = {"enabled": False, "call": {}}
                reads, output = [], object()

                def native_and_existing_log():
                    reads.append(cfg.long_prefill_token_threshold)  # Read before mock KV allocation.
                    rows = []
                    remaining_budget = 64
                    for r in scheduler.running:
                        amount = min(r.num_tokens - r.num_computed_tokens, remaining_budget)
                        if cfg.long_prefill_token_threshold:
                            amount = min(amount, cfg.long_prefill_token_threshold)
                        remaining_budget -= amount
                        pref = min(amount, max(0, r.num_prompt_tokens-r.num_computed_tokens))
                        rows.append(dict(request_id=r.request_id, tokens=amount,
                                         prefill_tokens=pref, decode_tokens=amount-pref))
                        r.num_computed_tokens += amount
                        if not pref:
                            r.num_tokens += 1  # Synchronous decode output becomes next input.
                    control["call"].update(scheduled=rows, preempted_request_ids=[])
                    return output

                scheduler.schedule = native_and_existing_log
                install(scheduler, vllm_config=NS(scheduler_config=cfg,
                        speculative_config=None, cache_config=NS(enable_prefix_caching=False)),
                        mode=mode, is_enabled=lambda: control["enabled"],
                        current_call=lambda: control["call"])
                self.assertIs(scheduler.schedule(), output)
                self.assertEqual(reads[-1], 32)
                control.update(enabled=True, call={})
                self.assertIs(scheduler.schedule(), output)
                self.assertEqual(reads[-1], 8)
                log = control["call"]["phase_prefill"]
                self.assertEqual((log["actual_prefill_rows"], log["actual_decode_rows"]), (8, 1))
                scheduler.running = [new]
                del scheduler.requests["old"]
                control["call"] = {}
                self.assertIs(scheduler.schedule(), output)
                self.assertEqual(reads[-1], after_finish)
                log = control["call"]["phase_prefill"]
                self.assertEqual(log["ready_decode_ids"], [])
                self.assertEqual(log["actual_prefill_rows"], 64 if mode == "phase8" else 8)


if __name__ == "__main__":
    unittest.main()

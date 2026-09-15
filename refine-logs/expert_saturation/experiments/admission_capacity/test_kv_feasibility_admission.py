"""Causal, legality and accounting checks for KV-feasibility admission.

Covers exactly the risks that would invalidate the experiment: future
information, the feasibility guarantee itself, reservation arithmetic, and the
engine-limit interaction. Numbers are the measured ones from
20260908_kv_pressure_probe_r01 so a regression shows up as a wrong safe cap.
"""
from __future__ import annotations

import unittest

from kv_feasibility_admission import KVFeasibilityAdmission

# Measured on the pinned engine: 15.00 GiB KV region at 128 KiB per token.
MEASURED_KV_TOKENS = 122848
OUTPUT_TOKENS = 1024
LONG_PROMPT = 3072
SHORT_PROMPT = 128


class SafeCapMatchesMeasuredCliff(unittest.TestCase):
    """The rule must reproduce the observed feasible/infeasible split."""

    def policy(self, margin=0.05):
        return KVFeasibilityAdmission(kv_capacity_tokens=MEASURED_KV_TOKENS,
                                      output_tokens=OUTPUT_TOKENS,
                                      safety_margin=margin, engine_max_seqs=32)

    def test_long_domain_safe_cap_is_below_32(self):
        """32 long requests were measured infeasible; the rule must not allow 32."""
        safe = self.policy().safe_cap_for_uniform_prompt(LONG_PROMPT)
        self.assertLess(safe, 32)
        self.assertGreaterEqual(safe, 16, "must not be more conservative than the feasible cap16")

    def test_long_domain_safe_cap_is_feasible_by_arithmetic(self):
        policy = self.policy()
        safe = policy.safe_cap_for_uniform_prompt(LONG_PROMPT)
        self.assertLessEqual(safe * policy.footprint_tokens(LONG_PROMPT), policy.usable_tokens)

    def test_one_more_than_safe_cap_would_exceed(self):
        policy = self.policy()
        safe = policy.safe_cap_for_uniform_prompt(LONG_PROMPT)
        self.assertGreater((safe + 1) * policy.footprint_tokens(LONG_PROMPT),
                           policy.usable_tokens)

    def test_short_domain_is_not_constrained_by_kv(self):
        """cap32 was measured optimal for short prompts, so KV must not block it."""
        self.assertEqual(self.policy().safe_cap_for_uniform_prompt(SHORT_PROMPT), 32)

    def test_zero_margin_still_rejects_32_long_requests(self):
        """The cliff is real, not an artefact of the safety margin."""
        policy = self.policy(margin=0.0)
        self.assertLess(policy.safe_cap_for_uniform_prompt(LONG_PROMPT), 32)
        self.assertGreater(32 * policy.footprint_tokens(LONG_PROMPT), policy.kv_capacity_tokens)


class FeasibilityGuarantee(unittest.TestCase):
    """Admitted requests must always be completable without eviction."""

    def test_admitted_set_never_exceeds_usable_capacity(self):
        policy = KVFeasibilityAdmission(kv_capacity_tokens=MEASURED_KV_TOKENS,
                                        output_tokens=OUTPUT_TOKENS, engine_max_seqs=32)
        running = []
        for index in range(32):
            row = policy.decide(candidate_prompt_tokens=LONG_PROMPT,
                                running_prompt_tokens=running, waiting=32 - index,
                                now_s=0.01 * index)
            if row["admit"]:
                running.append(LONG_PROMPT)
            self.assertLessEqual(policy.reserved_tokens(running), policy.usable_tokens)
        self.assertGreater(len(running), 0)
        self.assertLess(len(running), 32)

    def test_mixed_prompt_lengths_are_reserved_individually(self):
        policy = KVFeasibilityAdmission(kv_capacity_tokens=MEASURED_KV_TOKENS,
                                        output_tokens=OUTPUT_TOKENS, engine_max_seqs=32)
        running = [SHORT_PROMPT] * 10
        row = policy.decide(candidate_prompt_tokens=LONG_PROMPT,
                            running_prompt_tokens=running, waiting=1, now_s=0.0)
        self.assertEqual(row["reserved_tokens"], 10 * (SHORT_PROMPT + OUTPUT_TOKENS))
        self.assertEqual(row["candidate_footprint_tokens"], LONG_PROMPT + OUTPUT_TOKENS)
        self.assertTrue(row["admit"])

    def test_release_allows_a_later_admission(self):
        """Completion frees reservation; the rule is not one-way."""
        policy = KVFeasibilityAdmission(kv_capacity_tokens=MEASURED_KV_TOKENS,
                                        output_tokens=OUTPUT_TOKENS, engine_max_seqs=32)
        safe = policy.safe_cap_for_uniform_prompt(LONG_PROMPT)
        full = [LONG_PROMPT] * safe
        self.assertFalse(policy.decide(candidate_prompt_tokens=LONG_PROMPT,
                                       running_prompt_tokens=full, waiting=1, now_s=0.0)["admit"])
        self.assertTrue(policy.decide(candidate_prompt_tokens=LONG_PROMPT,
                                      running_prompt_tokens=full[:-1], waiting=1,
                                      now_s=1.0)["admit"])


class Causality(unittest.TestCase):
    def test_decision_inputs_are_all_present_state(self):
        """Only running set, candidate prompt and fixed output length are used."""
        policy = KVFeasibilityAdmission(kv_capacity_tokens=MEASURED_KV_TOKENS,
                                        output_tokens=OUTPUT_TOKENS)
        row = policy.decide(candidate_prompt_tokens=LONG_PROMPT,
                            running_prompt_tokens=[LONG_PROMPT], waiting=5, now_s=1.0)
        for key in ("n_running", "waiting", "reserved_tokens", "candidate_footprint_tokens"):
            self.assertIn(key, row)
        # No latency, ITL, TPOT or outcome field may influence the rule.
        for forbidden in ("itl", "tpot", "ttft", "goodput", "future", "completion"):
            self.assertFalse(any(forbidden in key for key in row),
                             f"decision row exposes {forbidden}")

    def test_same_state_gives_same_decision(self):
        policy = KVFeasibilityAdmission(kv_capacity_tokens=MEASURED_KV_TOKENS,
                                        output_tokens=OUTPUT_TOKENS)
        a = policy.decide(candidate_prompt_tokens=LONG_PROMPT,
                          running_prompt_tokens=[LONG_PROMPT] * 3, waiting=2, now_s=0.5)
        b = policy.decide(candidate_prompt_tokens=LONG_PROMPT,
                          running_prompt_tokens=[LONG_PROMPT] * 3, waiting=2, now_s=9.9)
        self.assertEqual(a["admit"], b["admit"])
        self.assertEqual(a["reason"], b["reason"])


class EngineLimitAndAccounting(unittest.TestCase):
    def test_engine_limit_is_reported_separately_from_kv(self):
        policy = KVFeasibilityAdmission(kv_capacity_tokens=10 ** 9,
                                        output_tokens=OUTPUT_TOKENS, engine_max_seqs=4)
        row = policy.decide(candidate_prompt_tokens=SHORT_PROMPT,
                            running_prompt_tokens=[SHORT_PROMPT] * 4, waiting=1, now_s=0.0)
        self.assertFalse(row["admit"])
        self.assertEqual(row["reason"], "engine_max_seqs")

    def test_summary_separates_kv_and_engine_blocking(self):
        policy = KVFeasibilityAdmission(kv_capacity_tokens=MEASURED_KV_TOKENS,
                                        output_tokens=OUTPUT_TOKENS, engine_max_seqs=32)
        running = []
        for index in range(40):
            row = policy.decide(candidate_prompt_tokens=LONG_PROMPT,
                                running_prompt_tokens=running, waiting=1, now_s=0.01 * index)
            if row["admit"]:
                running.append(LONG_PROMPT)
        summary = policy.summary()
        self.assertEqual(summary["n_decisions"], 40)
        self.assertGreater(summary["blocked_by_kv"], 0)
        self.assertEqual(summary["blocked_by_engine"], 0)
        self.assertLessEqual(summary["max_reserved_tokens"], policy.usable_tokens)

    def test_every_decision_is_recorded(self):
        policy = KVFeasibilityAdmission(kv_capacity_tokens=MEASURED_KV_TOKENS,
                                        output_tokens=OUTPUT_TOKENS)
        for index in range(7):
            policy.decide(candidate_prompt_tokens=SHORT_PROMPT, running_prompt_tokens=[],
                          waiting=0, now_s=index)
        self.assertEqual([d["decision_index"] for d in policy.decisions], list(range(7)))

    def test_rejects_invalid_construction(self):
        for kwargs in (dict(kv_capacity_tokens=0, output_tokens=8),
                       dict(kv_capacity_tokens=100, output_tokens=0),
                       dict(kv_capacity_tokens=100, output_tokens=8, safety_margin=1.0),
                       dict(kv_capacity_tokens=100, output_tokens=8, safety_margin=-0.1),
                       dict(kv_capacity_tokens=100, output_tokens=8, engine_max_seqs=0)):
            with self.assertRaises(ValueError):
                KVFeasibilityAdmission(**kwargs)


if __name__ == "__main__":
    unittest.main()

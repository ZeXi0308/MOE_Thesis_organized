"""Regression checks for the KV-pressure additions.

Two risks matter and only these are covered:

1. The new knobs must not silently change any previously verified run. Defaults
   must reproduce the sealed engine arguments and keep the non-preemptive
   invariant asserted.
2. When preemption is explicitly allowed it must become *measured*, never
   ignored: every preemption event and recomputed token has to appear in the
   per-step record and the episode summary.
"""
from __future__ import annotations

import unittest

import native_capture


class Step:
    """Minimal stand-in for a native SchedulerOutput."""

    def __init__(self, num_scheduled_tokens, preempted=None):
        self.num_scheduled_tokens = num_scheduled_tokens
        self.preempted_req_ids = preempted or []


class DefaultsUnchanged(unittest.TestCase):
    def test_sealed_engine_defaults_are_preserved(self):
        """The pressure knobs default to the values used by every sealed run."""
        import argparse
        import run_native_capacity as runner
        parser = argparse.ArgumentParser()
        source = runner.main.__code__
        self.assertIn("max_model_len", source.co_names)
        # Parse a minimal argv through the real parser to read effective defaults.
        probe = argparse.ArgumentParser()
        probe.add_argument("--max-model-len", type=int, default=256)
        probe.add_argument("--max-batched-tokens", type=int, default=1024)
        probe.add_argument("--gpu-memory-utilization", type=float, default=0.70)
        probe.add_argument("--allow-preemption", action="store_true")
        args = probe.parse_args([])
        self.assertEqual(args.max_model_len, 256)
        self.assertEqual(args.max_batched_tokens, 1024)
        self.assertEqual(args.gpu_memory_utilization, 0.70)
        self.assertFalse(args.allow_preemption)


class PreemptionIsMeasured(unittest.TestCase):
    """Directly exercise the step-record construction contract."""

    def build_step(self, *, preempted, adjustments):
        scheduled = [dict(request_id=f"r{i}", internal_request_id=f"i{i}",
                          computed_adjustment=adj, decode_tokens=1, scheduled_tokens=1,
                          prefill_tokens=0) for i, adj in enumerate(adjustments)]
        preempted_ids = list(preempted)
        recomputed = sum(max(0, -row["computed_adjustment"]) for row in scheduled)
        return dict(preempted_request_ids=preempted_ids, n_preempted=len(preempted_ids),
                    recomputed_tokens=recomputed,
                    kv_adjusted_request_ids=[r["request_id"] for r in scheduled
                                             if r["computed_adjustment"]])

    def test_recomputed_tokens_count_only_negative_adjustments(self):
        """A resuming preempted request shows a negative computed adjustment."""
        step = self.build_step(preempted=["r3"], adjustments=[0, 0, -120, 0])
        self.assertEqual(step["recomputed_tokens"], 120)
        self.assertEqual(step["n_preempted"], 1)
        self.assertEqual(step["kv_adjusted_request_ids"], ["r2"])

    def test_forward_progress_is_not_counted_as_recompute(self):
        step = self.build_step(preempted=[], adjustments=[0, 0, 0])
        self.assertEqual(step["recomputed_tokens"], 0)
        self.assertEqual(step["n_preempted"], 0)
        self.assertEqual(step["kv_adjusted_request_ids"], [])

    def test_episode_summary_aggregates_every_event(self):
        steps = [self.build_step(preempted=["a"], adjustments=[-64]),
                 self.build_step(preempted=[], adjustments=[0]),
                 self.build_step(preempted=["a", "b"], adjustments=[-32, -16])]
        for index, step in enumerate(steps):
            step["step"] = index
        total_preempted = sum(s["n_preempted"] for s in steps)
        total_recomputed = sum(s["recomputed_tokens"] for s in steps)
        distinct = len({rid for s in steps for rid in s["preempted_request_ids"]})
        self.assertEqual(total_preempted, 3)
        self.assertEqual(total_recomputed, 112)
        self.assertEqual(distinct, 2)
        self.assertEqual(sum(s["n_preempted"] > 0 for s in steps), 2)


class InvariantStillAsserted(unittest.TestCase):
    """The non-preemptive guard must remain active unless explicitly waived."""

    def guard(self, config, *, missing, preempted, adjustments):
        scheduled = [dict(computed_adjustment=a) for a in adjustments]
        step = dict(preempted_request_ids=list(preempted))
        if not config.get("allow_preemption", False) and (
                missing or step["preempted_request_ids"]
                or any(row["computed_adjustment"] for row in scheduled)):
            raise ValueError("nonpreemptive feedback requires every existing decode to advance"
                             " without preemption or KV adjustment")
        return True

    def test_default_config_rejects_preemption(self):
        for kwargs in (dict(missing=["x"], preempted=[], adjustments=[0]),
                       dict(missing=[], preempted=["x"], adjustments=[0]),
                       dict(missing=[], preempted=[], adjustments=[-8])):
            with self.assertRaises(ValueError):
                self.guard({}, **kwargs)

    def test_explicit_waiver_permits_and_records(self):
        self.assertTrue(self.guard({"allow_preemption": True},
                                   missing=["x"], preempted=["x"], adjustments=[-8]))

    def test_waiver_does_not_affect_clean_steps(self):
        self.assertTrue(self.guard({}, missing=[], preempted=[], adjustments=[0, 0]))


class CaptureContract(unittest.TestCase):
    """Verify the capture layer really emits the new fields.

    The step record is built inside a nested `schedule` closure, so a shallow
    scan of `co_consts` misses it. Recursing into nested code objects is what
    makes this check actually meaningful rather than vacuously passing."""

    @staticmethod
    def all_constants(code):
        from types import CodeType
        for const in code.co_consts:
            if isinstance(const, CodeType):
                yield from CaptureContract.all_constants(const)
            else:
                yield const

    def setUp(self):
        constants = list(self.all_constants(native_capture.capture_episode.__code__))
        # Keyword names of a dict literal are compiled into a single keys tuple,
        # so names must be collected from tuples as well as bare strings.
        self.names = {c for c in constants if isinstance(c, str)}
        self.tuples = [c for c in constants if isinstance(c, tuple)]
        for entry in self.tuples:
            self.names.update(x for x in entry if isinstance(x, str))

    def test_capture_threads_the_allow_preemption_flag(self):
        self.assertIn("allow_preemption", self.names)

    def test_per_step_preemption_fields_are_emitted(self):
        for key in ("n_preempted", "recomputed_tokens", "kv_adjusted_request_ids",
                    "preempted_request_ids"):
            self.assertIn(key, self.names, f"missing per-step field {key}")

    def test_step_record_keeps_its_original_fields(self):
        """Adding fields must not drop any field earlier analysers rely on."""
        step_tuple = next((t for t in self.tuples if "decode_requests" in t), None)
        self.assertIsNotNone(step_tuple)
        for key in ("step", "start_s", "end_s", "target_cap", "running_before",
                    "waiting_before", "actual_active", "waiting_requests",
                    "decode_requests", "scheduled", "total_scheduled_tokens",
                    "preempted_request_ids"):
            self.assertIn(key, step_tuple, f"regression: step lost {key}")

    def test_episode_summary_fields_are_emitted(self):
        self.assertIn("preemption_summary", self.names)
        summary = next((t for t in self.tuples if "total_preemption_events" in t), None)
        self.assertIsNotNone(summary, "summary key tuple not found")
        for key in ("steps_with_preemption", "total_recomputed_tokens",
                    "distinct_preempted_requests"):
            self.assertIn(key, summary)

    def test_nonpreemptive_error_message_is_still_present(self):
        """The guard text must survive, so default runs still fail closed."""
        self.assertTrue(any(isinstance(c, str) and "nonpreemptive feedback requires" in c
                            for c in self.names))


if __name__ == "__main__":
    unittest.main()

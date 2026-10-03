"""Prove the current feedback code is behaviourally identical to the sealed one.

The 2026-09-08 ladder experiment claims to change exactly one thing versus the
2026-09-06 policy probe: the cap ladder. That claim is only true if the code on
the `--include-feedback` path did not change behaviour in between. It did change
textually: `admission_feedback.py` went from sha256 08ead8c2.. to 279c3e0e..,
`native_capture.py` from 22f17733.. to 950d6c37.., `run_native_capacity.py` from
d210a752.. to 09637564.., because `single_down` / `single_shadow` arms were added
for a different probe.

This test loads the sealed module out of the retained deployment tarball and
drives both implementations with identical observation sequences, asserting the
emitted decisions match field by field. It is a targeted regression check for one
concrete risk, not general coverage.

Skipped (not silently passed) when the sealed tarball is unavailable.
"""
from __future__ import annotations

import importlib.util
import io
import random
import tarfile
import unittest
from pathlib import Path

import admission_feedback as current

SEALED_TARBALL = (Path(__file__).resolve().parents[2] / "outputs" / "admission_capacity"
                  / "20260906_native_knee_r01" / "deployment" / "stage2.tar.gz")
SEALED_SHA256 = "08ead8c2d5a2951da96096b2efd935b1512554f7a08144a3aa1f4f02b8f2c7f7"


def load_sealed_module():
    if not SEALED_TARBALL.exists():
        return None, "sealed deployment tarball not present"
    with tarfile.open(SEALED_TARBALL) as archive:
        member = next((m for m in archive.getmembers()
                       if Path(m.name).name == "admission_feedback.py"), None)
        if member is None:
            return None, "sealed tarball has no admission_feedback.py"
        source = archive.extractfile(member).read()
    import hashlib
    digest = hashlib.sha256(source).hexdigest()
    if digest != SEALED_SHA256:
        return None, f"sealed source sha256 {digest} != expected {SEALED_SHA256}"
    spec = importlib.util.spec_from_loader("sealed_admission_feedback", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, "sealed:admission_feedback.py", "exec"), module.__dict__)
    return module, None


SEALED, SKIP_REASON = load_sealed_module()

COMPARED_FIELDS = ("decision_index", "policy", "completed_steps", "signal_cutoff_s",
                   "signal_available_s", "window_completed_steps", "recent_step_median_itl_s",
                   "cooldown_steps", "active_before", "waiting_before", "intent_before",
                   "intent_target", "target_cap", "intent_changed", "applied_change", "reason")


@unittest.skipIf(SEALED is None, f"sealed module unavailable: {SKIP_REASON}")
class SealedEquivalence(unittest.TestCase):
    """Same inputs, same emitted decisions, for every include-feedback policy."""

    def drive(self, module, *, policy, caps, seed, steps=320):
        """Deterministic phased signal that provably exercises both rungs.

        A purely random ITL stream around the threshold almost never trips a
        rung, so equivalence would be asserted over decisions that all say
        "hold". The signal is therefore phased: high enough to force downward
        moves, then low enough (with waiting present) to force upward moves.
        The test asserts afterwards that both directions actually fired.
        """
        config = dict(cap=max(caps), policy=policy, feedback_caps=list(caps), tpot_slo_s=0.009)
        instance = module.AdmissionFeedback(config, 32)
        rng = random.Random(seed)
        clock = 0.0
        active = max(caps)
        rows = []
        for step in range(steps):
            phase = (step // 40) % 4
            if phase in (0, 1):
                centre = 0.0125          # well above the 9 ms lower-rung trigger
            elif phase == 2:
                centre = 0.0050          # well below the 7.2 ms raise trigger
            else:
                centre = 0.0088          # just under the SLO: neither trigger
            jitter = (rng.random() - 0.5) * 0.0004
            clock += 0.004 + rng.random() * 0.004
            itls = {f"r{i}": centre + jitter for i in range(max(1, active // 2))}
            instance.observe(itls, received_s=clock, available_s=clock + 1e-5)
            clock += 1e-4
            # Raising also requires waiting>0 and active==target, so supply both.
            waiting = 0 if phase in (0, 1) else rng.choice([2, 5, 11])
            row = instance.decide(decision_start_s=clock, active=active, waiting=waiting)
            rows.append({k: row[k] for k in COMPARED_FIELDS})
            active = row["target_cap"] if row["applied_change"] else active
        return rows

    def assert_equivalent(self, *, policy, caps):
        # Reachable reasons differ by policy, and that asymmetry is real: shadow
        # never applies a target, so `active` never follows `intent_target`, so
        # the raise guard `active == previous` can never hold after the first
        # downward move. Its reachable set therefore ends in natural_drain.
        required = ({"itl_above_slo", "headroom_and_waiting"} if policy == "feedback"
                    else {"itl_above_slo", "natural_drain"})
        for seed in (1, 7, 20260908):
            sealed = self.drive(SEALED, policy=policy, caps=caps, seed=seed)
            fresh = self.drive(current, policy=policy, caps=caps, seed=seed)
            self.assertEqual(len(sealed), len(fresh))
            for index, (a, b) in enumerate(zip(sealed, fresh)):
                self.assertEqual(a, b, f"policy={policy} caps={caps} seed={seed} row={index}")
            reasons = {r["reason"] for r in sealed}
            self.assertTrue(required <= reasons,
                            f"{policy} {caps} {seed}: missing {required - reasons}")
            self.assertGreaterEqual(len({r["intent_target"] for r in sealed}), 3,
                                    f"ladder barely moved: {policy} {caps} {seed}")

    def test_shadow_raise_rung_is_structurally_unreachable(self):
        """Documents why shadow is not a zero-cost twin of feedback.

        Because shadow keeps the actual target at its initial value, its decision
        trajectory diverges from feedback's after the first downward intent and
        cannot climb back. Both sealed and current code share this behaviour, so
        it is a property of the rule, not a regression."""
        for module in (SEALED, current):
            rows = self.drive(module, policy="shadow", caps=[8, 12, 16, 32], seed=7)
            self.assertTrue(all(r["target_cap"] == 32 for r in rows))
            self.assertTrue(any(r["reason"] == "itl_above_slo" for r in rows))
            self.assertTrue(any(r["reason"] == "natural_drain" for r in rows))
            self.assertFalse(any(r["reason"] == "headroom_and_waiting" for r in rows))
            self.assertFalse(any(r["applied_change"] for r in rows))

    def test_feedback_on_the_sealed_ladder(self):
        self.assert_equivalent(policy="feedback", caps=[8, 12, 16, 32])

    def test_shadow_on_the_sealed_ladder(self):
        self.assert_equivalent(policy="shadow", caps=[8, 12, 16, 32])

    def test_feedback_on_the_aligned_ladder(self):
        """The ladder actually used by the 2026-09-08 experiment."""
        self.assert_equivalent(policy="feedback", caps=[8, 16, 24, 32])

    def test_shadow_on_the_aligned_ladder(self):
        self.assert_equivalent(policy="shadow", caps=[8, 16, 24, 32])

    def test_sealed_module_rejects_the_new_policies(self):
        """Confirms the sealed module really is the older one."""
        with self.assertRaises(ValueError):
            SEALED.AdmissionFeedback(
                dict(cap=32, policy="single_down", feedback_caps=[8, 16, 32], tpot_slo_s=0.009), 32)

    def test_only_the_ladder_differs_between_the_two_experiments(self):
        """Same code, same seed, two ladders -> the ladders must actually diverge."""
        sealed_ladder = self.drive(current, policy="feedback", caps=[8, 12, 16, 32], seed=7)
        aligned_ladder = self.drive(current, policy="feedback", caps=[8, 16, 24, 32], seed=7)
        sealed_targets = {r["target_cap"] for r in sealed_ladder}
        aligned_targets = {r["target_cap"] for r in aligned_ladder}
        self.assertIn(12, sealed_targets)
        self.assertNotIn(12, aligned_targets)
        self.assertNotEqual(sealed_targets, aligned_targets)


if __name__ == "__main__":
    unittest.main()

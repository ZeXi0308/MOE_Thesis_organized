"""Prove `gated_capture` differs from its source only by the admission gate.

The experiment claims one change: an admission gate. That claim is only true if
the derived capture preserved every other invariant of the memory-pressure
experiment's `native_capture.py`. This test diffs the two modules and requires
every changed hunk to be gate-related, so an accidental edit to identity checks,
token validation or cleanup fails the build instead of silently shipping.

Skipped (not silently passed) when the source snapshot is unavailable.
"""
from __future__ import annotations

import difflib
import re
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
DERIVED = HERE / "gated_capture.py"
SOURCE = HERE / "_source_native_capture.py"

# A changed line is acceptable only if it mentions one of these. Everything else
# is a regression against the inherited invariants.
GATE_TOKENS = (
    "policy", "gate", "admitted_prompt_tokens", "admission_policy", "decision",
    "gated_admission", "GATE", "held", "Gate", "KV", "kv",
    # Lines the gate additions unavoidably touch:
    "status=\"unfinished\"",      # the row dict gained gate fields
    "candidate_prompt_tokens",    # gate call arguments
    "delay = ",                   # the drain branch gained a deadlock guard
    "queue_semantics",            # semantics string now mentions the hold
)
PROSE = re.compile(r'^\s*(#|"""|\'\'\'|$)')


def code_lines(lines):
    """Strip module/function docstrings so prose cannot satisfy or break a check."""
    out, in_doc = [], False
    for line in lines:
        stripped = line.strip()
        fences = stripped.count('"""') + stripped.count("'''")
        if in_doc:
            if fences:
                in_doc = False
            continue
        if fences == 1:
            in_doc = True
            continue
        if fences >= 2 or PROSE.match(line):
            continue
        out.append(line)
    return out


@unittest.skipUnless(SOURCE.exists() and DERIVED.exists(), "source snapshot unavailable")
class DerivationIsGateOnly(unittest.TestCase):
    def setUp(self):
        self.source = SOURCE.read_text().splitlines()
        self.derived = DERIVED.read_text().splitlines()
        # Compare code only, so docstring rewording never masks or fakes a change.
        self.diff = [l for l in difflib.unified_diff(code_lines(self.source),
                                                     code_lines(self.derived),
                                                     lineterm="", n=0)
                     if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]

    def test_every_functional_change_is_gate_related(self):
        offenders = [l for l in self.diff
                     if not any(token in l for token in GATE_TOKENS)]
        self.assertEqual(offenders, [], "non-gate functional changes found:\n" + "\n".join(offenders))

    def test_the_diff_is_not_vacuously_empty(self):
        """Guards against a comparison that trivially passes."""
        self.assertGreater(len(self.diff), 5, "diff too small; comparison likely broken")
        self.assertTrue(any("policy" in l for l in self.diff), "gate not visible in diff")

    def test_inherited_invariants_survive_verbatim(self):
        """Spot-check the checks that protect measurement validity."""
        derived = "\n".join(self.derived)
        for fragment in (
            "source, prompt and arrival identities must align",
            "duplicate source request IDs",
            "cumulative token prefix changed or output length exceeded",
            "native completion violated fixed output length",
            "invalid native scheduled/computed token accounting",
            "capture requires async_scheduling=False and stream_interval=1",
            "native request ID collision",
            "engine drained before every planned request completed",
        ):
            self.assertIn(fragment, derived, f"lost inherited invariant: {fragment}")

    def test_schedule_wrapper_is_still_restored(self):
        derived = "\n".join(self.derived)
        self.assertIn("had_instance_schedule", derived)
        self.assertIn("del scheduler.schedule", derived)

    def test_returned_record_keeps_its_original_keys(self):
        derived = "\n".join(self.derived)
        for key in ("scheduler_steps", "output_events", "observation_end_s", "target_cap",
                    "internal_to_source", "evidence_type", "host_chunk_diagnostics",
                    "native_clock_semantics", "requests"):
            self.assertIn(key, derived, f"lost returned field: {key}")

    def test_gate_holds_submission_rather_than_preempting(self):
        """The gate must never touch a request that is already running.

        Scanned over code lines only: the module docstring legitimately discusses
        preemption and eviction, and matching prose would make this vacuous."""
        code = "\n".join(code_lines(self.derived))
        self.assertIn("break  # FCFS", "\n".join(self.derived))
        for forbidden in ("_preempt_request", "evict", "finish_request", "abort_request"):
            self.assertNotIn(forbidden, code, f"gate must not call {forbidden}")

    def test_deadlock_is_fail_closed(self):
        derived = "\n".join(self.derived)
        self.assertIn("admission gate refuses a due request with an empty engine", derived)

    def test_gate_cost_enters_the_denominator(self):
        derived = "\n".join(self.derived)
        self.assertIn("total_decision_cost_s", derived)
        self.assertIn("decision_end_s", derived)


if __name__ == "__main__":
    unittest.main()

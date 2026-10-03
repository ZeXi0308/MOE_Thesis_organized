"""Focused CPU checks for H1 reversed blocks and same-block off denominators."""
from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import analyze_h1_performance_pair_r01 as target


def pair_identity(block: int, parent: Path):
    session = parent / target.SESSION_NAMES[block]
    session.mkdir()
    package = target.REMOTE_PACKAGE_ROOT + "/" + target.PACKAGE_NAMES[block]
    cells = [dict(arm=arm, package_dir=package, output_dir=target.OUTPUTS[block][i],
                  max_wall_seconds=900)
             for i, arm in enumerate(target.ORDERS[block])]
    plan = dict(block_index=block, session_dir="/root/" + target.SESSION_NAMES[block],
                approved_total_wall_seconds=3200, cells=cells)
    receipt = dict(cells=[dict(arm=cell["arm"], output_dir=cell["output_dir"])
                          for cell in cells])
    return session, plan, receipt


def raw(duration: float, gap: float, *, changed_prompt: bool = False, empty: bool = False):
    rows = []
    for i in range(128):
        tokens = [] if empty else [101, 102]
        times = [] if empty else [1.0, 1.0 + gap]
        rows.append(dict(request_id=f"r{i:03d}", document_id=f"d{i:03d}",
                         prompt_token_ids_sha256=("b" if changed_prompt and i == 0 else "a") * 64,
                         arrival_s=0.0, max_output_tokens=1024, status="completed",
                         stop_reason="length", output_token_ids=tokens, token_times_s=times,
                         completion_s=4.0))
    return dict(status="COMPLETE", observation_end_s=duration, requests=rows)


class H1PairChecks(unittest.TestCase):
    def test_each_frozen_block_order_passes_and_reversed_receipt_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            for block in (1, 2):
                session, plan, receipt = pair_identity(block, Path(tmp))
                self.assertEqual(target.check_pair_identity(session, plan, receipt)[2], block)
                receipt["cells"].reverse()
                with self.assertRaisesRegex(target.common.AuditError, "order changed"):
                    target.check_pair_identity(session, plan, receipt)

    def test_package_output_and_plan_order_mismatch_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, plan, receipt = pair_identity(1, Path(tmp))
            plan["cells"][0]["package_dir"] = target.REMOTE_PACKAGE_ROOT + "/wrong"
            with self.assertRaisesRegex(target.common.AuditError, "package/output"):
                target.check_pair_identity(session, plan, receipt)
            plan["cells"][0]["package_dir"] = (target.REMOTE_PACKAGE_ROOT + "/" +
                                                 target.PACKAGE_NAMES[1])
            receipt["cells"][0]["output_dir"] = "/root/wrong"
            with self.assertRaisesRegex(target.common.AuditError, "package/output"):
                target.check_pair_identity(session, plan, receipt)
            receipt["cells"][0]["output_dir"] = target.OUTPUTS[1][0]
            plan["cells"].reverse()
            with self.assertRaisesRegex(target.common.AuditError, "order changed"):
                target.check_pair_identity(session, plan, receipt)

    def test_on_off_ratio_uses_off_in_same_pair_despite_mapping_order(self):
        off, on = raw(100.0, 2.0), raw(90.0, 1.0)
        result = target.summarize_pair({"on": on, "off": off})
        self.assertTrue(math.isclose(result["output_rate_ratio_on_off"], 100 / 90))
        self.assertEqual(result["mean_flow_ratio_on_off"], 1.0)
        self.assertEqual(result["ratio_denominator"], "off_cell_in_same_block")
        self.assertTrue(result["meets_block_criterion"])
        slower = target.summarize_pair({"off": off, "on": raw(105.0, 1.0)})
        self.assertFalse(slower["within_efficiency_budget"])
        self.assertFalse(slower["meets_block_criterion"])

    def test_paired_request_identity_and_zero_off_denominator_fail(self):
        with self.assertRaisesRegex(ValueError, "paired capture cohorts differ"):
            target.summarize_pair({"off": raw(100.0, 2.0),
                                   "on": raw(90.0, 1.0, changed_prompt=True)})
        with self.assertRaisesRegex(target.common.AuditError, "denominator"):
            target.summarize_pair({"off": raw(100.0, 2.0, empty=True),
                                   "on": raw(90.0, 1.0)})

    def test_unpinned_qualification_sha_fails_before_session_read(self):
        with self.assertRaisesRegex(target.common.AuditError, "qualification audit identity"):
            target.analyze(Path("/missing"), "0" * 64, Path("/missing"), "0" * 64)


if __name__ == "__main__":
    unittest.main()

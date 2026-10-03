#!/usr/bin/env python3
"""Fail-closed entry gate for the RouteShape-SLO P2 Oracle replay.

P2 is intentionally unavailable until an eligible P1 bundle reports at least a
weak incremental route signal. This prevents smoke correlations from being
promoted into capacity-control evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ELIGIBLE_P1 = {"P1_INCREMENTAL_SIGNAL_PASS", "WEAK_SIGNAL_NEEDS_MORE_EVENTS"}
P1_SCHEMA = "route-shape-slo-p1-summary-v1"
ACTION = "next_window_active_token_budget"
REQUIRED_P1_ELIGIBILITY_CHECKS = {
    "arrival_episode_disjoint",
    "arrival_regimes_per_model",
    "fresh_holdout_sealed",
    "gate_weight_available",
    "independent_arrival_episodes",
    "instrumentation_overhead_measured",
    "matched_cell_coverage",
    "observed_real_runtime_only",
    "predeclared_split",
    "request_document_disjoint",
    "runtime_representative",
    "two_frozen_models",
}


class ProtocolError(RuntimeError):
    pass


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError("P1 summary must be a JSON object")
    return value


def gate(summary: dict[str, Any]) -> dict[str, Any]:
    schema_valid = summary.get("schema") == P1_SCHEMA
    action_valid = summary.get("action") == ACTION
    eligible = (
        summary.get("scientific_result_eligible") is True
        and summary.get("p1_gate_eligible") is True
    )
    checks = summary.get("eligibility_checks")
    checks_valid = (
        isinstance(checks, dict)
        and set(checks) == REQUIRED_P1_ELIGIBILITY_CHECKS
        and all(value is True for value in checks.values())
        and summary.get("eligibility_blockers") == []
    )
    p1_status = str(summary.get("p1_status", "MISSING"))
    ready = (
        schema_valid
        and action_valid
        and eligible
        and checks_valid
        and p1_status in ELIGIBLE_P1
    )
    if not schema_valid:
        reason = "P1 summary schema is missing or unsupported"
    elif not action_valid:
        reason = "P1 action does not match the frozen active-token-budget bound"
    elif not eligible or not checks_valid:
        reason = "P1 evidence is smoke-only or runtime-nonrepresentative"
    elif p1_status not in ELIGIBLE_P1:
        reason = f"P1 status {p1_status!r} is not weak-positive or passing"
    else:
        reason = "P1 gate passed; Oracle implementation still requires a causal capacity-action trace"
    return {
        "schema": "route-shape-slo-p2-gate-v1",
        "status": "READY_TO_IMPLEMENT_P2_REPLAY" if ready else "BLOCKED_P1_NOT_ELIGIBLE",
        "executed": False,
        "input_p1_status": p1_status,
        "input_scientific_result_eligible": eligible,
        "action": ACTION,
        "reason": reason,
        "claim_boundary": "No Oracle headroom or capacity-control result was measured.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = gate(load_object(Path(args.p1_summary)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

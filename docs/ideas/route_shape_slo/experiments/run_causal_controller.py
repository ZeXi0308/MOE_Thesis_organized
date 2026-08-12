#!/usr/bin/env python3
"""Fail-closed entry gate for RouteShape-SLO P3 controller experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class ProtocolError(RuntimeError):
    pass


P2_SCHEMA = "route-shape-slo-p2-summary-v1"
ACTION = "next_window_active_token_budget"


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError("P2 summary must be a JSON object")
    return value


def gate(summary: dict[str, Any]) -> dict[str, Any]:
    p2_status = str(summary.get("status", "MISSING"))
    p2_pass = (
        summary.get("schema") == P2_SCHEMA
        and summary.get("action") == ACTION
        and summary.get("scientific_result_eligible") is True
        and summary.get("executed") is True
        and p2_status in {"P2_ORACLE_HEADROOM_PASS", "GO_TO_CAUSAL_CONTROLLER"}
    )
    return {
        "schema": "route-shape-slo-p3-gate-v1",
        "status": "READY_TO_IMPLEMENT_P3_CONTROLLER" if p2_pass else "BLOCKED_P2_NOT_PASSED",
        "executed": False,
        "input_p2_status": p2_status,
        "action": ACTION,
        "reason": (
            "P2 gate passed; implement the frozen historical-route policy"
            if p2_pass
            else "P3 is forbidden until an eligible future-route Oracle clears the frozen headroom gate"
        ),
        "claim_boundary": "No causal route-aware controller result was measured.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = gate(load_object(Path(args.p2_summary)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

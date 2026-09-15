from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    from .schema import ROUTE_COLUMNS
    from .test_schema import route_row
except ImportError:
    from schema import ROUTE_COLUMNS
    from test_schema import route_row


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENTS = Path(__file__).resolve().parent
CONFIG = EXPERIMENTS / "configs" / "gate0_v1.json"
CENSUS = EXPERIMENTS / "census.py"


class CensusCliTest(unittest.TestCase):
    def test_development_cli_writes_boolean_and_suppresses_rank_claims(self) -> None:
        rows = []
        for slot in range(8):
            base = route_row(
                slot=slot,
                expert=slot,
                target_rank=slot,
            )
            rows.append(
                replace(
                    base,
                    model="olmoe",
                    model_revision="6d84c48581ece794365f2b8e9cfb043c68ade9c5",
                    placement_id="UNRESOLVED_PHYSICAL_PLACEMENT_ID",
                    target_rank=-1,
                    gate_weight=0.125,
                )
            )

        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            routes = temp / "routes.csv"
            output = temp / "census.json"
            with routes.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=ROUTE_COLUMNS)
                writer.writeheader()
                writer.writerows(row.__dict__ for row in rows)

            subprocess.run(
                [
                    sys.executable,
                    str(CENSUS),
                    "--config",
                    str(CONFIG),
                    "--routes",
                    str(routes),
                    "--output",
                    str(output),
                    "--development-expert-only",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertIs(payload["formal_gate_result"], False)
        self.assertIsNone(payload["request_rows"][0]["rank_max_share"])
        self.assertEqual(
            payload["service_work_gate"]["status"],
            "BLOCKED_MISSING_SERVICE_WEIGHTED_CAUSAL_LEDGER",
        )


if __name__ == "__main__":
    unittest.main()

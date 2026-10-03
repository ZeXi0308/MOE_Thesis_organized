from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    from .raw_recompute import sha256_file
    from .test_raw_recompute import (
        RAW_SOURCE,
        artifact_entry,
        raw_fixture,
        write_jsonl,
    )
except ImportError:
    from raw_recompute import sha256_file
    from test_raw_recompute import RAW_SOURCE, artifact_entry, raw_fixture, write_jsonl


EXPERIMENTS = Path(__file__).resolve().parent
RUNNER = EXPERIMENTS / "run_gate0.py"
CONFIG = EXPERIMENTS / "configs" / "gate0_v1.json"
ROOT = Path(__file__).resolve().parents[4]
CANONICAL_VERDICTS = (
    "QUALIFIED_FOR_8XA100_EXISTENCE_GATE",
    "NO_GO_PHENOMENON",
    "NO_GO_ORACLE",
    "SIMPLE_BASELINE_WINS",
    "NO_GO_BATCHING_TAX",
)


class GateCliTest(unittest.TestCase):
    def test_invalid_config_fails_closed_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "invalid.json"
            output = root / "result.json"
            config.write_text('{"schema":"wrong"}\n', encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--config",
                    str(config),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "INVALID_CONFIG")
        self.assertFalse(payload["formal_result"])
        self.assertNotIn("Traceback", completed.stderr)

    def test_aggregate_smoke_does_not_leak_canonical_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke.json"
            subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--config",
                    str(CONFIG),
                    "--smoke",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "SMOKE_ONLY")
        for verdict in CANONICAL_VERDICTS:
            self.assertNotIn(verdict, str(payload))

    def test_raw_bundle_cli_is_smoke_only(self) -> None:
        config, requests, blocks = raw_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            config_path = root / "config.json"
            request_path = bundle / "requests.jsonl"
            block_path = bundle / "blocks.jsonl"
            manifest_path = bundle / "manifest.json"
            output = root / "result.json"
            config_path.write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            write_jsonl(request_path, requests)
            write_jsonl(block_path, blocks)
            manifest = {
                "schema": "routeshield-raw-bundle-v1",
                "mode": "DEVELOPMENT",
                "config_sha256": sha256_file(config_path),
                "evaluator_source_sha256": sha256_file(RAW_SOURCE),
                "artifacts": {
                    "requests": artifact_entry(
                        request_path,
                        schema="routeshield-raw-request-v1",
                        config_key="required_evidence.raw_request_ledger_sha256",
                    ),
                    "blocks": artifact_entry(
                        block_path,
                        schema="routeshield-raw-block-v1",
                        config_key="required_evidence.raw_block_ledger_sha256",
                    ),
                },
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--config",
                    str(config_path),
                    "--raw-bundle",
                    str(manifest_path),
                    "--smoke",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "RAW_RECOMPUTE_SMOKE_ONLY")
        self.assertFalse(payload["formal_result"])
        for verdict in CANONICAL_VERDICTS:
            self.assertNotIn(verdict, str(payload))


if __name__ == "__main__":
    unittest.main()

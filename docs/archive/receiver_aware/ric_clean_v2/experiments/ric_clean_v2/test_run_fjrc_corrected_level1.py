from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

try:
    from .fjrc_corrected_level1 import ServiceLUT, select_holdout_scenarios
    from .fjrc_corrected_replay import ReplayConfig, materialize_replay, run_campaign
    from .run_fjrc_corrected_level1 import REQUIRED_ARTIFACTS, RunnerError, write_artifacts
    from .test_fjrc_corrected_level1 import joins_fixture
except ImportError:  # pragma: no cover
    from fjrc_corrected_level1 import ServiceLUT, select_holdout_scenarios  # type: ignore
    from fjrc_corrected_replay import ReplayConfig, materialize_replay, run_campaign  # type: ignore
    from run_fjrc_corrected_level1 import REQUIRED_ARTIFACTS, RunnerError, write_artifacts  # type: ignore
    from test_fjrc_corrected_level1 import joins_fixture  # type: ignore


def fixture_report():
    service = ServiceLUT("olmoe", 1.0, 0.25, 2.0, 0.5, 3.25, "a" * 64)
    config = ReplayConfig(bootstrap_replicates=10)
    native = select_holdout_scenarios("olmoe", joins_fixture(), service)
    scenarios = [materialize_replay(value, service, config) for value in native]
    return config, run_campaign(scenarios, config)


class CorrectedRunnerTests(unittest.TestCase):
    def test_artifact_bundle_is_complete_and_json_parseable(self):
        config, report = fixture_report()
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "run"
            write_artifacts(
                output,
                model="olmoe",
                config=config,
                report=report,
                environment={"test": True},
                manifest={"test": True},
                dry_run=True,
            )
            self.assertEqual({path.name for path in output.iterdir()}, set(REQUIRED_ARTIFACTS))
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["status"], "LOGICAL_TRACE_REPLAY_ONLY")
            self.assertEqual(metrics["aggregate"]["Q"]["request_count"], 32)
            rows = (output / "raw_results.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 48)
            for row in rows:
                json.loads(row)

    def test_refuses_to_overwrite_existing_output(self):
        config, report = fixture_report()
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "run"
            output.mkdir()
            with self.assertRaisesRegex(RunnerError, "refusing"):
                write_artifacts(
                    output,
                    model="olmoe",
                    config=config,
                    report=report,
                    environment={},
                    manifest={},
                    dry_run=True,
                )


if __name__ == "__main__":
    unittest.main()

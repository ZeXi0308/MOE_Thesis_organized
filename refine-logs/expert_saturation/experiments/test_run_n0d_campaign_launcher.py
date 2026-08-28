#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import unittest


SCRIPT = Path(__file__).with_name("run_n0d_campaign.sh")


class N0dCampaignLauncherTests(unittest.TestCase):
    def test_bash_syntax(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")
        result = subprocess.run(
            [bash, "-n", str(SCRIPT)], text=True, capture_output=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_launcher_freezes_capture_helper_and_binds_evaluator(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('cp "$script_dir/n0d_capture_contract.py"', source)
        self.assertIn('cp "$script_dir/N0D_MATCHED_ROUTER_GATE.md"', source)
        self.assertIn("n0d_capture_contract.py", source)
        self.assertIn("N0D_MATCHED_ROUTER_GATE.md", source)
        self.assertIn("> control-files.sha256", source)
        self.assertIn("run_stage verify-capture 1m", source)
        self.assertIn("capture_parent=/root/autodl-tmp/expert-saturation/tmp", source)
        self.assertIn('export TMPDIR="$capture_parent"', source)
        self.assertIn("bcrd-gate0-smoke-${campaign_root##*/}", source)
        self.assertIn('--capture-dir "$capture_dir"', source)
        self.assertIn('--output "$campaign_root/n0d-verdict.json"', source)

    def test_timeout_and_process_group_cleanup_are_fail_closed(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("setsid --wait timeout --signal=TERM --kill-after=30s", source)
        self.assertIn('kill -TERM -- "-$pgid"', source)
        self.assertIn('kill -KILL -- "-$pgid"', source)
        self.assertIn("wait_for_process_group_exit", source)
        self.assertIn("CAMPAIGN_ABORTED.json", source)
        self.assertNotIn("rm -rf", source)

    def test_success_seal_binds_artifacts_and_is_final(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"status": "CAMPAIGN_COMPLETE"', source)
        self.assertIn('"capture_complete_sha256"', source)
        self.assertIn('"capture_dir": str(capture)', source)
        self.assertIn('"process_outputs_sha256"', source)
        self.assertIn('"verdict_sha256"', source)
        self.assertIn('"status": verdict["status"]', source)
        self.assertIn('"cleanup_evidence_sha256"', source)
        self.assertIn("export PYTHONDONTWRITEBYTECODE=1", source)
        self.assertIn('os.replace(str(temporary), str(output))', source)
        self.assertIn("sha256sum -c control-files.sha256", source)
        self.assertIn("sha256sum -c CAMPAIGN_FILES.sha256", source)
        sentinel = source.index(
            "# The completion sentinel is the final campaign filesystem mutation."
        )
        tail = source[sentinel:].replace(
            'echo "N0D_CAMPAIGN_COMPLETE=$campaign_root"', ""
        )
        self.assertNotIn(">", tail)


if __name__ == "__main__":
    unittest.main()

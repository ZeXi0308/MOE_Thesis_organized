"""CPU failure fixtures only; these are not native experiment measurements."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics import summarize_episode_requests


class FailureRetention(unittest.TestCase):
    def analyze(self, files):
        with tempfile.TemporaryDirectory(prefix='moe-analysis-failure-') as temporary:
            root = Path(temporary)
            for label, artifacts in files.items():
                directory = root/label
                directory.mkdir()
                for name, value in artifacts.items():
                    (directory/name).write_text(json.dumps(value))
            subprocess.run([sys.executable, str(HERE/'analyze_probe.py'), '--results-dir', str(root),
                '--output-dir', str(root/'analysis')], check=True, capture_output=True, text=True)
            return json.loads((root/'analysis/analysis.json').read_text())

    def test_status_files_cannot_substitute_for_missing_raw(self):
        files = {f'repeat{r}-{d}-cap{c}': {'status.json': {'status': 'COMPLETE'}}
            for r in [0, 1] for d in ['short', 'long'] for c in [16, 32]}
        result = self.analyze(files)
        self.assertEqual(result['status'], 'INCOMPLETE_OR_UNRUN')
        self.assertTrue(all(c['status']=='INCOMPLETE' for c in result['cells']))

    def test_saved_raw_without_metrics_retains_failure(self):
        result = self.analyze({'repeat0-short-cap16': {
            'status.json': {'status': 'INCOMPLETE', 'error': 'post-capture GPU isolation check'},
            'raw.json': {'status': 'COMPLETE', 'requests': [{'status': 'completed'}]}}})
        cell = result['cells'][0]
        self.assertEqual(cell['status'], 'INCOMPLETE')
        self.assertEqual(cell['requests_completed'], 1)
        self.assertIn('metrics.json', cell['missing_artifacts'])

    def test_terminal_failure_overrides_successful_capture(self):
        raw = dict(status='COMPLETE', requests=[], observation_end_s=1, memory_trace=[],
            scheduler_steps=[], capacity_boundary=None, strict_nonpreemptive_status='COMPLETE_NO_PREEMPTION')
        files = dict.fromkeys(['engine_args.json', 'memory-before.json', 'memory-after.json', 'gpu-after.json'], {})
        files.update({'raw.json': raw, 'status.json': {'status': 'INCOMPLETE', 'error': 'post-capture failure'},
            'metrics.json': summarize_episode_requests([], observation_end_s=1, ttft_slo_s=5, tpot_slo_s=.2)})
        cell = self.analyze({'repeat0-short-cap16': files})['cells'][0]
        self.assertEqual(cell['status'], 'INCOMPLETE')
        self.assertFalse(cell['full_episode_comparison_eligible'])


if __name__ == '__main__':
    unittest.main()

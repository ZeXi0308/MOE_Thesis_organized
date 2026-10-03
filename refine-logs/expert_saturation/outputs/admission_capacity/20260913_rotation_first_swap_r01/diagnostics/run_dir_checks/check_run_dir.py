#!/usr/bin/env python3
"""Bounded CPU path checks; mocked qualification metadata has no raw or metrics."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'diagnose_paths.py'
spec = importlib.util.spec_from_file_location('path_diagnostics', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
manifest = module.read(HERE.parent.parent/'preparation/source/campaign.json')
checks = []

with tempfile.TemporaryDirectory(prefix='first-swap-run-dir-cpu-') as temporary:
    bundle = Path(temporary)/'bundle'
    primary = bundle/'primary.json'
    primary.parent.mkdir()
    primary.write_text('{}')
    metadata = dict(status='MEASUREMENT_ONLY', comparisons_eligible=True, cells=[
        dict(label=c['label'], status='COMPLETE', full_episode_comparison_eligible=True)
        for c in manifest['cells']])

    def blocked_case(name, run_dir, execution):
        selected = run_dir or bundle/'execution'
        if execution is not None:
            selected.mkdir(parents=True)
            (selected/'execution.json').write_text(json.dumps(execution))
        reads = []
        original_read = module.read
        def mocked_read(path):
            reads.append(path)
            if path == primary:
                return metadata
            if path == bundle/'preparation/source/campaign.json':
                return manifest
            return original_read(path)
        with patch.object(module, 'read', side_effect=mocked_read):
            result = module.diagnose(bundle, primary, run_dir)
        assert result['status'] == 'UNRUN_OR_UNQUALIFIED'
        assert result['comparisons'] == [] and 'cells' not in result
        if execution is not None:
            assert selected/'execution.json' in reads
        if run_dir is not None:
            assert bundle/'execution/execution.json' not in reads
        checks.append(dict(name=name, status='PASS'))

    blocked_case('default_execution_incomplete', None, dict(status='RUNNING', cells=[]))
    blocked_case('explicit_execution_missing', bundle/'missing', None)
    blocked_case('explicit_execution_staged', bundle/'staged', dict(status='STAGED', cells=[]))

    class Captured(Exception):
        pass
    for argument in (None, bundle/'execution02_weste_23478'):
        argv = ['diagnose_paths.py', '--bundle', str(bundle), '--primary', str(primary),
                '--output-dir', str(bundle/'must-not-be-created')]
        if argument is not None:
            argv += ['--run-dir', str(argument)]
        def capture(b, p, r):
            assert (b, p, r) == (bundle, primary, argument)
            raise Captured()
        with patch.object(sys, 'argv', argv), patch.object(module, 'diagnose', side_effect=capture):
            try:
                module.main()
            except Captured:
                pass
            else:
                raise AssertionError('CLI did not call diagnose')
        assert not (bundle/'must-not-be-created').exists()
        checks.append(dict(name='cli_explicit_run_dir' if argument else 'cli_legacy_default', status='PASS'))

result = dict(status='PASS', source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), checks=checks,
    scope='CPU parameter routing and missing/incomplete execution rejection only; no raw loaded, no GPU rerun, no measured output. Existing manifest/raw hash gate code retained, not re-executed on qualified data.')
with (HERE/'checks.json').open('x') as output:
    json.dump(result, output, indent=2)
    output.write('\n')
print(json.dumps(result))

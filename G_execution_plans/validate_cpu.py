"""Validate syntax, frozen files, and the no-CUDA-on-busy-lock boundary."""
import ast
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
checks = {}
for path in ROOT.glob('*.py'):
    ast.parse(path.read_text(), filename=str(path))
checks['python_syntax'] = 'passed'
manifest = json.loads((ROOT / 'evidence/source_sha256.json').read_text())
for path, expected in manifest.items():
    assert hashlib.sha256((ROOT / 'native_sources' / path).read_bytes()).hexdigest() == expected
checks['archived_source_sha256'] = len(manifest)
hist = json.loads((ROOT / 'evidence/historical_feasibility.json').read_text())
for load in ('low', 'high'):
    expected = hist['source_sha256'][f'D_prefill_budget_20261004/workload_{load}.json']
    assert hashlib.sha256((ROOT / 'inputs' / f'workload_{load}.json').read_bytes()).hexdigest() == expected
checks['frozen_input_sha256'] = 'passed'
assert 'torch' not in sys.modules and 'vllm' not in sys.modules
for name in ('moe_probe', 'startup_probe'):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix='g-lock-boundary-') as tmp:
        tmp = Path(tmp)
        lock_path = tmp / 'shared.lock'
        output = tmp / 'blocked.json'
        with lock_path.open('a+') as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if name == 'moe_probe':
                module.LOCK = str(lock_path)
                argv = [name, '--synthetic', '--output', str(output)]
            else:
                module.LOCK_PATH = lock_path
                argv = [name, '--plan', 'native', '--output', str(output)]
            with patch.object(sys, 'argv', argv), patch.object(
                    module.subprocess, 'check_output', side_effect=AssertionError('GPU process query before lock')):
                assert module.main() == 75
            assert json.loads(output.read_text())['status'] == 'LOCK_BUSY_NO_GPU_INITIALIZED'
            assert 'torch' not in sys.modules and 'vllm' not in sys.modules
        if name == 'startup_probe' and module._LOCK_HANDLE:
            module._LOCK_HANDLE.close()  # Only the unsuccessful temporary test handle.
        checks[name + '_busy_lock'] = 'exit75; no torch, vllm or nvidia-smi'
    if name == 'startup_probe':
        assert {max(v) for v in module.GRAPH_SETS.values()} == {512}
        assert {k: len(v) for k, v in module.GRAPH_SETS.items()} == dict(compact=10, native=51, dense=67)
        assert all(sorted(set(v)) == v for v in module.GRAPH_SETS.values())
        for plan in module.GRAPH_SETS:
            kwargs = module.engine_kwargs(plan, 'triton', module.MODEL)
            assert 'kv_cache_memory_bytes' not in kwargs
            assert kwargs['gpu_memory_utilization'] == 0.9
        checks['graph_sets_and_natural_KV_budget'] = 'passed'
checks['gpu_validation'] = 'not_run'
(ROOT / 'evidence/cpu_validation.json').write_text(json.dumps(checks, indent=2) + '\n')
print(json.dumps(checks, indent=2))

#!/usr/bin/env bash
# Default: CPU-only command/config verification. Pass --run to the root owner
# when the rented GPU is available; run_locked owns locking, timeout and budget.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$task_root"
# The rented host does not expose a python/python3 alias in its SSH PATH.
# Local --plan remains usable on the workstation.
task_python=/root/miniconda3/bin/python
if [[ ! -x "$task_python" ]]; then
    task_python=python3
fi
"$task_python" - "$task_root" "${1:---plan}" "${2:-configs/olmoe_awq_common_r01.json}" <<'PY'
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

root, mode = Path(sys.argv[1]), sys.argv[2]
if mode not in ('--plan', '--run'):
    raise SystemExit('Usage: bash configs/run_olmoe_awq_common_r01.sh [--plan|--run] [config.json]')
config_path = root / sys.argv[3]
c = json.loads(config_path.read_text())
m = json.loads((root / c['source_manifest']).read_text())
if (m['checkpoint_fingerprint'] != c['expected_source_checkpoint_fingerprint'] or
        m['identity']['repo_id'] != c['base_checkpoint'] or
        m['identity']['revision'] != c['base_revision'] or m['format'] != 'bf16'):
    raise SystemExit('Source manifest differs from this fixed calibration configuration')
data_hash = hashlib.sha256((root / c['calibration_jsonl']).read_bytes()).hexdigest()
if data_hash != c['calibration_jsonl_sha256']:
    raise SystemExit('Calibration data SHA256 differs from the fixed configuration')
command = [c['calibration_python'], 'code/prepare_quantization.py', 'calibrate',
    '--source-manifest', c['source_manifest'], '--format', c['format'],
    '--calibration-jsonl', c['calibration_jsonl'], '--samples', str(c['samples']),
    '--max-seq-length', str(c['max_seq_length']), '--seed', str(c['seed']),
    '--output', c['output'], '--common-basis-bf16-output', c['common_basis_bf16_output'],
    '--reserve-gib', str(c['reserve_gib']),
    '--moe-calibrate-all-experts' if c['moe_calibrate_all_experts'] else '--no-moe-calibrate-all-experts']
locked = [sys.executable, 'code/run_locked.py', '--root', str(root),
          '--name', c['run_name'], '--timeout', str(c['timeout_s']),
          '--wait-seconds', str(c['wait_seconds']), '--', *command]
print(json.dumps({'mode': mode, 'config': str(config_path),
    'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
    'source_manifest_sha256': hashlib.sha256((root / c['source_manifest']).read_bytes()).hexdigest(),
    'calibration_data_sha256_verified': True,
    'command': shlex.join(locked), 'storage_plan': c['storage_plan'],
    'gpu_work_started': False}, indent=2), flush=True)
if mode == '--run':
    if not (root / c['calibration_python']).is_file():
        raise SystemExit('Pinned calibration_venv is missing')
    raise SystemExit(subprocess.run(locked, cwd=root).returncode)
PY

#!/usr/bin/env python3
"""Freeze existing four policies at the measured d6 KV pool; no GPU work."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def main():
    out = Path(__file__).resolve().parents[2] / 'outputs/admission_capacity'
    pressure = out/'20260913_pressure_sweep_r01/execution_review_westc_r02/pkg'
    strong = out/'20260913_rotation_strong_baseline_r01/preparation/source'
    dest = out/'20260914_d6_strong_baselines_r01/preparation'
    pkg = dest/'pkg'
    pkg.mkdir(parents=True, exist_ok=False)
    names = ('run_probe.py', 'native_capture.py', 'memory_telemetry.py',
             'safe_static.py', 'completion_headroom.py', 'absence_rotation.py',
             'rotation_native.py', 'metrics.py')
    sources = {}
    for name in names:
        src = (strong if name in ('absence_rotation.py', 'rotation_native.py') else pressure)/name
        shutil.copy2(src, pkg/name)
        sources[name] = dict(path=str(src), sha256=hashlib.sha256(src.read_bytes()).hexdigest())
    shutil.copytree(pressure/'inputs_preparation', pkg/'inputs_preparation')
    f = pkg/'run_probe.py'
    s = f.read_text()
    changes = {
        "parser.add_argument('--output-dir', type=Path, required=True)":
            "parser.add_argument('--victim-order', choices=['least_progress', 'most_output'], default='least_progress')\n    parser.add_argument('--output-dir', type=Path, required=True)",
        'args = parser.parse_args()':
            "args = parser.parse_args()\n    if args.kv_cache_bytes != 13960740864 or args.cap != 32 or args.domain != 'long' or args.gpu_memory_utilization != 0.9 or args.completion_policy not in ('native', 'headroom', 'rotate'):\n        raise ValueError('frozen d6 dimensions differ')\n    if args.completion_policy != 'rotate' and args.victim_order != 'least_progress':\n        raise ValueError('victim order only applies to rotation')",
        "gpu_memory_utilization=args.gpu_memory_utilization,\n                  preemption_mode=":
            "gpu_memory_utilization=args.gpu_memory_utilization,\n                  rotation_victim_order=args.victim_order,\n                  preemption_mode=",
        "expected_requests=32, rotation_config=RotationConfig(**config['rotation_config']))":
            "expected_requests=32, rotation_config=RotationConfig(**config['rotation_config']),\n                victim_order=args.victim_order)",
    }
    for old, new in changes.items():
        assert s.count(old) == 1, old
        s = s.replace(old, new)
    ast.parse(s)
    f.write_text(s)
    roles = ('native', 'headroom', 'least_progress', 'most_output')
    cells = [dict(label=f'block{b}-d6-{role}', block=b, role=role,
                  policy=role if role in ('native', 'headroom') else 'rotate',
                  victim_order='most_output' if role == 'most_output' else 'least_progress',
                  kv_cache_bytes=13960740864, expected_usable_blocks=6656)
             for b in (0, 1) for role in (roles if b == 0 else roles[::-1])]
    (pkg/'campaign.json').write_text(json.dumps(dict(cells=cells), indent=2)+'\n')
    manifest = {str(f.relative_to(pkg)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in pkg.rglob('*') if f.is_file()}
    (dest/'status.json').write_text(json.dumps(dict(status='PREPARED_UNRUN', gpu_executions=0,
        ancestor_sources=sources, files=manifest, cells=cells), indent=2)+'\n')
    with tarfile.open(dest/'package.tar.gz', 'w:gz') as archive:
        archive.add(pkg, arcname='pkg')
    print(dest)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Run both routing distributions for one format with the same graph protocol."""
import argparse
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--model', choices=('olmoe', 'qwen3'), required=True)
p.add_argument('--format', choices=('bf16', 'fp8', 'awq', 'nvfp4'), required=True)
p.add_argument('--weights')
p.add_argument('--tokens', default='1,8,32,128,512,2048,8192')
p.add_argument('--out-dir', required=True)
a = p.parse_args()
for routing in ('uniform', 'zipf'):
    cmd = [sys.executable, str(Path(__file__).with_name('bench_expert_formats.py')),
           '--model', a.model, '--format', a.format, '--tokens', a.tokens,
           '--routing', routing, '--cuda-graph', '--energy-seconds', '3',
           '--out', str(Path(a.out_dir) / f'{a.model}_{routing}_{a.format}.jsonl')]
    if a.weights:
        cmd.extend(['--weights', a.weights])
    if a.format == 'nvfp4':
        cmd.extend(['--nvfp4-backend', 'cutlass'])
    subprocess.run(cmd, check=True)

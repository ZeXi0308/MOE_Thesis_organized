#!/usr/bin/env python3
"""Bounded first two-format survey, under run_locked.py's single shared lock."""
import argparse
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--out-dir', required=True)
a = p.parse_args()
out = Path(a.out_dir)
out.mkdir(parents=True, exist_ok=True)
for model in ('olmoe', 'qwen3'):
    for routing in ('uniform', 'zipf'):
        for fmt in ('bf16', 'fp8'):
            cmd = [sys.executable, 'code/bench_expert_formats.py', '--model', model,
                '--format', fmt, '--tokens', '1,8,32,128,512,2048,8192',
                '--routing', routing, '--cuda-graph', '--energy-seconds', '3',
                '--out', str(out / f'{model}_{routing}_{fmt}.jsonl')]
            print('RUN', cmd, flush=True)
            subprocess.run(cmd, check=True)

#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--out-dir', required=True)
p.add_argument('--weights', required=True)
a = p.parse_args()
for routing in ('uniform', 'zipf'):
    subprocess.run([sys.executable, 'code/bench_expert_formats.py', '--model', 'qwen3',
        '--format', 'nvfp4', '--weights', a.weights, '--nvfp4-backend', 'cutlass',
        '--tokens', '1,8,32,128,512,2048,8192', '--routing', routing, '--cuda-graph',
        '--energy-seconds', '3', '--out', str(Path(a.out_dir) / f'qwen3_{routing}_nvfp4.jsonl')], check=True)

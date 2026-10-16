#!/usr/bin/env python3
"""Contrast cold-start and continuously warm NVML energy windows."""
import argparse
import json
import math
from pathlib import Path
import time

from bench_expert_formats import event_time, make_routing
from format_backends import SHAPES, build_backend
from power import PowerSampler

p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
a = p.parse_args()
import torch
shape = SHAPES['olmoe']
backend = build_backend('fp8', shape)
ids, weights, routing = make_routing(512, shape, 'uniform', 20261015)
ids, weights = ids.cuda(), weights.cuda()
x = torch.randn((512, shape.d), device='cuda', dtype=torch.bfloat16)
with torch.inference_mode():
    for _ in range(20):
        y = backend(x, weights, ids)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = backend(x, weights, ids)
    fn = graph.replay
    latency, _ = event_time(fn, 100)
    iterations = max(1, math.ceil(3000 / latency))
    device = str(torch.cuda.get_device_properties(0).uuid)
    meter = PowerSampler(device=device)
    records = []
    try:
        torch.cuda.synchronize()
        time.sleep(2)
        idle = meter.measure_idle(3, synchronize=torch.cuda.synchronize)
        for name in ('cold_after_idle', 'continuous_warm_1', 'continuous_warm_2'):
            with meter.measure(synchronize=torch.cuda.synchronize) as m:
                for _ in range(iterations):
                    fn()
            records.append({'name': name, 'iterations': iterations, 'power': m.result,
                'sample_vs_counter_pct': 100 * (m.result['sampled_energy_j'] / m.result['energy_j'] - 1)})
            # Avoid filesystem work/printing between the continuous windows.
    finally:
        meter.close()
data = {'routing': routing, 'latency_ms': latency, 'idle': idle, 'windows': records,
    'scope': 'same synthetic FP8 MoE workload, one run, sensor-window diagnostic',
    'nvml_reference': 'https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html'}
Path(a.out).write_text(json.dumps(data, indent=2) + '\n')
print(json.dumps([{'name': r['name'], 'sample_vs_counter_pct': r['sample_vs_counter_pct']} for r in records]))

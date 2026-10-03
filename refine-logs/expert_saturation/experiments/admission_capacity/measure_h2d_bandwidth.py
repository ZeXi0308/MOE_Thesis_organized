#!/usr/bin/env python3
"""Measure pinned host-to-device bandwidth at expert-weight transfer sizes.

The residency-budget argument compares idle expert bytes against a transfer
cost, so the bandwidth must be measured rather than assumed. Sizes are chosen
to bracket a single OLMoE expert (12 MiB), one layer's experts (768 MiB) and
the whole instantaneous idle set (~1.2 GiB).

Reports the best of several trials per size: a paging implementation would use
pinned memory and a dedicated stream, so the optimistic figure is the right one
for a feasibility ceiling. Using a pessimistic bandwidth would make the
infeasibility argument too easy.
"""
import json
import sys
import time
from pathlib import Path

import torch

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/h2d")
OUT.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise SystemExit("needs a GPU")

MIB = 1024 ** 2
SIZES_MIB = [12, 48, 192, 768, 1212]
TRIALS = 7

rows = []
stream = torch.cuda.Stream()
for mib in SIZES_MIB:
    n = mib * MIB
    host = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(n, dtype=torch.uint8, device="cuda")
    # warm up the path before timing
    for _ in range(2):
        dev.copy_(host, non_blocking=True)
    torch.cuda.synchronize()
    best = None
    times = []
    for _ in range(TRIALS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.stream(stream):
            dev.copy_(host, non_blocking=True)
        stream.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        best = dt if best is None else min(best, dt)
    rows.append(dict(size_mib=mib, best_s=best,
                     best_gb_s=n / best / 1e9,
                     median_s=sorted(times)[len(times) // 2],
                     median_gb_s=n / sorted(times)[len(times) // 2] / 1e9,
                     trials=TRIALS))
    del host, dev
    torch.cuda.empty_cache()

peak = max(r["best_gb_s"] for r in rows)
report = dict(device=torch.cuda.get_device_name(0),
              torch=torch.__version__, rows=rows,
              peak_best_gb_s=peak,
              note="pinned, non_blocking, dedicated stream; best-of-trials is an "
                   "optimistic ceiling chosen so the feasibility argument is not "
                   "made easy by a pessimistic bandwidth")
(OUT / "h2d.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))

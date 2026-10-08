"""Passive 1 Hz GPU snapshots; never initializes CUDA or changes device state."""
import argparse
import json
import os
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--parent-pid', type=int, required=True)
a = p.parse_args()
while os.getppid() == a.parent_pid:
    start = time.perf_counter()
    row = dict(monotonic_s=start, unix_s=time.time(), observer_pid=os.getpid())
    for name, query in (
        ('gpu', '--query-gpu=uuid,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory,memory.used'),
        ('compute_processes', '--query-compute-apps=pid,process_name,used_gpu_memory'),
    ):
        try:
            result = subprocess.run(['nvidia-smi', query, '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=3)
            row[name] = dict(returncode=result.returncode, stdout=result.stdout,
                             stderr=result.stderr)
        except Exception as exc:
            row[name] = dict(error=repr(exc))
    row['query_wall_s'] = time.perf_counter() - start
    print(json.dumps(row), flush=True)
    time.sleep(max(0, 1 - row['query_wall_s']))

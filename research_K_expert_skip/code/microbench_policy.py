"""Measure post-selection policy overhead (not MoE/service speedup).

Run only through the project's GPU-lock launcher. Eager wall, eager main-thread
CPU, and captured-device time include the counters. Main-thread CPU measures
Python/driver submission in the warmed eager loop, excluding its final explicit
CUDA synchronization. It is not all-thread or full-service CPU consumption.
CUDA graph capture excludes Python, allocation, and individual call dispatch.
These are separate clock boundaries, not components to subtract from each other.
"""
import argparse
import json
from pathlib import Path
import time


TIMING_BOUNDARIES = {
    "eager_wall": {"clock": "time.perf_counter",
        "scope": "warmed eager Python loop through final explicit CUDA synchronization; includes clock instrumentation"},
    "eager_thread_cpu": {"clock": "time.thread_time_ns",
        "scope": "CPU consumed by the main thread during the same warmed eager Python/driver submission loop; final explicit CUDA synchronization is outside the interval",
        "excludes": "off-CPU waiting, other threads, GPU execution, warmup and graph capture; not all-thread or full-service CPU time"},
    "captured_device": {"clock": "torch.cuda.Event.elapsed_time",
        "scope": "CUDA graph replay device interval per captured policy call; excludes Python, allocation and individual call dispatch"},
    "comparison": "report each metric independently; wall minus device is not CPU time",
}


def measure_eager_calls(call, repetitions, synchronize):
    """Time warmed calls; the caller completes warmup/synchronization first.

    The CPU clock stops before the final explicit device synchronization. Any
    CPU work inside a policy/driver call remains included; blocked time does not
    advance the thread CPU clock. No GPU API is needed to test this boundary.
    """
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    wall_start = time.perf_counter()
    cpu_start = time.thread_time_ns()
    for _ in range(repetitions):
        call()
    cpu_ns = time.thread_time_ns() - cpu_start
    synchronize()
    wall_seconds = time.perf_counter() - wall_start
    return {"eager_wall_us_per_call": wall_seconds * 1e6 / repetitions,
            "eager_thread_cpu_us_per_call": cpu_ns / 1000 / repetitions,
            "eager_thread_cpu_clock": "time.thread_time_ns"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 64])
    parser.add_argument("--backends", nargs="+", default=["torch", "triton_multi", "triton"])
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    import torch
    from expert_skip import ExpertSkipPolicy, SkipConfig
    torch.manual_seed(20261016)
    rows = []
    for batch in args.batches:
        weights = torch.rand(batch, args.top_k, device="cuda", dtype=torch.float32)
        weights /= weights.sum(1, keepdim=True)
        ids = torch.stack([torch.randperm(args.experts, device="cuda")[:args.top_k]
                           for _ in range(batch)]).to(torch.int32)
        for backend in args.backends:
            policy = ExpertSkipPolicy(SkipConfig(mode="marginal", tail_rank=args.top_k//2,
                                                 gate_threshold=.15, mass_budget=.2,
                                                 backend=backend))
            def call():
                return policy("micro", weights, ids, args.experts, True, 1000)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(5):
                    call()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            eager_timing = measure_eager_calls(call, args.repetitions, torch.cuda.synchronize)
            graph = torch.cuda.CUDAGraph()
            calls_per_graph = 10
            with torch.cuda.graph(graph):
                for _ in range(calls_per_graph):
                    call()
            for _ in range(3):
                graph.replay()
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
            for _ in range(args.repetitions):
                graph.replay()
            end.record()
            end.synchronize()
            device_us = begin.elapsed_time(end)*1000/(args.repetitions*calls_per_graph)
            row = {"batch": batch, "experts": args.experts, "top_k": args.top_k,
                   "backend": backend, **eager_timing,
                   "captured_device_us_per_call": device_us,
                   "repetitions": args.repetitions, "calls_per_graph": calls_per_graph}
            rows.append(row)
            output.write_text(json.dumps({"status": "RUNNING", "rows": rows,
                                          "timing_boundaries": TIMING_BOUNDARIES}, indent=2))
            print(json.dumps(row), flush=True)
    output.write_text(json.dumps({"status": "PASS", "rows": rows,
        "scope": "synthetic routing post-select overhead with counters; not MoE/service performance",
        "timing_boundaries": TIMING_BOUNDARIES,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}, indent=2))


if __name__ == "__main__":
    main()

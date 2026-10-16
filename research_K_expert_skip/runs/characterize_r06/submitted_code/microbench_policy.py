"""Measure post-selection policy overhead (not MoE/service speedup).

Run only through the project's GPU-lock launcher. Both eager wall dispatch and
captured-device time include the counters. CUDA graph capture excludes Python,
allocation, and launch dispatch between individual calls; disclose both values.
"""
import argparse
import json
from pathlib import Path
import time


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
            start = time.perf_counter()
            for _ in range(args.repetitions):
                call()
            torch.cuda.synchronize()
            eager_us = (time.perf_counter()-start)*1e6/args.repetitions
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
                   "backend": backend, "eager_wall_us_per_call": eager_us,
                   "captured_device_us_per_call": device_us,
                   "repetitions": args.repetitions, "calls_per_graph": calls_per_graph}
            rows.append(row)
            output.write_text(json.dumps({"status": "RUNNING", "rows": rows}, indent=2))
            print(json.dumps(row), flush=True)
    output.write_text(json.dumps({"status": "PASS", "rows": rows,
        "scope": "synthetic routing post-select overhead with counters; not MoE/service performance",
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}, indent=2))


if __name__ == "__main__":
    main()

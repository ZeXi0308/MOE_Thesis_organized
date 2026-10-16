#!/usr/bin/env python3
"""Characterize real vLLM fused MoE calls; write each finished case immediately.

Example:
  python bench_expert_formats.py --model olmoe --format fp8 \
    --tokens 1,8,32,128,256,1024,4096 --routing uniform --out runs/fp8.jsonl

Histogram JSON accepts {"counts": [per-expert counts]} or
{"topk_ids": [[...], ...], "topk_weights": [[...], ...]} (weights optional).
Counts are realized exactly with distinct experts per token. The resulting
synthetic co-occurrence pattern is explicitly not a full original route trace.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import heapq
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

from format_backends import ExpertShape, SHAPES, UnsupportedFormat, build_backend


def realize_histogram(counts, top_k):
    """Bipartite Havel-Hakimi: each token has k distinct expert neighbours."""
    if not counts or any(not isinstance(c, int) or c < 0 for c in counts):
        raise ValueError('Counts must be nonnegative integers')
    total = sum(counts)
    if total == 0 or total % top_k:
        raise ValueError('Histogram total must be positive and divisible by top_k')
    tokens = total // top_k
    if max(counts) > tokens:
        raise ValueError('An expert cannot occur twice in one token top-k')
    heap = [(-count, expert) for expert, count in enumerate(counts) if count]
    heapq.heapify(heap)
    rows = []
    for _ in range(tokens):
        if len(heap) < top_k:
            raise ValueError('Histogram is not realizable without duplicate experts')
        chosen = [heapq.heappop(heap) for _ in range(top_k)]
        rows.append([expert for _, expert in chosen])
        for negative, expert in chosen:
            if negative < -1:
                heapq.heappush(heap, (negative + 1, expert))
    if heap:
        raise AssertionError('Unconsumed histogram edges')
    return rows


def make_routing(tokens, shape, kind, seed, alpha=1.2, histogram=None):
    import torch
    generator = torch.Generator(device='cpu').manual_seed(seed)
    scores = None
    if kind == 'histogram':
        data = json.loads(Path(histogram).read_text())
        if isinstance(data, list):
            data = {'counts': data}
        if 'topk_ids' in data:
            ids = torch.tensor(data['topk_ids'], dtype=torch.int32)
            if 'topk_weights' in data:
                scores = torch.tensor(data['topk_weights'], dtype=torch.float32)
            provenance = 'supplied_topk_routes'
        else:
            counts = data['counts']
            if len(counts) != shape.experts:
                raise ValueError('Histogram length must equal expert count')
            ids = torch.tensor(realize_histogram(counts, shape.top_k), dtype=torch.int32)
            ids = ids[torch.randperm(ids.shape[0], generator=generator)]
            provenance = 'exact_histogram_synthetic_cooccurrence'
        if ids.shape != (tokens, shape.top_k):
            raise ValueError(f'Routes have shape {list(ids.shape)}, requested {(tokens, shape.top_k)}')
    elif kind == 'uniform':
        ids = (torch.arange(tokens * shape.top_k).reshape(tokens, shape.top_k)
               % shape.experts).to(torch.int32)
        # Permute expert identity without changing the balanced count pattern.
        permutation = torch.randperm(shape.experts, generator=generator)
        ids = permutation[ids.long()].to(torch.int32)
        provenance = 'synthetic_balanced_exact'
    elif kind == 'zipf':
        probabilities = torch.arange(1, shape.experts + 1, dtype=torch.float64).pow(-alpha)
        ids = torch.multinomial(probabilities.expand(tokens, -1), shape.top_k,
                                replacement=False, generator=generator).to(torch.int32)
        provenance = 'synthetic_zipf_weighted_without_replacement'
    else:
        raise ValueError(kind)
    if (ids < 0).any() or (ids >= shape.experts).any():
        raise ValueError('Route expert index out of range')
    sorted_ids = ids.sort(dim=1).values
    if (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any():
        raise ValueError('Duplicate expert in a token route')
    if scores is None:
        scores = torch.full(ids.shape, 1.0 / shape.top_k, dtype=torch.float32)
        score_source = 'uniform_normalized'
    else:
        if scores.shape != ids.shape or not torch.isfinite(scores).all() or (scores < 0).any():
            raise ValueError('Invalid topk_weights')
        score_source = 'supplied_unchanged'
    counts = torch.bincount(ids.long().flatten(), minlength=shape.experts).tolist()
    return ids.contiguous(), scores.contiguous(), {
        'kind': kind, 'provenance': provenance, 'seed': seed,
        'zipf_alpha': alpha if kind == 'zipf' else None,
        'histogram_file': str(Path(histogram).resolve()) if histogram else None,
        'topk_weight_source': score_source, 'counts': counts,
        'assignments': sum(counts), 'mean_per_expert': sum(counts) / shape.experts,
        'max_per_expert': max(counts), 'active_experts': sum(c > 0 for c in counts),
    }


def event_time(fn, iterations):
    import torch
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall = time.perf_counter()
    begin.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / iterations, (time.perf_counter() - wall) * 1000 / iterations


def profile_call(fn, trace_path):
    import torch
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))
    trace = json.loads(trace_path.read_text())
    events = [event for event in trace.get('traceEvents', [])
              if event.get('cat') == 'kernel']
    names = sorted({event['name'] for event in events})
    return {'trace': str(trace_path), 'kernel_names': names,
            'kernel_event_count': len(events),
            'kernel_duration_sum_us': sum(event.get('dur', 0) for event in events),
            'tensor_core_status': 'not established by kernel names alone'}


def measure_energy(fn, iterations, device, idle_power, idle_seconds, interval):
    import torch
    from power import PowerSampler
    sampler = PowerSampler(device=device, interval_s=interval, idle_power_w=idle_power)
    try:
        if idle_power is None:
            sampler.measure_idle(duration_s=idle_seconds, synchronize=torch.cuda.synchronize)
        with sampler.measure(synchronize=torch.cuda.synchronize) as measurement:
            for _ in range(iterations):
                fn()
        return measurement.result
    finally:
        sampler.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', choices=list(SHAPES), default='olmoe')
    p.add_argument('--d', type=int)
    p.add_argument('--intermediate', type=int)
    p.add_argument('--experts', type=int)
    p.add_argument('--top-k', type=int)
    p.add_argument('--format', choices=['bf16', 'fp8', 'nvfp4', 'awq'], required=True)
    p.add_argument('--tokens', default='1,8,32,128,256,1024,4096')
    p.add_argument('--routing', choices=['uniform', 'zipf', 'histogram'], default='uniform')
    p.add_argument('--zipf-alpha', type=float, default=1.2)
    p.add_argument('--histogram', type=Path)
    p.add_argument('--weights', type=Path, help='Calibrated, backend-processed tensor artifact')
    p.add_argument('--nvfp4-backend', choices=['marlin', 'flashinfer_b12x', 'cutlass'], default='marlin')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--iterations', type=int, default=0, help='0 chooses calls for target-repeat-ms')
    p.add_argument('--target-repeat-ms', type=float, default=100)
    p.add_argument('--max-iterations', type=int, default=10000)
    p.add_argument('--energy-seconds', type=float, default=2.0)
    p.add_argument('--idle-seconds', type=float, default=1.0)
    p.add_argument('--idle-power-w', type=float)
    p.add_argument('--power-interval-s', type=float, default=0.05)
    p.add_argument('--no-power', action='store_true')
    p.add_argument('--no-profile', action='store_true')
    p.add_argument('--cuda-graph', action='store_true', help='Capture each fixed routing/shape for replay; record mode')
    p.add_argument('--seed', type=int, default=20261015)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--nvml-device', help='NVML UUID or physical ordinal; default resolves CUDA UUID')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--continue-on-error', action='store_true')
    p.add_argument('--routing-only', action='store_true', help='Validate/generate route metadata on CPU')
    return p


def append_record(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        stream.flush()
    print(json.dumps({key: record[key] for key in ['status', 'format', 'tokens', 'latency_ms_median']
                      if key in record}), flush=True)


def main():
    args = parser().parse_args()
    base = SHAPES[args.model]
    shape = ExpertShape(args.d or base.d, args.intermediate or base.intermediate,
                        args.experts or base.experts, args.top_k or base.top_k)
    shape.validate()
    token_counts = [int(item) for item in args.tokens.split(',')]
    if min(token_counts) <= 0 or args.repeats <= 0 or args.warmup < 1:
        raise ValueError('Tokens/repeats/warmup must be positive')
    if args.target_repeat_ms <= 0 or args.max_iterations <= 0 or args.iterations < 0:
        raise ValueError('Timing iteration limits must be positive')
    if args.energy_seconds <= 0 or args.idle_seconds <= 0 or args.power_interval_s <= 0:
        raise ValueError('Power measurement durations must be positive')
    if args.routing == 'histogram' and not args.histogram:
        raise ValueError('--histogram is required')
    import torch
    if args.routing_only:
        for tokens in token_counts:
            _, _, route = make_routing(tokens, shape, args.routing, args.seed,
                                       args.zipf_alpha, args.histogram)
            append_record(args.out, {'status': 'routing_only', 'tokens': tokens,
                'format': args.format, 'shape': asdict(shape), 'routing': route})
        return
    import vllm
    import triton
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; use --routing-only for CPU checks')
    torch.cuda.set_device(args.device)
    device = f'cuda:{args.device}'
    properties = torch.cuda.get_device_properties(args.device)
    nvml_device = args.nvml_device
    if nvml_device is None:
        uuid = getattr(properties, 'uuid', None)
        if uuid is not None:
            nvml_device = str(uuid)
            if not nvml_device.startswith('GPU-'):
                nvml_device = 'GPU-' + nvml_device
        elif os.environ.get('CUDA_VISIBLE_DEVICES') and not args.no_power:
            raise ValueError('CUDA UUID unavailable under CUDA_VISIBLE_DEVICES; provide --nvml-device')
        else:
            nvml_device = args.device
    elif nvml_device.isdigit():
        nvml_device = int(nvml_device)
    environment = {
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'vllm': vllm.__version__, 'triton': triton.__version__,
        'gpu': properties.name, 'compute_capability': list(torch.cuda.get_device_capability(args.device)),
        'gpu_total_memory_bytes': properties.total_memory, 'nvml_device': nvml_device,
    }
    try:
        backend = build_backend(args.format, shape, device=device, seed=args.seed,
            weights=args.weights, nvfp4_backend=args.nvfp4_backend, max_tokens=max(token_counts))
    except Exception as exc:
        append_record(args.out, {'status': 'unsupported' if isinstance(exc, UnsupportedFormat) else 'error',
            'format': args.format, 'shape': asdict(shape), 'environment': environment,
            'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()})
        raise
    for tokens in token_counts:
        record = {'schema': 1, 'started_utc': datetime.now(timezone.utc).isoformat(),
            'format': args.format, 'model_shape': args.model, 'shape': asdict(shape),
            'tokens': tokens, 'environment': environment, 'backend': backend.metadata,
            'timing_mode': 'cuda_graph_replay' if args.cuda_graph else 'eager_cuda_events',
            'repetition_scope': 'within one process and weight residency; not independent system runs',
            'warmup_calls': args.warmup, 'repeats': args.repeats,
            'cache_policy': 'same resident weights/inputs/routes reused; no explicit cache flush'}
        try:
            torch.cuda.reset_peak_memory_stats()
            ids, scores, route = make_routing(tokens, shape, args.routing, args.seed,
                                              args.zipf_alpha, args.histogram)
            ids, scores = ids.to(device), scores.to(device)
            record['routing'] = route
            torch.manual_seed(args.seed + tokens)
            x = torch.randn((tokens, shape.d), dtype=torch.bfloat16, device=device)
            eager = lambda: backend(x, scores, ids)
            with torch.inference_mode():
                for _ in range(args.warmup):
                    result = eager()
                torch.cuda.synchronize()
                record['output'] = {'shape': list(result.shape),
                    'finite': bool(torch.isfinite(result).all().item()),
                    'abs_mean': result.float().abs().mean().item()}
                if result.shape != x.shape or not record['output']['finite']:
                    raise RuntimeError('Invalid output shape or nonfinite values')
                fn = eager
                if args.cuda_graph:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        graph_output = eager()
                    fn = graph.replay
                    fn()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(graph_output, result, rtol=0, atol=0)
                    record['cuda_graph_eager_output_match'] = True
                pilot_ms, _ = event_time(fn, 3)
                iterations = args.iterations or min(args.max_iterations,
                    max(1, math.ceil(args.target_repeat_ms / max(pilot_ms, 1e-6))))
                values = [event_time(fn, iterations) for _ in range(args.repeats)]
                latency = statistics.median(value[0] for value in values)
                record.update(status='ok', iterations_per_repeat=iterations,
                    latency_ms=[value[0] for value in values],
                    wall_ms_per_call=[value[1] for value in values], latency_ms_median=latency)
                flops = 6 * tokens * shape.top_k * shape.d * shape.intermediate
                active_fraction = route['active_experts'] / shape.experts
                weight_bytes = backend.metadata['resident_weight_and_scale_bytes'] * active_fraction
                record['roofline'] = {'projection_flops_per_call': flops,
                    'achieved_projection_tflops': flops / (latency / 1000) / 1e12,
                    'active_expert_weight_and_scale_bytes': weight_bytes,
                    'weight_only_arithmetic_intensity_flops_per_byte': flops / weight_bytes,
                    'effective_active_weight_bandwidth_tb_s': weight_bytes / (latency / 1000) / 1e12,
                    'note': 'Analytical projection FLOPs and one-read active weight bytes; '
                            'not hardware-counter DRAM traffic or measured theoretical roof.'}
                if not args.no_power:
                    energy_iters = max(1, math.ceil(args.energy_seconds * 1000 / latency))
                    power = measure_energy(fn, energy_iters, nvml_device, args.idle_power_w,
                                           args.idle_seconds, args.power_interval_s)
                    record['power'] = power
                    record['power_iterations'] = energy_iters
                    for key in ('energy_j', 'dynamic_energy_j'):
                        if power.get(key) is not None:
                            record[key + '_per_call'] = power[key] / energy_iters
                            record[key + '_per_token'] = power[key] / energy_iters / tokens
                if not args.no_profile:
                    trace_dir = args.out.parent / (args.out.stem + '_traces')
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    record['profile'] = profile_call(fn, trace_dir / f'{args.model}_{args.format}_{args.routing}_{tokens}.json')
                record['peak_memory_allocated_bytes'] = torch.cuda.max_memory_allocated()
        except Exception as exc:
            record.update(status='error', error_type=type(exc).__name__, error=str(exc),
                          traceback=traceback.format_exc())
            append_record(args.out, record)
            if not args.continue_on_error:
                raise
        else:
            append_record(args.out, record)


if __name__ == '__main__':
    main()

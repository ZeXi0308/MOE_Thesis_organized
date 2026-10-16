#!/usr/bin/env python3
"""Calibrate one real OLMoE BF16 layer on measured speculative route snapshots.

Uses the engine's vLLM 0.26 modular Triton path: FusedMoEKernel with
MoEPrepareAndFinalizeNoDPEPModular and TritonExperts. Routing, attention and
draft production are outside this microbenchmark. It must not be used as the
controller's complete-step calibration table.

Example (execution and GPU lock are managed by the campaign owner)::

  python code/bench_moe_routes.py --model /models/OLMoE-1B-7B-0924 \
    --routes 'runs/r02/gamma*/routes-b*.json' --layer 0 \
    --repeats 5 --out runs/kernel_routes_r01.jsonl

Exact topk_ids are used when saved by keep_token_routes=True. Otherwise an
exact degree realization of the measured histogram preserves each expert's
load but has synthetic token co-occurrence and rank ordering. The distinction
is recorded on every result. Hidden states are synthetic unit-RMS normals;
gate weights are uniform unless supplied in the trace. No token-quality claim.

CUDA events measure a stream interval (including possible host launch gaps),
not the sum of individual GPU kernel durations. Optional profiler output gives
the latter on a separate call with the same pre-call cache policy. Optional
--execution-mode cudagraph captures the identical fixed-input layer operation,
then times graph.replay; it is not a compiled model or service benchmark.
Five repeat means and their median
are reported; repeats share one process, resident weights and route snapshot.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
import glob
import hashlib
import heapq
import inspect
import json
import math
from pathlib import Path
import random
import re
import statistics
import struct
import time
import traceback


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def read_safetensors_header(path):
    with Path(path).open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if not 2 <= length <= min(100 << 20, Path(path).stat().st_size - 8):
            raise ValueError(f"Invalid safetensors header: {path}")
        return json.loads(stream.read(length))


def inspect_checkpoint(directory, layer):
    directory = Path(directory).resolve()
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("model_type") != "olmoe" or config.get("quantization_config"):
        raise ValueError("This benchmark currently accepts original unquantized OLMoE weights")
    if not 0 <= layer < int(config["num_hidden_layers"]):
        raise ValueError("Layer index outside checkpoint")
    if config.get("hidden_act", "silu") != "silu":
        raise ValueError("Expected OLMoE gated SiLU")
    shape = {"experts": int(config["num_experts"]),
             "top_k": int(config["num_experts_per_tok"]),
             "hidden": int(config["hidden_size"]),
             "intermediate": int(config["intermediate_size"])}
    index = directory / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text())["weight_map"]
    else:
        mapping = {}
        for shard in sorted(directory.glob("*.safetensors")):
            mapping.update({name: shard.name for name in read_safetensors_header(shard)
                            if name != "__metadata__"})
    names = {}
    for expert in range(shape["experts"]):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
            if name not in mapping:
                raise ValueError(f"Missing checkpoint tensor {name}")
            names[(expert, projection)] = name
    headers = {}
    for shard in sorted({mapping[name] for name in names.values()}):
        if Path(shard).is_absolute() or ".." in Path(shard).parts:
            raise ValueError("Expected checkpoint-relative shard path")
        headers[shard] = read_safetensors_header(directory / shard)
    for (expert, projection), name in names.items():
        header = headers[mapping[name]][name]
        expected = ([shape["hidden"], shape["intermediate"]]
                    if projection == "down_proj" else
                    [shape["intermediate"], shape["hidden"]])
        if header["dtype"] != "BF16" or header["shape"] != expected:
            raise ValueError(f"Unexpected original BF16 shape/dtype: {name}: {header}")
    info = {"checkpoint": str(directory), "model_type": "olmoe", "layer": layer,
            "shape": shape, "config_sha256": sha256_file(config_path),
            "index_sha256": sha256_file(index) if index.exists() else None,
            "source_shards": [{"name": shard, "bytes": (directory / shard).stat().st_size}
                              for shard in sorted(headers)],
            "layout": "w13=[gate_proj;up_proj],w2=down_proj; all experts resident BF16"}
    return directory, shape, mapping, names, info


def load_weights(checkpoint, device):
    import torch
    from safetensors import safe_open
    directory, shape, mapping, names, info = checkpoint
    e, h, n = shape["experts"], shape["hidden"], shape["intermediate"]
    w13 = torch.empty((e, 2 * n, h), dtype=torch.bfloat16)
    w2 = torch.empty((e, h, n), dtype=torch.bfloat16)
    digest = hashlib.sha256()
    with ExitStack() as stack:
        handles = {shard: stack.enter_context(safe_open(directory / shard, framework="pt", device="cpu"))
                   for shard in {mapping[name] for name in names.values()}}
        for (expert, projection), name in names.items():
            tensor = handles[mapping[name]].get_tensor(name).contiguous()
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Nonfinite checkpoint weight: {name}")
            digest.update(name.encode() + b"\n")
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
            target = (w2[expert] if projection == "down_proj" else
                      w13[expert, :n] if projection == "gate_proj" else w13[expert, n:])
            target.copy_(tensor)
    info = {**info, "selected_weights_sha256": digest.hexdigest(),
            "selected_hash_order": "expert_id then gate_proj/up_proj/down_proj",
            "weight_source": "original_checkpoint_payload",
            "bytes_per_expert": (w13[0].numel() + w2[0].numel()) * 2}
    return w13.to(device), w2.to(device), info


def realize_histogram(counts, top_k):
    """Bipartite Havel-Hakimi degree realization; no duplicate expert per row.

    Adapted from research_C_format_sched/code/bench_expert_formats.py. Histogram
    equality is exact; token association and rank are newly constructed.
    """
    if (not counts or top_k < 1 or any(type(c) is not int or c < 0 for c in counts)
            or sum(counts) == 0 or sum(counts) % top_k):
        raise ValueError("Expected nonnegative integer histogram and positive divisible edge count")
    tokens = sum(counts) // top_k
    if max(counts) > tokens:
        raise ValueError("Histogram would repeat an expert within a token")
    heap = [(-count, expert) for expert, count in enumerate(counts) if count]
    heapq.heapify(heap)
    rows = []
    for _ in range(tokens):
        if len(heap) < top_k:
            raise ValueError("Histogram not realizable with unique experts per token")
        chosen = [heapq.heappop(heap) for _ in range(top_k)]
        rows.append([expert for _, expert in chosen])
        for neg_count, expert in chosen:
            if neg_count < -1:
                heapq.heappush(heap, (neg_count + 1, expert))
    if heap:
        raise AssertionError("Unused route edges")
    return rows


def unpack_records(obj, context=None):
    context = dict(context or {})
    if isinstance(obj, list):
        return [row for item in obj for row in unpack_records(item, context)]
    if "expert_token_counts" in obj or "expert_counts" in obj or "topk_ids" in obj:
        # Native controller row.gamma is the actual step maximum; keep the
        # outer episode gamma separately so conditions do not merge silently.
        return [{**context, **obj}]
    if "nominal_gamma" in obj or "gamma" in obj:
        context["nominal_gamma"] = obj.get("nominal_gamma", obj.get("gamma"))
    if "batch" in obj:
        context["nominal_batch"] = obj["batch"]
    if "episodes" in obj:
        return [row for episode in obj["episodes"] for row in unpack_records(episode, context)]
    if "plugin" in obj:
        return unpack_records(obj["plugin"], context)
    for key in ("routes", "records"):
        if key in obj:
            return unpack_records(obj[key], context)
    return []


def load_route_records(patterns):
    paths = sorted({Path(path).resolve() for pattern in patterns
                    for path in glob.glob(pattern, recursive=True)})
    if not paths:
        raise ValueError("No route input files matched")
    rows = []
    for path in paths:
        if path.suffix in {".jsonl", ".ndjson"}:
            records = [row for line in path.read_text().splitlines() if line.strip()
                       for row in unpack_records(json.loads(line))]
        else:
            records = unpack_records(json.loads(path.read_text()))
        source_hash = sha256_file(path)
        rows.extend({**row, "source_file": str(path), "source_file_sha256": source_hash,
                     "source_record_index": index} for index, row in enumerate(records))
    return rows


def layer_index(name):
    if isinstance(name, int) or str(name).isdigit():
        return int(name)
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", str(name))
    return int(match.group(1)) if match else None


def prepare_route(record, shape, seed):
    counts = record.get("expert_token_counts", record.get("expert_counts"))
    if int(record.get("invalid_expert_assignments", 0)):
        raise ValueError("Route snapshot contains invalid expert assignments")
    k, e = shape["top_k"], shape["experts"]
    if int(record.get("top_k", k)) != k or int(record.get("num_experts", e)) != e:
        raise ValueError("Trace and checkpoint expert/top-k shapes differ")
    if "topk_ids" in record:
        rows = record["topk_ids"]
        kind = "exact_kernel_input_topk_ids"
    else:
        if counts is None or len(counts) != e:
            raise ValueError("Missing expert histogram for this model")
        rows = realize_histogram(counts, k)
        rng = random.Random(seed)
        rng.shuffle(rows)
        for row in rows:
            rng.shuffle(row)
        kind = "measured_histogram_synthetic_cooccurrence"
    if not rows or any(len(row) != k or len(set(row)) != k
                       or any(type(x) is not int or not 0 <= x < e for x in row) for row in rows):
        raise ValueError("Invalid top-k rows")
    actual = Counter(x for row in rows for x in row)
    actual_counts = [actual[x] for x in range(e)]
    if counts is not None and list(counts) != actual_counts:
        raise ValueError("Exact top-k rows do not match recorded expert histogram")
    recorded_tokens = record.get("tokens", record.get("num_tokens", len(rows)))
    if int(recorded_tokens) != len(rows):
        raise ValueError("Recorded token count disagrees with routing edges")
    weights = record.get("topk_weights")
    if weights is not None and (len(weights) != len(rows)
            or any(len(row) != k or any(not math.isfinite(x) or x < 0 for x in row) for row in weights)):
        raise ValueError("Invalid supplied top-k weights")
    return rows, weights, {"route_kind": kind, "tokens": len(rows),
        "expert_token_counts": actual_counts, "distinct_experts": len(actual),
        "topk_weight_source": "supplied_trace" if weights is not None else "uniform_normalized",
        "actual_gamma_counts": record.get("actual_gamma_counts"),
        "cooccurrence_preserved": kind == "exact_kernel_input_topk_ids"}


def select_cases(records, shape, layer, samples_per_cell, seed, batches=None, gammas=None,
                 include_token_extremes=False):
    groups = defaultdict(list)
    for row in records:
        if (layer_index(row.get("layer")) != layer
                or row.get("model_phase", "target") != "target"
                or row.get("phase", "decode") != "decode"):
            continue
        batch = int(row["batch"])
        gamma = int(row["nominal_gamma"] if "nominal_gamma" in row else row["gamma"])
        if batch < 1 or gamma < 0 or (batches and batch not in batches) or (gammas and gamma not in gammas):
            continue
        rows, weights, metadata = prepare_route(row, shape, seed)
        groups[(batch, gamma)].append((row, rows, weights, metadata))
    cases = []
    for (batch, gamma), values in sorted(groups.items()):
        values.sort(key=lambda item: (item[3]["distinct_experts"], item[3]["tokens"],
                                     item[0]["source_file"], item[0]["source_record_index"]))
        count = min(samples_per_cell, len(values))
        # Prespecified empirical quantiles, including the median for one sample.
        indices = sorted({min(len(values) - 1, int((i + 0.5) * len(values) / count))
                          for i in range(count)})
        selections = {index: ["midpoint_quantile"] for index in indices}
        if include_token_extremes:
            # Token extremes are observed snapshots, never constructed routes.
            # Within each tied S choose the upper median D (same source-order
            # tie-break used by the quantiles). A snapshot can carry >1 labels.
            for label, token_count in (("minimum_tokens", min(v[3]["tokens"] for v in values)),
                                       ("maximum_tokens", max(v[3]["tokens"] for v in values))):
                tied = [i for i, value in enumerate(values) if value[3]["tokens"] == token_count]
                index = tied[len(tied) // 2]
                selections.setdefault(index, []).append(label)
        # Values have stable source/index identities. Multiple selection rules
        # add labels to one case rather than timing the same snapshot repeatedly.
        for ordinal, index in enumerate(sorted(selections)):
            source, ids, weights, route = values[index]
            cases.append({"case_id": f"b{batch}-g{gamma}-q{ordinal}", "batch": batch,
                "gamma": gamma,
                "gamma_semantics": ("nominal_episode_gamma_not_assumed_realized_length"
                    if "nominal_gamma" in source else "legacy_row_gamma_no_episode_cap_available"),
                "nominal_batch": source.get("nominal_batch"),
                "step_gamma": source.get("gamma"),
                "source_file": source["source_file"], "source_file_sha256": source["source_file_sha256"],
                "source_record_index": source["source_record_index"], "source_step": source.get("step"),
                "source_episode": source.get("episode"), "layer": source["layer"],
                "cell_snapshot_population": len(values), "selected_rank_in_sorted_population": index,
                "snapshot_selection": ("midpoint_quantiles_plus_observed_token_extremes"
                    if include_token_extremes else "prespecified_midpoint_quantiles_of_distinct_then_tokens"),
                "selection_labels": selections[index],
                "extreme_tie_break": "upper_median_distinct_then_source_file_and_record_index",
                "all_cell_distinct_range": [values[0][3]["distinct_experts"], values[-1][3]["distinct_experts"]],
                "all_cell_token_range": [min(v[3]["tokens"] for v in values), max(v[3]["tokens"] for v in values)],
                "nominal_full_draft_tokens": batch * (gamma + 1),
                "actual_slots_match_full_draft": route["tokens"] == batch * (gamma + 1),
                **route, "_ids": ids, "_weights": weights})
    if not cases:
        raise ValueError("No target pure-decode snapshots matched requested layer/batch/gamma")
    return cases


def build_kernel(shape, device, max_tokens):
    import torch
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig, FusedMoEParallelConfig, FusedMoEQuantConfig, RoutingMethodType)
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import MoEPrepareAndFinalizeNoDPEPModular
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
    from vllm.v1.worker.workspace import init_workspace_manager
    init_workspace_manager(torch.device(device), num_ubatches=1)
    config = FusedMoEConfig(num_experts=shape["experts"], experts_per_token=shape["top_k"],
        hidden_dim=shape["hidden"], intermediate_size=shape["intermediate"],
        num_local_experts=shape["experts"], num_logical_experts=shape["experts"],
        activation=MoEActivation.SILU, device=device, routing_method=RoutingMethodType.RenormalizeNaive,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16, moe_backend="triton", max_num_tokens=max_tokens)
    experts = TritonExperts(config, FusedMoEQuantConfig.make())
    kernel = FusedMoEKernel(MoEPrepareAndFinalizeNoDPEPModular(), experts)
    sources = {}
    for cls in (FusedMoEKernel, TritonExperts, MoEPrepareAndFinalizeNoDPEPModular):
        source = inspect.getsourcefile(cls)
        sources[cls.__module__ + "." + cls.__name__] = {"source_file": source,
                                                       "source_file_sha256": sha256_file(source)}
    return kernel, MoEActivation.SILU, {"entrypoint": "FusedMoEKernel.apply",
        "experts_class": "TritonExperts", "prepare_finalize_class": "MoEPrepareAndFinalizeNoDPEPModular",
        "auto_backend_selection": False, "kernel_tuning": "installed vLLM lookup/defaults unchanged",
        "source_files": sources}


def resolved_tile_configuration(w13, w2, top_k, tokens):
    """Same unquantized BF16 lookup as TritonExperts.apply, outside timers."""
    import torch
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config
    name = FusedMoEQuantConfig.make().config_name(torch.bfloat16)
    source = inspect.getsourcefile(try_get_optimal_moe_config)
    return {"parameters": dict(try_get_optimal_moe_config(w13.size(), w2.size(),
                top_k, name, tokens, block_shape=None)),
            "lookup_dtype": name, "lookup_tokens": tokens,
            "source_file": source, "source_file_sha256": sha256_file(source),
            "scope": "resolved host configuration; not a hardware padding counter"}


def prepare_call_cache(cache_buffer):
    """Outside both timing events and the profiler; no exact cache-state claim."""
    import torch
    if cache_buffer is not None:
        cache_buffer.add_(1)
    torch.cuda.synchronize()


class CaseExecution:
    """Own one case's graph/output while caller retains its fixed input tensors."""
    def __init__(self, native_call, mode, device):
        if mode not in ('eager', 'cudagraph'):
            raise ValueError('execution mode must be eager or cudagraph')
        self.native_call, self.mode, self.device = native_call, mode, device
        self.graph = self.output = self.stream = self.call = None
        self.metadata = {"enabled": mode == 'cudagraph', "capture_succeeded": False,
            "scope": "same fixed-input MoE layer; no model graph padding or router capture"}

    def prepare(self, warmup):
        import torch
        if self.mode == 'eager':
            for _ in range(warmup):
                self.output = self.native_call()
            torch.cuda.synchronize()
            self.call = self.native_call
            return
        # Warmup on a side stream follows PyTorch's CUDA graph contract. Inputs,
        # weights and vLLM workspace remain referenced through the final replay.
        self.stream = torch.cuda.Stream(device=self.device)
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for _ in range(warmup):
                self.output = self.native_call()
        torch.cuda.current_stream().wait_stream(self.stream)
        torch.cuda.synchronize()
        reference = self.output.clone()
        torch.cuda.synchronize()
        begin = time.perf_counter_ns()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.output = self.native_call()
        torch.cuda.synchronize()
        self.metadata.update(capture_succeeded=True,
            capture_wall_ms=(time.perf_counter_ns()-begin)/1e6,
            capture_wall_scope='graph creation, stream capture/instantiation and final synchronization; excludes native warmup and first replay',
            native_warmup_calls=warmup, native_warmup_stream='side_stream')
        self.call = self.graph.replay
        begin = time.perf_counter_ns()
        self.call()
        torch.cuda.synchronize()
        self.metadata['first_replay_wall_ms'] = (time.perf_counter_ns()-begin)/1e6
        self.metadata['first_replay_outside_measurement'] = True
        self.metadata['matches_eager_warmup'] = bool(torch.allclose(
            self.output, reference, rtol=1e-2, atol=1e-2))
        self.metadata['comparison_tolerance'] = {'rtol':1e-2,'atol':1e-2}
        if not self.metadata['matches_eager_warmup']:
            raise RuntimeError('Captured layer output disagrees with eager warmup')

    def close(self):
        import torch
        # No graph can survive into the next case and retain stale workspace.
        torch.cuda.synchronize()
        if self.graph is not None:
            self.graph.reset()
        self.call = self.output = self.graph = self.stream = self.native_call = None


def time_repeat(fn, iterations, cache_buffer):
    import torch
    events, walls = [], []
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iterations):
        prepare_call_cache(cache_buffer)
        wall_begin = time.perf_counter_ns()
        begin.record()
        fn()
        end.record()
        end.synchronize()
        walls.append((time.perf_counter_ns() - wall_begin) / 1e6)
        events.append(begin.elapsed_time(end))
    return {"cuda_event_ms_mean": statistics.mean(events),
            "wall_ms_mean": statistics.mean(walls), "cuda_event_ms_calls": events,
            "wall_ms_calls": walls}


def profile_once(fn, path, cache_buffer=None, execution_mode='eager'):
    import torch
    prepare_call_cache(cache_buffer)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    events = [row for row in json.loads(path.read_text()).get("traceEvents", [])
              if row.get("cat") == "kernel"]
    return {"trace_file": str(path), "kernel_names": sorted({row["name"] for row in events}),
            "kernel_duration_sum_us": sum(row.get("dur", 0) for row in events),
            "kernel_count": len(events), "timing_scope": "separate_profiler_diagnostic_call",
            "execution_mode":execution_mode,
            "cache_policy":"turnover" if cache_buffer is not None else "resident",
            "cache_turnover_before_profile":cache_buffer is not None,
            "cache_turnover_bytes":cache_buffer.numel()*cache_buffer.element_size() if cache_buffer is not None else 0,
            "cache_preparation_scope":"same policy as timed calls; turnover and synchronization before entering profiler"}


def append(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
    print(json.dumps({key: value[key] for key in
          ("status", "case_id", "tokens", "distinct_experts", "cuda_event_ms_median") if key in value}), flush=True)


def self_test():
    rng = random.Random(41)
    for tokens in (1, 2, 5, 32):
        for k in (1, 3, 8):
            rows = [rng.sample(range(12), k) for _ in range(tokens)]
            counts = [sum(e in row for row in rows) for e in range(12)]
            replay = realize_histogram(counts, k)
            assert [sum(e in row for row in replay) for e in range(12)] == counts
            assert all(len(set(row)) == k for row in replay)
    shape = {"experts": 4, "top_k": 2}
    record = {"expert_token_counts": [2, 2, 2, 0], "tokens": 3,
              "topk_ids": [[0, 1], [1, 2], [0, 2]]}
    assert prepare_route(record, shape, 1)[2]["cooccurrence_preserved"]
    record.pop("topk_ids")
    assert not prepare_route(record, shape, 1)[2]["cooccurrence_preserved"]
    for bad in ([3, 1, 0, 0], [1, -1, 2, 0], [1, 0, 0, 0]):
        try:
            realize_histogram(bad, 2)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid histogram {bad}")
    assert len(unpack_records({"episodes": [{"plugin": {"routes": [record]}}]})) == 1
    inherited = unpack_records({"gamma": 7, "batch": 2,
        "plugin": {"routes": [dict(record, gamma=0)]}})[0]
    assert inherited["nominal_gamma"] == 7 and inherited["gamma"] == 0
    population = []
    for i, (tokens, histogram) in enumerate(((1, [1, 1, 0, 0]),
            (1, [1, 0, 1, 0]), (2, [2, 2, 0, 0]),
            (3, [3, 3, 0, 0]), (3, [2, 2, 1, 1]))):
        population.append({"batch": 1, "gamma": tokens - 1, "nominal_gamma": 7,
            "layer": 0, "tokens": tokens, "expert_token_counts": histogram,
            "source_file": "test", "source_file_sha256": "testhash", "source_record_index": i})
    cases = select_cases(population, shape, 0, 1, 1, include_token_extremes=True)
    assert {case["gamma"] for case in cases} == {7}
    assert {case["tokens"] for case in cases} == {1, 2, 3}
    assert len({case["source_record_index"] for case in cases}) == len(cases) == 3
    assert next(case for case in cases if "minimum_tokens" in case["selection_labels"])["source_record_index"] == 1
    assert next(case for case in cases if "maximum_tokens" in case["selection_labels"])["source_record_index"] == 4
    same = select_cases(population[:1], shape, 0, 1, 1, include_token_extremes=True)
    assert len(same) == 1 and set(same[0]["selection_labels"]) == {"midpoint_quantile", "minimum_tokens", "maximum_tokens"}
    assert layer_index("model.layers.12.mlp.experts") == 12
    print("CPU self-test passed: exact histogram realization, route identity, validation, schema extraction")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--routes", nargs="+")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--batches", help="Optional comma-separated observed B values")
    parser.add_argument("--gammas", help="Optional comma-separated controller gamma values")
    parser.add_argument("--samples-per-cell", type=int, default=1)
    parser.add_argument("--include-token-extremes", action="store_true",
                        help="Also select observed min/max S per cell; tied S uses median D; deduplicate source snapshots")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--execution-mode", choices=("eager", "cudagraph"), default="eager")
    parser.add_argument("--cache-policy", choices=("turnover", "resident"), default="turnover")
    parser.add_argument("--cache-turnover-mib", type=int, default=256,
                        help="Read/write unrelated buffer before each timed call, outside both timers")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--profile", action="store_true", help="Separate profiler call per selected snapshot")
    parser.add_argument("--plan-only", action="store_true", help="CPU checkpoint-header and route planning; no torch/CUDA")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.model is None or not args.routes or args.out is None:
        parser.error("--model, --routes and --out are required")
    if min(args.samples_per_cell, args.warmup, args.repeats, args.iterations, args.cache_turnover_mib) < 1:
        parser.error("Sample counts, warmup, repeat counts and cache size must be positive")
    if args.out.exists():
        raise FileExistsError(f"Preserve prior results; choose a new --out path: {args.out}")
    checkpoint = inspect_checkpoint(args.model, args.layer)
    shape = checkpoint[1]
    batches = set(map(int, args.batches.split(","))) if args.batches else None
    gammas = set(map(int, args.gammas.split(","))) if args.gammas else None
    cases = select_cases(load_route_records(args.routes), shape, args.layer,
                         args.samples_per_cell, args.seed, batches, gammas, args.include_token_extremes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.plan_only:
        for case in cases:
            append(args.out, {"status": "planned_cpu_only", "checkpoint": checkpoint[-1],
                              "execution_mode":args.execution_mode,"execution_mode_realized":False,
                              **{k: v for k, v in case.items() if not k.startswith("_")}})
        return
    import torch
    import triton
    import vllm
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; use --plan-only for CPU preparation")
    if vllm.__version__.split("+")[0] != "0.26.0":
        raise RuntimeError(f"This modular constructor targets vLLM 0.26.0; found {vllm.__version__}")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    environment = {"torch": str(torch.__version__), "vllm": vllm.__version__,
        "triton": triton.__version__, "cuda": torch.version.cuda, "gpu": props.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "gpu_memory_bytes": props.total_memory}
    w13, w2, weights_info = load_weights(checkpoint, device)
    kernel, activation, backend = build_kernel(shape, device, max(c["tokens"] for c in cases))
    cache_buffer = (torch.zeros(args.cache_turnover_mib << 18, dtype=torch.float32, device=device)
                    if args.cache_policy == "turnover" else None)
    failures = 0
    with torch.inference_mode():
        for case_index, case in enumerate(cases):
            record = {"schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
                **{k: v for k, v in case.items() if not k.startswith("_")},
                "environment": environment, "weights": weights_info, "backend": backend,
                "execution_mode":args.execution_mode,"execution_mode_realized":False,
                "input_source": "synthetic_unit_RMS_normal", "seed": args.seed + case_index,
                "repeats": args.repeats, "iterations_per_repeat": args.iterations,
                "warmup_calls": args.warmup, "cache_policy": args.cache_policy,
                "cache_turnover_bytes": cache_buffer.numel() * 4 if cache_buffer is not None else 0,
                "cache_turnover_scope": "outside timers; read/write unrelated buffer; no claim of exact cache state",
                "repetition_scope": "same process, weights and snapshot; not independent service runs",
                "measurement_scope": "one MoE layer: prepare, alignment, two expert GEMMs, activation and reduction; excludes router/attention/drafter",
                "selected_expert_storage_bytes": case["distinct_experts"] * weights_info["bytes_per_expert"],
                "selected_bytes_semantics": "distinct weight storage; not measured HBM traffic"}
            execution = None
            generator = x = ids = scores = output = run = None
            try:
                record['resolved_tile_configuration'] = resolved_tile_configuration(
                    w13, w2, shape['top_k'], case['tokens'])
                generator = torch.Generator(device=device).manual_seed(args.seed + case_index)
                x = torch.randn((case["tokens"], shape["hidden"]), generator=generator, device=device)
                x = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
                ids = torch.tensor(case["_ids"], device=device, dtype=torch.int32)
                scores = (torch.tensor(case["_weights"], device=device, dtype=torch.float32)
                          if case["_weights"] is not None else
                          torch.full(ids.shape, 1 / shape["top_k"], device=device, dtype=torch.float32))
                def run():
                    return kernel.apply(hidden_states=x, w1=w13, w2=w2, topk_weights=scores,
                        topk_ids=ids, activation=activation, global_num_experts=shape["experts"],
                        expert_map=None, apply_router_weight_on_input=False)
                execution = CaseExecution(run,args.execution_mode,device)
                record['graph_capture'] = execution.metadata
                execution.prepare(args.warmup)
                output = execution.output
                if output.shape != x.shape or not bool(torch.isfinite(output).all()):
                    raise RuntimeError("Unexpected output shape or nonfinite kernel output")
                record.update(execution_mode_realized=True,output_shape=list(output.shape),
                    output_dtype=str(output.dtype),output_finite_before_timing=True)
                repeats = [time_repeat(execution.call, args.iterations, cache_buffer) for _ in range(args.repeats)]
                # Separate untimed validation call reads the same output buffer
                # for graph replay, or the newest eager result.
                prepare_call_cache(cache_buffer)
                latest = execution.call()
                if args.execution_mode == 'eager':
                    execution.output = latest
                latest = None
                torch.cuda.synchronize()
                record['output_finite_after_timing'] = bool(torch.isfinite(execution.output).all())
                if not record['output_finite_after_timing']:
                    raise RuntimeError('Nonfinite layer output after measured calls')
                event_ms = statistics.median(r["cuda_event_ms_mean"] for r in repeats)
                record.update(status="ok", timing_repeats=repeats, cuda_event_ms_median=event_ms,
                    wall_ms_median=statistics.median(r["wall_ms_mean"] for r in repeats),
                    selected_storage_gb_per_event_second=record["selected_expert_storage_bytes"] / event_ms / 1e6,
                    proxy_rate_semantics="selected-storage/time ratio, not measured memory bandwidth")
                if args.profile:
                    record["profile"] = profile_once(execution.call,
                        args.out.with_name(args.out.stem + "." + case["case_id"] + ".trace.json"),
                        cache_buffer,args.execution_mode)
            except Exception as exc:
                failures += 1
                record.update(status="error", error=repr(exc), traceback=traceback.format_exc())
            finally:
                try:
                    if execution is not None:
                        execution.close()
                    record['case_resources_released'] = True
                except Exception as exc:
                    failures += 1
                    record.update(status='error',resource_release_error=repr(exc),case_resources_released=False)
                # Shared weights, turnover buffer and the vLLM workspace remain
                # resident. The caching allocator may retain free blocks.
                execution = generator = x = ids = scores = output = run = None
                record['resource_release_scope'] = 'graph reset and per-case input/output/closure references dropped; shared weights, turnover buffer, workspace and allocator cache retained'
            append(args.out, record)
    if failures:
        raise RuntimeError(f"{failures} kernel cases require follow-up; raw errors retained in {args.out}")


if __name__ == "__main__":
    main()

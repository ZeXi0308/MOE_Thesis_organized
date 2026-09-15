#!/usr/bin/env python3
"""Validity-first GPU audit for routing-guided MoE expert prefetch.

This experiment intentionally fixes the optimistic assumptions in the earlier
prototype before spending more effort on predictors:

1. Cache identity is (layer_id, expert_id), never expert_id alone.
2. The required working set is evaluated for both top-1 and the model's full
   top-k routing set.
3. Caches persist across a stream of decode-sized chunks. We evaluate both the
   legacy global-capacity interpretation and a per-layer partition.
4. Transition prefetch is compared with frequency prefetch, static frequency
   pinning, random prefetch, reactive LRU, and a one-layer oracle prefetcher.
5. The overlap window comes from a GPU-measured sparse padded batched expert
   compute proxy, not the earlier all-token x all-expert dense BMM.
6. H2D copies are measured both alone and concurrently with compute. A safe
   runtime budget may be zero; it is never forced to one.

The simulator still is not an end-to-end serving implementation. Its purpose
is to decide whether the direction survives basic validity and strong-baseline
checks before building one.
"""
from __future__ import annotations


# --- shared-lib bootstrap (auto) ---
import sys
from pathlib import Path as _Path

def _ensure_shared_on_path() -> None:
    here = _Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        cand = p / "experiments" / "shared"
        if (cand / "capture_moe.py").exists():
            s = str(cand)
            if s not in sys.path:
                sys.path.insert(0, s)
            return
        if (p / "capture_moe.py").exists():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
            return

_ensure_shared_on_path()
del _ensure_shared_on_path, _Path
# --- end bootstrap ---

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoConfig


ObjectKey = tuple[int, int]


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = max(0, int(capacity))
        self.order: list[ObjectKey] = []

    def contains(self, key: ObjectKey) -> bool:
        return key in self.order

    def touch(self, key: ObjectKey) -> ObjectKey | None:
        if self.capacity <= 0:
            return key
        if key in self.order:
            self.order.remove(key)
        self.order.append(key)
        if len(self.order) > self.capacity:
            return self.order.pop(0)
        return None


@dataclass
class LayerBatch:
    sequence: list[int]
    needed: set[int]
    current_top1: list[int]


@dataclass
class StreamChunk:
    sample_id: int
    chunk_id: int
    layers: dict[int, LayerBatch]


def load_routes(path: Path) -> pd.DataFrame:
    usecols = ["sample_id", "token_position", "layer", "expert_id", "rank"]
    df = pd.read_csv(path, usecols=usecols)
    for col in usecols:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
    return df


def split_calibration_test(routes: pd.DataFrame, calib_docs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    docs = sorted(routes["sample_id"].unique().tolist())
    if len(docs) <= calib_docs:
        raise ValueError(f"need >{calib_docs} documents, found {len(docs)}")
    calib_ids = set(docs[:calib_docs])
    return routes[routes["sample_id"].isin(calib_ids)].copy(), routes[~routes["sample_id"].isin(calib_ids)].copy()


def build_predictors(
    calib: pd.DataFrame,
    num_layers: int,
    num_experts: int,
    rank_limit: int,
) -> tuple[dict[int, np.ndarray], dict[int, dict[int, np.ndarray]]]:
    limited = calib[calib["rank"] <= rank_limit]
    frequency: dict[int, np.ndarray] = {}
    for layer in range(num_layers):
        counts = limited[limited["layer"] == layer]["expert_id"].value_counts()
        arr = np.zeros(num_experts, dtype=np.float64)
        for expert, count in counts.items():
            arr[int(expert)] = float(count)
        total = arr.sum()
        frequency[layer] = arr / total if total > 0 else arr

    top1 = calib[calib["rank"] == 1][["sample_id", "token_position", "layer", "expert_id"]]
    transition: dict[int, dict[int, np.ndarray]] = {}
    for layer in range(num_layers - 1):
        cur = top1[top1["layer"] == layer][["sample_id", "token_position", "expert_id"]].rename(
            columns={"expert_id": "prev_expert"}
        )
        nxt = limited[limited["layer"] == layer + 1][["sample_id", "token_position", "expert_id"]].rename(
            columns={"expert_id": "next_expert"}
        )
        joined = cur.merge(nxt, on=["sample_id", "token_position"], how="inner")
        table: dict[int, np.ndarray] = {}
        for prev_expert, group in joined.groupby("prev_expert"):
            arr = np.zeros(num_experts, dtype=np.float64)
            counts = group["next_expert"].value_counts()
            for expert, count in counts.items():
                arr[int(expert)] = float(count)
            denom = max(float((cur["prev_expert"] == prev_expert).sum()), 1.0)
            table[int(prev_expert)] = arr / denom
        transition[layer] = table
    return frequency, transition


def build_stream(
    test: pd.DataFrame,
    batch_tokens: int,
    rank_limit: int,
    num_layers: int,
) -> list[StreamChunk]:
    limited = test[test["rank"] <= rank_limit]
    stream: list[StreamChunk] = []
    for sample_id, doc in limited.groupby("sample_id", sort=True):
        positions = sorted(doc["token_position"].unique().tolist())
        chunks = [positions[i : i + batch_tokens] for i in range(0, len(positions), batch_tokens)]
        for chunk_id, positions_chunk in enumerate(chunks):
            if len(positions_chunk) < max(4, batch_tokens // 4):
                continue
            subset = doc[doc["token_position"].isin(positions_chunk)]
            layers: dict[int, LayerBatch] = {}
            for layer in range(num_layers):
                layer_rows = subset[subset["layer"] == layer].sort_values(["token_position", "rank"])
                if layer_rows.empty:
                    continue
                sequence = layer_rows["expert_id"].astype(int).tolist()
                top1_values = layer_rows[layer_rows["rank"] == 1]["expert_id"].astype(int).tolist()
                layers[layer] = LayerBatch(
                    sequence=sequence,
                    needed=set(sequence),
                    current_top1=top1_values,
                )
            if layers:
                stream.append(StreamChunk(int(sample_id), chunk_id, layers))
    if not stream:
        raise ValueError("test stream is empty")
    return stream


def ordered_unique(values: Iterable[int]) -> list[int]:
    return list(dict.fromkeys(int(v) for v in values))


def rank_candidates(
    policy: str,
    layer: int,
    current_top1: list[int],
    next_needed: set[int],
    frequency: dict[int, np.ndarray],
    transition: dict[int, dict[int, np.ndarray]],
    num_experts: int,
    rng: np.random.Generator,
) -> list[int]:
    target_layer = layer + 1
    if policy == "oracle_prefetch":
        return sorted(next_needed, key=lambda e: frequency[target_layer][e], reverse=True)
    if policy == "random_prefetch":
        values = np.arange(num_experts)
        rng.shuffle(values)
        return values.astype(int).tolist()
    if policy == "frequency_prefetch":
        scores = frequency[target_layer]
        return np.argsort(-scores).astype(int).tolist()
    if policy == "transition_prefetch":
        scores = np.zeros(num_experts, dtype=np.float64)
        table = transition.get(layer, {})
        for prev_expert in current_top1:
            scores += table.get(int(prev_expert), frequency[target_layer])
        if not np.any(scores):
            scores = frequency[target_layer]
        return np.argsort(-scores).astype(int).tolist()
    raise ValueError(policy)


def init_caches(
    scope: str,
    capacity: int,
    num_layers: int,
) -> tuple[LRUCache | None, dict[int, LRUCache]]:
    if scope == "global":
        return LRUCache(capacity), {}
    if scope == "per_layer":
        return None, {layer: LRUCache(capacity) for layer in range(num_layers)}
    raise ValueError(scope)


def get_cache(
    scope: str,
    layer: int,
    global_cache: LRUCache | None,
    layer_caches: dict[int, LRUCache],
) -> LRUCache:
    return global_cache if scope == "global" else layer_caches[layer]


def concurrent_exposed_us(hardware: dict, copies: int) -> float:
    if copies <= 0:
        return 0.0
    curve = hardware["concurrent_curve"]
    key = str(copies)
    compute_us = float(hardware["sparse_compute_us_median"])
    if key in curve:
        return max(0.0, float(curve[key]["total_us"]) - compute_us)
    max_n = max(int(k) for k in curve)
    max_total = float(curve[str(max_n)]["total_us"])
    extra = (copies - max_n) * float(hardware["h2d_one_expert_us"])
    return max(0.0, max_total + extra - compute_us)


def simulate(
    stream: list[StreamChunk],
    policy: str,
    cache_scope: str,
    capacity: int,
    prefetch_budget: int,
    num_layers: int,
    num_experts: int,
    frequency: dict[int, np.ndarray],
    transition: dict[int, dict[int, np.ndarray]],
    hardware: dict,
    seed: int,
) -> dict[str, float | int | str]:
    h2d_us = float(hardware["h2d_one_expert_us"])
    compute_us = float(hardware["sparse_compute_us_median"])
    global_cache, layer_caches = init_caches(cache_scope, capacity, num_layers)
    pending_prefetch: set[ObjectKey] = set()
    rng = np.random.default_rng(seed)

    if policy == "static_frequency_pin":
        if cache_scope == "per_layer":
            for layer in range(num_layers):
                cache = layer_caches[layer]
                for expert in np.argsort(-frequency[layer])[:capacity]:
                    cache.touch((layer, int(expert)))
        else:
            ranked_objects: list[tuple[float, ObjectKey]] = []
            for layer in range(num_layers):
                ranked_objects.extend((float(frequency[layer][e]), (layer, e)) for e in range(num_experts))
            ranked_objects.sort(reverse=True)
            assert global_cache is not None
            for _, key in ranked_objects[:capacity]:
                global_cache.touch(key)

    total_latency_us = 0.0
    total_compute_us = 0.0
    total_demand_us = 0.0
    total_exposed_prefetch_us = 0.0
    total_needed = 0
    total_misses = 0
    total_prefetches = 0
    useful_prefetches = 0
    wasted_prefetches = 0

    dynamic_cache = policy != "static_frequency_pin"
    prefetch_policy = policy in {
        "frequency_prefetch",
        "transition_prefetch",
        "random_prefetch",
        "oracle_prefetch",
    }

    for chunk in stream:
        for layer in range(num_layers):
            batch = chunk.layers.get(layer)
            if batch is None:
                continue
            cache = get_cache(cache_scope, layer, global_cache, layer_caches)
            needed_keys = {(layer, expert) for expert in batch.needed}
            total_needed += len(needed_keys)
            hits = {key for key in needed_keys if cache.contains(key)}
            useful = hits & pending_prefetch
            useful_prefetches += len(useful)
            pending_prefetch.difference_update(useful)
            misses = needed_keys - hits
            total_misses += len(misses)
            demand_us = len(misses) * h2d_us
            total_demand_us += demand_us
            total_latency_us += demand_us + compute_us
            total_compute_us += compute_us

            if dynamic_cache:
                for expert in batch.sequence:
                    evicted = cache.touch((layer, int(expert)))
                    if evicted in pending_prefetch:
                        pending_prefetch.remove(evicted)
                        wasted_prefetches += 1

            if prefetch_policy and layer + 1 < num_layers:
                next_batch = chunk.layers.get(layer + 1)
                if next_batch is None:
                    continue
                target_cache = get_cache(cache_scope, layer + 1, global_cache, layer_caches)
                candidates = rank_candidates(
                    policy,
                    layer,
                    batch.current_top1,
                    next_batch.needed,
                    frequency,
                    transition,
                    num_experts,
                    rng,
                )
                selected: list[int] = []
                for expert in candidates:
                    key = (layer + 1, int(expert))
                    if key in pending_prefetch or target_cache.contains(key):
                        continue
                    selected.append(int(expert))
                    if len(selected) >= prefetch_budget:
                        break
                total_prefetches += len(selected)
                exposed_us = concurrent_exposed_us(hardware, len(selected))
                total_exposed_prefetch_us += exposed_us
                total_latency_us += exposed_us
                for expert in selected:
                    key = (layer + 1, expert)
                    evicted = target_cache.touch(key)
                    if evicted in pending_prefetch:
                        pending_prefetch.remove(evicted)
                        wasted_prefetches += 1
                    pending_prefetch.add(key)

    wasted_prefetches += len(pending_prefetch)
    return {
        "policy": policy,
        "cache_scope": cache_scope,
        "capacity": capacity,
        "prefetch_budget": prefetch_budget,
        "total_latency_ms": total_latency_us / 1000.0,
        "total_compute_ms": total_compute_us / 1000.0,
        "total_demand_ms": total_demand_us / 1000.0,
        "total_exposed_prefetch_ms": total_exposed_prefetch_us / 1000.0,
        "miss_rate": total_misses / max(total_needed, 1),
        "total_needed_objects": total_needed,
        "total_misses": total_misses,
        "total_prefetches": total_prefetches,
        "useful_prefetches": useful_prefetches,
        "wasted_prefetches": wasted_prefetches,
        "prefetch_precision": useful_prefetches / max(total_prefetches, 1),
    }


def event_median_us(values_ms: list[float]) -> float:
    return float(np.median(values_ms) * 1000.0)


def measure_gpu_hardware(
    config,
    representative_shapes: list[tuple[int, int, int]],
    max_prefetch: int,
    repeats: int,
    device: str,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this audit")
    hidden = int(config.hidden_size)
    intermediate = int(getattr(config, "moe_intermediate_size", getattr(config, "intermediate_size")))
    dtype = torch.bfloat16
    dev = torch.device(device)

    max_active = max(shape[0] for shape in representative_shapes)
    gate_w = torch.empty((max_active, intermediate, hidden), device=dev, dtype=dtype)
    up_w = torch.empty((max_active, intermediate, hidden), device=dev, dtype=dtype)
    down_w = torch.empty((max_active, hidden, intermediate), device=dev, dtype=dtype)
    gate_w.uniform_(-0.02, 0.02)
    up_w.uniform_(-0.02, 0.02)
    down_w.uniform_(-0.02, 0.02)

    compute_measurements: list[dict] = []
    for active, max_tokens_per_expert, assignments in representative_shapes:
        x = torch.empty((active, max_tokens_per_expert, hidden), device=dev, dtype=dtype)
        x.uniform_(-0.02, 0.02)

        def compute_once():
            gate = torch.bmm(x, gate_w[:active].transpose(1, 2))
            up = torch.bmm(x, up_w[:active].transpose(1, 2))
            act = F.silu(gate) * up
            return torch.bmm(act, down_w[:active].transpose(1, 2))

        for _ in range(5):
            compute_once()
        torch.cuda.synchronize()
        times_ms: list[float] = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            compute_once()
            end.record()
            end.synchronize()
            times_ms.append(start.elapsed_time(end))
        compute_measurements.append(
            {
                "active_experts": active,
                "max_tokens_per_expert": max_tokens_per_expert,
                "actual_assignments": assignments,
                "padded_assignments": active * max_tokens_per_expert,
                "padding_factor": (active * max_tokens_per_expert) / max(assignments, 1),
                "compute_us": event_median_us(times_ms),
            }
        )
    compute_us = float(np.median([row["compute_us"] for row in compute_measurements]))
    median_shape = min(
        representative_shapes,
        key=lambda shape: abs(shape[0] * shape[1] - np.median([a * m for a, m, _ in representative_shapes])),
    )
    active, max_tokens_per_expert, _ = median_shape
    x_concurrent = torch.empty((active, max_tokens_per_expert, hidden), device=dev, dtype=dtype)
    x_concurrent.uniform_(-0.02, 0.02)

    def compute_concurrent_once():
        gate = torch.bmm(x_concurrent, gate_w[:active].transpose(1, 2))
        up = torch.bmm(x_concurrent, up_w[:active].transpose(1, 2))
        act = F.silu(gate) * up
        return torch.bmm(act, down_w[:active].transpose(1, 2))

    expert_numel = 3 * hidden * intermediate
    cpu_copy = torch.empty((max_prefetch, expert_numel), dtype=dtype, pin_memory=True)
    gpu_copy = torch.empty((max_prefetch, expert_numel), dtype=dtype, device=dev)
    copy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()

    def measure_copy_count(n: int) -> float:
        times_ms: list[float] = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            copy_stream.wait_event(start)
            with torch.cuda.stream(copy_stream):
                for idx in range(n):
                    gpu_copy[idx].copy_(cpu_copy[idx], non_blocking=True)
                end.record()
            end.synchronize()
            times_ms.append(start.elapsed_time(end))
        return event_median_us(times_ms)

    h2d_one_us = measure_copy_count(1)
    concurrent_curve: dict[str, dict[str, float]] = {}
    for n in range(1, max_prefetch + 1):
        totals_ms: list[float] = []
        compute_times_ms: list[float] = []
        copy_times_ms: list[float] = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            comp_end = torch.cuda.Event(enable_timing=True)
            copy_end = torch.cuda.Event(enable_timing=True)
            start.record()
            compute_stream.wait_event(start)
            copy_stream.wait_event(start)
            with torch.cuda.stream(compute_stream):
                compute_concurrent_once()
                comp_end.record()
            with torch.cuda.stream(copy_stream):
                for idx in range(n):
                    gpu_copy[idx].copy_(cpu_copy[idx], non_blocking=True)
                copy_end.record()
            comp_end.synchronize()
            copy_end.synchronize()
            comp_ms = start.elapsed_time(comp_end)
            copy_ms = start.elapsed_time(copy_end)
            compute_times_ms.append(comp_ms)
            copy_times_ms.append(copy_ms)
            totals_ms.append(max(comp_ms, copy_ms))
        concurrent_curve[str(n)] = {
            "total_us": event_median_us(totals_ms),
            "compute_stream_us": event_median_us(compute_times_ms),
            "copy_stream_us": event_median_us(copy_times_ms),
            "exposed_vs_compute_us": max(0.0, event_median_us(totals_ms) - compute_us),
        }

    safe_budget = 0
    for n in range(1, max_prefetch + 1):
        if concurrent_curve[str(n)]["total_us"] <= 1.05 * compute_us:
            safe_budget = n

    expert_bytes = expert_numel * torch.tensor([], dtype=dtype).element_size()
    del gate_w, up_w, down_w, x_concurrent, cpu_copy, gpu_copy
    torch.cuda.empty_cache()
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "expert_bytes": expert_bytes,
        "h2d_one_expert_us": h2d_one_us,
        "sparse_compute_us_median": compute_us,
        "safe_overlap_budget_5pct": safe_budget,
        "compute_measurements": compute_measurements,
        "concurrent_curve": concurrent_curve,
    }


def representative_shapes(
    test: pd.DataFrame,
    batch_tokens: int,
    rank_limit: int,
    limit: int = 128,
) -> list[tuple[int, int, int]]:
    limited = test[test["rank"] <= rank_limit]
    shapes: list[tuple[int, int, int]] = []
    for _, doc in limited.groupby("sample_id", sort=True):
        positions = sorted(doc["token_position"].unique().tolist())
        for start in range(0, len(positions), batch_tokens):
            chunk = positions[start : start + batch_tokens]
            if len(chunk) < max(4, batch_tokens // 4):
                continue
            rows = doc[doc["token_position"].isin(chunk)]
            for _, layer_rows in rows.groupby("layer"):
                counts = layer_rows["expert_id"].value_counts()
                if counts.empty:
                    continue
                shapes.append((len(counts), int(counts.max()), int(counts.sum())))
                if len(shapes) >= limit:
                    break
            if len(shapes) >= limit:
                break
        if len(shapes) >= limit:
            break
    if not shapes:
        raise ValueError("could not derive representative routed shapes")
    padded = np.array([a * m for a, m, _ in shapes])
    quantiles = [0.25, 0.5, 0.95]
    selected = []
    for quantile in quantiles:
        target = float(np.quantile(padded, quantile))
        selected.append(min(shapes, key=lambda shape: abs(shape[0] * shape[1] - target)))
    return selected


def working_set_summary(stream: list[StreamChunk], num_experts: int) -> dict[str, float]:
    sizes = np.array([len(batch.needed) for chunk in stream for batch in chunk.layers.values()], dtype=float)
    return {
        "mean_unique_experts": float(sizes.mean()),
        "p50_unique_experts": float(np.quantile(sizes, 0.5)),
        "p95_unique_experts": float(np.quantile(sizes, 0.95)),
        "mean_fraction_of_all_experts": float((sizes / num_experts).mean()),
        "fraction_layers_requiring_all_experts": float((sizes >= num_experts).mean()),
        "n_layer_batches": int(len(sizes)),
    }


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--routes", required=True)
    parser.add_argument("--calib-docs", type=int, default=12)
    parser.add_argument("--batch-tokens", type=int, default=32)
    parser.add_argument("--capacities", default="8,16,32")
    parser.add_argument("--requested-prefetch-budget", type=int, default=8)
    parser.add_argument("--max-concurrent-copies", type=int, default=8)
    parser.add_argument("--gpu-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    num_experts_value = getattr(config, "num_experts", None)
    if num_experts_value is None:
        num_experts_value = getattr(config, "num_local_experts")
    num_experts = int(num_experts_value)
    top_k = int(config.num_experts_per_tok)
    num_layers = int(getattr(config, "num_hidden_layers"))
    routes = load_routes(Path(args.routes))
    calib, test = split_calibration_test(routes, args.calib_docs)

    shapes = representative_shapes(test, args.batch_tokens, top_k)
    hardware = measure_gpu_hardware(
        config,
        shapes,
        args.max_concurrent_copies,
        args.gpu_repeats,
        "cuda",
    )
    (output / "hardware.json").write_text(json.dumps(hardware, indent=2), encoding="utf-8")

    all_rows: list[dict] = []
    working_rows: list[dict] = []
    capacities = parse_int_list(args.capacities)
    policies = [
        "reactive_lru",
        "static_frequency_pin",
        "frequency_prefetch",
        "transition_prefetch",
        "random_prefetch",
        "oracle_prefetch",
    ]
    safe_budget = int(hardware["safe_overlap_budget_5pct"])
    budgets = sorted(set([0, safe_budget, args.requested_prefetch_budget]))

    for rank_mode, rank_limit in [("top1", 1), ("full_topk", top_k)]:
        frequency, transition = build_predictors(calib, num_layers, num_experts, rank_limit)
        stream = build_stream(test, args.batch_tokens, rank_limit, num_layers)
        working_rows.append(
            {
                "model": args.model_key,
                "rank_mode": rank_mode,
                "rank_limit": rank_limit,
                **working_set_summary(stream, num_experts),
            }
        )
        for cache_scope in ["global", "per_layer"]:
            for capacity in capacities:
                for budget in budgets:
                    for policy in policies:
                        if policy in {"reactive_lru", "static_frequency_pin"} and budget != 0:
                            continue
                        if policy not in {"reactive_lru", "static_frequency_pin"} and budget == 0:
                            continue
                        result = simulate(
                            stream,
                            policy,
                            cache_scope,
                            capacity,
                            budget,
                            num_layers,
                            num_experts,
                            frequency,
                            transition,
                            hardware,
                            args.seed + capacity + rank_limit + budget,
                        )
                        result.update(
                            {
                                "model": args.model_key,
                                "rank_mode": rank_mode,
                                "rank_limit": rank_limit,
                                "safe_overlap_budget": safe_budget,
                            }
                        )
                        all_rows.append(result)

    results = pd.DataFrame(all_rows)
    baseline = results[results["policy"] == "reactive_lru"][
        ["model", "rank_mode", "cache_scope", "capacity", "total_latency_ms"]
    ].rename(columns={"total_latency_ms": "reactive_latency_ms"})
    results = results.merge(baseline, on=["model", "rank_mode", "cache_scope", "capacity"], how="left")
    results["latency_saving_vs_reactive_pct"] = 100.0 * (
        1.0 - results["total_latency_ms"] / results["reactive_latency_ms"]
    )
    results.to_csv(output / "simulation_results.csv", index=False)
    working = pd.DataFrame(working_rows)
    working.to_csv(output / "working_set_summary.csv", index=False)

    best = (
        results.sort_values("latency_saving_vs_reactive_pct", ascending=False)
        .groupby(["rank_mode", "cache_scope", "capacity"], as_index=False)
        .first()
    )
    best.to_csv(output / "best_policy_by_setting.csv", index=False)

    print(json.dumps({"model": args.model_key, "hardware": hardware, "working_set": working_rows}, indent=2))
    print("\nBest policy by setting:")
    print(
        best[
            [
                "rank_mode",
                "cache_scope",
                "capacity",
                "policy",
                "prefetch_budget",
                "latency_saving_vs_reactive_pct",
                "miss_rate",
                "prefetch_precision",
            ]
        ].to_string(index=False)
    )
    print(f"\nsaved to {output}")


if __name__ == "__main__":
    main()

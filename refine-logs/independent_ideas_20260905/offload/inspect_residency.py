#!/usr/bin/env python3
"""Read OLMoE tensor headers and bound weights + dense BF16 KV; no weight load.

This is a capacity planning calculation, not an offload runtime or a measurement
of available HBM, cache misses, exposed fetch stalls, or serving performance.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct

GIB = 2 ** 30
EXPERT = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(down|gate|up)_proj\.weight$")


def read_weights(snapshot):
    config_bytes = (snapshot / "config.json").read_bytes()
    index_bytes = (snapshot / "model.safetensors.index.json").read_bytes()
    config, index = json.loads(config_bytes), json.loads(index_bytes)
    if config.get("model_type") != "olmoe" or config.get("torch_dtype") != "bfloat16":
        raise ValueError("this preparation supports the pinned BF16 OLMoE layout only")
    tensors, headers = {}, {}
    for name in sorted(set(index["weight_map"].values())):
        if Path(name).name != name:
            raise ValueError("shard must be a local filename")
        shard = snapshot / name
        with shard.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            if not 0 < size <= 16 * 2 ** 20:
                raise ValueError("unexpected tensor header size")
            raw = stream.read(size)
        headers[name] = hashlib.sha256(raw).hexdigest()
        for key, tensor in json.loads(raw).items():
            if key == "__metadata__":
                continue
            start, end = tensor["data_offsets"]
            shape = tensor["shape"]
            if (key in tensors or index["weight_map"].get(key) != name or tensor["dtype"] != "BF16"
                    or any(type(dim) is not int or dim <= 0 for dim in shape)
                    or not 0 <= start < end <= shard.stat().st_size - 8 - size
                    or end - start != math.prod(shape) * 2):
                raise ValueError("tensor index/shape/storage mismatch")
            tensors[key] = end - start
    if set(tensors) != set(index["weight_map"]):
        raise ValueError("incomplete tensor inventory")
    experts, projections = {}, {}
    for key, size in tensors.items():
        match = EXPERT.fullmatch(key)
        if match:
            layer, expert, projection = match.groups()
            identity = (int(layer), int(expert))
            experts[identity] = experts.get(identity, 0) + size
            projections.setdefault(identity, set()).add(projection)
    expected = {(layer, expert) for layer in range(config["num_hidden_layers"])
                for expert in range(config["num_experts"])}
    if set(experts) != expected or any(p != {"down", "gate", "up"} for p in projections.values()):
        raise ValueError("expert identity must include layer and all three projections")
    if len(set(experts.values())) != 1:
        raise ValueError("uniform expert-size formula is not valid for this checkpoint")
    total = sum(tensors.values())
    if total != index["metadata"]["total_size"]:
        raise ValueError("index total disagrees with tensor byte inventory")
    return config, dict(weight_bytes=total, expert_bytes=sum(experts.values()),
        nonexpert_bytes=total - sum(experts.values()), bytes_per_expert=next(iter(experts.values())),
        expert_objects=len(experts), config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        index_sha256=hashlib.sha256(index_bytes).hexdigest(), header_sha256=headers,
        weights_loaded=False, weight_payload_hashed=False)


def capacity(config, inventory, active, context, hbm_bytes, reserved_bytes=0):
    if (type(active) is not int or active < 1 or type(context) is not int
            or not 1 <= context <= config["max_position_embeddings"]):
        raise ValueError("positive concurrency and context within checkpoint support are required")
    if (not math.isfinite(hbm_bytes) or hbm_bytes <= 0 or not math.isfinite(reserved_bytes)
            or not 0 <= reserved_bytes < hbm_bytes):
        raise ValueError("finite positive HBM and a smaller nonnegative reserve required")
    heads = config["num_attention_heads"]
    if config["hidden_size"] % heads:
        raise ValueError("head dimension is not integral")
    per_token = 2 * config["num_hidden_layers"] * config["num_key_value_heads"] * (config["hidden_size"] // heads) * 2
    kv = active * context * per_token
    lower_bound = inventory["weight_bytes"] + kv
    expert_capacity = hbm_bytes - reserved_bytes - inventory["nonexpert_bytes"] - kv
    return dict(active_requests=active, cached_tokens_per_request=context, dense_bf16_kv_bytes=kv,
        hbm_scenario_bytes=hbm_bytes, assumed_other_memory_bytes=reserved_bytes,
        weights_plus_kv_lower_bound_bytes=lower_bound,
        remaining_before_runtime_overhead_bytes=hbm_bytes - lower_bound,
        optimistic_expert_slots=max(0, min(inventory["expert_objects"], int(expert_capacity // inventory["bytes_per_expert"]))),
        necessary_pressure=lower_bound > hbm_bytes,
        status="LOWER_BOUND_EXCEEDS_HBM" if lower_bound > hbm_bytes else "FIT_NOT_PROVEN",
        action_space="UNMEASURED", fetch_stall_fraction=None, full_request_gain=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--hbm-gib", type=float, required=True, help="scenario capacity, not measured free memory")
    parser.add_argument("--caps", default="2,4,8,16,32")
    parser.add_argument("--contexts", default="144,4096", help="total cached tokens, including generation")
    parser.add_argument("--reserved-runtime-gib", type=float, default=0, help="explicit planning assumption only")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config, inventory = read_weights(args.snapshot)
    rows = [capacity(config, inventory, active, context, args.hbm_gib * GIB, args.reserved_runtime_gib * GIB)
            for context in map(int, args.contexts.split(",")) for active in map(int, args.caps.split(","))]
    result = dict(evidence_type="STRUCTURAL_CAPACITY_CALCULATION", scientific_status="UNRUN",
        snapshot_name=args.snapshot.name, inventory=inventory, cells=rows,
        limits=["HBM size and reserve are supplied scenarios, not measurements of free memory.",
            "Dense BF16 KV formula excludes activation, workspace, allocator, graphs, and fragmentation.",
            "Context beyond the checkpoint limit is rejected; shared-prefix KV savings are not modeled.",
            "Capacity pressure does not establish natural misses, exposed stalls, or SLO usefulness.",
            "No native offload producer, LRU/pinning execution, or future-known transfer Oracle is implemented."])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "capacity.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["evidence_type"], "weight_gib": inventory["weight_bytes"] / GIB,
                      "cells": len(rows), "scientific_status": "UNRUN"}))


if __name__ == "__main__":
    main()

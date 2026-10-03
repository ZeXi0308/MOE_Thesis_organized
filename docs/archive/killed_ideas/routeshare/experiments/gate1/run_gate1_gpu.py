#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from route_pairing import Request, pair_greedy, pair_random, union_invocations  # noqa: E402


SPECS = {
    "olmoe": {"path": "/root/autodl-tmp/models/olmoe", "experts": 64, "set_size": 8},
    "llmjp": {"path": "/root/autodl-tmp/models/llmjp", "experts": 32, "set_size": 16},
}


def experts_at(model, layer: int):
    block = model.model.layers[layer]
    if hasattr(block, "block_sparse_moe"):
        return block.block_sparse_moe.experts
    return block.mlp.experts


def projections(expert):
    if all(hasattr(expert, name) for name in ("gate_proj", "up_proj", "down_proj")):
        return expert.gate_proj, expert.up_proj, expert.down_proj
    if all(hasattr(expert, name) for name in ("w1", "w3", "w2")):
        return expert.w1, expert.w3, expert.w2
    raise RuntimeError("unsupported expert layout")


def clustered_requests(num_experts: int, set_size: int, count: int, seed: int) -> list[Request]:
    if count % 4 or 4 * set_size > num_experts * 2:
        raise ValueError("invalid clustered request grid")
    rng = random.Random(seed)
    bases = []
    stride = max(1, num_experts // 4)
    for cluster in range(4):
        bases.append(frozenset((cluster * stride + j) % num_experts for j in range(set_size)))
    requests = []
    for request_id in range(count):
        base = set(bases[request_id % 4])
        # Replace one expert deterministically to avoid an all-identical toy workload.
        removed = sorted(base)[request_id % set_size]
        base.remove(removed)
        candidates = [expert for expert in range(num_experts) if expert not in base]
        base.add(candidates[rng.randrange(len(candidates))])
        requests.append(Request(request_id, frozenset(base)))
    return requests


def uniform_requests(num_experts: int, set_size: int, count: int, seed: int) -> list[Request]:
    rng = random.Random(seed)
    return [
        Request(request_id, frozenset(rng.sample(range(num_experts), set_size)))
        for request_id in range(count)
    ]


def request_inputs(requests, hidden: int, rows: int, seed: int):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return {
        (request.request_id, expert): torch.randn(
            rows, hidden, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        for request in requests for expert in sorted(request.experts)
    }


def build_eager_pair(pair, experts, inputs):
    a, b = pair
    grouped = {}
    for request in pair:
        for expert in request.experts:
            grouped.setdefault(expert, []).append(inputs[(request.request_id, expert)])
    packed = {expert: torch.cat(parts, dim=0) for expert, parts in grouped.items()}

    def operation():
        return [experts[expert](packed[expert]) for expert in sorted(packed)]
    return operation


def build_bmm_pair(pair, experts, inputs):
    a, b = pair
    grouped = {}
    for request in pair:
        for expert in request.experts:
            grouped.setdefault(expert, []).append(inputs[(request.request_id, expert)])
    expert_ids = sorted(grouped)
    counts = [sum(part.shape[0] for part in grouped[expert]) for expert in expert_ids]
    maximum = max(counts)
    hidden = next(iter(inputs.values())).shape[1]
    x = torch.zeros(len(expert_ids), maximum, hidden, device="cuda", dtype=torch.bfloat16)
    for index, expert in enumerate(expert_ids):
        value = torch.cat(grouped[expert], dim=0)
        x[index, : value.shape[0]] = value
    triplets = [projections(experts[expert]) for expert in expert_ids]
    gate = torch.stack([item[0].weight.T for item in triplets])
    up = torch.stack([item[1].weight.T for item in triplets])
    down = torch.stack([item[2].weight.T for item in triplets])

    def operation():
        intermediate = torch.nn.functional.silu(torch.bmm(x, gate)) * torch.bmm(x, up)
        return torch.bmm(intermediate, down)
    return operation, counts


def cuda_time(operation, inner: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(inner):
        operation()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / inner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=SPECS, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--rows-per-request-expert", type=int, default=2)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--inner", type=int, default=10)
    parser.add_argument("--workload", choices=("clustered", "uniform"), default="clustered")
    parser.add_argument("--workload-seed", type=int, default=20260723)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite output")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5090":
        raise RuntimeError("wrong GPU")
    spec = SPECS[args.model_key]
    model = AutoModelForCausalLM.from_pretrained(
        spec["path"], local_files_only=True, torch_dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()
    experts = experts_at(model, args.layer)
    if len(experts) != spec["experts"]:
        raise RuntimeError("expert count mismatch")
    hidden = projections(experts[0])[0].in_features
    request_builder = clustered_requests if args.workload == "clustered" else uniform_requests
    requests = request_builder(
        spec["experts"], spec["set_size"], args.requests, args.workload_seed
    )
    inputs = request_inputs(requests, hidden, args.rows_per_request_expert, 20260724)
    policies = {
        "max_overlap": pair_greedy(requests, maximize=True),
        "random": pair_random(requests, 20260725),
        "min_overlap": pair_greedy(requests, maximize=False),
    }
    cpu_us = {}
    for name in policies:
        samples = []
        for _ in range(200):
            start = time.perf_counter_ns()
            if name == "random": pair_random(requests, 20260725)
            else: pair_greedy(requests, maximize=name == "max_overlap")
            samples.append((time.perf_counter_ns() - start) / 1000.0)
        cpu_us[name] = statistics.median(samples)
    eager = {name: [build_eager_pair(pair, experts, inputs) for pair in pairs] for name, pairs in policies.items()}
    fused = {}
    counts = {}
    for name, pairs in policies.items():
        built = [build_bmm_pair(pair, experts, inputs) for pair in pairs]
        fused[name] = [item[0] for item in built]
        counts[name] = [item[1] for item in built]
    closure = {}
    with torch.inference_mode():
        for name in policies:
            reference = eager[name][0]()
            candidate = fused[name][0]()
            if len(reference) != candidate.shape[0]:
                raise RuntimeError("eager/BMM expert axis mismatch")
            deltas = []
            relatives = []
            for index, expected in enumerate(reference):
                actual = candidate[index, : counts[name][0][index]]
                delta = (expected.float() - actual.float()).abs().max().item()
                scale = max(expected.float().abs().max().item(), 1e-12)
                deltas.append(delta)
                relatives.append(delta / scale)
            closure[name] = {"max_abs": max(deltas), "max_rel": max(relatives)}
            if closure[name]["max_abs"] > 0.08 or closure[name]["max_rel"] > 0.05:
                raise RuntimeError(f"BMM surrogate closure failed: {name} {closure[name]}")
    def whole(operations):
        def operation():
            return [op() for op in operations]
        return operation
    operations = {
        backend: {name: whole(items) for name, items in table.items()}
        for backend, table in (("eager", eager), ("bmm_surrogate", fused))
    }
    with torch.inference_mode():
        for table in operations.values():
            for operation in table.values():
                for _ in range(3): operation()
        torch.cuda.synchronize()
        raw = []
        rng = random.Random(20260726)
        arms = [(backend, policy) for backend in operations for policy in policies]
        for trial in range(args.trials):
            rng.shuffle(arms)
            for backend, policy in arms:
                raw.append({
                    "trial": trial, "backend": backend, "policy": policy,
                    "latency_ms": cuda_time(operations[backend][policy], args.inner),
                })
    summary = {}
    for backend in operations:
        summary[backend] = {}
        for policy in policies:
            values = [row["latency_ms"] for row in raw if row["backend"] == backend and row["policy"] == policy]
            summary[backend][policy] = {
                "median_ms": statistics.median(values),
                "union_invocations": union_invocations(policies[policy]),
                "scheduler_cpu_us": cpu_us[policy],
            }
        baseline = summary[backend]["random"]["median_ms"]
        summary[backend]["max_overlap"]["speedup_vs_random"] = baseline / summary[backend]["max_overlap"]["median_ms"]
    source = hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_name("route_pairing.py").read_bytes()).hexdigest()
    payload = {
        "model": args.model_key, "layer": args.layer, "gpu": torch.cuda.get_device_name(0),
        "config": vars(args) | {"output": str(args.output)}, "summary": summary,
        "bmm_closure": closure,
        "raw": raw, "source_sha256": source,
        "evidence_boundary": "controlled expert-set pairing; eager real experts plus optimistic padded-bmm surrogate; not serving or production fused kernel",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

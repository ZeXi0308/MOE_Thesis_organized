"""Trace-replay go/no-go experiments for three alternative MoE systems ideas.

The experiments use captured OLMoE routing traces and analytical service models.
They are intended to reject weak ideas early; they do not replace multi-GPU
DeepEP/SGLang measurements.

Experiments:
  1. DeadlineEP: request/phase-aware asynchronous EP ingress scheduling.
  2. HedgeEP: selective duplicate expert execution under sender vs receiver tails.
  3. Complementary batching: history-predicted routing-complementary batch packing.
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
import heapq
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument(
        "--routes",
        default="experiments/idea_a_mac/outputs/paper_validation/olmoe_signal_mxfp4_n32/test_routes.csv",
    )
    p.add_argument("--ep-size", type=int, default=8)
    p.add_argument("--gpus-per-node", type=int, default=4)
    p.add_argument("--trials", type=int, default=300)
    p.add_argument("--seed", type=int, default=20260713)
    p.add_argument("--offline", action="store_true")
    p.add_argument(
        "--output-dir",
        default="experiments/idea_a_mac/outputs/paper_validation/alternative_ideas_2026-07-13",
    )
    return p.parse_args()


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in df[columns].iterrows():
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def load_trace(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, int]]:
    routes = pd.read_csv(args.routes)
    required = {"sample_id", "layer", "token_position", "rank", "expert_id"}
    missing = required - set(routes.columns)
    if missing:
        raise ValueError(f"route trace missing columns: {sorted(missing)}")
    cfg = AutoConfig.from_pretrained(args.model, local_files_only=args.offline)
    num_experts = int(getattr(cfg, "num_experts", getattr(cfg, "num_local_experts", 0)))
    top_k = int(getattr(cfg, "num_experts_per_tok", getattr(cfg, "num_experts_per_token", 0)))
    meta = {
        "hidden_size": int(cfg.hidden_size),
        "num_layers": int(routes["layer"].max()) + 1,
        "num_experts": num_experts,
        "top_k": top_k,
    }
    routes = routes.copy()
    routes["owner_rank"] = np.minimum(
        routes["expert_id"].astype(int) * args.ep_size // num_experts,
        args.ep_size - 1,
    ).astype(int)
    return routes, meta


def route_tensor(routes: pd.DataFrame, meta: dict[str, int], ep_size: int) -> dict[tuple[int, int], np.ndarray]:
    tensors: dict[tuple[int, int], np.ndarray] = {}
    grouped = routes.groupby(["sample_id", "token_position", "layer", "owner_rank"]).size()
    for (sample_id, token_position), values in grouped.groupby(level=[0, 1]):
        matrix = np.zeros((meta["num_layers"], ep_size), dtype=np.float64)
        for (_, _, layer, owner), count in values.items():
            matrix[int(layer), int(owner)] = float(count)
        tensors[(int(sample_id), int(token_position))] = matrix
    return tensors


def available_positions(tensors: dict[tuple[int, int], np.ndarray]) -> dict[int, list[int]]:
    positions: dict[int, list[int]] = {}
    for sample_id, pos in tensors:
        positions.setdefault(sample_id, []).append(pos)
    return {sample: sorted(values) for sample, values in positions.items()}


# ---------------------------------------------------------------------------
# DeadlineEP


@dataclass
class DeadlineRequest:
    request_id: int
    phase: str
    receiver: int
    deadline_us: float
    services_us: np.ndarray


def request_services(
    matrices: list[np.ndarray],
    receiver: int,
    hidden_size: int,
    gpus_per_node: int,
    inter_node_gbps: float = 200.0,
    intra_node_gbps: float = 900.0,
    message_overhead_us: float = 0.35,
) -> np.ndarray:
    traffic = np.sum(np.stack(matrices), axis=0)
    services = []
    inter_bytes_per_us = inter_node_gbps * 1e9 / 8 / 1e6
    intra_bytes_per_us = intra_node_gbps * 1e9 / 8 / 1e6
    for layer_load in traffic:
        owners = np.flatnonzero(layer_load)
        local = sum(layer_load[o] for o in owners if o // gpus_per_node == receiver // gpus_per_node)
        remote = sum(layer_load[o] for o in owners if o // gpus_per_node != receiver // gpus_per_node)
        transfer = max(local * hidden_size / intra_bytes_per_us, remote * hidden_size / inter_bytes_per_us)
        services.append(float(transfer + len(owners) * message_overhead_us))
    return np.asarray(services, dtype=np.float64)


def build_deadline_workload(
    rng: np.random.Generator,
    tensors: dict[tuple[int, int], np.ndarray],
    positions: dict[int, list[int]],
    hidden_size: int,
    gpus_per_node: int,
    ep_size: int,
    origin_mode: str,
    num_decode: int = 24,
    num_prefill: int = 8,
    prefill_chunk: int = 16,
) -> list[DeadlineRequest]:
    samples = sorted(positions)
    requests: list[DeadlineRequest] = []
    total = num_decode + num_prefill
    for request_id in range(total):
        phase = "decode" if request_id < num_decode else "prefill"
        sample = int(rng.choice(samples))
        sample_positions = positions[sample]
        if phase == "decode":
            pos = int(rng.choice(sample_positions))
            matrices = [tensors[(sample, pos)]]
            deadline = float(rng.uniform(180.0, 320.0))
        else:
            start_idx = int(rng.integers(0, max(1, len(sample_positions) - prefill_chunk + 1)))
            chosen = sample_positions[start_idx : start_idx + prefill_chunk]
            matrices = [tensors[(sample, pos)] for pos in chosen]
            deadline = float(rng.uniform(1800.0, 3200.0))
        if origin_mode == "balanced":
            receiver = request_id % ep_size
        elif origin_mode == "hotspot":
            receiver = 0 if request_id < math.ceil(total * 0.5) else 1 + request_id % max(ep_size - 1, 1)
        else:
            raise ValueError(origin_mode)
        services = request_services(matrices, receiver, hidden_size, gpus_per_node)
        requests.append(DeadlineRequest(request_id, phase, receiver, deadline, services))
    rng.shuffle(requests)
    return requests


def simulate_bsp(requests: list[DeadlineRequest]) -> dict[int, float]:
    num_layers = len(requests[0].services_us)
    total = 0.0
    for layer in range(num_layers):
        by_receiver: dict[int, float] = {}
        for request in requests:
            by_receiver[request.receiver] = by_receiver.get(request.receiver, 0.0) + request.services_us[layer]
        total += max(by_receiver.values(), default=0.0)
    return {request.request_id: total for request in requests}


def simulate_async(
    requests: list[DeadlineRequest],
    policy: str,
    scheduler_overhead_us: float,
    fragmentation_penalty: float = 0.10,
) -> dict[int, float]:
    """Per-receiver request/layer scheduler with cross-layer request progress."""
    completion: dict[int, float] = {}
    by_receiver: dict[int, list[DeadlineRequest]] = {}
    for request in requests:
        by_receiver.setdefault(request.receiver, []).append(request)

    for receiver_requests in by_receiver.values():
        now = 0.0
        ready: list[tuple[int, int]] = [(request.request_id, 0) for request in receiver_requests]
        request_map = {request.request_id: request for request in receiver_requests}
        fifo_order = {request.request_id: idx for idx, request in enumerate(receiver_requests)}
        age = {request.request_id: idx for idx, request in enumerate(receiver_requests)}
        sequence = len(ready)
        while ready:
            if policy == "fifo":
                idx = min(range(len(ready)), key=lambda i: age[ready[i][0]])
            elif policy == "decode_priority":
                idx = min(
                    range(len(ready)),
                    key=lambda i: (
                        request_map[ready[i][0]].phase != "decode",
                        age[ready[i][0]],
                    ),
                )
            elif policy == "edf":
                idx = min(
                    range(len(ready)),
                    key=lambda i: (request_map[ready[i][0]].deadline_us, age[ready[i][0]]),
                )
            else:
                raise ValueError(policy)
            request_id, layer = ready.pop(idx)
            request = request_map[request_id]
            service = request.services_us[layer] * (1.0 + fragmentation_penalty) + scheduler_overhead_us
            now += service
            next_layer = layer + 1
            if next_layer == len(request.services_us):
                completion[request_id] = now
            else:
                age[request_id] = sequence
                sequence += 1
                ready.append((request_id, next_layer))
    return completion


def deadline_experiment(
    args: argparse.Namespace,
    tensors: dict[tuple[int, int], np.ndarray],
    positions: dict[int, list[int]],
    meta: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(args.seed)
    rows = []
    for origin_mode in ("balanced", "hotspot"):
        for trial in range(args.trials):
            workload = build_deadline_workload(
                rng, tensors, positions, meta["hidden_size"], args.gpus_per_node, args.ep_size, origin_mode
            )
            phase = {r.request_id: r.phase for r in workload}
            deadlines = {r.request_id: r.deadline_us for r in workload}
            for overhead in (0.0, 1.0, 3.0, 5.0):
                policy_results = {"bsp": simulate_bsp(workload)}
                for policy in ("fifo", "decode_priority", "edf"):
                    policy_results[policy] = simulate_async(workload, policy, overhead)
                for policy, completion in policy_results.items():
                    decode = np.asarray([v for rid, v in completion.items() if phase[rid] == "decode"])
                    prefill = np.asarray([v for rid, v in completion.items() if phase[rid] == "prefill"])
                    misses = np.mean([v > deadlines[rid] for rid, v in completion.items() if phase[rid] == "decode"])
                    rows.append(
                        {
                            "origin_mode": origin_mode,
                            "trial": trial,
                            "scheduler_overhead_us": overhead,
                            "policy": policy,
                            "decode_p99_us": float(np.quantile(decode, 0.99)),
                            "decode_mean_us": float(decode.mean()),
                            "prefill_p95_us": float(np.quantile(prefill, 0.95)),
                            "makespan_us": float(max(completion.values())),
                            "decode_deadline_miss_rate": float(misses),
                        }
                    )
    raw = pd.DataFrame(rows)
    summary_rows = []
    group_cols = ["origin_mode", "scheduler_overhead_us", "policy"]
    for key, group in raw.groupby(group_cols):
        origin, overhead, policy = key
        bsp = raw[
            (raw.origin_mode == origin)
            & (raw.scheduler_overhead_us == overhead)
            & (raw.policy == "bsp")
        ].set_index("trial")
        aligned = group.set_index("trial")
        reduction = 1.0 - aligned["decode_p99_us"] / bsp["decode_p99_us"]
        makespan_delta = aligned["makespan_us"] / bsp["makespan_us"] - 1.0
        summary_rows.append(
            {
                "origin_mode": origin,
                "scheduler_overhead_us": overhead,
                "policy": policy,
                "median_decode_p99_us": float(group.decode_p99_us.median()),
                "median_decode_p99_reduction_vs_bsp": float(reduction.median()),
                "p10_decode_p99_reduction_vs_bsp": float(reduction.quantile(0.10)),
                "median_makespan_delta_vs_bsp": float(makespan_delta.median()),
                "median_deadline_miss_rate": float(group.decode_deadline_miss_rate.median()),
            }
        )
    return raw, pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# HedgeEP


def sample_snapshot_batch(
    rng: np.random.Generator,
    tensors: dict[tuple[int, int], np.ndarray],
    batch_size: int,
) -> np.ndarray:
    keys = list(tensors)
    chosen = rng.choice(len(keys), size=batch_size, replace=len(keys) < batch_size)
    return np.stack([tensors[keys[int(i)]] for i in chosen])


def hedge_step_latency(
    selection_rng: np.random.Generator,
    tail_rng: np.random.Generator,
    batch: np.ndarray,
    mode: str,
    budget_fraction: float,
    policy: str,
    hedge_delay_us: float = 4.0,
) -> tuple[np.ndarray, float]:
    """Return per-request 16-layer proxy latency and issued-hedge fraction."""
    batch_size, num_layers, ep_size = batch.shape
    request_latency = np.zeros(batch_size, dtype=np.float64)
    issued = 0
    total_calls = 0
    for layer in range(num_layers):
        owner_load = batch[:, layer, :].sum(axis=0)
        max_load = max(owner_load.max(), 1.0)
        call_records = []
        for request_id in range(batch_size):
            for owner in np.flatnonzero(batch[request_id, layer]):
                multiplicity = int(batch[request_id, layer, owner])
                for _ in range(multiplicity):
                    risk = float(owner_load[owner] / max_load)
                    call_records.append((request_id, int(owner), risk))
        total_calls += len(call_records)
        hedge_count = int(round(len(call_records) * budget_fraction))
        if policy == "none":
            hedged: set[int] = set()
        elif policy == "load_selective":
            order = np.argsort([-record[2] for record in call_records])
            hedged = set(int(i) for i in order[:hedge_count])
        elif policy == "random":
            chosen = selection_rng.choice(
                len(call_records), size=min(hedge_count, len(call_records)), replace=False
            )
            hedged = set(int(i) for i in chosen)
        elif policy == "all":
            hedged = set(range(len(call_records)))
        else:
            raise ValueError(policy)
        issued += len(hedged)

        receiver_tail = np.zeros(batch_size)
        if mode in ("receiver_only", "mixed"):
            recv_prob = 0.035 if mode == "receiver_only" else 0.018
            receiver_tail = (tail_rng.random(batch_size) < recv_prob) * tail_rng.lognormal(
                3.2, 0.35, batch_size
            )

        per_request_calls: list[list[float]] = [[] for _ in range(batch_size)]
        for index, (request_id, owner, risk) in enumerate(call_records):
            base = 3.0 + 0.65 * owner_load[owner]
            sender_tail = 0.0
            backup_sender_tail = 0.0
            if mode in ("sender_only", "mixed"):
                base_prob = 0.04 if mode == "sender_only" else 0.022
                probability = min(0.25, base_prob * (0.25 + 1.75 * risk))
                if tail_rng.random() < probability:
                    sender_tail = float(tail_rng.lognormal(3.15, 0.45))
                if tail_rng.random() < base_prob * 0.5:
                    backup_sender_tail = float(tail_rng.lognormal(3.0, 0.40))
            shared_receiver = receiver_tail[request_id]
            primary = base + sender_tail + shared_receiver
            if index in hedged:
                backup = base + backup_sender_tail + shared_receiver
                latency = min(primary, hedge_delay_us + backup)
            else:
                latency = primary
            per_request_calls[request_id].append(latency)
        for request_id, calls in enumerate(per_request_calls):
            if calls:
                request_latency[request_id] += max(calls)
    return request_latency, issued / max(total_calls, 1)


def hedge_experiment(
    args: argparse.Namespace,
    tensors: dict[tuple[int, int], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(args.seed + 1)
    rows = []
    for mode in ("sender_only", "mixed", "receiver_only"):
        for trial in range(args.trials):
            batch = sample_snapshot_batch(rng, tensors, batch_size=32)
            # Reuse identical random draws fairly by recording many requests per policy;
            # stochastic tails still differ, so aggregate over 300 trials.
            for budget in (0.05, 0.10, 0.20):
                policies = ("none", "random", "load_selective", "all")
                for policy in policies:
                    effective_budget = 0.0 if policy == "none" else (1.0 if policy == "all" else budget)
                    common_seed = args.seed + 10_000 * trial + 100 * int(budget * 100) + (
                        0 if mode == "sender_only" else 1 if mode == "mixed" else 2
                    )
                    latency, issued = hedge_step_latency(
                        np.random.default_rng(common_seed + 7),
                        np.random.default_rng(common_seed),
                        batch,
                        mode,
                        effective_budget,
                        policy,
                    )
                    for request_id, value in enumerate(latency):
                        rows.append(
                            {
                                "tail_mode": mode,
                                "trial": trial,
                                "request_id": request_id,
                                "budget_fraction": budget,
                                "policy": policy,
                                "step_latency_us": float(value),
                                "issued_hedge_fraction": issued,
                            }
                        )
    raw = pd.DataFrame(rows)
    summary_rows = []
    for (mode, budget, policy), group in raw.groupby(["tail_mode", "budget_fraction", "policy"]):
        baseline = raw[
            (raw.tail_mode == mode) & (raw.budget_fraction == budget) & (raw.policy == "none")
        ]["step_latency_us"]
        p99 = float(group.step_latency_us.quantile(0.99))
        baseline_p99 = float(baseline.quantile(0.99))
        summary_rows.append(
            {
                "tail_mode": mode,
                "budget_fraction": budget,
                "policy": policy,
                "p99_step_latency_us": p99,
                "p99_reduction_vs_none": 1.0 - p99 / baseline_p99,
                "mean_step_latency_us": float(group.step_latency_us.mean()),
                "mean_issued_hedge_fraction": float(group.issued_hedge_fraction.mean()),
            }
        )
    return raw, pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Complementary routing batching


@dataclass
class RoutingUnit:
    key: tuple[int, int]
    predicted: np.ndarray
    actual: np.ndarray


def routing_units(
    tensors: dict[tuple[int, int], np.ndarray],
    positions: dict[int, list[int]],
    history: int = 4,
) -> list[RoutingUnit]:
    units = []
    for sample, sample_positions in positions.items():
        position_set = set(sample_positions)
        for pos in sample_positions:
            history_positions = list(range(pos - history, pos))
            if pos < history or not all(value in position_set for value in history_positions):
                continue
            predicted = np.mean([tensors[(sample, value)] for value in history_positions], axis=0)
            units.append(RoutingUnit((sample, pos), predicted, tensors[(sample, pos)]))
    return units


def load_objective(matrices: list[np.ndarray]) -> float:
    combined = np.sum(np.stack(matrices), axis=0)
    return float(combined.max(axis=1).sum())


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.ravel(), b.ravel()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denom) if denom > 0 else 0.0


def partition_units(
    units: list[RoutingUnit],
    batch_size: int,
    mode: str,
    rng: np.random.Generator,
) -> list[list[RoutingUnit]]:
    remaining = list(units)
    if mode == "random":
        rng.shuffle(remaining)
        return [remaining[i : i + batch_size] for i in range(0, len(remaining), batch_size)]
    batches = []
    while remaining:
        seed_idx = max(
            range(len(remaining)),
            key=lambda i: load_objective([remaining[i].predicted if mode != "oracle" else remaining[i].actual]),
        )
        batch = [remaining.pop(seed_idx)]
        while remaining and len(batch) < batch_size:
            if mode == "complementary":
                candidate_idx = min(
                    range(len(remaining)),
                    key=lambda i: load_objective([u.predicted for u in batch] + [remaining[i].predicted]),
                )
            elif mode == "oracle":
                candidate_idx = min(
                    range(len(remaining)),
                    key=lambda i: load_objective([u.actual for u in batch] + [remaining[i].actual]),
                )
            elif mode == "affinity":
                current = np.sum([u.predicted for u in batch], axis=0)
                candidate_idx = max(
                    range(len(remaining)), key=lambda i: cosine(current, remaining[i].predicted)
                )
            else:
                raise ValueError(mode)
            batch.append(remaining.pop(candidate_idx))
        batches.append(batch)
    return batches


def complementary_experiment(
    args: argparse.Namespace,
    units: list[RoutingUnit],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(args.seed + 2)
    stability = {
        "num_units": len(units),
        "mean_history_to_current_cosine": float(np.mean([cosine(u.predicted, u.actual) for u in units])),
        "median_history_to_current_cosine": float(np.median([cosine(u.predicted, u.actual) for u in units])),
        "hot_owner_match_rate": float(
            np.mean(
                [
                    np.mean(u.predicted.argmax(axis=1) == u.actual.argmax(axis=1))
                    for u in units
                ]
            )
        ),
    }
    rows = []
    pool_size = min(64, len(units))
    for batch_size in (4, 8, 16):
        usable_pool = pool_size - pool_size % batch_size
        for trial in range(args.trials):
            chosen = rng.choice(len(units), size=usable_pool, replace=False)
            pool = [units[int(i)] for i in chosen]
            for mode in ("random", "affinity", "complementary", "oracle"):
                batches = partition_units(pool, batch_size, mode, rng)
                objectives = [load_objective([u.actual for u in batch]) for batch in batches]
                rows.append(
                    {
                        "batch_size": batch_size,
                        "trial": trial,
                        "mode": mode,
                        "mean_actual_bottleneck": float(np.mean(objectives)),
                        "p95_actual_bottleneck": float(np.quantile(objectives, 0.95)),
                        "max_actual_bottleneck": float(np.max(objectives)),
                    }
                )
    raw = pd.DataFrame(rows)
    summary_rows = []
    for (batch_size, mode), group in raw.groupby(["batch_size", "mode"]):
        random = raw[(raw.batch_size == batch_size) & (raw["mode"] == "random")].set_index("trial")
        aligned = group.set_index("trial")
        mean_gain = 1.0 - aligned.mean_actual_bottleneck / random.mean_actual_bottleneck
        p95_gain = 1.0 - aligned.p95_actual_bottleneck / random.p95_actual_bottleneck
        summary_rows.append(
            {
                "batch_size": batch_size,
                "mode": mode,
                "median_mean_bottleneck_reduction_vs_random": float(mean_gain.median()),
                "p10_mean_bottleneck_reduction_vs_random": float(mean_gain.quantile(0.10)),
                "median_p95_bottleneck_reduction_vs_random": float(p95_gain.median()),
                "median_actual_bottleneck": float(group.mean_actual_bottleneck.median()),
            }
        )
    return raw, pd.DataFrame(summary_rows), stability


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    routes, meta = load_trace(args)
    tensors = route_tensor(routes, meta, args.ep_size)
    positions = available_positions(tensors)
    units = routing_units(tensors, positions)

    print(f"trace units={len(tensors)}, history units={len(units)}, meta={meta}", flush=True)
    print("running DeadlineEP proxy...", flush=True)
    deadline_raw, deadline_summary = deadline_experiment(args, tensors, positions, meta)
    deadline_raw.to_csv(out / "deadline_ep_raw.csv", index=False)
    deadline_summary.to_csv(out / "deadline_ep_summary.csv", index=False)

    print("running HedgeEP proxy...", flush=True)
    hedge_raw, hedge_summary = hedge_experiment(args, tensors)
    hedge_raw.to_csv(out / "hedge_ep_raw.csv", index=False)
    hedge_summary.to_csv(out / "hedge_ep_summary.csv", index=False)

    print("running complementary batching proxy...", flush=True)
    batching_raw, batching_summary, stability = complementary_experiment(args, units)
    batching_raw.to_csv(out / "complementary_batching_raw.csv", index=False)
    batching_summary.to_csv(out / "complementary_batching_summary.csv", index=False)
    (out / "routing_stability.json").write_text(json.dumps(stability, indent=2), encoding="utf-8")

    deadline_head = deadline_summary[
        (deadline_summary.policy.isin(["decode_priority", "edf"]))
        & (deadline_summary.scheduler_overhead_us.isin([1.0, 3.0, 5.0]))
    ]
    hedge_head = hedge_summary[
        (hedge_summary.policy.isin(["load_selective", "random"]))
        & (hedge_summary.budget_fraction.isin([0.10, 0.20]))
    ]
    batching_head = batching_summary[batching_summary["mode"].isin(["affinity", "complementary", "oracle"])]

    report = f"""# 三个替代 MoE 系统 Idea 的代理生死实验（2026-07-13）

## 共同边界

- 输入：`{args.routes}`，OLMoE held-out routing trace；
- EP={args.ep_size}，按 expert id 映射 owner；
- 这些是 trace replay / analytical service models，不是 GPU、DeepEP、RDMA 或真实 P99；
- 目的只是在投入多 GPU 工程前发现机制雷点，正结果不能作为论文性能 claim。

## 1. DeadlineEP

模拟 24 个 decode + 8 个 16-token prefill 请求，16 个 MoE layers；比较 BSP、FIFO async、decode-priority 和 EDF。异步策略额外加入 10% fragmentation penalty，并扫描每 request-layer 0/1/3/5 us 调度开销。

{markdown_table(deadline_head, ["origin_mode", "scheduler_overhead_us", "policy", "median_decode_p99_reduction_vs_bsp", "p10_decode_p99_reduction_vs_bsp", "median_makespan_delta_vs_bsp", "median_deadline_miss_rate"])}

解释边界：收益包含打破跨请求 layer barrier 的效果；这要求异步执行/小粒度通信，可能损害 GroupGEMM batching、CUDA Graph 和 overlap。真实实现若只能重排一个不透明 collective 内部 packet，而不能让请求提前进入下一层，收益会显著缩水。

## 2. HedgeEP

将尾延迟拆成 sender/expert straggler、receiver-shared congestion 和 mixed 三种机制。backup 与 primary 共享 token-origin receiver，因此只能独立规避 sender/expert tail，不能规避 receiver tail。

{markdown_table(hedge_head, ["tail_mode", "budget_fraction", "policy", "p99_reduction_vs_none", "mean_issued_hedge_fraction", "p99_step_latency_us"])}

解释边界：额外执行比例是 issued hedge 的上界；尚未模拟取消延迟、replica HBM、备份流量对主路径的反压。receiver-only 场景若无收益，说明该方向不能与“receiver incast”混成一个故事。

## 3. Complementary Routing Batching

使用前 4 个 token 的 routing owner-load fingerprint 预测当前 token；比较 random、affinity、predicted complementary 和 current-routing oracle 分组。

- history→current mean cosine：`{stability['mean_history_to_current_cosine']:.4f}`；
- median cosine：`{stability['median_history_to_current_cosine']:.4f}`；
- per-layer hottest-owner match：`{stability['hot_owner_match_rate']:.4f}`。

{markdown_table(batching_head, ["batch_size", "mode", "median_mean_bottleneck_reduction_vs_random", "p10_mean_bottleneck_reduction_vs_random", "median_p95_bottleneck_reduction_vs_random", "median_actual_bottleneck"])}

解释边界：trace 来自 teacher-forced WikiText token positions，不是真实 autoregressive continuous-serving trace；若 oracle headroom 都很低，方向直接停止。若 oracle 高但 predicted complementary 低，说明 routing 不可预测，必须停止或引入代价更高的 predictor。
"""
    (out / "三方向代理生死实验报告_2026-07-13.md").write_text(report, encoding="utf-8")
    print(deadline_head.to_string(index=False), flush=True)
    print(hedge_head.to_string(index=False), flush=True)
    print(batching_head.to_string(index=False), flush=True)
    print(stability, flush=True)
    print(f"saved to {out}", flush=True)


if __name__ == "__main__":
    main()

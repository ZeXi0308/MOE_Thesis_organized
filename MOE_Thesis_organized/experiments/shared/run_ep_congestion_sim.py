"""Trace-replay proxy for correct EP combine semantics and multi-MoE congestion.

The input route CSV must contain per-token expert selections. The simulator maps:
  combine sender   = expert_owner_rank
  combine receiver = token_origin_rank

It evaluates repeated MoE layers and multiple concurrent MoE requests/replicas
sharing the same fabric. Results are analytical trace replay, not measured GPU
latency. The critical-port policies are upper-bound/idea probes.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument(
        "--calibration-routes",
        default="experiments/idea_a_mac/outputs/paper_validation/signal_comparison/calibration_routes.csv",
    )
    p.add_argument(
        "--test-routes",
        default="experiments/idea_a_mac/outputs/paper_validation/signal_comparison/test_routes.csv",
    )
    p.add_argument("--ep-size", type=int, default=8)
    p.add_argument("--gpus-per-node", type=int, default=4)
    p.add_argument("--num-jobs", default="1,2,4,8,16")
    p.add_argument("--origin-modes", default="balanced,hotspot")
    p.add_argument("--tail-budget-fraction", type=float, default=0.5)
    p.add_argument("--inter-node-gbps", type=float, default=400.0)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        default="experiments/idea_a_mac/outputs/paper_validation/ep_congestion",
    )
    return p.parse_args()


def dataframe_to_markdown(df: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in df[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def add_placement(
    routes: pd.DataFrame,
    ep_size: int,
    num_experts: int,
    receiver_by_sample: dict[int, int],
) -> pd.DataFrame:
    rows = routes.copy()
    rows["sender_rank"] = np.minimum(
        (rows["expert_id"].astype(int) * ep_size // num_experts), ep_size - 1
    )
    rows["receiver_rank"] = rows["sample_id"].astype(int).map(receiver_by_sample)
    if rows["receiver_rank"].isna().any():
        missing = rows.loc[rows["receiver_rank"].isna(), "sample_id"].unique().tolist()
        raise RuntimeError(f"missing receiver assignment for samples: {missing}")
    rows["receiver_rank"] = rows["receiver_rank"].astype(int)
    return rows


def calibration_scores(
    calibration_routes: pd.DataFrame,
    ep_size: int,
    num_experts: int,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    sample_ids = sorted(int(v) for v in calibration_routes["sample_id"].unique())
    receivers = {sample_id: idx % ep_size for idx, sample_id in enumerate(sample_ids)}
    rows = add_placement(calibration_routes, ep_size, num_experts, receivers)
    sender = rows.groupby(["layer", "sender_rank"]).size().to_dict()
    receiver = rows.groupby(["layer", "receiver_rank"]).size().to_dict()
    return sender, receiver


def concurrent_scenario(
    test_routes: pd.DataFrame,
    num_jobs: int,
    origin_mode: str,
    ep_size: int,
    num_experts: int,
) -> pd.DataFrame:
    sample_ids = sorted(int(v) for v in test_routes["sample_id"].unique())
    jobs = []
    receiver_by_synthetic_sample: dict[int, int] = {}
    for job_id in range(num_jobs):
        source_sample = sample_ids[job_id % len(sample_ids)]
        job_rows = test_routes[test_routes["sample_id"] == source_sample].copy()
        synthetic_sample = job_id
        job_rows["source_sample_id"] = source_sample
        job_rows["sample_id"] = synthetic_sample
        job_rows["job_id"] = job_id
        if origin_mode == "balanced":
            receiver = job_id % ep_size
        elif origin_mode == "hotspot":
            hotspot_jobs = max(1, math.ceil(num_jobs * 0.5))
            receiver = 0 if job_id < hotspot_jobs else 1 + ((job_id - hotspot_jobs) % max(ep_size - 1, 1))
        else:
            raise ValueError(f"unknown origin mode: {origin_mode}")
        receiver_by_synthetic_sample[synthetic_sample] = receiver
        jobs.append(job_rows)
    rows = pd.concat(jobs, ignore_index=True)
    return add_placement(rows, ep_size, num_experts, receiver_by_synthetic_sample)


def score_candidates(
    rows: pd.DataFrame,
    policy: str,
    profile_sender: dict[tuple[int, int], float],
    profile_receiver: dict[tuple[int, int], float],
    rng: np.random.Generator,
) -> pd.Series:
    if policy == "tail_budget_random":
        return pd.Series(rng.random(len(rows)), index=rows.index)

    actual_sender = rows.groupby(["layer", "sender_rank"]).size().to_dict()
    actual_receiver = rows.groupby(["layer", "receiver_rank"]).size().to_dict()
    if policy == "tail_budget_profile_ports":
        return rows.apply(
            lambda r: profile_sender.get((int(r["layer"]), int(r["sender_rank"])), 0.0)
            + profile_receiver.get((int(r["layer"]), int(r["receiver_rank"])), 0.0),
            axis=1,
        )
    if policy == "tail_budget_scheduler_receiver":
        return rows.apply(
            lambda r: actual_receiver.get((int(r["layer"]), int(r["receiver_rank"])), 0.0),
            axis=1,
        )
    raise ValueError(f"unsupported candidate policy: {policy}")


def greedy_critical_port_indices(
    rows: pd.DataFrame,
    tail_mask: pd.Series,
    tail_budget_fraction: float,
    gpus_per_node: int,
) -> list[int]:
    """Greedily spend tail-INT4 budget on current critical ingress/egress ports."""
    selected: list[int] = []
    for layer, layer_rows in rows.groupby("layer"):
        all_layer_candidates = layer_rows[tail_mask.loc[layer_rows.index]]
        budget = int(round(len(all_layer_candidates) * tail_budget_fraction))
        if budget <= 0:
            continue

        remote_mask = (layer_rows["sender_rank"] // gpus_per_node) != (
            layer_rows["receiver_rank"] // gpus_per_node
        )
        remote_rows = layer_rows[remote_mask]
        layer_candidates = all_layer_candidates.loc[all_layer_candidates.index.intersection(remote_rows.index)]

        sender_load = remote_rows.groupby("sender_rank").size().astype(float).to_dict()
        receiver_load = remote_rows.groupby("receiver_rank").size().astype(float).to_dict()
        flow_indices = {
            (int(sender), int(receiver)): list(group.index)
            for (sender, receiver), group in layer_candidates.groupby(["sender_rank", "receiver_rank"])
        }
        while budget > 0 and flow_indices:
            best_flow = max(
                flow_indices,
                key=lambda flow: (
                    max(sender_load.get(flow[0], 0.0), receiver_load.get(flow[1], 0.0)),
                    sender_load.get(flow[0], 0.0) + receiver_load.get(flow[1], 0.0),
                ),
            )
            available = flow_indices[best_flow]
            chunk = min(len(available), budget, max(1, budget // 100))
            chosen = available[:chunk]
            selected.extend(chosen)
            del available[:chunk]
            sender_load[best_flow[0]] = sender_load.get(best_flow[0], 0.0) - 0.5 * chunk
            receiver_load[best_flow[1]] = receiver_load.get(best_flow[1], 0.0) - 0.5 * chunk
            budget -= chunk
            if not available:
                del flow_indices[best_flow]
        if budget > 0:
            local_remaining = all_layer_candidates.index.difference(selected)
            selected.extend(list(local_remaining[:budget]))
    return selected


def assign_bytes(
    scenario: pd.DataFrame,
    policy: str,
    hidden_size: int,
    top_k: int,
    tail_budget_fraction: float,
    profile_sender: dict[tuple[int, int], float],
    profile_receiver: dict[tuple[int, int], float],
    gpus_per_node: int,
    seed: int,
) -> pd.DataFrame:
    rows = scenario.copy()
    rows["bytes_per_element"] = 2.0 if policy == "bf16" else 1.0
    tail_mask = rows["rank"].astype(int) > (top_k - max(1, top_k // 2))
    if policy == "rank_tail_all":
        rows.loc[tail_mask, "bytes_per_element"] = 0.5
    elif policy == "tail_budget_greedy_ports":
        selected = greedy_critical_port_indices(
            rows, tail_mask, tail_budget_fraction, gpus_per_node
        )
        rows.loc[selected, "bytes_per_element"] = 0.5
    elif policy.startswith("tail_budget_"):
        candidates = rows[tail_mask].copy()
        scores = score_candidates(
            rows,
            policy,
            profile_sender,
            profile_receiver,
            np.random.default_rng(seed),
        )
        if policy != "tail_budget_random":
            remote = (
                (rows["sender_rank"] // gpus_per_node)
                != (rows["receiver_rank"] // gpus_per_node)
            )
            scores = scores + remote.astype(float) * (float(scores.max()) + 1.0)
        candidates["priority"] = scores.loc[candidates.index]
        for _, layer_candidates in candidates.groupby("layer"):
            budget = int(round(len(layer_candidates) * tail_budget_fraction))
            selected = layer_candidates.nlargest(max(budget, 0), "priority").index
            rows.loc[selected, "bytes_per_element"] = 0.5
    elif policy not in ("bf16", "uniform_fp8"):
        raise ValueError(f"unknown policy: {policy}")

    rows["payload_bytes"] = rows["bytes_per_element"] * hidden_size
    return rows


def summarize(
    rows: pd.DataFrame,
    policy: str,
    origin_mode: str,
    num_jobs: int,
    ep_size: int,
    gpus_per_node: int,
    inter_node_gbps: float,
) -> dict[str, float | int | str]:
    remote = rows[
        (rows["sender_rank"] // gpus_per_node) != (rows["receiver_rank"] // gpus_per_node)
    ].copy()
    full_total = float((rows.shape[0] * 2.0 * rows.attrs["hidden_size"]))
    total = float(rows["payload_bytes"].sum())
    full_remote = float((remote.shape[0] * 2.0 * rows.attrs["hidden_size"]))
    remote_total = float(remote["payload_bytes"].sum())

    if remote.empty:
        return {
            "policy": policy,
            "origin_mode": origin_mode,
            "num_jobs": num_jobs,
            "payload_saving_vs_bf16": 1.0 - total / max(full_total, 1e-12),
            "remote_wire_saving_vs_bf16": 0.0,
            "remote_wire_bytes": 0.0,
            "sum_layer_bottleneck_us": 0.0,
            "p99_layer_receiver_bytes": 0.0,
            "mean_layer_receiver_imbalance": 1.0,
            "max_layer_receiver_imbalance": 1.0,
        }

    ingress = remote.groupby(["layer", "receiver_rank"])["payload_bytes"].sum()
    egress = remote.groupby(["layer", "sender_rank"])["payload_bytes"].sum()
    ingress_max = ingress.groupby("layer").max()
    egress_max = egress.groupby("layer").max()
    layer_bottleneck = pd.concat([ingress_max, egress_max], axis=1).max(axis=1)
    bw_bytes_per_us = inter_node_gbps * 1e9 / 8 / 1e6
    ingress_mean = ingress.groupby("layer").mean()
    imbalance = ingress_max / ingress_mean.clip(lower=1.0)

    return {
        "policy": policy,
        "origin_mode": origin_mode,
        "num_jobs": num_jobs,
        "payload_saving_vs_bf16": 1.0 - total / max(full_total, 1e-12),
        "remote_wire_saving_vs_bf16": 1.0 - remote_total / max(full_remote, 1e-12),
        "remote_wire_bytes": remote_total,
        "sum_layer_bottleneck_us": float((layer_bottleneck / bw_bytes_per_us).sum()),
        "p99_layer_receiver_bytes": float(ingress.quantile(0.99)),
        "mean_layer_receiver_imbalance": float(imbalance.mean()),
        "max_layer_receiver_imbalance": float(imbalance.max()),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    calibration_routes = pd.read_csv(args.calibration_routes)
    test_routes = pd.read_csv(args.test_routes)
    cfg = AutoConfig.from_pretrained(args.model, local_files_only=args.offline)
    hidden_size = int(cfg.hidden_size)
    num_experts = int(getattr(cfg, "num_experts", getattr(cfg, "num_local_experts", 0)))
    top_k = int(getattr(cfg, "num_experts_per_tok", getattr(cfg, "num_experts_per_token", 0)))
    if num_experts <= 0 or top_k <= 0:
        raise ValueError(f"cannot resolve MoE config: num_experts={num_experts}, top_k={top_k}")
    profile_sender, profile_receiver = calibration_scores(
        calibration_routes, args.ep_size, num_experts
    )

    policies = [
        "bf16",
        "uniform_fp8",
        "rank_tail_all",
        "tail_budget_random",
        "tail_budget_profile_ports",
        "tail_budget_scheduler_receiver",
        "tail_budget_greedy_ports",
    ]
    job_counts = [int(value) for value in args.num_jobs.split(",") if value]
    origin_modes = [value.strip() for value in args.origin_modes.split(",") if value.strip()]
    results: list[dict[str, float | int | str]] = []

    for origin_mode in origin_modes:
        for num_jobs in job_counts:
            scenario = concurrent_scenario(
                test_routes, num_jobs, origin_mode, args.ep_size, num_experts
            )
            for policy in policies:
                policy_rows = assign_bytes(
                    scenario,
                    policy,
                    hidden_size,
                    top_k,
                    args.tail_budget_fraction,
                    profile_sender,
                    profile_receiver,
                    args.gpus_per_node,
                    seed=args.seed + num_jobs,
                )
                policy_rows.attrs["hidden_size"] = hidden_size
                results.append(
                    summarize(
                        policy_rows,
                        policy,
                        origin_mode,
                        num_jobs,
                        args.ep_size,
                        args.gpus_per_node,
                        args.inter_node_gbps,
                    )
                )

    df = pd.DataFrame(results)
    bf16 = df[df["policy"] == "bf16"][
        ["origin_mode", "num_jobs", "sum_layer_bottleneck_us"]
    ].rename(columns={"sum_layer_bottleneck_us": "bf16_bottleneck_us"})
    fp8 = df[df["policy"] == "uniform_fp8"][
        ["origin_mode", "num_jobs", "sum_layer_bottleneck_us"]
    ].rename(columns={"sum_layer_bottleneck_us": "fp8_bottleneck_us"})
    df = df.merge(bf16, on=["origin_mode", "num_jobs"], how="left")
    df = df.merge(fp8, on=["origin_mode", "num_jobs"], how="left")
    df["bottleneck_saving_vs_bf16"] = 1.0 - df["sum_layer_bottleneck_us"] / df[
        "bf16_bottleneck_us"
    ].clip(lower=1e-12)
    df["bottleneck_saving_vs_fp8"] = 1.0 - df["sum_layer_bottleneck_us"] / df[
        "fp8_bottleneck_us"
    ].clip(lower=1e-12)
    df.to_csv(out / "congestion_simulation.csv", index=False)

    config = {
        "model": args.model,
        "ep_size": args.ep_size,
        "gpus_per_node": args.gpus_per_node,
        "hidden_size": hidden_size,
        "num_experts": num_experts,
        "top_k": top_k,
        "tail_budget_fraction_of_safe_tail": args.tail_budget_fraction,
        "inter_node_gbps": args.inter_node_gbps,
        "boundary": "analytical trace replay; not measured GPU latency",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    focus = df[
        (df["num_jobs"] == max(job_counts))
        & df["policy"].isin(
            [
                "uniform_fp8",
                "rank_tail_all",
                "tail_budget_random",
                "tail_budget_profile_ports",
                "tail_budget_scheduler_receiver",
                "tail_budget_greedy_ports",
            ]
        )
    ][
        [
            "origin_mode",
            "policy",
            "payload_saving_vs_bf16",
            "remote_wire_saving_vs_bf16",
            "sum_layer_bottleneck_us",
            "bottleneck_saving_vs_fp8",
            "mean_layer_receiver_imbalance",
        ]
    ]
    table = dataframe_to_markdown(focus, list(focus.columns))
    report = f"""# EP Congestion Trace-Replay Report

## Boundary

This report reconstructs combine traffic as `expert_owner_rank -> token_origin_rank` and models repeated MoE layers plus concurrent MoE requests/replicas. It is an analytical trace replay, not measured GPU latency or queueing.

## Configuration

- model: `{args.model}`
- EP size: `{args.ep_size}`
- GPUs per node: `{args.gpus_per_node}`
- inter-node bandwidth: `{args.inter_node_gbps} Gbps`
- concurrent jobs: `{job_counts}`
- origin modes: `{origin_modes}`
- selective tail budget: `{args.tail_budget_fraction:.2f}` of already-safe tail pairs

## Largest-concurrency comparison

{table}

## Policy interpretation

- `rank_tail_all`: all fixed tail ranks use INT4; this has more payload reduction than selective policies.
- `tail_budget_random/profile/scheduler/greedy`: use the same limited INT4 count and only change where that safe budget lands.
- `profile_ports` is deployable only if calibration transfers.
- `scheduler_receiver` assumes the request scheduler knows active token-owner counts.
- `greedy_ports` uses current-window sender/receiver loads and is a trace-level upper-bound probe, not a deployable online algorithm yet.

The optimization question suggested by this simulation is therefore broader than receiver-only: allocate a safe tail-INT4 budget to the current critical sender/receiver port while preserving a regular rank-segmented kernel.
"""
    (out / "congestion_report.md").write_text(report, encoding="utf-8")
    print(focus.to_string(index=False), flush=True)
    print(f"saved to {out}", flush=True)


if __name__ == "__main__":
    main()

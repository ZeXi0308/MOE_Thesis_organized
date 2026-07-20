"""Compose quality-safe INT4 selectors with topology-aware flow budgeting.

This is an analytical trace replay.  It compares four quality selectors
(global rank, layer-wise rank budget, gate threshold, cumulative tail mass)
and asks where a limited subset of each selector's INT4 budget should land.
It does not model a collective algorithm, queueing, or kernel overhead.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoConfig

from run_ep_congestion_sim import concurrent_scenario, dataframe_to_markdown, summarize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--test-routes", required=True)
    p.add_argument("--signal-calibration", required=True)
    p.add_argument("--signal-results", required=True)
    p.add_argument("--layer-allocations", required=True)
    p.add_argument("--layer-results", required=True)
    p.add_argument("--layer-strategy", default="kl_profile_2_4_6")
    p.add_argument("--ep-size", type=int, default=8)
    p.add_argument("--gpus-per-node", type=int, default=4)
    p.add_argument("--num-jobs", default="1,2,4,8,16")
    p.add_argument("--origin-modes", default="balanced,hotspot")
    p.add_argument("--budget-fractions", default="0.25,0.5,0.75,1.0")
    p.add_argument("--inter-node-gbps", type=float, default=200.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--offline", action="store_true")
    p.add_argument(
        "--output-dir",
        default="experiments/idea_a_mac/outputs/paper_validation/quality_safe_congestion",
    )
    return p.parse_args()


def safe_mask(
    rows: pd.DataFrame,
    selector: str,
    top_k: int,
    gate_threshold: float,
    tail_mass_budget: float,
    layer_counts: list[int],
) -> pd.Series:
    if selector == "fixed_rank":
        return rows["rank"].astype(int) > top_k - max(1, top_k // 2)
    if selector == "layer_rank":
        cutoffs = rows["layer"].astype(int).map(
            {layer: top_k - count for layer, count in enumerate(layer_counts)}
        )
        return rows["rank"].astype(int) > cutoffs
    if selector == "gate_threshold":
        return rows["gate_weight"].astype(float) <= gate_threshold
    if selector == "gate_tailmass":
        keys = ["job_id", "layer", "token_position"]
        ordered = rows.sort_values(keys + ["rank"])
        suffix = ordered.groupby(keys, sort=False)["gate_weight"].transform(
            lambda values: values.iloc[::-1].cumsum().iloc[::-1]
        )
        result = pd.Series(False, index=rows.index)
        result.loc[ordered.index] = (suffix <= tail_mass_budget).to_numpy()
        return result
    raise ValueError(f"unknown selector: {selector}")


def critical_indices(
    rows: pd.DataFrame,
    candidates: pd.Series,
    fraction: float,
    gpus_per_node: int,
) -> list[int]:
    """Greedily spend a fixed per-layer budget on critical remote ports."""
    selected: list[int] = []
    for _, layer_rows in rows.groupby("layer"):
        layer_candidates = layer_rows[candidates.loc[layer_rows.index]]
        budget = int(round(len(layer_candidates) * fraction))
        if budget <= 0:
            continue
        remote = layer_rows[
            (layer_rows["sender_rank"] // gpus_per_node)
            != (layer_rows["receiver_rank"] // gpus_per_node)
        ]
        remote_candidates = layer_candidates.loc[
            layer_candidates.index.intersection(remote.index)
        ]
        sender_load = remote.groupby("sender_rank").size().astype(float).to_dict()
        receiver_load = remote.groupby("receiver_rank").size().astype(float).to_dict()
        flows = {
            (int(sender), int(receiver)): list(group.index)
            for (sender, receiver), group in remote_candidates.groupby(
                ["sender_rank", "receiver_rank"]
            )
        }
        while budget > 0 and flows:
            flow = max(
                flows,
                key=lambda pair: (
                    max(sender_load.get(pair[0], 0.0), receiver_load.get(pair[1], 0.0)),
                    sender_load.get(pair[0], 0.0) + receiver_load.get(pair[1], 0.0),
                ),
            )
            available = flows[flow]
            chunk = min(len(available), budget, max(1, budget // 100))
            chosen = available[:chunk]
            selected.extend(chosen)
            del available[:chunk]
            sender_load[flow[0]] = sender_load.get(flow[0], 0.0) - 0.5 * chunk
            receiver_load[flow[1]] = receiver_load.get(flow[1], 0.0) - 0.5 * chunk
            budget -= chunk
            if not available:
                del flows[flow]
        if budget > 0:
            remaining = layer_candidates.index.difference(selected)
            selected.extend(list(remaining[:budget]))
    return selected


def assign_selector_bytes(
    scenario: pd.DataFrame,
    candidates: pd.Series,
    spending: str,
    fraction: float,
    hidden_size: int,
    gpus_per_node: int,
    seed: int,
) -> tuple[pd.DataFrame, float]:
    rows = scenario.copy()
    rows["bytes_per_element"] = 1.0
    if spending == "all_safe":
        selected = list(rows.index[candidates])
    elif spending == "remote_only":
        remote = (rows["sender_rank"] // gpus_per_node) != (
            rows["receiver_rank"] // gpus_per_node
        )
        selected = list(rows.index[candidates & remote])
    elif spending == "random_budget":
        rng = np.random.default_rng(seed)
        selected = []
        for _, layer_rows in rows.groupby("layer"):
            indices = layer_rows.index[candidates.loc[layer_rows.index]].to_numpy()
            budget = int(round(len(indices) * fraction))
            if budget > 0:
                selected.extend(rng.choice(indices, size=budget, replace=False).tolist())
    elif spending == "critical_budget":
        selected = critical_indices(rows, candidates, fraction, gpus_per_node)
    else:
        raise ValueError(f"unknown spending policy: {spending}")
    rows.loc[selected, "bytes_per_element"] = 0.5
    rows["payload_bytes"] = rows["bytes_per_element"] * hidden_size
    rows.attrs["hidden_size"] = hidden_size
    return rows, len(selected) / max(len(rows), 1)


def quality_lookup(
    signal_results: pd.DataFrame,
    layer_results: pd.DataFrame,
    layer_strategy: str,
) -> dict[str, float]:
    signal = signal_results.set_index("strategy")
    layer = layer_results.set_index("strategy")
    return {
        "fixed_rank": float(signal.loc["rank_tail4_int4", "mean_token_kl"]),
        "layer_rank": float(layer.loc[layer_strategy, "mean_token_kl"]),
        "gate_threshold": float(signal.loc["gate_threshold_int4", "mean_token_kl"]),
        "gate_tailmass": float(signal.loc["gate_tailmass_int4", "mean_token_kl"]),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    routes = pd.read_csv(args.test_routes)
    calibration = json.loads(Path(args.signal_calibration).read_text(encoding="utf-8"))
    allocation_doc = json.loads(Path(args.layer_allocations).read_text(encoding="utf-8"))
    layer_counts = allocation_doc["allocations"][args.layer_strategy]
    signal_results = pd.read_csv(args.signal_results)
    layer_results = pd.read_csv(args.layer_results)
    selector_quality = quality_lookup(signal_results, layer_results, args.layer_strategy)

    cfg = AutoConfig.from_pretrained(args.model, local_files_only=args.offline)
    hidden_size = int(cfg.hidden_size)
    num_experts = int(getattr(cfg, "num_experts", getattr(cfg, "num_local_experts", 0)))
    top_k = int(getattr(cfg, "num_experts_per_tok", getattr(cfg, "num_experts_per_token", 0)))
    gate_threshold = float(calibration["gate_threshold"])
    tail_mass_budget = float(calibration["gate_tailmass"])
    selectors = ["fixed_rank", "layer_rank", "gate_threshold", "gate_tailmass"]
    job_counts = [int(value) for value in args.num_jobs.split(",") if value]
    origin_modes = [value.strip() for value in args.origin_modes.split(",") if value.strip()]
    fractions = [float(value) for value in args.budget_fractions.split(",") if value]

    results: list[dict[str, float | int | str]] = []
    for origin_mode in origin_modes:
        for num_jobs in job_counts:
            scenario = concurrent_scenario(
                routes, num_jobs, origin_mode, args.ep_size, num_experts
            )
            fp8 = scenario.copy()
            fp8["payload_bytes"] = float(hidden_size)
            fp8.attrs["hidden_size"] = hidden_size
            fp8_row = summarize(
                fp8,
                "uniform_fp8",
                origin_mode,
                num_jobs,
                args.ep_size,
                args.gpus_per_node,
                args.inter_node_gbps,
            )
            fp8_row.update(
                {
                    "selector": "none",
                    "spending": "uniform_fp8",
                    "budget_fraction": 0.0,
                    "safe_fraction_of_pairs": 0.0,
                    "int4_fraction_of_pairs": 0.0,
                    "selector_full_mean_token_kl": 0.0,
                }
            )
            results.append(fp8_row)

            for selector in selectors:
                candidates = safe_mask(
                    scenario,
                    selector,
                    top_k,
                    gate_threshold,
                    tail_mass_budget,
                    layer_counts,
                )
                safe_fraction = float(candidates.mean())
                policies = [("all_safe", 1.0), ("remote_only", 1.0)]
                for fraction in fractions:
                    policies.extend(
                        [("random_budget", fraction), ("critical_budget", fraction)]
                    )
                seen: set[tuple[str, float]] = set()
                for spending, fraction in policies:
                    if (spending, fraction) in seen:
                        continue
                    seen.add((spending, fraction))
                    policy_rows, int4_fraction = assign_selector_bytes(
                        scenario,
                        candidates,
                        spending,
                        fraction,
                        hidden_size,
                        args.gpus_per_node,
                        args.seed + num_jobs,
                    )
                    row = summarize(
                        policy_rows,
                        f"{selector}:{spending}:{fraction:.2f}",
                        origin_mode,
                        num_jobs,
                        args.ep_size,
                        args.gpus_per_node,
                        args.inter_node_gbps,
                    )
                    row.update(
                        {
                            "selector": selector,
                            "spending": spending,
                            "budget_fraction": fraction,
                            "safe_fraction_of_pairs": safe_fraction,
                            "int4_fraction_of_pairs": int4_fraction,
                            "selector_full_mean_token_kl": selector_quality[selector],
                        }
                    )
                    results.append(row)

    df = pd.DataFrame(results)
    fp8_ref = df[df["spending"] == "uniform_fp8"][
        ["origin_mode", "num_jobs", "sum_layer_bottleneck_us"]
    ].rename(columns={"sum_layer_bottleneck_us": "fp8_bottleneck_us"})
    df = df.merge(fp8_ref, on=["origin_mode", "num_jobs"], how="left")
    df["bottleneck_saving_vs_uniform_fp8"] = 1.0 - df["sum_layer_bottleneck_us"] / df[
        "fp8_bottleneck_us"
    ].clip(lower=1e-12)
    df.to_csv(out / "quality_safe_congestion_frontier.csv", index=False)

    largest = max(job_counts)
    view = df[
        (df["num_jobs"] == largest)
        & (
            (df["spending"].isin(["uniform_fp8", "all_safe", "remote_only"]))
            | ((df["spending"].isin(["random_budget", "critical_budget"])) & (df["budget_fraction"] == 0.5))
        )
    ].copy()
    columns = [
        "origin_mode",
        "selector",
        "spending",
        "safe_fraction_of_pairs",
        "int4_fraction_of_pairs",
        "payload_saving_vs_bf16",
        "remote_wire_saving_vs_bf16",
        "bottleneck_saving_vs_uniform_fp8",
        "selector_full_mean_token_kl",
    ]
    report = f"""# Quality-Safe Critical-Flow Congestion Frontier

## Boundary

This is bandwidth-only trace replay with correct combine direction
`expert_owner_rank -> token_origin_rank`.  `selector_full_mean_token_kl` is
copied from the corresponding full-selector fake-quant experiment; subset
spending policies were not re-evaluated for quality and must not inherit that
number as a measured KL.

## Setup

- model: `{args.model}`; EP=`{args.ep_size}`; GPUs/node=`{args.gpus_per_node}`
- concurrent jobs: `{job_counts}`; origin modes: `{origin_modes}`
- layer selector: `{args.layer_strategy}` = `{layer_counts}`
- calibrated gate threshold: `{gate_threshold:.6f}`
- calibrated cumulative tail-mass budget: `{tail_mass_budget:.6f}`

## Largest-concurrency view

{dataframe_to_markdown(view, columns)}

## Interpretation

- `all_safe` applies INT4 to every pair admitted by the quality selector.
- `remote_only` applies INT4 only to admitted inter-node pairs.
- `random_budget` and `critical_budget` use the same admitted-pair count per layer.
- `critical_budget` is an online trace-level upper bound, not a deployable scheduler yet.
"""
    (out / "quality_safe_congestion_report.md").write_text(report, encoding="utf-8")
    print(view[columns].to_string(index=False), flush=True)
    print(f"saved to {out}", flush=True)


if __name__ == "__main__":
    main()

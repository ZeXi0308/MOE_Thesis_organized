#!/usr/bin/env python3
"""Trace-only oracle for exact hierarchical MoE combine aggregation.

For each token at each MoE layer, the standard logical combine has one
contribution per selected expert.  Because combine is a weighted sum, outputs
owned by the same GPU or NVLink domain can in principle be accumulated before
crossing the next topology level.  This script reports the resulting copy-count
upper bound.  It does not model kernels, synchronization, links, or latency.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY = ["sample_id", "layer", "token_position"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ep-sizes", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--gpus-per-node", type=int, nargs="+", default=[4, 8])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    routes = pd.read_csv(args.routes)
    required = set(KEY + ["expert_id", "rank"])
    missing = required - set(routes.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    rows: list[dict[str, float | int]] = []
    for ep_size in args.ep_sizes:
        for gpus_per_node in args.gpus_per_node:
            if gpus_per_node > ep_size or ep_size % gpus_per_node:
                continue
            current = routes[KEY + ["expert_id"]].copy()
            # Modular placement is the same simple placement used by the
            # existing congestion proxies.  Real systems must repeat this for
            # their actual expert map.
            current["owner_rank"] = current["expert_id"] % ep_size
            current["owner_node"] = current["owner_rank"] // gpus_per_node
            grouped = current.groupby(KEY, sort=False)
            copies = grouped.size()
            owner_ranks = grouped["owner_rank"].nunique()
            owner_nodes = grouped["owner_node"].nunique()
            rows.append(
                {
                    "ep_size": ep_size,
                    "gpus_per_node": gpus_per_node,
                    "num_nodes": ep_size // gpus_per_node,
                    "token_layers": int(len(copies)),
                    "mean_selected_experts": float(copies.mean()),
                    "mean_distinct_owner_ranks": float(owner_ranks.mean()),
                    "owner_rank_copy_reduction": float(
                        1.0 - owner_ranks.sum() / copies.sum()
                    ),
                    "mean_distinct_owner_nodes": float(owner_nodes.mean()),
                    "owner_node_copy_reduction": float(
                        1.0 - owner_nodes.sum() / copies.sum()
                    ),
                    "p50_distinct_owner_nodes": float(owner_nodes.median()),
                    "p90_distinct_owner_nodes": float(owner_nodes.quantile(0.9)),
                }
            )

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

"""Measure lane-count regularity for threshold and fixed-quota EP layouts.

This is a trace/layout analysis, not a latency benchmark.  It uses the routed
pair order saved by ``run_signal_comparison.py`` and asks how much a calibrated
gate threshold makes the FP8/MXFP4 lane lengths fluctuate inside each
owner-local tile.  The fixed-quota policy has the same dynamic membership, but
an exact low-bit cardinality conditional on the routed-pair count.
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
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--tile-pairs", type=int, default=64)
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument(
        "--group-mapping", choices=("contiguous", "mod"), default="contiguous"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        action="append",
        default=[],
        help="Hidden sizes used only for metadata-overhead estimates.",
    )
    return parser.parse_args()


def _group_ids(
    expert_ids: pd.Series, num_experts: int, num_groups: int, mapping: str
) -> np.ndarray:
    values = expert_ids.to_numpy(dtype=np.int64)
    if mapping == "mod":
        return values % num_groups
    return np.minimum(values * num_groups // num_experts, num_groups - 1)


def _summary(frame: pd.DataFrame, level: str) -> dict[str, float | int | str]:
    fractions = frame["threshold_low_fraction"].to_numpy(dtype=np.float64)
    deviations = frame["threshold_low_count"] - frame["quota_low_count"]
    return {
        "level": level,
        "units": int(len(frame)),
        "pairs": int(frame["pair_count"].sum()),
        "threshold_low_fraction_weighted": float(
            frame["threshold_low_count"].sum() / frame["pair_count"].sum()
        ),
        "threshold_low_fraction_p01": float(np.quantile(fractions, 0.01)),
        "threshold_low_fraction_p05": float(np.quantile(fractions, 0.05)),
        "threshold_low_fraction_p50": float(np.quantile(fractions, 0.50)),
        "threshold_low_fraction_p95": float(np.quantile(fractions, 0.95)),
        "threshold_low_fraction_p99": float(np.quantile(fractions, 0.99)),
        "mean_abs_lane_count_deviation": float(np.abs(deviations).mean()),
        "p95_abs_lane_count_deviation": float(np.quantile(np.abs(deviations), 0.95)),
        "any_lane_overflow_fraction": float(np.mean(deviations != 0)),
        "low_lane_overflow_fraction": float(np.mean(deviations > 0)),
        "high_lane_overflow_fraction": float(np.mean(deviations < 0)),
    }


def _markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.tile_pairs < 1:
        raise ValueError("--tile-pairs must be positive")
    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    calibration = json.loads(
        (run_dir / "calibration.json").read_text(encoding="utf-8")
    )
    routes = pd.read_csv(run_dir / "test_routes.csv")
    num_groups = int(config["num_receiver_groups"])
    inferred_experts = int(routes["expert_id"].max()) + 1
    num_experts = args.num_experts or inferred_experts
    if int(routes["expert_id"].max()) >= num_experts:
        raise ValueError("--num-experts is smaller than an observed expert id")
    threshold = float(calibration["gate_threshold"])
    target = float(calibration["target_low_bit_fraction"])

    routes = routes.sort_values(
        ["sample_id", "layer", "token_position", "rank"], kind="stable"
    ).reset_index(drop=True)
    routes["owner_group"] = _group_ids(
        routes["expert_id"], num_experts, num_groups, args.group_mapping
    )
    routes["threshold_low"] = routes["gate_weight"] <= threshold

    tile_rows: list[dict[str, int | float]] = []
    message_rows: list[dict[str, int | float]] = []
    group_columns = ["sample_id", "layer", "owner_group"]
    for key, message in routes.groupby(group_columns, sort=True):
        low_count = int(message["threshold_low"].sum())
        pair_count = int(len(message))
        quota_low = int(round(pair_count * target))
        message_rows.append(
            {
                **dict(zip(group_columns, key)),
                "pair_count": pair_count,
                "threshold_low_count": low_count,
                "quota_low_count": quota_low,
                "threshold_low_fraction": low_count / pair_count,
            }
        )
        low_values = message["threshold_low"].to_numpy(dtype=np.int64)
        for tile_id, start in enumerate(range(0, pair_count, args.tile_pairs)):
            tile = low_values[start : start + args.tile_pairs]
            tile_count = int(len(tile))
            tile_low = int(tile.sum())
            tile_rows.append(
                {
                    **dict(zip(group_columns, key)),
                    "tile_id": tile_id,
                    "pair_count": tile_count,
                    "threshold_low_count": tile_low,
                    "quota_low_count": int(round(tile_count * target)),
                    "threshold_low_fraction": tile_low / tile_count,
                }
            )

    tiles = pd.DataFrame(tile_rows)
    messages = pd.DataFrame(message_rows)
    summary = pd.DataFrame(
        [_summary(tiles, f"peer_tile_{args.tile_pairs}"), _summary(messages, "peer_message")]
    )
    tiles.to_csv(run_dir / "peer_layout_tiles.csv", index=False)
    messages.to_csv(run_dir / "peer_layout_messages.csv", index=False)
    summary.to_csv(run_dir / "peer_layout_regularity.csv", index=False)

    metadata_rows = []
    for hidden_size in sorted(set(args.hidden_size)):
        if hidden_size < 1:
            raise ValueError("--hidden-size values must be positive")
        # A 1-bit membership mask is close to the information-theoretic lower
        # bound for a 50/50 tile.  A gate scalar is only needed if the EP backend
        # does not already carry it to the owner.
        mask_bytes = math.ceil(args.tile_pairs / 8)
        mixed_payload = args.tile_pairs * hidden_size * (
            target * 0.5 + (1.0 - target) * 1.0
        )
        metadata_rows.append(
            {
                "hidden_size": hidden_size,
                "tile_pairs": args.tile_pairs,
                "membership_mask_bytes": mask_bytes,
                "mask_overhead_fraction_of_payload": mask_bytes / mixed_payload,
                "optional_fp16_gate_bytes": 2 * args.tile_pairs,
                "gate_overhead_fraction_of_payload": 2 * args.tile_pairs / mixed_payload,
            }
        )
    metadata = pd.DataFrame(metadata_rows)
    if not metadata.empty:
        metadata.to_csv(run_dir / "peer_layout_metadata_estimate.csv", index=False)

    report = f"""# Peer-Local Fixed-Quota Layout Regularity

- routing trace: `{run_dir / 'test_routes.csv'}`
- group semantics: one local-origin replay, expert ids mapped to {num_groups} synthetic owner groups by `{args.group_mapping}`
- expert count used for mapping: {num_experts} (observed max + 1: {inferred_experts})
- calibrated gate threshold: {threshold:.8f}
- target low-bit fraction: {target:.4f}
- fixed-quota tile: {args.tile_pairs} routed pairs

## Lane-count variability

{_markdown(summary)}

`any_lane_overflow_fraction` is the fraction of units whose threshold-selected
FP8/low-bit counts do not fit a preallocated exact-{target:.0%} two-lane split.
It is a layout regularity statistic, not a measured allocation or latency cost:
a threshold implementation can recover exact sizes with scans/count exchange.

## Metadata estimate

{_markdown(metadata) if not metadata.empty else '(no hidden size supplied)'}

The fixed quota still needs a membership mask because membership is dynamic; it
only removes variable lane cardinality.  Contribution selection may also need a
gate scalar at the expert owner if the dispatch protocol does not already carry
one.  Quantization scales, headers, alignment, scans, pack/unpack, and collective
startup are outside this estimate and require a real EP kernel benchmark.

## Evidence boundary

This trace is sufficient to test fixed-cardinality layout invariants and gate
lane-count variability.  It does not contain real token-origin ranks, queues,
RDMA/NVLink traffic, or timestamps, so it cannot establish TTFT, TPOT, TBT, P99,
or topology-aware benefit.
"""
    (run_dir / "peer_layout_regularity_report.md").write_text(
        report, encoding="utf-8"
    )
    print(report)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Offline replay: apply codec hard-gate to existing receiver *_steps.csv traces.

Recomputes optimistic / serialized_tiles net savings with measured pack+unpack(+h2d)
from Phase A lookup, then simulates blocking any step whose wire saving does not
cover the selected codec tax.

Accounting (corrected 2026-07-21):
  - Default codec mode is ``homo_int4`` (online low action is FP8→INT4).
  - ``once_per_step``: pay measured unit once (primary verdict; matches fused kernel).
  - ``serialized_tiles``: ``tiles * unit * (tile_rows / measured_rows)`` so a
    128-row measurement is not multiplied raw by thousands of 32-row tiles.

Does not re-run the model; action traces are taken as recorded (pre-gate).
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
from pathlib import Path

import numpy as np
import pandas as pd


def load_codec_unit_us(
    lookup_csv: Path,
    *,
    mode: str,
    hidden: int,
    preferred_rows: int,
) -> dict[str, float]:
    table = pd.read_csv(lookup_csv)
    subset = table[(table["mode"] == mode) & (table["hidden"] == hidden)]
    if subset.empty:
        raise ValueError(f"no codec rows for mode={mode} hidden={hidden} in {lookup_csv}")
    subset = subset.assign(row_dist=(subset["rows"] - preferred_rows).abs())
    row = subset.sort_values(["row_dist", "rows"]).iloc[0]
    return {
        "pack_us": float(row["pack_us"]),
        "unpack_us": float(row["unpack_us"]),
        "h2d_us": float(row["h2d_us"]),
        "lookup_rows": int(row["rows"]),
        "lookup_hidden": int(row["hidden"]),
        "mode": mode,
    }


def codec_costs(
    steps: pd.DataFrame,
    unit_us: float,
    tile_rows: int,
    measured_rows: int,
) -> pd.DataFrame:
    out = steps.copy()
    low = out["low_pairs"].astype(float).to_numpy()
    tiles = (
        out["codec_tiles"].astype(float).to_numpy()
        if "codec_tiles" in out.columns
        else np.zeros(len(out))
    )
    inferred = np.ceil(np.maximum(low, 0.0) / max(tile_rows, 1))
    tiles = np.where((tiles <= 0) & (low > 0), inferred, tiles)
    scale = float(tile_rows) / float(max(measured_rows, 1))
    optimistic = np.where(low > 0, unit_us, 0.0)
    # Corrected serialized: scale measured unit down to tile width.
    serialized = tiles * unit_us * scale
    # Deprecated raw multiply kept for audit only.
    serialized_raw_bug = tiles * unit_us
    out["replay_codec_tiles"] = tiles
    out["replay_codec_optimistic_us"] = optimistic
    out["replay_codec_serialized_us"] = serialized
    out["replay_codec_serialized_raw_unscaled_us"] = np.where(low > 0, serialized_raw_bug, 0.0)
    out["replay_tile_scale"] = scale
    baseline = out["baseline_step_us"].astype(float)
    bytes_per_us = out["baseline_bottleneck_bytes"].astype(float) / baseline.replace(0, np.nan)
    policy_wire_us = out["policy_bottleneck_bytes"].astype(float) / bytes_per_us
    policy_wire_us = policy_wire_us.fillna(0.0)
    out["replay_policy_wire_us"] = policy_wire_us
    out["replay_wire_saving_us"] = baseline - policy_wire_us
    out["replay_net_optimistic_us"] = out["replay_wire_saving_us"] - optimistic
    out["replay_net_serialized_us"] = out["replay_wire_saving_us"] - serialized
    return out


def apply_hard_gate(steps: pd.DataFrame, tax_mode: str) -> pd.DataFrame:
    out = steps.copy()
    if tax_mode == "once_per_step":
        tax = out["replay_codec_optimistic_us"]
        net = out["replay_net_optimistic_us"]
    else:
        tax = out["replay_codec_serialized_us"]
        net = out["replay_net_serialized_us"]
    would_act = out["low_pairs"].astype(float) > 0
    blocked = would_act & (out["replay_wire_saving_us"] <= tax)
    out["hardgate_blocked"] = blocked.astype(int)
    out["hardgate_net_us"] = np.where(blocked, 0.0, net)
    out["hardgate_low_pairs"] = np.where(blocked, 0.0, out["low_pairs"].astype(float))
    return out


def summarize_file(
    path: Path,
    unit: dict[str, float],
    tile_rows: int,
) -> list[dict]:
    steps = pd.read_csv(path)
    required = {
        "low_pairs",
        "baseline_step_us",
        "baseline_bottleneck_bytes",
        "policy_bottleneck_bytes",
    }
    missing = required - set(steps.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    unit_us = unit["pack_us"] + unit["unpack_us"] + unit["h2d_us"]
    measured_rows = int(unit["lookup_rows"])
    priced = codec_costs(steps, unit_us, tile_rows, measured_rows)
    rows = []
    for tax_mode in ("once_per_step", "serialized_tiles"):
        gated = apply_hard_gate(priced, tax_mode)
        would = int((priced["low_pairs"].astype(float) > 0).sum())
        blocked = int(gated["hardgate_blocked"].sum())
        unblocked = gated[gated["hardgate_blocked"] == 0]
        unblocked_acted = unblocked[unblocked["low_pairs"].astype(float) > 0]
        if tax_mode == "once_per_step":
            orig_net = priced["replay_net_optimistic_us"]
        else:
            orig_net = priced["replay_net_serialized_us"]
        rows.append(
            {
                "steps_csv": str(path),
                "arm": path.name.replace("_steps.csv", ""),
                "tax_mode": tax_mode,
                "n_steps": len(priced),
                "n_steps_with_low": would,
                "n_blocked": blocked,
                "block_rate_among_low": (blocked / would) if would else 0.0,
                "orig_net_mean_us": float(orig_net.mean()),
                "orig_net_p50_us": float(orig_net.median()),
                "orig_net_p95_us": float(orig_net.quantile(0.95)),
                "hardgate_net_mean_us": float(gated["hardgate_net_us"].mean()),
                "hardgate_net_p50_us": float(gated["hardgate_net_us"].median()),
                "unblocked_acted_net_median_us": (
                    float(unblocked_acted["hardgate_net_us"].median())
                    if len(unblocked_acted)
                    else float("nan")
                ),
                "codec_unit_us": unit_us,
                "tile_scale": float(tile_rows) / float(measured_rows),
                "lookup_rows": unit["lookup_rows"],
                "lookup_hidden": unit["lookup_hidden"],
                "lookup_mode": unit["mode"],
            }
        )
    return rows


def decide_verdict(summary: pd.DataFrame) -> dict:
    """Primary verdict uses once_per_step; serialized is sensitivity."""
    opt = summary[summary["tax_mode"] == "once_per_step"]
    ser = summary[summary["tax_mode"] == "serialized_tiles"]
    acted_opt = opt[opt["n_steps_with_low"] > 0]
    acted_ser = ser[ser["n_steps_with_low"] > 0]
    mean_block_opt = float(opt["block_rate_among_low"].mean()) if len(opt) else 0.0
    mean_block_ser = float(ser["block_rate_among_low"].mean()) if len(ser) else 0.0
    med_net_opt = (
        float(acted_opt["orig_net_p50_us"].median()) if len(acted_opt) else float("nan")
    )
    med_net_ser = (
        float(acted_ser["orig_net_p50_us"].median()) if len(acted_ser) else float("nan")
    )
    unblocked_nonpos_opt = bool(
        len(acted_opt)
        and (acted_opt["unblocked_acted_net_median_us"].fillna(0.0) <= 0).all()
    )

    if mean_block_opt > 0.8 and unblocked_nonpos_opt:
        conclusion = (
            "NO_SYSTEM_NET_GAIN_FUSED: >80% low actions blocked under once_per_step "
            "and remaining acted nets non-positive"
        )
    elif mean_block_opt < 0.2 and med_net_opt > 0:
        conclusion = (
            "POSITIVE_NET_REGION_FUSED: once_per_step leaves stable positive net; "
            "serialized_tiles is a pessimistic sensitivity only"
        )
    elif mean_block_opt > 0.8:
        conclusion = "MOSTLY_BLOCKED_FUSED: inspect remaining cells"
    else:
        conclusion = (
            "MIXED_FUSED: see per-file table; do not expand online controller yet"
        )

    return {
        "mean_block_rate_once_per_step": mean_block_opt,
        "mean_block_rate_serialized_scaled": mean_block_ser,
        "median_orig_net_p50_once_per_step": med_net_opt,
        "median_orig_net_p50_serialized_scaled": med_net_ser,
        "unblocked_nonpositive_once_per_step": unblocked_nonpos_opt,
        "conclusion": conclusion,
        # Keep old key name pointing at primary metric for downstream readers.
        "mean_block_rate_serialized": mean_block_ser,
        "unblocked_nonpositive_serialized": bool(
            len(acted_ser)
            and (acted_ser["unblocked_acted_net_median_us"].fillna(0.0) <= 0).all()
        ),
    }


def write_report(output: Path, summary: pd.DataFrame, verdict: dict, metadata: dict) -> None:
    lines = [
        "# Receiver Codec Hard-Gate Offline Replay (corrected accounting)",
        "",
        "## Evidence boundary",
        "",
        metadata["evidence_boundary"],
        "",
        "## Verdict (primary = once_per_step / fused kernel)",
        "",
        f"- block_rate_among_low (once_per_step): "
        f"**{verdict['mean_block_rate_once_per_step']:.1%}**",
        f"- median orig_net_p50 (once_per_step): "
        f"**{verdict['median_orig_net_p50_once_per_step']:.3f} µs**",
        f"- block_rate_among_low (serialized_tiles, scaled): "
        f"**{verdict['mean_block_rate_serialized_scaled']:.1%}**",
        f"- conclusion: **{verdict['conclusion']}**",
        "",
        "## Per-file summary",
        "",
        "| arm | tax_mode | n_low | blocked | block_rate | orig_net_p50 | "
        "hardgate_net_p50 | unblocked_acted_net_med |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['arm']} | {row['tax_mode']} | {int(row['n_steps_with_low'])} | "
            f"{int(row['n_blocked'])} | {row['block_rate_among_low']:.1%} | "
            f"{row['orig_net_p50_us']:.3f} | {row['hardgate_net_p50_us']:.3f} | "
            f"{row['unblocked_acted_net_median_us']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Accounting note",
            "",
            "- Online policy baseline is FP8 high; low action is INT4 → lookup "
            "`homo_int4` by default.",
            "- `serialized_tiles` uses `tiles * unit * (tile_rows / measured_rows)`.",
            "- Primary go/no-go uses `once_per_step`; scaled serialized is sensitivity.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps-glob", nargs="+", required=True)
    parser.add_argument("--codec-lookup-csv", required=True)
    parser.add_argument(
        "--codec-mode",
        default="homo_int4",
        choices=("homo_fp8", "homo_int4"),
        help="Online low action is INT4; default homo_int4",
    )
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--preferred-rows", type=int, default=128)
    parser.add_argument("--tile-rows", type=int, default=32)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    unit = load_codec_unit_us(
        Path(args.codec_lookup_csv),
        mode=args.codec_mode,
        hidden=args.hidden,
        preferred_rows=args.preferred_rows,
    )

    summaries: list[dict] = []
    for path_str in args.steps_glob:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(path)
        summaries.extend(summarize_file(path, unit, args.tile_rows))

    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary.csv", index=False)
    verdict = decide_verdict(summary)
    metadata = {
        "codec_unit": unit,
        "tile_rows": args.tile_rows,
        "steps_files": args.steps_glob,
        "verdict": verdict,
        "accounting": (
            "serialized_us = tiles * unit * (tile_rows / measured_rows); "
            "primary verdict = once_per_step; codec-mode default homo_int4"
        ),
        "evidence_boundary": (
            "offline replay of recorded policy action traces with Phase-A measured "
            "pack+unpack+h2d tax; analytic wire times from baseline/policy bottleneck "
            "bytes (FP8 high → INT4 low); not real NCCL/RDMA latency; H2D is a "
            "pessimistic host-staging bound"
        ),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(output, summary, verdict, metadata)
    print(summary.to_string(index=False))
    print(f"\nconclusion={verdict['conclusion']}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

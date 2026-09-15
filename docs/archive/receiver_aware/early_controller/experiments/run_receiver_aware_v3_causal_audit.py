#!/usr/bin/env python3
"""Causal audit of the Receiver-Aware v3 regime controller.

The original v3 detector observed an initial warm-up fraction but then applied
the selected policy retroactively to the entire scenario. It also constructed
the static profile separately inside each ground-truth origin regime.

This audit fixes both issues:

1. Warm-up steps always use random selection. The detected policy applies only
   after the detection cutoff.
2. A single structural static profile, calibrated from hotspot calibration
   traffic, is used in every test regime. A balanced test misclassified as
   structural therefore pays the real misclassification cost.
3. Detection fractions are swept in one run.
4. Results include whole-scenario saving, post-warm-up saving, detection delay,
   and a ground-truth regime-switch oracle upper bound.

This remains a bandwidth-only trace replay, not an RDMA latency experiment.
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
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoConfig

from run_receiver_aware_v2_systematic import (
    build_scenario,
    load_model_config,
    placement_map,
    remote_loads_by_step,
    select_indices,
    step_us_from_counts,
)
from run_receiver_aware_v3_adaptive import receiver_concentration


def build_static_profile(
    routes: pd.DataFrame,
    calib_docs: list[int],
    num_experts: int,
    ep_size: int,
    placement: str,
    gpus_per_node: int,
) -> dict[str, pd.Series]:
    arrivals = np.zeros(len(calib_docs), dtype=int)
    scenario = build_scenario(routes, calib_docs, arrivals, "hotspot", ep_size)
    scenario["sender_rank"] = placement_map(
        scenario["expert_id"].to_numpy(), num_experts, ep_size, placement
    )
    remote = scenario[
        (scenario["sender_rank"] // gpus_per_node)
        != (scenario["receiver_rank"] // gpus_per_node)
    ]
    n_steps = max(scenario["g"].nunique(), 1)
    return {
        "sender": remote.groupby("sender_rank").size() / n_steps,
        "receiver": remote.groupby("receiver_rank").size() / n_steps,
    }


def calibrate_detector(
    routes: pd.DataFrame,
    calib_docs: list[int],
    num_experts: int,
    ep_size: int,
    placement: str,
    gpus_per_node: int,
) -> tuple[float, dict[str, float]]:
    levels: dict[str, float] = {}
    for origin_mode in ("hotspot", "balanced"):
        arrivals = np.zeros(len(calib_docs), dtype=int)
        scenario = build_scenario(routes, calib_docs, arrivals, origin_mode, ep_size)
        scenario["sender_rank"] = placement_map(
            scenario["expert_id"].to_numpy(), num_experts, ep_size, placement
        )
        remote = scenario[
            (scenario["sender_rank"] // gpus_per_node)
            != (scenario["receiver_rank"] // gpus_per_node)
        ]
        g_max = int(remote["g"].max()) if not remote.empty else 0
        levels[origin_mode] = receiver_concentration(remote, g_max, ep_size)
    return 0.5 * (levels["hotspot"] + levels["balanced"]), levels


def select_subset(
    subset: pd.DataFrame,
    loads_by_step: dict,
    static_profile: dict[str, pd.Series],
    selection_mode: str,
    info_mode: str,
    fraction: float,
    seed: int,
) -> pd.Index:
    if subset.empty:
        return pd.Index([], dtype=int)
    return select_indices(
        subset,
        loads_by_step,
        static_profile,
        selection_mode,
        info_mode,
        fraction,
        seed,
    )


def run_scenarios(
    routes: pd.DataFrame,
    test_docs: list[int],
    num_experts: int,
    top_k: int,
    hidden_size: int,
    ep_size: int,
    gpus_per_node: int,
    placement: str,
    origin_mode: str,
    fraction: float,
    inter_node_gbps: float,
    static_profile: dict[str, pd.Series],
    threshold: float,
    detect_fracs: list[float],
    num_jobs: int,
    num_scenario_seeds: int,
    num_random_controls: int,
    seed_base: int,
) -> list[dict]:
    bw_bytes_per_us = inter_node_gbps * 1e9 / 8 / 1e6
    bytes_to_us = hidden_size / bw_bytes_per_us
    rows: list[dict] = []

    for scenario_seed in range(num_scenario_seeds):
        rng = np.random.default_rng(seed_base + scenario_seed)
        chosen_docs = rng.choice(
            test_docs,
            size=min(num_jobs, len(test_docs)),
            replace=len(test_docs) < num_jobs,
        ).tolist()
        num_layers = int(routes["layer"].max()) + 1
        max_stagger = max(1, num_layers // 2)
        arrivals = rng.integers(0, max_stagger + 1, size=len(chosen_docs))
        scenario = build_scenario(routes, chosen_docs, arrivals, origin_mode, ep_size)
        scenario["sender_rank"] = placement_map(
            scenario["expert_id"].to_numpy(), num_experts, ep_size, placement
        )
        remote = scenario[
            (scenario["sender_rank"] // gpus_per_node)
            != (scenario["receiver_rank"] // gpus_per_node)
        ]
        base_ingress = remote.groupby(["g", "receiver_rank"]).size().astype(float)
        base_egress = remote.groupby(["g", "sender_rank"]).size().astype(float)
        fp8_step_us = (
            pd.concat(
                [
                    base_ingress.groupby(level="g").max(),
                    base_egress.groupby(level="g").max(),
                ],
                axis=1,
            ).max(axis=1)
            * bytes_to_us
        )
        fp8_total = float(fp8_step_us.sum())
        tail_mask = remote["rank"].astype(int) > (top_k - max(1, top_k // 2))
        candidates = remote[tail_mask]
        loads_by_step = remote_loads_by_step(remote)
        g_max = int(remote["g"].max()) if not remote.empty else 0

        def evaluate(selected: pd.Index, cutoff: int) -> tuple[float, float]:
            step_us = step_us_from_counts(
                base_ingress, base_egress, candidates, selected, bytes_to_us
            )
            full_saving = 1.0 - float(step_us.sum()) / max(fp8_total, 1e-12)
            post_mask = step_us.index > cutoff
            post_fp8 = float(fp8_step_us[fp8_step_us.index > cutoff].sum())
            post_total = float(step_us[post_mask].sum())
            post_saving = (
                1.0 - post_total / max(post_fp8, 1e-12) if post_fp8 > 0 else 0.0
            )
            return full_saving, post_saving

        always_static_idx = select_subset(
            candidates,
            loads_by_step,
            static_profile,
            "hot",
            "calib_static",
            fraction,
            0,
        )
        always_causal_idx = select_subset(
            candidates,
            loads_by_step,
            static_profile,
            "hot",
            "causal_prev_step",
            fraction,
            0,
        )
        always_random_values: list[float] = []
        for trial in range(num_random_controls):
            idx = select_subset(
                candidates,
                loads_by_step,
                static_profile,
                "random",
                "random",
                fraction,
                seed_base + scenario_seed * 100 + trial,
            )
            value, _ = evaluate(idx, -1)
            always_random_values.append(value)
        always_static, _ = evaluate(always_static_idx, -1)
        always_causal, _ = evaluate(always_causal_idx, -1)
        always_random = float(np.mean(always_random_values))
        genie_idx = (
            always_static_idx
            if origin_mode == "hotspot"
            else select_subset(
                candidates,
                loads_by_step,
                static_profile,
                "random",
                "random",
                fraction,
                seed_base + scenario_seed,
            )
        )
        genie_saving, _ = evaluate(genie_idx, -1)

        for detect_frac in detect_fracs:
            cutoff = int(round(g_max * detect_frac))
            detected_score = receiver_concentration(remote, cutoff, ep_size)
            detected_regime = (
                "structural" if detected_score >= threshold else "transient"
            )
            warm_candidates = candidates[candidates["g"] <= cutoff]
            post_candidates = candidates[candidates["g"] > cutoff]
            warm_idx = select_subset(
                warm_candidates,
                loads_by_step,
                static_profile,
                "random",
                "random",
                fraction,
                seed_base + scenario_seed * 10 + int(detect_frac * 100),
            )
            if detected_regime == "structural":
                post_idx = select_subset(
                    post_candidates,
                    loads_by_step,
                    static_profile,
                    "hot",
                    "calib_static",
                    fraction,
                    0,
                )
            else:
                post_idx = select_subset(
                    post_candidates,
                    loads_by_step,
                    static_profile,
                    "random",
                    "random",
                    fraction,
                    seed_base + scenario_seed * 10 + 7,
                )
            adaptive_idx = warm_idx.append(post_idx)
            adaptive_full, adaptive_post = evaluate(adaptive_idx, cutoff)
            rows.append(
                {
                    "origin_mode": origin_mode,
                    "placement": placement,
                    "budget_fraction": fraction,
                    "scenario_seed": scenario_seed,
                    "detect_frac": detect_frac,
                    "detect_cutoff_g": cutoff,
                    "g_max": g_max,
                    "detected_score": detected_score,
                    "detected_regime": detected_regime,
                    "detection_correct": detected_regime
                    == ("structural" if origin_mode == "hotspot" else "transient"),
                    "adaptive_causal_saving": adaptive_full,
                    "adaptive_post_warmup_saving": adaptive_post,
                    "always_structural_profile_saving": always_static,
                    "always_causal_prev_step_saving": always_causal,
                    "always_random_saving": always_random,
                    "genie_regime_saving": genie_saving,
                    "adaptive_regret_vs_genie": genie_saving - adaptive_full,
                }
            )
    return rows


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--routes", required=True)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--num-jobs", type=int, default=16)
    parser.add_argument("--placement", default="contiguous")
    parser.add_argument("--budget-fraction", type=float, default=0.5)
    parser.add_argument("--inter-node-gbps", type=float, default=200.0)
    parser.add_argument("--calib-jobs", type=int, default=12)
    parser.add_argument("--detect-fracs", default="0.1,0.2,0.3,0.5")
    parser.add_argument("--num-scenario-seeds", type=int, default=24)
    parser.add_argument("--num-random-controls", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    num_experts, top_k = load_model_config(args.model)
    hidden_size = int(
        AutoConfig.from_pretrained(args.model, local_files_only=True).hidden_size
    )
    routes = pd.read_csv(args.routes)
    docs = sorted(routes["sample_id"].unique().tolist())
    calib_docs = docs[: args.calib_jobs]
    test_docs = docs[args.calib_jobs :]
    threshold, levels = calibrate_detector(
        routes,
        calib_docs,
        num_experts,
        args.ep_size,
        args.placement,
        args.gpus_per_node,
    )
    static_profile = build_static_profile(
        routes,
        calib_docs,
        num_experts,
        args.ep_size,
        args.placement,
        args.gpus_per_node,
    )
    detect_fracs = parse_float_list(args.detect_fracs)
    rows: list[dict] = []
    for origin_mode in ("hotspot", "balanced"):
        rows.extend(
            run_scenarios(
                routes,
                test_docs,
                num_experts,
                top_k,
                hidden_size,
                args.ep_size,
                args.gpus_per_node,
                args.placement,
                origin_mode,
                args.budget_fraction,
                args.inter_node_gbps,
                static_profile,
                threshold,
                detect_fracs,
                args.num_jobs,
                args.num_scenario_seeds,
                args.num_random_controls,
                2026072000 if origin_mode == "hotspot" else 2026072050,
            )
        )

    raw = pd.DataFrame(rows)
    raw.insert(0, "model", args.model_key)
    raw.to_csv(output / f"{args.model_key}_causal_raw.csv", index=False)
    summary = (
        raw.groupby(["model", "detect_frac", "origin_mode"], as_index=False)
        .agg(
            detection_accuracy=("detection_correct", "mean"),
            adaptive_causal_saving=("adaptive_causal_saving", "mean"),
            adaptive_post_warmup_saving=("adaptive_post_warmup_saving", "mean"),
            always_structural_profile_saving=(
                "always_structural_profile_saving",
                "mean",
            ),
            always_causal_prev_step_saving=(
                "always_causal_prev_step_saving",
                "mean",
            ),
            always_random_saving=("always_random_saving", "mean"),
            genie_regime_saving=("genie_regime_saving", "mean"),
            adaptive_regret_vs_genie=("adaptive_regret_vs_genie", "mean"),
        )
    )
    pooled = (
        raw.groupby(["model", "detect_frac"], as_index=False)
        .agg(
            detection_accuracy=("detection_correct", "mean"),
            adaptive_causal_saving=("adaptive_causal_saving", "mean"),
            adaptive_post_warmup_saving=("adaptive_post_warmup_saving", "mean"),
            always_structural_profile_saving=(
                "always_structural_profile_saving",
                "mean",
            ),
            always_causal_prev_step_saving=(
                "always_causal_prev_step_saving",
                "mean",
            ),
            always_random_saving=("always_random_saving", "mean"),
            genie_regime_saving=("genie_regime_saving", "mean"),
            adaptive_regret_vs_genie=("adaptive_regret_vs_genie", "mean"),
        )
    )
    summary.to_csv(output / f"{args.model_key}_causal_by_regime.csv", index=False)
    pooled.to_csv(output / f"{args.model_key}_causal_pooled.csv", index=False)
    metadata = pd.DataFrame(
        [
            {
                "model": args.model_key,
                "threshold": threshold,
                "calib_hotspot_score": levels["hotspot"],
                "calib_balanced_score": levels["balanced"],
                "static_profile_source": "hotspot_calibration_only",
                "warmup_policy": "random",
            }
        ]
    )
    metadata.to_csv(output / f"{args.model_key}_metadata.csv", index=False)

    print(f"detector levels={levels}, threshold={threshold:.6f}")
    print("\nPooled causal result:")
    print(pooled.to_string(index=False))
    print("\nPer-regime causal result:")
    print(summary.to_string(index=False))
    print(f"\nsaved to {output}")


if __name__ == "__main__":
    main()

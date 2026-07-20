"""Receiver-Aware v2: a more systematic, temporally-dynamic replay of the
confound-isolated receiver-aware(v1) mechanism.

v1 (``run_receiver_isolation_experiment.py``) already fixed the "prefer
remote" confound by restricting the candidate pool to tail-rank AND remote
pairs for every policy. It found that ``hot`` (real current-load scoring)
still beats ``random`` by a wide margin -- but it only used ONE model
(OLMoE), ONE synthetic scenario replicated from a single route file, and a
single, generous notion of "current load" that is available at the exact
instant the scheduling decision is made (a mild look-ahead: it assumes the
controller already knows the full concurrent load of the window it is
scoring).

v2 extends this along four axes that matter for whether the mechanism could
survive contact with a real system:

1. Cross-model: OLMoE (E64K8) AND LLM-jp (E32K16), on documents captured
   from wikitext-103 (never touched by any prior experiment in this
   project).
2. Cross-document: jobs are drawn from many independently captured real
   documents, not one route file replicated across synthetic job slots.
3. Temporal dynamics: jobs arrive with a staggered offset (continuous-batching
   style), so at any global time step, different jobs are at different local
   layers -- congestion state genuinely changes over time instead of being a
   single static snapshot.
4. Information staleness: three levels of "how much can the controller
   actually know when it makes the scoring decision":
     - ``oracle_same_step``: score using the SAME global-step's true remote
       load (the v1 assumption; an upper bound on what online information
       could provide).
     - ``causal_prev_step``: score using ONLY the previous global step's
       realized remote load (no knowledge of the future, the step being
       scored included) -- this is what an online controller could actually
       measure before acting.
     - ``calib_static``: score using a fixed profile pre-computed offline
       from an independent calibration scenario, with zero information about
       the current test scenario's actual state -- the pure/no-online-signal
       floor.

Performance note: the byte-assignment step is computed analytically as a
base (all-FP8) per-(g, rank) traffic count plus a sparse delta for whichever
candidate rows are upgraded to INT4, instead of copying and re-scanning the
full per-token route table for every one of the ~22 policy evaluations
(hot + cold + N random) per (fraction, info_mode) cell. This turns each
policy evaluation from O(all remote rows) into O(selected candidates),
which is what actually makes the full cross-model x cross-placement x
cross-seed sweep tractable on CPU.

Still bandwidth-only analytical trace replay: no collective/queueing/kernel
implementation, no real RDMA measurement.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--olmoe-model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--olmoe-routes", required=True)
    p.add_argument("--llmjp-model", default="llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M")
    p.add_argument("--llmjp-routes", required=True)
    p.add_argument("--ep-size", type=int, default=8)
    p.add_argument("--gpus-per-node", type=int, default=4)
    p.add_argument("--num-jobs", type=int, default=16)
    p.add_argument("--max-stagger-fraction", type=float, default=0.5,
                    help="max arrival stagger as a fraction of num_layers")
    p.add_argument("--placements", default="contiguous,round_robin")
    p.add_argument("--origin-modes", default="balanced,hotspot")
    p.add_argument("--budget-fractions", default="0.25,0.5,0.75")
    p.add_argument("--inter-node-gbps", type=float, default=200.0)
    p.add_argument("--num-scenario-seeds", type=int, default=24)
    p.add_argument("--num-random-controls", type=int, default=20)
    p.add_argument("--calib-jobs", type=int, default=12)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def placement_map(expert_id: np.ndarray, num_experts: int, ep_size: int, mapping: str) -> np.ndarray:
    if mapping == "contiguous":
        return np.minimum(expert_id * ep_size // num_experts, ep_size - 1)
    if mapping == "round_robin":
        return expert_id % ep_size
    raise ValueError(mapping)


def load_model_config(model_name: str) -> tuple[int, int]:
    cfg = AutoConfig.from_pretrained(model_name, local_files_only=True)
    num_experts = int(getattr(cfg, "num_experts", getattr(cfg, "num_local_experts", 0)))
    top_k = int(getattr(cfg, "num_experts_per_tok", getattr(cfg, "num_experts_per_token", 0)))
    return num_experts, top_k


def build_scenario(
    routes: pd.DataFrame,
    doc_ids: list[int],
    arrivals: np.ndarray,
    origin_mode: str,
    ep_size: int,
) -> pd.DataFrame:
    """One row per (job, local_layer, token_position, rank)."""
    frames = []
    for job_id, (doc_id, arrival) in enumerate(zip(doc_ids, arrivals)):
        rows = routes[routes["sample_id"] == doc_id].copy()
        rows["job_id"] = job_id
        rows["g"] = rows["layer"].astype(int) + int(arrival)
        if origin_mode == "balanced":
            receiver = job_id % ep_size
        elif origin_mode == "hotspot":
            hotspot_jobs = max(1, math.ceil(len(doc_ids) * 0.5))
            receiver = 0 if job_id < hotspot_jobs else 1 + ((job_id - hotspot_jobs) % max(ep_size - 1, 1))
        else:
            raise ValueError(origin_mode)
        rows["receiver_rank"] = receiver
        frames.append(rows[["job_id", "g", "rank", "expert_id", "receiver_rank"]])
    return pd.concat(frames, ignore_index=True)


def remote_loads_by_step(remote_rows: pd.DataFrame) -> dict[int, dict[str, pd.Series]]:
    out: dict[int, dict[str, pd.Series]] = {}
    for g, group in remote_rows.groupby("g"):
        out[g] = {
            "sender": group.groupby("sender_rank").size(),
            "receiver": group.groupby("receiver_rank").size(),
        }
    return out


def score_candidates(
    cand_rows: pd.DataFrame,
    loads_by_step: dict[int, dict[str, pd.Series]],
    info_mode: str,
    static_profile: dict[str, pd.Series] | None,
) -> pd.Series:
    scores = pd.Series(0.0, index=cand_rows.index)
    for g, group_idx in cand_rows.groupby("g").groups.items():
        sub = cand_rows.loc[group_idx]
        if info_mode == "oracle_same_step":
            loads = loads_by_step.get(g)
        elif info_mode == "causal_prev_step":
            loads = loads_by_step.get(g - 1)
        elif info_mode == "calib_static":
            loads = static_profile
        else:
            raise ValueError(info_mode)
        if loads is None:
            continue
        s = sub["sender_rank"].map(loads["sender"]).fillna(0.0)
        r = sub["receiver_rank"].map(loads["receiver"]).fillna(0.0)
        scores.loc[group_idx] = np.maximum(s.to_numpy(), r.to_numpy())
    return scores


def select_indices(
    cand_rows: pd.DataFrame,
    loads_by_step: dict[int, dict[str, pd.Series]],
    static_profile: dict[str, pd.Series] | None,
    selection_mode: str,
    info_mode: str,
    fraction: float,
    seed: int,
) -> pd.Index:
    """Pick, independently within each global step g, `fraction` of that
    step's candidates according to the given selection/info mode."""
    if selection_mode == "random":
        rng = np.random.default_rng(seed)
        chosen: list[np.ndarray] = []
        for g, group_idx in cand_rows.groupby("g").groups.items():
            budget = int(round(len(group_idx) * fraction))
            if budget <= 0:
                continue
            chosen.append(rng.choice(group_idx.to_numpy(), size=min(budget, len(group_idx)), replace=False))
        return pd.Index(np.concatenate(chosen)) if chosen else pd.Index([], dtype=cand_rows.index.dtype)

    scores = score_candidates(cand_rows, loads_by_step, info_mode, static_profile)
    chosen = []
    ascending = selection_mode == "cold"
    for g, group_idx in cand_rows.groupby("g").groups.items():
        budget = int(round(len(group_idx) * fraction))
        if budget <= 0:
            continue
        sub_scores = scores.loc[group_idx].sort_values(ascending=ascending)
        chosen.append(sub_scores.index[:budget].to_numpy())
    return pd.Index(np.concatenate(chosen)) if chosen else pd.Index([], dtype=cand_rows.index.dtype)


def step_us_from_counts(
    base_ingress: pd.Series,
    base_egress: pd.Series,
    cand_rows: pd.DataFrame,
    selected_idx: pd.Index,
    bytes_to_us: float,
) -> pd.Series:
    """Analytic byte accounting: start from the all-FP8 per-(g,rank) traffic
    COUNT, then subtract 0.5 for every selected row's (g, sender_rank) and
    (g, receiver_rank) bucket (since it moves from 1.0 to 0.5 bytes/elem).
    This avoids rebuilding the whole route table per policy evaluation."""
    if len(selected_idx) == 0:
        ingress, egress = base_ingress, base_egress
    else:
        sel = cand_rows.loc[selected_idx]
        sel_ingress = sel.groupby(["g", "receiver_rank"]).size().astype(float)
        sel_egress = sel.groupby(["g", "sender_rank"]).size().astype(float)
        ingress = base_ingress.subtract(0.5 * sel_ingress, fill_value=0.0)
        egress = base_egress.subtract(0.5 * sel_egress, fill_value=0.0)
    ingress_max = ingress.groupby(level="g").max()
    egress_max = egress.groupby(level="g").max()
    step_max = pd.concat([ingress_max, egress_max], axis=1).max(axis=1)
    return step_max * bytes_to_us


def job_p99(job_g_pairs: pd.DataFrame, step_us: pd.Series) -> float:
    if step_us.empty:
        return 0.0
    active = job_g_pairs[job_g_pairs["g"].isin(step_us.index)]
    if active.empty:
        return 0.0
    exposure = active.assign(v=active["g"].map(step_us)).groupby("job_id")["v"].sum()
    return float(np.quantile(exposure.to_numpy(), 0.99))


def run_one_cell(
    base_ingress: pd.Series,
    base_egress: pd.Series,
    cand_rows: pd.DataFrame,
    loads_by_step: dict[int, dict[str, pd.Series]],
    static_profile: dict[str, pd.Series] | None,
    job_g_pairs: pd.DataFrame,
    info_mode: str,
    bytes_to_us: float,
    fp8_total: float,
    fraction: float,
    num_random: int,
    seed: int,
) -> dict[str, object]:
    def total_and_step(selection_mode: str, rand_seed: int) -> tuple[float, pd.Series]:
        idx = select_indices(cand_rows, loads_by_step, static_profile, selection_mode, info_mode, fraction, rand_seed)
        step_us = step_us_from_counts(base_ingress, base_egress, cand_rows, idx, bytes_to_us)
        return float(step_us.sum()), step_us

    hot_total, hot_step_us = total_and_step("hot", 0)
    cold_total, _ = total_and_step("cold", 0)
    random_totals = []
    random_p99s = []
    for trial in range(num_random):
        total, step_us = total_and_step("random", seed + trial)
        random_totals.append(total)
        random_p99s.append(job_p99(job_g_pairs, step_us))

    hot_job_p99 = job_p99(job_g_pairs, hot_step_us)
    random_mean = float(np.mean(random_totals))
    random_ci_low = float(np.quantile(random_totals, 0.025))
    random_ci_high = float(np.quantile(random_totals, 0.975))
    hot_saving = 1.0 - hot_total / max(fp8_total, 1e-12)
    random_saving_mean = 1.0 - random_mean / max(fp8_total, 1e-12)
    cold_saving = 1.0 - cold_total / max(fp8_total, 1e-12)
    random_saving_ci_low = 1.0 - random_ci_high / max(fp8_total, 1e-12)
    random_saving_ci_high = 1.0 - random_ci_low / max(fp8_total, 1e-12)

    return {
        "info_mode": info_mode,
        "hot_saving": hot_saving,
        "cold_saving": cold_saving,
        "random_saving_mean": random_saving_mean,
        "random_saving_ci_low": random_saving_ci_low,
        "random_saving_ci_high": random_saving_ci_high,
        "hot_minus_random": hot_saving - random_saving_mean,
        "hot_within_random_ci": bool(random_saving_ci_low <= hot_saving <= random_saving_ci_high),
        "hot_job_p99_us": hot_job_p99,
        "random_job_p99_us_median": float(np.median(random_p99s)) if random_p99s else 0.0,
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    models = {
        "olmoe": (args.olmoe_model, args.olmoe_routes),
        "llmjp": (args.llmjp_model, args.llmjp_routes),
    }
    placements = [v.strip() for v in args.placements.split(",") if v.strip()]
    origin_modes = [v.strip() for v in args.origin_modes.split(",") if v.strip()]
    fractions = [float(v) for v in args.budget_fractions.split(",") if v]
    info_modes = ["oracle_same_step", "causal_prev_step", "calib_static"]
    bw_bytes_per_us = args.inter_node_gbps * 1e9 / 8 / 1e6

    t_start = time.time()
    all_rows: list[dict[str, object]] = []
    for model_key, (model_name, route_path) in models.items():
        num_experts, top_k = load_model_config(model_name)
        routes = pd.read_csv(route_path)
        hidden_size = int(AutoConfig.from_pretrained(model_name, local_files_only=True).hidden_size)
        bytes_to_us = hidden_size / bw_bytes_per_us
        num_layers = int(routes["layer"].max()) + 1
        doc_pool = sorted(routes["sample_id"].unique().tolist())
        max_stagger = max(1, int(round(num_layers * args.max_stagger_fraction)))

        calib_docs = doc_pool[: args.calib_jobs]
        test_docs = doc_pool[args.calib_jobs:]
        if len(test_docs) < 4:
            raise RuntimeError(f"{model_key}: not enough documents left for test after calibration split")

        for placement in placements:
            for origin_mode in origin_modes:
                calib_arrivals = np.zeros(len(calib_docs), dtype=int)
                calib_scn = build_scenario(routes, calib_docs, calib_arrivals, origin_mode, args.ep_size)
                calib_scn["sender_rank"] = placement_map(calib_scn["expert_id"].to_numpy(), num_experts, args.ep_size, placement)
                calib_remote = calib_scn[(calib_scn["sender_rank"] // args.gpus_per_node) != (calib_scn["receiver_rank"] // args.gpus_per_node)]
                n_steps_calib = max(calib_scn["g"].nunique(), 1)
                static_profile = {
                    "sender": calib_remote.groupby("sender_rank").size() / n_steps_calib,
                    "receiver": calib_remote.groupby("receiver_rank").size() / n_steps_calib,
                }

                for scenario_seed in range(args.num_scenario_seeds):
                    t_cell = time.time()
                    rng = np.random.default_rng(20260719_00 + scenario_seed)
                    chosen_docs = rng.choice(
                        test_docs, size=min(args.num_jobs, len(test_docs)), replace=len(test_docs) < args.num_jobs
                    ).tolist()
                    arrivals = rng.integers(0, max_stagger + 1, size=len(chosen_docs))
                    scn = build_scenario(routes, chosen_docs, arrivals, origin_mode, args.ep_size)
                    scn["sender_rank"] = placement_map(scn["expert_id"].to_numpy(), num_experts, args.ep_size, placement)

                    job_g_pairs = scn[["job_id", "g"]].drop_duplicates()
                    remote_rows = scn[(scn["sender_rank"] // args.gpus_per_node) != (scn["receiver_rank"] // args.gpus_per_node)]
                    base_ingress = remote_rows.groupby(["g", "receiver_rank"]).size().astype(float)
                    base_egress = remote_rows.groupby(["g", "sender_rank"]).size().astype(float)
                    fp8_step_us = pd.concat(
                        [base_ingress.groupby(level="g").max(), base_egress.groupby(level="g").max()], axis=1
                    ).max(axis=1) * bytes_to_us
                    fp8_total = float(fp8_step_us.sum())

                    tail_mask = remote_rows["rank"].astype(int) > (top_k - max(1, top_k // 2))
                    cand_rows = remote_rows[tail_mask]
                    loads_by_step = remote_loads_by_step(remote_rows)

                    for fraction in fractions:
                        for info_mode in info_modes:
                            result = run_one_cell(
                                base_ingress, base_egress, cand_rows, loads_by_step, static_profile,
                                job_g_pairs, info_mode, bytes_to_us, fp8_total,
                                fraction, args.num_random_controls, seed=1000 * scenario_seed,
                            )
                            result.update({
                                "model": model_key,
                                "placement": placement,
                                "origin_mode": origin_mode,
                                "budget_fraction": fraction,
                                "scenario_seed": scenario_seed,
                                "num_jobs": len(chosen_docs),
                                "num_layers": num_layers,
                                "max_stagger": max_stagger,
                            })
                            all_rows.append(result)
                    print(
                        f"[{time.time() - t_start:7.1f}s] {model_key} {placement} {origin_mode} "
                        f"seed={scenario_seed} done in {time.time() - t_cell:.2f}s "
                        f"(remote_rows={len(remote_rows)}, candidates={len(cand_rows)})",
                        flush=True,
                    )

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "receiver_aware_v2_raw.csv", index=False)

    summary = df.groupby(["model", "placement", "origin_mode", "budget_fraction", "info_mode"], as_index=False).agg(
        hot_minus_random_mean=("hot_minus_random", "mean"),
        hot_minus_random_std=("hot_minus_random", "std"),
        frac_seeds_hot_beats_random_ci=("hot_within_random_ci", lambda s: float((~s).mean())),
        hot_job_p99_us_mean=("hot_job_p99_us", "mean"),
        random_job_p99_us_median_mean=("random_job_p99_us_median", "mean"),
        n_seeds=("scenario_seed", "nunique"),
    )
    summary.to_csv(out / "receiver_aware_v2_summary.csv", index=False)

    pivot = summary.pivot_table(
        index=["model", "placement", "origin_mode", "budget_fraction"],
        columns="info_mode", values="hot_minus_random_mean",
    ).reset_index()
    if "oracle_same_step" in pivot.columns and "causal_prev_step" in pivot.columns:
        pivot["staleness_cost_oracle_minus_causal"] = pivot["oracle_same_step"] - pivot["causal_prev_step"]
    if "causal_prev_step" in pivot.columns and "calib_static" in pivot.columns:
        pivot["causal_advantage_over_calib_static"] = pivot["causal_prev_step"] - pivot["calib_static"]
    pivot.to_csv(out / "receiver_aware_v2_staleness_cost.csv", index=False)

    lines = ["# Receiver-Aware v2: Systematic Cross-Model Temporal Replay", ""]
    lines.append(f"models={list(models)}; placements={placements}; origin_modes={origin_modes}; "
                 f"budget_fractions={fractions}; scenario_seeds={args.num_scenario_seeds}; num_jobs={args.num_jobs}")
    lines.append("")
    lines.append("## hot - random saving (mean over scenario seeds); positive & 'frac_seeds...' close to 1.0 means hot robustly beats random")
    lines.append("")
    cols = ["model", "placement", "origin_mode", "budget_fraction", "info_mode",
            "hot_minus_random_mean", "hot_minus_random_std", "frac_seeds_hot_beats_random_ci", "n_seeds"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in summary.sort_values(["model", "placement", "origin_mode", "budget_fraction", "info_mode"]).iterrows():
        vals = [f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## Staleness cost: oracle_same_step vs causal_prev_step vs calib_static")
    lines.append("")
    scols = list(pivot.columns)
    lines.append("| " + " | ".join(scols) + " |")
    lines.append("|" + "|".join(["---"] * len(scols)) + "|")
    for _, row in pivot.iterrows():
        vals = [f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in scols]
        lines.append("| " + " | ".join(vals) + " |")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-40:]))
    print(f"\ntotal wall time: {time.time() - t_start:.1f}s")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()

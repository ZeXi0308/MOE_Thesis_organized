#!/usr/bin/env python3
"""Historical P0 for contribution-risk-aware shadow-expert placement.

MassCover-EP asks a deliberately narrow question:

    Under a fixed inactive-shadow memory budget, can a placement selected on
    calibration routes reduce the tail of *uncovered combine contribution*
    after a whole failure-domain loss better than load/frequency baselines?

This is a route-level, retrospective screen.  It does not emulate failure
detection, expert reload, mutable CUDA graphs, actual serving availability, or
end-to-end quality.  ``gate_weight / sum(top-k gate_weight)`` is used as an
online-observable contribution proxy; a separate small intervention artifact
is used only to audit that proxy.

The old ``p0b_sealed_*`` captures pre-date this hypothesis but have already
been inspected by the project.  They are therefore labelled historical test,
not a newly sealed confirmatory holdout.
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
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr


SCHEMA = "masscover_ep.historical_p0.v1"
POLICIES = ("frequency", "gate_mass", "cvar_greedy", "oracle_cvar")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_name: str
    num_experts: int
    top_k: int
    calibration_routes: Path
    historical_routes: Path
    placement_registry: Path
    contribution_audit: Path


@dataclass
class RiskMatrix:
    matrix_csr: sparse.csr_matrix
    matrix_csc: sparse.csc_matrix
    initial_residual: np.ndarray
    request_for_cell: np.ndarray
    request_ids: np.ndarray
    num_tokens: int
    num_failure_domains: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cvar(values: np.ndarray, alpha: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("CVaR requires at least one value")
    threshold = float(np.quantile(array, alpha))
    tail = array[array >= threshold]
    return float(tail.mean()) if tail.size else threshold


def normalize_route_table(path: Path, spec: ModelSpec) -> pd.DataFrame:
    required = {
        "sample_id",
        "layer",
        "token_position",
        "rank",
        "expert_id",
        "gate_weight",
        "home_rank",
    }
    table = pd.read_csv(
        path,
        usecols=sorted(required),
        dtype={
            "sample_id": "int64",
            "layer": "int16",
            "token_position": "int32",
            "rank": "int16",
            "expert_id": "int16",
            "gate_weight": "float32",
            "home_rank": "int16",
        },
    )
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    if table.empty:
        raise ValueError(f"{path} is empty")
    if table["expert_id"].min() < 0 or table["expert_id"].max() >= spec.num_experts:
        raise ValueError(f"{path} has expert outside E={spec.num_experts}")

    key = ["sample_id", "layer", "token_position"]
    sizes = table.groupby(key, sort=False, observed=True).size()
    if not bool((sizes == spec.top_k).all()):
        raise ValueError(f"{path} does not have exactly top_k={spec.top_k} rows per token")
    rank_ok = (
        table.groupby(key, sort=False, observed=True)["rank"]
        .agg(["min", "max", "nunique"])
    )
    if not bool(
        (
            (rank_ok["min"] == 1)
            & (rank_ok["max"] == spec.top_k)
            & (rank_ok["nunique"] == spec.top_k)
        ).all()
    ):
        raise ValueError(f"{path} has malformed rank rows")
    expert_unique = table.groupby(key, sort=False, observed=True)["expert_id"].nunique()
    if not bool((expert_unique == spec.top_k).all()):
        raise ValueError(f"{path} has duplicate experts within a token")
    home_unique = table.groupby(key, sort=False, observed=True)["home_rank"].nunique()
    if not bool((home_unique == 1).all()):
        raise ValueError(f"{path} has inconsistent home rank within a token")

    denominator = table.groupby(key, sort=False, observed=True)["gate_weight"].transform("sum")
    if bool((denominator <= 0).any()):
        raise ValueError(f"{path} contains non-positive top-k gate sum")
    table["gate_share"] = (table["gate_weight"] / denominator).astype("float32")
    token_keys = table[key].drop_duplicates(ignore_index=True)
    token_keys["token_index"] = np.arange(len(token_keys), dtype=np.int64)
    table = table.merge(token_keys, on=key, how="left", validate="many_to_one")
    return table.sort_values(key + ["rank"], kind="stable", ignore_index=True)


def load_placements(path: Path, ep_size: int) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    placements: dict[str, np.ndarray] = {}
    for row in payload["placements"]:
        name = str(row["name"])
        mapping = np.asarray(row["expert_to_rank"], dtype=np.int16)
        if mapping.ndim != 1 or np.any(mapping < 0) or np.any(mapping >= ep_size):
            raise ValueError(f"invalid placement {name} in {path}")
        placements[name] = mapping
    required = {
        "round_robin",
        "contiguous",
        "calibration_frequency_lpt",
        "calibration_coactivation_balanced",
    }
    missing = required - set(placements)
    if missing:
        raise ValueError(f"placement registry lacks {sorted(missing)}")
    return {name: placements[name] for name in sorted(required)}


def build_risk_matrix(
    routes: pd.DataFrame,
    expert_to_rank: np.ndarray,
    *,
    num_experts: int,
    gpus_per_failure_domain: int,
) -> RiskMatrix:
    if len(expert_to_rank) != num_experts:
        raise ValueError("expert mapping length does not match architecture")
    ep_size = int(expert_to_rank.max()) + 1
    if ep_size % gpus_per_failure_domain:
        raise ValueError("EP size must be divisible by ranks per failure domain")
    num_domains = ep_size // gpus_per_failure_domain
    if num_domains < 2:
        raise ValueError("failure analysis requires at least two domains")

    owner_rank = expert_to_rank[routes["expert_id"].to_numpy(dtype=np.int64)]
    failed_domain = owner_rank // gpus_per_failure_domain
    token_index = routes["token_index"].to_numpy(dtype=np.int64)
    layer = routes["layer"].to_numpy(dtype=np.int64)
    expert = routes["expert_id"].to_numpy(dtype=np.int64)
    candidate = layer * num_experts + expert
    cell = token_index * num_domains + failed_domain
    share = routes["gate_share"].to_numpy(dtype=np.float32)
    num_tokens = int(token_index.max()) + 1
    num_candidates = (int(layer.max()) + 1) * num_experts
    matrix = sparse.coo_matrix(
        (share, (cell, candidate)),
        shape=(num_tokens * num_domains, num_candidates),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    initial = np.asarray(matrix.sum(axis=1), dtype=np.float64).ravel()

    token_requests = (
        routes[["token_index", "sample_id"]]
        .drop_duplicates("token_index")
        .sort_values("token_index")["sample_id"]
        .to_numpy(dtype=np.int64)
    )
    if len(token_requests) != num_tokens:
        raise ValueError("token-to-request mapping is incomplete")
    request_for_cell = np.repeat(token_requests, num_domains)
    return RiskMatrix(
        matrix_csr=matrix,
        matrix_csc=matrix.tocsc(),
        initial_residual=initial,
        request_for_cell=request_for_cell,
        request_ids=np.unique(token_requests),
        num_tokens=num_tokens,
        num_failure_domains=num_domains,
    )


def static_order(matrix: RiskMatrix, policy: str) -> np.ndarray:
    if policy == "frequency":
        scores = np.asarray((matrix.matrix_csr != 0).sum(axis=0)).ravel()
    elif policy == "gate_mass":
        scores = np.asarray(matrix.matrix_csr.sum(axis=0)).ravel()
    else:
        raise ValueError(f"unknown static policy {policy}")
    candidate = np.arange(len(scores), dtype=np.int64)
    return np.lexsort((candidate, -scores)).astype(np.int64)


def cvar_greedy_order(
    matrix: RiskMatrix,
    *,
    max_candidates: int,
    alpha: float,
) -> np.ndarray:
    residual = matrix.initial_residual.copy()
    selected = np.zeros(matrix.matrix_csr.shape[1], dtype=bool)
    order: list[int] = []
    for _ in range(max_candidates):
        threshold = float(np.quantile(residual, alpha))
        tail = (residual >= threshold) & (residual > 1e-12)
        if not bool(tail.any()):
            remaining = np.flatnonzero(~selected)
            order.extend(remaining[: max_candidates - len(order)].tolist())
            break
        scores = np.asarray(matrix.matrix_csr[tail].sum(axis=0)).ravel()
        scores[selected] = -1.0
        best = int(np.argmax(scores))
        if scores[best] <= 0:
            remaining = np.flatnonzero(~selected)
            order.extend(remaining[: max_candidates - len(order)].tolist())
            break
        selected[best] = True
        order.append(best)
        column = matrix.matrix_csc.getcol(best)
        residual[column.indices] -= column.data
        np.maximum(residual, 0.0, out=residual)
    return np.asarray(order, dtype=np.int64)


def random_orders(num_candidates: int, max_candidates: int, seeds: Iterable[int]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        result[f"random_{seed}"] = rng.permutation(num_candidates)[:max_candidates]
    return result


def residual_after(matrix: RiskMatrix, selected: np.ndarray) -> np.ndarray:
    if len(selected) == 0:
        return matrix.initial_residual.copy()
    covered = np.asarray(matrix.matrix_csr[:, selected].sum(axis=1)).ravel()
    return np.maximum(matrix.initial_residual - covered, 0.0)


def request_level_cvar(matrix: RiskMatrix, residual: np.ndarray, alpha: float) -> np.ndarray:
    values: list[float] = []
    for request_id in matrix.request_ids:
        mask = matrix.request_for_cell == request_id
        values.append(cvar(residual[mask], alpha))
    return np.asarray(values, dtype=np.float64)


def summarize_residual(
    matrix: RiskMatrix,
    residual: np.ndarray,
    *,
    alpha: float,
) -> dict[str, float]:
    return {
        "mean_uncovered_mass": float(np.mean(residual)),
        "p50_uncovered_mass": float(np.quantile(residual, 0.50)),
        "p95_uncovered_mass": float(np.quantile(residual, 0.95)),
        "p99_uncovered_mass": float(np.quantile(residual, 0.99)),
        "cvar95_uncovered_mass": cvar(residual, alpha),
        "max_uncovered_mass": float(np.max(residual)),
        "fraction_mass_le_0p10": float(np.mean(residual <= 0.10 + 1e-12)),
        "fraction_mass_le_0p20": float(np.mean(residual <= 0.20 + 1e-12)),
        "fraction_fully_covered": float(np.mean(residual <= 1e-12)),
        "request_cvar95_mean": float(np.mean(request_level_cvar(matrix, residual, alpha))),
    }


def c1_cross_domain_fraction(
    routes: pd.DataFrame,
    expert_to_rank: np.ndarray,
    *,
    gpus_per_failure_domain: int,
) -> float:
    keys = ["sample_id", "layer", "token_position"]
    owner = expert_to_rank[routes["expert_id"].to_numpy(dtype=np.int64)]
    lowered = routes[keys + ["home_rank"]].copy()
    lowered["owner_rank"] = owner
    lowered = lowered.drop_duplicates(keys + ["owner_rank"])
    remote = (
        lowered["home_rank"].to_numpy(dtype=np.int64) // gpus_per_failure_domain
        != lowered["owner_rank"].to_numpy(dtype=np.int64) // gpus_per_failure_domain
    )
    return float(np.mean(remote))


def proxy_fidelity(path: Path, model_key: str) -> dict[str, float | str | int]:
    table = pd.read_csv(
        path,
        usecols=["sample_id", "layer", "expert_id", "gate", "contribution"],
    )
    rho = float(spearmanr(table["gate"], table["contribution"]).statistic)
    aggregated = table.groupby(["layer", "expert_id"], observed=True)[
        ["gate", "contribution"]
    ].sum()
    aggregate_rho = float(
        spearmanr(aggregated["gate"], aggregated["contribution"]).statistic
    )
    return {
        "model": model_key,
        "pair_count": int(len(table)),
        "request_count": int(table["sample_id"].nunique()),
        "pair_spearman_gate_vs_contribution": rho,
        "layer_expert_mass_spearman": aggregate_rho,
        "evidence_level": "small historical contribution audit",
    }


def evaluate_model(
    spec: ModelSpec,
    *,
    output_dir: Path,
    ep_size: int,
    gpus_per_failure_domain: int,
    budget_fractions: list[float],
    cvar_alpha: float,
    random_seed_count: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    calibration = normalize_route_table(spec.calibration_routes, spec)
    historical = normalize_route_table(spec.historical_routes, spec)
    placements = load_placements(spec.placement_registry, ep_size)
    if calibration["home_rank"].max() >= ep_size or historical["home_rank"].max() >= ep_size:
        raise ValueError(f"{spec.key}: observed home rank is incompatible with EP={ep_size}")

    result_rows: list[dict[str, object]] = []
    placement_rows: list[dict[str, object]] = []
    max_budget = max(1, math.ceil(max(budget_fractions) * calibration["layer"].nunique() * spec.num_experts))
    seeds = [2026071900 + index for index in range(random_seed_count)]

    for placement_name, mapping in placements.items():
        calibration_matrix = build_risk_matrix(
            calibration,
            mapping,
            num_experts=spec.num_experts,
            gpus_per_failure_domain=gpus_per_failure_domain,
        )
        historical_matrix = build_risk_matrix(
            historical,
            mapping,
            num_experts=spec.num_experts,
            gpus_per_failure_domain=gpus_per_failure_domain,
        )
        candidate_count = calibration_matrix.matrix_csr.shape[1]
        max_budget_here = min(max_budget, candidate_count)
        orders: dict[str, np.ndarray] = {
            "frequency": static_order(calibration_matrix, "frequency"),
            "gate_mass": static_order(calibration_matrix, "gate_mass"),
            "cvar_greedy": cvar_greedy_order(
                calibration_matrix,
                max_candidates=max_budget_here,
                alpha=cvar_alpha,
            ),
            "oracle_cvar": cvar_greedy_order(
                historical_matrix,
                max_candidates=max_budget_here,
                alpha=cvar_alpha,
            ),
        }
        orders.update(random_orders(candidate_count, max_budget_here, seeds))

        baseline = summarize_residual(
            historical_matrix,
            historical_matrix.initial_residual,
            alpha=cvar_alpha,
        )
        placement_rows.append(
            {
                "model": spec.key,
                "placement": placement_name,
                "c1_cross_failure_domain_fraction": c1_cross_domain_fraction(
                    historical,
                    mapping,
                    gpus_per_failure_domain=gpus_per_failure_domain,
                ),
                "no_shadow_mean_uncovered_mass": baseline["mean_uncovered_mass"],
                "no_shadow_cvar95_uncovered_mass": baseline["cvar95_uncovered_mass"],
                "no_shadow_p99_uncovered_mass": baseline["p99_uncovered_mass"],
            }
        )

        for fraction in budget_fractions:
            budget = min(candidate_count, int(round(fraction * candidate_count)))
            if fraction > 0 and budget == 0:
                budget = 1
            for policy_name, order in orders.items():
                selected = order[:budget]
                residual = residual_after(historical_matrix, selected)
                metrics = summarize_residual(
                    historical_matrix,
                    residual,
                    alpha=cvar_alpha,
                )
                row: dict[str, object] = {
                    "model": spec.key,
                    "placement": placement_name,
                    "policy": policy_name,
                    "policy_family": "random" if policy_name.startswith("random_") else policy_name,
                    "budget_fraction": float(fraction),
                    "shadow_count": int(budget),
                    "candidate_count": int(candidate_count),
                    "selection_split": "historical oracle" if policy_name == "oracle_cvar" else "calibration",
                    "evaluation_split": "historical test",
                    **metrics,
                }
                base_cvar = float(baseline["cvar95_uncovered_mass"])
                row["cvar95_reduction_vs_no_shadow"] = (
                    0.0 if base_cvar <= 0 else 1.0 - metrics["cvar95_uncovered_mass"] / base_cvar
                )
                result_rows.append(row)

        del calibration_matrix, historical_matrix

    metadata = {
        "model": spec.key,
        "model_name": spec.model_name,
        "num_experts": spec.num_experts,
        "top_k": spec.top_k,
        "calibration_request_count": int(calibration["sample_id"].nunique()),
        "historical_request_count": int(historical["sample_id"].nunique()),
        "calibration_token_count": int(calibration["token_index"].nunique()),
        "historical_token_count": int(historical["token_index"].nunique()),
        "calibration_sha256": sha256_file(spec.calibration_routes),
        "historical_sha256": sha256_file(spec.historical_routes),
        "placement_registry_sha256": sha256_file(spec.placement_registry),
    }
    return result_rows, placement_rows, metadata


def aggregate_random(results: pd.DataFrame) -> pd.DataFrame:
    random = results[results["policy_family"] == "random"]
    if random.empty:
        return pd.DataFrame()
    keys = ["model", "placement", "budget_fraction", "shadow_count", "candidate_count"]
    metrics = [
        "mean_uncovered_mass",
        "p95_uncovered_mass",
        "p99_uncovered_mass",
        "cvar95_uncovered_mass",
        "fraction_mass_le_0p10",
        "fraction_mass_le_0p20",
        "fraction_fully_covered",
        "request_cvar95_mean",
        "cvar95_reduction_vs_no_shadow",
    ]
    grouped = random.groupby(keys, as_index=False, observed=True)[metrics].median()
    grouped["policy"] = "random_median"
    grouped["policy_family"] = "random_median"
    grouped["selection_split"] = "calibration-independent random"
    grouped["evaluation_split"] = "historical test"
    return grouped


def make_verdict(
    results: pd.DataFrame,
    fidelity: pd.DataFrame,
    *,
    target_budget: float,
) -> dict[str, object]:
    target = results[np.isclose(results["budget_fraction"], target_budget)].copy()
    deployable = target[~target["policy_family"].isin(["random", "oracle_cvar"])]
    random_summary = aggregate_random(target)
    combined = pd.concat([deployable, random_summary], ignore_index=True, sort=False)
    comparison_rows: list[dict[str, object]] = []
    pass_by_model: dict[str, bool] = {}
    for model, model_rows in combined.groupby("model", observed=True):
        placement_pass: list[bool] = []
        for placement, rows in model_rows.groupby("placement", observed=True):
            ours = rows[rows["policy"] == "cvar_greedy"]
            baselines = rows[rows["policy"].isin(["frequency", "gate_mass", "random_median"])]
            if ours.empty or baselines.empty:
                continue
            ours_value = float(ours.iloc[0]["cvar95_uncovered_mass"])
            best_base = baselines.sort_values("cvar95_uncovered_mass").iloc[0]
            base_value = float(best_base["cvar95_uncovered_mass"])
            improvement = 0.0 if base_value <= 0 else 1.0 - ours_value / base_value
            passed = improvement >= 0.10
            placement_pass.append(passed)
            comparison_rows.append(
                {
                    "model": model,
                    "placement": placement,
                    "budget_fraction": target_budget,
                    "cvar_greedy": ours_value,
                    "best_baseline_policy": str(best_base["policy"]),
                    "best_baseline": base_value,
                    "relative_improvement": improvement,
                    "passes_10pct_gate": passed,
                }
            )
        pass_by_model[str(model)] = bool(placement_pass) and sum(placement_pass) >= math.ceil(0.75 * len(placement_pass))

    proxy_pass = bool(
        (fidelity["pair_spearman_gate_vs_contribution"] >= 0.50).all()
    )
    route_pass = bool(pass_by_model) and all(pass_by_model.values())
    passed = proxy_pass and route_pass
    return {
        "target_budget_fraction": target_budget,
        "proxy_gate": {
            "threshold": "pair Spearman >= 0.50 in both models",
            "passed": proxy_pass,
        },
        "route_risk_gate": {
            "threshold": "CVaR-greedy improves >=10% over best frequency/gate/random baseline in >=75% placements for each model",
            "passed": route_pass,
            "model_pass": pass_by_model,
            "comparisons": comparison_rows,
        },
        "p0_verdict": "PASS_TO_END_TO_END_FAILURE_MASK_P1" if passed else "KILL_OR_REFORMULATE_MASSCOVER",
        "claim_boundary": (
            "Passing licenses only an end-to-end expert-failure masking experiment. "
            "It is not evidence of actual recovery latency, availability, TPOT, P99, or quality."
        ),
    }


def write_report(
    output_dir: Path,
    verdict: dict[str, object],
    placements: pd.DataFrame,
    fidelity: pd.DataFrame,
    results_with_random: pd.DataFrame,
) -> None:
    lines = [
        "# MassCover-EP 历史 P0：有限 shadow 预算能否约束故障质量爆炸半径",
        "",
        f"> 自动判决：**{verdict['p0_verdict']}**",
        "",
        "## 证据边界",
        "",
        "- 这是 route-level historical retrospective screen，不是新 sealed confirmatory experiment。",
        "- C1、gate share 和 missing-mass 都是逻辑/语义代理；没有测 wire、恢复时延、TTFT、TPOT 或 P99。",
        "- shadow expert 在正常路径上假定 inactive；这里只计 expert-copy 数量，不计真实 HBM、加载和路由表更新成本。",
        "- `oracle_cvar` 在 historical test 上选副本，只表示上限，不能部署。",
        "",
        "## 核心假设与门槛",
        "",
        "在 10% layer-expert shadow budget 下，calibration-only CVaR greedy 必须在每个模型至少 75% 的冻结 primary placements 上，相对 frequency、gate-mass 和 random median 中的最佳者再降低至少 10% 的 historical-test CVaR95；同时 gate 对真实 contribution 的 pair-level Spearman 必须在两模型均不低于 0.50。",
        "",
        "## Gate → contribution proxy 审计",
        "",
        "| model | pairs | requests | pair Spearman | layer-expert mass Spearman |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in fidelity.to_dict("records"):
        lines.append(
            f"| {row['model']} | {row['pair_count']} | {row['request_count']} | "
            f"{row['pair_spearman_gate_vs_contribution']:.3f} | {row['layer_expert_mass_spearman']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Primary placement 的正常通信与故障风险",
            "",
            "| model | placement | C1 cross-domain fraction | no-shadow mean missing mass | no-shadow CVaR95 | no-shadow P99 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in placements.sort_values(["model", "placement"]).to_dict("records"):
        lines.append(
            f"| {row['model']} | {row['placement']} | {row['c1_cross_failure_domain_fraction']:.3f} | "
            f"{row['no_shadow_mean_uncovered_mass']:.3f} | {row['no_shadow_cvar95_uncovered_mass']:.3f} | "
            f"{row['no_shadow_p99_uncovered_mass']:.3f} |"
        )

    comparisons = verdict["route_risk_gate"]["comparisons"]
    lines.extend(
        [
            "",
            "## 10% shadow budget 生死比较",
            "",
            "| model | placement | MassCover CVaR95 | best baseline | baseline CVaR95 | relative gain | gate |",
            "|---|---|---:|---|---:|---:|---|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['model']} | {row['placement']} | {row['cvar_greedy']:.4f} | "
            f"{row['best_baseline_policy']} | {row['best_baseline']:.4f} | "
            f"{row['relative_improvement']:.1%} | {'PASS' if row['passes_10pct_gate'] else 'FAIL'} |"
        )

    lines.extend(
        [
            "",
            "## 判决解释",
            "",
            f"- Proxy gate：{'PASS' if verdict['proxy_gate']['passed'] else 'FAIL'}。",
            f"- Route-risk gate：{'PASS' if verdict['route_risk_gate']['passed'] else 'FAIL'}；按模型为 `{verdict['route_risk_gate']['model_pass']}`。",
            f"- **最终：{verdict['p0_verdict']}**。",
            "",
            "即使本 P0 通过，下一步仍必须直接 mask 整个失败域的 expert outputs，在相同 shadow HBM 下比较 frequency/CRAFT-like、gate-mass、random、MassCover 与 test oracle 的 KL/PPL/任务质量；随后才轮到 2–4 GPU 的 failure detection、mutable routing 和恢复时延。若端到端质量排序不复现 route proxy，方向立即死亡。",
            "",
            "## 输出",
            "",
            "- `risk_results.csv`：每模型、placement、policy、budget 的完整指标。",
            "- `placement_tradeoff.csv`：无 shadow 时的通信/故障风险。",
            "- `proxy_fidelity.csv`：gate 与真实 contribution 的小样本相关性。",
            "- `verdict.json`：冻结门槛与机器判决。",
            "- `manifest.json`：输入哈希与实验元数据。",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/idea_a_mac/outputs/masscover_ep_p0_2026-07-19"),
    )
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--gpus-per-failure-domain", type=int, default=4)
    parser.add_argument("--budget-fractions", default="0,0.02,0.05,0.10,0.20")
    parser.add_argument("--cvar-alpha", type=float, default=0.95)
    parser.add_argument("--random-seed-count", type=int, default=12)
    parser.add_argument("--target-budget", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    route_root = root / "experiments/idea_a_mac/outputs/route_fidelity_p0_2026-07-18"
    audit_root = root / "experiments/idea_a_mac/outputs/paper_validation"
    specs = [
        ModelSpec(
            key="olmoe_e64k8",
            model_name="allenai/OLMoE-1B-7B-0924",
            num_experts=64,
            top_k=8,
            calibration_routes=route_root / "p0b_calibration_olmoe/routes.csv",
            historical_routes=route_root / "p0b_sealed_olmoe/routes.csv",
            placement_registry=route_root / "p0b_placement_lock_v2/placements_olmoe.json",
            contribution_audit=audit_root / "olmoe_selector_causal_audit_2026-07-14/pair_interventions.csv",
        ),
        ModelSpec(
            key="llmjp_e32k16",
            model_name="llm-jp-MoE historical local artifact",
            num_experts=32,
            top_k=16,
            calibration_routes=route_root / "p0b_calibration_llmjp/routes.csv",
            historical_routes=route_root / "p0b_sealed_llmjp/routes.csv",
            placement_registry=route_root / "p0b_placement_lock_v2/placements_llmjp.json",
            contribution_audit=audit_root / "llmjp_top16_selector_causal_audit_2026-07-14/pair_interventions.csv",
        ),
    ]
    budgets = [float(value) for value in args.budget_fractions.split(",")]
    if not budgets or min(budgets) < 0 or max(budgets) > 1:
        raise ValueError("budget fractions must lie in [0, 1]")
    if args.target_budget not in budgets:
        raise ValueError("target budget must be included in budget fractions")
    if not 0 < args.cvar_alpha < 1:
        raise ValueError("CVaR alpha must lie in (0, 1)")

    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, object]] = []
    all_placements: list[dict[str, object]] = []
    model_metadata: list[dict[str, object]] = []
    fidelity_rows: list[dict[str, object]] = []
    for spec in specs:
        results, placements, metadata = evaluate_model(
            spec,
            output_dir=output_dir,
            ep_size=args.ep_size,
            gpus_per_failure_domain=args.gpus_per_failure_domain,
            budget_fractions=budgets,
            cvar_alpha=args.cvar_alpha,
            random_seed_count=args.random_seed_count,
        )
        all_results.extend(results)
        all_placements.extend(placements)
        model_metadata.append(metadata)
        fidelity_rows.append(proxy_fidelity(spec.contribution_audit, spec.key))

    results_table = pd.DataFrame(all_results)
    random_summary = aggregate_random(results_table)
    published_results = pd.concat(
        [results_table[results_table["policy_family"] != "random"], random_summary],
        ignore_index=True,
        sort=False,
    )
    placements_table = pd.DataFrame(all_placements)
    fidelity_table = pd.DataFrame(fidelity_rows)
    verdict = make_verdict(
        results_table,
        fidelity_table,
        target_budget=args.target_budget,
    )

    published_results.to_csv(output_dir / "risk_results.csv", index=False)
    placements_table.to_csv(output_dir / "placement_tradeoff.csv", index=False)
    fidelity_table.to_csv(output_dir / "proxy_fidelity.csv", index=False)
    (output_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "schema": SCHEMA,
        "evidence_level": "historical retrospective route-level P0",
        "ep_size": args.ep_size,
        "gpus_per_failure_domain": args.gpus_per_failure_domain,
        "budget_fractions": budgets,
        "cvar_alpha": args.cvar_alpha,
        "random_seed_count": args.random_seed_count,
        "target_budget": args.target_budget,
        "models": model_metadata,
        "limitations": [
            "old sealed captures are historical test, not newly sealed holdout",
            "gate share is a deployable proxy, not exact output contribution",
            "whole-domain failure is simulated only in logical route space",
            "shadow-copy count is not measured HBM or recovery cost",
            "no backend, GPU recovery, TTFT, TPOT, P99, energy, or task-quality evidence",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(output_dir, verdict, placements_table, fidelity_table, published_results)
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()

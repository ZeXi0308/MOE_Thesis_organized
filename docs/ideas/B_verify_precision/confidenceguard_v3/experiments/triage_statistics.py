"""Document-level paired statistics and executable Gate-M verdict."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np


class StatisticsError(RuntimeError):
    pass


PRIMARY_POLICIES = (
    "triage_2_4_8",
    "hash_budget_matched_2_4_8",
    "fixed_2",
    "fixed_4",
    "fixed_8",
)


def _interval(values: np.ndarray, point: float) -> dict[str, float]:
    return {
        "point": float(point),
        "lcb": float(np.quantile(values, 0.025)),
        "ucb": float(np.quantile(values, 0.975)),
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, name in enumerate(ordered):
        value = min(1.0, (count - index) * float(p_values[name]))
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def analyze_model(
    rows: Sequence[Mapping[str, object]],
    *,
    bootstrap_repeats: int,
    seed: int,
) -> dict[str, object]:
    if bootstrap_repeats < 100:
        raise StatisticsError("at least 100 bootstraps are required")
    by_policy: dict[str, dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        policy = str(row.get("policy"))
        document = str(row.get("text_sha256"))
        if policy not in PRIMARY_POLICIES or len(document) != 64:
            continue
        if document in by_policy[policy]:
            raise StatisticsError(f"duplicate document/policy row: {policy}/{document}")
        by_policy[policy][document] = row
    if set(by_policy) != set(PRIMARY_POLICIES):
        raise StatisticsError("missing primary policy")
    documents = sorted(by_policy[PRIMARY_POLICIES[0]])
    if not documents or any(sorted(by_policy[policy]) != documents for policy in PRIMARY_POLICIES):
        raise StatisticsError("policy document sets do not close")

    fields = (
        "document_cvar90_kl",
        "total_candidate_forward_calls",
        "dangerous_step_recall",
        "threshold_violation_fraction",
    )
    matrices: dict[str, dict[str, np.ndarray]] = {}
    for policy in PRIMARY_POLICIES:
        matrices[policy] = {}
        for field in fields:
            values = np.asarray([float(by_policy[policy][doc][field]) for doc in documents])
            if not np.isfinite(values).all():
                raise StatisticsError(f"non-finite {field} in {policy}")
            if field in {"document_cvar90_kl", "total_candidate_forward_calls"} and np.any(values < 0):
                raise StatisticsError(f"negative {field} in {policy}")
            if field in {"dangerous_step_recall", "threshold_violation_fraction"} and (
                np.any(values < 0) or np.any(values > 1)
            ):
                raise StatisticsError(f"out-of-range {field} in {policy}")
            matrices[policy][field] = values

    triage = matrices["triage_2_4_8"]
    hashed = matrices["hash_budget_matched_2_4_8"]

    def metrics(indices: np.ndarray) -> tuple[float, float, float, float, float]:
        tq = triage["document_cvar90_kl"][indices]
        hq = hashed["document_cvar90_kl"][indices]
        quality_ratio = float(np.median((tq + 1e-12) / (hq + 1e-12)))
        tc = triage["total_candidate_forward_calls"][indices]
        hc = hashed["total_candidate_forward_calls"][indices]
        forward_reduction = float(np.median(1.0 - tc / np.maximum(hc, 1e-12)))
        baseline_policies = ("hash_budget_matched_2_4_8", "fixed_2", "fixed_4", "fixed_8")
        triage_quality_mean = float(tq.mean())
        triage_cost_mean = float(tc.mean())
        baseline_points = [
            (
                float(matrices[policy]["document_cvar90_kl"][indices].mean()),
                float(matrices[policy]["total_candidate_forward_calls"][indices].mean()),
            )
            for policy in baseline_policies
        ]
        eligible_costs = [
            cost for quality, cost in baseline_points
            if quality <= triage_quality_mean * (1.0 + 1e-12)
        ]
        # If triage is better-quality than all baseline points, require it to
        # beat the cheapest baseline anyway; do not grant an automatic pass.
        comparison_cost = min(eligible_costs or [cost for _, cost in baseline_points])
        pareto_reduction = 1.0 - triage_cost_mean / max(comparison_cost, 1e-12)
        recall_difference = float(np.median(
            triage["dangerous_step_recall"][indices] - hashed["dangerous_step_recall"][indices]
        ))
        violation_difference = float(np.median(
            triage["threshold_violation_fraction"][indices] - hashed["threshold_violation_fraction"][indices]
        ))
        return quality_ratio, forward_reduction, pareto_reduction, recall_difference, violation_difference

    all_indices = np.arange(len(documents))
    point = metrics(all_indices)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty((bootstrap_repeats, 5), dtype=np.float64)
    for repeat in range(bootstrap_repeats):
        indices = rng.integers(0, len(documents), size=len(documents))
        bootstrap[repeat] = metrics(indices)
    names = (
        "quality_ratio_vs_hash",
        "forward_reduction_vs_hash",
        "pareto_forward_reduction",
        "dangerous_recall_difference",
        "violation_fraction_difference",
    )
    intervals = {name: _interval(bootstrap[:, index], point[index]) for index, name in enumerate(names)}
    # One-sided null-centered bootstrap tests. Directly counting how often a
    # confidence bootstrap crosses a boundary is not a valid p-value.
    nulls = np.asarray([1.05, 0.10, 0.10, -0.05, 0.0], dtype=np.float64)
    observed = np.asarray(point, dtype=np.float64)
    null_centered = bootstrap - observed[np.newaxis, :] + nulls[np.newaxis, :]
    denominator = bootstrap_repeats + 1
    raw_p = {
        "quality_noninferiority": float((1 + np.sum(null_centered[:, 0] <= observed[0])) / denominator),
        "forward_reduction": float((1 + np.sum(null_centered[:, 1] >= observed[1])) / denominator),
        "pareto_reduction": float((1 + np.sum(null_centered[:, 2] >= observed[2])) / denominator),
        "dangerous_recall": float((1 + np.sum(null_centered[:, 3] >= observed[3])) / denominator),
        "violation_nonincrease": float((1 + np.sum(null_centered[:, 4] <= observed[4])) / denominator),
    }
    adjusted = holm_adjust(raw_p)
    checks = {
        "quality_noninferiority": intervals["quality_ratio_vs_hash"]["ucb"] <= 1.05,
        "forward_reduction": intervals["forward_reduction_vs_hash"]["lcb"] >= 0.10,
        "pareto_reduction": intervals["pareto_forward_reduction"]["lcb"] >= 0.10,
        "dangerous_recall": intervals["dangerous_recall_difference"]["lcb"] >= -0.05,
        "violation_nonincrease": intervals["violation_fraction_difference"]["ucb"] <= 0.0,
        "holm_all": all(value <= 0.05 for value in adjusted.values()),
    }
    return {
        "documents": len(documents),
        "bootstrap_repeats": bootstrap_repeats,
        "effect_estimator": "median_of_document_level_paired_effects",
        "pareto_effect_estimator": "policy_level_mean_points_rebuilt_per_document_bootstrap",
        "intervals": intervals,
        "raw_p_values": raw_p,
        "holm_adjusted_p_values": adjusted,
        "checks": checks,
        "model_go": all(checks.values()),
    }


def cross_model_decision(model_results: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    if set(model_results) != {"olmoe", "llmjp"}:
        raise StatisticsError("cross-model decision requires exactly olmoe and llmjp")
    llmjp_go = bool(model_results["llmjp"].get("model_go"))
    olmoe_checks = model_results["olmoe"].get("checks")
    if not isinstance(olmoe_checks, Mapping):
        raise StatisticsError("OLMoE checks missing")
    olmoe_safe = all(
        bool(olmoe_checks.get(name))
        for name in ("quality_noninferiority", "dangerous_recall", "violation_nonincrease")
    )
    go = llmjp_go and olmoe_safe
    return {
        "llmjp_go": llmjp_go,
        "olmoe_safe_degradation": olmoe_safe,
        "go": go,
        "verdict": "GO_TO_NATIVE_LOW_PRECISION_GATE_S" if go else "NO_GO_PREFILL_RISK_RANKING_FOR_AUDIT_ALLOCATION",
    }

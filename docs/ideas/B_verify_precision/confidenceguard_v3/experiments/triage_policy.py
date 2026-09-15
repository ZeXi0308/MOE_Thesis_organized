"""Pure policy/statistics core for the TriageAudit-MoE mechanism probe."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence

import numpy as np


FEATURE_NAMES = (
    "full_route_top1_weight_mean",
    "full_route_top1_weight_std",
    "full_route_top1_top2_margin_mean",
    "full_route_tail_mass_mean",
    "full_route_routing_entropy_mean",
    "full_route_rank1_hhi_mean",
    "full_route_active_expert_fraction_mean",
    "full_route_same_id_adjacent_layer_rate",
    "full_mean_nll",
)


class TriagePolicyError(RuntimeError):
    pass


def _finite_vector(row: Mapping[str, float]) -> np.ndarray:
    missing = [name for name in FEATURE_NAMES if name not in row]
    if missing:
        raise TriagePolicyError(f"missing predictor features: {missing}")
    vector = np.asarray([row[name] for name in FEATURE_NAMES], dtype=np.float64)
    if vector.shape != (len(FEATURE_NAMES),) or not np.isfinite(vector).all():
        raise TriagePolicyError("predictor features must be finite scalars")
    return vector


def cvar(values: Sequence[float], fraction: float = 0.1) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise TriagePolicyError("CVaR input must be a non-empty finite vector")
    if not 0.0 < fraction <= 1.0:
        raise TriagePolicyError("CVaR fraction must be in (0, 1]")
    count = max(1, int(np.ceil(len(array) * fraction)))
    return float(np.sort(array)[-count:].mean())


@dataclass(frozen=True)
class FrozenRidgeTriage:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    lower_cut: float
    upper_cut: float

    def __post_init__(self) -> None:
        width = len(FEATURE_NAMES)
        for name, value in (
            ("mean", self.mean),
            ("scale", self.scale),
            ("coefficients", self.coefficients),
        ):
            if len(value) != width or not np.isfinite(np.asarray(value)).all():
                raise TriagePolicyError(f"invalid frozen ridge {name}")
        if np.any(np.asarray(self.scale) <= 0):
            raise TriagePolicyError("frozen ridge scales must be positive")
        if not np.isfinite([self.intercept, self.lower_cut, self.upper_cut]).all():
            raise TriagePolicyError("frozen ridge scalars must be finite")
        if self.lower_cut > self.upper_cut:
            raise TriagePolicyError("lower_cut must not exceed upper_cut")

    def score(self, row: Mapping[str, float]) -> float:
        x = _finite_vector(row)
        standardized = (x - np.asarray(self.mean)) / np.asarray(self.scale)
        return float(self.intercept + standardized @ np.asarray(self.coefficients))

    def period(self, row: Mapping[str, float]) -> int:
        value = self.score(row)
        if value < self.lower_cut:
            return 8
        if value < self.upper_cut:
            return 4
        return 2

    def to_dict(self) -> dict:
        return {
            "schema_version": "triage-ridge-v1",
            "feature_names": list(FEATURE_NAMES),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "lower_cut": self.lower_cut,
            "upper_cut": self.upper_cut,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "FrozenRidgeTriage":
        if value.get("schema_version") != "triage-ridge-v1":
            raise TriagePolicyError("wrong frozen ridge schema")
        if value.get("feature_names") != list(FEATURE_NAMES):
            raise TriagePolicyError("frozen ridge feature order mismatch")

        def numbers(name: str) -> tuple[float, ...]:
            raw = value.get(name)
            if not isinstance(raw, list) or any(type(item) not in (int, float) for item in raw):
                raise TriagePolicyError(f"{name} must be a numeric list")
            return tuple(float(item) for item in raw)

        for scalar in ("intercept", "lower_cut", "upper_cut"):
            if type(value.get(scalar)) not in (int, float):
                raise TriagePolicyError(f"{scalar} must be numeric")
        return cls(
            mean=numbers("mean"),
            scale=numbers("scale"),
            coefficients=numbers("coefficients"),
            intercept=float(value["intercept"]),
            lower_cut=float(value["lower_cut"]),
            upper_cut=float(value["upper_cut"]),
        )


@dataclass(frozen=True)
class FrozenConfidenceGuard:
    point_model: FrozenRidgeTriage
    bootstrap_models: tuple[FrozenRidgeTriage, ...]
    safe_cuts: tuple[float, ...]
    safe_probability_min: float
    risk_probability_max: float

    def __post_init__(self) -> None:
        if len(self.bootstrap_models) < 100 or len(self.bootstrap_models) != len(self.safe_cuts):
            raise TriagePolicyError("confidence guard needs aligned bootstrap models and cuts")
        if not np.isfinite(np.asarray(self.safe_cuts, dtype=np.float64)).all():
            raise TriagePolicyError("confidence guard cuts must be finite")
        if not (
            0.0 <= self.risk_probability_max < self.safe_probability_min <= 1.0
            and self.risk_probability_max < 0.5 < self.safe_probability_min
        ):
            raise TriagePolicyError("confidence guard probability thresholds are invalid")

    def safe_probability(self, row: Mapping[str, float]) -> float:
        safe_votes = sum(
            model.score(row) <= cut
            for model, cut in zip(self.bootstrap_models, self.safe_cuts)
        )
        return safe_votes / len(self.bootstrap_models)

    def period(self, row: Mapping[str, float]) -> int:
        probability = self.safe_probability(row)
        if probability >= self.safe_probability_min:
            return 8
        if probability <= self.risk_probability_max:
            return 2
        return 4

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "confidence-guard-v3",
            "feature_names": list(FEATURE_NAMES),
            "point_model": self.point_model.to_dict(),
            "bootstrap_models": [model.to_dict() for model in self.bootstrap_models],
            "safe_cuts": list(self.safe_cuts),
            "safe_probability_min": self.safe_probability_min,
            "risk_probability_max": self.risk_probability_max,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "FrozenConfidenceGuard":
        if value.get("schema_version") != "confidence-guard-v3":
            raise TriagePolicyError("wrong confidence guard schema")
        if value.get("feature_names") != list(FEATURE_NAMES):
            raise TriagePolicyError("confidence guard feature order mismatch")
        point = value.get("point_model")
        models = value.get("bootstrap_models")
        cuts = value.get("safe_cuts")
        if not isinstance(point, Mapping) or not isinstance(models, list) or not isinstance(cuts, list):
            raise TriagePolicyError("confidence guard model payload is malformed")
        if any(not isinstance(model, Mapping) for model in models):
            raise TriagePolicyError("confidence guard bootstrap model is malformed")
        if any(type(cut) not in (int, float) for cut in cuts):
            raise TriagePolicyError("confidence guard cuts must be numeric")
        for name in ("safe_probability_min", "risk_probability_max"):
            if type(value.get(name)) not in (int, float):
                raise TriagePolicyError(f"{name} must be numeric")
        return cls(
            point_model=FrozenRidgeTriage.from_dict(point),
            bootstrap_models=tuple(FrozenRidgeTriage.from_dict(model) for model in models),
            safe_cuts=tuple(float(cut) for cut in cuts),
            safe_probability_min=float(value["safe_probability_min"]),
            risk_probability_max=float(value["risk_probability_max"]),
        )


def fit_frozen_ridge(
    feature_rows: Sequence[Mapping[str, float]],
    document_cvar90: Sequence[float],
    alpha: float = 1.0,
) -> FrozenRidgeTriage:
    if len(feature_rows) != len(document_cvar90) or len(feature_rows) < len(FEATURE_NAMES) + 2:
        raise TriagePolicyError("insufficient or mismatched calibration rows")
    if type(alpha) not in (int, float) or alpha < 0 or not np.isfinite(alpha):
        raise TriagePolicyError("ridge alpha must be finite and non-negative")
    x = np.stack([_finite_vector(row) for row in feature_rows])
    y_raw = np.asarray(document_cvar90, dtype=np.float64)
    if np.any(y_raw < 0) or not np.isfinite(y_raw).all():
        raise TriagePolicyError("calibration labels must be finite and non-negative")
    y = np.log10(y_raw + 1e-12)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    # ``einsum`` avoids a noisy Accelerate/BLAS matmul warning observed on the
    # project Mac for perfectly collinear tiny dry-run matrices.  It computes
    # the same normal equation and is negligible at this nine-feature scale.
    normal_matrix = np.einsum("ni,nj->ij", design, design) + penalty
    rhs = np.einsum("ni,n->i", design, y)
    try:
        beta = np.linalg.solve(normal_matrix, rhs)
    except np.linalg.LinAlgError as exc:
        raise TriagePolicyError("ridge normal equation is singular") from exc
    if not np.isfinite(beta).all():
        raise TriagePolicyError("ridge fit produced non-finite coefficients")
    scores = design @ beta
    lower, upper = np.quantile(scores, [1.0 / 3.0, 2.0 / 3.0])
    return FrozenRidgeTriage(
        mean=tuple(float(v) for v in mean),
        scale=tuple(float(v) for v in scale),
        coefficients=tuple(float(v) for v in beta[1:]),
        intercept=float(beta[0]),
        lower_cut=float(lower),
        upper_cut=float(upper),
    )


def hash_control_period(document_sha256: str) -> int:
    if len(document_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in document_sha256):
        raise TriagePolicyError("document_sha256 must be lowercase hex")
    return (8, 4, 2)[int(document_sha256[:16], 16) % 3]


def common_audit_phase(document_sha256: str, period: int) -> int:
    """Policy-independent phase for a document/period pair."""
    if len(document_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in document_sha256):
        raise TriagePolicyError("document_sha256 must be lowercase hex")
    if period <= 0:
        raise TriagePolicyError("period must be positive")
    payload = f"audit-phase-v2|{document_sha256}|{period}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % period


def budget_matched_hash_periods(
    document_sha256s: Sequence[str],
    triage_periods: Sequence[int],
    *,
    model_key: str,
    split: str,
) -> dict[str, int]:
    """Remove risk-document correspondence while preserving exact budget.

    The hash arm receives the exact period multiset produced by triage. Only
    the mapping from period to document is changed.
    """
    if len(document_sha256s) != len(triage_periods) or len(set(document_sha256s)) != len(document_sha256s):
        raise TriagePolicyError("budget matching requires aligned unique documents")
    if not model_key or not split:
        raise TriagePolicyError("model_key and split are required")
    allowed = {2, 4, 8}
    if any(period not in allowed for period in triage_periods):
        raise TriagePolicyError("triage periods must be in {2,4,8}")
    for digest in document_sha256s:
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise TriagePolicyError("document_sha256 must be lowercase hex")
    ordered_documents = sorted(
        document_sha256s,
        key=lambda digest: hashlib.sha256(
            f"budget-control|{model_key}|{split}|{digest}".encode("utf-8")
        ).digest(),
    )
    ordered_periods = sorted(int(period) for period in triage_periods)
    result = dict(zip(ordered_documents, ordered_periods))
    if sorted(result.values()) != sorted(triage_periods):
        raise AssertionError("period multiset did not close")
    return result


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise TriagePolicyError("invalid Spearman inputs")
    rank_a = _rankdata(a)
    rank_b = _rankdata(b)
    if rank_a.std() <= 1e-12 or rank_b.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def calibration_stability(
    feature_rows: Sequence[Mapping[str, float]],
    document_cvar90: Sequence[float],
    *,
    alpha: float,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    if repeats < 100:
        raise TriagePolicyError("calibration stability needs at least 100 bootstraps")
    frozen = fit_frozen_ridge(feature_rows, document_cvar90, alpha=alpha)
    full_periods = np.asarray([frozen.period(row) for row in feature_rows], dtype=np.int64)
    full_scores = np.asarray([frozen.score(row) for row in feature_rows], dtype=np.float64)
    labels = np.asarray(document_cvar90, dtype=np.float64)
    rng = np.random.default_rng(seed)
    same = np.zeros(len(feature_rows), dtype=np.int64)
    rhos = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        indices = rng.integers(0, len(feature_rows), size=len(feature_rows))
        sampled_rows = [feature_rows[int(index)] for index in indices]
        sampled_labels = [float(labels[int(index)]) for index in indices]
        model = fit_frozen_ridge(sampled_rows, sampled_labels, alpha=alpha)
        periods = np.asarray([model.period(row) for row in feature_rows], dtype=np.int64)
        same += periods == full_periods
        sampled_scores = full_scores[indices]
        rhos[repeat] = spearman(sampled_scores, labels[indices])
    probabilities = same.astype(np.float64) / repeats
    return {
        "frozen_model": frozen.to_dict(),
        "assignment_probabilities": probabilities.tolist(),
        "median_assignment_probability": float(np.median(probabilities)),
        "fraction_documents_probability_ge_0_6": float(np.mean(probabilities >= 0.6)),
        "spearman_point": spearman(full_scores, labels),
        "spearman_lcb": float(np.quantile(rhos, 0.025)),
        "spearman_ucb": float(np.quantile(rhos, 0.975)),
        "bootstrap_repeats": repeats,
    }


def confidence_guard_stability(
    feature_rows: Sequence[Mapping[str, float]],
    document_cvar90: Sequence[float],
    *,
    alpha: float,
    repeats: int,
    seed: int,
    safe_probability_min: float,
    risk_probability_max: float,
) -> dict[str, object]:
    """Fit a frozen selective guard and audit binary assignment stability.

    Each bootstrap model defines ``safe`` using its median score on the full
    calibration feature set.  This preserves a 50/50 binary budget while the
    bootstrap is estimating assignment uncertainty.  New documents abstain to
    period 4 unless their safe-vote posterior crosses a frozen threshold.
    """
    if repeats < 100:
        raise TriagePolicyError("confidence guard needs at least 100 bootstraps")
    point_model = fit_frozen_ridge(feature_rows, document_cvar90, alpha=alpha)
    full_scores = np.asarray([point_model.score(row) for row in feature_rows], dtype=np.float64)
    full_safe = full_scores <= float(np.median(full_scores))
    labels = np.asarray(document_cvar90, dtype=np.float64)
    rng = np.random.default_rng(seed)
    same = np.zeros(len(feature_rows), dtype=np.int64)
    safe_votes = np.zeros(len(feature_rows), dtype=np.int64)
    rhos = np.empty(repeats, dtype=np.float64)
    models: list[FrozenRidgeTriage] = []
    cuts: list[float] = []
    for repeat in range(repeats):
        indices = rng.integers(0, len(feature_rows), size=len(feature_rows))
        model = fit_frozen_ridge(
            [feature_rows[int(index)] for index in indices],
            [float(labels[int(index)]) for index in indices],
            alpha=alpha,
        )
        scores = np.asarray([model.score(row) for row in feature_rows], dtype=np.float64)
        cut = float(np.median(scores))
        safe = scores <= cut
        same += safe == full_safe
        safe_votes += safe
        rhos[repeat] = spearman(full_scores[indices], labels[indices])
        models.append(model)
        cuts.append(cut)
    probabilities = same.astype(np.float64) / repeats
    guard = FrozenConfidenceGuard(
        point_model=point_model,
        bootstrap_models=tuple(models),
        safe_cuts=tuple(cuts),
        safe_probability_min=safe_probability_min,
        risk_probability_max=risk_probability_max,
    )
    return {
        "frozen_guard": guard.to_dict(),
        "binary_assignment_probabilities": probabilities.tolist(),
        "calibration_safe_probabilities": (safe_votes.astype(np.float64) / repeats).tolist(),
        "median_binary_assignment_probability": float(np.median(probabilities)),
        "fraction_documents_probability_ge_0_6": float(np.mean(probabilities >= 0.6)),
        "spearman_point": spearman(full_scores, labels),
        "spearman_lcb": float(np.quantile(rhos, 0.025)),
        "spearman_ucb": float(np.quantile(rhos, 0.975)),
        "bootstrap_repeats": repeats,
    }


def audit_phase(policy_name: str, document_sha256: str, period: int) -> int:
    if not policy_name or period <= 0:
        raise TriagePolicyError("policy_name and positive period are required")
    payload = f"{policy_name}|{document_sha256}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % period


@dataclass
class AuditState:
    period: int
    phase: int
    max_unaudited_steps: int = 8
    lockout_following_steps: int = 3
    steps_since_audit: int = 0
    lockout_remaining: int = 0
    audit_events: int = 0
    high_forward_calls: int = 0
    low_forward_calls: int = 0
    cache_clone_events: int = 0
    served_high_steps: int = 0
    served_low_steps: int = 0
    lockout_steps: int = 0

    def __post_init__(self) -> None:
        if self.period <= 0 or not 0 <= self.phase < self.period:
            raise TriagePolicyError("invalid audit period/phase")
        if self.max_unaudited_steps <= 0 or self.period > self.max_unaudited_steps:
            raise TriagePolicyError("period exceeds maximum unaudited interval")
        if self.lockout_following_steps < 0:
            raise TriagePolicyError("negative lockout is invalid")

    def decision(self, step: int) -> str:
        if step < 0:
            raise TriagePolicyError("step must be non-negative")
        if self.lockout_remaining > 0:
            return "lockout_high"
        scheduled = step % self.period == self.phase
        forced = self.steps_since_audit >= self.max_unaudited_steps - 1
        return "audit" if scheduled or forced else "low"

    def record_audit(self, discrepancy: float, threshold: float) -> str:
        if discrepancy < 0 or not np.isfinite(discrepancy) or threshold < 0 or not np.isfinite(threshold):
            raise TriagePolicyError("audit discrepancy/threshold must be finite and non-negative")
        self.audit_events += 1
        self.high_forward_calls += 1
        self.low_forward_calls += 1
        self.cache_clone_events += 2
        self.steps_since_audit = 0
        if discrepancy > threshold:
            self.served_high_steps += 1
            self.lockout_remaining = self.lockout_following_steps
            return "high"
        self.served_low_steps += 1
        return "low"

    def record_single(self, action: str) -> None:
        if action == "lockout_high":
            if self.lockout_remaining <= 0:
                raise TriagePolicyError("lockout counter underflow")
            self.high_forward_calls += 1
            self.served_high_steps += 1
            self.lockout_steps += 1
            self.lockout_remaining -= 1
        elif action == "low":
            self.low_forward_calls += 1
            self.served_low_steps += 1
        else:
            raise TriagePolicyError(f"invalid single action: {action}")
        self.steps_since_audit += 1

    def counters(self) -> dict[str, int]:
        return {
            "audit_events": self.audit_events,
            "high_forward_calls": self.high_forward_calls,
            "low_forward_calls": self.low_forward_calls,
            "total_candidate_forward_calls": self.high_forward_calls + self.low_forward_calls,
            "cache_clone_events": self.cache_clone_events,
            "served_high_steps": self.served_high_steps,
            "served_low_steps": self.served_low_steps,
            "lockout_steps": self.lockout_steps,
        }

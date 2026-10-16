#!/usr/bin/env python3
"""Working-set predictions and a measured-table/EWMA gamma selector.

Only the analytic CLI produces predictions. It never labels them measurements.
``q[e]`` is P(expert e is included in a token's top-k set), sum(q)=k.
Independence is assumed BETWEEN tokens, not between the k routes of a token.
Logical expert-weight working-set bytes are NOT measured HBM traffic.

Runtime API (standard library only)::

    table = CalibrationTable.from_json("calibration.json")
    acceptance = AcceptanceEWMA(alpha=0.7)
    policy = AdaptiveGammaPolicy(table, acceptance)
    decision = policy.choose(batch=8, candidates=[0, 1, 3, 7])
    acceptance.update(accepted_lengths=[2, 0], proposed_lengths=[3, 3])

For truncation AFTER generating a full draft, pass ``draft_gamma=7`` to
``choose``. This uses budget-specific timings, or replaces the draft cost
with the FULL gamma=7 draft cost; it does not pretend truncation saves drafting.
Calibration must describe the same model, device, workload/context regime,
draft method and timing boundary. Do not pool these regimes into one table.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def _probabilities(values: Sequence[float]) -> list[float]:
    q = [float(v) for v in values]
    if not q or any(not math.isfinite(v) or not 0 <= v <= 1 for v in q):
        raise ValueError("probabilities must be a nonempty finite vector in [0,1]")
    return q


def _slots(batch: int, gamma: int) -> int:
    if int(batch) != batch or batch < 1 or int(gamma) != gamma or gamma < 0:
        raise ValueError("batch must be a positive integer; gamma a nonnegative integer")
    return int(batch) * (int(gamma) + 1)


def _occupancy(probability: float, count: int) -> float:
    return 1.0 if probability == 1 else -math.expm1(count * math.log1p(-probability))


def expected_distinct(q_e: Sequence[float], batch: int, gamma: int) -> float:
    """Sum[1-(1-q_e)**(B*(gamma+1))], assuming independent token sets.

    For gamma=0,B=1 this is exactly sum(q_e)=k, even though top-k routes
    within a token are sampled without replacement.
    """
    slots = _slots(batch, gamma)
    return math.fsum(_occupancy(q, slots) for q in _probabilities(q_e))


def expected_distinct_slot_approx(p_e: Sequence[float], batch: int, gamma: int,
                                  top_k: int) -> float:
    """Legacy IID route-slot approximation; generally undercounts B=1,gamma=0."""
    if top_k < 1 or int(top_k) != top_k:
        raise ValueError("top_k must be positive")
    p = _probabilities(p_e)
    if not math.isclose(sum(p), 1.0, abs_tol=1e-6):
        raise ValueError("slot marginal p_e must sum to one")
    return math.fsum(_occupancy(v, _slots(batch, gamma) * top_k) for v in p)


def expected_tokens_geometric(alpha: float, gamma: int) -> float:
    """1 + sum(alpha**j,j=1..gamma), an IID acceptance approximation.

    Assumes one correction/bonus token and no EOS/output-budget truncation.
    """
    _slots(1, gamma)
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0,1]")
    return math.fsum(alpha ** j for j in range(gamma + 1))


@dataclass
class AcceptanceEWMA:
    """Empirical accepted-prefix survival with a geometric prior for unseen ranks.

    A rank j is exposed only when proposed_length >= j. P(L>=j) is updated
    from this eligible population, including earlier rejections as failures.
    The scalar alpha uses tested positions, not accepted/proposed: positions
    beyond the first rejection are untested. EOS-censored observations should
    be omitted by the caller. Decisions from mixed workloads remain approximate.
    """
    alpha: float = 0.7
    decay: float = 0.9

    def __post_init__(self) -> None:
        expected_tokens_geometric(self.alpha, 0)
        if not 0 <= self.decay < 1:
            raise ValueError("decay must be in [0,1)")
        self.survival: dict[int, float] = {}
        self.exposures: dict[int, int] = defaultdict(int)
        self.update_count = 0
        self.observation_count = 0

    def update(self, accepted_lengths: Sequence[int], proposed_lengths: Sequence[int]) -> None:
        if len(accepted_lengths) != len(proposed_lengths):
            raise ValueError("accepted and proposed arrays must have equal length")
        pairs = list(zip(accepted_lengths, proposed_lengths))
        if any(int(a) != a or int(p) != p or a < 0 or p < a for a, p in pairs):
            raise ValueError("each observation must satisfy 0 <= accepted <= proposed")
        pairs = [(int(a), int(p)) for a, p in pairs if p > 0]
        if not pairs:
            return
        tested = sum(min(a + 1, p) for a, p in pairs)
        new_alpha = sum(a for a, _ in pairs) / tested
        self.alpha = self.decay * self.alpha + (1 - self.decay) * new_alpha
        for rank in range(1, max(p for _, p in pairs) + 1):
            eligible = [a for a, p in pairs if p >= rank]
            observed = sum(a >= rank for a in eligible) / len(eligible)
            prior = self.survival.get(rank, self.alpha ** rank)
            self.survival[rank] = self.decay * prior + (1 - self.decay) * observed
            self.exposures[rank] += len(eligible)
        self.update_count += 1
        self.observation_count += len(pairs)

    def prefix_survival(self, gamma: int) -> list[float]:
        _slots(1, gamma)
        values, previous = [], 1.0
        for rank in range(1, gamma + 1):
            # Sampling populations can differ across ranks; monotone projection
            # prevents a nonphysical survival curve without inventing exposures.
            value = min(previous, self.survival.get(rank, previous * self.alpha))
            values.append(value)
            previous = value
        return values

    def expected_tokens(self, gamma: int) -> float:
        return 1.0 + math.fsum(self.prefix_survival(gamma))

    def snapshot(self) -> dict:
        return {"alpha": self.alpha, "decay": self.decay,
                "prefix_survival": dict(self.survival), "exposures": dict(self.exposures),
                "updates": self.update_count, "observations": self.observation_count,
                "unseen_rank_model": "geometric conditional continuation",
                "boundary": "one correction token; no terminal/output-cap censoring"}


@dataclass(frozen=True)
class CostEstimate:
    batch: int
    gamma: int
    total_step_ms: float
    verify_ms: float | None
    draft_ms: float | None
    control_ms: float | None
    source: str
    samples: int
    proposal_budget: int | None = None


def _number(record: Mapping, key: str, default=None):
    v = record.get(key)
    if v is None or v == "":
        return default
    value = float(v)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{key} must be finite and nonnegative")
    return value


class CalibrationTable:
    """Measured step times, median across repeats, linear interpolation in B.

    Rows: batch, gamma, total_step_ms OR verify_ms+draft_ms+control_ms.
    ``total_step_ms`` MUST include draft, verify, control and non-overlapped
    host gaps, with a consistent observed timing boundary. Component-only rows
    require ``components_nonoverlap=true`` (no double-counting GPU/CPU overlap).
    ``proposal_budget`` identifies full-budget drafting followed by truncation.
    Missing cells are not fabricated and no batch extrapolation is performed.
    """
    def __init__(self, records: Iterable[Mapping], metadata: Mapping | None = None):
        self.metadata = dict(metadata or {})
        grouped = defaultdict(list)
        regimes = defaultdict(set)
        for record in records:
            for key in ("model", "device", "draft_source", "workload", "context_regime"):
                if record.get(key) is not None:
                    regimes[key].add(str(record[key]))
            b, g = int(record["batch"]), int(record["gamma"])
            _slots(b, g)
            budget = record.get("proposal_budget")
            budget = None if budget in (None, "") else int(budget)
            if budget is not None and budget < g:
                raise ValueError("proposal_budget cannot be smaller than gamma")
            total = _number(record, "total_step_ms")
            verify = _number(record, "verify_ms")
            draft = _number(record, "draft_ms")
            control = _number(record, "control_ms", 0.0)
            if total is None:
                if verify is None or draft is None:
                    raise ValueError("need total_step_ms or explicit verify_ms and draft_ms")
                if str(record.get("components_nonoverlap", "")).lower() not in ("true", "1"):
                    raise ValueError("component sums require components_nonoverlap=true")
                total = verify + draft + control
            if total <= 0:
                raise ValueError("total_step_ms must be positive")
            if draft is not None and draft > total + 1e-9:
                raise ValueError("draft_ms exceeds complete-step total_step_ms")
            grouped[(b, g, budget)].append((total, verify, draft, control))
        if any(len(values) > 1 for values in regimes.values()):
            raise ValueError(f"mixed calibration regimes: {dict(regimes)}")
        if not grouped:
            raise ValueError("calibration table is empty")
        self.cells: dict[tuple[int, int, int | None], CostEstimate] = {}
        self.online_updates: dict[tuple[int, int, int | None], int] = defaultdict(int)
        for (b, g, budget), rows in grouped.items():
            medians = [statistics.median(r[i] for r in rows)
                       if all(r[i] is not None for r in rows) else None for i in range(4)]
            self.cells[(b, g, budget)] = CostEstimate(
                b, g, *medians, "measured_cell_median", len(rows), budget)

    @classmethod
    def from_json(cls, path: str | Path) -> "CalibrationTable":
        obj = json.loads(Path(path).read_text())
        return cls(obj if isinstance(obj, list) else obj["records"],
                   {} if isinstance(obj, list) else obj.get("metadata", {}))

    @classmethod
    def from_csv(cls, path: str | Path) -> "CalibrationTable":
        with Path(path).open(newline="") as f:
            return cls(csv.DictReader(f))

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationTable":
        return cls.from_csv(path) if Path(path).suffix == ".csv" else cls.from_json(path)

    @property
    def gammas(self) -> list[int]:
        return sorted({g for _, g, _ in self.cells})

    def lookup(self, batch: int, gamma: int, proposal_budget: int | None = None) -> CostEstimate:
        _slots(batch, gamma)
        key = (batch, gamma, proposal_budget)
        if key in self.cells:
            return self.cells[key]
        available = sorted(b for b, g, p in self.cells if g == gamma and p == proposal_budget)
        left = [b for b in available if b < batch]
        right = [b for b in available if b > batch]
        if not left or not right:
            raise KeyError(f"uncalibrated batch/gamma/budget {key}; extrapolation disabled")
        lo = self.cells[(max(left), gamma, proposal_budget)]
        hi = self.cells[(min(right), gamma, proposal_budget)]
        weight = (batch - lo.batch) / (hi.batch - lo.batch)
        def mix(name):
            a, b = getattr(lo, name), getattr(hi, name)
            return None if a is None or b is None else a + weight * (b - a)
        return CostEstimate(batch, gamma, *(mix(n) for n in
            ("total_step_ms", "verify_ms", "draft_ms", "control_ms")),
            "batch_linear_interpolation", min(lo.samples, hi.samples), proposal_budget)

    def estimate(self, batch: int, gamma: int, draft_gamma: int | None = None) -> CostEstimate:
        if draft_gamma is None:
            return self.lookup(batch, gamma)
        if draft_gamma < gamma:
            raise ValueError("draft_gamma cannot be smaller than chosen gamma")
        try:
            return self.lookup(batch, gamma, proposal_budget=draft_gamma)
        except KeyError:
            pass
        selected = self.lookup(batch, gamma)
        full = self.lookup(batch, draft_gamma)
        if gamma == draft_gamma:
            return selected
        if selected.draft_ms is None or full.draft_ms is None:
            raise KeyError("post-proposal truncation requires budget-specific complete timings "
                           "or non-overlapped draft_ms for each candidate and full budget")
        # This replacement requires isolated/non-overlapped draft timing, stated
        # explicitly at table level. It is a model estimate, not a measured cell.
        if not self.metadata.get("draft_cost_nonoverlap", False):
            raise KeyError("draft-cost replacement requires metadata.draft_cost_nonoverlap=true")
        total = selected.total_step_ms - selected.draft_ms + full.draft_ms
        return CostEstimate(batch, gamma, total, selected.verify_ms, full.draft_ms,
                            selected.control_ms, "full_draft_cost_replacement_estimate",
                            min(selected.samples, full.samples), draft_gamma)

    def update(self, batch: int, gamma: int, total_step_ms: float, decay: float = 0.9,
               proposal_budget: int | None = None) -> None:
        if not 0 <= decay < 1 or not math.isfinite(total_step_ms) or total_step_ms <= 0:
            raise ValueError("invalid online cost EWMA observation")
        key = (batch, gamma, proposal_budget)
        old = self.lookup(batch, gamma, proposal_budget)
        # Component times no longer have the same calibration boundary after a
        # total-only update, so don't use them in draft-cost replacement.
        self.cells[key] = CostEstimate(batch, gamma,
            decay * old.total_step_ms + (1 - decay) * total_step_ms,
            None, None, None, "measured_cost_ewma", old.samples + 1, proposal_budget)
        self.online_updates[key] += 1


@dataclass(frozen=True)
class Decision:
    gamma: int
    batch: int
    expected_tokens_per_request: float
    total_step_ms: float
    ms_per_output_token_per_request: float
    source: str
    acceptance_source: str
    decision_cpu_us: float
    candidates: list[dict]
    skipped: list[dict]


class AdaptiveGammaPolicy:
    def __init__(self, table: CalibrationTable, acceptance: AcceptanceEWMA | None = None):
        self.table = table
        self.acceptance = acceptance if acceptance is not None else AcceptanceEWMA()

    def choose(self, batch: int, candidates: Sequence[int] | None = None,
               draft_gamma: int | None = None) -> Decision:
        start = time.perf_counter_ns()
        alternatives, skipped = [], []
        for gamma in sorted(set(self.table.gammas if candidates is None else candidates)):
            if draft_gamma is not None and gamma > draft_gamma:
                continue
            try:
                cost = self.table.estimate(batch, gamma, draft_gamma)
            except KeyError as exc:
                skipped.append({"gamma": gamma, "reason": str(exc)})
                continue
            expected = self.acceptance.expected_tokens(gamma)
            alternatives.append({"gamma": gamma, "expected_tokens_per_request": expected,
                "total_step_ms": cost.total_step_ms,
                "ms_per_output_token_per_request": cost.total_step_ms / expected,
                "source": cost.source})
        if not alternatives:
            raise ValueError(f"no calibrated candidate for batch={batch}: {skipped}")
        best = min(alternatives, key=lambda r: (r["ms_per_output_token_per_request"], r["gamma"]))
        return Decision(**best, batch=batch,
            acceptance_source="empirical_prefix_survival_ewma" if self.acceptance.update_count
                              else "geometric_prior_unvalidated",
            decision_cpu_us=(time.perf_counter_ns() - start) / 1000,
            candidates=alternatives, skipped=skipped)


@dataclass(frozen=True)
class LayerWorkingSet:
    q_e: Sequence[float]
    expert_weight_bytes: float


@dataclass
class WorkingSetCostModel:
    """Illustrative bandwidth/compute model; rates must be calibrated separately.

    effective_weight_bytes_per_ms is a fitted conversion for LOGICAL working
    set bytes. It must not be called measured HBM bandwidth. Attention_ms is a
    supplied cost, not a claim that KV reads are shared perfectly across gamma.
    Layers' memory/compute maxima are summed (layers execute in sequence).
    """
    layers: Sequence[LayerWorkingSet]
    effective_weight_bytes_per_ms: float
    compute_ms_per_token_per_layer: float = 0.0
    fixed_ms: float = 0.0

    def predict(self, batch: int, gamma: int, *, draft_ms: float,
                attention_ms: float, control_ms: float = 0.0) -> dict:
        slots = _slots(batch, gamma)
        if self.effective_weight_bytes_per_ms <= 0 or not self.layers:
            raise ValueError("need layers and positive effective byte-rate")
        costs = [draft_ms, attention_ms, control_ms, self.fixed_ms,
                 self.compute_ms_per_token_per_layer]
        if any(not math.isfinite(x) or x < 0 for x in costs):
            raise ValueError("costs must be finite and nonnegative")
        layer_rows = []
        for layer in self.layers:
            if not math.isfinite(layer.expert_weight_bytes) or layer.expert_weight_bytes <= 0:
                raise ValueError("expert_weight_bytes must be finite and positive")
            distinct = expected_distinct(layer.q_e, batch, gamma)
            logical_bytes = distinct * layer.expert_weight_bytes
            layer_rows.append({"distinct": distinct, "logical_weight_working_set_bytes": logical_bytes,
                "moe_ms": max(logical_bytes / self.effective_weight_bytes_per_ms,
                              slots * self.compute_ms_per_token_per_layer)})
        verify = sum(row["moe_ms"] for row in layer_rows) + attention_ms + self.fixed_ms
        return {"batch": batch, "gamma": gamma, "token_slots": slots,
                "verify_ms": verify, "draft_ms": draft_ms, "control_ms": control_ms,
                "total_step_ms": verify + draft_ms + control_ms, "layers": layer_rows,
                "logical_weight_working_set_bytes": sum(
                    row["logical_weight_working_set_bytes"] for row in layer_rows),
                "evidence_kind": "analytic_prediction_not_measured",
                "assumptions": ["independent token route sets", "supplied non-overlapped costs",
                                "logical working-set bytes are not actual HBM traffic"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    decide = sub.add_parser("choose", help="make one decision from a calibrated timing table")
    decide.add_argument("--calibration", required=True)
    decide.add_argument("--batch", required=True, type=int)
    decide.add_argument("--alpha", type=float, default=0.7)
    decide.add_argument("--gammas", type=int, nargs="+")
    decide.add_argument("--draft-gamma", type=int)
    predict = sub.add_parser("predict", help="write explicitly unmeasured D and acceptance predictions")
    predict.add_argument("--routes", required=True, help="analyze_routes.py probability JSON")
    predict.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    predict.add_argument("--gammas", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 7])
    predict.add_argument("--alpha", type=float, default=0.7)
    predict.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "choose":
        policy = AdaptiveGammaPolicy(CalibrationTable.load(args.calibration), AcceptanceEWMA(args.alpha))
        print(json.dumps(asdict(policy.choose(args.batch, args.gammas, args.draft_gamma)), indent=2))
    else:
        routes = json.loads(Path(args.routes).read_text())
        rows = [{"batch": b, "gamma": g, "layer": layer["layer"],
                 "predicted_distinct": expected_distinct(layer["q_e"], b, g),
                 "expected_tokens_geometric": expected_tokens_geometric(args.alpha, g),
                 "evidence_kind": "analytic_prediction_not_measured"}
                for layer in routes["layers"] for b in args.batches for g in args.gammas]
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix == ".csv":
            with output.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        else:
            output.write_text(json.dumps({"records": rows, "route_source": args.routes}, indent=2))


if __name__ == "__main__":
    main()

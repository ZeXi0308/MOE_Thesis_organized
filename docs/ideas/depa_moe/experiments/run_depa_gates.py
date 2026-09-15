from __future__ import annotations

"""Serial, fail-closed DEPA-MoE development gate runner."""

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from depa_policy import (
    DEPARollingPolicy,
    DeterministicRandomPolicy,
    EDFPolicy,
    FCFSPolicy,
    GreedyPressureSlackPolicy,
    LeastLaxityPolicy,
    ProtocolError,
    RequestSpec,
    ScheduleMetrics,
    ServiceCatalog,
    ServiceCurve,
    SurfacePoint,
    ThresholdPolicy,
    bootstrap_fraction_lcb95,
    exact_slo_goodput_oracle,
    paired_bootstrap_lcb95,
    relative_gain,
    schedule_metrics,
    simulate_causal,
)


@dataclass(frozen=True)
class Episode:
    episode_id: str
    model: str
    cell: str
    seed: int
    window_start_us: float
    window_end_us: float
    requests: tuple[RequestSpec, ...]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _json_safe(payload: Any) -> Any:
    if isinstance(payload, float) and not math.isfinite(payload):
        return None
    if isinstance(payload, dict):
        return {key: _json_safe(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_json_safe(value) for value in payload]
    return payload


def load_config(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("protocol_version") != "depa-moe-v1":
        raise ProtocolError("config protocol_version must be depa-moe-v1")
    return payload


def load_surfaces(path: Path) -> dict[str, ServiceCatalog]:
    payload = _read_json(path)
    if payload.get("schema_version") != "depa-service-surface-v1":
        raise ProtocolError("surface schema_version must be depa-service-surface-v1")
    catalogs: dict[str, ServiceCatalog] = {}
    for model, raw in payload.get("models", {}).items():
        default_raw = raw.get("default_curve")
        default_curve = _parse_curve(default_raw) if default_raw else None
        curves = {
            int(expert_id): _parse_curve(points)
            for expert_id, points in raw.get("expert_curves", {}).items()
        }
        catalogs[model] = ServiceCatalog(
            curves,
            default_curve=default_curve,
            launch_overhead_us=float(raw.get("launch_overhead_us", 0.0)),
            execution_model=raw.get("execution_model", "serial_experts"),
        )
    if not catalogs:
        raise ProtocolError("surface has no model catalogs")
    return catalogs


def _parse_curve(points: Sequence[Mapping[str, Any]]) -> ServiceCurve:
    return ServiceCurve(
        tuple(
            SurfacePoint(
                rows=int(point["rows"]),
                latency_us=float(point["latency_us"]),
                latency_ucb95_us=(
                    float(point["latency_ucb95_us"])
                    if point.get("latency_ucb95_us") is not None
                    else None
                ),
            )
            for point in points
        )
    )


def load_episodes(path: Path) -> tuple[Episode, ...]:
    payload = _read_json(path)
    if payload.get("schema_version") != "depa-episodes-v1":
        raise ProtocolError("episode schema_version must be depa-episodes-v1")
    episodes: list[Episode] = []
    for raw in payload.get("episodes", []):
        model = str(raw["model"])
        cell = str(raw["cell"])
        requests = tuple(
            RequestSpec(
                request_id=str(item["request_id"]),
                model=model,
                cell=cell,
                arrival_us=float(item["arrival_us"]),
                deadline_us=float(item["deadline_us"]),
                expert_rows=tuple(
                    sorted((int(expert_id), int(rows)) for expert_id, rows in item["expert_rows"].items())
                ),
                request_class=str(item.get("request_class", "default")),
                activation_sha256=item.get("activation_sha256"),
            )
            for item in raw["requests"]
        )
        episode = Episode(
            episode_id=str(raw["episode_id"]),
            model=model,
            cell=cell,
            seed=int(raw["seed"]),
            window_start_us=float(raw["window_start_us"]),
            window_end_us=float(raw["window_end_us"]),
            requests=requests,
        )
        if episode.window_end_us <= episode.window_start_us:
            raise ProtocolError("episode window must be positive")
        if any(
            item.arrival_us < episode.window_start_us
            or item.arrival_us >= episode.window_end_us
            for item in episode.requests
        ):
            raise ProtocolError("every request arrival must lie inside its observation window")
        episodes.append(episode)
    if not episodes:
        raise ProtocolError("episode file is empty")
    if len({episode.episode_id for episode in episodes}) != len(episodes):
        raise ProtocolError("duplicate episode_id")
    return tuple(episodes)


def load_breakdown(path: Path) -> tuple[dict[str, Any], ...]:
    payload = _read_json(path)
    if payload.get("schema_version") != "depa-breakdown-v1":
        raise ProtocolError("breakdown schema_version must be depa-breakdown-v1")
    records = tuple(payload.get("records", ()))
    if not records:
        raise ProtocolError("breakdown file is empty")
    for record in records:
        total = float(record["total_critical_path_us"])
        exposed = float(record["target_exposed_us"])
        if not math.isfinite(total) or total <= 0 or not math.isfinite(exposed) or exposed < 0 or exposed > total:
            raise ProtocolError("breakdown times must satisfy 0 <= target <= total")
        if not str(record.get("model", "")) or not str(record.get("cell", "")):
            raise ProtocolError("breakdown model and cell must be non-empty")
    return records


def _group_key(model: str, cell: str) -> str:
    return f"{model}::{cell}"


def gate1_bottleneck_share(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[int, float]] = {}
    for record in records:
        key = (str(record["model"]), str(record["cell"]))
        seed = int(record["seed"])
        if seed in grouped.setdefault(key, {}):
            raise ProtocolError(f"duplicate Gate 1 seed {seed} for {key}")
        grouped[key][seed] = float(record["target_exposed_us"]) / float(record["total_critical_path_us"])
    required_models = tuple(config["required_models"])
    minimum_seeds = int(config["minimum_seeds_per_model_cell"])
    stats: dict[str, Any] = {}
    for (model, cell), by_seed in sorted(grouped.items()):
        values = list(by_seed.values())
        mean, lcb = bootstrap_fraction_lcb95(
            values,
            replicates=int(config["bootstrap_replicates"]),
            seed=int(config["bootstrap_seed"]) + sum(ord(char) for char in model + cell),
        )
        stats[_group_key(model, cell)] = {"n": len(values), "mean_share": mean, "lcb95": lcb}
    cells_by_model = {
        model: {cell for (candidate_model, cell), by_seed in grouped.items() if candidate_model == model and len(by_seed) >= minimum_seeds}
        for model in required_models
    }
    common_cells = sorted(set.intersection(*(cells_by_model[model] for model in required_models))) if required_models else []
    qualifying = []
    for cell in common_cells:
        if all(
            stats[_group_key(model, cell)]["mean_share"] >= float(config["pass_mean_share_min"])
            and stats[_group_key(model, cell)]["lcb95"] >= float(config["pass_lcb95_min"])
            for model in required_models
        ):
            qualifying.append(cell)
    if qualifying:
        decision = "PASS"
        reason = "common natural cell meets bottleneck-share thresholds for every required model"
    elif not common_cells:
        decision = "BLOCKED_INSUFFICIENT_COMMON_CELLS"
        reason = "no adequately replicated common cell across required models"
    elif all(
        any(
            stats[_group_key(model, cell)]["mean_share"] < float(config["kill_mean_share_below"])
            for model in required_models
        )
        for cell in common_cells
    ):
        decision = "FAIL_KILL"
        reason = "every common cell has at least one required model below the kill threshold"
    else:
        decision = "FAIL_INCONCLUSIVE"
        reason = "effect is between kill and pass thresholds or confidence is insufficient"
    return {
        "gate": 1,
        "name": "bottleneck_share",
        "decision": decision,
        "reason": reason,
        "common_cells": common_cells,
        "qualifying_cells": qualifying,
        "groups": stats,
    }


def _run_arm(
    episode: Episode,
    surface: ServiceCatalog,
    policy: Any,
    arm: str,
) -> ScheduleMetrics:
    result = simulate_causal(episode.requests, surface, policy, arm=arm)
    return schedule_metrics(
        episode.requests,
        result,
        window_start_us=episode.window_start_us,
        window_end_us=episode.window_end_us,
    )


def evaluate_episodes(
    episodes: Sequence[Episode],
    surfaces: Mapping[str, ServiceCatalog],
    config: Mapping[str, Any],
) -> dict[str, dict[str, ScheduleMetrics]]:
    max_batch = int(config["policy"]["max_batch"])
    evaluations: dict[str, dict[str, ScheduleMetrics]] = {}
    replicate_keys: set[tuple[str, str, int]] = set()
    for episode in episodes:
        replicate_key = (episode.model, episode.cell, episode.seed)
        if replicate_key in replicate_keys:
            raise ProtocolError(f"duplicate episode replicate {replicate_key}")
        replicate_keys.add(replicate_key)
        if episode.model not in surfaces:
            raise ProtocolError(f"missing surface for model {episode.model}")
        surface = surfaces[episode.model]
        policies = {
            "current_fcfs": FCFSPolicy(max_batch),
            "random": DeterministicRandomPolicy(max_batch, episode.seed),
            "threshold": ThresholdPolicy(
                max_batch,
                int(config["policy"]["threshold_batch"]),
                float(config["policy"]["threshold_wait_us"]),
            ),
            "edf": EDFPolicy(max_batch),
            "least_laxity": LeastLaxityPolicy(max_batch),
            "greedy_pressure_slack": GreedyPressureSlackPolicy(max_batch),
            "depa_rolling": DEPARollingPolicy(
                max_batch,
                max_candidates=int(config["policy"]["depa_max_candidates"]),
                min_batch=int(config["policy"]["depa_min_batch"]),
                max_wait_us=float(config["policy"]["depa_max_wait_us"]),
                reject_infeasible=bool(config["policy"]["depa_reject_infeasible"]),
            ),
        }
        arms = {
            name: _run_arm(episode, surface, policy, name)
            for name, policy in policies.items()
        }
        oracle = exact_slo_goodput_oracle(
            episode.requests,
            surface,
            max_batch=max_batch,
            max_exact_requests=int(config["oracle"]["max_exact_requests"]),
        )
        arms["oracle"] = schedule_metrics(
            episode.requests,
            oracle,
            window_start_us=episode.window_start_us,
            window_end_us=episode.window_end_us,
        )
        evaluations[episode.episode_id] = arms
    return evaluations


def gate2_oracle_space(
    episodes: Sequence[Episode],
    evaluations: Mapping[str, Mapping[str, ScheduleMetrics]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[int, Episode]] = {}
    for episode in episodes:
        key = (episode.model, episode.cell)
        if episode.seed in grouped.setdefault(key, {}):
            raise ProtocolError(f"duplicate Gate 2 seed {episode.seed} for {key}")
        grouped[key][episode.seed] = episode
    required_models = tuple(config["required_models"])
    minimum_seeds = int(config["minimum_seeds_per_model_cell"])
    stats: dict[str, Any] = {}
    for (model, cell), by_seed in sorted(grouped.items()):
        group = list(by_seed.values())
        baseline = [evaluations[item.episode_id]["current_fcfs"].slo_goodput_per_s for item in group]
        oracle = [evaluations[item.episode_id]["oracle"].slo_goodput_per_s for item in group]
        mean, lcb = paired_bootstrap_lcb95(
            baseline,
            oracle,
            replicates=int(config["bootstrap_replicates"]),
            seed=int(config["bootstrap_seed"]) + sum(ord(char) for char in model + cell),
        )
        stats[_group_key(model, cell)] = {"n": len(group), "mean_gain": mean, "lcb95": lcb}
    cells_by_model = {
        model: {cell for (candidate_model, cell), by_seed in grouped.items() if candidate_model == model and len(by_seed) >= minimum_seeds}
        for model in required_models
    }
    common_cells = sorted(set.intersection(*(cells_by_model[model] for model in required_models))) if required_models else []
    qualifying = [
        cell
        for cell in common_cells
        if all(
            stats[_group_key(model, cell)]["mean_gain"] >= float(config["pass_mean_gain_min"])
            and stats[_group_key(model, cell)]["lcb95"] >= float(config["pass_lcb95_min"])
            for model in required_models
        )
    ]
    if qualifying:
        decision = "PASS"
        reason = "exact oracle exposes enough SLO-goodput headroom in a common cell"
    elif not common_cells:
        decision = "BLOCKED_INSUFFICIENT_COMMON_CELLS"
        reason = "no adequately replicated common cell across required models"
    elif all(
        any(
            stats[_group_key(model, cell)]["mean_gain"] < float(config["kill_mean_gain_below"])
            for model in required_models
        )
        for cell in common_cells
    ):
        decision = "FAIL_KILL"
        reason = "oracle headroom is below the kill threshold"
    else:
        decision = "FAIL_INCONCLUSIVE"
        reason = "oracle headroom or confidence does not meet the pass threshold"
    return {
        "gate": 2,
        "name": "oracle_space",
        "decision": decision,
        "reason": reason,
        "common_cells": common_cells,
        "qualifying_cells": qualifying,
        "groups": stats,
    }


def gate3_strategy_gap(
    episodes: Sequence[Episode],
    evaluations: Mapping[str, Mapping[str, ScheduleMetrics]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    simple_arms = tuple(config["simple_arms"])
    rows = []
    for episode in episodes:
        arms = evaluations[episode.episode_id]
        current = arms["current_fcfs"].slo_goodput_per_s
        oracle_gain = relative_gain(arms["oracle"].slo_goodput_per_s, current)
        depa_gain = relative_gain(arms["depa_rolling"].slo_goodput_per_s, current)
        simple_gains = {name: relative_gain(arms[name].slo_goodput_per_s, current) for name in simple_arms}
        best_simple_arm = max(simple_gains, key=lambda name: (simple_gains[name], name))
        best_simple_gain = simple_gains[best_simple_arm]
        depa_capture = depa_gain / oracle_gain if oracle_gain > 0 else 0.0
        simple_capture = best_simple_gain / oracle_gain if oracle_gain > 0 else 0.0
        current_p99 = arms["current_fcfs"].p99_completion_latency_us
        depa_p99 = arms["depa_rolling"].p99_completion_latency_us
        rows.append(
            {
                "episode_id": episode.episode_id,
                "model": episode.model,
                "cell": episode.cell,
                "oracle_gain": oracle_gain,
                "depa_gain": depa_gain,
                "best_simple_arm": best_simple_arm,
                "best_simple_gain": best_simple_gain,
                "depa_capture": depa_capture,
                "simple_capture": simple_capture,
                "depa_minus_simple": depa_gain - best_simple_gain,
                "p99_ratio": depa_p99 / current_p99 if current_p99 > 0 else math.inf,
                "fairness_delta": arms["depa_rolling"].jain_class_fairness - arms["current_fcfs"].jain_class_fairness,
                "decision_overhead_fraction": arms["depa_rolling"].decision_overhead_fraction,
            }
        )
    required_models = tuple(config["required_models"])
    by_model = {model: [row for row in rows if row["model"] == model] for model in required_models}
    aggregates = {}
    for model, model_rows in by_model.items():
        if not model_rows:
            continue
        aggregates[model] = {
            key: sum(float(row[key]) for row in model_rows) / len(model_rows)
            for key in (
                "oracle_gain",
                "depa_gain",
                "best_simple_gain",
                "depa_capture",
                "simple_capture",
                "depa_minus_simple",
                "p99_ratio",
                "fairness_delta",
                "decision_overhead_fraction",
            )
        }
    missing = [model for model in required_models if model not in aggregates]
    simple_saturates = any(
        aggregates[model]["simple_capture"] >= float(config["simple_capture_abandon_at"])
        for model in aggregates
    )
    pass_all = not missing and all(
        aggregates[model]["depa_gain"] >= float(config["depa_gain_min"])
        and aggregates[model]["depa_capture"] >= float(config["depa_oracle_capture_min"])
        and aggregates[model]["depa_minus_simple"] >= float(config["depa_minus_simple_min"])
        and aggregates[model]["p99_ratio"] <= float(config["p99_ratio_max"])
        and aggregates[model]["fairness_delta"] >= float(config["fairness_delta_min"])
        and aggregates[model]["decision_overhead_fraction"] <= float(config["decision_overhead_fraction_max"])
        for model in required_models
    )
    if missing:
        decision = "BLOCKED_MISSING_MODEL"
        reason = f"missing required model evaluations: {missing}"
    elif simple_saturates:
        decision = "FAIL_ABANDON_COMPLEX_POLICY"
        reason = "a simple causal strategy captures at least the configured fraction of oracle headroom"
    elif pass_all:
        decision = "PASS"
        reason = "DEPA clears gain, oracle-capture, simple-gap, overhead, P99, and fairness thresholds"
    else:
        decision = "FAIL_INCONCLUSIVE"
        reason = "DEPA does not clear every frozen strategy-gap safeguard"
    return {
        "gate": 3,
        "name": "strategy_gap",
        "decision": decision,
        "reason": reason,
        "aggregates_by_model": aggregates,
        "episodes": rows,
    }


def run_serial_gates(
    config: Mapping[str, Any],
    breakdown: Sequence[Mapping[str, Any]],
    episodes: Sequence[Episode],
    surfaces: Mapping[str, ServiceCatalog],
    *,
    development: bool,
    inputs_scientific_eligible: bool = False,
) -> dict[str, Any]:
    if not development:
        if config.get("scientific_result_eligible") is not True:
            raise ProtocolError("formal run blocked; config is not scientifically eligible")
        if not inputs_scientific_eligible:
            raise ProtocolError("formal run blocked; input artifacts are not scientifically eligible")
        capabilities = config.get("formal_capabilities", {})
        missing = sorted(name for name, available in capabilities.items() if not available)
        if missing:
            raise ProtocolError(f"formal run blocked; missing capabilities: {missing}")
    gate1 = gate1_bottleneck_share(breakdown, config["gate1"])
    result: dict[str, Any] = {
        "protocol_version": config["protocol_version"],
        "scientific_result_eligible": not development,
        "result_boundary": (
            "DEVELOPMENT_ONLY_NOT_SCIENTIFIC" if development else "FORMAL_CAPABILITIES_DECLARED"
        ),
        "gates": [gate1],
    }
    if gate1["decision"] != "PASS":
        result["overall_decision"] = "STOP_AFTER_GATE_1"
        return result
    evaluations = evaluate_episodes(episodes, surfaces, config)
    gate2 = gate2_oracle_space(episodes, evaluations, config["gate2"])
    result["gates"].append(gate2)
    if gate2["decision"] != "PASS":
        result["overall_decision"] = "STOP_AFTER_GATE_2"
        result["episode_metrics"] = _serialize_evaluations(evaluations)
        return result
    gate3_episodes = tuple(
        episode for episode in episodes if episode.cell in set(gate2["qualifying_cells"])
    )
    gate3 = gate3_strategy_gap(gate3_episodes, evaluations, config["gate3"])
    result["gates"].append(gate3)
    result["overall_decision"] = "PASS_ALL_GATES" if gate3["decision"] == "PASS" else "STOP_AFTER_GATE_3"
    result["episode_metrics"] = _serialize_evaluations(evaluations)
    return result


def _serialize_evaluations(
    evaluations: Mapping[str, Mapping[str, ScheduleMetrics]]
) -> dict[str, Any]:
    return {
        episode_id: {arm: asdict(metrics) for arm, metrics in arms.items()}
        for episode_id, arms in evaluations.items()
    }


def development_fixture() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    models = ("olmoe-dev", "llmjp-dev")
    breakdown = {
        "schema_version": "depa-breakdown-v1",
        "development_only": True,
        "scientific_result_eligible": False,
        "records": [
            {
                "model": model,
                "cell": "natural-bursty",
                "seed": seed,
                "total_critical_path_us": 100.0,
                "target_exposed_us": 25.0 + (seed % 3),
            }
            for model in models
            for seed in range(1, 7)
        ],
    }
    surface = {
        "schema_version": "depa-service-surface-v1",
        "development_only": True,
        "scientific_result_eligible": False,
        "models": {
            model: {
                "execution_model": "serial_experts",
                "launch_overhead_us": 2.0,
                "default_curve": [
                    {"rows": 1, "latency_us": 10.0, "latency_ucb95_us": 10.5},
                    {"rows": 2, "latency_us": 12.0, "latency_ucb95_us": 12.5},
                    {"rows": 4, "latency_us": 16.0, "latency_ucb95_us": 16.5},
                    {"rows": 8, "latency_us": 24.0, "latency_ucb95_us": 24.5},
                ],
            }
            for model in models
        },
    }
    episode_rows = []
    for model in models:
        for seed in range(1, 7):
            rng = random.Random(seed)
            requests = []
            for index in range(8):
                arrival = float((index // 2) * 5 + rng.randrange(0, 2))
                requests.append(
                    {
                        "request_id": f"{model}-{seed}-{index}",
                        "arrival_us": arrival,
                        "deadline_us": arrival + (38.0 if index % 3 else 27.0),
                        "expert_rows": {str(index % 2): 1},
                        "request_class": "tight" if index % 3 == 0 else "normal",
                    }
                )
            episode_rows.append(
                {
                    "episode_id": f"{model}-natural-bursty-{seed}",
                    "model": model,
                    "cell": "natural-bursty",
                    "seed": seed,
                    "window_start_us": 0.0,
                    "window_end_us": 80.0,
                    "requests": requests,
                }
            )
    episodes = {
        "schema_version": "depa-episodes-v1",
        "development_only": True,
        "scientific_result_eligible": False,
        "episodes": episode_rows,
    }
    return breakdown, episodes, surface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fixture = subparsers.add_parser("make-development-fixture")
    fixture.add_argument("--output-dir", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--breakdown", type=Path, required=True)
    run.add_argument("--episodes", type=Path, required=True)
    run.add_argument("--surface", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--development", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "make-development-fixture":
        breakdown, episodes, surface = development_fixture()
        _write_json(args.output_dir / "breakdown.json", breakdown)
        _write_json(args.output_dir / "episodes.json", episodes)
        _write_json(args.output_dir / "surface.json", surface)
        return 0
    config = load_config(args.config)
    input_payloads = (
        _read_json(args.breakdown),
        _read_json(args.episodes),
        _read_json(args.surface),
    )
    result = run_serial_gates(
        config,
        load_breakdown(args.breakdown),
        load_episodes(args.episodes),
        load_surfaces(args.surface),
        development=args.development,
        inputs_scientific_eligible=all(
            payload.get("scientific_result_eligible") is True for payload in input_payloads
        ),
    )
    _write_json(args.output, result)
    print(json.dumps({
        "overall_decision": result["overall_decision"],
        "scientific_result_eligible": result["scientific_result_eligible"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

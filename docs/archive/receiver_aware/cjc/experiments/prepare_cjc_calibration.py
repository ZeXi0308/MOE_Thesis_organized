#!/usr/bin/env python3
"""Build the cjc-v1 calibration-only workload and ACK timing manifest.

The script consumes identity-complete *calibration* routes.  It never opens a
sealed route trace.  Route/LUT-derived workload constants, the calibration-only
static baseline and SLO, measured host controller costs, and analytic wire time
remain separate provenance components.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Callable, Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(candidate for candidate in HERE.parents if (candidate / "experiments/shared").is_dir())
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from cjc_policy import (  # noqa: E402
    JOIN_BLIND_ARMS,
    AckWireRecord,
    CJCValidationError,
    PlacementManifest,
    RouteContribution,
    ServiceLUT,
    WorkloadSpec,
    ack_message_bytes,
    build_tasks_from_routes,
    decode_ack_message,
    encode_ack_message,
    episode_metrics,
    simulate,
    validate_route_contributions,
)
from run_cjc_oracle import (  # noqa: E402
    load_json,
    load_lut,
    load_placement,
    load_routes,
    select_replay_routes,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "docs/archive/receiver_aware/cjc/configs/cjc_v1.json",
    )
    parser.add_argument("--calibration-route-trace", action="append", required=True)
    parser.add_argument("--lut", type=Path, required=True)
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument("--data-registry", type=Path, required=True)
    parser.add_argument(
        "--arrival-trace",
        type=Path,
        required=True,
        help="Calibration-only measured/public arrival timestamps used only to fit MMPP shape.",
    )
    parser.add_argument("--mode", choices=("dev", "formal"), default="dev")
    parser.add_argument("--signoff", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _quantile(values: Sequence[float], q: float) -> float:
    if not values or not 0.0 <= q <= 1.0:
        raise CJCValidationError("invalid calibration quantile")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise CJCValidationError("non-finite calibration value")
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    alpha = position - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


def load_arrival_trace(path: Path) -> tuple[list[float], Mapping[str, object]]:
    raw = load_json(path)
    if raw.get("schema_version") != "cjc-arrival-trace-v1":
        raise CJCValidationError("arrival trace schema mismatch")
    if raw.get("protocol_split") != "calibration":
        raise CJCValidationError("MMPP fit may consume calibration arrivals only")
    if raw.get("source") not in {
        "measured_same_run_host_serving",
        "public_trace_calibration_slice",
    }:
        raise CJCValidationError("synthetic or unlabelled arrival trace is forbidden")
    supplied = raw.get("manifest_sha256")
    unhashed = dict(raw)
    unhashed.pop("manifest_sha256", None)
    if supplied != hashlib.sha256(_canonical_json(unhashed)).hexdigest():
        raise CJCValidationError("arrival trace self-hash mismatch")
    timestamps = raw.get("timestamps_us")
    if not isinstance(timestamps, list) or len(timestamps) < 65:
        raise CJCValidationError("arrival trace needs at least 65 timestamps")
    values = [float(value) for value in timestamps]
    if any(not math.isfinite(value) for value in values):
        raise CJCValidationError("arrival trace contains non-finite timestamp")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise CJCValidationError("arrival timestamps must be strictly increasing")
    return values, raw


def fit_two_state_mmpp(timestamps_us: Sequence[float]) -> dict[str, float]:
    """Fit the frozen symmetric two-state proxy without using outcome metrics."""
    intervals = [right - left for left, right in zip(timestamps_us, timestamps_us[1:])]
    threshold = statistics.median(intervals)
    states = [interval <= threshold for interval in intervals]
    low_intervals = [value for value, high in zip(intervals, states) if not high]
    high_intervals = [value for value, high in zip(intervals, states) if high]
    if not low_intervals or not high_intervals:
        raise CJCValidationError("arrival trace cannot identify two non-empty rate states")
    low_rate = 1.0 / statistics.fmean(low_intervals)
    high_rate = 1.0 / statistics.fmean(high_intervals)
    if high_rate <= low_rate:
        raise CJCValidationError("fitted high arrival state is not faster than low state")
    base_rate = 0.5 * (low_rate + high_rate)
    switches = sum(left != right for left, right in zip(states, states[1:]))
    switch_probability = switches / (len(states) - 1)
    if not 0.0 < switch_probability < 1.0:
        raise CJCValidationError("arrival trace produced a degenerate MMPP transition rate")
    return {
        "mmpp_low_multiplier": low_rate / base_rate,
        "mmpp_high_multiplier": high_rate / base_rate,
        "mmpp_switch_probability": switch_probability,
        "fit_interval_threshold_us": threshold,
        "fit_low_rate_per_us": low_rate,
        "fit_high_rate_per_us": high_rate,
    }


def derive_layer_period_us(
    routes: Sequence[RouteContribution], lut: ServiceLUT, quantile: float
) -> float:
    groups: dict[tuple[str, int, int, int], int] = defaultdict(int)
    for route in routes:
        groups[(route.request_id, route.layer_id, route.sender_rank, route.expert_id)] += 1
    sender_load: dict[tuple[str, int, int], float] = defaultdict(float)
    for (request_id, layer_id, sender_rank, _expert_id), rows in groups.items():
        point = lut.lookup(routes[0].model_revision, layer_id, rows)
        sender_load[(request_id, layer_id, sender_rank)] += point.expert_us
    layer_durations: dict[tuple[str, int], float] = defaultdict(float)
    for (request_id, layer_id, _sender_rank), duration in sender_load.items():
        layer_durations[(request_id, layer_id)] = max(
            layer_durations[(request_id, layer_id)], duration
        )
    period = _quantile(list(layer_durations.values()), quantile)
    if period <= 0:
        raise CJCValidationError("calibration-derived layer period must be positive")
    return period


def derive_arrival_rate_per_us(
    routes: Sequence[RouteContribution],
    *,
    lut: ServiceLUT,
    placement: PlacementManifest,
    hidden_size: int,
    dtype_bytes: int,
    descriptor_bytes: int,
    alignment_bytes: int,
    link_gbps: float,
    target_rho: float,
) -> tuple[float, dict[str, float]]:
    if not 0.0 < target_rho < 1.0:
        raise CJCValidationError("target utilization must be in (0,1)")
    bytes_per_us = link_gbps * 1e9 / 8.0 / 1e6
    wire_bytes = hidden_size * dtype_bytes + descriptor_bytes + alignment_bytes
    demand: dict[tuple[str, str], float] = defaultdict(float)
    requests = sorted({route.request_id for route in routes})
    for route in routes:
        point = lut.lookup(route.model_revision, route.layer_id, 1)
        service = (
            point.pack_us
            + point.launch_us
            + point.host_staging_us
            + point.reduction_us
            + wire_bytes / bytes_per_us
        )
        resource = placement.receiver_resource(route.receiver_rank)
        demand[(route.request_id, resource)] += service
    resources = sorted({resource for _, resource in demand})
    mean_demand_by_resource = {
        resource: sum(demand.get((request_id, resource), 0.0) for request_id in requests)
        / len(requests)
        for resource in resources
    }
    bottleneck = max(mean_demand_by_resource.values(), default=0.0)
    if bottleneck <= 0:
        raise CJCValidationError("zero receiver service demand in calibration routes")
    return target_rho / bottleneck, mean_demand_by_resource


def _pooled_p99(tasks: Sequence[object], result: object, slo_us: float) -> float:
    rows = episode_metrics(tasks, result, slo_us=slo_us)  # type: ignore[arg-type]
    latencies = [value for row in rows for value in row.token_latencies_us]
    return _quantile(latencies, 0.99)


def select_calibration_static_arm(
    routes: Sequence[RouteContribution],
    *,
    lut: ServiceLUT,
    placement: PlacementManifest,
    workload: WorkloadSpec,
    seeds: Sequence[int],
    hidden_size: int,
    dtype_bytes: int,
    descriptor_bytes: int,
    alignment_bytes: int,
    link_gbps: float,
) -> tuple[str, float, Mapping[str, float]]:
    arms = JOIN_BLIND_ARMS[:-1]
    values: dict[str, list[float]] = {arm: [] for arm in arms}
    for seed in seeds:
        tasks = build_tasks_from_routes(
            routes,
            lut=lut,
            placement=placement,
            workload=workload,
            seed=int(seed),
            hidden_size=hidden_size,
            dtype_bytes=dtype_bytes,
            descriptor_bytes=descriptor_bytes,
            alignment_bytes=alignment_bytes,
            link_gbps=link_gbps,
        )
        for arm in arms:
            result = simulate(tasks, arm=arm, starvation_us=math.inf)
            values[arm].append(_pooled_p99(tasks, result, workload.slo_us))
    aggregate = {arm: statistics.fmean(samples) for arm, samples in values.items()}
    winner = min(arms, key=lambda arm: (aggregate[arm], arm))
    return winner, aggregate[winner], aggregate


def _timed_us(operation: Callable[[], object], iterations: int, repeats: int) -> float:
    if iterations < 1 or repeats < 1:
        raise CJCValidationError("invalid ACK microbenchmark repetitions")
    per_call: list[float] = []
    checksum = 0
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _index in range(iterations):
            checksum ^= hash(operation())
        elapsed = time.perf_counter_ns() - start
        per_call.append(elapsed / iterations / 1000.0)
    if checksum == -1:  # keep operations observably consumed without affecting timing output
        raise AssertionError("unreachable checksum")
    return statistics.median(per_call)


def measure_ack_components(
    *, model_key: str, iterations: int, repeats: int, link_gbps: float
) -> Mapping[str, object]:
    record = AckWireRecord(0x123456789ABCDEF0, 1, 1, 1)
    payload = encode_ack_message((record,))
    if decode_ack_message(payload) != (record,):
        raise AssertionError("ACK codec calibration self-check failed")
    visible = {(model_key, 1, 1): 1}
    build_us = _timed_us(
        lambda: AckWireRecord(0x123456789ABCDEF0, 1, 1, 1), iterations, repeats
    )
    serialize_us = _timed_us(lambda: encode_ack_message((record,)), iterations, repeats)
    parse_us = _timed_us(lambda: decode_ack_message(payload), iterations, repeats)
    policy_lookup_us = _timed_us(lambda: visible[(model_key, 1, 1)], iterations, repeats)
    message_bytes = ack_message_bytes(1)
    wire_us = message_bytes * 8.0 / (link_gbps * 1000.0)
    measured_source = "measured_same_run_host_monotonic_ns"
    return {
        "schema_version": "cjc-ack-timing-v1",
        "components": {
            "build_us": {"value_us": build_us, "source": measured_source},
            "serialize_us": {"value_us": serialize_us, "source": measured_source},
            "wire_us": {
                "value_us": wire_us,
                "source": "analytic_link",
                "message_bytes": message_bytes,
                "link_gbps": link_gbps,
            },
            "parse_us": {"value_us": parse_us, "source": measured_source},
            "policy_lookup_us": {
                "value_us": policy_lookup_us,
                "source": measured_source,
            },
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "clock": "perf_counter_ns",
            "iterations": iterations,
            "repeats": repeats,
        },
    }


def _require_signoff(path: Path | None, bindings: Mapping[str, str]) -> None:
    if path is None or not path.is_file():
        raise CJCValidationError("formal calibration requires Phase-4 SIGNED-OFF")
    raw = load_json(path)
    if raw.get("status") != "SIGNED-OFF":
        raise CJCValidationError("calibration Phase-4 status is not SIGNED-OFF")
    for key, expected in bindings.items():
        if raw.get(key) != expected:
            raise CJCValidationError(f"calibration signoff hash mismatch: {key}")


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    if config.get("protocol_version") != "cjc-v1":
        raise CJCValidationError("calibration config is not cjc-v1")
    calibration_cfg = config.get("calibration")
    replay_cfg = config.get("replay_selection")
    required_models = config.get("required_models")
    wire_cfg = config.get("wire")
    topology_cfg = config.get("topology")
    if not all(
        isinstance(value, dict)
        for value in (calibration_cfg, replay_cfg, required_models, wire_cfg, topology_cfg)
    ):
        raise CJCValidationError("incomplete calibration config")
    if calibration_cfg.get("arrival_trace_required_for_bursty_mmpp") is not True:
        raise CJCValidationError("bursty MMPP must fail closed without an arrival trace")

    registry = load_json(args.data_registry)
    calibration_manifest_sha = str(registry.get("calibration_manifest_sha256", ""))
    sealed_manifest_sha = str(registry.get("sealed_manifest_sha256", ""))
    if len(calibration_manifest_sha) != 64 or len(sealed_manifest_sha) != 64:
        raise CJCValidationError("data registry lacks calibration/sealed manifest hashes")
    routes = load_routes(args.calibration_route_trace)
    if any(route.data_manifest_sha256 == sealed_manifest_sha for route in routes):
        raise CJCValidationError("sealed route appeared in calibration preparation")
    if any(route.data_manifest_sha256 != calibration_manifest_sha for route in routes):
        raise CJCValidationError("route is not bound to the calibration manifest")
    placement = load_placement(args.placement, config)
    lut = load_lut(args.lut, formal=args.mode == "formal")
    timestamps, arrival_trace = load_arrival_trace(args.arrival_trace)
    mmpp = fit_two_state_mmpp(timestamps)

    hashes = {
        "cjc_calibration_config_sha256": sha256_file(args.config),
        "cjc_calibration_source_sha256": sha256_file(Path(__file__)),
        "cjc_calibration_lut_sha256": sha256_file(args.lut),
        "cjc_calibration_placement_sha256": sha256_file(args.placement),
        "cjc_calibration_data_registry_sha256": sha256_file(args.data_registry),
        "cjc_calibration_arrival_trace_sha256": sha256_file(args.arrival_trace),
        "cjc_calibration_route_trace_sha256": hashlib.sha256(
            "".join(sha256_file(path) for path in args.calibration_route_trace).encode("ascii")
        ).hexdigest(),
    }
    if args.mode == "formal":
        _require_signoff(args.signoff, hashes)

    by_revision: dict[str, list[RouteContribution]] = defaultdict(list)
    for route in routes:
        by_revision[route.model_revision].append(route)
    revision_to_key = {
        str(value["revision"]): str(key)
        for key, value in required_models.items()
        if isinstance(value, dict)
    }
    if set(by_revision) != set(revision_to_key):
        raise CJCValidationError("calibration routes must contain exactly both frozen models")

    models_out: dict[str, object] = {}
    for revision, model_routes in sorted(by_revision.items()):
        model_key = revision_to_key[revision]
        model_cfg = required_models[model_key]
        assert isinstance(model_cfg, dict)
        validate_route_contributions(
            model_routes,
            expected_model_revision=revision,
            top_k=int(model_cfg["top_k"]),
            num_experts=int(model_cfg["num_experts"]),
            placement=placement,
            expected_data_manifest_sha256=calibration_manifest_sha,
            formal=True,
        )
        request_ids = {route.request_id for route in model_routes}
        if len(request_ids) != 64:
            raise CJCValidationError("calibration route trace must contain 64 requests/model")
        positions_by_forward_layer: dict[tuple[str, str, int], set[int]] = defaultdict(set)
        forwards_by_request: dict[str, set[str]] = defaultdict(set)
        layers_by_request: dict[str, set[int]] = defaultdict(set)
        for route in model_routes:
            positions_by_forward_layer[
                (route.request_id, route.forward_id, route.layer_id)
            ].add(route.token_position)
            forwards_by_request[route.request_id].add(route.forward_id)
            layers_by_request[route.request_id].add(route.layer_id)
        if any(len(values) != 1 for values in forwards_by_request.values()):
            raise CJCValidationError("calibration request must have exactly one independent forward")
        if any(values != set(range(128)) for values in positions_by_forward_layer.values()):
            raise CJCValidationError("calibration forward/layer must contain positions 0..127")
        layer_sets = {tuple(sorted(values)) for values in layers_by_request.values()}
        if len(layer_sets) != 1:
            raise CJCValidationError("calibration requests expose inconsistent MoE layer sets")
        selected = select_replay_routes(model_routes, replay_cfg)
        layer_period = derive_layer_period_us(
            selected, lut, float(calibration_cfg["layer_period_quantile"])
        )
        cells: dict[str, object] = {}
        for cell, target_rho in (("steady_rho50", 0.50), ("bursty_rho80", 0.80)):
            arrival_rate, resource_demands = derive_arrival_rate_per_us(
                selected,
                lut=lut,
                placement=placement,
                hidden_size=int(model_cfg["hidden_size"]),
                dtype_bytes=int(wire_cfg["dtype_bytes"]),
                descriptor_bytes=int(wire_cfg["descriptor_bytes_per_contribution"]),
                alignment_bytes=int(wire_cfg["alignment_bytes_per_contribution"]),
                link_gbps=float(topology_cfg["link_gbps_primary"]),
                target_rho=target_rho,
            )
            workload = WorkloadSpec(
                cell=cell,
                arrival_rate_per_us=arrival_rate,
                layer_period_us=layer_period,
                slo_us=1e12,
                mmpp_low_multiplier=(
                    float(mmpp["mmpp_low_multiplier"]) if cell == "bursty_rho80" else 0.5
                ),
                mmpp_high_multiplier=(
                    float(mmpp["mmpp_high_multiplier"]) if cell == "bursty_rho80" else 1.5
                ),
                mmpp_switch_probability=(
                    float(mmpp["mmpp_switch_probability"]) if cell == "bursty_rho80" else 0.10
                ),
            )
            arm, p99_us, all_arms = select_calibration_static_arm(
                selected,
                lut=lut,
                placement=placement,
                workload=workload,
                seeds=[int(value) for value in calibration_cfg["seeds"]],
                hidden_size=int(model_cfg["hidden_size"]),
                dtype_bytes=int(wire_cfg["dtype_bytes"]),
                descriptor_bytes=int(wire_cfg["descriptor_bytes_per_contribution"]),
                alignment_bytes=int(wire_cfg["alignment_bytes_per_contribution"]),
                link_gbps=float(topology_cfg["link_gbps_primary"]),
            )
            cells[cell] = {
                "arrival_rate_per_us": arrival_rate,
                "layer_period_us": layer_period,
                "slo_us": 1.10 * p99_us,
                "calibration_best_joinblind_p99_us": p99_us,
                "slo_definition": "1.10_x_calibration_best_joinblind_p99",
                "target_rho": target_rho,
                "calib_best_static": arm,
                "all_joinblind_p99_us": all_arms,
                "receiver_mean_demand_us_per_global_request": resource_demands,
                "mmpp_low_multiplier": workload.mmpp_low_multiplier,
                "mmpp_high_multiplier": workload.mmpp_high_multiplier,
                "mmpp_switch_probability": workload.mmpp_switch_probability,
            }
        models_out[model_key] = {
            "model_revision": revision,
            "cells": cells,
            "ack_timing": measure_ack_components(
                model_key=model_key,
                iterations=int(calibration_cfg["ack_microbenchmark_iterations"]),
                repeats=int(calibration_cfg["ack_microbenchmark_repeats"]),
                link_gbps=float(topology_cfg["link_gbps_primary"]),
            ),
        }

    output = {
        "schema_version": "cjc-calibration-v1",
        "status": "CALIBRATION_ONLY" if args.mode == "formal" else "NOT_TESTED",
        "scientific_result": False,
        "mode": args.mode,
        "models": models_out,
        "mmpp_fit": {
            **mmpp,
            "source": arrival_trace["source"],
            "arrival_trace_manifest_sha256": arrival_trace["manifest_sha256"],
        },
        "provenance": {
            **hashes,
            "calibration_data_manifest_sha256": calibration_manifest_sha,
            "sealed_data_manifest_sha256_not_opened": sealed_manifest_sha,
            "selection": replay_cfg,
            "static_metric": calibration_cfg["static_baseline_metric"],
            "layer_period_quantile": calibration_cfg["layer_period_quantile"],
        },
    }
    if args.output.exists():
        raise CJCValidationError("refusing to overwrite calibration manifest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": output["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

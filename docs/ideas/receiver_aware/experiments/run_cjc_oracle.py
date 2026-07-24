#!/usr/bin/env python3
"""Run the cjc-v1 causal-oracle screen.

Formal mode is fail-closed on Phase-4 attestation and every frozen artifact
hash.  Development mode is useful for implementation smoke tests only and can
never emit a scientific verdict.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Iterable, Mapping

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from cjc_policy import (  # noqa: E402
    ACK_ALIGNMENT_BYTES,
    ACK_HEADER_BYTES,
    ACK_RECORD_BYTES,
    JOIN_BLIND_ARMS,
    AckConfig,
    CJCValidationError,
    EpisodeMetrics,
    LUTPoint,
    PlacementManifest,
    RouteContribution,
    ServiceLUT,
    WorkloadSpec,
    assert_arm_equivalence,
    build_tasks_from_routes,
    episode_metrics,
    paired_hierarchical_bootstrap,
    simulate,
    validate_route_contributions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "docs/ideas/receiver_aware/configs/cjc_v1.json"),
    )
    parser.add_argument("--route-trace", action="append", required=True)
    parser.add_argument("--lut", required=True)
    parser.add_argument("--placement", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--phase4-signoff")
    parser.add_argument("--mode", choices=("dev", "formal"), default="dev")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, object]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise CJCValidationError(f"expected JSON object: {path}")
    return value


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def load_routes(paths: Iterable[str | Path]) -> list[RouteContribution]:
    rows: list[RouteContribution] = []
    for path in paths:
        route_path = Path(path)
        if route_path.suffix.lower() != ".jsonl":
            raise CJCValidationError("legacy CSV is forbidden; CJC route input must be JSONL")
        with route_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CJCValidationError(f"invalid JSONL {route_path}:{line_number}") from exc
                if not isinstance(raw, dict):
                    raise CJCValidationError("route JSONL row must be an object")
                forbidden_timing = {
                    "arrival_us",
                    "expert_ready_us",
                    "service_us",
                    "ack_us",
                    "completion_us",
                } & set(raw)
                if forbidden_timing:
                    raise CJCValidationError(
                        "route producer must not inject replay timing fields: "
                        + ",".join(sorted(forbidden_timing))
                    )
                rows.append(RouteContribution.from_mapping(raw))
    return rows


def load_lut(path: str | Path, *, formal: bool = False) -> ServiceLUT:
    required = {
        "model_revision",
        "layer_id",
        "rows",
        "expert_us",
        "pack_us",
        "launch_us",
        "host_staging_us",
        "reduction_us",
        "source",
    }
    points: list[LUTPoint] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise CJCValidationError("LUT CSV is missing frozen columns")
        for row in reader:
            points.append(
                LUTPoint(
                    model_revision=str(row["model_revision"]),
                    layer_id=int(row["layer_id"]),
                    rows=int(row["rows"]),
                    expert_us=float(row["expert_us"]),
                    pack_us=float(row["pack_us"]),
                    launch_us=float(row["launch_us"]),
                    host_staging_us=float(row["host_staging_us"]),
                    reduction_us=float(row["reduction_us"]),
                    source=str(row["source"]),
                    expert_source=(str(row["expert_source"]) if row.get("expert_source") else None),
                    pack_source=(str(row["pack_source"]) if row.get("pack_source") else None),
                    launch_source=(str(row["launch_source"]) if row.get("launch_source") else None),
                    host_staging_source=(
                        str(row["host_staging_source"])
                        if row.get("host_staging_source") else None
                    ),
                    reduction_source=(
                        str(row["reduction_source"]) if row.get("reduction_source") else None
                    ),
                )
            )
    return ServiceLUT(points, formal=formal)


def load_placement(path: str | Path, config: Mapping[str, object]) -> PlacementManifest:
    raw = load_json(path)
    topology = config["topology"]
    if not isinstance(topology, dict):
        raise CJCValidationError("invalid topology config")
    supplied_hash = raw.get("manifest_sha256")
    unhashed = dict(raw)
    unhashed.pop("manifest_sha256", None)
    actual_hash = hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if supplied_hash != actual_hash:
        raise CJCValidationError("placement manifest self-hash mismatch")
    expert_map_raw = raw.get("expert_to_sender_by_model")
    request_map_raw = raw.get("request_to_receiver")
    if not isinstance(expert_map_raw, dict) or not isinstance(request_map_raw, dict):
        raise CJCValidationError("placement manifest lacks expert/request ownership maps")
    expert_map: dict[tuple[str, int], int] = {}
    for model_revision, model_map in expert_map_raw.items():
        if not isinstance(model_map, dict):
            raise CJCValidationError("per-model expert placement must be an object")
        for expert_id, sender_rank in model_map.items():
            expert_map[(str(model_revision), int(expert_id))] = int(sender_rank)
    return PlacementManifest(
        sha256=str(supplied_hash),
        ep_size=int(topology["ep_size"]),
        gpus_per_node=int(topology["gpus_per_node"]),
        expert_to_sender=expert_map,
        request_to_receiver={str(key): int(value) for key, value in request_map_raw.items()},
    )


def validate_data_manifest(raw: Mapping[str, object], *, formal: bool) -> None:
    calibration = raw.get("calibration_hashes")
    sealed = raw.get("sealed_hashes")
    historical = raw.get("historical_hashes")
    if not all(isinstance(value, list) for value in (calibration, sealed, historical)):
        raise CJCValidationError("data manifest requires calibration/sealed/historical hash lists")
    calib_set = set(str(value) for value in calibration)  # type: ignore[arg-type]
    sealed_set = set(str(value) for value in sealed)  # type: ignore[arg-type]
    historical_set = set(str(value) for value in historical)  # type: ignore[arg-type]
    if calib_set & sealed_set or sealed_set & historical_set or calib_set & historical_set:
        raise CJCValidationError("calibration/sealed/historical data overlap")
    if formal and raw.get("sealed") is not True:
        raise CJCValidationError("formal data manifest must be sealed")
    if raw.get("dataset") != "wikitext/wikitext-103-raw-v1" or raw.get("dataset_split") != "train":
        raise CJCValidationError("dataset does not match cjc-v1")
    for name in ("calibration_manifest_sha256", "sealed_manifest_sha256"):
        value = raw.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise CJCValidationError(f"data registry lacks {name}")
    if not calib_set or not sealed_set:
        raise CJCValidationError("empty calibration or sealed split")
    frozen = {
        "selection_seed": 20260722,
        "calibration_window": [20000, 22000],
        "sealed_window": [40000, 44000],
        "calibration_selected_count": 64,
        "sealed_selected_count": 128,
        "tokens_per_request": 128,
        "tokenizer_min_tokens": 129,
    }
    if formal:
        for key, expected in frozen.items():
            if raw.get(key) != expected:
                raise CJCValidationError(f"data manifest drift for frozen field {key}")


def validate_source_manifest(raw: Mapping[str, object]) -> None:
    files = raw.get("files")
    if not isinstance(files, dict) or not files:
        raise CJCValidationError("source manifest must bind source files")
    required_names = {
        "cjc_policy.py",
        "run_cjc_oracle.py",
        "test_cjc_policy.py",
        "cjc_v1.json",
        "capture_cjc_routes_gpu.py",
        "prepare_cjc_data_manifest.py",
        "build_cjc_data_registry.py",
        "prepare_cjc_calibration.py",
        "run_cjc_lut_gpu.py",
        "merge_cjc_luts.py",
        "capture_moe.py",
    }
    observed_names: set[str] = set()
    for path_value, expected_hash in files.items():
        path = Path(str(path_value))
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file() or sha256_file(path) != str(expected_hash):
            raise CJCValidationError(f"source hash mismatch: {path}")
        observed_names.add(path.name)
    if not required_names.issubset(observed_names):
        raise CJCValidationError("source manifest does not bind all CJC implementation/config files")


def validate_environment(raw: Mapping[str, object], config: Mapping[str, object]) -> None:
    formal_cfg = config.get("formal")
    if not isinstance(formal_cfg, dict):
        raise CJCValidationError("invalid formal config")
    capabilities = raw.get("capabilities")
    if not isinstance(capabilities, dict):
        raise CJCValidationError("environment manifest lacks capabilities")
    missing = [
        name
        for name in formal_cfg["required_capabilities"]  # type: ignore[index]
        if capabilities.get(str(name)) is not True
    ]
    if missing:
        raise CJCValidationError("G0/G1 capability missing: " + ",".join(missing))
    if raw.get("gpu_name") != "NVIDIA GeForce RTX 5090":
        raise CJCValidationError("measured LUT environment is not the frozen RTX 5090")
    if raw.get("h2d_boundary") != "NOT_RDMA":
        raise CJCValidationError("environment must preserve H2D != RDMA boundary")


def validate_signoff(
    signoff_path: str | None,
    *,
    bindings: Mapping[str, str],
) -> None:
    if signoff_path is None:
        raise CJCValidationError("formal mode requires Phase-4 SIGNED-OFF attestation")
    raw = load_json(signoff_path)
    if raw.get("status") != "SIGNED-OFF":
        raise CJCValidationError("Phase-4 status is not SIGNED-OFF")
    for name, expected in bindings.items():
        if raw.get(name) != expected:
            raise CJCValidationError(f"Phase-4 attestation hash drift: {name}")


def calibration_entry(
    calibration: Mapping[str, object], model_key: str, cell: str
) -> tuple[WorkloadSpec, str, dict[str, float]]:
    models = calibration.get("models")
    if not isinstance(models, dict) or not isinstance(models.get(model_key), dict):
        raise CJCValidationError(f"missing calibration model {model_key}")
    model = models[model_key]
    cells = model.get("cells")  # type: ignore[union-attr]
    if not isinstance(cells, dict) or not isinstance(cells.get(cell), dict):
        raise CJCValidationError(f"missing calibration cell {model_key}/{cell}")
    raw = cells[cell]
    workload = WorkloadSpec(
        cell=cell,
        arrival_rate_per_us=float(raw["arrival_rate_per_us"]),
        layer_period_us=float(raw["layer_period_us"]),
        slo_us=float(raw["slo_us"]),
        mmpp_low_multiplier=float(raw.get("mmpp_low_multiplier", 0.5)),
        mmpp_high_multiplier=float(raw.get("mmpp_high_multiplier", 1.5)),
        mmpp_switch_probability=float(raw.get("mmpp_switch_probability", 0.10)),
    )
    static_arm = str(raw["calib_best_static"])
    if static_arm not in JOIN_BLIND_ARMS[:-1]:
        raise CJCValidationError("calibration selected an invalid static baseline")
    if str(raw.get("slo_definition")) != "1.10_x_calibration_best_joinblind_p99":
        raise CJCValidationError("SLO was not frozen from calibration best join-blind P99")
    baseline_p99 = float(raw["calibration_best_joinblind_p99_us"])
    if abs(workload.slo_us - 1.10 * baseline_p99) > max(1e-9, 1e-9 * workload.slo_us):
        raise CJCValidationError("SLO value does not equal 1.10 x calibration P99")
    target_rho = float(raw["target_rho"])
    expected_rho = 0.50 if cell == "steady_rho50" else 0.80
    if abs(target_rho - expected_rho) > 1e-12:
        raise CJCValidationError("workload utilization target drift")
    ack_timing = model.get("ack_timing")  # type: ignore[union-attr]
    if not isinstance(ack_timing, dict) or ack_timing.get("schema_version") != "cjc-ack-timing-v1":
        raise CJCValidationError("ACK timing lacks component-level provenance")
    components = ack_timing.get("components")
    if not isinstance(components, dict):
        raise CJCValidationError("ACK timing components are missing")
    expected_sources = {
        "build_us": "measured_same_run_host_monotonic_ns",
        "serialize_us": "measured_same_run_host_monotonic_ns",
        "wire_us": "analytic_link",
        "parse_us": "measured_same_run_host_monotonic_ns",
        "policy_lookup_us": "measured_same_run_host_monotonic_ns",
    }
    timings: dict[str, float] = {}
    for key, source in expected_sources.items():
        component = components.get(key)
        if not isinstance(component, dict) or component.get("source") != source:
            raise CJCValidationError(f"ACK timing source mismatch for {key}")
        value = float(component.get("value_us", -1.0))
        if value < 0:
            raise CJCValidationError(f"ACK timing is negative for {key}")
        timings[key] = value
    wire = components["wire_us"]
    if int(wire.get("message_bytes", -1)) != ACK_HEADER_BYTES + ACK_RECORD_BYTES:
        raise CJCValidationError("analytic ACK wire bytes do not match one-record aligned message")
    return workload, static_arm, timings


def select_replay_routes(
    routes: list[RouteContribution], selection: Mapping[str, object]
) -> list[RouteContribution]:
    if not routes:
        raise CJCValidationError("cannot select from an empty route trace")
    if selection.get("method") != "smallest_sha256_of_seed_and_identity":
        raise CJCValidationError("unknown replay selection method")
    if selection.get("preserve_all_topk_siblings") is not True:
        raise CJCValidationError("replay selection must preserve complete join sets")
    seed = int(selection["seed"])
    layer_count = int(selection["layers_per_model"])
    token_count = int(selection["token_positions_per_request_layer"])
    layers = sorted({route.layer_id for route in routes})
    selected_layers = set(sorted(
        layers,
        key=lambda layer: hashlib.sha256(
            f"{seed}:{routes[0].model_revision}:{layer}".encode("utf-8")
        ).digest(),
    )[:layer_count])
    if len(selected_layers) != layer_count:
        raise CJCValidationError("route trace has too few layers for frozen replay selection")
    selected_tokens: set[tuple[str, int, int]] = set()
    positions_by_request_layer: dict[tuple[str, int], set[int]] = {}
    for route in routes:
        if route.layer_id in selected_layers:
            positions_by_request_layer.setdefault(
                (route.request_id, route.layer_id), set()
            ).add(route.token_position)
    for (request_id, layer_id), position_set in sorted(positions_by_request_layer.items()):
        positions = sorted(position_set)
        chosen = sorted(
            positions,
            key=lambda position: hashlib.sha256(
                f"{seed}:{request_id}:{layer_id}:{position}".encode("utf-8")
            ).digest(),
        )[:token_count]
        if len(chosen) != token_count:
            raise CJCValidationError("route trace has too few token positions for replay selection")
        selected_tokens.update((request_id, layer_id, position) for position in chosen)
    selected = [
        route for route in routes
        if (route.request_id, route.layer_id, route.token_position) in selected_tokens
    ]
    if not selected:
        raise CJCValidationError("frozen replay selection produced no routes")
    return selected


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _route_row(route: RouteContribution) -> dict[str, object]:
    return asdict(route)


def _task_row(task: object) -> dict[str, object]:
    row = asdict(task)
    row["route_key"] = list(row["route_key"])
    row["join_key"] = list(row["join_key"])
    return row


def gate_pass(summary: Mapping[str, object], config: Mapping[str, object]) -> bool:
    statistics = config["statistics"]
    assert isinstance(statistics, dict)
    return (
        float(summary["p99_gain_ci_low"]) >= float(statistics["p99_gain_lcb_gate"])
        or float(summary["violation_reduction_ci_low"])
        >= float(statistics["violation_reduction_lcb_gate"])
    )


def positive_sensitivity(summary: Mapping[str, object]) -> bool:
    return (
        float(summary["p99_gain_ci_low"]) > 0
        or float(summary["violation_reduction_ci_low"]) > 0
    )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    if config.get("protocol_version") != "cjc-v1":
        raise CJCValidationError("not the frozen cjc-v1 config")
    protocol_path = REPO_ROOT / str(config["protocol_path"])
    if not protocol_path.is_file():
        raise CJCValidationError("frozen protocol file missing")

    data_manifest = load_json(args.data_manifest)
    source_manifest = load_json(args.source_manifest)
    environment = load_json(args.environment)
    calibration = load_json(args.calibration_manifest)
    validate_data_manifest(data_manifest, formal=args.mode == "formal")
    validate_source_manifest(source_manifest)

    placement = load_placement(args.placement, config)
    routes = load_routes(args.route_trace)
    lut = load_lut(args.lut, formal=args.mode == "formal")
    data_sha = sha256_file(args.data_manifest)
    sealed_route_manifest_sha = str(data_manifest["sealed_manifest_sha256"])
    route_by_revision: dict[str, list[RouteContribution]] = {}
    for route in routes:
        route_by_revision.setdefault(route.model_revision, []).append(route)

    required_models = config.get("required_models")
    if not isinstance(required_models, dict):
        raise CJCValidationError("invalid required_models config")
    revision_to_key = {
        str(model["revision"]): key
        for key, model in required_models.items()
        if isinstance(model, dict)
    }
    unexpected = set(route_by_revision) - set(revision_to_key)
    if unexpected:
        raise CJCValidationError(f"unexpected model revisions: {sorted(unexpected)}")
    if args.mode == "formal" and set(route_by_revision) != set(revision_to_key):
        raise CJCValidationError("formal run must contain exactly both frozen models")

    for revision, model_routes in route_by_revision.items():
        model_key = revision_to_key[revision]
        model_cfg = required_models[model_key]
        assert isinstance(model_cfg, dict)
        validate_route_contributions(
            model_routes,
            expected_model_revision=revision,
            top_k=int(model_cfg["top_k"]),
            num_experts=int(model_cfg["num_experts"]),
            placement=placement,
            expected_data_manifest_sha256=sealed_route_manifest_sha,
            formal=args.mode == "formal",
        )
        if args.mode == "formal":
            request_ids = {route.request_id for route in model_routes}
            if len(request_ids) != 128:
                raise CJCValidationError("formal route trace must contain 128 sealed requests per model")
            by_forward_layer: dict[tuple[str, str, int], set[int]] = {}
            for route in model_routes:
                by_forward_layer.setdefault(
                    (route.request_id, route.forward_id, route.layer_id), set()
                ).add(route.token_position)
            expected_positions = set(range(128))
            if any(positions != expected_positions for positions in by_forward_layer.values()):
                raise CJCValidationError("formal forward/layer must contain exactly 128 valid tokens")

    bindings = {
        "protocol_sha256": sha256_file(protocol_path),
        "config_sha256": sha256_file(config_path),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "data_manifest_sha256": data_sha,
        "placement_manifest_sha256": sha256_file(args.placement),
        "lut_sha256": sha256_file(args.lut),
        "calibration_manifest_sha256": sha256_file(args.calibration_manifest),
        "environment_sha256": sha256_file(args.environment),
        "route_trace_sha256": hashlib.sha256(
            "".join(sha256_file(path) for path in args.route_trace).encode("ascii")
        ).hexdigest(),
    }
    if args.mode == "formal":
        validate_environment(environment, config)
        validate_signoff(args.phase4_signoff, bindings=bindings)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dump_json(output / "protocol.json", {"protocol_version": "cjc-v1", **bindings})
    for source, target in (
        (args.environment, "environment.json"),
        (args.source_manifest, "source_manifest.json"),
        (args.data_manifest, "data_manifest.json"),
        (args.placement, "placement.json"),
        (args.lut, "lut.csv"),
    ):
        shutil.copyfile(source, output / target)
    with (output / "route_trace.jsonl").open("w", encoding="utf-8") as handle:
        for route in routes:
            handle.write(json.dumps(_route_row(route), ensure_ascii=False, sort_keys=True) + "\n")

    wire_cfg = config["wire"]
    topology_cfg = config["topology"]
    workload_cfg = config["workload"]
    ack_cfg = config["ack"]
    arms_cfg = config["arms"]
    statistics_cfg = config["statistics"]
    replay_selection = config["replay_selection"]
    assert all(
        isinstance(value, dict)
        for value in (
            wire_cfg, topology_cfg, workload_cfg, ack_cfg, arms_cfg,
            statistics_cfg, replay_selection,
        )
    )
    if (
        int(ack_cfg["header_bytes"]) != ACK_HEADER_BYTES
        or int(ack_cfg["record_bytes"]) != ACK_RECORD_BYTES
        or int(ack_cfg["alignment_bytes"]) != ACK_ALIGNMENT_BYTES
    ):
        raise CJCValidationError("ACK wire constants drifted from frozen implementation")

    task_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    accounting_rows: list[dict[str, object]] = []
    all_episode_metrics: list[EpisodeMetrics] = []
    action_signatures: dict[tuple[str, str, int, str], tuple[str, ...]] = {}
    main_seeds = [int(value) for value in workload_cfg["main_seeds"]]
    cells = [str(value) for value in workload_cfg["cells"]]
    baseline_arms = [str(value) for value in arms_cfg["join_blind"]]

    for revision, model_routes in sorted(route_by_revision.items()):
        model_key = revision_to_key[revision]
        model_cfg = required_models[model_key]
        assert isinstance(model_cfg, dict)
        model_routes = select_replay_routes(model_routes, replay_selection)
        for cell in cells:
            workload, calib_static_arm, timings = calibration_entry(calibration, model_key, cell)
            for run_seed in main_seeds:
                tasks = build_tasks_from_routes(
                    model_routes,
                    lut=lut,
                    placement=placement,
                    workload=workload,
                    seed=run_seed,
                    hidden_size=int(model_cfg["hidden_size"]),
                    dtype_bytes=int(wire_cfg["dtype_bytes"]),
                    descriptor_bytes=int(wire_cfg["descriptor_bytes_per_contribution"]),
                    alignment_bytes=int(wire_cfg["alignment_bytes_per_contribution"]),
                    link_gbps=float(topology_cfg["link_gbps_primary"]),
                )
                task_rows.extend(_task_row(task) for task in tasks)
                results = []
                variants: list[tuple[str, object]] = []
                for baseline in baseline_arms:
                    result = simulate(
                        tasks,
                        arm=baseline,
                        calib_static_arm=calib_static_arm,
                        fallback_arm=str(arms_cfg["fallback"]),
                        starvation_us=workload.slo_us,
                    )
                    variants.append((baseline, result))
                    results.append(result)

                zero = simulate(
                    tasks,
                    arm="global_causal_join",
                    ack=AckConfig(enabled=False),
                    fallback_arm=str(arms_cfg["fallback"]),
                    starvation_us=workload.slo_us,
                )
                variants.append(("global_causal_join_zero_tax", zero))
                results.append(zero)
                sensitivity_values = sorted(
                    {
                        float(ack_cfg["main_staleness_us"]),
                        *[float(value) for value in ack_cfg["sensitivity_staleness_us"]],
                    }
                )
                for staleness in sensitivity_values:
                    charged = simulate(
                        tasks,
                        arm="global_causal_join",
                        ack=AckConfig(
                            enabled=True,
                            staleness_us=staleness,
                            build_us=timings["build_us"],
                            serialize_us=timings["serialize_us"],
                            wire_us=timings["wire_us"],
                            parse_us=timings["parse_us"],
                            policy_lookup_us=timings["policy_lookup_us"],
                        ),
                        fallback_arm=str(arms_cfg["fallback"]),
                        starvation_us=workload.slo_us,
                    )
                    alias = f"global_causal_join_s{int(staleness):02d}_charged"
                    variants.append((alias, charged))
                    results.append(charged)
                assert_arm_equivalence(results, len(tasks))

                for alias, result in variants:
                    action_signatures[(model_key, cell, run_seed, alias)] = tuple(
                        action.task_id for action in result.action_trace
                    )
                    for action in result.action_trace:
                        row = asdict(action)
                        row.update(
                            {
                                "model": model_key,
                                "cell": cell,
                                "seed": run_seed,
                                "variant": alias,
                            }
                        )
                        action_rows.append(row)
                    accounting_rows.append(
                        {
                            "model": model_key,
                            "cell": cell,
                            "seed": run_seed,
                            "arm": alias,
                            "task_count": len(tasks),
                            "task_fingerprint": result.task_fingerprint,
                            "data_bytes": result.data_bytes,
                            "ack_bytes": result.ack_bytes,
                            "ack_messages": result.ack_messages,
                            "coordination_charged_us": result.coordination_charged_us,
                            "policy_charged_us": result.policy_charged_us,
                            "fallback_count": result.fallback_count,
                            "stale_decisions": result.stale_decisions,
                            "starvation_overrides": result.starvation_overrides,
                            "evidence_boundary": config["evidence_boundary"],
                        }
                    )
                    metrics = episode_metrics(tasks, result, slo_us=workload.slo_us)
                    all_episode_metrics.extend(replace(metric, arm=alias) for metric in metrics)

    summaries: dict[str, dict[str, object]] = {}
    for revision in sorted(route_by_revision):
        model_key = revision_to_key[revision]
        for cell in cells:
            subset = [
                row
                for row in all_episode_metrics
                if row.model_revision == revision and row.cell == cell
            ]
            candidate_aliases = [
                "global_causal_join_zero_tax",
                *[
                    f"global_causal_join_s{int(value):02d}_charged"
                    for value in sorted(
                        {
                            float(ack_cfg["main_staleness_us"]),
                            *[
                                float(item)
                                for item in ack_cfg["sensitivity_staleness_us"]
                            ],
                        }
                    )
                ],
            ]
            for candidate_alias in dict.fromkeys(candidate_aliases):
                summary = paired_hierarchical_bootstrap(
                    subset,
                    candidate_arm=candidate_alias,
                    baseline_arms=baseline_arms,
                    n_bootstrap=int(statistics_cfg["n_bootstrap"]),
                    seed=int(statistics_cfg["bootstrap_seed"]),
                )
                summaries[f"{model_key}/{cell}/{candidate_alias}"] = asdict(summary)

    information_collapse: dict[str, bool] = {}
    for model_key in sorted(revision_to_key.values()):
        for cell in cells:
            zero_signatures = [
                action_signatures.get((model_key, cell, seed, "global_causal_join_zero_tax"))
                for seed in main_seeds
            ]
            collapsed = any(
                all(
                    action_signatures.get((model_key, cell, seed, baseline))
                    == zero_signatures[index]
                    for index, seed in enumerate(main_seeds)
                )
                for baseline in baseline_arms
            )
            information_collapse[f"{model_key}/{cell}"] = collapsed

    required_keys = [
        (model_key, cell)
        for model_key in required_models
        for cell in cells
    ]
    zero_pass: dict[str, bool] = {}
    charged_pass: dict[str, bool] = {}
    stale20_positive: dict[str, bool] = {}
    main_stale = int(float(ack_cfg["main_staleness_us"]))
    for model_key, cell in required_keys:
        prefix = f"{model_key}/{cell}"
        zero = summaries.get(prefix + "/global_causal_join_zero_tax")
        charged = summaries.get(prefix + f"/global_causal_join_s{main_stale:02d}_charged")
        stale20 = summaries.get(prefix + "/global_causal_join_s20_charged")
        zero_pass[prefix] = bool(zero and gate_pass(zero, config))
        charged_pass[prefix] = bool(charged and gate_pass(charged, config))
        stale20_positive[prefix] = bool(stale20 and positive_sensitivity(stale20))

    if args.mode == "dev":
        verdict = "NOT_TESTED"
        formal_run_valid = False
        go = False
    else:
        formal_run_valid = True
        go = (
            all(zero_pass.values())
            and all(charged_pass.values())
            and all(stale20_positive.values())
            and not any(information_collapse.values())
        )
        verdict = "GO_TO_STREAMING_RUNTIME_PROTOTYPE" if go else "NO_GO_CJC_V1"

    decision = {
        "protocol_version": "cjc-v1",
        "mode": args.mode,
        "verdict": verdict,
        "go": go,
        "formal_run_valid": formal_run_valid,
        "g2_zero_tax_by_cell": zero_pass,
        "g3_charged_by_cell": charged_pass,
        "staleness20_positive_by_cell": stale20_positive,
        "information_collapse_by_cell": information_collapse,
        "thresholds_unchanged": {
            "p99_gain_lcb": statistics_cfg["p99_gain_lcb_gate"],
            "violation_reduction_lcb": statistics_cfg["violation_reduction_lcb_gate"],
        },
        "runtime_invariants": {
            "full_drain": True,
            "task_set_equivalence": True,
            "data_bytes_equal_across_arms": True,
        },
        "evidence_boundary": config["evidence_boundary"],
    }
    status = {
        "protocol_version": "cjc-v1",
        "phase": 3 if args.mode == "dev" else 5,
        "mode": args.mode,
        "formal_run_valid": formal_run_valid,
        "scientific_verdict": verdict if formal_run_valid else None,
        "go": go,
        "status": "PARTIAL" if args.mode == "dev" else verdict,
        "not_rdma": True,
        "not_serving_p99": True,
    }

    with (output / "task_trace.jsonl").open("w", encoding="utf-8") as handle:
        for row in task_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    _write_csv(
        output / "action_trace.csv",
        action_rows,
        [
            "model", "cell", "seed", "variant", "arm", "task_id", "resource_id",
            "decision_us", "start_us", "completion_us", "visible_missing", "fallback",
            "starvation_override",
        ],
    )
    _write_csv(
        output / "accounting.csv",
        accounting_rows,
        list(accounting_rows[0].keys()) if accounting_rows else [],
    )
    per_episode_rows = [
        {
            "model_revision": row.model_revision,
            "cell": row.cell,
            "seed": row.seed,
            "episode_id": row.episode_id,
            "arm": row.arm,
            "token_count": len(row.token_latencies_us),
            "p99_us": row.p99_us,
            "violation_fraction": row.violation_fraction,
            "slo_us": row.slo_us,
        }
        for row in all_episode_metrics
    ]
    _write_csv(
        output / "per_episode.csv",
        per_episode_rows,
        list(per_episode_rows[0].keys()) if per_episode_rows else [],
    )
    dump_json(output / "paired_bootstrap.json", summaries)
    dump_json(output / "decision.json", decision)
    dump_json(output / "status.json", status)
    (output / "report.md").write_text(
        "# CJC replay status\n\n"
        f"- status: **{status['status']}**\n"
        f"- mode: `{args.mode}`\n"
        "- boundary: route-real/LUT-calibrated single-host replay; NOT RDMA; NOT serving P99.\n"
        "- Phase 3 dev output is not a scientific conclusion.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

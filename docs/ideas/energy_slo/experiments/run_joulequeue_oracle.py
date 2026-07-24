from __future__ import annotations

"""JouleQueue v1 oracle-development runner.

Formal execution is deliberately fail-closed.  This Phase-3 bundle does not
contain a native RTX 5090 activation/surface producer or a real board-energy
queue executor, so it cannot emit a scientific GO/NO-GO verdict.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from joulequeue_policy import (  # noqa: E402
    AmoEStylePolicy,
    CausalJouleQueuePolicy,
    EDFPolicy,
    FestinaLikeProfiledPolicy,
    FixedTimeoutPolicy,
    ImmediatePolicy,
    Job,
    JobIdentity,
    ProtocolError,
    StaticRowsPolicy,
    SurfaceCatalog,
    SurfaceCurve,
    SurfacePoint,
    ThroughputMuQueuePolicy,
    exact_clairvoyant_oracle,
    schedule_metrics,
    simulate_causal,
    validate_review_attestation,
)


PROTOCOL_VERSION = "joulequeue-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=HERE / "configs" / "joulequeue_v1.json",
    )
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--surface", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--review-attestation", type=Path)
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def _status(output_dir: Path, status: str, reason: str, **extra: object) -> None:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "reason": reason,
        "scientific_result_eligible": False,
        "boundary": (
            "CPU scheduling/accounting development only; not full serving, "
            "not measured RTX 5090 board energy, not a paper GO/NO-GO"
        ),
        **extra,
    }
    _write_new(output_dir / "status.json", json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    _write_new(
        output_dir / "report.md",
        "# JouleQueue runner status\n\n"
        f"- status: **{status}**\n"
        f"- reason: {reason}\n"
        "- scientific result eligible: **false**\n",
    )


def _formal_files(config_path: Path, config: Mapping[str, object]) -> dict[str, Path]:
    repository_root = HERE.parents[3]
    formal_gate = config.get("formal_gate")
    if not isinstance(formal_gate, dict):
        raise ProtocolError("formal_gate must be a JSON object")
    logical_names = formal_gate.get("hash_manifest")
    if not isinstance(logical_names, list) or not logical_names:
        raise ProtocolError("formal hash manifest is absent or empty")
    files = {str(name): repository_root / str(name) for name in logical_names}
    if config_path.resolve() not in {path.resolve() for path in files.values()}:
        raise ProtocolError("active config is outside formal hash manifest")
    if any(not path.is_file() for path in files.values()):
        raise ProtocolError("formal hash manifest references a missing file")
    return files


def validate_formal_gate(
    config_path: Path,
    config: Mapping[str, object],
    attestation_path: Path | None,
) -> dict[str, str]:
    if attestation_path is None or not attestation_path.is_file():
        raise ProtocolError("formal run requires a review attestation")
    files = _formal_files(config_path, config)
    attestation = _load_json(attestation_path)
    if not isinstance(attestation, dict):
        raise ProtocolError("review attestation must be a JSON object")
    return validate_review_attestation(
        attestation,
        protocol_version=PROTOCOL_VERSION,
        files=files,
    )


def validate_formal_capabilities(config: Mapping[str, object]) -> None:
    capabilities = config.get("formal_capabilities")
    required = {
        "identity_complete_native_route_producer",
        "native_5090_surface_producer",
        "real_board_energy_queue_executor",
        "full_dependency_replay",
    }
    if not isinstance(capabilities, dict):
        raise ProtocolError("formal_capabilities must be a JSON object")
    if set(capabilities) != required or any(
        value is not True for value in capabilities.values()
    ):
        missing = sorted(
            required - set(capabilities)
            | {key for key, value in capabilities.items() if value is not True}
        )
        raise ProtocolError(
            "formal producers/executor are not implemented: " + ",".join(missing)
        )


def _load_surface(path: Path, *, formal: bool) -> SurfaceCatalog:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ProtocolError("surface artifact must be a JSON object")
    metadata = payload.get("metadata", payload)
    curves = payload.get("curves")
    pooled_surface = payload.get("surface")
    if not isinstance(metadata, dict):
        raise ProtocolError("surface artifact lacks metadata")
    if not isinstance(curves, list) and not isinstance(pooled_surface, dict):
        raise ProtocolError("surface artifact lacks per-expert curves or pooled surface")
    if formal:
        if not isinstance(curves, list) or not curves:
            raise ProtocolError("formal surface requires non-empty per-expert curves")
        required = {
            "artifact_schema": "joulequeue-expert-surface-v1",
            "evidence_level": "REAL_5090_NATIVE_ACTIVATIONS",
            "native_activations": True,
            "formal_eligible": True,
            "phase4_signoff_verified": True,
            "energy_basis": "total_during_launch",
            "all_measurements_valid": True,
            "all_numerical_gates_passed": True,
        }
        for key, value in required.items():
            if metadata.get(key) != value:
                raise ProtocolError(f"formal surface capability missing: {key}")
        if "RTX 5090" not in str(metadata.get("gpu_name", "")):
            raise ProtocolError("formal surface was not measured on RTX 5090")
        energy_source = metadata.get("energy_source")
        if isinstance(energy_source, list) and len(energy_source) == 1:
            energy_source = energy_source[0]
        if energy_source not in {
            "nvml_total_energy_counter",
            "monotonic_power_integral",
        }:
            raise ProtocolError("formal surface has invalid energy source")
        formal_metadata = {
            "paired_order": "AB_BA",
            "counter_sample_logical_window_bracketed": True,
            "counter_sample_boundary_relation": "SEQUENTIAL_BRACKETING_NOT_ATOMIC",
            "background_sampler_exceptions_propagated": True,
        }
        for key, value in formal_metadata.items():
            if metadata.get(key) != value:
                raise ProtocolError(f"formal surface accounting gate missing: {key}")
        if float(metadata.get("minimum_window_s", 0)) < 2.0:
            raise ProtocolError("formal surface window is shorter than two seconds")
        if int(metadata.get("independent_trials", 0)) < 10:
            raise ProtocolError("formal surface has fewer than ten independent trials")
        if float(metadata.get("sampling_interval_ms", float("inf"))) != 5.0:
            raise ProtocolError("formal surface sampling interval is not 5 ms")
        if float(metadata.get("max_observed_gap_ms", float("inf"))) > 20.0:
            raise ProtocolError("formal surface observed NVML gap exceeds 20 ms")
        if not str(metadata.get("gpu_uuid", "")):
            raise ProtocolError("formal surface lacks GPU UUID")
        signoff_sha256 = str(metadata.get("phase4_signoff_sha256", ""))
        if len(signoff_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in signoff_sha256
        ):
            raise ProtocolError("formal surface lacks a valid Phase-4 signoff hash")
        if "@" not in str(metadata.get("model_revision", "")):
            raise ProtocolError("formal surface lacks a pinned model revision")
        if metadata.get("numerical_gate") != {
            "max_abs_error": 0.02,
            "mean_abs_error": 0.002,
            "max_cosine_error": 0.0001,
        }:
            raise ProtocolError("formal surface numerical thresholds drifted")
        failed_gates = metadata.get("formal_eligibility_failed_gates")
        if failed_gates != []:
            raise ProtocolError("formal surface records failed eligibility gates")
    catalog: dict[tuple[int, int], SurfaceCurve] = {}
    if isinstance(curves, list):
        frozen_rows = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        for curve in curves:
            if not isinstance(curve, dict) or not isinstance(curve.get("points"), list):
                raise ProtocolError("malformed surface curve")
            if formal and not (
                curve.get("energy_basis") == "total_during_launch"
                and curve.get("all_measurements_valid") is True
                and curve.get("all_numerical_gates_passed") is True
            ):
                raise ProtocolError("formal surface curve failed accounting or numerical gate")
            key = (int(curve["layer_id"]), int(curve["expert_id"]))
            if key in catalog:
                raise ProtocolError("duplicate layer/expert surface")
            if formal and [int(point["rows"]) for point in curve["points"]] != frozen_rows:
                raise ProtocolError("formal surface curve row grid drifted")
            points = []
            for point in curve["points"]:
                if not isinstance(point, dict):
                    raise ProtocolError("malformed surface point")
                if formal and not (
                    "energy_j" in point
                    and point.get("measurement_valid") is True
                    and point.get("numerical_gate_passed") is True
                    and point.get("paired_order_complete") is True
                    and int(point.get("independent_trials", 0)) >= 10
                ):
                    raise ProtocolError("formal surface point failed a frozen gate")
                energy_value = point.get("energy_j", point.get("dynamic_energy_j"))
                if energy_value is None:
                    raise ProtocolError("surface point lacks an energy value")
                points.append(SurfacePoint(
                    rows=int(point["rows"]),
                    energy_j=float(energy_value),
                    latency_us=float(point["latency_us"]),
                    energy_ucb95_j=(
                        float(point["energy_ucb95_j"])
                        if point.get("energy_ucb95_j") is not None
                        else None
                    ),
                    latency_ucb95_us=(
                        float(point["latency_ucb95_us"])
                        if point.get("latency_ucb95_us") is not None
                        else None
                    ),
                ))
            catalog[key] = SurfaceCurve(points)
        if formal:
            layers = {layer_id for layer_id, _expert_id in catalog}
            if len(catalog) != 16 or len(layers) != 4 or any(
                sum(layer_id == layer for layer_id, _expert_id in catalog) != 4
                for layer in layers
            ):
                raise ProtocolError("formal surface must contain four experts in each of four layers")
        energy_basis = str(metadata.get("energy_basis", "dynamic_incremental"))
        return SurfaceCatalog(catalog, energy_basis=energy_basis)

    # The independently implemented surface producer emits a calibration-pooled
    # curve under ``surface``.  It is accepted only as a default development
    # curve; the formal capability gate remains false until per-expert identity
    # and the reviewed executor are integrated.
    assert isinstance(pooled_surface, dict)
    pooled_points = []
    for raw_rows, point in pooled_surface.items():
        if not isinstance(point, dict):
            raise ProtocolError("malformed pooled surface point")
        pooled_points.append(SurfacePoint(
            rows=int(raw_rows),
            energy_j=float(point["coalesced_energy_mean_j"]),
            latency_us=float(point["coalesced_latency_mean_us"]),
            energy_ucb95_j=float(point["coalesced_energy_ucb95_j"]),
            latency_ucb95_us=float(point["coalesced_latency_ucb95_us"]),
        ))
    return SurfaceCatalog(
        {},
        default_curve=SurfaceCurve(pooled_points),
        energy_basis="total_during_launch",
    )


def _load_jobs(path: Path, *, formal: bool) -> list[Job]:
    payload = _load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("jobs"), list):
        raise ProtocolError("job artifact must contain a jobs list")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ProtocolError("job artifact lacks metadata")
    if formal and not (
        metadata.get("identity_complete") is True
        and metadata.get("native_route") is True
        and metadata.get("full_dependency_replay") is True
        and metadata.get("route_closure_validated") is True
    ):
        raise ProtocolError("formal job artifact is not identity-complete native replay")
    jobs: list[Job] = []
    for item in payload["jobs"]:
        identity = JobIdentity(
            request_id=str(item["request_id"]),
            forward_id=int(item["forward_id"]),
            layer_id=int(item["layer_id"]),
            token_id=int(item["token_id"]),
            route_slot=int(item["route_slot"]),
            expert_id=int(item["expert_id"]),
        )
        jobs.append(Job(
            identity=identity,
            arrival_us=float(item["arrival_us"]),
            rows=int(item["rows"]),
            deadline_us=float(item["deadline_us"]),
            activation_sha256=item.get("activation_sha256"),
        ))
    return jobs


def run_development(config: Mapping[str, object], jobs: list[Job], surfaces: SurfaceCatalog) -> dict[str, object]:
    policy = config["policy"]  # type: ignore[index]
    idle_power_w = float(config["accounting"]["idle_power_w_dev_only"])  # type: ignore[index]
    slo_us = float(config["slo"]["dev_slo_us"])  # type: ignore[index]
    arms = {
        "immediate": ImmediatePolicy(),
        "best_fixed_timeout": FixedTimeoutPolicy(float(policy["fixed_timeout_us"])),
        "best_static_rows": StaticRowsPolicy(
            int(policy["static_rows"]), float(policy["max_age_us"])
        ),
        "edf": EDFPolicy(),
        "throughput_muqueue": ThroughputMuQueuePolicy(
            int(policy["throughput_rows"]), float(policy["max_age_us"])
        ),
        "amoe_style": AmoEStylePolicy(
            int(policy["amoe_rows"]), float(policy["max_age_us"])
        ),
        "festina_like_profiled": FestinaLikeProfiledPolicy(
            int(policy["festina_rows"]), float(policy["max_age_us"])
        ),
        "causal_joulequeue": CausalJouleQueuePolicy(
            target_rows=int(policy["causal_target_rows"]),
            min_saving_fraction=float(policy["causal_min_saving_fraction"]),
            max_age_us=float(policy["max_age_us"]),
            urgent_margin_us=float(policy["urgent_margin_us"]),
        ),
    }
    metrics: dict[str, object] = {}
    for name, arm_policy in arms.items():
        result = simulate_causal(
            jobs, surfaces, arm_policy, idle_power_w=idle_power_w, arm=name
        )
        metric = schedule_metrics(jobs, result, slo_us=slo_us)
        metrics[name] = {
            "completed_jobs": metric.completed_jobs,
            "completed_tokens": metric.completed_tokens,
            "board_j_per_completed_token": metric.board_j_per_completed_token,
            "p99_token_completion_us": metric.p99_token_completion_us,
            "p99_tpot_proxy_us": metric.p99_tpot_proxy_us,
            "slo_violation_rate": metric.slo_violation_rate,
        }
    oracle = exact_clairvoyant_oracle(
        jobs,
        surfaces,
        idle_power_w=idle_power_w,
        max_jobs=int(config["oracle"]["max_exact_jobs"]),  # type: ignore[index]
        max_age_us=float(policy["max_age_us"]),
    )
    oracle_metric = schedule_metrics(jobs, oracle, slo_us=slo_us)
    metrics["clairvoyant_energy_oracle"] = {
        "completed_jobs": oracle_metric.completed_jobs,
        "completed_tokens": oracle_metric.completed_tokens,
        "board_j_per_completed_token": oracle_metric.board_j_per_completed_token,
        "p99_token_completion_us": oracle_metric.p99_token_completion_us,
        "p99_tpot_proxy_us": oracle_metric.p99_tpot_proxy_us,
        "slo_violation_rate": oracle_metric.slo_violation_rate,
    }
    return metrics


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise RuntimeError("refusing to overwrite an existing output directory")
    config = _load_json(args.config)
    if not isinstance(config, dict) or config.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError("config protocol version mismatch")
    try:
        if args.formal:
            hashes = validate_formal_gate(args.config, config, args.review_attestation)
            validate_formal_capabilities(config)
        else:
            hashes = {}
        if args.jobs is None or args.surface is None:
            raise ProtocolError("runner requires explicit job and surface artifacts")
        jobs = _load_jobs(args.jobs, formal=args.formal)
        surfaces = _load_surface(args.surface, formal=args.formal)
        metrics = run_development(config, jobs, surfaces)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        _write_new(args.output_dir / "development_metrics.json", json.dumps(metrics, indent=2) + "\n")
        _status(
            args.output_dir,
            "PARTIAL_DEVELOPMENT_ONLY",
            "logic replay completed; native GPU producer/executor evidence is absent",
            reviewed_file_sha256=hashes,
        )
        return 0
    except (OSError, ValueError, KeyError, ProtocolError) as exc:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        _status(args.output_dir, "BLOCKED", str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

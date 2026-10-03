#!/usr/bin/env python3
"""Frozen document-paired analysis and decision logic for RouteGuard-KV R0-A."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from r0a_artifacts import (
    ArtifactError,
    load_config,
    load_json,
    write_json_no_overwrite,
    write_jsonl_no_overwrite,
)


class AnalysisError(RuntimeError):
    pass


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise AnalysisError(f"non-finite {label}: {value}")
    return result


def _quantile_interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def paired_bootstrap(
    free: np.ndarray,
    router: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> dict[str, float]:
    if free.ndim != 1 or router.shape != free.shape or free.size < 2:
        raise AnalysisError("paired bootstrap requires equal 1-D arrays with at least two documents")
    if replicates <= 0:
        raise AnalysisError("bootstrap replicate count must be positive")
    rng = np.random.default_rng(seed)
    n = free.size
    router_means = np.empty(replicates, dtype=np.float64)
    shares = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.integers(0, n, size=n)
        free_mean = float(free[selected].mean())
        router_mean = float(router[selected].mean())
        router_means[index] = router_mean
        shares[index] = router_mean / free_mean if free_mean != 0.0 else np.nan
    if not np.isfinite(shares).all():
        raise AnalysisError("router share bootstrap has a zero/non-finite denominator")
    router_low, router_high = _quantile_interval(router_means, confidence)
    share_low, share_high = _quantile_interval(shares, confidence)
    return {
        "router_contrast_mean": float(router.mean()),
        "router_contrast_ci_low": router_low,
        "router_contrast_ci_high": router_high,
        "router_share_ratio_of_means": float(router.mean() / free.mean()),
        "router_share_ci_low": share_low,
        "router_share_ci_high": share_high,
    }


def _trajectory_key(row: Mapping[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(row["text_sha256"]),
        int(row["prompt_length"]),
        str(row["target"]),
        str(row["arm"]),
    )


def load_trajectory_rows(output_dir: Path) -> list[dict[str, Any]]:
    trajectory_dir = output_dir / "trajectories"
    if not trajectory_dir.is_dir():
        raise AnalysisError(f"trajectory directory is missing: {trajectory_dir}")
    rows: list[dict[str, Any]] = []
    for path in sorted(trajectory_dir.glob("*.json")):
        value = load_json(path)
        if not isinstance(value, dict):
            raise AnalysisError(f"trajectory is not an object: {path}")
        rows.append(value)
    return rows


def expected_keys(config: Mapping[str, Any], document_hashes: Iterable[str]) -> set[tuple[str, int, str, str]]:
    keys: set[tuple[str, int, str, str]] = set()
    prompt_lengths = [int(value) for value in config["dataset"]["prompt_lengths"]]
    for digest in document_hashes:
        for prompt_length in prompt_lengths:
            keys.add((digest, prompt_length, "bf16", "bf16_reference"))
            keys.add((digest, prompt_length, "identity", "identity_free"))
        for target in config["quantization"]["targets"]:
            for prompt_length in target["prompt_lengths"]:
                for arm in ("free", "set_locked", "fully_locked"):
                    keys.add((digest, int(prompt_length), str(target["name"]), arm))
    return keys


def _invalid_decision(integrity: Mapping[str, Any]) -> str | None:
    if integrity.get("status") == "PASS":
        return None
    code = str(integrity.get("decision_code", "INVALID_STATE_OR_NUMERICAL_CONTROL"))
    if not code.startswith("INVALID_"):
        raise AnalysisError(f"integrity failure did not provide INVALID code: {code}")
    return code


def analyze_rows(
    config: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    integrity: Mapping[str, Any],
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    invalid = _invalid_decision(integrity)
    by_key: dict[tuple[str, int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = _trajectory_key(row)
        if key in by_key:
            raise AnalysisError(f"duplicate trajectory key: {key}")
        by_key[key] = row
        if int(row.get("completed_steps", -1)) != int(config["dataset"]["decode_steps"]):
            invalid = "INVALID_INCOMPLETE_RUN"

    primary = config["statistics"]["primary_cell"]
    target = str(primary["target"])
    prompt_length = int(primary["prompt_length"])
    document_hashes = sorted(
        {
            str(row["text_sha256"])
            for row in rows
            if int(row["prompt_length"]) == prompt_length
            and str(row["target"]) == target
            and str(row["arm"]) == "free"
        }
    )
    expected_documents = int(config["dataset"]["sealed_documents"])
    if require_complete and len(document_hashes) != expected_documents:
        invalid = "INVALID_INCOMPLETE_RUN"
    if require_complete:
        expected = expected_keys(config, document_hashes)
        actual = set(by_key)
        if actual != expected:
            invalid = "INVALID_INCOMPLETE_RUN"

    per_document: list[dict[str, Any]] = []
    for digest in document_hashes:
        try:
            free_row = by_key[(digest, prompt_length, target, "free")]
            set_row = by_key[(digest, prompt_length, target, "set_locked")]
            full_row = by_key[(digest, prompt_length, target, "fully_locked")]
            identity_row = by_key[(digest, prompt_length, "identity", "identity_free")]
        except KeyError:
            invalid = "INVALID_INCOMPLETE_RUN"
            continue
        free_kl = _finite(free_row["mean_kl"], "free KL")
        set_kl = _finite(set_row["mean_kl"], "set-locked KL")
        full_kl = _finite(full_row["mean_kl"], "fully-locked KL")
        identity_kl = _finite(identity_row["mean_kl"], "identity KL")
        route = free_row.get("route_metrics", {})
        per_document.append(
            {
                "text_sha256": digest,
                "free_kl": free_kl,
                "set_locked_kl": set_kl,
                "fully_locked_kl": full_kl,
                "identity_kl": identity_kl,
                "delta_set": free_kl - set_kl,
                "delta_weight": set_kl - full_kl,
                "delta_router": free_kl - full_kl,
                "delta_numeric": full_kl,
                "set_flip_count": int(route.get("set_flip_count", 0)),
                "non_tie_set_flip_count": int(route.get("non_tie_set_flip_count", 0)),
                "non_tie_cell_count": int(route.get("non_tie_cell_count", 0)),
                "route_cell_count": int(route.get("route_cell_count", 0)),
            }
        )

    if not per_document:
        return {
            "schema_version": "routeguard-kv-r0a-decision-v1",
            "decision_code": invalid or "INVALID_INCOMPLETE_RUN",
            "integrity_status": integrity.get("status"),
            "completed_primary_documents": 0,
            "gates": {},
            "per_document": [],
            "evidence_boundary": config["evidence_boundary"],
        }

    free = np.asarray([row["free_kl"] for row in per_document], dtype=np.float64)
    identity = np.asarray([row["identity_kl"] for row in per_document], dtype=np.float64)
    router = np.asarray([row["delta_router"] for row in per_document], dtype=np.float64)
    bootstrap = paired_bootstrap(
        free,
        router,
        replicates=int(config["statistics"]["bootstrap_replicates"]),
        seed=int(config["statistics"]["bootstrap_seed"]),
        confidence=float(config["statistics"]["confidence_interval"]),
    )
    total_flips = sum(row["set_flip_count"] for row in per_document)
    non_tie_flips = sum(row["non_tie_set_flip_count"] for row in per_document)
    non_tie_cells = sum(row["non_tie_cell_count"] for row in per_document)
    route_cells = sum(row["route_cell_count"] for row in per_document)
    non_tie_flip_rate = non_tie_flips / non_tie_cells if non_tie_cells else 0.0
    non_tie_fraction_of_flips = non_tie_flips / total_flips if total_flips else 0.0
    loo_values = [float(np.delete(router, index).mean()) for index in range(router.size)]
    identity_mean = float(identity.mean())
    free_mean = float(free.mean())
    denominator = max(identity_mean, float(primary["identity_kl_denominator_floor"]))

    gates = {
        "total_effect": {
            "mean_free_kl": free_mean,
            "minimum": float(primary["mean_free_kl_min"]),
            "identity_mean_kl": identity_mean,
            "free_over_identity": free_mean / denominator,
            "ratio_minimum": float(primary["free_kl_over_identity_min"]),
            "pass": free_mean >= float(primary["mean_free_kl_min"])
            and free_mean / denominator >= float(primary["free_kl_over_identity_min"]),
        },
        "route_set": {
            "route_cell_count": route_cells,
            "non_tie_cell_count": non_tie_cells,
            "set_flip_count": total_flips,
            "non_tie_set_flip_count": non_tie_flips,
            "non_tie_set_flip_rate": non_tie_flip_rate,
            "minimum": float(primary["non_tie_set_flip_rate_min"]),
            "non_tie_fraction_of_flips": non_tie_fraction_of_flips,
            "non_tie_fraction_minimum": float(primary["non_tie_fraction_of_flips_min"]),
            "pass": non_tie_flip_rate >= float(primary["non_tie_set_flip_rate_min"])
            and non_tie_fraction_of_flips >= float(primary["non_tie_fraction_of_flips_min"]),
        },
        "router_mediation": {
            **bootstrap,
            "router_contrast_lcb_min_exclusive": float(
                primary["router_contrast_lcb_min_exclusive"]
            ),
            "router_share_point_min": float(primary["router_share_point_min"]),
            "router_share_lcb_min_exclusive": float(primary["router_share_lcb_min_exclusive"]),
            "leave_one_document_out_means": loo_values,
            "leave_one_document_out_all_positive": all(value > 0.0 for value in loo_values),
        },
    }
    gates["router_mediation"]["pass"] = (
        bootstrap["router_contrast_ci_low"]
        > float(primary["router_contrast_lcb_min_exclusive"])
        and bootstrap["router_share_ratio_of_means"] >= float(primary["router_share_point_min"])
        and bootstrap["router_share_ci_low"] > float(primary["router_share_lcb_min_exclusive"])
        and gates["router_mediation"]["leave_one_document_out_all_positive"]
    )

    if invalid:
        decision = invalid
    elif not gates["total_effect"]["pass"]:
        decision = "NO_GO_TOTAL_KV_EFFECT_TOO_SMALL"
    elif not gates["route_set"]["pass"]:
        decision = "NO_GO_ROUTESET_CHANGE_TOO_RARE"
    elif not gates["router_mediation"]["pass"]:
        decision = "NO_GO_ROUTE_MEDIATION_TOO_SMALL"
    else:
        decision = "PASS_R0A_ROUTE_MEDIATED_KV_EFFECT_R0B_ONLY"
    if decision not in config["decision_codes"]:
        raise AnalysisError(f"decision is not frozen in config: {decision}")
    return {
        "schema_version": "routeguard-kv-r0a-decision-v1",
        "decision_code": decision,
        "integrity_status": integrity.get("status"),
        "completed_primary_documents": len(per_document),
        "gates": gates,
        "per_document": per_document,
        "secondary_cells_can_rescue_primary": False,
        "evidence_boundary": config["evidence_boundary"],
    }


def flatten_steps(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        base = {key: row[key] for key in ("text_sha256", "prompt_length", "target", "arm")}
        for step in row.get("steps", []):
            flattened.append({**base, **step})
    return flattened


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    rows = load_trajectory_rows(args.output_dir)
    integrity_path = args.output_dir / "integrity.json"
    if not integrity_path.is_file():
        raise AnalysisError("integrity.json is missing")
    integrity = load_json(integrity_path)
    decision = analyze_rows(config, rows, integrity, require_complete=True)
    metadata = load_json(args.output_dir / "metadata.json")
    environment = load_json(args.output_dir / "environment.json")
    bindings = load_json(args.output_dir / "frozen_bindings.json")
    expected_count = int(config["dataset"]["sealed_documents"]) * 19
    decision["run_evidence"] = {
        "completed_trajectory_count": len(rows),
        "expected_trajectory_count": expected_count,
        "resume_count": integrity.get("resume_count"),
        "model": f"{config['model']['repo_id']}@{config['model']['revision']}",
        "dataset": (
            f"{config['dataset']['repo_id']}/{config['dataset']['config']}"
            f"@{config['dataset']['revision']}:{config['dataset']['split']}"
        ),
        "gpu": environment.get("gpu_name"),
        "compute_capability": environment.get("compute_capability"),
        "software": {
            key: environment.get(key)
            for key in ("python", "torch", "transformers", "datasets", "huggingface_hub", "numpy")
        },
        "manifest_sha256": metadata.get("manifest_sha256"),
        "data_provenance": metadata.get("data_provenance"),
        "frozen_bindings_sha256": metadata.get("frozen_bindings_sha256"),
        "bound_file_hashes": bindings.get("files"),
        "peak_allocated_gib": integrity.get("peak_allocated_gib"),
    }
    write_jsonl_no_overwrite(args.output_dir / "per_step.jsonl", flatten_steps(rows))
    write_jsonl_no_overwrite(args.output_dir / "per_document.jsonl", decision["per_document"])
    write_json_no_overwrite(args.output_dir / "summary.json", {"gates": decision["gates"]})
    write_json_no_overwrite(args.output_dir / "decision.json", decision)
    print(f"R0A_DECISION {decision['decision_code']}")


if __name__ == "__main__":
    try:
        main()
    except (ArtifactError, AnalysisError, KeyError, ValueError) as exc:
        raise SystemExit(f"R0A_ANALYSIS_FAILED: {exc}") from exc

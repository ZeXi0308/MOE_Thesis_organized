#!/usr/bin/env python3
"""Fail-closed upper-bound gate for the CPR-MoE RankLane actuator.

This program intentionally performs no GPU timing. It combines already-recorded
single-GPU quality/codec evidence with an exact Amdahl upper bound. It cannot
validate the existence or magnitude of an exposed EP return path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_QUALITY_COLUMNS = {
    "policy",
    "byte_saving",
    "mean_kl",
    "kl_ci_low",
    "kl_ci_high",
}


class EvidenceError(ValueError):
    """Raised when an input artifact violates the frozen evidence contract."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"JSON root must be an object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_relative_improvement(
    baseline_saving: float, candidate_saving: float, exposed_fraction: float
) -> float:
    """Return (T_baseline - T_candidate) / T_baseline.

    Total BF16 time is normalized to 1.0. ``exposed_fraction`` is the raw BF16
    return-path fraction. Byte saving is assumed to reduce that path linearly,
    with zero codec/launch/queueing overhead: an intentionally optimistic bound.
    """
    for name, value in (
        ("baseline_saving", baseline_saving),
        ("candidate_saving", candidate_saving),
        ("exposed_fraction", exposed_fraction),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise EvidenceError(f"{name} must be finite and within [0, 1]: {value}")
    if candidate_saving < baseline_saving:
        raise EvidenceError("candidate saving must not be below baseline saving")
    baseline_time = 1.0 - exposed_fraction * baseline_saving
    if baseline_time <= 0.0:
        raise EvidenceError("normalized baseline time must be positive")
    return exposed_fraction * (candidate_saving - baseline_saving) / baseline_time


def required_exposed_fraction(
    baseline_saving: float, candidate_saving: float, target_improvement: float
) -> float:
    """Invert ``exact_relative_improvement`` for the required raw path fraction."""
    if not 0.0 < target_improvement < 1.0:
        raise EvidenceError("target improvement must be within (0, 1)")
    delta = candidate_saving - baseline_saving
    if delta <= 0.0:
        return math.inf
    return target_improvement / (delta + target_improvement * baseline_saving)


def _number(row: dict[str, str], key: str, path: Path) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceError(f"invalid numeric field {key!r} in {path}") from exc
    if not math.isfinite(value):
        raise EvidenceError(f"non-finite field {key!r} in {path}")
    return value


def read_quality_input(repo_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    for key in ("model_key", "summary_csv", "metadata_json"):
        if not isinstance(spec.get(key), str) or not spec[key]:
            raise EvidenceError(f"quality input missing non-empty {key!r}")
    summary_path = repo_root / spec["summary_csv"]
    metadata_path = repo_root / spec["metadata_json"]
    metadata = load_json(metadata_path)
    if metadata.get("model_key") != spec["model_key"]:
        raise EvidenceError(
            f"model_key mismatch for {metadata_path}: "
            f"{metadata.get('model_key')!r} != {spec['model_key']!r}"
        )
    if not isinstance(metadata.get("samples"), int) or metadata["samples"] <= 0:
        raise EvidenceError(f"metadata samples must be a positive integer: {metadata_path}")
    boundary = metadata.get("evidence_boundary")
    if not isinstance(boundary, str) or "No decode-loop" not in boundary:
        raise EvidenceError(f"missing expected evidence boundary: {metadata_path}")
    metadata_savings = metadata.get("byte_saving")
    if not isinstance(metadata_savings, dict):
        raise EvidenceError(f"missing byte_saving map: {metadata_path}")

    try:
        with summary_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not REQUIRED_QUALITY_COLUMNS.issubset(reader.fieldnames):
                raise EvidenceError(
                    f"quality CSV missing columns {sorted(REQUIRED_QUALITY_COLUMNS)}: {summary_path}"
                )
            raw_rows = list(reader)
    except OSError as exc:
        raise EvidenceError(f"cannot read quality CSV: {summary_path}: {exc}") from exc
    if not raw_rows:
        raise EvidenceError(f"quality CSV is empty: {summary_path}")

    rows: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        policy = raw.get("policy", "")
        if not policy or policy in rows:
            raise EvidenceError(f"empty or duplicate policy in {summary_path}: {policy!r}")
        parsed = {
            "policy": policy,
            "byte_saving": _number(raw, "byte_saving", summary_path),
            "mean_kl": _number(raw, "mean_kl", summary_path),
            "kl_ci_low": _number(raw, "kl_ci_low", summary_path),
            "kl_ci_high": _number(raw, "kl_ci_high", summary_path),
        }
        if not 0.0 <= parsed["byte_saving"] <= 1.0:
            raise EvidenceError(f"byte saving outside [0, 1] in {summary_path}: {policy}")
        if parsed["kl_ci_low"] > parsed["mean_kl"] or parsed["mean_kl"] > parsed["kl_ci_high"]:
            raise EvidenceError(f"invalid KL interval ordering in {summary_path}: {policy}")
        meta_saving = metadata_savings.get(policy)
        try:
            meta_saving_number = float(meta_saving)
        except (TypeError, ValueError) as exc:
            raise EvidenceError(
                f"metadata byte saving is not numeric: {metadata_path}: {policy}"
            ) from exc
        if not math.isfinite(meta_saving_number) or not math.isclose(
            meta_saving_number, parsed["byte_saving"], rel_tol=0.0, abs_tol=1e-12
        ):
            raise EvidenceError(f"CSV/metadata byte-saving mismatch: {summary_path}: {policy}")
        rows[policy] = parsed

    return {
        "model_key": spec["model_key"],
        "model": metadata.get("model"),
        "samples": metadata["samples"],
        "top_k": metadata.get("top_k"),
        "evidence_boundary": boundary,
        "rows": rows,
        "summary_path": summary_path,
        "metadata_path": metadata_path,
    }


def choose_max_saving(rows: dict[str, dict[str, Any]], prefix: str) -> dict[str, Any]:
    candidates = [row for name, row in rows.items() if name.startswith(prefix)]
    if not candidates:
        raise EvidenceError(f"no candidate policy starts with {prefix!r}")
    return max(candidates, key=lambda row: (row["byte_saving"], -row["kl_ci_high"], row["policy"]))


def choose_under_quality_budget(
    rows: dict[str, dict[str, Any]], prefix: str, budget: float
) -> dict[str, Any] | None:
    candidates = [
        row
        for name, row in rows.items()
        if name.startswith(prefix) and row["kl_ci_high"] <= budget
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (row["byte_saving"], -row["kl_ci_high"], row["policy"]))


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "experiment_id",
        "hypothesis",
        "baseline_policy",
        "candidate_prefix",
        "exposed_return_fraction_grid",
        "exposed_return_fraction_max",
        "min_relative_improvement",
        "quality_kl_ci_high_budgets",
        "quality_inputs",
        "codec_metadata_json",
    }
    missing = required - config.keys()
    if missing:
        raise EvidenceError(f"config missing keys: {sorted(missing)}")
    if config["schema_version"] != 1:
        raise EvidenceError(f"unsupported schema_version: {config['schema_version']!r}")
    for key in ("experiment_id", "hypothesis", "baseline_policy", "candidate_prefix"):
        if not isinstance(config[key], str) or not config[key]:
            raise EvidenceError(f"{key} must be a non-empty string")
    grid = config["exposed_return_fraction_grid"]
    if not isinstance(grid, list) or not grid:
        raise EvidenceError("exposed_return_fraction_grid must be a non-empty list")
    if any(not isinstance(value, (int, float)) or not 0.0 < value < 1.0 for value in grid):
        raise EvidenceError("all exposed-return fractions must be numeric and within (0, 1)")
    if sorted(set(grid)) != grid:
        raise EvidenceError("exposed-return fractions must be unique and increasing")
    if not math.isclose(grid[-1], config["exposed_return_fraction_max"], abs_tol=1e-12):
        raise EvidenceError("grid maximum must equal exposed_return_fraction_max")
    target = config["min_relative_improvement"]
    if not isinstance(target, (int, float)) or not 0.0 < target < 1.0:
        raise EvidenceError("min_relative_improvement must be numeric and within (0, 1)")
    if not isinstance(config["quality_inputs"], list) or len(config["quality_inputs"]) < 2:
        raise EvidenceError("at least two quality inputs are required for cross-model AND")
    if any(not isinstance(spec, dict) for spec in config["quality_inputs"]):
        raise EvidenceError("each quality input must be an object")
    budgets = config["quality_kl_ci_high_budgets"]
    if not isinstance(budgets, list) or not budgets:
        raise EvidenceError("quality_kl_ci_high_budgets must be a non-empty list")
    if any(not isinstance(value, (int, float)) or value < 0.0 for value in budgets):
        raise EvidenceError("quality KL budgets must be non-negative numbers")
    if not isinstance(config["codec_metadata_json"], str) or not config["codec_metadata_json"]:
        raise EvidenceError("codec_metadata_json must be a non-empty string")


def analyze(repo_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    baseline_policy = config["baseline_policy"]
    prefix = config["candidate_prefix"]
    target = float(config["min_relative_improvement"])
    p_max = float(config["exposed_return_fraction_max"])
    quality_inputs = [read_quality_input(repo_root, spec) for spec in config["quality_inputs"]]

    model_results: list[dict[str, Any]] = []
    matrix: list[dict[str, Any]] = []
    for quality in quality_inputs:
        baseline = quality["rows"].get(baseline_policy)
        if baseline is None:
            raise EvidenceError(f"missing baseline policy {baseline_policy!r}: {quality['model_key']}")
        candidate = choose_max_saving(quality["rows"], prefix)
        if candidate["byte_saving"] <= baseline["byte_saving"]:
            raise EvidenceError(f"candidate does not save more bytes than baseline: {quality['model_key']}")
        upper_at_max = exact_relative_improvement(
            baseline["byte_saving"], candidate["byte_saving"], p_max
        )
        required_p = required_exposed_fraction(
            baseline["byte_saving"], candidate["byte_saving"], target
        )
        model_pass = upper_at_max >= target
        quality_sensitivity: list[dict[str, Any]] = []
        for budget in config["quality_kl_ci_high_budgets"]:
            selected = choose_under_quality_budget(quality["rows"], prefix, float(budget))
            quality_sensitivity.append(
                {
                    "kl_ci_high_budget": float(budget),
                    "policy": selected["policy"] if selected else None,
                    "byte_saving": selected["byte_saving"] if selected else None,
                    "upper_bound_at_p_max": (
                        exact_relative_improvement(
                            baseline["byte_saving"], selected["byte_saving"], p_max
                        )
                        if selected
                        else None
                    ),
                }
            )
        for exposed_fraction in config["exposed_return_fraction_grid"]:
            improvement = exact_relative_improvement(
                baseline["byte_saving"], candidate["byte_saving"], float(exposed_fraction)
            )
            matrix.append(
                {
                    "model_key": quality["model_key"],
                    "baseline_policy": baseline["policy"],
                    "candidate_policy": candidate["policy"],
                    "baseline_byte_saving": baseline["byte_saving"],
                    "candidate_byte_saving": candidate["byte_saving"],
                    "exposed_return_fraction": float(exposed_fraction),
                    "zero_codec_relative_improvement": improvement,
                    "min_required_improvement": target,
                    "pass": improvement >= target,
                }
            )
        model_results.append(
            {
                "model_key": quality["model_key"],
                "model": quality["model"],
                "samples": quality["samples"],
                "top_k": quality["top_k"],
                "baseline": baseline,
                "quality_free_max_saving_candidate": candidate,
                "zero_codec_upper_bound_at_p_max": upper_at_max,
                "required_exposed_fraction_for_target": required_p,
                "captured_remaining_return_headroom": (
                    (candidate["byte_saving"] - baseline["byte_saving"])
                    / (1.0 - baseline["byte_saving"])
                ),
                "pass": model_pass,
                "quality_sensitivity": quality_sensitivity,
                "quality_evidence_boundary": quality["evidence_boundary"],
            }
        )

    codec_path = repo_root / config["codec_metadata_json"]
    codec = load_json(codec_path)
    incremental = codec.get("incremental_fp8_to_int4")
    if not isinstance(incremental, dict):
        raise EvidenceError(f"codec metadata missing incremental_fp8_to_int4: {codec_path}")
    n_cells = incremental.get("n_serving_cells")
    viable_count = incremental.get("viable_count")
    if not isinstance(n_cells, int) or n_cells <= 0 or not isinstance(viable_count, int):
        raise EvidenceError(f"invalid codec cell summary: {codec_path}")
    source = codec.get("source", "")
    p95_independent = not (isinstance(source, str) and "p95 sample arrays not re-fetched" in source)
    codec_summary = {
        "gpu": codec.get("gpu"),
        "incremental_fp8_to_int4_viable_count": viable_count,
        "incremental_fp8_to_int4_n_cells": n_cells,
        "p95_is_independent_measurement": p95_independent,
        "source": source,
        "evidence_boundary": codec.get("evidence_boundary"),
        "role_in_primary_decision": "descriptive_only; primary bound assumes zero codec cost",
    }

    overall_pass = all(item["pass"] for item in model_results)
    p_tag = f"{p_max:.2f}".replace(".", "_")
    decision_code = (
        "GO_TO_8XA100_RANKLANE_GATE"
        if overall_pass
        else f"NO_GO_RANKLANE_ACTUATOR_UNDER_P_RETURN_MAX_{p_tag}"
    )
    return {
        "experiment_id": config["experiment_id"],
        "hypothesis": config["hypothesis"],
        "primary_gate": {
            "domain": f"raw BF16 exposed return fraction <= {p_max:.2f}",
            "assumption": "zero codec, launch, queueing, and metadata overhead",
            "cross_model_rule": "AND",
            "target_relative_improvement": target,
            "pass": overall_pass,
        },
        "decision": {
            "code": decision_code,
            "ranklane_hypothesis_status": (
                "SUPPORTED_WITHIN_FROZEN_DOMAIN" if overall_pass else "FALSIFIED_WITHIN_FROZEN_DOMAIN"
            ),
            "p1_return_path_existence_status": "NOT_TESTED_REQUIRES_8XA100",
            "receiver_ordering_status": "NOT_TESTED",
            "reopen_condition": (
                "8xA100 shows raw BF16 exposed return fraction at or above each model's "
                "recorded required fraction, and a fused codec has non-negative net benefit"
            ),
        },
        "model_results": model_results,
        "codec_summary": codec_summary,
        "matrix": matrix,
        "evidence_boundary": (
            "This is a deterministic upper-bound synthesis of prior single-GPU artifacts. "
            "It is not a new GPU run, EP/NCCL/RDMA measurement, decode SLO result, or proof "
            "that the return path is exposed in production."
        ),
        "source_paths": [
            path
            for quality in quality_inputs
            for path in (quality["summary_path"], quality["metadata_path"])
        ]
        + [codec_path],
    }


def _format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_report(result: dict[str, Any]) -> str:
    gate = result["primary_gate"]
    decision = result["decision"]
    lines = [
        "# CPR-MoE RankLane 5090 快验结果",
        "",
        f"- 决策：`{decision['code']}`",
        f"- 冻结主门槛：跨模型 AND；{gate['domain']}；相对 uniform FP8 的端到端改善 >= {_format_percent(gate['target_relative_improvement'])}",
        f"- 最有利假设：{gate['assumption']}",
        f"- RankLane 假设状态：`{decision['ranklane_hypothesis_status']}`",
        f"- P1 回传路径存在性：`{decision['p1_return_path_existence_status']}`",
        "",
        "## 主结果",
        "",
        "| 模型 | uniform FP8 节省 | 最乐观 RankLane | 节省 | p=20% 零 codec E2E 上界 | 达到 5% 所需 p | PASS |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for item in result["model_results"]:
        lines.append(
            "| {model} | {base} | `{candidate}` | {saving} | {upper} | {required} | {passed} |".format(
                model=item["model_key"],
                base=_format_percent(item["baseline"]["byte_saving"]),
                candidate=item["quality_free_max_saving_candidate"]["policy"],
                saving=_format_percent(item["quality_free_max_saving_candidate"]["byte_saving"]),
                upper=_format_percent(item["zero_codec_upper_bound_at_p_max"]),
                required=_format_percent(item["required_exposed_fraction_for_target"]),
                passed="PASS" if item["pass"] else "FAIL",
            )
        )
    codec = result["codec_summary"]
    lines.extend(
        [
            "",
            "## Codec 旁证（不参与主门槛）",
            "",
            f"- 既有 RTX 5090 FP8→INT4 增量 codec gate：{codec['incremental_fp8_to_int4_viable_count']}/{codec['incremental_fp8_to_int4_n_cells']} 可行。",
            f"- p95 是否为独立样本统计：`{str(codec['p95_is_independent_measurement']).lower()}`。因此这里只采用 p50 方向性旁证。",
            f"- 边界：{codec['evidence_boundary']}",
            "",
            "## 裁决解释",
            "",
            "本门槛已经允许任意 RankLane tail policy，并把 codec/launch/queueing/metadata 成本全部设为零。若该上界仍低于 5%，加入质量约束或真实执行开销不可能把它救回。该 FAIL 只否定冻结域内的固定 RankLane 执行器；不否定 P1 回传路径可能存在，也不替代 8×A100 的真实 EP profiling。",
            "",
            "## 证据边界",
            "",
            result["evidence_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    output_dir: Path, config_path: Path, config: dict[str, Any], result: dict[str, Any]
) -> None:
    if output_dir.exists():
        raise EvidenceError(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    manifest_paths = [Path(__file__).resolve(), config_path, *result.pop("source_paths")]
    source_manifest = {
        "generated_at_utc": generated_at,
        "files": [
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in manifest_paths
        ],
    }
    environment = {
        "generated_at_utc": generated_at,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version_if_installed": None,
        "gpu_timing_performed": False,
    }
    try:
        environment["torch_version_if_installed"] = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        pass

    with (output_dir / "matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["matrix"][0].keys()))
        writer.writeheader()
        writer.writerows(result["matrix"])
    (output_dir / "decision.json").write_text(
        json.dumps({"generated_at_utc": generated_at, **result}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "source_manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(render_report(result), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config_path = args.config.resolve()
        repo_root = args.repo_root.resolve()
        output_dir = args.output_dir.resolve()
        config = load_json(config_path)
        result = analyze(repo_root, config)
        write_outputs(output_dir, config_path, config, result)
    except EvidenceError as exc:
        print(f"EVIDENCE_ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"decision={result['decision']['code']}")
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

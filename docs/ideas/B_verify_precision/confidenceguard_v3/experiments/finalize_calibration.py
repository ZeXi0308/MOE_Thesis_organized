#!/usr/bin/env python3
"""Freeze TriageAudit calibration without reading any sealed artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from triage_artifacts import write_json_no_overwrite
from triage_policy import FEATURE_NAMES, calibration_stability


class CalibrationError(RuntimeError):
    pass


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file() or "sealed" in path.name.lower():
        raise CalibrationError("calibration input must exist and must not be named sealed")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise CalibrationError(f"line {line_number} is not an object")
        rows.append(value)
    return rows


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize(
    rows: list[Mapping[str, object]],
    config: Mapping[str, object],
    *,
    raw_sha256: str,
) -> dict[str, object]:
    dataset = config.get("dataset")
    predictor = config.get("predictor")
    stability_config = config.get("calibration_stability")
    controller = config.get("controller")
    models = config.get("models")
    if not all(isinstance(value, Mapping) for value in (dataset, predictor, stability_config, controller, models)):
        raise CalibrationError("config lacks required sections")
    expected_documents = int(dataset["calibration_documents"])
    results: dict[str, object] = {}
    for model_key in models:
        selected = [row for row in rows if row.get("model_key") == model_key]
        digests = [str(row.get("text_sha256")) for row in selected]
        if len(selected) != expected_documents or len(set(digests)) != expected_documents:
            raise CalibrationError(
                f"{model_key}: expected {expected_documents} unique calibration rows, got {len(selected)}"
            )
        feature_rows: list[dict[str, float]] = []
        labels: list[float] = []
        all_discrepancies: list[float] = []
        for row in selected:
            if row.get("split") != "calibration":
                raise CalibrationError(f"{model_key}: non-calibration split in raw input")
            feature_value = row.get("features")
            discrepancies_value = row.get("same_state_discrepancies")
            if not isinstance(feature_value, Mapping) or not isinstance(discrepancies_value, list):
                raise CalibrationError(f"{model_key}: missing features or discrepancies")
            feature_rows.append({name: float(feature_value[name]) for name in FEATURE_NAMES})
            discrepancies = np.asarray(discrepancies_value, dtype=np.float64)
            if (
                discrepancies.ndim != 1
                or len(discrepancies) != int(dataset["decode_steps"])
                or not np.isfinite(discrepancies).all()
                or np.any(discrepancies < 0)
            ):
                raise CalibrationError(f"{model_key}: invalid same-state discrepancy vector")
            tail_count = max(1, int(np.ceil(len(discrepancies) * 0.1)))
            labels.append(float(np.sort(discrepancies)[-tail_count:].mean()))
            all_discrepancies.extend(float(value) for value in discrepancies)
        stability = calibration_stability(
            feature_rows,
            labels,
            alpha=float(predictor["ridge_alpha"]),
            repeats=int(stability_config["bootstrap_replicates"]),
            seed=int(config["seed"]) + (0 if model_key == "olmoe" else 1),
        )
        checks = {
            "median_assignment_probability": stability["median_assignment_probability"]
            >= float(stability_config["median_assignment_probability_min"]),
            "stable_document_fraction": stability["fraction_documents_probability_ge_0_6"]
            >= float(stability_config["fraction_documents_probability_ge_0_6_min"]),
            "spearman_positive": stability["spearman_lcb"]
            > float(stability_config["spearman_lcb_min_exclusive"]),
        }
        results[str(model_key)] = {
            "document_count": len(selected),
            "document_sha256s": sorted(digests),
            "audit_threshold": float(
                np.quantile(np.asarray(all_discrepancies), float(controller["audit_threshold_quantile"]))
            ),
            "stability": stability,
            "checks": checks,
            "h0_pass": all(checks.values()),
        }
    calibration_sets = {tuple(value["document_sha256s"]) for value in results.values()}
    if len(calibration_sets) != 1:
        raise CalibrationError("calibration document sets differ across models")
    all_pass = all(bool(value["h0_pass"]) for value in results.values())
    config_canonical_sha256 = hashlib.sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "triage-calibration-lock-v2",
        "raw_calibration_sha256": raw_sha256,
        "config_schema_version": config.get("schema_version"),
        "config_canonical_sha256": config_canonical_sha256,
        "models": results,
        "h0_all_models_pass": all_pass,
        "sealed_run_allowed_by_calibration": all_pass,
        "verdict": "H0_PASS" if all_pass else "NO_GO_UNSTABLE_TRIAGE_SIGNAL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    lock = finalize(_read_jsonl(args.raw_calibration), config, raw_sha256=_file_sha256(args.raw_calibration))
    write_json_no_overwrite(args.output, lock)


if __name__ == "__main__":
    main()

from __future__ import annotations

"""Validate and seal a non-formal RouteSlack single-GPU evidence bundle."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from run_routeslack_dry_run import _environment, _git, _run_unit_tests


MODEL_REVISIONS = {
    "llmjp": "1d5983076dfc67aee4a77ec06a27027f5bab6055",
    "olmoe": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
}
CAPTURE_ROWS = {"llmjp": 2048, "olmoe": 1024}
EXACTNESS_CONTRIBUTIONS = {"llmjp": 1024, "olmoe": 512}
OPEN_P0 = tuple(f"P0-{number:02d}" for number in range(8, 17))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--source-package-sha256", required=True)
    parser.add_argument("--cjc-package-sha256", required=True)
    parser.add_argument("--compat-package-sha256", required=True)
    parser.add_argument(
        "--provenance-command",
        action="append",
        default=[],
        help="Repeat for each command used to produce the raw evidence.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_capture(raw: Path, model: str) -> dict[str, object]:
    csv_path = raw / f"{model}_dev_routes.csv"
    meta_path = raw / f"{model}_dev_routes.meta.json"
    meta = read_json(meta_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = sum(1 for _ in csv.DictReader(handle))
    expected_rows = CAPTURE_ROWS[model]
    checks = {
        "row_count_matches_frozen_expectation": rows == expected_rows,
        "row_count_matches_metadata": rows == meta.get("rows"),
        "csv_sha256_matches_metadata": sha256(csv_path) == meta.get("output_sha256"),
        "model_revision_matches_frozen_revision": meta.get("model_revision") == MODEL_REVISIONS[model],
        "formal_eligible_is_false": meta.get("formal_eligible") is False,
        "scientific_result_eligible_is_false": meta.get("scientific_result_eligible") is False,
    }
    if not all(checks.values()):
        raise SystemExit(f"capture validation failed for {model}: {checks}")
    return {
        "rows": rows,
        "requests": meta.get("requests"),
        "decode_steps_max": meta.get("decode_steps_max"),
        "seq_len": meta.get("seq_len"),
        "model_revision": meta.get("model_revision"),
        "csv_sha256": sha256(csv_path),
        "metadata_sha256": sha256(meta_path),
        "checks": checks,
    }


def validate_exactness(raw: Path, model: str) -> dict[str, object]:
    path = raw / f"{model}_gpu_exactness.json"
    value = read_json(path)
    checks = value.get("checks")
    route = value.get("route")
    if not isinstance(checks, dict) or not isinstance(route, dict):
        raise SystemExit(f"missing exactness checks or route summary: {path}")
    validation = {
        "all_exactness_checks_passed": checks.get("passed") is True,
        "zero_max_abs_error_all_steps": all(
            comparison.get("max_abs_error") == 0.0
            for comparison in value.get("comparisons", [])
        ),
        "contributions_match_frozen_expectation": (
            route.get("contributions") == EXACTNESS_CONTRIBUTIONS[model]
        ),
        "model_revision_matches_frozen_revision": value.get("model_revision") == MODEL_REVISIONS[model],
        "formal_result_is_false": value.get("formal_result") is False,
        "gate0_eligible_is_false": value.get("gate0_eligible") is False,
    }
    if not all(validation.values()):
        raise SystemExit(f"exactness validation failed for {model}: {validation}")
    return {
        "model": value.get("model"),
        "model_revision": value.get("model_revision"),
        "decode_steps": value.get("decode_steps"),
        "route": route,
        "reported_checks": checks,
        "validation": validation,
        "sha256": sha256(path),
    }


def validate_meter(raw: Path) -> dict[str, object]:
    path = raw / "gpu_meter_preflight.json"
    value = read_json(path)
    trace = value.get("trace")
    checks = value.get("checks")
    before = value.get("telemetry_before")
    after = value.get("telemetry_after")
    if not all(isinstance(item, dict) for item in (trace, checks, before, after)):
        raise SystemExit("GPU meter preflight is missing required objects")
    assert isinstance(trace, dict) and isinstance(checks, dict)
    assert isinstance(before, dict) and isinstance(after, dict)
    validation = {
        "all_capability_checks_passed": all(checks.values()),
        "sample_gap_within_20ms": float(trace["max_sample_gap_s"]) <= 0.020,
        "cumulative_energy_counter_present": trace.get("total_energy_counter_delta_j") is not None,
        "formal_result_is_false": value.get("formal_result") is False,
        "gate0_eligible_is_false": value.get("gate0_eligible") is False,
    }
    if not all(validation.values()):
        raise SystemExit(f"meter validation failed: {validation}")
    temperature_delta = float(after["temperature_c"]) - float(before["temperature_c"])
    return {
        "gpu": value.get("gpu"),
        "workload": value.get("workload"),
        "sample_count": trace.get("sample_count"),
        "max_sample_gap_s": trace.get("max_sample_gap_s"),
        "cumulative_energy_counter_delta_j": trace.get("total_energy_counter_delta_j"),
        "power_integral_j": trace.get("power_integral_j"),
        "temperature_before_c": before.get("temperature_c"),
        "temperature_after_c": after.get("temperature_c"),
        "temperature_delta_c": temperature_delta,
        "formal_pair_thermal_limit_c": 2.0,
        "formal_pair_thermal_check_passed": abs(temperature_delta) <= 2.0,
        "clock_throttle_reasons_after": after.get("clock_throttle_reasons"),
        "validation": validation,
        "sha256": sha256(path),
    }


def main() -> None:
    args = parse_args()
    artifact = Path(args.artifact_dir).resolve()
    if (artifact / "manifest.json").exists():
        raise SystemExit("refusing to overwrite an already sealed bundle")
    raw = artifact / "raw"
    logs = artifact / "logs"
    if not raw.is_dir() or not logs.is_dir():
        raise SystemExit("artifact must already contain raw/ and logs/")

    captures = {model: validate_capture(raw, model) for model in MODEL_REVISIONS}
    exactness = {model: validate_exactness(raw, model) for model in MODEL_REVISIONS}
    meter = validate_meter(raw)

    unit_tests, test_log = _run_unit_tests()
    (logs / "unit_tests.log").write_text(test_log, encoding="utf-8")
    if unit_tests != {
        "suites": unit_tests["suites"],
        "tests": 96,
        "failed_suites": 0,
        "all_passed": True,
    }:
        raise SystemExit(f"unit-test qualification failed: {unit_tests}")

    source_packages = {
        "routeslack_source_tar_sha256": args.source_package_sha256,
        "cjc_dependency_tar_sha256": args.cjc_package_sha256,
        "compat_dependency_tar_sha256": args.compat_package_sha256,
    }
    summary = {
        "schema": "routeslack-single-gpu-gate0-qualification-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GPU_DEVELOPMENT_QUALIFICATION_COMPLETE_GATE0_FAIL",
        "formal_result": False,
        "scientific_result_eligible": False,
        "gate0": "FAIL",
        "verdict": "MEASUREMENT_ONLY",
        "open_p0_count": len(OPEN_P0),
        "open_p0_ids": list(OPEN_P0),
        "evidence_boundary": (
            "Single-GPU batch-1 frozen-model route capture, cached-decode exactness, "
            "and NVML capability evidence only; no continuous serving, physical EP, "
            "matched completion set, service-energy surface, or policy comparison."
        ),
        "captures": captures,
        "cached_decode_exactness": exactness,
        "gpu_meter_preflight": meter,
        "unit_tests": unit_tests,
        "physical_model_energy_samples": 0,
        "route_capture_ratio": None,
        "formal_experiments_a_to_e": "NOT_RUN",
        "source_packages": source_packages,
        "git": {
            "head": _git(["rev-parse", "HEAD"]),
            "status_porcelain": _git(["status", "--porcelain"]),
        },
    }
    write_json(artifact / "processed/gpu_gate0_summary.json", summary)
    write_json(artifact / "environment.json", _environment())

    config = "\n".join(
        [
            "schema: routeslack-single-gpu-gate0-qualification-v1",
            "formal_result: false",
            "scientific_result_eligible: false",
            "gate0: FAIL",
            "verdict: MEASUREMENT_ONLY",
            "capture:",
            "  dataset: wikitext-103-raw-v1",
            "  split: train",
            "  samples_per_model: 2",
            "  seq_len: 128",
            "  decode_steps: 4",
            "  dtype: bfloat16",
            "exactness:",
            "  batch_size: 1",
            "  decode_steps: 4",
            "  dtype: bfloat16",
            "meter:",
            "  interval_ms: 10",
            "  duration_s: 3",
            "  matrix_size: 8192",
            "open_p0_ids: [" + ", ".join(OPEN_P0) + "]",
            "",
        ]
    )
    (artifact / "config.yaml").write_text(config, encoding="utf-8")

    commands = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    commands.extend(args.provenance_command)
    commands.extend(
        [
            "",
            "# Bundle finalization command:",
            "# " + shlex.join(sys.argv),
            "",
        ]
    )
    (artifact / "commands.sh").write_text("\n".join(commands), encoding="utf-8")
    (artifact / "git_diff.patch").write_text(_git(["diff", "--binary"]) + "\n", encoding="utf-8")
    source_snapshot = artifact / "source_snapshot"
    source_snapshot.mkdir(exist_ok=True)
    experiment_dir = Path(__file__).resolve().parent
    for filename in (
        "finalize_gpu_gate0_bundle.py",
        "run_frozen_model_gpu_exactness.py",
        "run_gpu_meter_preflight.py",
    ):
        shutil.copy2(experiment_dir / filename, source_snapshot / filename)
    figures = artifact / "figures"
    figures.mkdir(exist_ok=True)
    (figures / "README.md").write_text(
        "# Figures\n\nNo formal RouteSlack plots were generated. Experiments A-E were not run.\n",
        encoding="utf-8",
    )
    (artifact / "verdict.md").write_text(
        "# Verdict\n\n"
        "`MEASUREMENT_ONLY`\n\n"
        "The single-GPU development checks passed, but Gate 0 remains `FAIL` with "
        "nine open P0 blockers. This bundle is not scientific evidence for an "
        "Energy-SLO claim, controller benefit, or physical expert-parallel result.\n",
        encoding="utf-8",
    )

    files = {}
    for path in sorted(candidate for candidate in artifact.rglob("*") if candidate.is_file()):
        relative = path.relative_to(artifact).as_posix()
        if relative == "manifest.json":
            continue
        files[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest = {
        "schema": "routeslack-artifact-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_result": False,
        "gate0": "FAIL",
        "verdict": "MEASUREMENT_ONLY",
        "files": files,
        "source_packages": source_packages,
    }
    write_json(artifact / "manifest.json", manifest)
    print(json.dumps({"artifact": str(artifact), "files": len(files), "summary": summary["status"]}, sort_keys=True))
    print(f"manifest_sha256={sha256(artifact / 'manifest.json')}")


if __name__ == "__main__":
    main()

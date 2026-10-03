from __future__ import annotations

"""Seal the non-formal RTX 5090 expert-energy characterization bundle."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _verify_child_manifest(directory: Path) -> dict[str, object]:
    manifest = _json(directory / "manifest.json")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise SystemExit(f"child manifest has no file table: {directory}")
    for relative, expected in files.items():
        if not isinstance(expected, dict):
            raise SystemExit(f"malformed child manifest entry: {relative}")
        path = directory / str(relative)
        if not path.is_file() or _sha256(path) != expected.get("sha256"):
            raise SystemExit(f"child artifact hash mismatch: {path}")
    return manifest


def _trials(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise SystemExit(f"malformed trial row: {path}")
        rows.append(value)
    return rows


def _git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args), check=False, text=True, capture_output=True
    )
    if result.returncode == 0:
        return result.stdout.strip()
    detail = result.stderr.strip().replace("\n", " ")
    return f"UNAVAILABLE_EXIT_{result.returncode}: {detail}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact_dir.resolve()
    if (artifact / "manifest.json").exists():
        raise SystemExit("refusing to overwrite a sealed bundle")
    remote = artifact / "remote_artifacts"
    model_dirs = {
        "olmoe": remote / "olmoe_energy_characterization_retry3",
        "llm_jp": remote / "llmjp_energy_characterization",
    }
    child_manifests = {
        model: _verify_child_manifest(directory)
        for model, directory in model_dirs.items()
    }
    exactness_paths = {
        "olmoe": remote / "frozen_exactness_dev/olmoe.json",
        "llm_jp": remote / "frozen_exactness_dev/llm_jp.json",
    }
    exactness: dict[str, object] = {}
    for model, path in exactness_paths.items():
        value = _json(path)
        checks = value.get("checks")
        if (
            not isinstance(checks, dict)
            or checks.get("passed") is not True
            or value.get("formal_result") is not False
        ):
            raise SystemExit(f"exactness qualification failed: {model}")
        exactness[model] = {
            "sha256": _sha256(path),
            "decode_steps": value.get("decode_steps"),
            "route": value.get("route"),
            "checks": checks,
            "max_abs_error": max(
                float(row["max_abs_error"])
                for row in value.get("comparisons", [])
            ),
        }

    models: dict[str, object] = {}
    all_trials: list[dict[str, object]] = []
    for model, directory in model_dirs.items():
        summary = _json(directory / "processed/summary.json")
        trials = _trials(directory / "raw/trials.jsonl")
        all_trials.extend(trials)
        valid_by_rows = Counter(
            int(row["rows"]) for row in trials if bool(row["valid"])
        )
        reasons = Counter(
            str(reason)
            for row in trials
            for reason in row.get("invalid_reasons", [])
        )
        models[model] = {
            "status": summary.get("status"),
            "model_revision": child_manifests[model].get("model_revision"),
            "capture_sha256": child_manifests[model].get("capture_sha256"),
            "attempted_physical_windows": len(trials),
            "valid_physical_windows": sum(bool(row["valid"]) for row in trials),
            "invalid_physical_windows": sum(not bool(row["valid"]) for row in trials),
            "valid_windows_by_rows": {
                str(rows): valid_by_rows[rows] for rows in (1, 8, 32, 128)
            },
            "invalid_reason_counts": dict(sorted(reasons.items())),
            "cells": summary.get("cells"),
            "child_manifest_sha256": _sha256(directory / "manifest.json"),
        }

    source_dir = artifact / "source_snapshot"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_files = (
        Path(__file__).resolve(),
        Path(__file__).with_name("run_rtx5090_energy_characterization.py").resolve(),
        Path(__file__).with_name("test_rtx5090_energy_characterization.py").resolve(),
        Path(__file__).with_name("run_frozen_model_gpu_exactness.py").resolve(),
    )
    for source in source_files:
        shutil.copy2(source, source_dir / source.name)

    total_valid = sum(bool(row["valid"]) for row in all_trials)
    total_attempted = len(all_trials)
    reason_counts = Counter(
        str(reason)
        for row in all_trials
        for reason in row.get("invalid_reasons", [])
    )
    summary = {
        "schema": "routeslack-rtx5090-physical-development-bundle-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PHYSICAL_CHARACTERIZATION_INCOMPLETE_GATE0_FAIL",
        "formal_result": False,
        "scientific_result_eligible": False,
        "gate0": "FAIL",
        "gate1": "FAIL",
        "verdict": "MEASUREMENT_ONLY",
        "development_physical_windows": {
            "attempted": total_attempted,
            "valid": total_valid,
            "invalid": total_attempted - total_valid,
            "invalid_reason_counts": dict(sorted(reason_counts.items())),
            "minimum_host_window_s": min(
                float(row["host_elapsed_s"]) for row in all_trials
            ),
        },
        "formal_strategy_energy_samples": 0,
        "models": models,
        "cached_decode_exactness": exactness,
        "power_clock_actuator": "DENIED_BY_PROVIDER_PERMISSIONS",
        "capture_ratio": None,
        "evidence_boundary": (
            "Real RTX 5090 cumulative-counter expert-stage windows over captured "
            "BF16 prefill activations. Not continuous serving, not matched "
            "SLO-completed tokens, not a power-tier comparison, and not EP."
        ),
    }
    _write_json(artifact / "processed/summary.json", summary)
    (artifact / "config.yaml").write_text(
        "\n".join(
            (
                "schema: routeslack-rtx5090-physical-development-bundle-v1",
                "formal_result: false",
                "rows: [1, 8, 32, 128]",
                "outer_trials: 4",
                "minimum_window_seconds: 10",
                "sample_interval_ms: 5",
                "maximum_sample_gap_ms: 20",
                "thermal_warmup_seconds: 60",
                "thermal_stable_window_seconds: 30",
                "maximum_temperature_range_c: 2",
                "power_tier: provider_default_575W_uncontrolled",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (artifact / "commands.sh").write_text(
        "# Exact commands are preserved in remote_artifacts/*/commands.sh and remote_logs/.\n",
        encoding="utf-8",
    )
    (artifact / "verdict.md").write_text(
        "# Verdict\n\nMEASUREMENT_ONLY\n\n"
        "The physical expert-stage meter did not produce a complete valid "
        "two-model surface and cannot authorize RouteSlack Gate 1.\n",
        encoding="utf-8",
    )

    files = sorted(path for path in artifact.rglob("*") if path.is_file())
    manifest = {
        "schema": "routeslack-artifact-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": summary["status"],
        "formal_result": False,
        "gate0": "FAIL",
        "verdict": "MEASUREMENT_ONLY",
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "status_porcelain": _git("status", "--porcelain"),
        },
        "files": {
            str(path.relative_to(artifact)): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        },
    }
    _write_json(artifact / "manifest.json", manifest)
    print(f"files={len(files)}")
    print(f"manifest_sha256={_sha256(artifact / 'manifest.json')}")


if __name__ == "__main__":
    main()

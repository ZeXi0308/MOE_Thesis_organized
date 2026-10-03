#!/usr/bin/env python3
"""Aggregate CPR-MoE quick-validation outputs without upgrading claim scope."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_experiment import EXPECTED_QUANTIZATION_CONTRACT


COMPLETE = "COMPLETE"
SMOKE = "SMOKE_NOT_SCIENTIFIC"


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def decision_status(item: dict[str, Any] | None) -> str | None:
    return None if item is None else str(item.get("status", "MISSING"))


def decide(quality: dict[str, Any] | None, codec: dict[str, Any] | None) -> dict[str, Any]:
    decisions = [item for item in (quality, codec) if item is not None]
    if not decisions:
        raise ValueError("no quality_decision.json or codec_decision.json found")
    statuses = [decision_status(item) for item in decisions]
    if any(status == SMOKE for status in statuses):
        verdict = SMOKE
    elif any(status != COMPLETE for status in statuses):
        verdict = "INVALID_DECISION_STATUS"
    elif quality is not None and (
        quality.get("decision_mode") != "int4"
        or not isinstance(quality.get("all_models_passed"), bool)
        or quality.get("quantization_contract") != EXPECTED_QUANTIZATION_CONTRACT
        or not isinstance(quality.get("source_provenance"), list)
        or not quality.get("source_provenance")
    ):
        verdict = "INVALID_QUALITY_CONTRACT"
    elif codec is not None and (
        codec.get("decision_mode") != "int4"
        or not isinstance(codec.get("primary_mode_passed"), bool)
        or codec.get("quantization_contract") != EXPECTED_QUANTIZATION_CONTRACT
    ):
        verdict = "INVALID_CODEC_CONTRACT"
    elif quality is not None and codec is not None and (
        quality["quantization_contract"] != codec["quantization_contract"]
    ):
        verdict = "INVALID_MISMATCHED_QUANTIZATION_CONTRACT"
    elif quality is not None and not bool(quality.get("all_models_passed")):
        verdict = "NO_GO_CPR_QUALITY_SIGNAL"
    elif codec is not None and not bool(codec.get("primary_mode_passed")):
        verdict = "NO_GO_CURRENT_UNFUSED_INT4_CODEC_PATH"
    elif quality is None or codec is None:
        verdict = "INCOMPLETE_NECESSARY_GATES"
    else:
        verdict = "NOT_FALSIFIED_SINGLE_GPU_BLOCKED_EP_RETURN_PATH_GATE"
    return {
        "verdict": verdict,
        "quality_present": quality is not None,
        "codec_present": codec is not None,
        "quality_status": decision_status(quality),
        "codec_status": decision_status(codec),
        "ep_return_path_gate": "BLOCKED_NOT_TESTABLE_ON_SINGLE_GPU",
        "allowed_next_claim": (
            "At most a single-GPU necessary-condition result. No EP/NCCL/TPOT/P99 claim."
        ),
    }


def load_run(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    return (
        read_json(path / "run_manifest.json"),
        read_json(path / "quality_decision.json"),
        read_json(path / "codec_decision.json"),
    )


def validate_run_artifacts(
    manifest: dict[str, Any] | None,
    quality: dict[str, Any] | None,
    codec: dict[str, Any] | None,
) -> list[str]:
    errors: list[str] = []
    if manifest is None:
        return ["missing run_manifest.json"]
    status = manifest.get("status")
    if status not in {COMPLETE, SMOKE}:
        errors.append(f"invalid manifest status: {status!r}")
    if manifest.get("experiment") != "all":
        errors.append("scientific analysis requires manifest experiment=all")
    seed = manifest.get("seed")
    formal_seeds = manifest.get("formal_seeds")
    if (
        not isinstance(seed, int)
        or not isinstance(formal_seeds, list)
        or not all(isinstance(value, int) for value in formal_seeds)
        or seed not in formal_seeds
    ):
        errors.append("invalid seed/formal_seeds manifest contract")
    expected_decision_status = COMPLETE if status == COMPLETE else SMOKE
    if quality is None or codec is None:
        errors.append("experiment=all requires both decision files")
    else:
        if quality.get("status") != expected_decision_status:
            errors.append("quality decision status does not match manifest")
        if codec.get("status") != expected_decision_status:
            errors.append("codec decision status does not match manifest")
    expected_manifest_decisions = {"quality": quality, "codec": codec}
    if manifest.get("decisions") != expected_manifest_decisions:
        errors.append("manifest decisions do not exactly match decision files")
    if quality is not None and manifest.get("quality_source_provenance") != quality.get(
        "source_provenance"
    ):
        errors.append("manifest quality source provenance does not match decision")
    return errors


def aggregate_seed_runs(paths: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(paths) < 2:
        raise ValueError("multi-seed aggregation requires at least two input directories")
    records: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    source_provenance_payloads: list[str] = []
    for path in paths:
        manifest, quality, codec = load_run(path)
        if manifest is None:
            raise ValueError(f"missing run_manifest.json in {path}")
        manifests.append(manifest)
        artifact_errors = validate_run_artifacts(manifest, quality, codec)
        run_result = (
            {
                "verdict": "INVALID_RUN_ARTIFACTS",
                "ep_return_path_gate": "BLOCKED_NOT_TESTABLE_ON_SINGLE_GPU",
            }
            if artifact_errors
            else decide(quality, codec)
        )
        if quality is not None:
            source_provenance_payloads.append(
                json.dumps(quality.get("source_provenance"), sort_keys=True)
            )
        records.append(
            {
                "path": str(path.resolve()),
                "seed": manifest.get("seed"),
                "manifest_status": manifest.get("status"),
                "verdict": run_result["verdict"],
                "artifact_errors": artifact_errors,
            }
        )

    config_hashes = {item.get("config_sha256") for item in manifests}
    declared_seed_sets = {
        tuple(sorted(int(value) for value in item.get("formal_seeds", [])))
        for item in manifests
    }
    observed_seeds = [item.get("seed") for item in manifests]
    expected_seeds = set(next(iter(declared_seed_sets))) if len(declared_seed_sets) == 1 else set()
    valid_protocol = bool(
        len(config_hashes) == 1
        and None not in config_hashes
        and len(declared_seed_sets) == 1
        and expected_seeds
        and len(observed_seeds) == len(set(observed_seeds))
        and set(observed_seeds) == expected_seeds
        and all(item.get("status") == COMPLETE for item in manifests)
        and all(item.get("experiment") == "all" for item in manifests)
        and len(source_provenance_payloads) == len(paths)
        and len(set(source_provenance_payloads)) == 1
        and all(not item["artifact_errors"] for item in records)
    )
    per_seed_verdicts = [item["verdict"] for item in records]
    if not valid_protocol or any(
        verdict.startswith("INVALID") or verdict in {SMOKE, "INCOMPLETE_NECESSARY_GATES"}
        for verdict in per_seed_verdicts
    ):
        verdict = "INVALID_OR_INCOMPLETE_MULTI_SEED_PROTOCOL"
    elif any(verdict == "NO_GO_CPR_QUALITY_SIGNAL" for verdict in per_seed_verdicts):
        verdict = "NO_GO_CPR_QUALITY_SIGNAL"
    elif any(verdict == "NO_GO_CURRENT_UNFUSED_INT4_CODEC_PATH" for verdict in per_seed_verdicts):
        verdict = "NO_GO_CURRENT_UNFUSED_INT4_CODEC_PATH"
    elif all(
        verdict == "NOT_FALSIFIED_SINGLE_GPU_BLOCKED_EP_RETURN_PATH_GATE"
        for verdict in per_seed_verdicts
    ):
        verdict = "NOT_FALSIFIED_SINGLE_GPU_BLOCKED_EP_RETURN_PATH_GATE"
    else:
        verdict = "INVALID_OR_INCOMPLETE_MULTI_SEED_PROTOCOL"
    result = {
        "verdict": verdict,
        "multi_seed_protocol_valid": valid_protocol,
        "observed_seeds": observed_seeds,
        "expected_seeds": sorted(expected_seeds),
        "config_sha256": next(iter(config_hashes)) if len(config_hashes) == 1 else None,
        "ep_return_path_gate": "BLOCKED_NOT_TESTABLE_ON_SINGLE_GPU",
        "allowed_next_claim": (
            "At most a repeated single-GPU necessary-condition result. No EP/NCCL/TPOT/P99 claim."
        ),
    }
    return result, records


def write_report(
    path: Path,
    result: dict[str, Any],
    quality: dict[str, Any] | None = None,
    codec: dict[str, Any] | None = None,
    seed_records: list[dict[str, Any]] | None = None,
) -> None:
    lines = [
        "# CPR-MoE 5090 Quick Validation",
        "",
        f"- verdict: **{result['verdict']}**",
        f"- 8xA100 EP Gate 0: **{result['ep_return_path_gate']}**",
        "",
        "## Evidence boundary",
        "",
        result["allowed_next_claim"],
        "",
        "## Necessary gates",
        "",
    ]
    if seed_records is None:
        lines.extend(
            [
                f"- quality: {quality.get('verdict') if quality else 'NOT_RUN'}",
                f"- INT4 codec: {codec.get('verdict') if codec else 'NOT_RUN'}",
            ]
        )
    else:
        lines.append(f"- multi-seed protocol valid: {result['multi_seed_protocol_valid']}")
        for record in seed_records:
            lines.append(
                f"- seed {record['seed']}: {record['verdict']} ({record['manifest_status']})"
            )
    lines.extend(
        [
            "",
            "A PASS here never authorizes a CPR-MoE controller. The optimized multi-GPU return-path existence gate remains mandatory.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-dir", type=Path)
    group.add_argument("--input-dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if args.input_dir is not None:
        manifest, quality, codec = load_run(args.input_dir)
        artifact_errors = validate_run_artifacts(manifest, quality, codec)
        if artifact_errors:
            result = {
                "verdict": "INVALID_RUN_ARTIFACTS",
                "artifact_errors": artifact_errors,
                "ep_return_path_gate": "BLOCKED_NOT_TESTABLE_ON_SINGLE_GPU",
                "allowed_next_claim": "No scientific claim; run artifacts failed validation.",
            }
        else:
            result = decide(quality, codec)
        output = args.output_dir or args.input_dir
        output.mkdir(parents=True, exist_ok=True)
        (output / "analysis.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        write_report(output / "report.md", result, quality, codec)
        if result["verdict"].startswith("INVALID") or result["verdict"].startswith("INCOMPLETE"):
            raise SystemExit(2)
        return

    if args.output_dir is None:
        raise ValueError("--output-dir is required with --input-dirs")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty aggregate output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result, records = aggregate_seed_runs(list(args.input_dirs))
    (args.output_dir / "analysis.json").write_text(
        json.dumps({**result, "seed_runs": records}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(args.output_dir / "report.md", result, seed_records=records)
    if result["verdict"].startswith("INVALID") or result["verdict"].startswith("INCOMPLETE"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Combine two corrected-FJRC replay bundles with an explicit AND rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .fjrc_corrected_replay import ReplayError, decide_two_model
except ImportError:  # pragma: no cover
    from fjrc_corrected_replay import ReplayError, decide_two_model  # type: ignore


REQUIRED_FILES = {
    "config.yaml",
    "metrics.json",
    "raw_results.jsonl",
    "environment.json",
    "source_manifest.json",
    "stdout.log",
    "summary.md",
}


class TwoModelDecisionError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TwoModelDecisionError(f"cannot load {path}") from exc
    if not isinstance(value, Mapping):
        raise TwoModelDecisionError(f"JSON root is not an object: {path}")
    return value


def load_bundle(path: Path, model: str, expected_run_class: str) -> dict[str, Any]:
    if not path.is_dir() or path.is_symlink():
        raise TwoModelDecisionError(f"bundle is not a real directory: {path}")
    observed = {value.name for value in path.iterdir()}
    if not REQUIRED_FILES <= observed:
        raise TwoModelDecisionError(f"bundle is incomplete: {path}")
    config = _load(path / "config.yaml")
    metrics = _load(path / "metrics.json")
    environment = _load(path / "environment.json")
    manifest = _load(path / "source_manifest.json")
    if config.get("model") != model or config.get("run_class") != expected_run_class:
        raise TwoModelDecisionError(f"model or run class mismatch: {model}")
    if (
        metrics.get("schema_version") != "fjrc-corrected-level1-replay-v1"
        or metrics.get("status") != "LOGICAL_TRACE_REPLAY_ONLY"
        or environment.get("cuda_execution") is not False
        or environment.get("gpu_measurement") is not False
    ):
        raise TwoModelDecisionError(f"evidence boundary mismatch: {model}")
    aggregate = metrics.get("aggregate")
    decision = metrics.get("decision")
    calibration = metrics.get("deadline_calibration")
    if not isinstance(aggregate, Mapping) or not isinstance(decision, Mapping):
        raise TwoModelDecisionError(f"metrics missing aggregate or decision: {model}")
    if (
        aggregate.get("Q", {}).get("request_count") != 32
        or aggregate.get("R", {}).get("request_count") != 32
        or not isinstance(calibration, Mapping)
        or calibration.get("r_outcomes_read_for_selection") is not False
    ):
        raise TwoModelDecisionError(f"denominator or calibration audit mismatch: {model}")
    status = decision.get("status")
    gates = decision.get("gates")
    if status not in {"PASS", "FAIL", "INVALID_WORKLOAD_IDENTIFIABILITY"} or not isinstance(gates, Mapping):
        raise TwoModelDecisionError(f"decision is malformed: {model}")
    expected_gates = {
        "holdout_q_risk_nondegenerate",
        "effect_size",
        "paired_bootstrap_lower_gt_zero",
        "minimum_strict_action_flips",
        "r_not_worse_on_primary",
    }
    if set(gates) != expected_gates or any(type(value) is not bool for value in gates.values()):
        raise TwoModelDecisionError(f"decision gate surface drift: {model}")
    if calibration.get("status") != "FROZEN_FROM_SELECTION_Q_ONLY":
        raise TwoModelDecisionError(f"calibration status mismatch: {model}")
    expected_status = (
        "INVALID_WORKLOAD_IDENTIFIABILITY"
        if gates.get("holdout_q_risk_nondegenerate") is False
        else ("PASS" if all(value is True for value in gates.values()) else "FAIL")
    )
    if status != expected_status:
        raise TwoModelDecisionError(f"decision/gate inconsistency: {model}")
    protocol = manifest.get("protocol_binding")
    inputs = manifest.get("inputs")
    sources = manifest.get("sources")
    if not isinstance(protocol, Mapping) or not isinstance(inputs, Mapping) or not isinstance(sources, Mapping):
        raise TwoModelDecisionError(f"source manifest is incomplete: {model}")
    if protocol.get("scientific_boundary") != "LOGICAL_TRACE_REPLAY_NOT_NETWORK_OR_SERVING_MEASUREMENT":
        raise TwoModelDecisionError(f"source boundary drift: {model}")
    return {
        "model": model,
        "path": str(path.resolve()),
        "config": config,
        "metrics": metrics,
        "environment": environment,
        "manifest": manifest,
        "bundle_files": sorted(observed),
    }


def decide(
    olmoe_path: Path, llmjp_path: Path, *, expected_run_class: str
) -> dict[str, Any]:
    bundles = {
        "olmoe": load_bundle(olmoe_path, "olmoe", expected_run_class),
        "llmjp": load_bundle(llmjp_path, "llmjp", expected_run_class),
    }
    source_sets = [bundle["manifest"]["sources"] for bundle in bundles.values()]
    if source_sets[0] != source_sets[1]:
        raise TwoModelDecisionError("two models were not executed with the same source set")
    lut_hashes = {
        bundle["manifest"]["inputs"].get("lut_sha256") for bundle in bundles.values()
    }
    if len(lut_hashes) != 1 or None in lut_hashes:
        raise TwoModelDecisionError("two models do not share one validated LUT artifact")
    model_decisions = {
        model: bundle["metrics"]["decision"] for model, bundle in bundles.items()
    }
    try:
        combined = decide_two_model(model_decisions)
    except ReplayError as exc:
        raise TwoModelDecisionError(str(exc)) from exc
    value = {
        "schema_version": "fjrc-corrected-two-model-decision-v1",
        "status": combined["status"],
        "rule": combined["rule"],
        "run_class": expected_run_class,
        "scientific_boundary": "LOGICAL_TRACE_REPLAY_NOT_NETWORK_OR_SERVING_MEASUREMENT",
        "pooling": False,
        "lut_sha256": next(iter(lut_hashes)),
        "models": {
            model: {
                "bundle": bundle["path"],
                "decision": bundle["metrics"]["decision"],
                "aggregate": bundle["metrics"]["aggregate"],
                "deadline_calibration": bundle["metrics"]["deadline_calibration"],
            }
            for model, bundle in bundles.items()
        },
    }
    return {**value, "artifact_sha256": _sha(value)}


def write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise TwoModelDecisionError("decision output already exists")
    if value.get("artifact_sha256") != _sha(
        {key: item for key, item in value.items() if key != "artifact_sha256"}
    ):
        raise TwoModelDecisionError("decision self-hash mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise TwoModelDecisionError("decision output parent may not be a symlink")
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise TwoModelDecisionError("decision output write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise TwoModelDecisionError("decision output appeared during publish") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--olmoe-output", type=Path, required=True)
    parser.add_argument("--llmjp-output", type=Path, required=True)
    parser.add_argument(
        "--expected-run-class",
        choices=("CPU_DRY_RUN", "FORMAL_CPU_TRACE_REPLAY"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        value = decide(
            args.olmoe_output,
            args.llmjp_output,
            expected_run_class=args.expected_run_class,
        )
        write_atomic(args.output, value)
    except (OSError, ValueError, RuntimeError, TwoModelDecisionError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": value["status"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

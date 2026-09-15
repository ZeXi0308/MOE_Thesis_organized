#!/usr/bin/env python3
"""Formal CPU driver for corrected FJRC Level-1 logical trace replay.

The driver does not execute model inference or claim measured network latency.
It binds validated native route identities and a validated RTX-5090 primitive
LUT to an explicitly synthetic timing workload, then emits a non-overwriting
artifact bundle.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

try:
    from .explore_receiver_matched_milp import load_verified_joins
    from .fjrc_corrected_level1 import load_service_lut, select_split_scenarios
    from .fjrc_corrected_replay import (
        ReplayConfig,
        ReplayError,
        TIMING_SOURCE,
        calibrate_deadline_on_selection,
        materialize_replay,
        run_campaign,
    )
except ImportError:  # pragma: no cover
    from explore_receiver_matched_milp import load_verified_joins  # type: ignore
    from fjrc_corrected_level1 import load_service_lut, select_split_scenarios  # type: ignore
    from fjrc_corrected_replay import (  # type: ignore
        ReplayConfig,
        ReplayError,
        TIMING_SOURCE,
        calibrate_deadline_on_selection,
        materialize_replay,
        run_campaign,
    )


class RunnerError(RuntimeError):
    pass


REQUIRED_ARTIFACTS = (
    "config.yaml",
    "metrics.json",
    "raw_results.jsonl",
    "environment.json",
    "source_manifest.json",
    "stdout.log",
    "summary.md",
)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise RunnerError(f"value is not JSON serializable: {type(value).__name__}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return value or None


def environment_record() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "numpy", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": packages,
        "git_commit": _git_commit(),
        "cuda_execution": False,
        "gpu_measurement": False,
        "timing_source": TIMING_SOURCE,
    }


def source_manifest(
    lut_path: Path, route_root: Path, route_metadata: Mapping[str, Any]
) -> dict[str, Any]:
    here = Path(__file__).resolve().parent
    sources = [
        here / "fjrc_corrected_level0.py",
        here / "fjrc_corrected_level1.py",
        here / "fjrc_corrected_replay.py",
        Path(__file__).resolve(),
    ]
    if any(not path.is_file() for path in sources):
        raise RunnerError("source manifest cannot resolve every implementation file")
    return {
        "sources": {str(path): _sha256_file(path) for path in sources},
        "inputs": {
            "lut_path": str(lut_path.resolve()),
            "lut_sha256": _sha256_file(lut_path),
            "route_root": str(route_root.resolve()),
            "validated_route_metadata": _jsonable(route_metadata),
        },
        "protocol_binding": {
            "schema": "fjrc-corrected-level1-replay-v1",
            "primitive_lut_role": "CALIBRATION_INPUT_ONLY",
            "route_role": "NATIVE_IDENTITY_AND_FORK_JOIN_STRUCTURE_ONLY",
            "timing_role": TIMING_SOURCE,
            "scientific_boundary": "LOGICAL_TRACE_REPLAY_NOT_NETWORK_OR_SERVING_MEASUREMENT",
        },
    }


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise RunnerError(f"temporary artifact already exists: {temporary}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _summary(model: str, report: Mapping[str, Any], dry_run: bool) -> str:
    aggregate = report["aggregate"]
    decision = report["decision"]
    bootstrap = report["paired_bootstrap"]
    return "\n".join(
        (
            "# Corrected FJRC Level-1 Replay",
            "",
            f"- Model: `{model}`",
            f"- Run class: `{'CPU_DRY_RUN' if dry_run else 'FORMAL_CPU_TRACE_REPLAY'}`",
            f"- Status: `{decision['status']}`",
            f"- Timing source: `{TIMING_SOURCE}`",
            "- Evidence boundary: logical replay only; not a GPU, NCCL, RDMA, TPOT, or serving result.",
            f"- Requests: `{aggregate['Q'].request_count}`",
            f"- Q miss rate: `{aggregate['Q'].miss_rate:.8f}`",
            f"- R miss rate: `{aggregate['R'].miss_rate:.8f}`",
            f"- Q-R absolute miss reduction: `{aggregate['q_to_r_absolute_miss_reduction']:.8f}`",
            f"- Q-R relative CVaR90 reduction: `{aggregate['q_to_r_relative_cvar90_reduction']:.8f}`",
            f"- Strict action flips: `{aggregate['strict_flip_count']}/16`",
            f"- Paired bootstrap 95% CI: `{bootstrap['absolute_miss_reduction'][1:]}`",
            "",
        )
    )


def write_artifacts(
    output: Path,
    *,
    model: str,
    config: ReplayConfig,
    report: Mapping[str, Any],
    environment: Mapping[str, Any],
    manifest: Mapping[str, Any],
    dry_run: bool,
) -> None:
    if output.exists():
        raise RunnerError("output directory already exists; refusing to overwrite history")
    output.mkdir(parents=True, exist_ok=False)
    serial_report = _jsonable(report)
    _atomic_write(
        output / "config.yaml",
        json.dumps(
            {
                "model": model,
                "run_class": "CPU_DRY_RUN" if dry_run else "FORMAL_CPU_TRACE_REPLAY",
                "replay": _jsonable(config),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )
    _atomic_write(
        output / "metrics.json",
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "status": report["status"],
                "timing_source": report["timing_source"],
                "aggregate": serial_report["aggregate"],
                "baseline_aggregate": serial_report["baseline_aggregate"],
                "paired_bootstrap": serial_report["paired_bootstrap"],
                "decision": serial_report["decision"],
                "deadline_calibration": serial_report.get("deadline_calibration"),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )
    raw_lines = []
    for pair in serial_report["pair_reports"]:
        raw_lines.append(json.dumps({"record_type": "pair", **pair}, sort_keys=True, allow_nan=False))
    for baseline in serial_report["baseline_reports"]:
        raw_lines.append(
            json.dumps({"record_type": "baselines", **baseline}, sort_keys=True, allow_nan=False)
        )
    for control in serial_report["negative_controls"]:
        raw_lines.append(
            json.dumps({"record_type": "negative_controls", **control}, sort_keys=True, allow_nan=False)
        )
    _atomic_write(output / "raw_results.jsonl", "\n".join(raw_lines) + "\n")
    _atomic_write(
        output / "environment.json",
        json.dumps(_jsonable(environment), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_write(
        output / "source_manifest.json",
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_write(
        output / "stdout.log",
        "\n".join(
            (
                f"model={model}",
                f"run_class={'CPU_DRY_RUN' if dry_run else 'FORMAL_CPU_TRACE_REPLAY'}",
                f"scenario_count={len(report['pair_reports'])}",
                f"request_count={report['aggregate']['Q'].request_count}",
                f"decision={report['decision']['status']}",
                f"timing_source={TIMING_SOURCE}",
                "gpu_executed=false",
            )
        )
        + "\n",
    )
    _atomic_write(output / "summary.md", _summary(model, report, dry_run))
    missing = [name for name in REQUIRED_ARTIFACTS if not (output / name).is_file()]
    if missing:
        raise RunnerError(f"artifact set incomplete: {missing}")


def execute(
    *,
    route_root: Path,
    lut_path: Path,
    model: str,
    output: Path,
    config: ReplayConfig,
    dry_run: bool,
) -> Mapping[str, Any]:
    if output.exists():
        raise RunnerError("output directory already exists; refusing to overwrite history")
    joins, metadata = load_verified_joins(route_root, model)
    service = load_service_lut(lut_path, model)
    selection_native = select_split_scenarios(model, joins, service, split="selection")
    holdout_native = select_split_scenarios(model, joins, service, split="holdout")
    effective_config, calibration = calibrate_deadline_on_selection(
        selection_native, service, config
    )
    scenarios = [materialize_replay(value, service, effective_config) for value in holdout_native]
    report = dict(run_campaign(scenarios, effective_config))
    report["deadline_calibration"] = calibration
    write_artifacts(
        output,
        model=model,
        config=effective_config,
        report=report,
        environment=environment_record(),
        manifest=source_manifest(lut_path, route_root, metadata),
        dry_run=dry_run,
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--lut", type=Path, required=True)
    parser.add_argument("--model", choices=("olmoe", "llmjp"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--future-release-factor", type=float, default=1.5)
    parser.add_argument("--arrival-jitter-factor", type=float, default=0.75)
    parser.add_argument("--deadline-factor", type=float, default=0.85)
    parser.add_argument("--deadline-jitter", type=float, default=0.20)
    parser.add_argument("--background-depth", type=int, default=2)
    parser.add_argument("--background-service-factor", type=float, default=1.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    replicates = min(args.bootstrap_replicates, 50) if args.dry_run else args.bootstrap_replicates
    config = ReplayConfig(
        future_release_factor=args.future_release_factor,
        arrival_jitter_factor=args.arrival_jitter_factor,
        deadline_factor=args.deadline_factor,
        deadline_jitter=args.deadline_jitter,
        background_depth=args.background_depth,
        background_service_factor=args.background_service_factor,
        bootstrap_replicates=replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    try:
        report = execute(
            route_root=args.route_root,
            lut_path=args.lut,
            model=args.model,
            output=args.output,
            config=config,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError, RuntimeError, ReplayError, RunnerError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report["decision"]["status"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fail-closed readiness audit for corrected FJRC GPU work.

``static`` mode runs without CUDA and verifies the reviewed source bundle.
``gpu`` mode additionally verifies the frozen Python/CUDA environment, model
trees, one-shot route state, primitive LUT, and native replay prerequisites.
The tool never launches model inference or a scientific experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

try:
    from . import prepare_clean_v2_data as data_core
    from .explore_receiver_matched_milp import load_verified_joins
    from .fjrc_corrected_level1 import load_service_lut
except ImportError:  # pragma: no cover
    import prepare_clean_v2_data as data_core  # type: ignore
    from explore_receiver_matched_milp import load_verified_joins  # type: ignore
    from fjrc_corrected_level1 import load_service_lut  # type: ignore


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE.parents[1] / "configs/fjrc_corrected_gpu_readiness_v1.json"
GIB = 1024**3


class PreflightError(RuntimeError):
    pass


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot load config: {path}") from exc
    if not isinstance(value, Mapping):
        raise PreflightError("preflight config root must be an object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise PreflightError(f"cannot hash source: {path}") from exc
    return digest.hexdigest()


def _object_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_output_path(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise PreflightError("preflight report output already exists")
    if path.parent.exists() and path.parent.is_symlink():
        raise PreflightError("preflight report parent may not be a symlink")


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    _safe_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
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
                raise PreflightError("preflight report write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise PreflightError("preflight report appeared during publish") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _new_report(mode: str, repo_root: Path, config_path: Path) -> dict[str, Any]:
    return {
        "schema_version": "fjrc-corrected-gpu-preflight-v1",
        "mode": mode,
        "repo_root": str(repo_root.resolve(strict=False)),
        "config_path": str(config_path.resolve(strict=False)),
        "checks": [],
        "blockers": [],
        "planned_actions": [],
        "warnings": [],
    }


def _check(report: dict[str, Any], name: str, passed: bool, detail: Any) -> None:
    report["checks"].append({"name": name, "passed": bool(passed), "detail": detail})
    if not passed:
        report["blockers"].append(name)


def validate_static_sources(
    repo_root: Path, config: Mapping[str, Any], report: dict[str, Any]
) -> None:
    expected = config.get("reviewed_sources")
    if not isinstance(expected, Mapping) or not expected:
        _check(report, "reviewed_source_manifest", False, "missing reviewed_sources")
        return
    observed: dict[str, str | None] = {}
    for relative, digest in sorted(expected.items()):
        path = repo_root / str(relative)
        if not _is_sha256(digest) or not path.is_file() or path.is_symlink():
            observed[str(relative)] = None
            continue
        observed[str(relative)] = _sha256_file(path)
    mismatches = {
        relative: {"expected": expected[relative], "observed": observed.get(relative)}
        for relative in expected
        if observed.get(str(relative)) != expected[relative]
    }
    _check(report, "reviewed_source_manifest", not mismatches, mismatches or observed)


def validate_config_identity(config: Mapping[str, Any], report: dict[str, Any]) -> None:
    if config.get("schema_version") != "fjrc-corrected-gpu-readiness-v1":
        return
    expected = config.get("artifact_sha256")
    payload = {key: value for key, value in config.items() if key != "artifact_sha256"}
    observed = _object_sha256(payload)
    _check(
        report,
        "readiness_config_self_hash",
        _is_sha256(expected) and expected == observed,
        {"expected": expected, "observed": observed},
    )


def _query_compute_apps() -> list[dict[str, Any]]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreflightError("nvidia-smi compute-app query failed") from exc
    if not output or "No running processes found" in output:
        return []
    rows = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise PreflightError("malformed compute-app query")
        rows.append(
            {
                "pid": int(fields[0]),
                "gpu_uuid": fields[1],
                "process_name": fields[2],
                "used_gpu_memory_mib": float(fields[3]),
            }
        )
    return rows


def validate_gpu_environment(config: Mapping[str, Any], report: dict[str, Any]) -> None:
    environment = config.get("environment")
    if not isinstance(environment, Mapping):
        _check(report, "environment_config", False, "missing environment object")
        return
    expected_python = Path(os.path.abspath(str(environment.get("python_executable", ""))))
    actual_python = Path(os.path.abspath(sys.executable))
    _check(
        report,
        "python_executable",
        actual_python == expected_python,
        {
            "expected": str(expected_python),
            "observed": str(actual_python),
            "resolved_observed": str(actual_python.resolve()),
        },
    )
    versions = environment.get("package_versions")
    observed_versions = {}
    if isinstance(versions, Mapping):
        for package, expected in versions.items():
            try:
                observed_versions[str(package)] = importlib.metadata.version(str(package))
            except importlib.metadata.PackageNotFoundError:
                observed_versions[str(package)] = None
        _check(
            report,
            "package_versions",
            all(observed_versions.get(str(key)) == value for key, value in versions.items()),
            {"expected": versions, "observed": observed_versions},
        )
    else:
        _check(report, "package_versions", False, "missing package version map")

    try:
        import torch
    except ImportError:
        _check(report, "torch_cuda", False, "torch unavailable")
        return
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_name = torch.cuda.get_device_name(0) if device_count else None
    expected_name = environment.get("gpu_name")
    _check(
        report,
        "torch_cuda",
        device_count == 1 and device_name == expected_name and torch.version.cuda is not None,
        {
            "device_count": device_count,
            "device_name": device_name,
            "expected_name": expected_name,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    )
    try:
        apps = _query_compute_apps()
    except PreflightError as exc:
        _check(report, "exclusive_gpu", False, str(exc))
    else:
        foreign = [row for row in apps if row["pid"] != os.getpid()]
        _check(report, "exclusive_gpu", not foreign, foreign or apps)


def _resolve_remote_path(remote_root: Path, configured: object) -> Path:
    value = str(configured)
    if value.startswith("$REMOTE_ROOT/"):
        return remote_root / value.removeprefix("$REMOTE_ROOT/")
    return Path(value)


def validate_remote_inputs(
    remote_root: Path,
    config: Mapping[str, Any],
    report: dict[str, Any],
    *,
    deep_model_hash: bool,
) -> None:
    models = config.get("models")
    if not isinstance(models, Mapping):
        _check(report, "model_roots", False, "missing models map")
        return
    for model, spec in models.items():
        if not isinstance(spec, Mapping):
            _check(report, f"model_{model}", False, "malformed model spec")
            continue
        path = Path(str(spec.get("path", "")))
        exists = path.is_dir() and not path.is_symlink()
        _check(report, f"model_{model}_root", exists, str(path))
        if exists and deep_model_hash:
            try:
                observed = data_core.model_tree_sha256(path)
            except RuntimeError as exc:
                _check(report, f"model_{model}_tree_hash", False, str(exc))
            else:
                expected = spec.get("tree_sha256")
                _check(
                    report,
                    f"model_{model}_tree_hash",
                    observed == expected,
                    {"expected": expected, "observed": observed},
                )
        elif exists:
            report["warnings"].append(f"model_{model}_tree_hash_not_run_without_--deep-model-hash")

    required = config.get("required_remote_inputs")
    if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
        _check(report, "required_remote_inputs", False, "missing input list")
    else:
        missing = []
        for value in required:
            path = _resolve_remote_path(remote_root, value)
            if not path.is_file() or path.is_symlink():
                missing.append(str(path))
        _check(report, "required_remote_inputs", not missing, missing or list(required))


def plan_artifact_state(
    remote_root: Path, config: Mapping[str, Any], report: dict[str, Any]
) -> None:
    artifacts = config.get("artifacts")
    if not isinstance(artifacts, Mapping):
        _check(report, "artifact_config", False, "missing artifact map")
        return
    route_root = _resolve_remote_path(remote_root, artifacts["route_root"])
    state_root = _resolve_remote_path(remote_root, artifacts["route_state_root"])
    for model in ("olmoe", "llmjp"):
        route_dir = route_root / model
        ledger = state_root / f"route_calibration_{model}_consumption.json"
        if route_dir.exists():
            if route_dir.is_symlink():
                _check(report, f"route_{model}_root", False, "route root is a symlink")
                continue
            if not ledger.is_file():
                _check(report, f"route_{model}_one_shot_state", False, "output exists without ledger")
                continue
            try:
                joins, metadata = load_verified_joins(route_root, model)
            except RuntimeError as exc:
                _check(report, f"route_{model}_validation", False, str(exc))
            else:
                _check(
                    report,
                    f"route_{model}_validation",
                    True,
                    {"join_count": len(joins), "manifest_sha256": metadata.get("manifest_sha256")},
                )
        elif ledger.exists():
            _check(
                report,
                f"route_{model}_one_shot_state",
                False,
                "reservation ledger exists but route output is absent",
            )
        else:
            report["planned_actions"].append(f"capture_route_{model}")

    lut_path = _resolve_remote_path(remote_root, artifacts["lut"])
    if lut_path.exists():
        if lut_path.is_symlink():
            _check(report, "lut_root", False, "LUT path is a symlink")
            return
        for model in ("olmoe", "llmjp"):
            try:
                service = load_service_lut(lut_path, model)
            except RuntimeError as exc:
                _check(report, f"lut_{model}_validation", False, str(exc))
            else:
                _check(
                    report,
                    f"lut_{model}_validation",
                    True,
                    {"artifact_sha256": service.source_artifact_sha256},
                )
    else:
        report["planned_actions"].append("capture_primitive_lut")

    replay_outputs = artifacts.get("native_dry_run_outputs")
    if not isinstance(replay_outputs, Mapping):
        _check(report, "native_dry_run_outputs", False, "missing output paths")
    else:
        required_files = {
            "config.yaml",
            "metrics.json",
            "raw_results.jsonl",
            "environment.json",
            "source_manifest.json",
            "stdout.log",
            "summary.md",
        }
        for model, configured in replay_outputs.items():
            output = _resolve_remote_path(remote_root, configured)
            if not output.exists():
                report["planned_actions"].append(f"native_cpu_dry_run_{model}")
                continue
            if output.is_symlink():
                _check(
                    report,
                    f"native_cpu_dry_run_{model}_artifacts",
                    False,
                    "output directory is a symlink",
                )
                continue
            observed = {path.name for path in output.iterdir()} if output.is_dir() else set()
            _check(
                report,
                f"native_cpu_dry_run_{model}_artifacts",
                required_files <= observed,
                {"required": sorted(required_files), "observed": sorted(observed)},
            )


def validate_disk_capacity(
    remote_root: Path, config: Mapping[str, Any], report: dict[str, Any]
) -> None:
    environment = config.get("environment")
    if not isinstance(environment, Mapping):
        _check(report, "free_disk", False, "missing environment config")
        return
    thresholds = environment.get("minimum_free_disk_gib_by_action")
    if not isinstance(thresholds, Mapping):
        _check(report, "free_disk", False, "missing action-aware disk thresholds")
        return
    actions = set(report["planned_actions"])
    if any(value.startswith("capture_route_") for value in actions):
        action_class = "route_capture"
    elif actions & {
        "capture_primitive_lut",
        "native_cpu_dry_run_olmoe",
        "native_cpu_dry_run_llmjp",
    }:
        action_class = "lut_and_replay"
    else:
        action_class = "validation_only"
    try:
        minimum_free = int(thresholds[action_class])
        free = shutil.disk_usage(remote_root.parent).free
    except (KeyError, TypeError, ValueError, OSError) as exc:
        _check(report, "free_disk", False, str(exc))
        return
    _check(
        report,
        "free_disk",
        minimum_free >= 0 and free >= minimum_free * GIB,
        {
            "action_class": action_class,
            "planned_actions": sorted(actions),
            "free_gib": free / GIB,
            "minimum_gib": minimum_free,
        },
    )


def run_preflight(
    *,
    mode: str,
    repo_root: Path,
    remote_root: Path,
    config_path: Path,
    deep_model_hash: bool,
) -> dict[str, Any]:
    if mode not in {"static", "gpu"}:
        raise PreflightError("mode must be static or gpu")
    config = _load_json(config_path)
    report = _new_report(mode, repo_root, config_path)
    validate_config_identity(config, report)
    validate_static_sources(repo_root, config, report)
    if mode == "gpu":
        validate_gpu_environment(config, report)
        validate_remote_inputs(
            remote_root, config, report, deep_model_hash=deep_model_hash
        )
        plan_artifact_state(remote_root, config, report)
        validate_disk_capacity(remote_root, config, report)
    report["status"] = "READY" if not report["blockers"] else "BLOCKED"
    report["gpu_work_executed"] = False
    report["scientific_result"] = False
    report["approval_scope"] = (
        "STATIC_BUNDLE_ONLY"
        if mode == "static"
        else "CALIBRATION_CAPTURE_PREFLIGHT_ONLY_NOT_FORMAL_PHYSICAL_RUN"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "gpu"), required=True)
    parser.add_argument("--repo-root", type=Path, default=HERE.parents[4])
    parser.add_argument(
        "--remote-root", type=Path, default=Path("/root/autodl-tmp/ric_clean_v2_20260723")
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--deep-model-hash", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_preflight(
            mode=args.mode,
            repo_root=args.repo_root,
            remote_root=args.remote_root,
            config_path=args.config,
            deep_model_hash=args.deep_model_hash,
        )
        if args.output is not None:
            _write_report(args.output, report)
    except (OSError, ValueError, RuntimeError, PreflightError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())

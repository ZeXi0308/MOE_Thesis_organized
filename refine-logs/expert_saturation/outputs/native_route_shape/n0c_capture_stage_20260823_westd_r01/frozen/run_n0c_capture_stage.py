#!/usr/bin/env python3
"""Run the sealed N0c capture-enabled/no-export versus full-export triage."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "n0c-capture-stage-orchestration-v1"
CLAIM_CEILING = "FRESH_PROCESS_ASSOCIATIONAL_CAPTURE_TRIAGE_ONLY"
N0B_CAMPAIGN_COMPLETE_SHA256 = "484c63bfabf3b8efc5de0b9244eda2709fd81f99d5e688c4016f699ae3a163cb"
N0B_RUNTIME_MANIFEST_SHA256 = "b44c562b64c9e11eda72ec5b56a664c166b59836bbb2e80ed70e5f01505abed2"
WORKLOAD_SHA256 = "2bf4b4897c15b165fea90d730ed9136d0777535daab7f6807336c09a7c70cdbe"
DEVICE_PATCH_ID = "device-capture-no-export-v1"
RUNTIME_VARIANTS = ("stock", "stock-device", "valid-window", "valid-window-device")

HERE = Path(__file__).resolve().parent
ARM_RUNNER = HERE / "run_n0c_capture_stage_arm.py"
EVALUATOR = HERE / "evaluate_n0c_capture_stage.py"
DEVICE_PATCH = HERE / "vllm_patches" / "vllm-0.26-device-capture-only.patch"
COMMON_PATH = HERE / "run_valid_window_telemetry_gate.py"
COMMON_VALIDATOR_PATH = HERE / "vllm_patches" / "validate_valid_window_patch.py"

TARGETS: dict[str, dict[str, Any]] = {
    "stock_p512_b8_g2_w0": {
        "target_runtime": "stock",
        "source_bundle": "stock_on-r1",
        "source_batches_sha256": "b165c5d1c26ad97f4f15411cd8a1f8721aecea866f30849c51d8297a72bf30fa",
        "batch_id": "r01-p512-b8-g02-w00",
        "execution_order": 11,
        "batch_size": 8,
        "target_input_artifact_sha256": "37b9d87d894f4b314f97c3b6e1fda893c15ab3ee40a9b7870845ebb9de1ed491",
        "target_prompt_token_ids_sha256": "a9a36f51ab8a18bdf9178517cf74e7b4ac36de48738dea61eee66cee9e10aef0",
        "base_runtime_patch_id": "stock-vllm-0.26.0",
    },
    "valid_window_p512_b16_g1_w0": {
        "target_runtime": "valid-window",
        "source_bundle": "optimized_on-r1",
        "source_batches_sha256": "b3eb50db81a37b702aadd7731e1b13a4d7d1a1e8f5777d1b0bf49655594d749a",
        "batch_id": "r01-p512-b16-g01-w00",
        "execution_order": 35,
        "batch_size": 16,
        "target_input_artifact_sha256": "8ddb890932ca0b4f4eb712d7df9ecebc62283b732f19ed71abd4161f3022d1d2",
        "target_prompt_token_ids_sha256": "9bae1e480b20cc210da2f3cc14ea34f3f82632b339cb8b9a452b410756f430c0",
        "base_runtime_patch_id": "valid-window-clear-v1",
    },
}

LATIN_ORDERS = (
    ("n_a", "capture_only", "full_export", "n_b"),
    ("capture_only", "n_b", "n_a", "full_export"),
    ("full_export", "n_a", "n_b", "capture_only"),
    ("n_b", "full_export", "capture_only", "n_a"),
)
CAPTURE_MODES = {
    "n_a": "off",
    "capture_only": "device",
    "full_export": "full_export",
    "n_b": "off",
}


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = _load_module(COMMON_PATH, "n0c_common_orchestration")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _write_json_once(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def frozen_schedule() -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    target_ids = tuple(TARGETS)
    for round_index, arm_order in enumerate(LATIN_ORDERS):
        target_order = target_ids if round_index % 2 == 0 else tuple(reversed(target_ids))
        for target_id in target_order:
            target = TARGETS[target_id]
            for arm in arm_order:
                capture_mode = CAPTURE_MODES[arm]
                base_patch = target["base_runtime_patch_id"]
                patch_id = (
                    f"{base_patch}+{DEVICE_PATCH_ID}" if capture_mode == "device" else base_patch
                )
                schedule.append(
                    {
                        "target_id": target_id,
                        "target_runtime": target["target_runtime"],
                        "round": round_index,
                        "arm": arm,
                        "capture_mode": capture_mode,
                        "runtime_patch_id": patch_id,
                        "bundle": f"{target_id}-r{round_index}-{arm}",
                    }
                )
    return schedule


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"empty JSONL: {path}")
    return rows


def _safe_copy(source: Path, root: Path, relative: str) -> None:
    destination = root / relative
    if not destination.resolve().is_relative_to(root.resolve()):
        raise RuntimeError(f"unsafe input path: {relative}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _freeze_target(n0b: Path, frozen: Path, target_id: str, target: dict[str, Any]) -> Path:
    source_bundle = n0b / "bundles" / target["source_bundle"]
    batches = source_bundle / "batches.jsonl"
    if _sha256(batches) != target["source_batches_sha256"]:
        raise RuntimeError(f"N0b source batch manifest drifted for {target_id}")
    rows = _read_jsonl(batches)
    selected = [row for row in rows if int(row["execution_order"]) <= target["execution_order"]]
    selected.sort(key=lambda row: int(row["execution_order"]))
    if [int(row["execution_order"]) for row in selected] != list(
        range(target["execution_order"] + 1)
    ):
        raise RuntimeError(f"N0b prefix is incomplete for {target_id}")
    if selected[-1]["batch_id"] != target["batch_id"]:
        raise RuntimeError(f"wrong frozen terminal cell for {target_id}")

    keep = (
        "batch_id",
        "execution_order",
        "prompt_length",
        "batch_size",
        "group",
        "within_process_repeat",
        "input_artifact",
        "input_artifact_sha256",
        "prompt_token_ids_sha256",
    )
    plan: list[dict[str, Any]] = []
    target_root = frozen / "targets" / target_id
    target_root.mkdir(parents=True)
    for row in selected:
        reduced = {key: row[key] for key in keep}
        input_source = source_bundle / str(row["input_artifact"])
        if _sha256(input_source) != row["input_artifact_sha256"]:
            raise RuntimeError(f"N0b input artifact drifted: {input_source}")
        _safe_copy(input_source, target_root, str(row["input_artifact"]))
        plan.append(reduced)
    terminal = plan[-1]
    if (
        int(terminal["batch_size"]) != target["batch_size"]
        or terminal["input_artifact_sha256"] != target["target_input_artifact_sha256"]
        or terminal["prompt_token_ids_sha256"] != target["target_prompt_token_ids_sha256"]
    ):
        raise RuntimeError(f"target identity mismatch for {target_id}")
    payload = {
        "schema": "n0c-capture-target-spec-v1",
        "target_id": target_id,
        "target_runtime": target["target_runtime"],
        "source_bundle": target["source_bundle"],
        "source_batches_sha256": target["source_batches_sha256"],
        "prefix_plan_sha256": _json_sha256(plan),
        "prefix_records": plan,
        "target_record": terminal,
    }
    path = target_root / "target-spec.json"
    _write_json_once(path, payload)
    return path


def _freeze_inputs(root: Path, n0b: Path) -> tuple[dict[str, Path], dict[str, str]]:
    frozen = root / "frozen"
    patches = frozen / "vllm_patches"
    patches.mkdir(parents=True)
    paths: dict[str, Path] = {}
    sources = {
        "orchestrator": (Path(__file__).resolve(), frozen / Path(__file__).name),
        "common_orchestration": (COMMON_PATH, frozen / COMMON_PATH.name),
        "common_validator": (
            COMMON_VALIDATOR_PATH,
            patches / COMMON_VALIDATOR_PATH.name,
        ),
        "runner": (ARM_RUNNER, frozen / ARM_RUNNER.name),
        "evaluator": (EVALUATOR, frozen / EVALUATOR.name),
        "device_patch": (DEVICE_PATCH, patches / DEVICE_PATCH.name),
        "n0b_runtime_manifest": (
            n0b / "runtime-package-manifest.json",
            frozen / "n0b-runtime-package-manifest.json",
        ),
        "workload": (n0b / "frozen" / "workload_manifest.json", frozen / "workload.json"),
    }
    hashes: dict[str, str] = {}
    for role, (source, destination) in sources.items():
        shutil.copyfile(source, destination)
        paths[role] = destination
        hashes[str(destination.relative_to(root))] = _sha256(destination)
    if _sha256(paths["n0b_runtime_manifest"]) != N0B_RUNTIME_MANIFEST_SHA256:
        raise RuntimeError("N0b runtime manifest SHA-256 mismatch")
    if _sha256(paths["workload"]) != WORKLOAD_SHA256:
        raise RuntimeError("frozen workload SHA-256 mismatch")
    for target_id, target in TARGETS.items():
        target_spec = _freeze_target(n0b, frozen, target_id, target)
        paths[f"target:{target_id}"] = target_spec
        for artifact in sorted(target_spec.parent.rglob("*")):
            if artifact.is_file():
                hashes[str(artifact.relative_to(root))] = _sha256(artifact)
    return paths, hashes


def _copy_runtime(source: Path, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"runtime destination exists: {destination}")
    shutil.copytree(
        source / "vllm",
        destination / "vllm",
        symlinks=False,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def _create_runtimes(root: Path, n0b: Path, patch: Path) -> dict[str, dict[str, str]]:
    n0b_manifest = json.loads((n0b / "runtime-package-manifest.json").read_text())
    runtime_root = root / "runtime"
    manifests: dict[str, dict[str, str]] = {}
    for base in ("stock", "valid-window"):
        source = n0b / "runtime" / base
        expected_key = "stock" if base == "stock" else "optimized"
        if COMMON._package_manifest(source) != n0b_manifest[expected_key]:
            raise RuntimeError(f"sealed N0b {base} runtime does not match its manifest")
        destination = runtime_root / base
        _copy_runtime(source, destination)
        manifests[base] = COMMON._package_manifest(destination)
        if manifests[base] != n0b_manifest[expected_key]:
            raise RuntimeError(f"copied {base} runtime changed")

        device_name = f"{base}-device"
        device = runtime_root / device_name
        _copy_runtime(destination, device)
        COMMON._run_checked(
            ["git", "-C", str(device), "apply", "--check", "--include=vllm/**", str(patch)]
        )
        COMMON._run_checked(
            ["git", "-C", str(device), "apply", "--include=vllm/**", str(patch)]
        )
        manifests[device_name] = COMMON._package_manifest(device)
        differing = sorted(
            key
            for key in set(manifests[base]) | set(manifests[device_name])
            if manifests[base].get(key) != manifests[device_name].get(key)
        )
        if differing != ["vllm/v1/worker/gpu_model_runner.py"]:
            raise RuntimeError(f"device patch escaped its allowlist: {differing}")
    _write_json_once(root / "runtime-package-manifest.json", manifests)
    return manifests


def _runtime_variant(row: dict[str, Any]) -> str:
    suffix = "-device" if row["capture_mode"] == "device" else ""
    return f"{row['target_runtime']}{suffix}"


def _base_environment() -> dict[str, str]:
    environment = COMMON._base_environment()
    environment.pop("PYTHONPATH", None)
    environment.pop("N0C_DEVICE_CAPTURE_ONLY", None)
    return environment


def _arm_command(
    python: Path,
    runner: Path,
    root: Path,
    paths: dict[str, Path],
    row: dict[str, Any],
    package_manifest_sha256: str,
) -> list[str]:
    target_root = paths[f"target:{row['target_id']}"].parent
    return [
        str(python),
        str(runner),
        "--output-dir",
        str(root / "bundles" / row["bundle"]),
        "--target-spec",
        str(paths[f"target:{row['target_id']}"]),
        "--input-root",
        str(target_root),
        "--workload-manifest",
        str(paths["workload"]),
        "--target-id",
        row["target_id"],
        "--target-runtime",
        row["target_runtime"],
        "--round",
        str(row["round"]),
        "--arm",
        row["arm"],
        "--capture-mode",
        row["capture_mode"],
        "--runtime-patch-id",
        row["runtime_patch_id"],
        "--runtime-package-manifest-sha256",
        package_manifest_sha256,
        "--logical-runtime-variant",
        _runtime_variant(row),
        "--expected-runtime-root",
        str(root / "runtime" / _runtime_variant(row)),
        "--require-exclusive-gpu",
    ]


def _verify_runtime_manifests(root: Path, manifests: dict[str, dict[str, str]]) -> None:
    for name, expected in manifests.items():
        if COMMON._package_manifest(root / "runtime" / name) != expected:
            raise RuntimeError(f"runtime package drifted: {name}")


def _verify_runtime_key_files(root: Path, manifests: dict[str, dict[str, str]]) -> None:
    keys = (
        "vllm/__init__.py",
        "vllm/model_executor/layers/fused_moe/routed_experts_capturer.py",
        "vllm/v1/worker/gpu_model_runner.py",
    )
    for name, manifest in manifests.items():
        for relative in keys:
            path = root / "runtime" / name / relative
            if _sha256(path) != manifest[relative]:
                raise RuntimeError(f"runtime key source drifted: {name}:{relative}")


def _verify_campaign_runtime_imports(
    python: Path, root: Path, base_environment: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """Prove that every arm-visible overlay imports its own frozen vLLM package."""
    probes: dict[str, dict[str, Any]] = {}
    for variant in RUNTIME_VARIANTS:
        expected_root = (root / "runtime" / variant).resolve()
        environment = COMMON._overlay_environment(base_environment, expected_root)
        probe = COMMON._import_probe(python, environment)
        actual_root = Path(probe["source_root"]).resolve()
        if actual_root != expected_root:
            raise RuntimeError(
                f"campaign runtime import escaped expected root: "
                f"{variant}:{actual_root} != {expected_root}"
            )
        if probe["version"] != "0.26.0":
            raise RuntimeError(f"wrong vLLM version in campaign runtime {variant}")
        probes[variant] = {
            **probe,
            "logical_runtime_variant": variant,
            "expected_runtime_root": str(expected_root),
            "runtime_import_root_verified": True,
        }
    return probes


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    python = COMMON._interpreter_path(args.python)
    n0b = Path(args.n0b_campaign_root).resolve()
    if not python.is_file():
        raise RuntimeError(f"Python does not exist: {python}")
    if _sha256(n0b / "CAMPAIGN_COMPLETE.json") != N0B_CAMPAIGN_COMPLETE_SHA256:
        raise RuntimeError("N0b campaign seal mismatch")
    if _sha256(n0b / "runtime-package-manifest.json") != N0B_RUNTIME_MANIFEST_SHA256:
        raise RuntimeError("N0b runtime manifest mismatch")
    if _sha256(n0b / "frozen" / "workload_manifest.json") != WORKLOAD_SHA256:
        raise RuntimeError("N0b workload mismatch")
    if COMMON._gpu_processes():
        raise RuntimeError("GPU is not isolated")
    environment = _base_environment()
    probes = {}
    for name in ("stock", "valid-window"):
        overlay = COMMON._overlay_environment(environment, n0b / "runtime" / name)
        probe = COMMON._import_probe(python, overlay)
        if Path(probe["source_root"]).resolve() != (n0b / "runtime" / name).resolve():
            raise RuntimeError(f"N0b {name} runtime import escaped the overlay")
        if probe["version"] != "0.26.0":
            raise RuntimeError(f"wrong vLLM version in {name}")
        probes[name] = probe
    model_cache = COMMON._model_cache_probe(python, environment)
    return {
        "python": str(python),
        "n0b_campaign_root": str(n0b),
        "n0b_campaign_complete_sha256": N0B_CAMPAIGN_COMPLETE_SHA256,
        "n0b_runtime_manifest_sha256": N0B_RUNTIME_MANIFEST_SHA256,
        "workload_sha256": WORKLOAD_SHA256,
        "runtime_imports": probes,
        "model_cache": model_cache,
        "exclusive_gpu_verified": True,
    }


def _append_event(path: Path, value: dict[str, Any]) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def execute(args: argparse.Namespace) -> int:
    root = Path(args.output_root).resolve()
    if root.exists():
        raise RuntimeError(f"output root already exists: {root}")
    root.mkdir(parents=True)
    events = root / "events.jsonl"
    completed: list[str] = []
    try:
        preflight = _preflight(args)
        n0b = Path(args.n0b_campaign_root).resolve()
        paths, frozen_hashes = _freeze_inputs(root, n0b)
        manifests = _create_runtimes(root, n0b, paths["device_patch"])
        manifest_sha = {name: _json_sha256(value) for name, value in manifests.items()}
        if set(manifests) != set(RUNTIME_VARIANTS):
            raise RuntimeError(f"campaign runtime variants drifted: {sorted(manifests)}")
        base_environment = _base_environment()
        _verify_runtime_manifests(root, manifests)
        campaign_runtime_imports = _verify_campaign_runtime_imports(
            Path(preflight["python"]), root, base_environment
        )
        schedule = frozen_schedule()
        plan = {
            "schema": SCHEMA,
            "claim_ceiling": CLAIM_CEILING,
            "research_question": (
                "Does the N0b token drift already appear when route capture is enabled "
                "without D2H/export, or only on the full export path?"
            ),
            "prestate_boundary": "fresh processes replay an identical semantic prefix; physical runtime state is not restored",
            "targets": TARGETS,
            "rounds": 4,
            "schedule": schedule,
            "runtime_package_manifest_sha256": manifest_sha,
            "campaign_runtime_imports": campaign_runtime_imports,
            "frozen_input_sha256": frozen_hashes,
            "preflight": preflight,
            "anti_claims": [
                "fresh-process association is not same-physical-prestate causality",
                "capture-only retains scheduler route-manager bookkeeping and is not a pure device-kernel treatment",
                "this Gate does not attribute a first hidden/router/logit divergence",
                "this Gate does not measure timing overhead or request-level serving benefit",
                "no result unlocks decode-cap or an admission Controller",
            ],
        }
        _write_json_once(root / "run_plan.json", plan)
        python = Path(preflight["python"])
        (root / "bundles").mkdir()
        (root / "logs").mkdir()
        for index, row in enumerate(schedule):
            _verify_runtime_key_files(root, manifests)
            variant = _runtime_variant(row)
            environment = COMMON._overlay_environment(
                base_environment, root / "runtime" / variant
            )
            if row["capture_mode"] == "device":
                environment["N0C_DEVICE_CAPTURE_ONLY"] = "1"
            command = _arm_command(
                python,
                paths["runner"],
                root,
                paths,
                row,
                manifest_sha[variant],
            )
            _append_event(
                events,
                {"event": "ARM_START", "schedule_index": index, "bundle": row["bundle"], "time": time.time()},
            )
            code, elapsed, cleanup = COMMON._run_arm(
                command,
                root / "logs" / f"{row['bundle']}.log",
                args.arm_timeout_seconds,
                environment,
            )
            if code != 0 or not (root / "bundles" / row["bundle"] / "RUN_COMPLETE.json").is_file():
                raise RuntimeError(f"arm failed without retry: {row['bundle']} exit={code}")
            completed.append(row["bundle"])
            _append_event(
                events,
                {
                    "event": "ARM_COMPLETE",
                    "schedule_index": index,
                    "bundle": row["bundle"],
                    "elapsed_seconds": elapsed,
                    "cleanup": cleanup,
                    "time": time.time(),
                },
            )
        _verify_runtime_manifests(root, manifests)
        verdict = root / "n0c-stage-a-verdict.json"
        evaluation = subprocess.run(
            [
                str(python),
                str(paths["evaluator"]),
                "--campaign-root",
                str(root),
                "--output",
                str(verdict),
            ],
            text=True,
            capture_output=True,
            timeout=args.evaluation_timeout_seconds,
            check=False,
            env=base_environment,
        )
        (root / "logs" / "evaluator.log").write_text(evaluation.stdout + evaluation.stderr)
        if evaluation.returncode != 0 or not verdict.is_file():
            raise RuntimeError(f"evaluator rejected campaign: exit={evaluation.returncode}")
        _write_json_once(
            root / "CAMPAIGN_COMPLETE.json",
            {
                "status": "CAMPAIGN_COMPLETE",
                "schema": SCHEMA,
                "claim_ceiling": CLAIM_CEILING,
                "completed_arms": completed,
                "all_32_arms_retained": len(completed) == 32,
                "run_plan_sha256": _sha256(root / "run_plan.json"),
                "runtime_package_manifest_sha256": _sha256(root / "runtime-package-manifest.json"),
                "verdict_sha256": _sha256(verdict),
                "gpu_idle_after_campaign": not COMMON._gpu_processes(),
            },
        )
        return 0
    except BaseException as exc:
        _write_json_once(
            root / "CAMPAIGN_ABORTED.json",
            {
                "status": "CAMPAIGN_ABORTED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_arms": completed,
                "retry_performed": False,
            },
        )
        return 2


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--python", default=sys.executable)
    common.add_argument("--n0b-campaign-root", required=True)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", parents=[common])
    run = commands.add_parser("run", parents=[common])
    run.add_argument("--output-root", required=True)
    run.add_argument("--arm-timeout-seconds", type=int, default=1800)
    run.add_argument("--evaluation-timeout-seconds", type=int, default=300)
    return parser


def _raise_interrupt(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "preflight":
            print(json.dumps(_preflight(args), indent=2, sort_keys=True))
            return
        if args.arm_timeout_seconds <= 0 or args.evaluation_timeout_seconds <= 0:
            raise RuntimeError("timeouts must be positive")
        signal.signal(signal.SIGTERM, _raise_interrupt)
        signal.signal(signal.SIGHUP, _raise_interrupt)
        raise SystemExit(execute(args))
    except RuntimeError as exc:
        print(json.dumps({"status": "INVALID_ORCHESTRATION", "error": str(exc)}))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()

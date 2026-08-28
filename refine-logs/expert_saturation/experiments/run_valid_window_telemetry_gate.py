#!/usr/bin/env python3
"""Run the frozen two-repeat stock versus valid-window telemetry Gate.

The driver freezes its inputs, builds a patched vLLM package overlay without
modifying the stock environment, executes all eight fresh-process arms, keeps
every bundle, and calls the existing fail-closed comparator exactly once.
"""

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


SCHEMA = "vllm-valid-window-telemetry-orchestration-v1"
CLAIM_CEILING = "NATIVE_OFFLINE_FIXED_BATCH_TELEMETRY_IMPLEMENTATION_ONLY"
PROCESS_REPEATS = (0, 1)
FROZEN_MAX_P95_ABSOLUTE_TIMING_DEVIATION_PCT = 5.0
FROZEN_WORKLOAD_SHA256 = "2bf4b4897c15b165fea90d730ed9136d0777535daab7f6807336c09a7c70cdbe"
MODEL = "allenai/OLMoE-1B-7B-0924"
REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
DTYPE = "bfloat16"
BATCH_SIZES = (4, 8, 16)
PROMPT_LENGTHS = (128, 512)
OUTPUT_TOKENS = 16
GROUPS = 6
SEED = 20260823
ORDER_SEED = 20260823
MAX_MODEL_LEN = 1024
MAX_NUM_SEQS = 32
MAX_NUM_BATCHED_TOKENS = 8192
GPU_MEMORY_UTILIZATION = 0.80
HF_HOME = "/root/autodl-tmp/hf-cache"
HF_HUB_CACHE = f"{HF_HOME}/hub"
MODEL_SNAPSHOT = (
    Path(HF_HUB_CACHE)
    / "models--allenai--OLMoE-1B-7B-0924"
    / "snapshots"
    / REVISION
)
CACHE_ALIAS_KEYS = (
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "PYTORCH_TRANSFORMERS_CACHE",
    "PYTORCH_PRETRAINED_BERT_CACHE",
)

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_vllm_route_shape_probe.py"
COMPARATOR = HERE / "compare_vllm_telemetry_implementations.py"
PAIR_COMPARATOR = HERE / "compare_vllm_route_probe_runs.py"
VALIDATOR_PATH = HERE / "vllm_patches" / "validate_valid_window_patch.py"
PATCH_PATH = HERE / "vllm_patches" / "vllm-0.26-valid-route-window.patch"
E0_REVIEWED_SHA256 = {
    "runner": "9a83209363a0fb68568a3c85cc42dfb578c407f4ae8c4306d78f147ffe433e44",
    "pair_comparator": "c6d16a1f219438a0525055096bd77278bc1a83119e9079fe37775abcc3b7013d",
    "comparator": "f263366b895c213ba95f42f4004a948e8697a3a90e63b99f4122404cd3040c80",
    "validator": "55e4a1f7d2d51054213e3af78f8b4a09fe5d58e84d30619af7a422e43532aca7",
    "patch": "862b3ff7732fd4ccac4ffeba923174ab3d662e57834a981eb329aba893e0d87b",
}

# r1 reverses both implementation and capture order. Workload order inside
# every arm remains identical because ORDER_SEED is frozen across repeats.
FROZEN_SCHEDULE = (
    ("stock_off", 0, "original", False, "stock-vllm-0.26.0"),
    ("stock_on", 0, "original", True, "stock-vllm-0.26.0"),
    ("optimized_off", 0, "patched", False, "valid-window-clear-v1"),
    ("optimized_on", 0, "patched", True, "valid-window-clear-v1"),
    ("optimized_on", 1, "patched", True, "valid-window-clear-v1"),
    ("optimized_off", 1, "patched", False, "valid-window-clear-v1"),
    ("stock_on", 1, "original", True, "stock-vllm-0.26.0"),
    ("stock_off", 1, "original", False, "stock-vllm-0.26.0"),
)


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_module(VALIDATOR_PATH, "valid_window_source_validator")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _interpreter_path(value: str) -> Path:
    """Return an absolute interpreter path without dereferencing a venv symlink."""
    return Path(os.path.abspath(os.path.expanduser(value)))


def _write_json_once(path: Path, payload: Any) -> None:
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _append_event(path: Path, event: dict[str, Any]) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def _run_checked(command: Sequence[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            list(command), text=True, capture_output=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"command timed out after {timeout}s: {command[0]}") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-2000:]
        raise RuntimeError(f"command failed ({result.returncode}): {detail}")
    return result


def _base_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for key in CACHE_ALIAS_KEYS:
        environment.pop(key, None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "VLLM_BATCH_INVARIANT": "0",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "PYTHONHASHSEED": str(SEED),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HOME": HF_HOME,
            "HF_HUB_CACHE": HF_HUB_CACHE,
            "PYTHONNOUSERSITE": "1",
        }
    )
    return environment


def _overlay_environment(base: dict[str, str], overlay_root: Path) -> dict[str, str]:
    environment = base.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(overlay_root) if not existing else str(overlay_root) + os.pathsep + existing
    )
    return environment


def _import_probe(python: Path, environment: dict[str, str]) -> dict[str, str]:
    code = (
        "import json,pathlib,vllm;"
        "print(json.dumps({'source_root':str(pathlib.Path(vllm.__file__).resolve().parent.parent),"
        "'module_file':str(pathlib.Path(vllm.__file__).resolve()),'version':vllm.__version__}))"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", code], text=True, capture_output=True,
            timeout=120, check=False, env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("vLLM import probe timed out") from exc
    if result.returncode:
        raise RuntimeError(f"vLLM import probe failed: {(result.stderr or result.stdout)[-2000:]}")
    try:
        payload = json.loads(result.stdout.strip())
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid vLLM import probe output: {result.stdout!r}") from exc
    return {key: str(payload[key]) for key in ("source_root", "module_file", "version")}


def _model_cache_probe(python: Path, environment: dict[str, str]) -> dict[str, str]:
    code = (
        "import json,pathlib;from huggingface_hub import snapshot_download;"
        f"p=snapshot_download({MODEL!r},revision={REVISION!r},local_files_only=True);"
        "print(json.dumps({'snapshot_path':str(pathlib.Path(p).resolve())}))"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", code], text=True, capture_output=True,
            timeout=120, check=False, env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("frozen model-cache probe timed out") from exc
    if result.returncode:
        raise RuntimeError(
            f"frozen model revision is unavailable offline: "
            f"{(result.stderr or result.stdout)[-2000:]}"
        )
    try:
        payload = json.loads(result.stdout.strip())
        snapshot = _interpreter_path(str(payload["snapshot_path"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid model-cache probe output: {result.stdout!r}") from exc
    expected = _interpreter_path(str(MODEL_SNAPSHOT))
    if not snapshot.is_dir() or snapshot != expected:
        raise RuntimeError(f"model-cache probe resolved the wrong revision: {snapshot}")
    return {
        "hf_home": environment["HF_HOME"],
        "hf_hub_cache": environment["HF_HUB_CACHE"],
        "cleared_cache_aliases": list(CACHE_ALIAS_KEYS),
        "snapshot_path": str(snapshot),
    }


def _gpu_processes() -> list[str]:
    result = _run_checked(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _source_report(source_root: Path, patch_path: Path = PATCH_PATH) -> dict[str, Any]:
    return VALIDATOR.validate(source_root.resolve(), patch_path.resolve())


def _require_source_state(
    source_root: Path, expected: str, patch_path: Path = PATCH_PATH
) -> dict[str, Any]:
    report = _source_report(source_root, patch_path)
    if not report.get("valid") or report.get("source_state") != expected:
        raise RuntimeError(
            f"vLLM source state is not exact {expected}: {report.get('source_state')}:"
            f"{report.get('errors')}"
        )
    return report


def _copy_package(source_root: Path, destination_root: Path) -> None:
    if destination_root.exists():
        raise RuntimeError(f"runtime snapshot already exists: {destination_root}")
    shutil.copytree(
        source_root / "vllm", destination_root / "vllm", symlinks=False,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for relative in VALIDATOR.FILES:
        if (destination_root / relative).is_symlink():
            raise RuntimeError(f"patch target is a symlink: {relative}")


def _package_manifest(root: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for path in sorted((root / "vllm").rglob("*")):
        if not path.is_file() or path.suffix == ".pyc" or "__pycache__" in path.parts:
            continue
        manifest[str(path.relative_to(root))] = _sha256(path)
    if not manifest:
        raise RuntimeError(f"empty vLLM package snapshot: {root}")
    return manifest


def _create_runtime_snapshots(
    installed_root: Path, runtime_root: Path, patch_path: Path
) -> dict[str, Any]:
    stock_root = runtime_root / "stock"
    optimized_root = runtime_root / "valid-window"
    _copy_package(installed_root, stock_root)
    stock_validation = _require_source_state(stock_root, "original", patch_path)
    _copy_package(stock_root, optimized_root)
    optimized_before = _require_source_state(optimized_root, "original", patch_path)
    base = ["git", "-C", str(optimized_root), "apply"]
    include = ["--include=vllm/**"]
    _run_checked(base + ["--check"] + include + [str(patch_path)])
    _run_checked(base + include + [str(patch_path)])
    optimized_validation = _require_source_state(optimized_root, "patched", patch_path)
    stock_manifest = _package_manifest(stock_root)
    optimized_manifest = _package_manifest(optimized_root)
    differing = sorted(
        relative for relative in set(stock_manifest) | set(optimized_manifest)
        if stock_manifest.get(relative) != optimized_manifest.get(relative)
    )
    expected_differing = sorted(VALIDATOR.FILES)
    if differing != expected_differing:
        raise RuntimeError(
            f"runtime snapshots differ outside the frozen patch surface: {differing}"
        )
    return {
        "stock_root": str(stock_root),
        "optimized_root": str(optimized_root),
        "stock_validation": stock_validation,
        "optimized_before_patch_validation": optimized_before,
        "optimized_validation": optimized_validation,
        "stock_manifest": stock_manifest,
        "optimized_manifest": optimized_manifest,
        "differing_files": differing,
    }


def _verify_runtime_snapshots(runtime: dict[str, Any], patch_path: Path) -> None:
    stock_root = Path(runtime["stock_root"])
    optimized_root = Path(runtime["optimized_root"])
    _require_source_state(stock_root, "original", patch_path)
    _require_source_state(optimized_root, "patched", patch_path)
    if _package_manifest(stock_root) != runtime["stock_manifest"]:
        raise RuntimeError("stock runtime snapshot drifted")
    if _package_manifest(optimized_root) != runtime["optimized_manifest"]:
        raise RuntimeError("optimized runtime snapshot drifted")


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    python = _interpreter_path(args.python)
    if not python.is_file():
        raise RuntimeError(f"Python interpreter does not exist: {python}")
    environment = _base_environment()
    probe = _import_probe(python, environment)
    detected_root = Path(probe["source_root"]).resolve()
    source_root = Path(args.vllm_source_root).resolve() if args.vllm_source_root else detected_root
    if source_root != detected_root:
        raise RuntimeError(
            f"requested source root {source_root} does not back interpreter {detected_root}"
        )
    if probe["version"] != "0.26.0":
        raise RuntimeError(f"expected vLLM 0.26.0, found {probe['version']}")
    source = _require_source_state(source_root, "original")
    model_cache = _model_cache_probe(python, environment)
    processes = _gpu_processes()
    if processes:
        raise RuntimeError(f"GPU is not isolated: {processes}")
    return {
        "python": str(python),
        "vllm_source_root": str(source_root),
        "vllm_import": probe,
        "source": source,
        "model_cache": model_cache,
        "exclusive_gpu_verified": True,
        "compute_processes": processes,
    }


def _freeze_inputs(root: Path, workload_bytes: bytes) -> tuple[dict[str, str], dict[str, Path]]:
    frozen = root / "frozen"
    patches = frozen / "vllm_patches"
    patches.mkdir(parents=True)
    sources = {
        "orchestrator": (Path(__file__).resolve(), frozen / "run_valid_window_telemetry_gate.py"),
        "runner": (RUNNER, frozen / RUNNER.name),
        "comparator": (COMPARATOR, frozen / COMPARATOR.name),
        "pair_comparator": (PAIR_COMPARATOR, frozen / PAIR_COMPARATOR.name),
        "validator": (VALIDATOR_PATH, patches / VALIDATOR_PATH.name),
        "patch": (PATCH_PATH, patches / PATCH_PATH.name),
    }
    hashes: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for name, (source, destination) in sources.items():
        shutil.copyfile(source, destination)
        relative = str(destination.relative_to(root))
        hashes[relative] = _sha256(destination)
        paths[name] = destination
    workload = frozen / "workload_manifest.json"
    workload.write_bytes(workload_bytes)
    workload_relative = str(workload.relative_to(root))
    hashes[workload_relative] = _sha256(workload)
    paths["workload"] = workload
    if hashes[workload_relative] != FROZEN_WORKLOAD_SHA256:
        raise RuntimeError("staged workload differs from the frozen SHA-256")
    drift = {
        role: {"actual": _sha256(paths[role]), "expected": expected}
        for role, expected in E0_REVIEWED_SHA256.items()
        if _sha256(paths[role]) != expected
    }
    if drift:
        raise RuntimeError(f"control source differs from the E0-reviewed snapshot: {drift}")
    return hashes, paths


def _verify_frozen(root: Path, hashes: dict[str, str]) -> None:
    drift = [relative for relative, digest in hashes.items() if _sha256(root / relative) != digest]
    if drift:
        raise RuntimeError(f"frozen input drift: {drift}")


def _arm_command(
    python: Path, runner: Path, workload: Path, output: Path,
    repeat: int, capture_routes: bool, patch_id: str,
) -> list[str]:
    return [
        str(python), str(runner), "--output-dir", str(output),
        "--workload-manifest", str(workload), "--model", MODEL,
        "--revision", REVISION, "--dtype", DTYPE, "--batch-sizes",
        *map(str, BATCH_SIZES), "--prompt-lengths", *map(str, PROMPT_LENGTHS),
        "--output-tokens", str(OUTPUT_TOKENS), "--groups", str(GROUPS),
        "--within-process-repeats", "1", "--process-repeat", str(repeat),
        "--seed", str(SEED), "--order-seed", str(ORDER_SEED),
        "--max-model-len", str(MAX_MODEL_LEN), "--max-num-seqs", str(MAX_NUM_SEQS),
        "--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--runtime-patch-id", patch_id,
        "--capture-routes" if capture_routes else "--no-capture-routes",
        "--enforce-eager", "--require-exclusive-gpu",
    ]


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - cannot occur for our child group
        return True
    return True


def _wait_for_process_group_exit(
    pgid: int, timeout: float, leader: subprocess.Popen[bytes] | None = None
) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        if leader is not None:
            leader.poll()  # Reap a dead leader so it cannot keep the PGID visible as a zombie.
        if not _process_group_exists(pgid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


def _wait_for_gpu_idle(timeout: float) -> list[str]:
    deadline = time.monotonic() + timeout
    while True:
        processes = _gpu_processes()
        if not processes:
            return []
        if time.monotonic() >= deadline:
            return processes
        time.sleep(0.5)


def _terminate_process_group(
    process: subprocess.Popen[bytes], *, term_timeout: float = 10.0,
    kill_timeout: float = 10.0, gpu_idle_timeout: float = 30.0,
) -> dict[str, Any]:
    """Reap the entire session and prove that it released the experiment GPU."""
    pgid = process.pid
    sent_sigterm = False
    sent_sigkill = False
    if _process_group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGTERM)
            sent_sigterm = True
        except ProcessLookupError:
            pass
    if not _wait_for_process_group_exit(pgid, term_timeout, process):
        try:
            os.killpg(pgid, signal.SIGKILL)
            sent_sigkill = True
        except ProcessLookupError:
            pass
        if not _wait_for_process_group_exit(pgid, kill_timeout, process):
            raise RuntimeError(f"process group {pgid} survived SIGKILL")
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - inconsistent OS state
        raise RuntimeError(f"process-group leader {pgid} was not reaped") from exc
    remaining = _wait_for_gpu_idle(gpu_idle_timeout)
    if remaining:
        raise RuntimeError(f"GPU processes remained after process-group cleanup: {remaining}")
    return {
        "pgid": pgid,
        "sent_sigterm": sent_sigterm,
        "sent_sigkill": sent_sigkill,
        "process_group_absent": True,
        "gpu_idle_verified": True,
        "gpu_processes_after_cleanup": [],
    }


def _run_arm(
    command: Sequence[str], log_path: Path, timeout: int, environment: dict[str, str]
) -> tuple[int, float, dict[str, Any]]:
    started = time.monotonic()
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            list(command), stdout=log, stderr=subprocess.STDOUT,
            env=environment, start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout)
            cleanup = _terminate_process_group(process)
            log.write(
                ("ORCHESTRATOR_PROCESS_CLEANUP=" + json.dumps(cleanup, sort_keys=True) + "\n").encode()
            )
            return returncode, time.monotonic() - started, cleanup
        except subprocess.TimeoutExpired:
            log.write(f"\nORCHESTRATOR_TIMEOUT_SECONDS={timeout}\n".encode())
            cleanup = _terminate_process_group(process)
            log.write(
                ("ORCHESTRATOR_PROCESS_CLEANUP=" + json.dumps(cleanup, sort_keys=True) + "\n").encode()
            )
            return 124, time.monotonic() - started, cleanup
        except BaseException as exc:
            try:
                cleanup = _terminate_process_group(process)
                log.write(
                    ("ORCHESTRATOR_PROCESS_CLEANUP=" + json.dumps(cleanup, sort_keys=True) + "\n").encode()
                )
            except BaseException as cleanup_exc:
                raise RuntimeError(
                    f"arm interruption cleanup failed after {type(exc).__name__}: {cleanup_exc}"
                ) from cleanup_exc
            raise


def _verify_bundle(pair_comparator: Path, bundle: Path) -> dict[str, Any]:
    module = _load_module(pair_comparator, f"pair_comparator_{time.monotonic_ns()}")
    report = module.verify_bundle(bundle)
    if not report.get("valid"):
        raise RuntimeError(f"bundle integrity failed: {bundle.name}:{report.get('errors')}")
    return report


def _comparison_command(python: Path, comparator: Path, root: Path) -> list[str]:
    bundles = {
        arm: [root / "bundles" / f"{arm}-r{repeat}" for repeat in PROCESS_REPEATS]
        for arm in ("stock_off", "stock_on", "optimized_off", "optimized_on")
    }
    return [
        str(python), str(comparator),
        "--stock-off", *map(str, bundles["stock_off"]),
        "--stock-on", *map(str, bundles["stock_on"]),
        "--optimized-off", *map(str, bundles["optimized_off"]),
        "--optimized-on", *map(str, bundles["optimized_on"]),
        "--output", str(root / "valid-window-gate.json"),
    ]


def _run_comparator(
    command: Sequence[str], log: Path, timeout: int, environment: dict[str, str], report: Path
) -> tuple[int, dict[str, Any]]:
    with log.open("xb") as stream:
        try:
            result = subprocess.run(
                list(command), stdout=stream, stderr=subprocess.STDOUT,
                env=environment, timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stream.write(f"\nORCHESTRATOR_TIMEOUT_SECONDS={timeout}\n".encode())
            raise RuntimeError("comparator timed out") from exc
    if not report.is_file() or result.returncode not in {0, 1, 2}:
        raise RuntimeError("comparator did not produce a valid write-once report")
    try:
        payload = json.loads(report.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError("comparator report is not valid JSON") from exc
    status = str(payload.get("status", ""))
    expected = 0 if status == "VALID_WINDOW_TELEMETRY_QUALIFIED" else (
        2 if status.startswith("INVALID_") else 1
    )
    if result.returncode != expected:
        raise RuntimeError(f"comparator status/exit mismatch: {status}:{result.returncode}!={expected}")
    return result.returncode, payload


def _plan(
    preflight: dict[str, Any], runtime: dict[str, Any], hashes: dict[str, str]
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "claim_ceiling": CLAIM_CEILING,
        "process_repeats": list(PROCESS_REPEATS),
        "schedule": [
            {"arm": arm, "process_repeat": repeat, "source_state": state,
             "capture_routes": capture, "runtime_patch_id": patch_id}
            for arm, repeat, state, capture, patch_id in FROZEN_SCHEDULE
        ],
        "workload_sha256": FROZEN_WORKLOAD_SHA256,
        "model": MODEL, "revision": REVISION, "dtype": DTYPE,
        "batch_sizes": list(BATCH_SIZES), "prompt_lengths": list(PROMPT_LENGTHS),
        "output_tokens": OUTPUT_TOKENS, "groups": GROUPS,
        "within_process_repeats": 1, "seed": SEED, "order_seed": ORDER_SEED,
        "max_model_len": MAX_MODEL_LEN, "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_p95_absolute_timing_deviation_pct": FROZEN_MAX_P95_ABSOLUTE_TIMING_DEVIATION_PCT,
        "environment_overrides": {
            key: value for key, value in _base_environment().items()
            if key in {"CUDA_VISIBLE_DEVICES", "VLLM_BATCH_INVARIANT",
                       "VLLM_USE_FLASHINFER_SAMPLER", "PYTHONHASHSEED",
                       "HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE",
                       "TRANSFORMERS_OFFLINE", "PYTHONNOUSERSITE"}
        },
        "preflight": preflight,
        "runtime_snapshots": {
            key: value for key, value in runtime.items()
            if key not in {"stock_manifest", "optimized_manifest"}
        },
        "frozen_input_sha256": hashes,
        "anti_claims": [
            "telemetry qualification is not pressure-latency evidence",
            "telemetry qualification is not scheduling or admission headroom",
            "single-GPU fixed batching is not online serving or Expert Parallel evidence",
        ],
    }


def _write_abort(root: Path, error: BaseException, completed: list[str]) -> None:
    path = root / "CAMPAIGN_ABORTED.json"
    if not path.exists():
        _write_json_once(
            path,
            {"schema": SCHEMA, "status": "INVALID_ORCHESTRATION",
             "failure": f"{type(error).__name__}: {error}",
             "completed_arms": completed, "no_automatic_retry": True},
        )


def execute(args: argparse.Namespace) -> int:
    workload = Path(args.workload_manifest).resolve()
    try:
        workload_bytes = workload.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read workload manifest: {workload}") from exc
    if hashlib.sha256(workload_bytes).hexdigest() != FROZEN_WORKLOAD_SHA256:
        raise RuntimeError("workload manifest is missing or differs from the frozen SHA-256")
    root = Path(args.output_root).resolve()
    if root.exists():
        raise RuntimeError(f"output root already exists: {root}")
    preflight = _preflight(args)
    python = Path(preflight["python"])
    stock_root = Path(preflight["vllm_source_root"])
    base_environment = _base_environment()
    root.mkdir(parents=True, exist_ok=False)
    for directory in ("bundles", "logs", "bundle_integrity"):
        (root / directory).mkdir()
    events = root / "events.jsonl"
    completed: list[str] = []
    try:
        hashes, frozen = _freeze_inputs(root, workload_bytes)
        runtime = _create_runtime_snapshots(stock_root, root / "runtime", frozen["patch"])
        runtime_manifest_path = root / "runtime-package-manifest.json"
        _write_json_once(
            runtime_manifest_path,
            {"stock": runtime["stock_manifest"],
             "optimized": runtime["optimized_manifest"],
             "differing_files": runtime["differing_files"]},
        )
        runtime_manifest_relative = str(runtime_manifest_path.relative_to(root))
        hashes[runtime_manifest_relative] = _sha256(runtime_manifest_path)
        stock_snapshot = Path(runtime["stock_root"]).resolve()
        optimized_snapshot = Path(runtime["optimized_root"]).resolve()
        stock_environment = _overlay_environment(base_environment, stock_snapshot)
        optimized_environment = _overlay_environment(base_environment, optimized_snapshot)
        stock_probe = _import_probe(python, stock_environment)
        optimized_probe = _import_probe(python, optimized_environment)
        if Path(stock_probe["source_root"]).resolve() != stock_snapshot or stock_probe["version"] != "0.26.0":
            raise RuntimeError(f"stock import did not resolve to the campaign snapshot: {stock_probe}")
        if Path(optimized_probe["source_root"]).resolve() != optimized_snapshot or optimized_probe["version"] != "0.26.0":
            raise RuntimeError(f"optimized import did not resolve to the patched overlay: {optimized_probe}")
        runtime["stock_import_probe"] = stock_probe
        runtime["optimized_import_probe"] = optimized_probe
        _write_json_once(root / "run_plan.json", _plan(preflight, runtime, hashes))
        for arm, repeat, state, capture, patch_id in FROZEN_SCHEDULE:
            _verify_frozen(root, hashes)
            _verify_runtime_snapshots(runtime, frozen["patch"])
            processes = _gpu_processes()
            if processes:
                raise RuntimeError(f"GPU lost isolation before {arm}:r{repeat}:{processes}")
            output = root / "bundles" / f"{arm}-r{repeat}"
            command = _arm_command(
                python, frozen["runner"], frozen["workload"], output,
                repeat, capture, patch_id,
            )
            environment = stock_environment if state == "original" else optimized_environment
            _append_event(events, {"event": "arm_start", "arm": arm,
                                   "process_repeat": repeat, "source_state": state,
                                   "command": command})
            returncode, elapsed, cleanup = _run_arm(
                command, root / "logs" / f"{arm}-r{repeat}.log",
                args.arm_timeout_seconds, environment,
            )
            complete = (output / "RUN_COMPLETE.json").is_file()
            _append_event(events, {"event": "arm_finish", "arm": arm,
                                   "process_repeat": repeat, "returncode": returncode,
                                   "elapsed_seconds": elapsed, "bundle_complete": complete,
                                   "process_cleanup": cleanup})
            if returncode or not complete:
                raise RuntimeError(f"arm failed or incomplete: {arm}:r{repeat}:exit={returncode}")
            integrity = _verify_bundle(frozen["pair_comparator"], output)
            _write_json_once(root / "bundle_integrity" / f"{arm}-r{repeat}.json", integrity)
            completed.append(f"{arm}:r{repeat}")
        remaining = _gpu_processes()
        if remaining:
            raise RuntimeError(f"GPU processes remained after final arm: {remaining}")
        _verify_frozen(root, hashes)
        _verify_runtime_snapshots(runtime, frozen["patch"])
        report_path = root / "valid-window-gate.json"
        code, report = _run_comparator(
            _comparison_command(python, frozen["comparator"], root),
            root / "logs" / "comparator.log", args.comparison_timeout_seconds,
            base_environment, report_path,
        )
        final = {
            "schema": SCHEMA, "status": report.get("status"),
            "failure_category": report.get("failure_category"),
            "claim_ceiling": CLAIM_CEILING, "completed_arms": completed,
            "all_eight_arms_retained": len(completed) == len(FROZEN_SCHEDULE),
            "stock_source_untouched_and_original": True,
            "comparison_exit_code": code,
            "run_plan_sha256": _sha256(root / "run_plan.json"),
            "runtime_package_manifest_sha256": _sha256(runtime_manifest_path),
            "comparison_sha256": _sha256(report_path),
            "arm_seal_sha256": {
                item: _sha256(root / "bundles" / item.replace(":", "-") / "RUN_COMPLETE.json")
                for item in completed
            },
        }
        _write_json_once(root / "CAMPAIGN_COMPLETE.json", final)
        return code
    except KeyboardInterrupt as exc:
        _write_abort(root, exc, completed)
        return 130
    except Exception as exc:
        _write_abort(root, exc, completed)
        return 2


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--python", default=sys.executable)
    common.add_argument("--vllm-source-root")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight", parents=[common])
    run = subparsers.add_parser("run", parents=[common])
    run.add_argument("--output-root", required=True)
    run.add_argument("--workload-manifest", required=True)
    run.add_argument("--arm-timeout-seconds", type=int, default=1800)
    run.add_argument("--comparison-timeout-seconds", type=int, default=300)
    return parser


def _raise_keyboard_interrupt(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "preflight":
            print(json.dumps(_preflight(args), indent=2, sort_keys=True))
            return
        if args.arm_timeout_seconds <= 0 or args.comparison_timeout_seconds <= 0:
            raise RuntimeError("timeouts must be positive")
        signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
        signal.signal(signal.SIGHUP, _raise_keyboard_interrupt)
        raise SystemExit(execute(args))
    except RuntimeError as exc:
        print(json.dumps({"status": "INVALID_ORCHESTRATION", "error": str(exc)}))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()

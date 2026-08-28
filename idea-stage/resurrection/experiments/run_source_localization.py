#!/usr/bin/env python3
"""Localize real serial-vs-batched OLMoE route differences.

The runner is a development execution-conformance diagnostic, not a capacity
experiment.  It validates two sealed raw captures, deterministically replays
their original batch schedule up to each selected event, and then forks the
same target pre-step logical state into three one-step arms:

* A: target request alone;
* B: target plus the original companions in the original row order;
* C: target at the same row with the original companion rows reversed.

The raw capture does not contain serializable KV tensors.  Consequently,
"canonical state" here means a deterministic replay that must match every
captured input token, next token, padding extent, and ordered route before the
event.  Any mismatch fails closed.  Arm C tests row-order/physical-layout
sensitivity; it does not test replacement-companion identity externality.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence


CAPTURE_FILES = (
    "routes.csv",
    "decode_batches.jsonl",
    "request_ledger.jsonl",
    "workload_manifest.json",
    "preregistration.json",
    "environment.json",
    "serial_audit.json",
)
SIGNALS = (
    "pre_router_hidden",
    "router_logits",
    "topk_margin",
    "selected_experts",
    "combined_expert_output",
    "next_token_logits",
)
ATOL = 1e-6
RTOL = 1e-5
NEAR_TIE_MARGIN = 1e-2
NEAR_TIE_CONCENTRATION_FRACTION = 0.75
GPU_POLL_SECONDS = 0.20
MIN_FREE_BYTES = 24 * 1024**3
PRIMARY_SOURCE_CLASSES = (
    "UPSTREAM_BATCH_CONTEXT_EFFECT",
    "ROUTER_KERNEL_SHAPE_EFFECT",
)


class ProtocolError(RuntimeError):
    """An input, replay, or arm violates the frozen diagnostic contract."""


def _repo_root() -> Path:
    return next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{path} must contain one JSON object")
    return dict(value)


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    identity_lines: dict[str, int] = {}
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ProtocolError(f"{path}:{line_number} is not an object")
                row = dict(value)
                rows.append(row)
                request_id = str(row.get("request_id", ""))
                if request_id:
                    if request_id in identity_lines:
                        raise ProtocolError(f"duplicate request ledger identity: {request_id}")
                    identity_lines[request_id] = line_number
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot parse {path}: {exc}") from exc
    return rows, identity_lines


def load_events_config(path: Path, profile: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = _read_json(path)
    if config.get("schema") != "moe-execution-conformance-events-v1":
        raise ProtocolError("unexpected event-selection schema")
    raw_events = config.get("events")
    profiles = config.get("profiles")
    if not isinstance(raw_events, list) or not isinstance(profiles, Mapping):
        raise ProtocolError("events and profiles must be present")
    by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_events:
        if not isinstance(raw, Mapping):
            raise ProtocolError("every event must be an object")
        event = dict(raw)
        event_id = str(event.get("event_id", ""))
        if not event_id or event_id in by_id:
            raise ProtocolError("event IDs must be non-empty and unique")
        by_id[event_id] = event
    selected_ids = profiles.get(profile)
    if not isinstance(selected_ids, list) or not selected_ids:
        raise ProtocolError(f"unknown or empty profile: {profile}")
    if len(selected_ids) != len(set(str(value) for value in selected_ids)):
        raise ProtocolError(f"profile {profile} repeats event IDs")
    try:
        selected = [by_id[str(value)] for value in selected_ids]
    except KeyError as exc:
        raise ProtocolError(f"profile references unknown event: {exc}") from exc
    regimes = {str(event.get("arrival_regime", "")) for event in selected}
    if regimes != {"steady", "bursty"}:
        raise ProtocolError("every execution profile must cover steady and bursty")
    return config, selected


def _load_producer() -> Any:
    path = _repo_root() / "docs/ideas/bcrd/experiments/capture_continuous_decode.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("resurrection_source_localization_producer", path)
    if spec is None or spec.loader is None:
        raise ProtocolError(f"cannot import capture producer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_capture(capture_dir: Path, expected: Mapping[str, Any], producer: Any) -> dict[str, Any]:
    if not capture_dir.is_dir():
        raise ProtocolError(f"capture directory does not exist: {capture_dir}")
    if _read_json(capture_dir / "RUN_STATUS.json") != {
        "required_sentinel": "CAPTURE_COMPLETE.json",
        "status": "COMPLETE",
    }:
        raise ProtocolError(f"capture is not closed COMPLETE: {capture_dir}")
    complete = _read_json(capture_dir / "CAPTURE_COMPLETE.json")
    if (
        complete.get("schema") != "bcrd-continuous-capture-complete-v1"
        or complete.get("status") != "CAPTURE_COMPLETE"
        or complete.get("run_class") != "development"
    ):
        raise ProtocolError("source is not a completed development capture")
    sentinel_hashes = complete.get("files")
    expected_hashes = expected.get("files_sha256")
    if not isinstance(sentinel_hashes, Mapping) or set(sentinel_hashes) != set(CAPTURE_FILES):
        raise ProtocolError("capture sentinel does not close the seven raw files")
    if not isinstance(expected_hashes, Mapping) or dict(sentinel_hashes) != dict(expected_hashes):
        raise ProtocolError("event selection and capture sentinel hashes differ")
    for name in CAPTURE_FILES:
        if _sha256_file(capture_dir / name) != str(sentinel_hashes[name]):
            raise ProtocolError(f"source capture hash mismatch: {name}")

    manifest = producer.load_workload_manifest(capture_dir / "workload_manifest.json")
    marker = manifest.get("route_capacity_envelope")
    if not isinstance(marker, Mapping):
        raise ProtocolError("capture workload lacks route-capacity marker")
    if (
        marker.get("episode_id") != expected.get("episode_id")
        or marker.get("arrival_regime") != expected.get("arrival_regime")
    ):
        raise ProtocolError("capture episode/regime differs from event selection")
    serial_audit = _read_json(capture_dir / "serial_audit.json")
    if complete.get("serial_audit") != serial_audit:
        raise ProtocolError("serial audit differs from the capture sentinel")
    batch_rows, _ = _read_jsonl(capture_dir / "decode_batches.jsonl")
    ledger_list, ledger_lines = _read_jsonl(capture_dir / "request_ledger.jsonl")
    ledger = {str(row.get("request_id", "")): row for row in ledger_list}
    if not ledger or "" in ledger or len(ledger) != len(ledger_list):
        raise ProtocolError("request ledger identities are not closed")

    route_index: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    with (capture_dir / "routes.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for line_number, raw in enumerate(reader, start=2):
            row = dict(raw)
            row["_source_line"] = line_number
            key = (
                str(row["request_id"]),
                int(row["decode_step"]),
                int(row["layer_id"]),
            )
            route_index.setdefault(key, []).append(row)
    return {
        "capture_dir": capture_dir,
        "complete": complete,
        "manifest": manifest,
        "environment": _read_json(capture_dir / "environment.json"),
        "serial_audit": serial_audit,
        "batch_rows": batch_rows,
        "ledger": ledger,
        "ledger_lines": ledger_lines,
        "route_index": route_index,
    }


def validate_event_alignment(event: Mapping[str, Any], capture: Mapping[str, Any]) -> dict[str, Any]:
    """Bind one selected event to batch, ledger, route, and serial-audit rows."""

    batch_index = int(event["batch_index"])
    batches = capture["batch_rows"]
    if batch_index < 0 or batch_index >= len(batches):
        raise ProtocolError("event batch index is outside the source capture")
    batch = batches[batch_index]
    if int(batch.get("batch_index", -1)) != batch_index:
        raise ProtocolError("decode-batch indices are not contiguous")
    source_rows = event.get("source_rows")
    if not isinstance(source_rows, Mapping):
        raise ProtocolError("event source rows are missing")
    if int(source_rows.get("decode_batches_jsonl_line", -1)) != batch_index + 1:
        raise ProtocolError("event decode-batch source line drifted")
    request_ids = [str(value) for value in batch.get("request_ids", [])]
    expected_batch = [str(value) for value in event.get("original_batch_request_ids", [])]
    if request_ids != expected_batch:
        raise ProtocolError("event original companion identities/order drifted")
    target = str(event["target_request_id"])
    target_row = int(event["target_row_index"])
    if target_row < 0 or target_row >= len(request_ids) or request_ids[target_row] != target:
        raise ProtocolError("target row identity does not close")
    decode_steps = [int(value) for value in batch.get("decode_steps", [])]
    step = int(event["decode_step"])
    if len(decode_steps) != len(request_ids) or decode_steps[target_row] != step:
        raise ProtocolError("target decode-step identity does not close")

    ledger = capture["ledger"]
    target_ledger = ledger.get(target)
    if not isinstance(target_ledger, Mapping):
        raise ProtocolError("target is absent from request ledger")
    if capture["ledger_lines"].get(target) != int(source_rows["request_ledger_jsonl_line"]):
        raise ProtocolError("target request-ledger source line drifted")
    steps = target_ledger.get("steps")
    if not isinstance(steps, list) or step >= len(steps):
        raise ProtocolError("target event step is absent from request ledger")
    ledger_step = steps[step]
    if (
        int(source_rows["request_ledger_step_index"]) != step
        or int(ledger_step.get("decode_step", -1)) != step
        or int(ledger_step.get("batch_index", -1)) != batch_index
    ):
        raise ProtocolError("target ledger step/batch alignment drifted")

    serial_examples = {
        (str(row.get("request_id", "")), int(row.get("decode_step", -1)), int(row.get("layer", -1))): row
        for row in capture["serial_audit"].get("difference_examples", [])
    }
    differences = event.get("recorded_difference_examples")
    if not isinstance(differences, list) or len(differences) != int(event["recorded_mismatch_count"]):
        raise ProtocolError("event recorded-difference count drifted")
    verified_layers: list[int] = []
    for difference in differences:
        layer = int(difference["layer"])
        recorded = serial_examples.get((target, step, layer))
        if recorded is None:
            raise ProtocolError("event is not a real recorded serial-audit difference")
        for key in ("batched_experts", "serial_experts"):
            if [int(value) for value in recorded[key]] != [int(value) for value in difference[key]]:
                raise ProtocolError(f"event {key} differs from serial audit")
        route_rows = capture["route_index"].get((target, step, layer), [])
        route_rows = sorted(route_rows, key=lambda row: int(row["topk_slot"]))
        if [int(row["topk_slot"]) for row in route_rows] != list(range(len(difference["batched_experts"]))):
            raise ProtocolError("target route top-k slots are incomplete")
        if [int(row["expert_id"]) for row in route_rows] != [int(value) for value in difference["batched_experts"]]:
            raise ProtocolError("target routes.csv experts differ from recorded example")
        expected_range = source_rows["routes_csv_rows_by_layer"].get(str(layer))
        observed_lines = [int(row["_source_line"]) for row in route_rows]
        if expected_range != [min(observed_lines), max(observed_lines)]:
            raise ProtocolError("target routes.csv physical source rows drifted")
        verified_layers.append(layer)
    return {
        "event_id": str(event["event_id"]),
        "batch_index": batch_index,
        "target_request_id": target,
        "target_decode_step": step,
        "original_batch_request_ids": request_ids,
        "recorded_difference_layers": verified_layers,
        "source_rows": dict(source_rows),
    }


def build_replay_plan(batch_rows: Sequence[Mapping[str, Any]], event_batch_index: int) -> list[Mapping[str, Any]]:
    if event_batch_index < 0 or event_batch_index >= len(batch_rows):
        raise ProtocolError("event batch is outside replay plan")
    for index, row in enumerate(batch_rows):
        if int(row.get("batch_index", -1)) != index:
            raise ProtocolError("source batch indices are not contiguous")
    plan = list(batch_rows[:event_batch_index])
    if any(int(row["batch_index"]) >= event_batch_index for row in plan):
        raise ProtocolError("replay plan reads the event or future batch")
    return plan


def deterministic_arm_orders(original: Sequence[str], target: str) -> dict[str, list[str]]:
    original = [str(value) for value in original]
    if len(original) < 3 or original.count(target) != 1 or len(set(original)) != len(original):
        raise ProtocolError("original batch must contain one target and unique companions")
    target_row = original.index(target)
    companions = [value for value in original if value != target]
    shuffled = list(original)
    for row, value in zip((index for index in range(len(original)) if index != target_row), reversed(companions)):
        shuffled[row] = value
    if shuffled == original:
        raise ProtocolError("deterministic companion shuffle did not change row order")
    return {"A_serial": [target], "B_original": original, "C_shuffled": shuffled}


def classify_facts(facts: Mapping[str, Any]) -> dict[str, Any]:
    """Convert measured comparisons into bounded, non-exclusive localization labels."""

    scope = {
        "companion_row_order_or_layout_effect": "TESTED_BY_DETERMINISTIC_SHUFFLE",
        "companion_identity_externality": "NOT_TESTED",
        "physical_shape_effect": "NOT_IDENTIFIED_BY_SHUFFLE_ONLY_C",
        "width_vs_companion_context": "UNRESOLVED_BY_A_B_C",
    }
    if not bool(facts.get("target_state_identical", False)):
        return {
            "status": "INVALID_TARGET_PRESTATE_DRIFT",
            "frozen_classification": [],
            "secondary_findings": [],
            "scope": scope,
        }
    if not bool(facts.get("original_arm_reproduced", False)):
        return {
            "status": "STOP_ORIGINAL_ARM_NOT_REPRODUCED",
            "frozen_classification": ["NOT_REPRODUCED"],
            "secondary_findings": [],
            "scope": scope,
        }
    if not bool(facts.get("within_arm_stable", False)):
        return {
            "status": "STOP_WITHIN_ARM_UNSTABLE",
            "frozen_classification": ["NONDETERMINISTIC_RUNTIME"],
            "secondary_findings": [],
            "scope": scope,
        }
    if not bool(facts.get("pairwise_repeat_consistent", False)):
        return {
            "status": "STOP_PAIRWISE_REPEAT_INCONSISTENT",
            "frozen_classification": ["NONDETERMINISTIC_RUNTIME"],
            "secondary_findings": [],
            "scope": scope,
        }
    if not bool(facts.get("ab_assignment_changed", False)):
        return {
            "status": "STOP_NO_REPRODUCED_ASSIGNMENT_DIVERGENCE",
            "frozen_classification": ["NOT_REPRODUCED"],
            "secondary_findings": [],
            "scope": scope,
        }
    labels: list[str] = []
    first_signal = facts.get("ab_first_divergence_signal")
    if first_signal == "pre_router_hidden":
        labels.append("UPSTREAM_BATCH_CONTEXT_EFFECT")
    elif first_signal == "router_logits":
        if not bool(
            facts.get("ab_router_candidate_pre_hidden_exact_digest_equal", False)
        ):
            return {
                "status": "STABLE_DIVERGENCE_INPUT_DELTA_VS_KERNEL_UNRESOLVED",
                "frozen_classification": ["UNRESOLVED_INPUT_DELTA_VS_KERNEL"],
                "secondary_findings": [],
                "scope": scope,
            }
        labels.append("ROUTER_KERNEL_SHAPE_EFFECT")
    else:
        return {
            "status": "STABLE_DIVERGENCE_SOURCE_UNRESOLVED",
            "frozen_classification": ["UNRESOLVED_FIRST_DIVERGENCE"],
            "secondary_findings": [
                f"FIRST_DIVERGENCE_{str(first_signal or 'UNKNOWN').upper()}"
            ],
            "scope": scope,
        }
    if bool(facts.get("ab_near_tie_concentrated_crossing", False)):
        labels.append("NEAR_TIE_AMPLIFICATION")
    secondary: list[str] = []
    if bool(facts.get("bc_assignment_changed", False)):
        secondary.append("COMPANION_ROW_ORDER_OR_LAYOUT_EFFECT")
    else:
        secondary.append("WIDTH_VS_COMPANION_CONTEXT_UNRESOLVED")
    return {
        "status": "LOCALIZED_DEVELOPMENT_SIGNAL",
        "frozen_classification": labels,
        "secondary_findings": secondary,
        "scope": scope,
    }


def _reset_rng(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _tensor_bytes(tensor: Any) -> bytes:
    value = tensor.detach().contiguous().cpu()
    return value.view(-1).view(__import__("torch").uint8).numpy().tobytes()


def _tensor_digest(tensor: Any) -> str:
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape)).encode())
    digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def _clone_cache(cache: Any, torch: Any) -> Any:
    if not hasattr(cache, "to_legacy_cache") or not hasattr(type(cache), "from_legacy_cache"):
        raise ProtocolError(f"unsupported cache type for isolated forks: {type(cache).__name__}")
    legacy = cache.to_legacy_cache()
    cloned = tuple(
        (key.detach().clone(), value.detach().clone()) for key, value in legacy
    )
    return type(cache).from_legacy_cache(cloned)


def clone_state(state: Any, torch: Any) -> SimpleNamespace:
    return SimpleNamespace(
        spec=state.spec,
        cache=_clone_cache(state.cache, torch),
        attention_mask=state.attention_mask.detach().clone(),
        next_token=state.next_token.detach().clone(),
        prompt_length=int(state.prompt_length),
        decode_step=int(state.decode_step),
    )


def state_fingerprint(state: Any) -> str:
    digest = hashlib.sha256()
    digest.update(str(state.spec.request_id).encode())
    digest.update(str(int(state.decode_step)).encode())
    digest.update(_tensor_digest(state.attention_mask).encode())
    digest.update(_tensor_digest(state.next_token).encode())
    for key, value in state.cache.to_legacy_cache():
        digest.update(_tensor_digest(key).encode())
        digest.update(_tensor_digest(value).encode())
    return digest.hexdigest()


def _query_gpu_processes() -> tuple[tuple[str, str, str], ...]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ProtocolError(f"cannot query GPU processes: {result.stderr.strip()}")
    rows: list[tuple[str, str, str]] = []
    for line in result.stdout.splitlines():
        text = line.strip()
        if not text or text.upper() in {"N/A", "[N/A]"}:
            continue
        fields = tuple(value.strip() for value in text.split(",", 2))
        if len(fields) != 3 or not fields[1].isdigit():
            raise ProtocolError(f"unexpected nvidia-smi process row: {text!r}")
        rows.append(fields)
    return tuple(sorted(set(rows)))


class GpuIsolationMonitor:
    def __init__(self, expected: tuple[tuple[str, str, str], ...]) -> None:
        if len(expected) != 1:
            raise ProtocolError("model load must create exactly one isolated GPU process")
        self.expected = expected
        self.samples = 0
        self.violations: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def check(self, label: str) -> None:
        try:
            observed = _query_gpu_processes()
        except Exception as exc:
            self.violations.append({"label": label, "error": str(exc)})
            return
        self.samples += 1
        if observed != self.expected:
            self.violations.append(
                {
                    "label": label,
                    "expected": [list(row) for row in self.expected],
                    "observed": [list(row) for row in observed],
                }
            )

    def require_clean(self) -> None:
        if self.violations:
            raise ProtocolError(f"GPU isolation failed: {self.violations[:3]}")

    def start(self) -> None:
        self.check("monitor_start")

        def poll() -> None:
            while not self._stop.wait(GPU_POLL_SECONDS):
                self.check("periodic")

        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.check("monitor_stop")

    def summary(self) -> dict[str, Any]:
        return {
            "status": "PASS_SAMPLED_PROCESS_ISOLATION" if not self.violations else "FAIL",
            "poll_interval_seconds": GPU_POLL_SECONDS,
            "samples": self.samples,
            "expected": [list(row) for row in self.expected],
            "violations": self.violations,
        }


def _git_state() -> dict[str, Any]:
    root = _repo_root()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.splitlines()
    return {"head": head, "dirty": bool(status), "status": status}


def _runtime_environment(torch: Any, transformers: Any) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": sys.version,
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "cuda_version": torch.version.cuda,
        "gpu_count": int(torch.cuda.device_count()),
        "gpus": [
            {
                "index": 0,
                "name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "total_memory_bytes": int(properties.total_memory),
            }
        ],
    }


def _validate_runtime(captured: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    captured_gpus = captured.get("gpus")
    current_gpus = current.get("gpus")
    if not isinstance(captured_gpus, list) or not isinstance(current_gpus, list):
        raise ProtocolError("GPU environment identity is missing")
    if len(captured_gpus) != 1 or len(current_gpus) != 1:
        raise ProtocolError("diagnostic requires exactly one captured and current GPU")
    checks = {
        "python_version": (str(captured.get("python", "")).split()[0], str(current["python"]).split()[0]),
        "torch": (captured.get("torch"), current.get("torch")),
        "transformers": (captured.get("transformers"), current.get("transformers")),
        "cuda_version": (captured.get("cuda_version"), current.get("cuda_version")),
        "gpu_name": (captured_gpus[0].get("name"), current_gpus[0].get("name")),
        "gpu_capability": (
            list(captured_gpus[0].get("capability", [])),
            list(current_gpus[0].get("capability", [])),
        ),
    }
    drift = {
        key: {"captured": left, "current": right}
        for key, (left, right) in checks.items()
        if left != right
    }
    if drift:
        raise ProtocolError(f"runtime differs from the capture: {drift}")
    return {"status": "MATCH_CAPTURED_RUNTIME", "checked_fields": sorted(checks)}


def load_exact_model(model_spec: Mapping[str, Any]) -> tuple[Any, Any, Any, Any, float]:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise ProtocolError("PyTorch and Transformers are required") from exc
    if not torch.cuda.is_available() or int(torch.cuda.device_count()) != 1:
        raise ProtocolError("source localization requires exactly one visible CUDA GPU")
    torch.cuda.set_device(0)
    free_bytes, _ = torch.cuda.mem_get_info(0)
    if int(free_bytes) < MIN_FREE_BYTES:
        raise ProtocolError(f"need 24 GiB free before model load; found {free_bytes / 1024**3:.2f}")
    shared = _repo_root() / "experiments/shared"
    sys.path.insert(0, str(shared))
    from modeling import load_model, load_tokenizer

    tokenizer = load_tokenizer(
        str(model_spec["id"]),
        local_files_only=True,
        revision=str(model_spec["tokenizer_revision"]),
    )
    model, load_seconds = load_model(
        str(model_spec["id"]),
        dtype_name=str(model_spec["dtype"]),
        local_files_only=True,
        revision=str(model_spec["revision"]),
    )
    model.eval()
    if not str(model.device).startswith("cuda") or next(model.parameters()).dtype != torch.bfloat16:
        raise ProtocolError("loaded model is not CUDA BF16")
    return torch, transformers, tokenizer, model, float(load_seconds)


def _tensor_record(tensor: Any) -> dict[str, Any]:
    value = tensor.detach().float().cpu().contiguous()
    return {
        "dtype_before_float32_copy": str(tensor.dtype),
        "shape": list(value.shape),
        "sha256_float32": _tensor_digest(value),
        "values": [float(item) for item in value.reshape(-1).tolist()],
    }


class TargetSignalHooks:
    """Capture only the six authorized target-token signals."""

    def __init__(self, model: Any, target_row: int, expected_rows: int) -> None:
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None:
            raise ProtocolError("model does not expose OLMoE decoder layers")
        if expected_rows < 1 or target_row < 0 or target_row >= expected_rows:
            raise ProtocolError("target hook row is outside the measured batch")
        self.layers = list(layers)
        self.target_row = target_row
        self.expected_rows = expected_rows
        self.hidden_size = int(getattr(getattr(model, "config", None), "hidden_size", 0))
        self.num_experts = int(getattr(getattr(model, "config", None), "num_experts", 0))
        if self.hidden_size <= 0 or self.num_experts <= 0:
            raise ProtocolError("model config lacks positive hidden_size/num_experts")
        self.handles: list[Any] = []
        self.pre_router_hidden: dict[str, Any] = {}
        self.router_logits: dict[str, Any] = {}
        self.combined_expert_output: dict[str, Any] = {}

    def __enter__(self) -> "TargetSignalHooks":
        for layer_index, layer in enumerate(self.layers):
            mlp = getattr(layer, "mlp", None)
            gate = getattr(mlp, "gate", None)
            if mlp is None or gate is None:
                raise ProtocolError("decoder layer lacks OLMoE mlp.gate")

            def gate_pre(_module: Any, inputs: tuple[Any, ...], index: int = layer_index) -> None:
                if not inputs:
                    raise ProtocolError(f"layer {index} gate pre-hook received no input")
                hidden = inputs[0]
                if (
                    getattr(hidden, "ndim", None) != 2
                    or int(hidden.shape[0]) != self.expected_rows
                    or int(hidden.shape[1]) != self.hidden_size
                ):
                    raise ProtocolError(
                        f"layer {index} gate input shape is not "
                        f"[{self.expected_rows}, {self.hidden_size}]"
                    )
                self.pre_router_hidden[str(index)] = hidden[self.target_row].detach().clone()

            def gate_post(_module: Any, _inputs: tuple[Any, ...], output: Any, index: int = layer_index) -> None:
                if (
                    getattr(output, "ndim", None) != 2
                    or int(output.shape[0]) != self.expected_rows
                    or int(output.shape[1]) != self.num_experts
                ):
                    raise ProtocolError(
                        f"layer {index} router-logit shape is not "
                        f"[{self.expected_rows}, {self.num_experts}]"
                    )
                self.router_logits[str(index)] = output[self.target_row].detach().clone()

            def mlp_post(_module: Any, _inputs: tuple[Any, ...], output: Any, index: int = layer_index) -> None:
                combined = output[0] if isinstance(output, tuple) else output
                if (
                    getattr(combined, "ndim", None) != 3
                    or int(combined.shape[0]) != self.expected_rows
                    or int(combined.shape[1]) != 1
                    or int(combined.shape[2]) != self.hidden_size
                ):
                    raise ProtocolError(
                        f"layer {index} combined expert output shape is not "
                        f"[{self.expected_rows}, 1, {self.hidden_size}]"
                    )
                self.combined_expert_output[str(index)] = combined[self.target_row, -1].detach().clone()

            self.handles.append(gate.register_forward_pre_hook(gate_pre))
            self.handles.append(gate.register_forward_hook(gate_post))
            self.handles.append(mlp.register_forward_hook(mlp_post))
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def finish(self, route_batches: Sequence[Mapping[str, Any]], next_logits: Any) -> dict[str, Any]:
        expected_layers = {str(index) for index in range(len(self.layers))}
        for captured in (self.pre_router_hidden, self.router_logits, self.combined_expert_output):
            if set(captured) != expected_layers:
                raise ProtocolError("target hook did not close every decoder layer")
        margins: dict[str, float] = {}
        experts: dict[str, list[int]] = {}
        for batch in route_batches:
            layer = str(int(batch["layer"]))
            selected_tensor = batch["selected_experts"]
            if (
                getattr(selected_tensor, "ndim", None) != 2
                or int(selected_tensor.shape[0]) != self.expected_rows
                or int(selected_tensor.shape[1]) < 1
            ):
                raise ProtocolError(f"layer {layer} selected-expert shape violates the hook contract")
            selected = [int(value) for value in selected_tensor[self.target_row].tolist()]
            logits = self.router_logits[layer].detach().float()
            if logits.ndim != 1 or int(logits.shape[0]) != self.num_experts:
                raise ProtocolError(f"layer {layer} target router logits are not one expert vector")
            if len(selected) >= self.num_experts:
                raise ProtocolError(f"layer {layer} top-k leaves no boundary expert")
            ordered = __import__("torch").topk(logits, k=len(selected) + 1).values
            margins[layer] = float((ordered[len(selected) - 1] - ordered[len(selected)]).item())
            experts[layer] = selected
        if getattr(next_logits, "ndim", None) != 1:
            raise ProtocolError("target next-token logits are not one vocabulary vector")
        return {
            "pre_router_hidden": {key: _tensor_record(value) for key, value in self.pre_router_hidden.items()},
            "router_logits": {key: _tensor_record(value) for key, value in self.router_logits.items()},
            "topk_margin": margins,
            "selected_experts": experts,
            "combined_expert_output": {
                key: _tensor_record(value) for key, value in self.combined_expert_output.items()
            },
            "next_token_logits": _tensor_record(next_logits),
        }


def _prefill_states(
    torch: Any,
    producer: Any,
    model: Any,
    prepared: Sequence[Any],
    ledger: Mapping[str, Mapping[str, Any]],
) -> dict[str, SimpleNamespace]:
    states: dict[str, SimpleNamespace] = {}
    for spec in prepared:
        with torch.inference_mode():
            output, _ = producer._timed_call(
                model,
                "source_localization_prefill",
                1,
                None,
                input_ids=spec.input_ids,
                attention_mask=spec.attention_mask,
                use_cache=True,
                output_router_logits=False,
                return_dict=True,
            )
        cache = getattr(output, "past_key_values", None)
        logits = getattr(output, "logits", None)
        prompt_length = int(spec.input_ids.shape[1])
        if cache is None or logits is None or producer._cache_length(cache) != prompt_length:
            raise ProtocolError(f"prefill did not close cache/logits for {spec.request_id}")
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        row = ledger.get(spec.request_id)
        steps = row.get("steps") if isinstance(row, Mapping) else None
        if not isinstance(steps, list) or not steps:
            raise ProtocolError(f"ledger has no decode trajectory for {spec.request_id}")
        if int(next_token.item()) != int(steps[0]["input_token_id"]):
            raise ProtocolError(f"prefill token does not reproduce capture for {spec.request_id}")
        states[spec.request_id] = SimpleNamespace(
            spec=spec,
            cache=cache,
            attention_mask=spec.attention_mask,
            next_token=next_token,
            prompt_length=prompt_length,
            decode_step=0,
        )
    return states


def _call_states(
    torch: Any,
    producer: Any,
    model: Any,
    states: Sequence[Any],
    *,
    hooks: TargetSignalHooks | None = None,
) -> dict[str, Any]:
    (
        input_ids,
        attention_mask,
        position_ids,
        cache,
        prior_lengths,
        prior_max,
    ) = producer._pad_decode_inputs(states)
    with torch.inference_mode():
        output, _ = producer._timed_call(
            model,
            "source_localization_decode",
            len(states),
            None,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=torch.tensor([prior_max], dtype=torch.long, device=input_ids.device),
            past_key_values=cache,
            use_cache=True,
            output_router_logits=True,
            return_dict=True,
        )
    logits = getattr(output, "logits", None)
    output_cache = getattr(output, "past_key_values", None)
    if logits is None or output_cache is None:
        raise ProtocolError("decode call returned no logits/cache")
    route_batches = producer._native_route_batches(
        output, expected_rows=len(states), config=getattr(model, "config")
    )
    predicted = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    return {
        "input_ids": input_ids,
        "prior_lengths": [int(value) for value in prior_lengths],
        "prior_max": int(prior_max),
        "output_cache": output_cache,
        "logits": logits,
        "predicted": predicted,
        "route_batches": route_batches,
        "signals": hooks.finish(route_batches, logits[hooks.target_row, -1]) if hooks else None,
    }


def _validate_call_against_source(
    call: Mapping[str, Any],
    states: Sequence[Any],
    source_batch: Mapping[str, Any],
    ledger: Mapping[str, Mapping[str, Any]],
    *,
    fail_closed: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    ids = [state.spec.request_id for state in states]
    if ids != [str(value) for value in source_batch.get("request_ids", [])]:
        errors.append("batch identities/order differ from source")
    if [int(state.decode_step) for state in states] != [
        int(value) for value in source_batch.get("decode_steps", [])
    ]:
        errors.append("decode steps differ from source")
    if call["prior_lengths"] != [int(value) for value in source_batch.get("prior_cache_lengths", [])]:
        errors.append("logical KV lengths differ from source")
    if [call["prior_max"] - value for value in call["prior_lengths"]] != [
        int(value) for value in source_batch.get("left_padding", [])
    ]:
        errors.append("physical padding differs from source")
    for row_index, state in enumerate(states):
        step = int(state.decode_step)
        source_step = ledger[state.spec.request_id]["steps"][step]
        if int(call["input_ids"][row_index].item()) != int(source_step["input_token_id"]):
            errors.append(f"input token differs for row {row_index}")
        if int(call["predicted"][row_index].item()) != int(source_step["predicted_next_token_id"]):
            errors.append(f"next token differs for row {row_index}")
        observed = [
            {
                "layer": int(batch["layer"]),
                "experts": [int(value) for value in batch["selected_experts"][row_index].tolist()],
            }
            for batch in call["route_batches"]
        ]
        expected = [
            {
                "layer": int(layer["layer"]),
                "experts": [int(value) for value in layer["experts"]],
            }
            for layer in source_step["route_signature"]
        ]
        if observed != expected:
            errors.append(
                f"ordered route differs for {state.spec.request_id}/step {step}"
            )
    result = {"passed": not errors, "errors": errors}
    if errors and fail_closed:
        raise ProtocolError("source replay validation failed: " + "; ".join(errors))
    return result


def _advance_source_batch(
    torch: Any,
    producer: Any,
    model: Any,
    state_by_id: Mapping[str, Any],
    source_batch: Mapping[str, Any],
    ledger: Mapping[str, Mapping[str, Any]],
) -> None:
    states = [state_by_id[str(value)] for value in source_batch["request_ids"]]
    call = _call_states(torch, producer, model, states)
    _validate_call_against_source(call, states, source_batch, ledger)
    split = producer.split_left_padded_cache(
        call["output_cache"],
        prior_lengths=call["prior_lengths"],
        prior_max_length=call["prior_max"],
    )
    for row_index, state in enumerate(states):
        state.cache = split[row_index]
        state.attention_mask = torch.cat(
            (state.attention_mask, state.attention_mask.new_ones((1, 1))), dim=1
        )
        state.next_token = call["predicted"][row_index : row_index + 1]
        state.decode_step += 1


def _run_arm_once(
    torch: Any,
    producer: Any,
    model: Any,
    canonical: Mapping[str, Any],
    order: Sequence[str],
    target: str,
    source_batch: Mapping[str, Any],
    ledger: Mapping[str, Mapping[str, Any]],
    *,
    capture_signals: bool,
    validate_original: bool,
) -> dict[str, Any] | None:
    states = [clone_state(canonical[request_id], torch) for request_id in order]
    target_row = list(order).index(target)
    target_before = state_fingerprint(states[target_row])
    for state in states:
        source_step = ledger[state.spec.request_id]["steps"][state.decode_step]
        if int(state.next_token.item()) != int(source_step["input_token_id"]):
            raise ProtocolError("forked arm input token is not capture-bound")
    if capture_signals:
        with TargetSignalHooks(model, target_row, len(states)) as hooks:
            call = _call_states(torch, producer, model, states, hooks=hooks)
    else:
        call = _call_states(torch, producer, model, states)
    source_reproduction = None
    if validate_original:
        source_reproduction = _validate_call_against_source(
            call, states, source_batch, ledger, fail_closed=True
        )
    if not capture_signals:
        return None
    if set(call["signals"]) != set(SIGNALS):
        raise ProtocolError("measured arm emitted signals outside the six-field contract")
    return {
        "request_order": list(order),
        "target_row_index": target_row,
        "target_prestate_fingerprint": target_before,
        "batch_width": len(order),
        "logical_kv_lengths": call["prior_lengths"],
        "left_padding": [call["prior_max"] - value for value in call["prior_lengths"]],
        "input_token_ids": [int(call["input_ids"][index].item()) for index in range(len(states))],
        "argmax_next_token_ids": [
            int(call["predicted"][index].item()) for index in range(len(states))
        ],
        "source_reproduction": source_reproduction,
        "signals": call["signals"],
    }


def _allclose_values(left: Sequence[float], right: Sequence[float]) -> tuple[bool, float]:
    if len(left) != len(right):
        raise ProtocolError("signal vector widths differ")
    maximum = 0.0
    matched = True
    for a, b in zip(left, right):
        delta = abs(float(a) - float(b))
        maximum = max(maximum, delta)
        if delta > ATOL + RTOL * abs(float(b)):
            matched = False
    return matched, maximum


def _tensor_records_exact_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Require an exact serialized tensor identity, not numerical allclose."""

    required = ("dtype_before_float32_copy", "shape", "sha256_float32")
    if any(key not in left or key not in right for key in required):
        raise ProtocolError("tensor record lacks fields required for exact digest comparison")
    return all(left[key] == right[key] for key in required)


def compare_runs(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_signals = left["signals"]
    right_signals = right["signals"]
    layers = sorted(left_signals["router_logits"], key=int)
    result: dict[str, Any] = {
        "first_pre_router_hidden_layer": None,
        "first_router_logit_layer": None,
        "first_topk_margin_layer": None,
        "first_selected_experts_layer": None,
        "first_combined_expert_output_layer": None,
        "assignment_difference_layers": [],
        "near_tie_boundary_crossing_layers": [],
        "near_tie_boundary_crossing_fraction": 0.0,
        "near_tie_concentrated_crossing": False,
        "assignment_margin_pairs": {},
        "first_cross_signal_divergence": None,
        "pre_router_hidden_exact_digest_equal_by_layer": {},
        "router_candidate_pre_hidden_exact_digest_equal": None,
        "max_abs_delta": {},
    }
    for layer in layers:
        result["pre_router_hidden_exact_digest_equal_by_layer"][layer] = (
            _tensor_records_exact_equal(
                left_signals["pre_router_hidden"][layer],
                right_signals["pre_router_hidden"][layer],
            )
        )
    tensor_first_keys = {
        "pre_router_hidden": "first_pre_router_hidden_layer",
        "router_logits": "first_router_logit_layer",
        "combined_expert_output": "first_combined_expert_output_layer",
    }
    for signal, first_key in tensor_first_keys.items():
        deltas: dict[str, float] = {}
        first = None
        for layer in layers:
            matched, maximum = _allclose_values(
                left_signals[signal][layer]["values"], right_signals[signal][layer]["values"]
            )
            deltas[layer] = maximum
            if not matched and first is None:
                first = int(layer)
        result["max_abs_delta"][signal] = deltas
        result[first_key] = first
    topk_deltas: dict[str, float] = {}
    for layer in layers:
        matched, maximum = _allclose_values(
            [float(left_signals["topk_margin"][layer])],
            [float(right_signals["topk_margin"][layer])],
        )
        topk_deltas[layer] = maximum
        if not matched and result["first_topk_margin_layer"] is None:
            result["first_topk_margin_layer"] = int(layer)
    result["max_abs_delta"]["topk_margin"] = topk_deltas
    for layer in layers:
        left_experts = sorted(int(value) for value in left_signals["selected_experts"][layer])
        right_experts = sorted(int(value) for value in right_signals["selected_experts"][layer])
        if left_experts != right_experts:
            layer_number = int(layer)
            if result["first_selected_experts_layer"] is None:
                result["first_selected_experts_layer"] = layer_number
            result["assignment_difference_layers"].append(layer_number)
            left_margin = abs(float(left_signals["topk_margin"][layer]))
            right_margin = abs(float(right_signals["topk_margin"][layer]))
            result["assignment_margin_pairs"][layer] = {
                "left_abs_margin": left_margin,
                "right_abs_margin": right_margin,
            }
            # An assignment change is the boundary crossing.  It is called a
            # near-tie crossing only when both executions had a small local
            # top-k boundary, rather than when either side happened to be small.
            if max(left_margin, right_margin) <= NEAR_TIE_MARGIN:
                result["near_tie_boundary_crossing_layers"].append(layer_number)
    changed_count = len(result["assignment_difference_layers"])
    if changed_count:
        result["near_tie_boundary_crossing_fraction"] = (
            len(result["near_tie_boundary_crossing_layers"]) / changed_count
        )
        result["near_tie_concentrated_crossing"] = bool(
            result["near_tie_boundary_crossing_fraction"]
            >= NEAR_TIE_CONCENTRATION_FRACTION
        )
    next_match, next_delta = _allclose_values(
        left_signals["next_token_logits"]["values"],
        right_signals["next_token_logits"]["values"],
    )
    result["next_token_logits_allclose"] = next_match
    result["next_token_logits_max_abs_delta"] = next_delta
    result["assignment_changed"] = bool(result["assignment_difference_layers"])

    # Order across signals, not within each signal independently.  For a layer,
    # pre-router hidden precedes router logits, the top-k boundary/selection,
    # and combined expert output.  Next-token logits are post-decoder.
    divergence_keys = (
        ("pre_router_hidden", "first_pre_router_hidden_layer"),
        ("router_logits", "first_router_logit_layer"),
        ("topk_margin", "first_topk_margin_layer"),
        ("selected_experts", "first_selected_experts_layer"),
        ("combined_expert_output", "first_combined_expert_output_layer"),
    )
    for layer in (int(value) for value in layers):
        for signal, key in divergence_keys:
            if result[key] == layer:
                result["first_cross_signal_divergence"] = {
                    "signal": signal,
                    "layer": layer,
                }
                break
        if result["first_cross_signal_divergence"] is not None:
            break
    if result["first_cross_signal_divergence"] is None and not next_match:
        result["first_cross_signal_divergence"] = {
            "signal": "next_token_logits",
            "layer": None,
        }
    first_divergence = result["first_cross_signal_divergence"] or {}
    if first_divergence.get("signal") == "router_logits":
        candidate_layer = str(int(first_divergence["layer"]))
        result["router_candidate_pre_hidden_exact_digest_equal"] = bool(
            result["pre_router_hidden_exact_digest_equal_by_layer"][candidate_layer]
        )
    result["all_six_signals_stable"] = bool(
        not result["assignment_changed"]
        and result["first_pre_router_hidden_layer"] is None
        and result["first_router_logit_layer"] is None
        and result["first_topk_margin_layer"] is None
        and result["first_combined_expert_output_layer"] is None
        and next_match
    )
    return result


def _comparison_signature(comparison: Mapping[str, Any]) -> dict[str, Any]:
    """Fields that must agree before a pairwise localization is repeat-qualified."""

    return {
        "first_cross_signal_divergence": comparison["first_cross_signal_divergence"],
        "assignment_difference_layers": comparison["assignment_difference_layers"],
        "near_tie_boundary_crossing_layers": comparison[
            "near_tie_boundary_crossing_layers"
        ],
        "router_candidate_pre_hidden_exact_digest_equal": comparison[
            "router_candidate_pre_hidden_exact_digest_equal"
        ],
    }


def _compare_repeat_pairs(
    left_runs: Sequence[Mapping[str, Any]],
    right_runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    left_by_repeat = {int(value["repeat"]): value for value in left_runs}
    right_by_repeat = {int(value["repeat"]): value for value in right_runs}
    if len(left_by_repeat) != len(left_runs) or len(right_by_repeat) != len(right_runs):
        raise ProtocolError("an arm contains duplicate repeat identities")
    if not left_by_repeat or set(left_by_repeat) != set(right_by_repeat):
        raise ProtocolError("pairwise arms do not contain the same repeat identities")
    comparisons = [
        {"repeat": repeat, **compare_runs(left_by_repeat[repeat], right_by_repeat[repeat])}
        for repeat in sorted(left_by_repeat)
    ]
    signatures = [_comparison_signature(value) for value in comparisons]
    repeat_consistent = all(value == signatures[0] for value in signatures[1:])
    return {
        "repeat_consistent": repeat_consistent,
        "all_repeats_assignment_changed": all(
            bool(value["assignment_changed"]) for value in comparisons
        ),
        "all_repeats_assignment_unchanged": all(
            not bool(value["assignment_changed"]) for value in comparisons
        ),
        "all_repeats_near_tie_concentrated_crossing": all(
            bool(value["near_tie_concentrated_crossing"]) for value in comparisons
        ),
        "consensus": comparisons[0] if repeat_consistent else None,
        "per_repeat": comparisons,
    }


def _summarize_event_runs(runs: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    stability: dict[str, Any] = {}
    within_stable = True
    for arm, arm_runs in runs.items():
        comparisons = [compare_runs(arm_runs[0], value) for value in arm_runs[1:]]
        passed = all(value["all_six_signals_stable"] for value in comparisons)
        within_stable = within_stable and passed
        stability[arm] = {"passed": passed, "repeat0_vs_later": comparisons}
    ab = _compare_repeat_pairs(runs["A_serial"], runs["B_original"])
    bc = _compare_repeat_pairs(runs["B_original"], runs["C_shuffled"])
    ab_consensus = ab["consensus"] or {}
    target_fingerprints = {
        value["target_prestate_fingerprint"]
        for arm_runs in runs.values()
        for value in arm_runs
    }
    facts = {
        "target_state_identical": len(target_fingerprints) == 1,
        "original_arm_reproduced": all(
            bool(value.get("source_reproduction", {}).get("passed", False))
            for value in runs["B_original"]
        ),
        "within_arm_stable": within_stable,
        "pairwise_repeat_consistent": bool(
            ab["repeat_consistent"] and bc["repeat_consistent"]
        ),
        "ab_assignment_changed": bool(
            ab["repeat_consistent"] and ab["all_repeats_assignment_changed"]
        ),
        "bc_assignment_changed": bool(
            bc["repeat_consistent"] and bc["all_repeats_assignment_changed"]
        ),
        "ab_first_pre_router_hidden_layer": ab_consensus.get(
            "first_pre_router_hidden_layer"
        ),
        "ab_first_router_logit_layer": ab_consensus.get("first_router_logit_layer"),
        "ab_first_divergence_signal": (
            ab_consensus.get("first_cross_signal_divergence") or {}
        ).get("signal"),
        "ab_first_divergence_layer": (
            ab_consensus.get("first_cross_signal_divergence") or {}
        ).get("layer"),
        "ab_router_candidate_pre_hidden_exact_digest_equal": ab_consensus.get(
            "router_candidate_pre_hidden_exact_digest_equal"
        ),
        "ab_near_tie_concentrated_crossing": bool(
            ab["repeat_consistent"]
            and ab["all_repeats_near_tie_concentrated_crossing"]
        ),
    }
    return {
        "within_arm_repeat_stability": stability,
        "A_vs_B": ab,
        "B_vs_C": bc,
        "facts": facts,
        "classification": classify_facts(facts),
    }


def _measured_arm_order(repeat: int) -> list[str]:
    orders = (
        ["A_serial", "B_original", "C_shuffled"],
        ["C_shuffled", "B_original", "A_serial"],
        ["B_original", "A_serial", "C_shuffled"],
    )
    return list(orders[repeat % len(orders)])


def _run_event(
    torch: Any,
    producer: Any,
    model: Any,
    capture: Mapping[str, Any],
    canonical: Mapping[str, Any],
    event: Mapping[str, Any],
    alignment: Mapping[str, Any],
    repeats: int,
    monitor: GpuIsolationMonitor,
) -> dict[str, Any]:
    target = str(event["target_request_id"])
    source_batch = capture["batch_rows"][int(event["batch_index"])]
    orders = deterministic_arm_orders(source_batch["request_ids"], target)
    seed = int(capture["manifest"]["seed"])
    canonical_target_fingerprint = state_fingerprint(canonical[target])

    def invoke(arm: str, measured: bool) -> dict[str, Any] | None:
        monitor.check(f"before_{event['event_id']}_{arm}")
        monitor.require_clean()
        _reset_rng(torch, seed)
        value = _run_arm_once(
            torch,
            producer,
            model,
            canonical,
            orders[arm],
            target,
            source_batch,
            capture["ledger"],
            capture_signals=measured,
            validate_original=arm == "B_original",
        )
        monitor.check(f"after_{event['event_id']}_{arm}")
        monitor.require_clean()
        if value is not None and value["target_prestate_fingerprint"] != canonical_target_fingerprint:
            raise ProtocolError("arm target state differs from canonical pre-step state")
        return value

    for arm in ("A_serial", "B_original", "C_shuffled"):
        invoke(arm, False)
    measured: dict[str, list[dict[str, Any]]] = {key: [] for key in orders}
    arm_orders: list[list[str]] = []
    started = time.monotonic()
    for repeat in range(repeats):
        order = _measured_arm_order(repeat)
        arm_orders.append(order)
        for arm in order:
            value = invoke(arm, True)
            if value is None:
                raise ProtocolError("measured arm returned no signal payload")
            measured[arm].append({"repeat": repeat, **value})
    elapsed = time.monotonic() - started
    summary = _summarize_event_runs(measured)
    return {
        "event_id": str(event["event_id"]),
        "arrival_regime": str(event["arrival_regime"]),
        "episode_id": str(event["episode_id"]),
        "target_request_id": target,
        "decode_step": int(event["decode_step"]),
        "batch_index": int(event["batch_index"]),
        "source_alignment": dict(alignment),
        "canonical_replay": {
            "target_prestate_fingerprint": canonical_target_fingerprint,
            "source_batches_replayed_before_event": int(event["batch_index"]),
            "maximum_source_batch_index_read_before_fork": int(event["batch_index"]) - 1,
            "event_batch_metadata_read_for_current_step": [
                "request_ids",
                "decode_steps",
                "prior_cache_lengths",
                "left_padding",
            ],
            "event_step_ledger_metadata_read_for_input_and_B_validation": True,
            "event_or_future_tensors_used_to_construct_canonical_prestate": False,
            "future_tensor_or_outcome_used_to_construct_or_evaluate_this_fork": False,
            "later_selected_event_identity_metadata_may_be_prevalidated": True,
        },
        "arms": {
            "A_serial": "target only",
            "B_original": "target plus original companions in original row order",
            "C_shuffled": "same target row; original companion rows reversed",
            "orders": orders,
            "warmup_arms_discarded": ["A_serial", "B_original", "C_shuffled"],
            "measured_order_by_repeat": arm_orders,
        },
        "repeats": repeats,
        "elapsed_seconds_excluding_prefill_and_canonical_replay": elapsed,
        "summary": summary,
        "traces": measured,
    }


def _run_capture_events(
    torch: Any,
    producer: Any,
    tokenizer: Any,
    model: Any,
    capture: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    repeats: int,
    monitor: GpuIsolationMonitor,
) -> list[dict[str, Any]]:
    events = sorted(events, key=lambda value: int(value["batch_index"]))
    batch_indices = [int(value["batch_index"]) for value in events]
    if len(batch_indices) != len(set(batch_indices)):
        raise ProtocolError("selected events in one capture must use distinct batches")
    alignments = {
        str(event["event_id"]): validate_event_alignment(event, capture) for event in events
    }
    for event in events:
        build_replay_plan(capture["batch_rows"], int(event["batch_index"]))

    prepared = producer._prepare_requests(capture["manifest"], tokenizer, model.device)
    results: list[dict[str, Any]] = []
    for event in events:
        event_batch = int(event["batch_index"])
        # Reconstruct each event independently from the sealed manifest.  A
        # measured A/B/C call may initialize CUDA allocator/kernel state, but it
        # must never mutate or otherwise condition the canonical prefix used by
        # a later selected event.
        _reset_rng(torch, int(capture["manifest"]["seed"]))
        canonical = _prefill_states(
            torch, producer, model, prepared, capture["ledger"]
        )
        cursor = 0
        while cursor < event_batch:
            monitor.check(f"before_canonical_batch_{cursor}")
            monitor.require_clean()
            _advance_source_batch(
                torch,
                producer,
                model,
                canonical,
                capture["batch_rows"][cursor],
                capture["ledger"],
            )
            monitor.check(f"after_canonical_batch_{cursor}")
            monitor.require_clean()
            cursor += 1
        target = str(event["target_request_id"])
        if int(canonical[target].decode_step) != int(event["decode_step"]):
            raise ProtocolError("canonical target state did not reach the selected pre-step")
        results.append(
            _run_event(
                torch,
                producer,
                model,
                capture,
                canonical,
                event,
                alignments[str(event["event_id"])],
                repeats,
                monitor,
            )
        )
    return results


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise ProtocolError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise ProtocolError(f"refusing to overwrite output: {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def summarize_profile_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen cross-regime continuation rule to event verdicts."""

    regimes: dict[str, dict[str, Any]] = {}
    all_qualifying_classes: list[str] = []
    for regime in ("steady", "bursty"):
        regime_results = [value for value in results if value["arrival_regime"] == regime]
        qualifying: list[dict[str, str]] = []
        for value in regime_results:
            classification = value["summary"]["classification"]
            source_classes = [
                label
                for label in classification["frozen_classification"]
                if label in PRIMARY_SOURCE_CLASSES
            ]
            if classification["status"] == "LOCALIZED_DEVELOPMENT_SIGNAL" and len(source_classes) == 1:
                qualifying.append(
                    {"event_id": str(value["event_id"]), "source_class": source_classes[0]}
                )
                all_qualifying_classes.append(source_classes[0])
        regimes[regime] = {
            "event_count": len(regime_results),
            "qualifying_events": qualifying,
            "has_repeat_qualified_localization": bool(qualifying),
            "source_classes": sorted({value["source_class"] for value in qualifying}),
        }
    both_regimes = all(
        regimes[regime]["has_repeat_qualified_localization"]
        for regime in ("steady", "bursty")
    )
    source_classes = sorted(set(all_qualifying_classes))
    consistent_class = source_classes[0] if both_regimes and len(source_classes) == 1 else None
    continue_gate = bool(both_regimes and consistent_class is not None)
    return {
        "status": (
            "LOCALIZED_DEVELOPMENT_SIGNAL"
            if continue_gate
            else "STOP_OR_MIXED_EVENT_RESULTS"
        ),
        "continue_gate_passed": continue_gate,
        "requires_at_least_one_qualifying_event_per_regime": True,
        "requires_one_consistent_primary_source_class_across_regimes": True,
        "consistent_primary_source_class": consistent_class,
        "observed_qualifying_source_classes": source_classes,
        "regimes": regimes,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.offline:
        raise ProtocolError("pass --offline; model/tokenizer downloads are forbidden")
    if args.repeats < 3:
        raise ProtocolError("every arm requires at least three repeats")
    output = Path(args.output).resolve()
    if output.suffix != ".json" or output.exists():
        raise ProtocolError("output must be one new .json file")
    events_path = Path(args.events).resolve()
    config, selected = load_events_config(events_path, args.profile)
    producer = _load_producer()
    capture_specs = config.get("captures")
    if not isinstance(capture_specs, Mapping):
        raise ProtocolError("event config lacks capture bindings")
    captures = {
        "steady": load_capture(Path(args.steady_capture_dir).resolve(), capture_specs["steady"], producer),
        "bursty": load_capture(Path(args.bursty_capture_dir).resolve(), capture_specs["bursty"], producer),
    }
    model_spec = config.get("model")
    if not isinstance(model_spec, Mapping):
        raise ProtocolError("event config lacks model identity")
    for capture in captures.values():
        manifest_model = capture["manifest"].get("model")
        if not isinstance(manifest_model, Mapping):
            raise ProtocolError("capture manifest lacks model identity")
        for key in ("id", "revision", "tokenizer_revision", "dtype"):
            if str(manifest_model.get(key, "")) != str(model_spec.get(key, "")):
                raise ProtocolError(f"capture model.{key} differs from event config")

    if _query_gpu_processes():
        raise ProtocolError("GPU is not idle before model load")
    torch, transformers, tokenizer, model, load_seconds = load_exact_model(model_spec)
    current_environment = _runtime_environment(torch, transformers)
    runtime_checks = {
        regime: _validate_runtime(capture["environment"], current_environment)
        for regime, capture in captures.items()
    }
    own_process = _query_gpu_processes()
    monitor = GpuIsolationMonitor(own_process)
    torch.cuda.reset_peak_memory_stats(0)
    monitor.start()
    started = time.monotonic()
    try:
        results: list[dict[str, Any]] = []
        for regime in ("steady", "bursty"):
            regime_events = [event for event in selected if event["arrival_regime"] == regime]
            results.extend(
                _run_capture_events(
                    torch,
                    producer,
                    tokenizer,
                    model,
                    captures[regime],
                    regime_events,
                    args.repeats,
                    monitor,
                )
            )
        torch.cuda.synchronize(0)
        elapsed = time.monotonic() - started
    finally:
        monitor.stop()
    monitor.require_clean()
    result_by_id = {value["event_id"]: value for value in results}
    ordered_results = [result_by_id[str(event["event_id"])] for event in selected]
    profile_summary = summarize_profile_results(ordered_results)
    status = str(profile_summary["status"])
    payload = {
        "schema": "moe-execution-conformance-source-localization-v1",
        "status": status,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": {
            "evidence_tier": "CUSTOM_CONTINUOUS_RUNTIME_DEVELOPMENT_DIAGNOSTIC",
            "capacity_claim_authorized": False,
            "controller_action_executed": False,
            "full_request_benefit_measured": False,
            "arm_c_tests_companion_identity": False,
            "captured_model_signals": list(SIGNALS),
        },
        "profile": args.profile,
        "profile_summary": profile_summary,
        "event_selection": {
            "path": str(events_path),
            "sha256": _sha256_file(events_path),
            "event_ids": [str(event["event_id"]) for event in selected],
        },
        "source_captures": {
            regime: {
                "path": str(capture["capture_dir"]),
                "capture_complete_sha256": _sha256_file(
                    capture["capture_dir"] / "CAPTURE_COMPLETE.json"
                ),
                "files_sha256": dict(capture["complete"]["files"]),
            }
            for regime, capture in captures.items()
        },
        "model": {**dict(model_spec), "offline": True, "load_seconds": load_seconds},
        "execution": {
            "repeats_per_arm": args.repeats,
            "allclose": {"atol": ATOL, "rtol": RTOL},
            "near_tie_margin": NEAR_TIE_MARGIN,
            "near_tie_concentration_fraction": NEAR_TIE_CONCENTRATION_FRACTION,
            "elapsed_seconds_excluding_model_load": elapsed,
            "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "runtime_environment": current_environment,
            "runtime_checks": runtime_checks,
            "gpu_isolation": monitor.summary(),
            "git": _git_state(),
            "runner_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "events": ordered_results,
    }
    _write_json_exclusive(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events", default=str(Path(__file__).with_name("events.json"))
    )
    parser.add_argument("--profile", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--steady-capture-dir", required=True)
    parser.add_argument("--bursty-capture-dir", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "profile": payload["profile"],
                "events": len(payload["events"]),
                "output": str(Path(args.output).resolve()),
                "capacity_claim_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

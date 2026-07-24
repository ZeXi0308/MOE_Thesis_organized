#!/usr/bin/env python3
"""Capture clean-v2 calibration routes from unmodified native MoE forwards.

This formal producer has no dev mode and exposes no input/output path knobs.
It emits calibration input artifacts only; it never emits a scientific result.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

try:
    from . import native_route_core as core
    from . import prepare_clean_v2_data as data
except ImportError:  # pragma: no cover - direct script execution
    import native_route_core as core  # type: ignore
    import prepare_clean_v2_data as data  # type: ignore


HERE = Path(__file__).resolve().parent
IDEA_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[4]
DEFAULT_CONFIG = IDEA_ROOT / "configs/ric_clean_v2.json"
BASE_PROTOCOL = IDEA_ROOT / "RIC_Clean_v2_Phase2_冻结实验协议_2026-07-23.md"
ROUTE_ADDENDUM = IDEA_ROOT / "RIC_Clean_v2_CalibrationRoute_Phase2_Addendum_2026-07-23.md"
CLEAN_BUNDLE_ROOT = Path("/root/autodl-tmp/ric_clean_v2_20260723")
CALIBRATION_MANIFEST = CLEAN_BUNDLE_ROOT / "clean_v2/data/calibration/manifest.json"
CALIBRATION_MANIFEST_SHA256 = (
    "e07d5b4d42b1f9e59d21ccab1b40fa31fb9052e19d4e8f83ec31a19b3b21d545"
)
MODEL_DIRS = {
    "olmoe": Path("/root/autodl-tmp/models/olmoe"),
    "llmjp": Path("/root/autodl-tmp/models/llmjp"),
}
ROUTE_OUTPUTS = {
    key: CLEAN_BUNDLE_ROOT / f"clean_v2/routes/calibration/{key}"
    for key in MODEL_DIRS
}
CLEAN_REVIEW_DIR = CLEAN_BUNDLE_ROOT / "review"
REVIEW_REPORT = CLEAN_REVIEW_DIR / "RIC_Clean_v2_Route_CodeReview.md"
TEST_REPORT = CLEAN_REVIEW_DIR / "RIC_Clean_v2_Route_TestReport.json"
REVIEWED_SOURCE_MANIFEST = CLEAN_REVIEW_DIR / "reviewed_source_manifest_route.json"
SIGNOFFS = {
    key: CLEAN_REVIEW_DIR / f"signoff_route_{key}.json" for key in MODEL_DIRS
}
ROUTE_EPOCH = 1
ROUTE_STATE_DIR = CLEAN_BUNDLE_ROOT / "state"
ROUTE_LEDGERS = {
    key: ROUTE_STATE_DIR / f"route_calibration_{key}_consumption.json"
    for key in MODEL_DIRS
}


class CleanRouteError(RuntimeError):
    """A clean route provenance, environment, or parity invariant failed."""


def load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        return data.load_mapping(path, label=label)
    except data.CleanDataError as exc:
        raise CleanRouteError(str(exc)) from exc


def add_self_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return data.add_self_hash(value)
    except data.CleanDataError as exc:
        raise CleanRouteError(str(exc)) from exc


def source_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        HERE / "native_route_core.py",
        HERE / "prepare_clean_v2_data.py",
    ):
        digest.update(path.resolve().relative_to(REPO_ROOT.resolve()).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_calibration_manifest() -> Mapping[str, Any]:
    try:
        value = data.validate_calibration_manifest(CALIBRATION_MANIFEST)
    except data.CleanDataError as exc:
        raise CleanRouteError(str(exc)) from exc
    if value["manifest_sha256"] != CALIBRATION_MANIFEST_SHA256:
        raise CleanRouteError("calibration manifest is not the addendum-frozen input")
    config = load_mapping(DEFAULT_CONFIG, label="clean-v2 config")
    requests = value.get("requests")
    selected_hashes = value.get("selected_text_sha256")
    selected_ids = value.get("selected_request_ids")
    expected_count = int(config["data"]["calibration"]["document_count"])
    if (
        not isinstance(requests, list)
        or len(requests) != expected_count
        or not isinstance(selected_hashes, list)
        or not isinstance(selected_ids, list)
        or len(selected_hashes) != expected_count
        or len(selected_ids) != expected_count
    ):
        raise CleanRouteError("calibration request census mismatch")
    observed_ids: list[str] = []
    observed_hashes: list[str] = []
    observed_rows: list[int] = []
    minimum = int(config["data"]["min_tokens_both_frozen_tokenizers"])
    start = int(config["data"]["calibration"]["candidate_row_start_inclusive"])
    stop = int(config["data"]["calibration"]["candidate_row_end_exclusive"])
    for request in requests:
        if not isinstance(request, Mapping):
            raise CleanRouteError("calibration request is not an object")
        request_id = request.get("request_id")
        text = request.get("text")
        text_hash = request.get("text_sha256")
        lengths = request.get("token_lengths")
        source_row = request.get("source_row")
        if (
            not isinstance(request_id, str)
            or not request_id.startswith("ric-clean-v2:calibration:")
            or not isinstance(text, str)
            or hashlib.sha256(text.encode("utf-8")).hexdigest() != text_hash
            or not isinstance(lengths, Mapping)
            or set(lengths) != set(MODEL_DIRS)
            or any(type(lengths[key]) is not int or lengths[key] < minimum for key in MODEL_DIRS)
            or type(source_row) is not int
            or not start <= source_row < stop
        ):
            raise CleanRouteError("calibration request identity/content mismatch")
        observed_ids.append(request_id)
        observed_hashes.append(str(text_hash))
        observed_rows.append(source_row)
    if (
        observed_ids != selected_ids
        or observed_hashes != selected_hashes
        or len(set(observed_ids)) != expected_count
        or len(set(observed_hashes)) != expected_count
        or len(set(observed_rows)) != expected_count
    ):
        raise CleanRouteError("calibration selected lists differ from request rows")
    if not _is_sha256(value.get("phase4_signoff_sha256")):
        raise CleanRouteError("calibration data Phase-4 signoff is missing")
    return value


def _reviewed_sources() -> dict[str, dict[str, str]]:
    paths = {
        "config": DEFAULT_CONFIG,
        "base_protocol": BASE_PROTOCOL,
        "route_addendum": ROUTE_ADDENDUM,
        "data_producer": HERE / "prepare_clean_v2_data.py",
        "native_route_core": HERE / "native_route_core.py",
        "route_producer": Path(__file__),
        "route_tests": HERE / "test_capture_clean_v2_routes_gpu.py",
    }
    return {
        key: {
            "path": path.resolve(strict=True).relative_to(REPO_ROOT.resolve()).as_posix(),
            "file_sha256": data.file_sha256(path),
        }
        for key, path in paths.items()
    }


def validate_route_signoff(
    model_key: str, manifest: Mapping[str, Any]
) -> Mapping[str, Any]:
    path = SIGNOFFS[model_key]
    if path.is_symlink() or path.resolve(strict=True) != path:
        raise CleanRouteError("route signoff path identity mismatch")
    value = load_mapping(path, label="clean-v2 route signoff")
    try:
        data.validate_self_hash(value, "signoff_sha256")
    except data.CleanDataError as exc:
        raise CleanRouteError(str(exc)) from exc
    for evidence in (REVIEW_REPORT, TEST_REPORT, REVIEWED_SOURCE_MANIFEST):
        if evidence.is_symlink() or evidence.resolve(strict=True) != evidence:
            raise CleanRouteError("route review evidence identity mismatch")
    source_manifest = load_mapping(
        REVIEWED_SOURCE_MANIFEST, label="clean-v2 route reviewed sources"
    )
    try:
        data.validate_self_hash(source_manifest)
    except data.CleanDataError as exc:
        raise CleanRouteError(str(exc)) from exc
    if (
        source_manifest.get("schema_version")
        != "ric-clean-v2-route-reviewed-source-manifest-v1"
        or source_manifest.get("status") != "REVIEWED"
        or source_manifest.get("sources") != _reviewed_sources()
    ):
        raise CleanRouteError("route reviewed-source closure mismatch")
    review_lines = REVIEW_REPORT.read_text(encoding="utf-8").splitlines()
    if (
        "STATUS: SIGNED-OFF" not in review_lines
        or "OPEN_P0: 0" not in review_lines
        or f"REVIEWED_SOURCE_MANIFEST_SHA256: {source_manifest['manifest_sha256']}"
        not in review_lines
    ):
        raise CleanRouteError("route review is not signed off")
    test_report = load_mapping(TEST_REPORT, label="clean-v2 route test report")
    try:
        data.validate_self_hash(test_report)
    except data.CleanDataError as exc:
        raise CleanRouteError(str(exc)) from exc
    if (
        test_report.get("schema_version") != "ric-clean-v2-route-test-report-v1"
        or test_report.get("status") != "PASS"
        or test_report.get("errors") != 0
        or test_report.get("failures") != 0
        or type(test_report.get("tests_run")) is not int
        or test_report["tests_run"] < 1
        or test_report.get("reviewed_source_manifest_sha256")
        != source_manifest["manifest_sha256"]
        or test_report.get("reviewed_source_manifest_file_sha256")
        != data.file_sha256(REVIEWED_SOURCE_MANIFEST)
    ):
        raise CleanRouteError("route test report is not bound to reviewed sources")
    expected = {
        "schema_version": "ric-clean-v2-route-phase4-signoff-v1",
        "status": "SIGNED-OFF",
        "open_p0": 0,
        "stage": "capture_calibration_routes",
        "model_key": model_key,
        "config_sha256": data.file_sha256(DEFAULT_CONFIG),
        "base_protocol_sha256": data.file_sha256(BASE_PROTOCOL),
        "route_addendum_sha256": data.file_sha256(ROUTE_ADDENDUM),
        "route_producer_source_sha256": source_sha256(),
        "data_manifest_sha256": manifest["manifest_sha256"],
        "data_manifest_file_sha256": data.file_sha256(CALIBRATION_MANIFEST),
        "data_phase4_signoff_sha256": manifest["phase4_signoff_sha256"],
        "model_tree_manifest_sha256": load_mapping(DEFAULT_CONFIG, label="clean-v2 config")["models"][model_key]["expected_local_model_tree_manifest_sha256"],
        "review_report_sha256": data.file_sha256(REVIEW_REPORT),
        "test_report_sha256": data.file_sha256(TEST_REPORT),
        "reviewed_source_manifest_sha256": source_manifest["manifest_sha256"],
        "reviewed_source_manifest_file_sha256": data.file_sha256(REVIEWED_SOURCE_MANIFEST),
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value or type(value.get(field)) is not type(expected_value):
            raise CleanRouteError(f"route signoff mismatch: {field}")
    return value


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
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise CleanRouteError("BLOCKED_GPU_ENVIRONMENT: compute-app query failed") from exc
    if not output or "No running processes found" in output:
        return []
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 4:
            raise CleanRouteError("BLOCKED_GPU_ENVIRONMENT: malformed compute-app query")
        try:
            rows.append(
                {
                    "pid": int(fields[0]),
                    "gpu_uuid": fields[1],
                    "process_name": fields[2],
                    "used_gpu_memory_mib": float(fields[3]),
                }
            )
        except ValueError as exc:
            raise CleanRouteError("BLOCKED_GPU_ENVIRONMENT: invalid compute-app row") from exc
    return sorted(rows, key=lambda row: (row["pid"], row["process_name"]))


def validate_compute_apps(rows: Sequence[Mapping[str, Any]], *, producer_pid: int) -> None:
    foreign = [row for row in rows if row.get("pid") != producer_pid]
    if foreign:
        raise CleanRouteError(f"BLOCKED_GPU_ENVIRONMENT: foreign GPU processes: {foreign}")


def capture_gpu_environment(
    torch: Any, *, compute_apps_before: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,clocks.sm,power.draw,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise CleanRouteError("BLOCKED_GPU_ENVIRONMENT: GPU query failed") from exc
    fields = [item.strip() for item in output.split(",")]
    if len(fields) != 7:
        raise CleanRouteError("BLOCKED_GPU_ENVIRONMENT: malformed GPU query")
    after = _query_compute_apps()
    validate_compute_apps(after, producer_pid=os.getpid())
    if (
        fields[1] != "NVIDIA GeForce RTX 5090"
        or torch.cuda.get_device_name(0) != fields[1]
        or torch.version.cuda is None
    ):
        raise CleanRouteError("BLOCKED_GPU_ENVIRONMENT: device identity drift")
    return {
        "producer_pid": os.getpid(),
        "gpu_uuid": fields[0],
        "gpu_name": fields[1],
        "driver_version": fields[2],
        "clock_sm_mhz": float(fields[3]),
        "power_draw_w": float(fields[4]),
        "memory_used_mib": float(fields[5]),
        "background_gpu_util_percent": float(fields[6]),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "compute_apps_before": [dict(row) for row in compute_apps_before],
        "compute_apps_after": after,
    }


def reserve_route(
    model_key: str,
    *,
    manifest: Mapping[str, Any],
    signoff: Mapping[str, Any],
) -> Mapping[str, Any]:
    ROUTE_STATE_DIR.mkdir(parents=True, exist_ok=True)
    ledger = ROUTE_LEDGERS[model_key]
    if ROUTE_STATE_DIR.is_symlink() or ledger.is_symlink():
        raise CleanRouteError("route state path may not be a symlink")
    record = add_self_hash(
        {
            "schema_version": "ric-clean-v2-route-consumption-v1",
            "state": "RESERVED_FAIL_CLOSED",
            "model_key": model_key,
            "data_manifest_sha256": manifest["manifest_sha256"],
            "route_phase4_signoff_sha256": signoff["signoff_sha256"],
            "config_sha256": data.file_sha256(DEFAULT_CONFIG),
            "base_protocol_sha256": data.file_sha256(BASE_PROTOCOL),
            "route_addendum_sha256": data.file_sha256(ROUTE_ADDENDUM),
        }
    )
    encoded = json.dumps(record, indent=2, sort_keys=True).encode() + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(ledger, flags, 0o400)
    except FileExistsError as exc:
        raise CleanRouteError(f"clean route {model_key} was already consumed") from exc
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise CleanRouteError("route reservation write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if ledger.read_bytes() != encoded:
        raise CleanRouteError("route reservation byte verification failed")
    return record


def _extract_router_logits(outputs: Any) -> list[Any]:
    value = getattr(outputs, "router_logits", None)
    if value is None and isinstance(outputs, Mapping):
        value = outputs.get("router_logits")
    if value is None:
        raise CleanRouteError("native output returned no router_logits")
    if hasattr(value, "shape"):
        return [value]
    if not isinstance(value, (tuple, list)):
        raise CleanRouteError("native router_logits has unknown container")
    tensors = [item for item in value if hasattr(item, "shape")]
    if not tensors:
        raise CleanRouteError("native router_logits container has no tensors")
    return tensors


def _load_model(torch: Any, config: Mapping[str, Any], model_key: str) -> tuple[Any, Any, str]:
    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise CleanRouteError("transformers is required") from exc
    spec = config["models"][model_key]
    model_path = MODEL_DIRS[model_key]
    actual_tree = data.model_tree_sha256(model_path)
    if actual_tree != spec["expected_local_model_tree_manifest_sha256"]:
        raise CleanRouteError("frozen model tree mismatch")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()
    model.config.output_router_logits = True
    return model, tokenizer, transformers.__version__


def _write_artifacts(
    *,
    output_dir: Path,
    route_rows: Sequence[Mapping[str, Any]],
    placement: Mapping[str, Any],
    parity: Mapping[str, Any],
    metadata_payload: Mapping[str, Any],
    signoff_bytes: bytes,
    expected_signoff_sha256: str,
) -> Mapping[str, Any]:
    if output_dir.exists():
        raise CleanRouteError("refusing to overwrite clean route output")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.partial-", dir=output_dir.parent
    ) as raw:
        temporary = Path(raw)
        trace_path = temporary / "route_trace.jsonl"
        with trace_path.open("x", encoding="utf-8") as handle:
            for row in route_rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
        (temporary / "placement.json").write_text(
            json.dumps(placement, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "route_parity.json").write_text(
            json.dumps(parity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        signoff_target = temporary / "producer_signoff.json"
        signoff_target.write_bytes(signoff_bytes)
        embedded = load_mapping(signoff_target, label="embedded route signoff")
        try:
            data.validate_self_hash(embedded, "signoff_sha256")
        except data.CleanDataError as exc:
            raise CleanRouteError(str(exc)) from exc
        if embedded.get("signoff_sha256") != expected_signoff_sha256:
            raise CleanRouteError("embedded route signoff self-hash mismatch")
        metadata = add_self_hash(
            {
                **metadata_payload,
                "route_trace_file_sha256": data.file_sha256(trace_path),
                "placement_file_sha256": data.file_sha256(temporary / "placement.json"),
                "route_parity_file_sha256": data.file_sha256(temporary / "route_parity.json"),
                "producer_signoff_file_sha256": data.file_sha256(signoff_target),
            }
        )
        (temporary / "capture_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(output_dir)
    return metadata


def validate_route_census(
    route_rows: Sequence[Mapping[str, Any]],
    parity_rows: Sequence[Mapping[str, Any]],
    *,
    request_ids: set[str],
    layer_ids: set[int],
    top_k: int,
) -> None:
    expected_rows = len(request_ids) * len(layer_ids) * 128 * top_k
    route_keys = {
        (
            row.get("request_id"),
            row.get("layer_id"),
            row.get("token_position"),
            row.get("topk_slot"),
        )
        for row in route_rows
    }
    parity_keys = {(row.get("request_id"), row.get("layer_id")) for row in parity_rows}
    if (
        len(route_rows) != expected_rows
        or len(route_keys) != expected_rows
        or any(
            request_id not in request_ids
            or layer_id not in layer_ids
            or type(token_position) is not int
            or not 0 <= token_position < 128
            or type(slot) is not int
            or not 0 <= slot < top_k
            for request_id, layer_id, token_position, slot in route_keys
        )
        or len(parity_rows) != len(request_ids) * len(layer_ids)
        or parity_keys != {(request_id, layer_id) for request_id in request_ids for layer_id in layer_ids}
    ):
        raise CleanRouteError("route/parity Cartesian identity mismatch")


def run(model_key: str) -> Mapping[str, Any]:
    if model_key not in MODEL_DIRS:
        raise CleanRouteError("unknown model key")
    if CLEAN_BUNDLE_ROOT.resolve(strict=True) != CLEAN_BUNDLE_ROOT:
        raise CleanRouteError("clean bundle root identity mismatch")
    output_dir = ROUTE_OUTPUTS[model_key]
    if output_dir.exists():
        raise CleanRouteError("clean route output already exists")
    for ancestor in (
        CLEAN_BUNDLE_ROOT / "clean_v2",
        CLEAN_BUNDLE_ROOT / "clean_v2/routes",
        CLEAN_BUNDLE_ROOT / "clean_v2/routes/calibration",
    ):
        if ancestor.exists() and (ancestor.is_symlink() or ancestor.resolve(strict=True) != ancestor):
            raise CleanRouteError("route output ancestor identity mismatch")
    manifest = validate_calibration_manifest()
    signoff = validate_route_signoff(model_key, manifest)
    config = load_mapping(DEFAULT_CONFIG, label="clean-v2 config")
    try:
        import datasets
        import tokenizers
        import torch
        import transformers
    except ImportError as exc:
        raise CleanRouteError("formal runtime dependencies are missing") from exc
    try:
        data.validate_runtime_identity(
            config["data"]["formal_dataset_identity"],
            datasets_version=datasets.__version__,
            transformers_version=transformers.__version__,
            tokenizers_version=tokenizers.__version__,
        )
    except data.CleanDataError as exc:
        raise CleanRouteError(str(exc)) from exc
    if not torch.cuda.is_available():
        raise CleanRouteError("BLOCKED_GPU_ENVIRONMENT: CUDA is unavailable")
    compute_apps_before = _query_compute_apps()
    validate_compute_apps(compute_apps_before, producer_pid=os.getpid())
    reservation = reserve_route(
        model_key, manifest=manifest, signoff=signoff
    )
    model, tokenizer, transformers_version = _load_model(torch, config, model_key)
    spec = config["models"][model_key]
    modules = core.discover_moe_modules(model)
    census = core.validate_model_config_layer_census(
        model.config,
        modules,
        expected_num_experts=int(spec["num_experts"]),
        expected_top_k=int(spec["top_k"]),
    )
    native_contract = core.validate_native_moe_implementation(
        modules, model_spec=spec, route_config=config["route_capture"]
    )
    layer_ids = list(census["expected_layers"])
    model_revision = f"{spec['repo_id']}@{spec['revision']}"
    replay_layers = core.selected_layers(
        layer_ids,
        selection_seed=int(config["data"]["selection_seed"]),
        model_revision=model_revision,
        count=int(config["route_capture"]["selected_layer_count_per_model"]),
    )
    module_by_layer = {layer: module for layer, _name, module in modules}
    normalize_by_layer = {layer: core.normalizes_topk(module) for layer, module in module_by_layer.items()}
    active_layer: dict[str, int | None] = {"value": None}
    native_topk = core.make_native_topk_capture_mode(
        active_layer=active_layer,
        expected_num_experts=int(spec["num_experts"]),
        expected_top_k=int(spec["top_k"]),
        expected_tokens=int(config["route_capture"]["valid_tokens_per_request"]),
    )
    gate_outputs: dict[int, Any] = {}
    moe_inputs: dict[int, Any] = {}
    moe_outputs: dict[int, Any] = {}
    handles: list[Any] = []

    def pre_hook(layer_id: int):
        def hook(_module: Any, _inputs: Any) -> None:
            if active_layer["value"] is not None:
                raise CleanRouteError("nested native MoE execution")
            active_layer["value"] = layer_id
        return hook

    def gate_hook(layer_id: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if layer_id in gate_outputs:
                raise CleanRouteError("native gate called twice")
            gate_outputs[layer_id] = output.detach()
        return hook

    def moe_hook(layer_id: int):
        def hook(_module: Any, inputs: Any, output: Any) -> None:
            if layer_id in moe_inputs:
                raise CleanRouteError("native MoE called twice")
            moe_inputs[layer_id] = core.first_tensor(inputs).detach()
            moe_outputs[layer_id] = core.first_tensor(output).detach()
        return hook

    def clear_hook(layer_id: int):
        def hook(_module: Any, _inputs: Any, _output: Any) -> None:
            if active_layer["value"] != layer_id:
                raise CleanRouteError("active native MoE layer drift")
            active_layer["value"] = None
        return hook

    for layer_id, _name, module in modules:
        handles.extend(
            (
                module.register_forward_pre_hook(pre_hook(layer_id)),
                module.gate.register_forward_hook(gate_hook(layer_id)),
                module.register_forward_hook(moe_hook(layer_id)),
                module.register_forward_hook(clear_hook(layer_id)),
            )
        )
    requests = manifest["requests"]
    ep_size = int(config["topology_proxy"]["ep_size"])
    request_to_receiver = core.origin_lpt(requests, ep_size)
    if set(request_to_receiver.values()) != set(range(ep_size)):
        raise CleanRouteError("receiver coverage does not include every rank")
    placement = add_self_hash(
        {
            "schema_version": "ric-clean-v2-placement-v1",
            "status": "CALIBRATION_INPUT_ONLY",
            "scientific_result": False,
            "model_key": model_key,
            "model_revision": model_revision,
            "ep_size": ep_size,
            "virtual_nodes": int(config["topology_proxy"]["virtual_nodes"]),
            "ranks_per_node": int(config["topology_proxy"]["ranks_per_node"]),
            "expert_placement": "contiguous",
            "request_origin": "route_blind_token_count_lpt",
            "expert_to_sender": {
                str(expert): core.expert_sender(expert, int(spec["num_experts"]), ep_size)
                for expert in range(int(spec["num_experts"]))
            },
            "request_to_receiver": request_to_receiver,
            "receiver_token_count_by_rank": {
                str(rank): sum(
                    128 for request_id in request_to_receiver
                    if request_to_receiver[request_id] == rank
                )
                for rank in range(ep_size)
            },
            "receiver_token_count_by_node": {
                str(node): sum(
                    128 for request_id in request_to_receiver
                    if request_to_receiver[request_id]
                    // int(config["topology_proxy"]["ranks_per_node"]) == node
                )
                for node in range(int(config["topology_proxy"]["virtual_nodes"]))
            },
        }
    )
    if (
        sum(placement["receiver_token_count_by_rank"].values()) != len(requests) * 128
        or sum(placement["receiver_token_count_by_node"].values()) != len(requests) * 128
        or any(value <= 0 for value in placement["receiver_token_count_by_rank"].values())
        or any(value <= 0 for value in placement["receiver_token_count_by_node"].values())
    ):
        raise CleanRouteError("receiver token-load accounting mismatch")
    route_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    try:
        for request in requests:
            request_id = str(request["request_id"])
            encoded = tokenizer(
                str(request["text"]), add_special_tokens=False, return_tensors="pt"
            )
            full_ids = encoded["input_ids"]
            observed_length = int(full_ids.shape[1])
            if (
                tuple(full_ids.shape[:1]) != (1,)
                or observed_length != int(request["token_lengths"][model_key])
                or observed_length < int(config["data"]["min_tokens_both_frozen_tokenizers"])
            ):
                raise CleanRouteError("tokenizer length differs from calibration manifest")
            input_ids = full_ids[:, :128].contiguous()
            if tuple(input_ids.shape) != (1, 128):
                raise CleanRouteError("route input is not exactly 128 tokens")
            gate_outputs.clear()
            moe_inputs.clear()
            moe_outputs.clear()
            native_topk.calls.clear()
            with torch.inference_mode(), native_topk:
                outputs = model(
                    input_ids=input_ids.to("cuda:0"),
                    use_cache=False,
                    output_router_logits=True,
                    return_dict=True,
                )
            torch.cuda.synchronize()
            router_outputs = _extract_router_logits(outputs)
            if (
                sorted(gate_outputs) != layer_ids
                or sorted(moe_inputs) != layer_ids
                or sorted(moe_outputs) != layer_ids
                or sorted(native_topk.calls) != layer_ids
                or len(router_outputs) != len(layer_ids)
                or active_layer["value"] is not None
            ):
                raise CleanRouteError("native route layer census is incomplete")
            assigned = core.assigned_layer(request_id, replay_layers)
            for route_event_index, (layer_id, output_logits) in enumerate(zip(layer_ids, router_outputs)):
                gate_logits = gate_outputs[layer_id]
                raw_identity = core.validate_raw_router_tensor_identity(
                    gate_logits,
                    output_logits,
                    expected_shape=(128, int(spec["num_experts"])),
                )
                native_values, native_experts = native_topk.calls[layer_id]
                native_weights = core.effective_route_weights(
                    native_values,
                    normalize_topk=normalize_by_layer[layer_id],
                    output_dtype=moe_inputs[layer_id].dtype,
                )
                replay_experts, replay_weights = core.routes_from_logits(
                    gate_logits,
                    top_k=int(spec["top_k"]),
                    normalize_topk=normalize_by_layer[layer_id],
                    selection_rule=core.NATIVE_TOPK_SELECTION_RULE,
                    output_dtype=moe_inputs[layer_id].dtype,
                )
                if not torch.equal(native_experts, replay_experts) or not torch.equal(native_weights, replay_weights):
                    raise CleanRouteError("native ordered top-k tuple differs from raw-logit replay")
                reconstructed, _, _ = core.reconstruct_native_moe_output(
                    moe=module_by_layer[layer_id],
                    hidden_states=moe_inputs[layer_id],
                    selected_experts=native_experts,
                    effective_weights=native_weights,
                )
                parity_evidence = core.validate_native_moe_output_parity(
                    moe_outputs[layer_id],
                    reconstructed,
                    tolerance_rule=config["route_capture"]["native_moe_output_tolerance"],
                )
                tuple_sha = core.route_tuple_sha256(native_experts, native_weights)
                parity_rows.append(
                    {
                        "request_id": request_id,
                        "layer_id": layer_id,
                        **raw_identity,
                        "native_route_tuple_sha256": tuple_sha,
                        "topk_expert_exact": True,
                        "topk_effective_weight_exact": True,
                        **parity_evidence,
                    }
                )
                fp32_weights = core.precast_route_weights(
                    native_values, normalize_topk=normalize_by_layer[layer_id]
                )
                for token_position in range(128):
                    experts = [int(value) for value in native_experts[token_position].detach().cpu().tolist()]
                    if len(experts) != int(spec["top_k"]) or len(set(experts)) != len(experts):
                        raise CleanRouteError("native top-k join set is invalid")
                    token_id = f"{request_id}:token:{token_position:03d}"
                    for slot, expert_id in enumerate(experts):
                        route_rows.append(
                            {
                                "schema_version": "ric-clean-v2-route-row-v1",
                                "model_key": model_key,
                                "model_revision": model_revision,
                                "data_manifest_sha256": manifest["manifest_sha256"],
                                "placement_manifest_sha256": placement["manifest_sha256"],
                                "request_id": request_id,
                                "forward_id": f"{request_id}:prefill:0",
                                "batch_id": f"batch:{request_id}:prefill:0",
                                "phase": "prefill",
                                "decode_step": 0,
                                "layer_id": layer_id,
                                "token_id": token_id,
                                "token_block_id": token_id,
                                "token_position": token_position,
                                "topk_slot": slot,
                                "expert_id": expert_id,
                                "sender_rank": core.expert_sender(expert_id, int(spec["num_experts"]), ep_size),
                                "receiver_rank": request_to_receiver[request_id],
                                "epoch": ROUTE_EPOCH,
                                "valid": True,
                                "route_weight": float(native_weights[token_position, slot]),
                                "route_weight_dtype": str(native_weights.dtype),
                                "route_weight_fp32_precast": float(fp32_weights[token_position, slot]),
                                "route_event_index": route_event_index,
                                "selected_for_replay": layer_id == assigned,
                                "native_route_tuple_sha256": tuple_sha,
                                "route_source": "native_aten_topk_plus_raw_logit_and_output_parity",
                            }
                        )
    except core.NativeRouteError as exc:
        raise CleanRouteError(str(exc)) from exc
    finally:
        for handle in handles:
            handle.remove()
    validate_route_census(
        route_rows,
        parity_rows,
        request_ids={str(row["request_id"]) for row in requests},
        layer_ids=set(layer_ids),
        top_k=int(spec["top_k"]),
    )
    parity = add_self_hash(
        {
            "schema_version": "ric-clean-v2-route-parity-v1",
            "status": "CALIBRATION_INPUT_ONLY",
            "scientific_result": False,
            "model_key": model_key,
            "model_revision": model_revision,
            **census,
            "selected_replay_layers": replay_layers,
            **native_contract,
            "parity_rows": parity_rows,
            "all_raw_logits_exact": True,
            "all_ordered_topk_exact": True,
            "all_native_outputs_within_frozen_tolerance": True,
        }
    )
    gpu_environment = capture_gpu_environment(
        torch, compute_apps_before=compute_apps_before
    )
    if data.model_tree_sha256(MODEL_DIRS[model_key]) != spec["expected_local_model_tree_manifest_sha256"]:
        raise CleanRouteError("model tree changed during capture")
    final_signoff = validate_route_signoff(model_key, manifest)
    if final_signoff["signoff_sha256"] != signoff["signoff_sha256"]:
        raise CleanRouteError("route signoff changed during capture")
    signoff_bytes = SIGNOFFS[model_key].read_bytes()
    captured_signoff = data.decode_json_bytes(
        signoff_bytes, label="captured route signoff"
    )
    if not isinstance(captured_signoff, Mapping):
        raise CleanRouteError("captured route signoff is not an object")
    try:
        data.validate_self_hash(captured_signoff, "signoff_sha256")
    except data.CleanDataError as exc:
        raise CleanRouteError(str(exc)) from exc
    if captured_signoff.get("signoff_sha256") != signoff["signoff_sha256"]:
        raise CleanRouteError("route signoff changed while snapshotting")
    metadata_payload = {
        "schema_version": "ric-clean-v2-route-capture-v1",
        "status": "CALIBRATION_INPUT_ONLY",
        "scientific_result": False,
        "evidence_boundary": "NATIVE_ROUTE_CAPTURE_ONLY_NO_NETWORK_OR_SCIENTIFIC_RESULT",
        "model_key": model_key,
        "model_revision": model_revision,
        "model_tree_manifest_sha256": spec["expected_local_model_tree_manifest_sha256"],
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "transformers_version": transformers_version,
        "config_sha256": data.file_sha256(DEFAULT_CONFIG),
        "base_protocol_sha256": data.file_sha256(BASE_PROTOCOL),
        "route_addendum_sha256": data.file_sha256(ROUTE_ADDENDUM),
        "route_producer_source_sha256": source_sha256(),
        "data_manifest_sha256": manifest["manifest_sha256"],
        "data_manifest_file_sha256": data.file_sha256(CALIBRATION_MANIFEST),
        "data_phase4_signoff_sha256": manifest["phase4_signoff_sha256"],
        "route_phase4_signoff_sha256": signoff["signoff_sha256"],
        "route_reservation_sha256": reservation["manifest_sha256"],
        "placement_manifest_sha256": placement["manifest_sha256"],
        "route_parity_sha256": parity["manifest_sha256"],
        "request_count": len(requests),
        "route_rows": len(route_rows),
        "join_sets": len(route_rows) // int(spec["top_k"]),
        "all_moe_layers_captured": layer_ids,
        "selected_replay_layers": replay_layers,
        "top_k": int(spec["top_k"]),
        "num_experts": int(spec["num_experts"]),
        "gpu_environment": gpu_environment,
    }
    return _write_artifacts(
        output_dir=output_dir,
        route_rows=route_rows,
        placement=placement,
        parity=parity,
        metadata_payload=metadata_payload,
        signoff_bytes=signoff_bytes,
        expected_signoff_sha256=signoff["signoff_sha256"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=tuple(MODEL_DIRS), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(run(args.model_key), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

"""Capture natural MoE routes from a mutable continuous-decode active set.

This runner is a Gate 0 producer qualification tool. It deliberately does not
measure dispatch/expert/combine service time and never emits a Gate 1 verdict.
"""

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

try:
    from .core import (
        Contribution,
        ProtocolError,
        sha256_file,
        validate_identity_conservation,
        write_json,
        write_routes,
    )
except ImportError:
    from core import (
        Contribution,
        ProtocolError,
        sha256_file,
        validate_identity_conservation,
        write_json,
        write_routes,
    )


DurationProvider = Callable[[str, int], float]


@dataclass(frozen=True)
class ContinuousRequest:
    request_id: str
    sample_id: int
    document_id: str
    arrival_us: float
    deadline_us: float
    input_ids: Any
    attention_mask: Any


@dataclass
class _ActiveRequest:
    spec: ContinuousRequest
    cache: Any
    attention_mask: Any
    next_token: Any
    prompt_length: int
    decode_step: int = 0


@dataclass
class ContinuousCapture:
    contributions: list[Contribution] = field(default_factory=list)
    batch_rows: list[dict[str, Any]] = field(default_factory=list)
    request_rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    serial_audit: dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-manifest", required=True)
    parser.add_argument(
        "--preregistration",
        default=str(
            Path(__file__).resolve().parent
            / "configs"
            / "gate0_continuous_decode_v1.json"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="development only; formal manifests always require CUDA",
    )
    return parser.parse_args()


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be an object")
    return value


def _require_resolved(value: object, name: str) -> str:
    text = str(value or "").strip()
    if not text or text.startswith("UNRESOLVED"):
        raise ProtocolError(f"{name} is unresolved")
    return text


def _arrival_trace_sha256(requests: Sequence[Mapping[str, Any]]) -> str:
    """Hash the shared model-independent arrival/deadline trace."""

    ordered = sorted(
        requests,
        key=lambda request: (float(request["arrival_us"]), int(request["sample_id"])),
    )
    payload = [
        {
            "sample_id": int(request["sample_id"]),
            "arrival_us": format(float(request["arrival_us"]), ".17g"),
            "deadline_us": format(float(request["deadline_us"]), ".17g"),
        }
        for request in ordered
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_workload_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read workload manifest: {exc}") from exc
    manifest = dict(_require_mapping(raw, "workload manifest"))
    if manifest.get("schema") != "bcrd-continuous-workload-v1":
        raise ProtocolError("workload manifest schema must be bcrd-continuous-workload-v1")
    run_class = manifest.get("run_class")
    if run_class not in {"development", "formal"}:
        raise ProtocolError("run_class must be development or formal")
    model = _require_mapping(manifest.get("model"), "model")
    dataset = _require_mapping(manifest.get("dataset"), "dataset")
    generation = _require_mapping(manifest.get("generation"), "generation")
    scheduler = _require_mapping(manifest.get("scheduler"), "scheduler")
    software = _require_mapping(manifest.get("software"), "software")
    for key in ("id", "revision", "tokenizer_revision", "dtype"):
        _require_resolved(model.get(key), f"model.{key}")
    for key in ("id", "revision", "split"):
        _require_resolved(dataset.get(key), f"dataset.{key}")
    if software.get("execution_policy") != "clean_committed_head":
        raise ProtocolError("software.execution_policy must be clean_committed_head")
    if generation.get("mode") != "greedy" or bool(generation.get("do_sample", False)):
        raise ProtocolError("continuous producer qualification requires greedy decoding")
    max_steps = int(generation.get("max_decode_steps", 0))
    max_batch = int(scheduler.get("max_batch_size", 0))
    if max_steps <= 0 or max_batch <= 0:
        raise ProtocolError("max_decode_steps and max_batch_size must be positive")
    requests = manifest.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ProtocolError("workload manifest requests must be a non-empty list")
    if int(manifest.get("expected_requests", -1)) != len(requests):
        raise ProtocolError("expected_requests does not match the frozen request list")
    request_ids: set[str] = set()
    sample_ids: set[int] = set()
    document_ids: set[str] = set()
    dataset_row_indices: set[int] = set()
    source_csv_rows: set[int] = set()
    for index, value in enumerate(requests):
        request = _require_mapping(value, f"requests[{index}]")
        request_id = _require_resolved(request.get("request_id"), f"requests[{index}].request_id")
        sample_id = int(request.get("sample_id", -1))
        prompt_value = request.get("prompt")
        if not isinstance(prompt_value, str) or not prompt_value.strip():
            raise ProtocolError(f"requests[{index}].prompt is unresolved")
        # Preserve exact prompt bytes.  _require_resolved strips whitespace and
        # would silently change raw WikiText prompts before hash validation.
        prompt = prompt_value
        prompt_hash = _require_resolved(
            request.get("prompt_sha256"), f"requests[{index}].prompt_sha256"
        )
        if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != prompt_hash:
            raise ProtocolError(f"request {request_id} prompt SHA-256 mismatch")
        document_id = _require_resolved(
            request.get("document_id"), f"requests[{index}].document_id"
        )
        arrival = float(request.get("arrival_us", -1.0))
        deadline = float(request.get("deadline_us", -1.0))
        if (
            sample_id < 0
            or not math.isfinite(arrival)
            or not math.isfinite(deadline)
            or arrival < 0
            or deadline <= arrival
        ):
            raise ProtocolError(f"request {request_id} has invalid sample/arrival/deadline")
        if request_id in request_ids or sample_id in sample_ids:
            raise ProtocolError("request_id and sample_id must be unique")
        request_ids.add(request_id)
        sample_ids.add(sample_id)
        document_ids.add(document_id)
        if run_class == "formal":
            document_hash = _require_resolved(
                request.get("document_sha256"),
                f"requests[{index}].document_sha256",
            )
            token_hash = _require_resolved(
                request.get("prompt_token_ids_sha256"),
                f"requests[{index}].prompt_token_ids_sha256",
            )
            dataset_row_index = int(request.get("dataset_row_index", -1))
            source_csv_row = int(request.get("source_csv_row", -1))
            _require_resolved(
                request.get("source_timestamp_s"),
                f"requests[{index}].source_timestamp_s",
            )
            if document_hash != prompt_hash:
                raise ProtocolError(
                    f"request {request_id} document and prompt SHA-256 differ"
                )
            if len(token_hash) != 64 or int(request.get("prompt_token_count", 0)) <= 0:
                raise ProtocolError(f"request {request_id} token contract is invalid")
            if dataset_row_index < 0 or source_csv_row < 2:
                raise ProtocolError(f"request {request_id} source row contract is invalid")
            if dataset_row_index in dataset_row_indices:
                raise ProtocolError("formal dataset_row_index values must be unique")
            if source_csv_row in source_csv_rows:
                raise ProtocolError("formal source_csv_row values must be unique")
            dataset_row_indices.add(dataset_row_index)
            source_csv_rows.add(source_csv_row)
    audit_ids = manifest.get("serial_audit_request_ids")
    if not isinstance(audit_ids, list) or not audit_ids:
        raise ProtocolError("serial_audit_request_ids must be frozen and non-empty")
    if len(set(str(value) for value in audit_ids)) != len(audit_ids):
        raise ProtocolError("serial_audit_request_ids contains duplicates")
    missing_audit = set(str(value) for value in audit_ids) - request_ids
    if missing_audit:
        raise ProtocolError(f"serial audit references unknown requests: {sorted(missing_audit)}")
    if int(manifest.get("seed", -1)) < 0:
        raise ProtocolError("seed must be non-negative")
    trace_hash = _require_resolved(
        scheduler.get("arrival_trace_sha256"), "scheduler.arrival_trace_sha256"
    )
    if trace_hash != _arrival_trace_sha256(
        [_require_mapping(value, "request") for value in requests]
    ):
        raise ProtocolError("arrival trace SHA-256 does not match frozen request timings")
    if run_class == "formal":
        arrival_source = _require_mapping(
            scheduler.get("arrival_source"), "scheduler.arrival_source"
        )
        if arrival_source.get("kind") != "real_world_llm_serving_trace":
            raise ProtocolError("formal arrivals must come from a real-world serving trace")
        for key in (
            "repository",
            "revision",
            "path",
            "git_blob_oid",
            "sha256",
            "selection_rule",
            "deadline_policy",
        ):
            _require_resolved(arrival_source.get(key), f"arrival_source.{key}")
        tokenizer = _require_mapping(manifest.get("tokenizer"), "tokenizer")
        for key in ("revision", "class", "files_sha256"):
            _require_resolved(tokenizer.get(key), f"tokenizer.{key}")
    if run_class == "formal" and len(document_ids) != len(requests):
        raise ProtocolError("formal workload requires one unique document per request")
    return manifest


def load_preregistration(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read preregistration: {exc}") from exc
    preregistration = dict(_require_mapping(raw, "preregistration"))
    if preregistration.get("schema") != "bcrd-gate0-continuous-decode-prereg-v1":
        raise ProtocolError("unexpected Gate 0-A preregistration schema")
    return preregistration


def validate_formal_contract(
    manifest: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    *,
    preregistration_sha256: str,
    preregistration_path: str | Path | None = None,
    canonical_preregistration_path: str | Path | None = None,
    committed_preregistration_sha256: str | None = None,
) -> None:
    """Bind one formal producer cell to the authorized preregistration bytes."""

    if manifest.get("run_class") != "formal":
        return
    if preregistration_path is None or canonical_preregistration_path is None:
        raise ProtocolError("formal execution requires the canonical preregistration path")
    actual_path = Path(preregistration_path).resolve()
    canonical_path = Path(canonical_preregistration_path).resolve()
    if actual_path != canonical_path:
        raise ProtocolError("formal execution rejects a non-canonical preregistration path")
    if committed_preregistration_sha256 is None:
        raise ProtocolError("canonical preregistration is not bound to the executing git commit")
    if preregistration_sha256 != committed_preregistration_sha256:
        raise ProtocolError("canonical preregistration bytes differ from the executing git commit")
    if preregistration.get("formal_execution_authorized") is not True:
        raise ProtocolError("formal execution is not authorized by the preregistration")
    if preregistration.get("formal_blockers"):
        raise ProtocolError("formal preregistration still lists unresolved blockers")
    if int(manifest.get("seed", -1)) != int(preregistration.get("seed", -2)):
        raise ProtocolError("formal workload seed differs from preregistration")
    generation = _require_mapping(manifest.get("generation"), "generation")
    frozen_generation = _require_mapping(preregistration.get("generation"), "prereg generation")
    if generation.get("mode") != frozen_generation.get("mode") or int(
        generation.get("max_decode_steps", -1)
    ) != int(frozen_generation.get("max_decode_steps", -2)):
        raise ProtocolError("formal generation parameters differ from preregistration")
    scheduler = _require_mapping(manifest.get("scheduler"), "scheduler")
    frozen_scheduler = _require_mapping(preregistration.get("scheduler"), "prereg scheduler")
    if int(scheduler.get("max_batch_size", -1)) != int(
        frozen_scheduler.get("max_batch_size", -2)
    ):
        raise ProtocolError("formal scheduler parameters differ from preregistration")
    frozen_trace_hash = _require_resolved(
        frozen_scheduler.get("arrival_trace_sha256"),
        "prereg scheduler.arrival_trace_sha256",
    )
    if str(scheduler.get("arrival_trace_sha256", "")) != frozen_trace_hash:
        raise ProtocolError("formal arrival trace differs from preregistration")
    if scheduler.get("arrival_source") != frozen_scheduler.get("arrival_source"):
        raise ProtocolError("formal arrival provenance differs from preregistration")
    dataset = _require_mapping(manifest.get("dataset"), "dataset")
    frozen_dataset = _require_mapping(preregistration.get("dataset"), "prereg dataset")
    for key in (
        "id",
        "config",
        "split",
        "revision",
        "arrow_sha256",
        "fingerprint",
        "selection_rule",
        "prompt_policy",
    ):
        if str(dataset.get(key, "")) != str(frozen_dataset.get(key, "")):
            raise ProtocolError(f"formal dataset.{key} differs from preregistration")
    if int(manifest.get("max_prompt_tokens", -1)) != int(
        frozen_dataset.get("max_prompt_tokens", -2)
    ):
        raise ProtocolError("formal max_prompt_tokens differs from preregistration")
    expected_requests = int(frozen_dataset.get("documents_per_model", -1))
    requests = manifest.get("requests")
    if not isinstance(requests, list) or len(requests) != expected_requests:
        raise ProtocolError("formal request count differs from preregistration")
    audit_ids = {str(value) for value in manifest.get("serial_audit_request_ids", [])}
    request_ids = {str(value["request_id"]) for value in requests}
    expected_audit = int(
        _require_mapping(preregistration.get("acceptance"), "prereg acceptance").get(
            "serial_audit_requests_per_cell", -1
        )
    )
    if audit_ids != request_ids or len(audit_ids) != expected_audit:
        raise ProtocolError("formal serial audit must cover every frozen request")
    model = _require_mapping(manifest.get("model"), "model")
    frozen_models = preregistration.get("models")
    if not isinstance(frozen_models, list):
        raise ProtocolError("preregistration models must be a list")
    candidates = [item for item in frozen_models if item.get("id") == model.get("id")]
    if len(candidates) != 1:
        raise ProtocolError("formal model is not one preregistered cell")
    frozen_model = candidates[0]
    for key in ("key", "id", "revision", "dtype"):
        if str(model.get(key, "")) != str(frozen_model.get(key, "")):
            raise ProtocolError(f"formal model.{key} differs from preregistration")
    if str(model.get("tokenizer_revision", "")) != str(frozen_model.get("revision", "")):
        raise ProtocolError("formal tokenizer revision must equal the frozen model revision")


def _git_state(repo_root: Path) -> dict[str, Any]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProtocolError(f"cannot bind capture to git state: {exc}") from exc
    return {"git_sha": sha, "working_tree_clean": not bool(status.strip())}


def _git_head_file_sha256(repo_root: Path, path: Path) -> str:
    """Return SHA-256 of one canonical file exactly as stored in HEAD."""

    try:
        relative = path.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ProtocolError("canonical preregistration must be inside the repository") from exc
    try:
        payload = subprocess.run(
            ["git", "show", f"HEAD:{relative.as_posix()}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProtocolError(
            "canonical preregistration is not tracked in the executing git commit"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def validate_formal_workload_source(
    manifest: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    *,
    repo_root: Path,
    workload_manifest_path: Path,
    workload_manifest_sha256: str,
    committed_workload_manifest_sha256: str | None = None,
) -> None:
    """Bind formal prompt/document bytes to one preregistered committed file."""

    if manifest.get("run_class") != "formal":
        return
    model = _require_mapping(manifest.get("model"), "model")
    model_key = _require_resolved(model.get("key"), "model.key")
    workloads = _require_mapping(
        preregistration.get("formal_workloads"), "prereg formal_workloads"
    )
    workload = _require_mapping(
        workloads.get(model_key), f"prereg formal_workloads.{model_key}"
    )
    relative_text = _require_resolved(
        workload.get("path"), f"prereg formal_workloads.{model_key}.path"
    )
    expected_relative = (
        Path("docs")
        / "ideas"
        / "bcrd"
        / "experiments"
        / "configs"
        / "workloads"
        / f"{model_key}.formal.json"
    )
    if Path(relative_text) != expected_relative:
        raise ProtocolError("formal workload path is not the canonical model path")
    canonical_path = (repo_root.resolve() / expected_relative).resolve()
    if workload_manifest_path.resolve() != canonical_path:
        raise ProtocolError("formal execution rejects a non-canonical workload manifest")
    expected_hash = _require_resolved(
        workload.get("sha256"), f"prereg formal_workloads.{model_key}.sha256"
    )
    if workload_manifest_sha256 != expected_hash:
        raise ProtocolError("formal workload bytes differ from the preregistered SHA-256")
    committed_hash = committed_workload_manifest_sha256
    if committed_hash is None:
        committed_hash = _git_head_file_sha256(repo_root, canonical_path)
    if committed_hash != expected_hash:
        raise ProtocolError("formal workload bytes differ from the executing git commit")


def validate_output_isolation(
    output_dir: Path,
    repo_root: Path,
    run_class: str,
    model_key: str,
) -> None:
    """Keep formal and development bundles in disjoint canonical roots."""

    formal_root = (repo_root / "artifacts" / "bcrd_gate0" / "formal").resolve()
    output = output_dir.resolve()
    if run_class != "formal":
        development_roots = {
            Path(tempfile.gettempdir()).resolve(),
            Path("/tmp").resolve(),
        }
        for development_root in development_roots:
            try:
                relative = output.relative_to(development_root)
            except ValueError:
                continue
            if len(relative.parts) == 1 and relative.name.startswith(
                "bcrd-gate0-smoke-"
            ):
                return
        allowed = ", ".join(sorted(str(path) for path in development_roots))
        raise ProtocolError(
            "development output must be a direct bcrd-gate0-smoke-* directory "
            f"below one of: {allowed}"
        )
    try:
        relative = output.relative_to(formal_root)
    except ValueError as exc:
        raise ProtocolError(
            "formal capture output must be below artifacts/bcrd_gate0/formal"
        ) from exc
    if len(relative.parts) != 2 or relative.parts[-1] != model_key:
        raise ProtocolError(
            "formal output must be <formal-root>/<run-id>/<model-key>"
        )


def _legacy_cache(cache: object) -> tuple[tuple[Any, Any], ...]:
    converter = getattr(cache, "to_legacy_cache", None)
    if callable(converter):
        return tuple(converter())
    if isinstance(cache, (tuple, list)):
        return tuple(cache)
    raise ProtocolError(f"unsupported cache type: {type(cache).__name__}")


def _dynamic_cache(legacy: tuple[tuple[Any, Any], ...]) -> object:
    try:
        from transformers import DynamicCache
    except ImportError as exc:
        raise ProtocolError("Transformers DynamicCache is required") from exc
    return DynamicCache.from_legacy_cache(legacy)


def _cache_length(cache: object) -> int:
    getter = getattr(cache, "get_seq_length", None)
    if callable(getter):
        return int(getter())
    legacy = _legacy_cache(cache)
    if not legacy:
        raise ProtocolError("cache contains no layers")
    return int(legacy[0][0].shape[-2])


def stack_left_padded_caches(caches: Sequence[object]) -> tuple[object, list[int], int]:
    """Merge batch-one caches after left padding their sequence dimension."""

    try:
        import torch
    except ImportError as exc:
        raise ProtocolError("cache stacking requires PyTorch") from exc
    if not caches:
        raise ProtocolError("cannot stack an empty cache list")
    legacies = [_legacy_cache(cache) for cache in caches]
    layer_count = len(legacies[0])
    if layer_count <= 0 or any(len(cache) != layer_count for cache in legacies):
        raise ProtocolError("cache layer closure failed")
    lengths = [int(cache[0][0].shape[-2]) for cache in legacies]
    if any(length <= 0 for length in lengths):
        raise ProtocolError("cache sequence lengths must be positive")
    max_length = max(lengths)
    stacked: list[tuple[Any, Any]] = []
    for layer_index in range(layer_count):
        keys: list[Any] = []
        values: list[Any] = []
        for cache, length in zip(legacies, lengths):
            key, value = cache[layer_index]
            if int(key.shape[0]) != 1 or int(value.shape[0]) != 1:
                raise ProtocolError("per-request cache must have batch size one")
            if int(key.shape[-2]) != length or int(value.shape[-2]) != length:
                raise ProtocolError("cache layer sequence lengths disagree")
            padding = max_length - length
            if padding:
                key_pad = key.new_zeros((*key.shape[:-2], padding, key.shape[-1]))
                value_pad = value.new_zeros((*value.shape[:-2], padding, value.shape[-1]))
                key = torch.cat((key_pad, key), dim=-2)
                value = torch.cat((value_pad, value), dim=-2)
            keys.append(key)
            values.append(value)
        stacked.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
    return _dynamic_cache(tuple(stacked)), lengths, max_length


def split_left_padded_cache(
    cache: object, *, prior_lengths: Sequence[int], prior_max_length: int
) -> list[object]:
    """Undo ``stack_left_padded_caches`` after one appended decode token."""

    legacy = _legacy_cache(cache)
    expected_total = prior_max_length + 1
    outputs: list[object] = []
    for batch_index, prior_length in enumerate(prior_lengths):
        new_length = int(prior_length) + 1
        start = expected_total - new_length
        layers: list[tuple[Any, Any]] = []
        for key, value in legacy:
            if int(key.shape[-2]) != expected_total or int(value.shape[-2]) != expected_total:
                raise ProtocolError("batched cache did not append exactly one decode position")
            layers.append(
                (
                    key[batch_index : batch_index + 1, :, start:, :].contiguous(),
                    value[batch_index : batch_index + 1, :, start:, :].contiguous(),
                )
            )
        result = _dynamic_cache(tuple(layers))
        if _cache_length(result) != new_length:
            raise ProtocolError("split cache length mismatch")
        outputs.append(result)
    return outputs


def _native_route_batches(
    output: object, *, expected_rows: int, config: object
) -> tuple[Mapping[str, Any], ...]:
    try:
        import torch
    except ImportError as exc:
        raise ProtocolError("native route extraction requires PyTorch") from exc
    router_logits = getattr(output, "router_logits", None)
    if not isinstance(router_logits, (tuple, list)) or not router_logits:
        raise ProtocolError("model did not return native router_logits")
    expected_layers = int(getattr(config, "num_hidden_layers", 0))
    if expected_layers <= 0 or len(router_logits) != expected_layers:
        raise ProtocolError(
            "native router layer closure failed: "
            f"observed={len(router_logits)}, expected={expected_layers}"
        )
    top_k = int(
        getattr(config, "num_experts_per_tok", 0)
        or getattr(config, "top_k", 0)
    )
    if top_k <= 0:
        raise ProtocolError("model config does not declare num_experts_per_tok")
    normalize = bool(getattr(config, "norm_topk_prob", False)) or str(
        getattr(config, "model_type", "")
    ) == "mixtral"
    batches: list[Mapping[str, Any]] = []
    for layer, logits in enumerate(router_logits):
        if logits is None:
            raise ProtocolError(f"router_logits layer {layer} is missing")
        if logits.ndim != 2 or int(logits.shape[0]) != expected_rows:
            raise ProtocolError(
                f"router layer {layer} rows={int(logits.shape[0])} != expected {expected_rows}"
            )
        probabilities = torch.softmax(logits.float(), dim=-1)
        weights, experts = torch.topk(probabilities, k=top_k, dim=-1)
        if normalize:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        batches.append(
            {
                "layer": layer,
                "selected_experts": experts.detach().cpu(),
                "routing_weights": weights.detach().cpu(),
            }
        )
    return tuple(batches)


def _sync_model(model: object) -> None:
    try:
        import torch
    except ImportError:
        return
    device = getattr(model, "device", None)
    if torch.cuda.is_available() and device is not None and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _timed_call(
    model: object,
    phase: str,
    batch_size: int,
    duration_provider: DurationProvider | None,
    **kwargs: Any,
) -> tuple[object, float]:
    _sync_model(model)
    start = time.perf_counter_ns()
    base_model = getattr(model, "model", None)
    lm_head = getattr(model, "lm_head", None)
    if base_model is None or lm_head is None:
        raise ProtocolError("native producer requires a causal LM with model and lm_head")
    kwargs.pop("return_dict", None)
    base_output = base_model(return_dict=True, **kwargs)
    output = SimpleNamespace(
        logits=lm_head(base_output.last_hidden_state),
        past_key_values=base_output.past_key_values,
        router_logits=base_output.router_logits,
    )
    _sync_model(model)
    elapsed = max((time.perf_counter_ns() - start) / 1000.0, 1e-3)
    if duration_provider is not None:
        elapsed = float(duration_provider(phase, batch_size))
        if elapsed <= 0:
            raise ProtocolError("duration provider must return positive microseconds")
    return output, elapsed


def _route_signature(
    batches: Sequence[Mapping[str, Any]], row_index: int
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    return tuple(
        (
            int(batch["layer"]),
            tuple(int(value) for value in batch["selected_experts"][row_index].tolist()),
        )
        for batch in batches
    )


def _decode_stop_reason(
    *,
    predicted_token_id: int,
    eos_token_id: int | None,
    completed_decode_steps: int,
    max_decode_steps: int,
) -> str | None:
    """Classify a completed decode step without hiding a terminal EOS."""

    if eos_token_id is not None and predicted_token_id == int(eos_token_id):
        return "eos"
    if completed_decode_steps >= max_decode_steps:
        return "max_decode_steps"
    return None


def _append_contributions(
    sink: list[Contribution],
    *,
    batches: Sequence[Mapping[str, Any]],
    states: Sequence[_ActiveRequest],
    model_key: str,
    batch_start_us: float,
    batch_end_us: float,
) -> None:
    for batch in batches:
        selected = batch["selected_experts"]
        weights = batch["routing_weights"]
        if int(selected.shape[0]) != len(states) or weights.shape != selected.shape:
            raise ProtocolError("batched router rows lost request identity")
        layer = int(batch["layer"])
        for request_index, state in enumerate(states):
            spec = state.spec
            event_id = f"{spec.request_id}:decode:{state.decode_step:06d}"
            for slot in range(int(selected.shape[1])):
                sink.append(
                    Contribution(
                        model=model_key,
                        phase="decode",
                        request_id=spec.request_id,
                        sample_id=spec.sample_id,
                        arrival_us=spec.arrival_us,
                        deadline_us=spec.deadline_us,
                        layer=layer,
                        token_position=state.prompt_length + state.decode_step,
                        rank=slot + 1,
                        expert_id=int(selected[request_index, slot].item()),
                        gate_weight=float(weights[request_index, slot].item()),
                        src_replica=0,
                        input_event_id=event_id,
                        token_id=int(state.next_token.item()),
                        decode_step=state.decode_step,
                        layer_id=layer,
                        topk_slot=slot,
                        source_rank=0,
                        target_replica=-1,
                        document_id=spec.document_id,
                        request_arrival_us=spec.arrival_us,
                        layer_ready_us=batch_start_us,
                        route_end_us=batch_end_us,
                        legal_replica_set=(0,),
                    )
                )


def _pad_decode_inputs(
    states: Sequence[_ActiveRequest],
) -> tuple[Any, Any, Any, object, list[int], int]:
    try:
        import torch
    except ImportError as exc:
        raise ProtocolError("continuous decode requires PyTorch") from exc
    cache, lengths, max_length = stack_left_padded_caches([state.cache for state in states])
    masks: list[Any] = []
    tokens: list[Any] = []
    positions: list[int] = []
    for state, length in zip(states, lengths):
        if (
            int(state.attention_mask.shape[0]) != 1
            or int(state.attention_mask.shape[1]) != length
        ):
            raise ProtocolError("request attention mask and cache length disagree")
        padding = max_length - length
        prefix = state.attention_mask.new_zeros((1, padding))
        masks.append(
            torch.cat(
                (prefix, state.attention_mask, state.attention_mask.new_ones((1, 1))),
                dim=1,
            )
        )
        tokens.append(state.next_token)
        positions.append(int(state.attention_mask.long().sum().item()))
    return (
        torch.cat(tokens, dim=0),
        torch.cat(masks, dim=0),
        torch.tensor(positions, dtype=torch.long, device=tokens[0].device).unsqueeze(1),
        cache,
        lengths,
        max_length,
    )


def run_continuous_decode(
    model: object,
    requests: Sequence[ContinuousRequest],
    *,
    model_key: str,
    max_decode_steps: int,
    max_batch_size: int,
    eos_token_id: int | None,
    serial_audit_request_ids: Sequence[str],
    duration_provider: DurationProvider | None = None,
) -> ContinuousCapture:
    """Execute arrival-ordered prefills and mutable batched cached decode."""

    try:
        import torch
    except ImportError as exc:
        raise ProtocolError("continuous decode requires PyTorch") from exc
    if max_decode_steps <= 0 or max_batch_size <= 0:
        raise ProtocolError("decode steps and max batch size must be positive")
    ordered = sorted(requests, key=lambda item: (item.arrival_us, item.request_id))
    if not ordered or len({item.request_id for item in ordered}) != len(ordered):
        raise ProtocolError("continuous requests must be non-empty with unique IDs")
    audit_ids = tuple(str(value) for value in serial_audit_request_ids)
    if not audit_ids or not set(audit_ids).issubset({item.request_id for item in ordered}):
        raise ProtocolError("serial audit request IDs must be frozen and present")

    result = ContinuousCapture()
    pending = list(ordered)
    active: list[_ActiveRequest] = []
    clock_us = float(pending[0].arrival_us)
    for spec in ordered:
        result.request_rows[spec.request_id] = {
            "request_id": spec.request_id,
            "sample_id": spec.sample_id,
            "document_id": spec.document_id,
            "arrival_us": spec.arrival_us,
            "deadline_us": spec.deadline_us,
            "prompt_tokens": int(spec.input_ids.shape[1]),
            "prompt_token_ids_sha256": _prompt_token_ids_sha256(spec.input_ids),
            "prefill_start_us": None,
            "prefill_end_us": None,
            "completion_us": None,
            "stop_reason": None,
            "steps": [],
        }

    while pending or active:
        if not active and pending and pending[0].arrival_us > clock_us:
            clock_us = float(pending[0].arrival_us)

        while pending and pending[0].arrival_us <= clock_us:
            spec = pending.pop(0)
            row = result.request_rows[spec.request_id]
            row["prefill_start_us"] = clock_us
            with torch.inference_mode():
                prefill, elapsed = _timed_call(
                    model,
                    "prefill",
                    1,
                    duration_provider,
                    input_ids=spec.input_ids,
                    attention_mask=spec.attention_mask,
                    use_cache=True,
                    output_router_logits=True,
                    return_dict=True,
                )
            clock_us += elapsed
            row["prefill_end_us"] = clock_us
            cache = getattr(prefill, "past_key_values", None)
            logits = getattr(prefill, "logits", None)
            if cache is None or logits is None:
                raise ProtocolError(f"request {spec.request_id} prefill returned no cache/logits")
            prompt_length = int(spec.input_ids.shape[1])
            if _cache_length(cache) != prompt_length:
                raise ProtocolError(f"request {spec.request_id} prefill cache length mismatch")
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
                row["completion_us"] = clock_us
                row["stop_reason"] = "eos_before_decode"
            else:
                active.append(
                    _ActiveRequest(
                        spec=spec,
                        cache=cache,
                        attention_mask=spec.attention_mask,
                        next_token=next_token,
                        prompt_length=prompt_length,
                    )
                )

        if not active:
            continue
        active.sort(key=lambda item: (item.spec.arrival_us, item.spec.request_id))
        batch = active[:max_batch_size]
        (
            input_ids,
            attention_mask,
            position_ids,
            cache,
            prior_lengths,
            prior_max,
        ) = _pad_decode_inputs(batch)
        start_us = clock_us
        with torch.inference_mode():
            output, elapsed = _timed_call(
                model,
                "decode",
                len(batch),
                duration_provider,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=torch.tensor(
                    [prior_max], dtype=torch.long, device=input_ids.device
                ),
                past_key_values=cache,
                use_cache=True,
                output_router_logits=True,
                return_dict=True,
            )
        clock_us += elapsed
        batches = _native_route_batches(
            output, expected_rows=len(batch), config=getattr(model, "config")
        )
        _append_contributions(
            result.contributions,
            batches=batches,
            states=batch,
            model_key=model_key,
            batch_start_us=start_us,
            batch_end_us=clock_us,
        )
        logits = getattr(output, "logits", None)
        output_cache = getattr(output, "past_key_values", None)
        if logits is None or output_cache is None:
            raise ProtocolError("decode batch returned no logits/cache")
        split_caches = split_left_padded_cache(
            output_cache, prior_lengths=prior_lengths, prior_max_length=prior_max
        )
        predicted = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        result.batch_rows.append(
            {
                "batch_index": len(result.batch_rows),
                "start_us": start_us,
                "end_us": clock_us,
                "batch_size": len(batch),
                "active_request_ids": [state.spec.request_id for state in active],
                "pending_request_count": len(pending),
                "request_ids": [state.spec.request_id for state in batch],
                "decode_steps": [state.decode_step for state in batch],
                "prior_cache_lengths": prior_lengths,
                "left_padding": [prior_max - length for length in prior_lengths],
            }
        )
        for index, state in enumerate(batch):
            row = result.request_rows[state.spec.request_id]
            signature = _route_signature(batches, index)
            input_token = int(state.next_token.item())
            predicted_token = int(predicted[index].item())
            row["steps"].append(
                {
                    "decode_step": state.decode_step,
                    "input_token_id": input_token,
                    "predicted_next_token_id": predicted_token,
                    "batch_index": len(result.batch_rows) - 1,
                    "route_signature": [
                        {"layer": layer, "experts": list(experts)}
                        for layer, experts in signature
                    ],
                }
            )
            state.cache = split_caches[index]
            state.attention_mask = torch.cat(
                (state.attention_mask, state.attention_mask.new_ones((1, 1))), dim=1
            )
            state.decode_step += 1
            state.next_token = predicted[index : index + 1]
            stop_reason = _decode_stop_reason(
                predicted_token_id=predicted_token,
                eos_token_id=eos_token_id,
                completed_decode_steps=state.decode_step,
                max_decode_steps=max_decode_steps,
            )
            if stop_reason is not None:
                row["completion_us"] = clock_us
                row["stop_reason"] = stop_reason
                active.remove(state)

    expected_events = {
        request_id: [
            f"{request_id}:decode:{int(step['decode_step']):06d}"
            for step in row["steps"]
        ]
        for request_id, row in result.request_rows.items()
        if row["steps"]
    }
    identity = validate_identity_conservation(
        result.contributions, expected_input_events=expected_events
    )
    result.serial_audit = run_serial_audit(
        model,
        requests=ordered,
        request_rows=result.request_rows,
        request_ids=audit_ids,
        eos_token_id=eos_token_id,
        duration_provider=duration_provider,
    )
    result.serial_audit["identity_summary"] = identity
    return result


def run_serial_audit(
    model: object,
    *,
    requests: Sequence[ContinuousRequest],
    request_rows: Mapping[str, Mapping[str, Any]],
    request_ids: Sequence[str],
    eos_token_id: int | None,
    duration_provider: DurationProvider | None = None,
) -> dict[str, Any]:
    """Re-execute a frozen request subset serially and require route/token parity."""

    try:
        import torch
    except ImportError as exc:
        raise ProtocolError("serial audit requires PyTorch") from exc
    by_id = {item.request_id: item for item in requests}
    audited_steps = 0
    for request_id in request_ids:
        spec = by_id[request_id]
        expected_steps = list(request_rows[request_id]["steps"])
        with torch.inference_mode():
            prefill, _ = _timed_call(
                model,
                "serial_prefill_audit",
                1,
                duration_provider,
                input_ids=spec.input_ids,
                attention_mask=spec.attention_mask,
                use_cache=True,
                output_router_logits=True,
                return_dict=True,
            )
        cache = getattr(prefill, "past_key_values", None)
        logits = getattr(prefill, "logits", None)
        if cache is None or logits is None:
            raise ProtocolError("serial audit prefill returned no cache/logits")
        attention_mask = spec.attention_mask
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        for expected in expected_steps:
            if int(next_token.item()) != int(expected["input_token_id"]):
                raise ProtocolError(f"serial token input mismatch for {request_id}")
            prior_length = _cache_length(cache)
            attention_mask = torch.cat(
                (attention_mask, attention_mask.new_ones((1, 1))), dim=1
            )
            position_ids = attention_mask.long().cumsum(-1)[:, -1:] - 1
            with torch.inference_mode():
                output, _ = _timed_call(
                    model,
                    "serial_decode_audit",
                    1,
                    duration_provider,
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_position=torch.tensor(
                        [prior_length], dtype=torch.long, device=next_token.device
                    ),
                    past_key_values=cache,
                    use_cache=True,
                    output_router_logits=True,
                    return_dict=True,
                )
            cache = getattr(output, "past_key_values", None)
            logits = getattr(output, "logits", None)
            if cache is None or logits is None or _cache_length(cache) != prior_length + 1:
                raise ProtocolError(f"serial cache closure failed for {request_id}")
            batches = _native_route_batches(
                output, expected_rows=1, config=getattr(model, "config")
            )
            observed_signature = [
                {"layer": layer, "experts": list(experts)}
                for layer, experts in _route_signature(batches, 0)
            ]
            if observed_signature != expected["route_signature"]:
                raise ProtocolError(f"serial route identity mismatch for {request_id}")
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            if int(next_token.item()) != int(expected["predicted_next_token_id"]):
                raise ProtocolError(f"serial greedy token mismatch for {request_id}")
            audited_steps += 1
        if expected_steps and eos_token_id is not None:
            expected_stop = request_rows[request_id]["stop_reason"]
            if expected_stop == "eos" and int(next_token.item()) != int(eos_token_id):
                raise ProtocolError(f"serial EOS stop mismatch for {request_id}")
        elif not expected_steps and request_rows[request_id]["stop_reason"] == "eos_before_decode":
            if eos_token_id is None or int(next_token.item()) != int(eos_token_id):
                raise ProtocolError(f"serial pre-decode EOS mismatch for {request_id}")
    return {
        "status": "PASS",
        "requests": len(request_ids),
        "steps": audited_steps,
        "token_match_fraction": 1.0,
        "route_identity_match_fraction": 1.0,
        "reference_type": "same-model serial cached-decode engineering equivalence",
        "scientific_ground_truth": False,
    }


def _prompt_token_ids_sha256(input_ids: object) -> str:
    values = getattr(input_ids, "tolist", None)
    if not callable(values):
        raise ProtocolError("tokenizer input_ids must support tolist")
    rows = values()
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], list):
        raise ProtocolError("tokenizer input_ids must have shape [1, sequence]")
    payload = [int(value) for value in rows[0]]
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_tokenizer_contract(
    manifest: Mapping[str, Any], tokenizer: object
) -> None:
    if manifest.get("run_class") != "formal":
        return
    frozen = _require_mapping(manifest.get("tokenizer"), "tokenizer")
    observed = {
        "revision": str(_require_mapping(manifest.get("model"), "model").get(
            "tokenizer_revision", ""
        )),
        "class": type(tokenizer).__name__,
        "vocab_size": int(getattr(tokenizer, "vocab_size", -1)),
        "length": int(len(tokenizer)),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "truncation_side": str(getattr(tokenizer, "truncation_side", "")),
    }
    for key, value in observed.items():
        if frozen.get(key) != value:
            raise ProtocolError(f"loaded tokenizer.{key} differs from frozen manifest")


def _prepare_requests(
    manifest: Mapping[str, Any], tokenizer: object, device: object
) -> list[ContinuousRequest]:
    requests: list[ContinuousRequest] = []
    _validate_tokenizer_contract(manifest, tokenizer)
    max_prompt_tokens = int(manifest.get("max_prompt_tokens", 0))
    if max_prompt_tokens <= 0:
        raise ProtocolError("max_prompt_tokens must be positive")
    for raw in manifest["requests"]:
        encoded = tokenizer(
            str(raw["prompt"]),
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_tokens,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)
        else:
            attention_mask = attention_mask.to(device)
        if manifest.get("run_class") == "formal":
            expected_count = int(raw.get("prompt_token_count", -1))
            expected_hash = str(raw.get("prompt_token_ids_sha256", ""))
            if int(input_ids.shape[1]) != expected_count:
                raise ProtocolError(
                    f"request {raw['request_id']} prompt token count drifted"
                )
            if _prompt_token_ids_sha256(input_ids) != expected_hash:
                raise ProtocolError(
                    f"request {raw['request_id']} prompt token IDs drifted"
                )
        requests.append(
            ContinuousRequest(
                request_id=str(raw["request_id"]),
                sample_id=int(raw["sample_id"]),
                document_id=str(raw["document_id"]),
                arrival_us=float(raw["arrival_us"]),
                deadline_us=float(raw["deadline_us"]),
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        )
    return requests


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _environment(
    torch_module: object,
    transformers_module: object,
    git_state: Mapping[str, Any],
) -> dict[str, Any]:
    cuda = bool(torch_module.cuda.is_available())
    environment: dict[str, Any] = {
        **git_state,
        "python": sys.version,
        "torch": getattr(torch_module, "__version__", "unknown"),
        "transformers": getattr(transformers_module, "__version__", "unknown"),
        "cuda_available": cuda,
        "cuda_version": getattr(getattr(torch_module, "version", None), "cuda", None),
        "gpu_count": int(torch_module.cuda.device_count()) if cuda else 0,
        "gpus": [],
    }
    if cuda:
        environment["gpus"] = [
            {
                "index": index,
                "name": torch_module.cuda.get_device_name(index),
                "capability": list(torch_module.cuda.get_device_capability(index)),
            }
            for index in range(torch_module.cuda.device_count())
        ]
    return environment


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    status_path = output_dir / "RUN_STATUS.json"
    write_json(status_path, {"status": "INCOMPLETE", "reason": "capture not finished"})
    completed_payload: dict[str, Any] | None = None
    try:
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise ProtocolError("continuous capture requires PyTorch and Transformers") from exc
        manifest_path = Path(args.workload_manifest).resolve()
        manifest = load_workload_manifest(manifest_path)
        manifest_hash = sha256_file(manifest_path)
        repo_root = next(
            parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()
        )
        git_state = _git_state(repo_root)
        run_class = str(manifest["run_class"])
        preregistration_path = Path(args.preregistration).resolve()
        preregistration = load_preregistration(preregistration_path)
        preregistration_hash = sha256_file(preregistration_path)
        canonical_preregistration_path = (
            repo_root
            / "docs"
            / "ideas"
            / "bcrd"
            / "experiments"
            / "configs"
            / "gate0_continuous_decode_v1.json"
        )
        validate_formal_contract(
            manifest,
            preregistration,
            preregistration_sha256=preregistration_hash,
            preregistration_path=preregistration_path,
            canonical_preregistration_path=canonical_preregistration_path,
            committed_preregistration_sha256=(
                _git_head_file_sha256(repo_root, canonical_preregistration_path)
                if run_class == "formal"
                else None
            ),
        )
        validate_formal_workload_source(
            manifest,
            preregistration,
            repo_root=repo_root,
            workload_manifest_path=manifest_path,
            workload_manifest_sha256=manifest_hash,
        )
        model_key = str(manifest["model"].get("key") or manifest["model"]["id"])
        validate_output_isolation(output_dir, repo_root, run_class, model_key)
        if run_class == "formal":
            if not torch.cuda.is_available():
                raise ProtocolError("formal continuous capture requires CUDA")
            if not git_state["working_tree_clean"]:
                raise ProtocolError("formal continuous capture requires a clean working tree")
            frozen_environment = _require_mapping(
                preregistration.get("formal_environment"),
                "prereg formal_environment",
            )
            if frozen_environment.get("device") != "cuda":
                raise ProtocolError("formal environment device must be cuda")
            expected_gpu_count = int(
                frozen_environment.get("visible_gpu_count", -1)
            )
            if int(torch.cuda.device_count()) != expected_gpu_count:
                raise ProtocolError("visible CUDA GPU count differs from preregistration")
            required_gpu_name = _require_resolved(
                frozen_environment.get("required_gpu_name_substring"),
                "prereg formal_environment.required_gpu_name_substring",
            )
            observed_gpu_names = [
                str(torch.cuda.get_device_name(index))
                for index in range(torch.cuda.device_count())
            ]
            if any(required_gpu_name not in name for name in observed_gpu_names):
                raise ProtocolError("formal execution requires the frozen RTX 5090 target")
            torch_version_prefix = _require_resolved(
                frozen_environment.get("required_torch_version_prefix"),
                "prereg formal_environment.required_torch_version_prefix",
            )
            if not str(getattr(torch, "__version__", "")).startswith(
                torch_version_prefix
            ):
                raise ProtocolError("formal PyTorch version differs from preregistration")
            transformers_version = _require_resolved(
                frozen_environment.get("required_transformers_version"),
                "prereg formal_environment.required_transformers_version",
            )
            if str(getattr(transformers, "__version__", "")) != transformers_version:
                raise ProtocolError(
                    "formal Transformers version differs from preregistration"
                )
            if len(manifest["requests"]) != 128:
                raise ProtocolError("formal Gate 0-A cell requires exactly 128 frozen requests")
        else:
            if not torch.cuda.is_available() and not args.allow_cpu:
                raise ProtocolError("CPU development capture requires --allow-cpu")

        seed = int(manifest["seed"])
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        shared = repo_root / "experiments" / "shared"
        sys.path.insert(0, str(shared))
        from modeling import load_model, load_tokenizer

        model_spec = manifest["model"]
        tokenizer = load_tokenizer(
            str(model_spec["id"]),
            local_files_only=args.offline,
            revision=str(model_spec["tokenizer_revision"]),
        )
        model, load_seconds = load_model(
            str(model_spec["id"]),
            dtype_name=str(model_spec["dtype"]),
            local_files_only=args.offline,
            revision=str(model_spec["revision"]),
        )
        model.eval()
        requests = _prepare_requests(manifest, tokenizer, model.device)
        capture = run_continuous_decode(
            model,
            requests,
            model_key=model_key,
            max_decode_steps=int(manifest["generation"]["max_decode_steps"]),
            max_batch_size=int(manifest["scheduler"]["max_batch_size"]),
            eos_token_id=tokenizer.eos_token_id,
            serial_audit_request_ids=[str(value) for value in manifest["serial_audit_request_ids"]],
        )

        routes_path = output_dir / "routes.csv"
        batches_path = output_dir / "decode_batches.jsonl"
        requests_path = output_dir / "request_ledger.jsonl"
        snapshot_path = output_dir / "workload_manifest.json"
        preregistration_snapshot_path = output_dir / "preregistration.json"
        environment_path = output_dir / "environment.json"
        audit_path = output_dir / "serial_audit.json"
        write_routes(routes_path, capture.contributions)
        _write_jsonl(batches_path, capture.batch_rows)
        _write_jsonl(requests_path, list(capture.request_rows.values()))
        write_json(snapshot_path, manifest)
        write_json(preregistration_snapshot_path, preregistration)
        write_json(environment_path, _environment(torch, transformers, git_state))
        write_json(audit_path, capture.serial_audit)

        completed = sum(
            row["completion_us"] is not None for row in capture.request_rows.values()
        )
        batch_histogram: dict[str, int] = {}
        for row in capture.batch_rows:
            key = str(row["batch_size"])
            batch_histogram[key] = batch_histogram.get(key, 0) + 1
        active_sets = [tuple(row["active_request_ids"]) for row in capture.batch_rows]
        active_set_changes = sum(
            left != right for left, right in zip(active_sets, active_sets[1:])
        )
        files = [
            routes_path,
            batches_path,
            requests_path,
            snapshot_path,
            preregistration_snapshot_path,
            environment_path,
            audit_path,
        ]
        stop_reason_counts: dict[str, int] = {}
        for row in capture.request_rows.values():
            reason = str(row["stop_reason"] or "missing")
            stop_reason_counts[reason] = stop_reason_counts.get(reason, 0) + 1
        router_invocations = {
            (
                contribution.request_id,
                contribution.decode_step,
                contribution.layer_id,
            )
            for contribution in capture.contributions
        }
        complete = {
            "schema": "bcrd-continuous-capture-complete-v1",
            "status": "CAPTURE_COMPLETE",
            "run_class": run_class,
            "formal_capture_candidate": run_class == "formal",
            "producer_formal_eligible": False,
            "scientific_result_eligible": False,
            "gate0_complete": False,
            "gate1_authorized": False,
            "qualification_scope": "producer candidate; independent audit still required",
            "model_load_seconds": load_seconds,
            "workload_manifest_sha256": manifest_hash,
            "preregistration_sha256": preregistration_hash,
            "expected_requests": len(requests),
            "admitted_requests": len(capture.request_rows),
            "completed_requests": completed,
            "failed_requests": len(requests) - completed,
            "filtered_requests": 0,
            "stop_reason_counts": stop_reason_counts,
            "decode_steps": sum(len(row["steps"]) for row in capture.request_rows.values()),
            "router_invocations": len(router_invocations),
            "identity_closed_router_invocations": len(router_invocations),
            "route_contributions": len(capture.contributions),
            "identity_closed_contributions": int(
                capture.serial_audit["identity_summary"]["contributions"]
            ),
            "serial_audit": capture.serial_audit,
            "decode_batch_size_histogram": batch_histogram,
            "active_set_changes": active_set_changes,
            "maximum_observed_decode_batch_size": max(
                (int(row["batch_size"]) for row in capture.batch_rows), default=0
            ),
            "timing_boundary": "model-call boundaries only; not layer/stage service timing",
            "formal_blockers_outside_gate0_a": [
                "no dispatch/expert/combine stage ledger",
                "no expert/dtype-complete service surface",
                "no full-path denominator",
                "Gate 0 completeness has not been audited",
            ],
            "files": {path.name: sha256_file(path) for path in files},
        }
        if completed != len(requests):
            raise ProtocolError("not all frozen requests completed")
        if complete["maximum_observed_decode_batch_size"] < 2:
            raise ProtocolError("frozen arrival replay produced no decode batch larger than one")
        if complete["active_set_changes"] < 1:
            raise ProtocolError("frozen arrival replay did not change the decode active set")
        write_json(
            status_path,
            {
                "status": "COMPLETE",
                "required_sentinel": "CAPTURE_COMPLETE.json",
            },
        )
        # The sentinel is deliberately the final filesystem mutation. A
        # consumer must require both RUN_STATUS=COMPLETE and this file.
        write_json(output_dir / "CAPTURE_COMPLETE.json", complete)
        completed_payload = complete
    except BaseException as exc:
        write_json(
            status_path,
            {
                "status": "INCOMPLETE",
                "error_type": type(exc).__name__,
                "reason": str(exc),
                "scientific_result_eligible": False,
            },
        )
        raise
    if completed_payload is not None:
        print(json.dumps(completed_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

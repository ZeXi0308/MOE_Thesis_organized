"""Frozen, route-only capture for the Route-Fidelity P0-B experiment.

The command deliberately has no knobs for model, dataset, split, sample range,
sequence length, dtype, or EP size.  Those values must be frozen in one JSON
file before either phase is opened.  A minimal accepted configuration is::

    {
      "schema_version": 1,
      "dataset": {
        "name": "wikitext2_docs",
        "split": "train",
        "seed": 20260717
      },
      "model": {
        "name": "allenai/OLMoE-1B-7B-0924",
        "revision": null
      },
      "dtype": "bfloat16",
      "seq_len": 256,
      "ep_size": 8,
      "compute_seed": 0,
      "historical_exclusion": {
        "registry": {
          "path": "experiments/idea_a_mac/outputs/creditreduce_p0_2026-07-17/frozen_historical_exclusion_registry.json",
          "sha256": "<sha256 of the registry file>"
        },
        "hashes": [],
        "combined_hashes_sha256": "<hash_lines(sorted(unique hashes))>"
      },
      "phases": {
        "calibration": {
          "offset": 192,
          "n": 32,
          "hash_of_hashes": "264360550fe1caa24f97411155812a746290ad5742f1fe9842555427c7a4ba2c"
        },
        "sealed": {
          "offset": 224,
          "n": 64,
          "hash_of_hashes": "255af39aa7bb32c6f972195b0399a3c59b90af0460c11f93f7f780fb07391b04"
        }
      }
    }

``historical_exclusion.hashes`` is useful for documents opened after an older
registry was frozen.  Registry inputs must contain ``unique_hashes``; JSON/CSV
manifest inputs may instead be supplied through ``manifests``.  Every external
input must be pinned by its file SHA-256, and the combined exclusion set must
also be pinned.  Relative paths are resolved against the current working
directory first and then against the frozen JSON's directory.

The only persisted model observation is the exact, full-precision top-k route.
No route statistic, synthetic workload, quality metric, or latency estimate is
computed here.  ``home_rank`` is assigned independently of dataset order by
sorting document SHA-256 values and round-robin assigning the sorted positions.
"""


# --- shared-lib bootstrap (auto) ---
import sys
from pathlib import Path as _Path

def _ensure_shared_on_path() -> None:
    here = _Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        cand = p / "experiments" / "shared"
        if (cand / "capture_moe.py").exists():
            s = str(cand)
            if s not in sys.path:
                sys.path.insert(0, s)
            return
        if (p / "capture_moe.py").exists():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
            return

_ensure_shared_on_path()
del _ensure_shared_on_path, _Path
# --- end bootstrap ---

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, TextIO

import torch

from capture_moe import MoeRecorder, patch_mixtral_moe
from modeling import load_model, load_tokenizer
from prompts import get_prompts


SCHEMA_VERSION = 1
PHASES = ("calibration", "sealed")
ROUTE_COLUMNS = (
    "sample_id",
    "layer",
    "token_position",
    "rank",
    "expert_id",
    "gate_weight",
    "home_rank",
)
HEX64 = re.compile(r"[0-9a-fA-F]{64}")
MODEL_REVISION = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_lines(values: Iterable[str]) -> str:
    return sha256_bytes(("\n".join(values) + "\n").encode("utf-8"))


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a 64-character SHA-256 hex string")
    return value.lower()


def _model_revision(value: object, label: str) -> str:
    if not isinstance(value, str) or MODEL_REVISION.fullmatch(value) is None:
        raise ValueError(f"{label} must be an exact 40- or 64-hex commit id")
    return value.lower()


def _positive_int(value: object, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    lower = 0 if allow_zero else 1
    if value < lower:
        comparison = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be {comparison}, got {value}")
    return value


def _require_dict(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _resolve_input_path(raw_path: object, frozen_path: Path, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise TypeError(f"{label}.path must be a non-empty string")
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        candidate = path
    elif path.exists():
        candidate = path
    else:
        candidate = frozen_path.parent / path
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} does not exist: {candidate}")
    return candidate


def _phase_spec(raw: dict[str, Any], phase: str) -> dict[str, Any]:
    phases = _require_dict(raw.get("phases"), "phases")
    spec = _require_dict(phases.get(phase), f"phases.{phase}")
    offset = _positive_int(
        spec.get("offset"), f"phases.{phase}.offset", allow_zero=True
    )
    count = _positive_int(spec.get("n"), f"phases.{phase}.n")
    expected_hash = _sha256(
        spec.get("hash_of_hashes"), f"phases.{phase}.hash_of_hashes"
    )
    return {"offset": offset, "n": count, "hash_of_hashes": expected_hash}


def _normalize_frozen_config(
    raw: dict[str, Any], phase: str
) -> dict[str, Any]:
    version = raw.get("schema_version", raw.get("schema"))
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"frozen schema_version must be {SCHEMA_VERSION}, got {version!r}"
        )

    dataset = _require_dict(raw.get("dataset"), "dataset")
    dataset_name = dataset.get("name")
    split = dataset.get("split")
    seed = dataset.get("seed")
    if dataset_name not in {"wikitext2_docs", "wikitext2", "builtin"}:
        raise ValueError(f"unsupported frozen dataset: {dataset_name!r}")
    if not isinstance(split, str) or not split:
        raise TypeError("dataset.split must be a non-empty string")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("dataset.seed must be an integer")

    model_raw = raw.get("model")
    if isinstance(model_raw, str):
        model_name = model_raw
        model_revision = raw.get("model_revision")
    else:
        model = _require_dict(model_raw, "model")
        model_name = model.get("name")
        model_revision = model.get("revision")
    if not isinstance(model_name, str) or not model_name:
        raise TypeError("model.name must be a non-empty string")
    if model_revision is not None:
        model_revision = _model_revision(model_revision, "model.revision")

    dtype = raw.get("dtype")
    if dtype not in {"float32", "float16", "bfloat16", "auto"}:
        raise ValueError(f"unsupported frozen dtype: {dtype!r}")
    seq_len = _positive_int(raw.get("seq_len"), "seq_len")
    ep_size = _positive_int(raw.get("ep_size"), "ep_size")
    compute_seed = raw.get("compute_seed", 0)
    if isinstance(compute_seed, bool) or not isinstance(compute_seed, int):
        raise TypeError("compute_seed must be an integer")

    selected = _phase_spec(raw, phase)
    # Range disjointness is a preregistration check and does not open the other
    # phase's documents.
    calibration = _phase_spec(raw, "calibration")
    sealed = _phase_spec(raw, "sealed")
    calibration_range = range(
        calibration["offset"], calibration["offset"] + calibration["n"]
    )
    sealed_range = range(sealed["offset"], sealed["offset"] + sealed["n"])
    if max(calibration_range.start, sealed_range.start) < min(
        calibration_range.stop, sealed_range.stop
    ):
        raise RuntimeError("calibration and sealed dataset ranges overlap")

    if "historical_exclusion" not in raw:
        raise KeyError("frozen config must contain historical_exclusion")
    return {
        "dataset": {"name": dataset_name, "split": split, "seed": seed},
        "model": {"name": model_name, "revision": model_revision},
        "dtype": dtype,
        "seq_len": seq_len,
        "ep_size": ep_size,
        "compute_seed": compute_seed,
        "phase": selected,
        "historical_exclusion": raw["historical_exclusion"],
    }


def _file_spec(
    value: object, frozen_path: Path, label: str
) -> tuple[Path, str]:
    spec = _require_dict(value, label)
    path = _resolve_input_path(spec.get("path"), frozen_path, label)
    expected = _sha256(spec.get("sha256"), f"{label}.sha256")
    observed = sha256_bytes(path.read_bytes())
    if observed != expected:
        raise RuntimeError(
            f"{label} file hash drift: observed {observed}, expected {expected}"
        )
    return path, observed


def _walk_document_hashes(value: object) -> list[str]:
    hashes: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "sha256" and isinstance(item, str) and HEX64.fullmatch(item):
                hashes.append(item.lower())
            else:
                hashes.extend(_walk_document_hashes(item))
    elif isinstance(value, list):
        for item in value:
            hashes.extend(_walk_document_hashes(item))
    return hashes


def _load_registry(path: Path, label: str) -> set[str]:
    payload = _require_dict(
        json.loads(path.read_text(encoding="utf-8")), label
    )
    values = payload.get("unique_hashes")
    if not isinstance(values, list) or not values:
        raise RuntimeError(f"{label} must contain a non-empty unique_hashes list")
    hashes = {_sha256(value, f"{label}.unique_hashes") for value in values}
    if len(hashes) != len(values):
        raise RuntimeError(f"{label}.unique_hashes contains duplicates")
    expected_count = payload.get("unique_hash_count")
    if expected_count is not None and int(expected_count) != len(hashes):
        raise RuntimeError(f"{label}.unique_hash_count is inconsistent")
    expected_digest = payload.get("unique_hashes_sha256")
    if expected_digest is not None:
        expected_digest = _sha256(
            expected_digest, f"{label}.unique_hashes_sha256"
        )
        observed_digest = hash_lines(sorted(hashes))
        if observed_digest != expected_digest:
            raise RuntimeError(
                f"{label} unique-hash digest drift: observed {observed_digest}, "
                f"expected {expected_digest}"
            )
    return hashes


def _load_manifest(path: Path, label: str) -> set[str]:
    if path.suffix.lower() == ".csv":
        hashes: list[str] = []
        with path.open(newline="", encoding="utf-8") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                value = row.get("sha256")
                if value:
                    hashes.append(_sha256(value, f"{label}:{row_number}.sha256"))
    elif path.suffix.lower() == ".json":
        hashes = _walk_document_hashes(
            json.loads(path.read_text(encoding="utf-8"))
        )
    else:
        raise ValueError(f"{label} must be a JSON or CSV manifest: {path}")
    if not hashes:
        raise RuntimeError(f"{label} contains no document SHA-256 values")
    return set(hashes)


def load_historical_exclusions(
    value: object, frozen_path: Path
) -> tuple[set[str], dict[str, Any]]:
    spec = _require_dict(value, "historical_exclusion")
    direct_raw = spec.get("hashes", [])
    if not isinstance(direct_raw, list):
        raise TypeError("historical_exclusion.hashes must be a list")
    hashes = {
        _sha256(item, "historical_exclusion.hashes") for item in direct_raw
    }
    if len(hashes) != len(direct_raw):
        raise RuntimeError("historical_exclusion.hashes contains duplicates")

    inputs: list[dict[str, Any]] = []
    registry_specs: list[object] = []
    if "registry" in spec:
        registry_specs.append(spec["registry"])
    registries = spec.get("registries", [])
    if not isinstance(registries, list):
        raise TypeError("historical_exclusion.registries must be a list")
    registry_specs.extend(registries)
    for index, item in enumerate(registry_specs):
        label = f"historical_exclusion.registries[{index}]"
        path, file_hash = _file_spec(item, frozen_path, label)
        values = _load_registry(path, label)
        hashes.update(values)
        inputs.append(
            {
                "kind": "registry",
                "path": str(path),
                "sha256": file_hash,
                "document_hash_count": len(values),
            }
        )

    manifest_specs = spec.get("manifests", [])
    if not isinstance(manifest_specs, list):
        raise TypeError("historical_exclusion.manifests must be a list")
    for index, item in enumerate(manifest_specs):
        label = f"historical_exclusion.manifests[{index}]"
        path, file_hash = _file_spec(item, frozen_path, label)
        values = _load_manifest(path, label)
        hashes.update(values)
        inputs.append(
            {
                "kind": "manifest",
                "path": str(path),
                "sha256": file_hash,
                "document_hash_count": len(values),
            }
        )

    if not hashes:
        raise RuntimeError("historical exclusion set is empty")
    expected_combined = _sha256(
        spec.get("combined_hashes_sha256"),
        "historical_exclusion.combined_hashes_sha256",
    )
    observed_combined = hash_lines(sorted(hashes))
    if observed_combined != expected_combined:
        raise RuntimeError(
            "historical exclusion digest drift: "
            f"observed {observed_combined}, expected {expected_combined}"
        )
    return hashes, {
        "unique_hash_count": len(hashes),
        "combined_hashes_sha256": observed_combined,
        "direct_hash_count": len(direct_raw),
        "inputs": inputs,
    }


def _moe_modules(model: object) -> list[object]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("model does not expose model.layers")
    modules: list[object] = []
    for layer_id, layer in enumerate(layers):
        if hasattr(layer, "block_sparse_moe"):
            modules.append(layer.block_sparse_moe)
        elif hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            modules.append(layer.mlp)
        else:
            raise TypeError(f"unsupported MoE layer structure at layer {layer_id}")
    if not modules:
        raise RuntimeError("model has no MoE layers")
    return modules


def _restore_forwards(modules: list[object], forwards: list[object]) -> None:
    for module, forward in zip(modules, forwards):
        module.forward = forward


def _disable_nonroute_recording(recorder: MoeRecorder) -> None:
    """Keep the patched arithmetic but retain only route batches in memory."""

    def no_op(*_args: object, **_kwargs: object) -> None:
        return None

    recorder.update_contrib = no_op  # type: ignore[method-assign]
    recorder.update_receiver = no_op  # type: ignore[method-assign]
    recorder.update_error = no_op  # type: ignore[method-assign]
    recorder.update_pair_audit = no_op  # type: ignore[method-assign]


def validate_exact_full_path(
    model: object,
    tokenizer: object,
    text: str,
    seq_len: int,
) -> dict[str, Any]:
    """Fail unless the instrumented full path is bit-identical to pretrained."""
    modules = _moe_modules(model)
    original_forwards = [module.forward for module in modules]
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=seq_len
    )
    try:
        with torch.no_grad():
            original = model(**inputs).logits.detach().cpu()
        recorder = patch_mixtral_moe(
            model, "full", num_receiver_groups=1, record_routes=False
        )
        _disable_nonroute_recording(recorder)
        with torch.no_grad():
            patched = model(**inputs).logits.detach().cpu()
    finally:
        _restore_forwards(modules, original_forwards)
    exact = torch.equal(original, patched)
    if exact:
        max_abs_diff = 0.0
        mean_abs_diff = 0.0
    else:
        difference = (patched.float() - original.float()).abs()
        max_abs_diff = float(difference.max().item())
        mean_abs_diff = float(difference.mean().item())
    result = {
        "torch_equal": bool(exact),
        "max_abs_logit_diff": max_abs_diff,
        "mean_abs_logit_diff": mean_abs_diff,
    }
    if not exact:
        raise RuntimeError(f"patched full path is not exact: {result}")
    return result


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_manifest(
    frozen_path: Path,
    frozen_hash: str,
    exclusion_meta: dict[str, Any],
) -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    source_names = (
        "capture_route_fidelity_p0b.py",
        "capture_moe.py",
        "modeling.py",
        "prompts.py",
        "policies.py",
        "fake_quant.py",
        "creditreduce_reference.py",
        "grouped_owner_combine.py",
    )
    sources: dict[str, str] = {}
    for name in source_names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"required source is missing: {path}")
        sources[name] = sha256_bytes(path.read_bytes())
    return {
        "schema_version": SCHEMA_VERSION,
        "python_sources": sources,
        "frozen_config": {"path": str(frozen_path), "sha256": frozen_hash},
        "historical_exclusion_inputs": exclusion_meta["inputs"],
    }


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )


def _write_bytes_atomic(path: Path, value: bytes) -> str:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to replace stale temporary file: {temporary}")
    temporary.write_bytes(value)
    temporary.replace(path)
    return sha256_bytes(value)


class _HashingTextWriter:
    """Text writer that hashes exactly the UTF-8 bytes sent to the CSV file."""

    def __init__(self, handle: TextIO) -> None:
        self.handle = handle
        self.digest = hashlib.sha256()

    def write(self, value: str) -> int:
        self.digest.update(value.encode("utf-8"))
        return self.handle.write(value)

    def hexdigest(self) -> str:
        return self.digest.hexdigest()


def _flush_route_batches(
    recorder: MoeRecorder,
    writer: csv.writer,
    expected_sample_id: int,
    expected_tokens: int,
    expected_home_rank: int,
    expected_layers: int,
    expected_top_k: int | None,
) -> tuple[int, int]:
    batches = recorder.route_batches
    if len(batches) != expected_layers:
        raise RuntimeError(
            f"sample {expected_sample_id} produced {len(batches)} route batches; "
            f"expected {expected_layers}"
        )
    observed_top_k = expected_top_k
    rows_written = 0
    for batch in batches:
        if int(batch["sample_id"]) != expected_sample_id:
            raise RuntimeError("recorder sample_id drift during route capture")
        experts = batch["selected_experts"]
        weights = batch["routing_weights"]
        if not isinstance(experts, torch.Tensor) or not isinstance(weights, torch.Tensor):
            raise TypeError("route recorder emitted non-tensor route data")
        if experts.ndim != 2 or weights.shape != experts.shape:
            raise RuntimeError("route recorder emitted an invalid route tensor shape")
        if int(experts.shape[0]) != expected_tokens:
            raise RuntimeError(
                f"sample {expected_sample_id} route token count drift: "
                f"{experts.shape[0]} != {expected_tokens}"
            )
        top_k = int(experts.shape[1])
        if observed_top_k is None:
            observed_top_k = top_k
        elif top_k != observed_top_k:
            raise RuntimeError(f"top-k drift: {top_k} != {observed_top_k}")
        layer_id = int(batch["layer"])
        for token_position in range(expected_tokens):
            for rank_index in range(top_k):
                writer.writerow(
                    (
                        expected_sample_id,
                        layer_id,
                        token_position,
                        rank_index + 1,
                        int(experts[token_position, rank_index].item()),
                        float(weights[token_position, rank_index].item()),
                        expected_home_rank,
                    )
                )
                rows_written += 1
    recorder.route_batches.clear()
    recorder.routing_weight_batches.clear()
    if observed_top_k is None:
        raise RuntimeError("no route data was captured")
    return rows_written, observed_top_k


def capture_routes(
    output_path: Path,
    model: object,
    tokenizer: object,
    texts_by_sample: list[tuple[int, str, int, int]],
    seq_len: int,
) -> dict[str, Any]:
    recorder = patch_mixtral_moe(
        model, "full", num_receiver_groups=1, record_routes=True
    )
    _disable_nonroute_recording(recorder)
    expected_layers = len(_moe_modules(model))
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to replace stale temporary file: {temporary}")

    row_count = 0
    token_count = 0
    top_k: int | None = None
    with temporary.open("w", newline="", encoding="utf-8") as raw_handle:
        hashing_handle = _HashingTextWriter(raw_handle)
        writer = csv.writer(hashing_handle, lineterminator="\n")
        writer.writerow(ROUTE_COLUMNS)
        for sample_id, text, expected_tokens, expected_home_rank in texts_by_sample:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=seq_len
            )
            observed_tokens = int(inputs["input_ids"].shape[-1])
            if observed_tokens != expected_tokens:
                raise RuntimeError(
                    f"tokenization drift for sample {sample_id}: "
                    f"{observed_tokens} != {expected_tokens}"
                )
            recorder.set_sample_id(sample_id)
            with torch.no_grad():
                model(**inputs)
            written, top_k = _flush_route_batches(
                recorder,
                writer,
                sample_id,
                expected_tokens,
                expected_home_rank,
                expected_layers,
                top_k,
            )
            row_count += written
            token_count += expected_tokens
        raw_handle.flush()
        os.fsync(raw_handle.fileno())
        route_hash = hashing_handle.hexdigest()
    temporary.replace(output_path)
    return {
        "sha256": route_hash,
        "rows": row_count,
        "requests": len(texts_by_sample),
        "tokens": token_count,
        "moe_layers": expected_layers,
        "top_k": top_k,
    }


def _request_manifest(
    texts: list[str],
    tokenizer: object,
    dataset: dict[str, Any],
    phase: str,
    phase_spec: dict[str, Any],
    ep_size: int,
    seq_len: int,
) -> tuple[dict[str, Any], list[tuple[int, str, int, int]]]:
    requests: list[dict[str, Any]] = []
    text_by_sample: dict[int, tuple[str, int]] = {}
    for local_index, text in enumerate(texts):
        digest = sha256_bytes(text.encode("utf-8"))
        sample_id = phase_spec["offset"] + local_index
        tokens = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=seq_len
        )
        tokens_before = len(
            tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        )
        tokens_used = int(tokens["input_ids"].shape[-1])
        requests.append(
            {
                "sample_id": sample_id,
                "phase_local_index": local_index,
                "dataset_index": sample_id,
                "sha256": digest,
                "characters": len(text),
                "tokens_before_truncation": tokens_before,
                "tokens_used": tokens_used,
            }
        )
        text_by_sample[sample_id] = (text, tokens_used)

    hashes_in_dataset_order = [request["sha256"] for request in requests]
    if len(set(hashes_in_dataset_order)) != len(hashes_in_dataset_order):
        raise RuntimeError("selected phase contains duplicate document hashes")
    observed = hash_lines(hashes_in_dataset_order)
    expected = phase_spec["hash_of_hashes"]
    if observed != expected:
        raise RuntimeError(
            f"{phase} document hash-of-hashes drift: observed {observed}, "
            f"expected {expected}"
        )

    sorted_requests = sorted(requests, key=lambda row: row["sha256"])
    for sha_order, request in enumerate(sorted_requests):
        request["sha_order"] = sha_order
        request["home_rank"] = sha_order % ep_size
    texts_by_sample = []
    for request in sorted(requests, key=lambda row: int(row["sample_id"])):
        sample_id = int(request["sample_id"])
        text, tokens_used = text_by_sample[sample_id]
        texts_by_sample.append(
            (sample_id, text, tokens_used, int(request["home_rank"]))
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "dataset": dataset,
        "offset": phase_spec["offset"],
        "n": phase_spec["n"],
        "hash_of_hashes_dataset_order": observed,
        "hash_of_hashes_sha_order": hash_lines(
            [request["sha256"] for request in sorted_requests]
        ),
        "home_rank_assignment": "sha256_lexicographic_round_robin",
        "ep_size": ep_size,
        "requests": sorted_requests,
    }, texts_by_sample


def _validate_model_revision(
    expected: str | None, model: object, tokenizer: object
) -> dict[str, Any]:
    observed_model = getattr(getattr(model, "config", None), "_commit_hash", None)
    tokenizer_kwargs = getattr(tokenizer, "init_kwargs", {})
    observed_tokenizer = (
        tokenizer_kwargs.get("_commit_hash")
        if isinstance(tokenizer_kwargs, dict)
        else None
    )
    if expected is not None:
        if observed_model is None:
            raise RuntimeError(
                "frozen model revision was supplied but the loaded model exposes no commit hash"
            )
        if str(observed_model).lower() != expected:
            raise RuntimeError(
                f"model revision drift: observed {observed_model}, expected {expected}"
            )
        if observed_tokenizer is not None and str(observed_tokenizer).lower() != expected:
            raise RuntimeError(
                "tokenizer revision differs from the frozen model revision: "
                f"{observed_tokenizer} != {expected}"
            )
    return {
        "expected_revision": expected,
        "observed_model_revision": observed_model,
        "observed_tokenizer_revision": observed_tokenizer,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-config", required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Required: use only locally cached datasets, tokenizers, and weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.offline:
        raise RuntimeError("route capture is offline-only; pass --offline")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    frozen_path = Path(args.frozen_config).expanduser().resolve()
    if not frozen_path.is_file():
        raise FileNotFoundError(f"frozen config does not exist: {frozen_path}")
    frozen_bytes = frozen_path.read_bytes()
    raw = _require_dict(json.loads(frozen_bytes), "frozen config")
    config = _normalize_frozen_config(raw, args.phase)

    output = Path(args.output_dir).expanduser().resolve()
    if args.phase == "sealed" and output.exists():
        raise FileExistsError(
            f"sealed output directory already exists; refusing to open it: {output}"
        )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output dir: {output}")

    exclusion_hashes, exclusion_meta = load_historical_exclusions(
        config["historical_exclusion"], frozen_path
    )
    phase_spec = config["phase"]
    dataset = config["dataset"]
    texts = get_prompts(
        dataset["name"],
        phase_spec["n"],
        offset=phase_spec["offset"],
        split=dataset["split"],
        seed=dataset["seed"],
    )
    document_hashes = [sha256_bytes(text.encode("utf-8")) for text in texts]
    observed_phase_hash = hash_lines(document_hashes)
    if observed_phase_hash != phase_spec["hash_of_hashes"]:
        raise RuntimeError(
            "frozen phase hash mismatch before model load: "
            f"{observed_phase_hash} != {phase_spec['hash_of_hashes']}"
        )
    overlap = sorted(set(document_hashes) & exclusion_hashes)
    if overlap:
        raise RuntimeError(
            f"fresh-document guard failed; historical overlap: {overlap}"
        )

    torch.manual_seed(config["compute_seed"])
    tokenizer = load_tokenizer(config["model"]["name"], local_files_only=True)
    load_start = time.perf_counter()
    model, modeled_load_seconds = load_model(
        config["model"]["name"],
        dtype_name=config["dtype"],
        local_files_only=True,
    )
    wall_load_seconds = time.perf_counter() - load_start
    revision = _validate_model_revision(
        config["model"]["revision"], model, tokenizer
    )
    exactness = validate_exact_full_path(
        model, tokenizer, texts[0], config["seq_len"]
    )
    request_manifest, texts_by_sample = _request_manifest(
        texts,
        tokenizer,
        dataset,
        args.phase,
        phase_spec,
        config["ep_size"],
        config["seq_len"],
    )

    # Create the directory only after all non-capture preflight checks pass.
    # This avoids burning a sealed output path for a missing local model while
    # preserving the rule that any actual sealed capture is single-shot.
    if args.phase == "sealed" and output.exists():
        raise FileExistsError(f"sealed output directory appeared during preflight: {output}")
    output.mkdir(parents=True, exist_ok=args.phase != "sealed")

    frozen_hash = sha256_bytes(frozen_bytes)
    source_manifest = _source_manifest(
        frozen_path, frozen_hash, exclusion_meta
    )
    source_hash = _write_bytes_atomic(
        output / "source_manifest.json", _json_bytes(source_manifest)
    )
    request_hash = _write_bytes_atomic(
        output / "request_manifest.json", _json_bytes(request_manifest)
    )
    route_meta = capture_routes(
        output / "routes.csv",
        model,
        tokenizer,
        texts_by_sample,
        config["seq_len"],
    )

    completed = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "phase": args.phase,
        "evidence_boundary": (
            "offline CPU/model route capture only; no backend, wire-byte, latency, "
            "quality, or route-distribution claim"
        ),
        "offline": True,
        "frozen_config": {"path": str(frozen_path), "sha256": frozen_hash},
        "dataset": dataset,
        "phase_spec": phase_spec,
        "observed_hash_of_hashes": observed_phase_hash,
        "historical_exclusion": exclusion_meta,
        "model": {
            "name": config["model"]["name"],
            "dtype": config["dtype"],
            "seq_len": config["seq_len"],
            "compute_seed": config["compute_seed"],
            **revision,
            "modeled_load_seconds": modeled_load_seconds,
            "wall_load_seconds": wall_load_seconds,
        },
        "ep_size": config["ep_size"],
        "home_rank_assignment": "sha256_lexicographic_round_robin",
        "exact_full_path": exactness,
        "capture": route_meta,
        "outputs": {
            "routes.csv": route_meta["sha256"],
            "request_manifest.json": request_hash,
            "source_manifest.json": source_hash,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "datasets": _package_version("datasets"),
        },
    }
    _write_bytes_atomic(output / "config.json", _json_bytes(completed))
    print(
        f"captured phase={args.phase} requests={route_meta['requests']} "
        f"rows={route_meta['rows']} into {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()

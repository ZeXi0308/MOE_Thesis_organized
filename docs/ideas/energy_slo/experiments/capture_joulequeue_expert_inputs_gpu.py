#!/usr/bin/env python3
"""Capture audited BF16 inputs for a frozen subset of real MoE experts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import torch
from torch import nn


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
RECEIVER_EXPERIMENTS = REPO_ROOT / "docs" / "ideas" / "receiver_aware" / "experiments"
SHARED = REPO_ROOT / "experiments" / "shared"
for path in (RECEIVER_EXPERIMENTS, SHARED):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from capture_cjc_routes_gpu import (  # noqa: E402
    MODEL_SPECS,
    _load_manifest,
    _load_model_and_tokenizer,
    sha256_file,
)
from capture_moe import patch_mixtral_moe  # noqa: E402


EXPERT_PATH = re.compile(r"(?:^|\.)layers\.(\d+)\..*experts\.(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=HERE.parent / "JouleQueue_Phase2_冻结实验协议_2026-07-22.md",
    )
    parser.add_argument("--mode", choices=("dev", "formal"), default="dev")
    parser.add_argument("--signoff", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def _is_expert(module: nn.Module) -> bool:
    triplets = (("gate_proj", "up_proj", "down_proj"), ("w1", "w3", "w2"))
    return hasattr(module, "act_fn") and any(
        all(isinstance(getattr(module, name, None), nn.Linear) for name in names)
        for names in triplets
    )


def _expert_modules(model: nn.Module) -> dict[tuple[int, int], nn.Module]:
    result: dict[tuple[int, int], nn.Module] = {}
    for name, module in model.named_modules():
        match = EXPERT_PATH.search(name)
        if match is None or not _is_expert(module):
            continue
        key = int(match.group(1)), int(match.group(2))
        if key in result:
            raise RuntimeError(f"duplicate expert module identity: {key}")
        result[key] = module
    if not result:
        raise RuntimeError("model exposes no auditable expert modules")
    return result


def _hash_rank(seed: int, *parts: object) -> str:
    return hashlib.sha256(":".join(map(str, (seed, *parts))).encode("utf-8")).hexdigest()


def select_experts(
    identities: set[tuple[int, int]], seed: int
) -> tuple[tuple[int, int], ...]:
    layers = sorted({layer for layer, _ in identities})
    selected_layers = sorted(layers, key=lambda layer: _hash_rank(seed, "layer", layer))[:4]
    if len(selected_layers) != 4:
        raise RuntimeError("frozen surface requires at least four MoE layers")
    selected: list[tuple[int, int]] = []
    for layer in selected_layers:
        experts = sorted(expert for candidate, expert in identities if candidate == layer)
        chosen = sorted(
            experts, key=lambda expert: _hash_rank(seed, "expert", layer, expert)
        )[:4]
        if len(chosen) != 4:
            raise RuntimeError(f"layer {layer} exposes fewer than four experts")
        selected.extend((layer, expert) for expert in chosen)
    return tuple(selected)


def _source_hash() -> str:
    paths = (
        Path(__file__),
        RECEIVER_EXPERIMENTS / "capture_cjc_routes_gpu.py",
        SHARED / "capture_moe.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _require_formal_signoff(
    path: Path | None, protocol_sha: str, source_sha: str, data_sha: str
) -> Mapping[str, Any]:
    if path is None or not path.is_file():
        raise RuntimeError("formal expert-input capture requires SIGNED-OFF")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "SIGNED-OFF",
        "joulequeue_protocol_sha256": protocol_sha,
        "joulequeue_capture_source_sha256": source_sha,
        "data_manifest_sha256": data_sha,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise RuntimeError(f"formal signoff mismatch for {key}")
    return value


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; synthetic activation fallback is forbidden")
    manifest = _load_manifest(args.data_manifest)
    if manifest.get("protocol_split") != "calibration":
        raise RuntimeError("expert surface inputs must come from calibration only")
    spec = MODEL_SPECS[args.model_key]
    model_revision = f"{spec['model_id']}@{spec['revision']}"
    if manifest.get("model_revisions", {}).get(args.model_key) != model_revision:
        raise RuntimeError("model/data revision mismatch")
    protocol_sha = sha256_file(args.protocol)
    source_sha = _source_hash()
    data_sha = str(manifest["manifest_sha256"])
    signoff = None
    if args.mode == "formal":
        signoff = _require_formal_signoff(
            args.signoff, protocol_sha, source_sha, data_sha
        )

    model, tokenizer = _load_model_and_tokenizer(args, spec)
    experts = _expert_modules(model)
    selected = select_experts(set(experts), args.seed)
    selected_set = set(selected)
    recorder = patch_mixtral_moe(
        model, policy_name="full", record_routes=True, record_diagnostics=False
    )

    context: dict[str, object] = {}
    captured: list[dict[str, object]] = []
    handles = []
    for (layer_id, expert_id), expert in experts.items():
        if (layer_id, expert_id) not in selected_set:
            continue

        def hook(
            _module: nn.Module,
            inputs: tuple[object, ...],
            *,
            layer_id: int = layer_id,
            expert_id: int = expert_id,
        ) -> None:
            if not context:
                raise RuntimeError("expert hook fired without request context")
            if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError("expert input hook expected one Tensor")
            activation = inputs[0]
            if activation.ndim != 2 or activation.shape[0] < 1:
                raise RuntimeError("captured expert activation must be non-empty [rows,hidden]")
            captured.append(
                {
                    "request_id": context["request_id"],
                    "forward_id": context["forward_id"],
                    "layer_id": layer_id,
                    "expert_id": expert_id,
                    "row_count": int(activation.shape[0]),
                    "activation": activation.detach().to(
                        device="cpu", dtype=torch.bfloat16
                    ).contiguous(),
                }
            )

        handles.append(expert.register_forward_pre_hook(hook))

    requests = manifest["requests"]
    assert isinstance(requests, list)
    try:
        for request_index, request in enumerate(requests):
            request_id = str(request["request_id"])
            encoded = tokenizer(
                str(request["text"]),
                add_special_tokens=False,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"]
            if tuple(input_ids.shape) != (1, 128):
                raise RuntimeError("manifest request is not exactly 128 tokens")
            forward_id = f"{request_id}:prefill:0"
            context.update(request_id=request_id, forward_id=forward_id)
            before_routes = len(recorder.route_batches)
            before_inputs = len(captured)
            recorder.set_sample_id(request_index)
            with torch.inference_mode():
                model(input_ids=input_ids.to("cuda:0"), use_cache=False)
            torch.cuda.synchronize()
            context.clear()

            counts: dict[tuple[int, int], int] = {}
            for batch in recorder.route_batches[before_routes:]:
                layer_id = int(batch["layer"])
                selected_experts = batch["selected_experts"]
                if not isinstance(selected_experts, torch.Tensor):
                    raise RuntimeError("route recorder emitted non-Tensor experts")
                bincount = torch.bincount(
                    selected_experts.reshape(-1).to(torch.int64),
                    minlength=int(spec["num_experts"]),
                )
                for expert_id in range(int(spec["num_experts"])):
                    counts[(layer_id, expert_id)] = int(bincount[expert_id].item())
            seen: set[tuple[int, int]] = set()
            for row in captured[before_inputs:]:
                key = int(row["layer_id"]), int(row["expert_id"])
                if key in seen:
                    raise RuntimeError(f"expert invoked more than once per forward: {key}")
                seen.add(key)
                if int(row["row_count"]) != counts.get(key, 0):
                    raise RuntimeError(f"hook/router row-count mismatch for {key}")
            expected_active = {key for key in selected_set if counts.get(key, 0) > 0}
            if seen != expected_active:
                raise RuntimeError(
                    f"selected expert hook coverage mismatch: missing={expected_active-seen}, "
                    f"extra={seen-expected_active}"
                )
    finally:
        context.clear()
        for handle in handles:
            handle.remove()

    if not captured:
        raise RuntimeError("capture contains no selected-expert activations")
    artifact = {
        "schema_version": 1,
        "metadata": {
            "status": "CAPTURE_ONLY" if args.mode == "formal" else "NOT_TESTED",
            "scientific_result": False,
            "capture_kind": "joulequeue_real_bf16_expert_inputs",
            "input_source": "measured_same_gpu_model_forward",
            "model_revision": model_revision,
            "data_split": "calibration",
            "data_manifest_sha256": data_sha,
            "protocol_sha256": protocol_sha,
            "capture_source_sha256": source_sha,
            "selection_seed": args.seed,
            "selected_experts": [list(key) for key in selected],
            "gpu_name": torch.cuda.get_device_name(0),
            "signoff_sha256": sha256_file(args.signoff) if signoff else None,
        },
        "records": captured,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError("refusing to overwrite expert-input capture")
    torch.save(artifact, args.output)
    print(
        json.dumps(
            {
                "status": artifact["metadata"]["status"],
                "records": len(captured),
                "selected_experts": len(selected),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


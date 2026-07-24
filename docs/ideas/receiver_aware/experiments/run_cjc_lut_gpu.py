#!/usr/bin/env python3
"""Measure the cjc-v1 component LUT from real calibration activations on CUDA.

The output intentionally does not collapse all timing into a misleading
``measured_same_gpu`` label.  Expert, pack, pinned H2D (NOT RDMA), canonical
reduction, and launch-accounting provenance are recorded independently.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Callable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
SHARED = REPO_ROOT / "experiments/shared"
for path in (HERE, SHARED):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cjc_policy import (  # noqa: E402
    CJCValidationError,
    LUT_COMPONENT_PROVENANCE,
    LUT_EXPERT_SOURCE,
    LUT_HOST_STAGING_SOURCE,
    LUT_LAUNCH_SOURCE,
    LUT_PACK_SOURCE,
    LUT_REDUCTION_SOURCE,
)
from run_cjc_oracle import load_json, load_routes, sha256_file  # noqa: E402


EXPERT_PATH = re.compile(r"(?:^|\.)layers\.(\d+)\..*experts\.(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=("olmoe", "llmjp"), required=True)
    parser.add_argument("--calibration-data-manifest", type=Path, required=True)
    parser.add_argument("--calibration-route-trace", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "docs/ideas/receiver_aware/configs/cjc_v1.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=HERE.parent / "CJC_Phase2_冻结实验协议_2026-07-22.md",
    )
    parser.add_argument("--mode", choices=("dev", "formal"), default="dev")
    parser.add_argument("--signoff", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _load_calibration_manifest(path: Path, model_key: str, revision: str) -> Mapping[str, Any]:
    raw = load_json(path)
    if raw.get("schema_version") != 1 or raw.get("protocol_split") != "calibration":
        raise CJCValidationError("LUT inputs must use the frozen calibration manifest")
    supplied = raw.get("manifest_sha256")
    unhashed = dict(raw)
    unhashed.pop("manifest_sha256", None)
    if supplied != hashlib.sha256(_canonical_json(unhashed)).hexdigest():
        raise CJCValidationError("calibration data manifest self-hash mismatch")
    revisions = raw.get("model_revisions")
    if not isinstance(revisions, dict):
        raise CJCValidationError("calibration manifest lacks model revisions")
    producer_key = "llm_jp" if model_key == "llmjp" else model_key
    if revisions.get(producer_key) != revision and revisions.get(model_key) != revision:
        raise CJCValidationError("calibration manifest/model revision mismatch")
    requests = raw.get("requests")
    if not isinstance(requests, list) or len(requests) != 64:
        raise CJCValidationError("CJC LUT calibration requires exactly 64 calibration requests")
    return raw


def select_ids(
    candidates: Sequence[int], *, count: int, seed: int, prefix: Sequence[object]
) -> tuple[int, ...]:
    unique = sorted(set(int(value) for value in candidates))
    ranked = sorted(
        unique,
        key=lambda value: hashlib.sha256(
            ":".join(map(str, (*prefix, seed, value))).encode("utf-8")
        ).digest(),
    )
    selected = tuple(ranked[:count])
    if len(selected) != count:
        raise CJCValidationError("too few identities for frozen LUT selection")
    return selected


def selected_route_layers(
    route_path: Path, *, revision: str, data_sha: str, seed: int, count: int
) -> tuple[int, ...]:
    routes = load_routes((route_path,))
    if any(route.model_revision != revision for route in routes):
        raise CJCValidationError("LUT route trace contains wrong model revision")
    if any(route.data_manifest_sha256 != data_sha for route in routes):
        raise CJCValidationError("LUT route trace is not calibration-bound")
    if any(route.route_source != "native_model_forward" for route in routes):
        raise CJCValidationError("LUT route source must be native_model_forward")
    request_ids = {route.request_id for route in routes}
    if len(request_ids) != 64:
        raise CJCValidationError("LUT route trace requires 64 calibration requests")
    layers = sorted({route.layer_id for route in routes})
    return tuple(
        sorted(
            layers,
            key=lambda layer: hashlib.sha256(
                f"{seed}:{revision}:{layer}".encode("utf-8")
            ).digest(),
        )[:count]
    )


def _is_expert(module: Any, torch_module: Any) -> bool:
    triplets = (("gate_proj", "up_proj", "down_proj"), ("w1", "w3", "w2"))
    return hasattr(module, "act_fn") and any(
        all(isinstance(getattr(module, name, None), torch_module.nn.Linear) for name in names)
        for names in triplets
    )


def _expert_modules(model: Any, torch_module: Any) -> dict[tuple[int, int], Any]:
    result: dict[tuple[int, int], Any] = {}
    for name, module in model.named_modules():
        match = EXPERT_PATH.search(name)
        if match is None or not _is_expert(module, torch_module):
            continue
        identity = int(match.group(1)), int(match.group(2))
        if identity in result:
            raise CJCValidationError(f"duplicate expert module identity {identity}")
        result[identity] = module
    if not result:
        raise CJCValidationError("model exposes no auditable expert modules")
    return result


def _cuda_trial_us(torch_module: Any, operation: Callable[[], object]) -> float:
    start = torch_module.cuda.Event(enable_timing=True)
    end = torch_module.cuda.Event(enable_timing=True)
    start.record()
    output = operation()
    end.record()
    end.synchronize()
    if output is None:
        raise AssertionError("measured operation returned no output")
    return float(start.elapsed_time(end)) * 1000.0


def _median_cuda_us(
    torch_module: Any,
    operation: Callable[[], object],
    *,
    warmups: int,
    trials: int,
) -> float:
    for _ in range(warmups):
        operation()
    torch_module.cuda.synchronize()
    samples = [_cuda_trial_us(torch_module, operation) for _ in range(trials)]
    if any(value < 0 for value in samples):
        raise CJCValidationError("negative CUDA timing")
    return statistics.median(samples)


def _source_hash() -> str:
    files = (
        Path(__file__),
        HERE / "capture_cjc_routes_gpu.py",
        HERE / "cjc_policy.py",
        HERE / "run_cjc_oracle.py",
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _require_signoff(path: Path | None, bindings: Mapping[str, str]) -> None:
    if path is None or not path.is_file():
        raise CJCValidationError("formal LUT measurement requires Phase-4 SIGNED-OFF")
    raw = load_json(path)
    if raw.get("status") != "SIGNED-OFF":
        raise CJCValidationError("LUT Phase-4 status is not SIGNED-OFF")
    for key, expected in bindings.items():
        if raw.get(key) != expected:
            raise CJCValidationError(f"LUT signoff hash mismatch: {key}")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise CJCValidationError("refusing to write empty LUT")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - GPU environment capability
        raise RuntimeError("PyTorch is required for CJC LUT measurement") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; analytical LUT fallback is forbidden")

    args = parse_args()
    config = load_json(args.config)
    required_models = config.get("required_models")
    lut_cfg = config.get("lut")
    replay_cfg = config.get("replay_selection")
    if not all(isinstance(value, dict) for value in (required_models, lut_cfg, replay_cfg)):
        raise CJCValidationError("CJC LUT config is incomplete")
    if lut_cfg.get("schema_version") != "cjc-lut-v1":
        raise CJCValidationError("CJC LUT schema drift")
    model_cfg = required_models.get(args.model_key)
    if not isinstance(model_cfg, dict):
        raise CJCValidationError("unknown frozen CJC model")
    revision = str(model_cfg["revision"])
    data_manifest = _load_calibration_manifest(
        args.calibration_data_manifest, args.model_key, revision
    )
    data_sha = str(data_manifest["manifest_sha256"])
    layer_ids = selected_route_layers(
        args.calibration_route_trace,
        revision=revision,
        data_sha=data_sha,
        seed=int(replay_cfg["seed"]),
        count=int(lut_cfg["layers_per_model"]),
    )
    if len(layer_ids) != int(lut_cfg["layers_per_model"]):
        raise CJCValidationError("route trace exposes too few layers for CJC LUT")

    hashes = {
        "cjc_lut_protocol_sha256": sha256_file(args.protocol),
        "cjc_lut_config_sha256": sha256_file(args.config),
        "cjc_lut_source_sha256": _source_hash(),
        "cjc_lut_data_manifest_sha256": sha256_file(args.calibration_data_manifest),
        "cjc_lut_route_trace_sha256": sha256_file(args.calibration_route_trace),
    }
    if args.mode == "formal":
        _require_signoff(args.signoff, hashes)

    from capture_cjc_routes_gpu import _load_model_and_tokenizer

    producer_key = "llm_jp" if args.model_key == "llmjp" else args.model_key
    from capture_cjc_routes_gpu import MODEL_SPECS

    spec = MODEL_SPECS[producer_key]
    model, tokenizer = _load_model_and_tokenizer(args, spec)
    experts = _expert_modules(model, torch)
    experts_by_layer = {
        layer: sorted(expert for candidate, expert in experts if candidate == layer)
        for layer in layer_ids
    }
    selected_experts = {
        layer: select_ids(
            experts_by_layer[layer],
            count=int(lut_cfg["experts_per_layer"]),
            seed=int(lut_cfg["selection_seed"]),
            prefix=(revision, layer, "expert"),
        )
        for layer in layer_ids
    }

    maximum_rows = max(int(value) for value in lut_cfg["rows"])
    pools: dict[int, list[Any]] = {layer: [] for layer in layer_ids}
    pool_rows = {layer: 0 for layer in layer_ids}
    handles = []
    for (layer, _expert), module in experts.items():
        if layer not in pools:
            continue

        def hook(
            _module: object,
            inputs: tuple[object, ...],
            *,
            layer: int = layer,
        ) -> None:
            if pool_rows[layer] >= maximum_rows:
                return
            if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
                raise CJCValidationError("expert hook expected one Tensor input")
            activation = inputs[0]
            if activation.ndim != 2 or activation.shape[1] != int(model_cfg["hidden_size"]):
                raise CJCValidationError("expert hook activation shape mismatch")
            take = min(maximum_rows - pool_rows[layer], int(activation.shape[0]))
            if take:
                pools[layer].append(
                    activation[:take].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                )
                pool_rows[layer] += take

        handles.append(module.register_forward_pre_hook(hook))

    requests_used = 0
    try:
        for request in data_manifest["requests"]:
            encoded = tokenizer(
                str(request["text"]),
                add_special_tokens=False,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            if tuple(encoded["input_ids"].shape) != (1, 128):
                raise CJCValidationError("LUT calibration request is not exactly 128 tokens")
            with torch.inference_mode():
                model(input_ids=encoded["input_ids"].to("cuda:0"), use_cache=False)
            torch.cuda.synchronize()
            requests_used += 1
            if all(value >= maximum_rows for value in pool_rows.values()):
                break
    finally:
        for handle in handles:
            handle.remove()
    if any(value < maximum_rows for value in pool_rows.values()):
        raise CJCValidationError(f"insufficient real calibration activation rows: {pool_rows}")

    rows_out: list[dict[str, object]] = []
    trial_detail: dict[str, object] = {}
    warmups = int(lut_cfg["warmup_calls"])
    trials = int(lut_cfg["independent_trials"])
    top_k = int(model_cfg["top_k"])
    with torch.inference_mode():
        for layer in layer_ids:
            pool = torch.cat(pools[layer], dim=0)
            for row_count in map(int, lut_cfg["rows"]):
                activation = pool[:row_count].to(device="cuda:0", dtype=torch.bfloat16)
                expert_samples: dict[str, float] = {}
                for expert_id in selected_experts[layer]:
                    expert = experts[(layer, expert_id)]
                    expert_samples[str(expert_id)] = _median_cuda_us(
                        torch,
                        lambda expert=expert, activation=activation: expert(activation),
                        warmups=warmups,
                        trials=trials,
                    )
                expert_us = max(expert_samples.values())

                index = torch.arange(row_count - 1, -1, -1, device="cuda:0")
                pack_us = _median_cuda_us(
                    torch,
                    lambda: torch.index_select(activation, 0, index),
                    warmups=warmups,
                    trials=trials,
                )
                host = activation.detach().cpu().pin_memory()
                device_staging = torch.empty_like(activation)
                host_staging_us = _median_cuda_us(
                    torch,
                    lambda: device_staging.copy_(host, non_blocking=True),
                    warmups=warmups,
                    trials=trials,
                )
                siblings = activation.unsqueeze(0).expand(top_k, -1, -1).contiguous()

                def canonical_reduce() -> object:
                    accumulator = siblings[0].clone()
                    for slot in range(1, top_k):
                        accumulator = accumulator + siblings[slot]
                    return accumulator

                reduction_total_us = _median_cuda_us(
                    torch, canonical_reduce, warmups=warmups, trials=trials
                )
                reduction_us = reduction_total_us / top_k
                rows_out.append(
                    {
                        "model_revision": revision,
                        "layer_id": layer,
                        "rows": row_count,
                        "expert_us": expert_us,
                        "pack_us": pack_us,
                        "launch_us": 0.0,
                        "host_staging_us": host_staging_us,
                        "reduction_us": reduction_us,
                        "source": LUT_COMPONENT_PROVENANCE,
                        "expert_source": LUT_EXPERT_SOURCE,
                        "pack_source": LUT_PACK_SOURCE,
                        "launch_source": LUT_LAUNCH_SOURCE,
                        "host_staging_source": LUT_HOST_STAGING_SOURCE,
                        "reduction_source": LUT_REDUCTION_SOURCE,
                    }
                )
                trial_detail[f"{layer}/{row_count}"] = {
                    "expert_median_us_by_expert": expert_samples,
                    "expert_aggregation": "max_of_four_outcome_blind_expert_medians",
                    "reduction_total_us": reduction_total_us,
                    "reduction_accounting": f"canonical_total_divided_by_top_k_{top_k}",
                }

    if args.output_dir.exists():
        raise CJCValidationError("refusing to overwrite CJC LUT output directory")
    args.output_dir.mkdir(parents=True)
    _write_csv(args.output_dir / "lut.csv", rows_out)
    metadata = {
        "schema_version": "cjc-lut-v1",
        "status": "LUT_ONLY" if args.mode == "formal" else "NOT_TESTED",
        "scientific_result": False,
        "mode": args.mode,
        "model_revision": revision,
        "gpu_name": torch.cuda.get_device_name(0),
        "calibration_data_manifest_sha256": data_sha,
        "selected_layers": list(layer_ids),
        "selected_experts_by_layer": {
            str(layer): list(values) for layer, values in selected_experts.items()
        },
        "activation_source": "real_calibration_model_forward",
        "requests_used_for_activation_pool": requests_used,
        "component_boundary": {
            "expert_pack_reduction": "same_cuda_gpu_event_measurement",
            "host_staging": "pinned_H2D_same_run_host_NOT_RDMA",
            "launch": "zero_separate_charge_kernel_launch_included_in_component_events",
            "wire": "not_in_LUT_analytic_link_added_by_replay",
        },
        "hashes": hashes,
        "trial_detail": trial_detail,
        "signoff_sha256": sha256_file(args.signoff) if args.mode == "formal" else None,
    }
    (args.output_dir / "lut_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": metadata["status"], "rows": len(rows_out)}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

"""Measure a native model expert row-service curve on CUDA."""

import argparse
import csv
import math
from pathlib import Path
import statistics
import sys
import time

try:
    from .core import ProtocolError, sha256_file, write_json
except ImportError:
    from core import ProtocolError, sha256_file, write_json


CURVE_COLUMNS = ("model", "layer", "expert", "rows", "median_us", "p95_us", "trials", "backend", "dtype")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", help="Hugging Face model id; runs the selected native expert")
    mode.add_argument("--smoke", action="store_true", help="deterministic analytical fixture, never evidence")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] if low == high else ordered[low] + (position - low) * (ordered[high] - ordered[low])


def _locate_expert(model: object, layer_index: int, expert_index: int):
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or layer_index < 0 or layer_index >= len(layers):
        raise ProtocolError(f"cannot locate model layer {layer_index}")
    layer = layers[layer_index]
    containers = [getattr(layer, "block_sparse_moe", None), getattr(layer, "mlp", None)]
    for container in containers:
        experts = getattr(container, "experts", None)
        if experts is not None:
            if expert_index < 0 or expert_index >= len(experts):
                raise ProtocolError(f"expert {expert_index} outside [0,{len(experts) - 1}]")
            return experts[expert_index]
    raise ProtocolError("selected layer has no supported experts container")


def _hidden_size(model: object) -> int:
    for name in ("hidden_size", "d_model"):
        value = getattr(getattr(model, "config", None), name, None)
        if value:
            return int(value)
    raise ProtocolError("model config has no hidden size")


def real_measurements(args: argparse.Namespace) -> list[dict[str, object]]:
    try:
        import torch
    except ImportError as exc:
        raise ProtocolError("native benchmark requires PyTorch") from exc
    if not torch.cuda.is_available():
        raise ProtocolError("CUDA is mandatory; this benchmark never falls back to CPU")
    shared = next(
        candidate / "experiments/shared"
        for candidate in Path(__file__).resolve().parents
        if (candidate / "experiments/shared").is_dir()
    )
    sys.path.insert(0, str(shared))
    from modeling import load_model

    model, _ = load_model(
        args.model,
        dtype_name=args.dtype,
        local_files_only=args.offline,
        revision=args.model_revision,
    )
    expert = _locate_expert(model, args.layer, args.expert)
    hidden = _hidden_size(model)
    dtype = getattr(torch, args.dtype)
    output: list[dict[str, object]] = []
    for rows in sorted(set(args.rows)):
        inputs = torch.randn((rows, hidden), device="cuda", dtype=dtype)
        with torch.inference_mode():
            for _ in range(args.warmups):
                expert(inputs)
            torch.cuda.synchronize()
            samples = []
            for _ in range(args.trials):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                expert(inputs)
                end.record()
                end.synchronize()
                samples.append(float(start.elapsed_time(end) * 1000.0))
        output.append(
            {
                "model": args.model_key,
                "layer": args.layer,
                "expert": args.expert,
                "rows": rows,
                "median_us": statistics.median(samples),
                "p95_us": _percentile(samples, 0.95),
                "trials": args.trials,
                "backend": "native-transformers-expert-forward",
                "dtype": args.dtype,
            }
        )
        print(f"rows={rows}: median={output[-1]['median_us']:.3f} us", flush=True)
    return output


def smoke_measurements(args: argparse.Namespace) -> list[dict[str, object]]:
    output = []
    for rows in sorted(set(args.rows)):
        latency = 8.0 + 1.8 * math.sqrt(rows) + 0.055 * rows
        output.append(
            {
                "model": args.model_key,
                "layer": -1,
                "expert": -1,
                "rows": rows,
                "median_us": latency,
                "p95_us": latency * 1.05,
                "trials": 0,
                "backend": "SMOKE_ONLY-analytical-fixture",
                "dtype": args.dtype,
            }
        )
    return output


def main() -> None:
    args = parse_args()
    if any(row <= 0 for row in args.rows) or args.warmups < 0 or args.trials <= 0:
        raise SystemExit("rows/trials must be positive and warmups non-negative")
    started = time.time()
    rows = smoke_measurements(args) if args.smoke else real_measurements(args)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CURVE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        path.with_suffix(".meta.json"),
        {
            "schema": "bcrd-service-curve-v1",
            "model": args.model_key,
            "model_revision": args.model_revision,
            "cuda_required": not args.smoke,
            "formal_eligible": not args.smoke,
            "smoke": bool(args.smoke),
            "warmups": 0 if args.smoke else args.warmups,
            "trials": 0 if args.smoke else args.trials,
            "elapsed_seconds": time.time() - started,
            "output_sha256": sha256_file(path),
            "evidence_boundary": (
                "SMOKE_ONLY deterministic curve; no GPU measurement"
                if args.smoke
                else "single-GPU native expert forward; not grouped EP, A2A, TPOT or P99 evidence"
            ),
        },
    )
    print(f"wrote {len(rows)} curve points to {path}")


if __name__ == "__main__":
    main()

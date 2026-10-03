"""Independent CPU tensor readback; no imported replay/gate implementation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import torch


def require(ok, message):
    if not ok:
        raise ValueError(message)


def metrics(a, b):
    require(a.shape == b.shape and a.dtype == b.dtype, "comparison shape/dtype")
    aa, bb = [t.contiguous().view(torch.uint8).reshape(t.numel(), t.element_size()) for t in (a, b)]
    count = int(torch.all(aa == bb, dim=1).sum())
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    delta = (a.to(torch.float32) - b.to(torch.float32)).reshape(-1)
    norm = float(torch.linalg.vector_norm(b.to(torch.float32))) if finite else None
    error = float(torch.linalg.vector_norm(delta)) if finite else None
    return dict(elements=a.numel(), bit_equal_elements=count, bit_equal=count == a.numel(), allfinite=finite,
        allclose=bool(torch.allclose(a, b, rtol=.01, atol=.01)), maxabs=float(delta.abs().max()) if finite else None,
        relative_l2=error / norm if norm else (0. if error == 0 else None))


def total(parts, fp32=False, reverse=False):
    values = list(reversed(parts)) if reverse else parts
    value = values[0].to(torch.float32 if fp32 else torch.bfloat16).clone()
    for item in values[1:]:
        value = torch.add(value, item.to(value.dtype))
    return value.to(torch.bfloat16)


def check(snapshot_path, diagnostic_path=None, *, expert_cap=48, num_experts=128, snapshot_origin="cuda"):
    path = Path(snapshot_path); diagnostic_path = Path(diagnostic_path or path.with_suffix(".json"))
    saved = torch.load(path, map_location="cpu", weights_only=True)
    d = json.loads(diagnostic_path.read_text())
    require(isinstance(saved, dict) and all(isinstance(t, torch.Tensor) and t.device.type == "cpu" for t in saved.values()), "CPU tensor snapshot")
    indexes = sorted(int(m[1]) for key in saved if (m := re.fullmatch(r"actual_partial_(\d+)", key)))
    require(indexes and indexes == list(range(len(indexes))), "contiguous partial indexes")
    base = {"x", "topk_weights", "topk_ids", "actual", "full_reference", "same_partition_full", "actual_sum_fp32_cast", "actual_sum_reverse_bf16"}
    prefixes = ("actual_partial", "actual_group_map", "same_partition_partial", "full_identity_group_map")
    require(set(saved) == base | {f"{name}_{i}" for name in prefixes for i in indexes}, "snapshot tensor schema")
    parts = [saved[f"actual_partial_{i}"] for i in indexes]
    refs = [saved[f"same_partition_partial_{i}"] for i in indexes]
    x, ids, weights = (saved[k] for k in ("x", "topk_ids", "topk_weights"))
    require(x.ndim == ids.ndim == weights.ndim == 2 and ids.shape == weights.shape and len(ids) == len(x), "input/router rows")
    require(ids.dtype in (torch.int32, torch.int64) and weights.is_floating_point(), "router dtypes")
    require(all(len(set(row)) == len(row) for row in ids.tolist()) and bool(((ids >= 0) & (ids < num_experts)).all()), "top-k expert identities")
    outputs = [saved[k] for k in base - {"topk_weights", "topk_ids"}] + parts + refs
    require(all(t.dtype == torch.bfloat16 and t.shape == x.shape for t in outputs), "BF16 output geometry")
    groups, covered = [], set()
    for i in indexes:
        actual_map, reference_map = saved[f"actual_group_map_{i}"], saved[f"full_identity_group_map_{i}"]
        require(actual_map.dtype == reference_map.dtype == torch.int32 and actual_map.shape == reference_map.shape == (num_experts,), "map dtype/shape")
        values = actual_map.tolist(); enabled = {e for e, slot in enumerate(values) if slot >= 0}
        slots = [slot for slot in values if slot >= 0]
        require(enabled and len(set(slots)) == len(slots) and all(-1 <= slot < expert_cap for slot in values), "scratch slot mapping")
        require(not covered & enabled, "overlapping expert groups")
        require(reference_map.tolist() == [e if e in enabled else -1 for e in range(num_experts)], "full-weight identity mask")
        groups.append(enabled); covered.update(enabled)
    require(covered == set(ids.flatten().tolist()), "active expert coverage")
    require(len(d["groups"]) == len(groups) and all(set(a) == b for a, b in zip(d["groups"], groups)), "reported mask coverage")
    require(d["rtol"] == d["atol"] == .01, "original allclose tolerance")
    actual = saved["actual"]; full = saved["full_reference"]; grouped = total(refs)
    reconstructed = metrics(actual, total(parts))
    require(reconstructed["bit_equal"], "actual partials do not reconstruct actual output")
    for key, value in (("same_partition_full", grouped), ("actual_sum_fp32_cast", total(parts, fp32=True)),
                       ("actual_sum_reverse_bf16", total(parts, reverse=True))):
        require(metrics(saved[key], value)["bit_equal"], "stored aggregation differs: " + key)
    computed = dict(actual_vs_full=metrics(actual, full), actual_vs_same_partition_full=metrics(actual, grouped),
        same_partition_full_vs_full=metrics(grouped, full), actual_vs_fp32_cast=metrics(actual, total(parts, fp32=True)),
        actual_vs_reverse_bf16=metrics(actual, total(parts, reverse=True)))
    partials = [metrics(a, b) for a, b in zip(parts, refs)]
    require(len(d["partial_comparisons"]) == len(partials), "reported partial count")
    norms = {}
    for name, value, claimed in [(k, v, d[k]) for k, v in computed.items()] + [(f"partial_{i}", v, d["partial_comparisons"][i]) for i, v in enumerate(partials)]:
        require(all(claimed[k] == value[k] for k in value if k != "relative_l2"), "reported discrete/maxabs metric mismatch: " + name)
        norms[name] = dict(reported_gpu_or_fixture_relative_l2=claimed["relative_l2"], recomputed_cpu_relative_l2=value["relative_l2"])
    counts = dict(rows=len(x), output_elements=actual.numel(), expected_partial_elements=len(parts)*actual.numel(),
        actual_partial_finite_elements=sum(int(torch.isfinite(t).sum()) for t in parts),
        reference_partial_finite_elements=sum(int(torch.isfinite(t).sum()) for t in refs))
    require(all(d[k] == v for k, v in counts.items()), "reported point counts")
    require(d["actual_partial_reconstruction_bit_equal"] and d["disjoint_complete_group_masks"], "reported reconstruction/coverage")
    size = lambda t: t.numel() * t.element_size()
    payload = sum(map(size, saved.values()))
    expected_accounting = dict(cpu_snapshot_tensor_payload_bytes=payload, serialized_file_bytes=path.stat().st_size,
        snapshot_d2h_payload_bytes=payload if snapshot_origin == "cuda" else 0,
        extra_identity_map_h2d_payload_bytes=len(parts)*num_experts*4 if snapshot_origin == "cuda" else 0,
        retained_partial_and_map_clone_payload_bytes=sum(map(size, parts))+len(parts)*num_experts*4,
        extra_masked_kernel_calls=len(parts), extra_full_weight_copy_bytes=0)
    require(snapshot_origin in ("cuda", "cpu") and all(d["accounting"][k] == v for k, v in expected_accounting.items()), "snapshot payload/file accounting")
    exact = all(v["bit_equal"] and v["allfinite"] for v in partials + [computed["actual_vs_same_partition_full"]]) and computed["actual_vs_full"]["allfinite"]
    gate = "PASS" if exact else "FAIL"
    require(d["gate"]["complete"] and d["gate"]["status"] == gate, "reported gate disagrees with CPU tensors")
    return dict(status="PASS", scope="Snapshot consistency only; independently derived gate below, no GEMM/task-quality validation",
        independently_recomputed_gate=gate, source_snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        tensor_point_counts=counts, group_sizes=list(map(len, groups)), partial_comparisons=partials,
        recomputed_metrics=computed, relative_l2_comparison=norms, accounting=expected_accounting,
        limits="CPU/GPU FP32 norm reductions are reported separately and do not affect gate; transfer payload assumes the declared source device and frozen capture operations, not measured DMA; full weight tensors are absent")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path); parser.add_argument("--diagnostic", type=Path)
    parser.add_argument("--expert-cap", type=int, default=48); parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--snapshot-origin", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.snapshot, args.diagnostic, expert_cap=args.expert_cap, num_experts=args.num_experts, snapshot_origin=args.snapshot_origin)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")

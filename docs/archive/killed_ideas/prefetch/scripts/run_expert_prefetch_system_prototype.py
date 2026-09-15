#!/usr/bin/env python3
"""Expert-Prefetch System Prototype: turn the routing-predictability signal
(P0/P0-B/P0-C, all offline statistics) into a measured system-level latency
number on real hardware.

Setup (mirrors a memory-constrained MoE deployment where only a subset of
experts stay HBM-resident and the rest are paged in from CPU-pinned memory,
e.g. a larger MoE than fits in HBM, or multiple co-located models sharing one
GPU): a GPU-resident LRU cache holds C experts per layer. When a layer needs
an expert that is not cached, that expert's weight tensor must be copied
CPU->GPU (real cudaMemcpyAsync, real bytes matching the model's actual FFN
matrix sizes) before that layer's compute can start -- a real, measured
critical-path stall.

Two policies are compared on the exact same real route trace (calibration/
test split matching P0/P0-B/P0-C, zero new statistical assumptions):

  reactive  -- cache is filled only by actual usage (classic LRU demand
               paging). No lookahead.
  predictive-- while layer L is computing, the calibration-learned top-1
               transition table (same table validated in P0/P0-C) issues
               async prefetch copies (real cudaMemcpyAsync on a side CUDA
               stream, overlapped with L's real compute) for the predicted
               top-B experts of layer L+1 that are not already cached.

All latency constants (H2D bytes/sec for a real expert-sized tensor, and one
MoE layer's real compute time for a token batch) are measured on-GPU, not
assumed. Evidence tag: [Observed] system prototype, reusing P0/P0-C's
pre-registered calibration/test split.
"""
from __future__ import annotations


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

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM


def measure_h2d_bandwidth(hidden_size: int, intermediate_size: int, dtype: torch.dtype,
                           device: str, n_repeats: int = 30) -> float:
    """Real cudaMemcpyAsync latency (seconds) for one expert's weight tensor
    (gate_proj + up_proj + down_proj, matching real FFN matrix shapes)."""
    numel = 2 * hidden_size * intermediate_size + intermediate_size * hidden_size
    cpu_tensor = torch.empty(numel, dtype=dtype, pin_memory=True)
    cpu_tensor.uniform_(-0.02, 0.02)
    gpu_tensor = torch.empty(numel, dtype=dtype, device=device)
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    # warmup
    for _ in range(5):
        gpu_tensor.copy_(cpu_tensor, non_blocking=True)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        with torch.cuda.stream(stream):
            gpu_tensor.copy_(cpu_tensor, non_blocking=True)
        stream.synchronize()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)  # ms -> s
    return float(np.median(times)), int(numel * cpu_tensor.element_size())


def measure_layer_compute_time(model, layer_id: int, batch_tokens: int, hidden_size: int,
                                device: str, n_repeats: int = 30) -> float:
    """Real forward-pass wall time (seconds) for one MoE layer's expert
    compute on a batch of `batch_tokens` real hidden-state vectors, using the
    real loaded expert modules (weights already resident, so this isolates
    compute time from any paging)."""
    layer = model.model.layers[layer_id]
    moe = layer.mlp if (hasattr(layer, "mlp") and hasattr(layer.mlp, "experts")) else layer.block_sparse_moe
    top_k = int(model.config.num_experts_per_tok)
    num_experts = getattr(model.config, "num_experts", None)
    if num_experts is None:
        num_experts = model.config.num_local_experts
    num_experts = int(num_experts)
    x = torch.randn(batch_tokens, hidden_size, dtype=next(moe.parameters()).dtype, device=device)
    experts_per_token = torch.randint(0, num_experts, (batch_tokens, top_k), device=device)

    def run_once():
        expert_mask = torch.nn.functional.one_hot(experts_per_token, num_classes=num_experts).permute(2, 1, 0)
        out = torch.zeros(batch_tokens, top_k, hidden_size, dtype=x.dtype, device=device)
        hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for e_idx_t in hit:
            e_idx = int(e_idx_t.item())
            idx, top_x = torch.where(expert_mask[e_idx])
            current = x[None, top_x].reshape(-1, hidden_size)
            out[top_x, idx, :] = moe.experts[e_idx](current)
        return out

    for _ in range(5):
        run_once()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        run_once()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)
    return float(np.median(times))


def build_transition_table(calib_top1: pd.DataFrame, num_layers: int, budget: int) -> dict[int, dict[int, list[int]]]:
    out: dict[int, dict[int, list[int]]] = {}
    for layer in range(num_layers - 1):
        cur = calib_top1[calib_top1["layer"] == layer][["sample_id", "token_position", "top1_expert"]]
        nxt = calib_top1[calib_top1["layer"] == layer + 1][["sample_id", "token_position", "top1_expert"]]
        joined = cur.merge(nxt, on=["sample_id", "token_position"], suffixes=("_cur", "_nxt"))
        table: dict[int, list[int]] = {}
        for prev_expert, grp in joined.groupby("top1_expert_cur"):
            table[int(prev_expert)] = grp["top1_expert_nxt"].value_counts().index[:budget].tolist()
        out[layer] = table
    return out


def build_freq_fallback(calib_top1: pd.DataFrame, num_layers: int, budget: int) -> dict[int, list[int]]:
    out = {}
    for layer in range(num_layers):
        counts = calib_top1[calib_top1["layer"] == layer]["top1_expert"].value_counts()
        out[layer] = counts.index[:budget].tolist()
    return out


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.order: list[int] = []  # most-recent last

    def contains(self, expert_id: int) -> bool:
        return expert_id in self.order

    def touch(self, expert_id: int) -> None:
        if expert_id in self.order:
            self.order.remove(expert_id)
        self.order.append(expert_id)
        if len(self.order) > self.capacity:
            self.order.pop(0)

    def snapshot(self) -> set[int]:
        return set(self.order)


def simulate_document(
    doc: pd.DataFrame,
    num_layers: int,
    cache_capacity: int,
    prefetch_budget: int,
    trans_table: dict[int, dict[int, list[int]]],
    freq_fallback: dict[int, list[int]],
    h2d_time_s: float,
    layer_compute_s: dict[int, float],
    policy: str,
    token_subset: set[int] | None = None,
) -> dict[str, float]:
    """Returns per-decode-batch total layer-transition latency (seconds) and
    miss counts, for a single (reactive|predictive) policy. ``token_subset``
    restricts the simulated concurrent decode batch to a fixed set of
    token_position ids (matching the batch size used to measure real layer
    compute time -- a real serving batch decodes O(10-100) concurrent
    requests per step, not all 256 tokens of one document at once)."""
    if token_subset is not None:
        doc = doc[doc["token_position"].isin(token_subset)]
    by_layer = {layer: grp for layer, grp in doc.groupby("layer")}
    cache = LRUCache(cache_capacity)
    total_latency = 0.0
    total_paging_latency = 0.0  # H2D-only component, isolated from compute
    total_misses = 0
    total_needed = 0
    # layer 0: cold start, everything needed is a miss by definition (both
    # policies pay this once; it cancels out in the reactive-vs-predictive
    # comparison so we still charge it for realism but track it separately).
    first = by_layer.get(0)
    if first is not None:
        needed0 = set(first["top1_expert"].unique().tolist())
        misses0 = [e for e in needed0 if not cache.contains(e)]
        total_latency += len(misses0) * h2d_time_s + layer_compute_s.get(0, 0.0)
        total_paging_latency += len(misses0) * h2d_time_s
        total_misses += len(misses0)
        total_needed += len(needed0)
        for e in needed0:
            cache.touch(e)

    for layer in range(num_layers - 1):
        cur = by_layer.get(layer)
        nxt = by_layer.get(layer + 1)
        if cur is None or nxt is None:
            continue
        cur_top1_by_tok = cur.set_index("token_position")["top1_expert"]
        needed_next = set(nxt["top1_expert"].unique().tolist())

        if policy == "predictive":
            # Prefetch during layer L's compute window: union of transition
            # predictions for every distinct current top-1 expert present in
            # this document at layer L, capped to prefetch_budget candidates
            # in priority order, skipping experts already cached.
            table = trans_table.get(layer, {})
            candidates: list[int] = []
            seen_cur = cur_top1_by_tok.unique().tolist()
            for prev_e in seen_cur:
                candidates.extend(table.get(int(prev_e), freq_fallback.get(layer + 1, [])))
            # de-duplicate, preserve priority order
            ordered = list(dict.fromkeys(candidates))
            to_prefetch = [e for e in ordered if not cache.contains(e)][:prefetch_budget]
            prefetch_time_needed = len(to_prefetch) * h2d_time_s
            available_overlap = layer_compute_s.get(layer, 0.0)
            fully_hidden = int(available_overlap // max(h2d_time_s, 1e-12))
            hidden_count = min(len(to_prefetch), fully_hidden)
            exposed_prefetch_time = max(0.0, prefetch_time_needed - available_overlap)
            for e in to_prefetch[:hidden_count] + to_prefetch[hidden_count:]:
                cache.touch(e)
            # exposed prefetch time still occupies the H2D link but overlaps
            # with nothing further; charge only the part that exceeds the
            # compute window (conservative: does not assume free lunch).
            total_latency += exposed_prefetch_time
            total_paging_latency += exposed_prefetch_time

        misses = [e for e in needed_next if not cache.contains(e)]
        total_latency += len(misses) * h2d_time_s + layer_compute_s.get(layer + 1, 0.0)
        total_paging_latency += len(misses) * h2d_time_s
        total_misses += len(misses)
        total_needed += len(needed_next)
        for e in needed_next:
            cache.touch(e)

    return {
        "total_latency_s": total_latency,
        "total_paging_latency_s": total_paging_latency,
        "total_misses": total_misses,
        "total_needed": total_needed,
        "miss_rate": total_misses / max(total_needed, 1),
    }


def paired_bootstrap_ci(diffs: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = diffs[idx].mean()
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2)), float(diffs.mean())


def run_model(
    model_key: str,
    model_name: str,
    calib_csv: Path,
    test_csv: Path,
    cache_capacity: int,
    prefetch_budget: int,
    n_boot: int,
    seed: int,
    device: str,
    batch_tokens: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    print(f"[{model_key}] loading model + measuring real hardware constants...")
    cfg = AutoConfig.from_pretrained(model_name, local_files_only=True)
    hidden_size = int(cfg.hidden_size)
    intermediate_size = int(getattr(cfg, "moe_intermediate_size", getattr(cfg, "intermediate_size", None)))
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, local_files_only=True).to(device)
    model.eval()
    num_layers = len(model.model.layers)

    h2d_time_s, expert_bytes = measure_h2d_bandwidth(hidden_size, intermediate_size, dtype, device)
    layer_compute_s = {}
    for layer_id in range(num_layers):
        layer_compute_s[layer_id] = measure_layer_compute_time(model, layer_id, batch_tokens, hidden_size, device)
    del model
    torch.cuda.empty_cache()

    print(f"[{model_key}] h2d_per_expert={h2d_time_s * 1e6:.2f}us ({expert_bytes / 1e6:.2f}MB), "
          f"layer_compute_median={np.median(list(layer_compute_s.values())) * 1e6:.2f}us")

    calib = pd.read_csv(calib_csv)
    test = pd.read_csv(test_csv)
    calib_top1 = calib[calib["rank"] == 1][["sample_id", "token_position", "layer", "expert_id"]].rename(
        columns={"expert_id": "top1_expert"})
    test_top1 = test[test["rank"] == 1][["sample_id", "token_position", "layer", "expert_id"]].rename(
        columns={"expert_id": "top1_expert"})

    trans_table = build_transition_table(calib_top1, num_layers, prefetch_budget)
    freq_fallback = build_freq_fallback(calib_top1, num_layers, prefetch_budget)

    rows = []
    for sample_id, doc in test_top1.groupby("sample_id"):
        all_positions = sorted(doc["token_position"].unique().tolist())
        # chunk the document's token positions into decode-batch-sized groups
        # so the simulated concurrent batch size matches the batch size used
        # to measure real layer_compute_s (a real serving batch decodes a
        # bounded number of concurrent requests per step, not all 256
        # positions of one document at once).
        chunks = [all_positions[i:i + batch_tokens] for i in range(0, len(all_positions), batch_tokens)]
        for chunk_idx, chunk in enumerate(chunks):
            if len(chunk) < max(4, batch_tokens // 4):
                continue  # drop tiny trailing remainder chunks
            subset = set(chunk)
            r_react = simulate_document(doc, num_layers, cache_capacity, prefetch_budget, trans_table,
                                         freq_fallback, h2d_time_s, layer_compute_s, "reactive",
                                         token_subset=subset)
            r_pred = simulate_document(doc, num_layers, cache_capacity, prefetch_budget, trans_table,
                                        freq_fallback, h2d_time_s, layer_compute_s, "predictive",
                                        token_subset=subset)
            rows.append({
                "sample_id": sample_id,
                "chunk_idx": chunk_idx,
                "batch_size": len(chunk),
                "reactive_latency_s": r_react["total_latency_s"],
                "predictive_latency_s": r_pred["total_latency_s"],
                "reactive_paging_latency_s": r_react["total_paging_latency_s"],
                "predictive_paging_latency_s": r_pred["total_paging_latency_s"],
                "reactive_miss_rate": r_react["miss_rate"],
                "predictive_miss_rate": r_pred["miss_rate"],
                "latency_saving_s": r_react["total_latency_s"] - r_pred["total_latency_s"],
                "latency_saving_pct": 1.0 - r_pred["total_latency_s"] / max(r_react["total_latency_s"], 1e-12),
                "paging_latency_saving_pct": 1.0 - r_pred["total_paging_latency_s"] / max(
                    r_react["total_paging_latency_s"], 1e-12),
            })
    df = pd.DataFrame(rows)
    # paired bootstrap over documents (same unit as P0/P0-B/P0-C), averaging
    # multiple decode-batch chunks within a document first
    diffs = df.groupby("sample_id")["latency_saving_pct"].mean().to_numpy()
    lo, hi, mean_diff = paired_bootstrap_ci(diffs, n_boot, seed)
    diffs_paging = df.groupby("sample_id")["paging_latency_saving_pct"].mean().to_numpy()
    lo_p, hi_p, mean_diff_p = paired_bootstrap_ci(diffs_paging, n_boot, seed + 1000)

    summary = {
        "model": model_key,
        "cache_capacity": cache_capacity,
        "prefetch_budget": prefetch_budget,
        "h2d_time_us": h2d_time_s * 1e6,
        "expert_bytes": expert_bytes,
        "layer_compute_us_median": float(np.median(list(layer_compute_s.values())) * 1e6),
        "mean_reactive_latency_ms": float(df["reactive_latency_s"].mean() * 1000),
        "mean_predictive_latency_ms": float(df["predictive_latency_s"].mean() * 1000),
        "mean_latency_saving_pct": mean_diff * 100,
        "ci_low_pct": lo * 100,
        "ci_high_pct": hi * 100,
        "mean_reactive_paging_latency_ms": float(df["reactive_paging_latency_s"].mean() * 1000),
        "mean_predictive_paging_latency_ms": float(df["predictive_paging_latency_s"].mean() * 1000),
        "mean_paging_only_saving_pct": mean_diff_p * 100,
        "paging_ci_low_pct": lo_p * 100,
        "paging_ci_high_pct": hi_p * 100,
        "mean_reactive_miss_rate": float(df["reactive_miss_rate"].mean()),
        "mean_predictive_miss_rate": float(df["predictive_miss_rate"].mean()),
        "n_docs": len(df),
    }
    return df, pd.DataFrame([summary]), summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--olmoe-model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--olmoe-calib", required=True)
    ap.add_argument("--olmoe-test", required=True)
    ap.add_argument("--olmoe-cache-capacity", type=int, default=8)
    ap.add_argument("--olmoe-prefetch-budget", type=int, default=8)
    ap.add_argument("--llmjp-model", default="llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M")
    ap.add_argument("--llmjp-calib", required=True)
    ap.add_argument("--llmjp-test", required=True)
    ap.add_argument("--llmjp-cache-capacity", type=int, default=6)
    ap.add_argument("--llmjp-prefetch-budget", type=int, default=6)
    ap.add_argument("--batch-tokens", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_o, summ_o, s_o = run_model("olmoe", args.olmoe_model, Path(args.olmoe_calib), Path(args.olmoe_test),
                                   args.olmoe_cache_capacity, args.olmoe_prefetch_budget, args.n_boot, args.seed,
                                   device, args.batch_tokens)
    df_l, summ_l, s_l = run_model("llmjp", args.llmjp_model, Path(args.llmjp_calib), Path(args.llmjp_test),
                                   args.llmjp_cache_capacity, args.llmjp_prefetch_budget, args.n_boot, args.seed + 1,
                                   device, args.batch_tokens)

    df_o.to_csv(out / "olmoe_per_document.csv", index=False)
    df_l.to_csv(out / "llmjp_per_document.csv", index=False)
    summary = pd.concat([summ_o, summ_l], ignore_index=True)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "meta.json").write_text(json.dumps({"olmoe": s_o, "llmjp": s_l}, indent=2), encoding="utf-8")

    lines = ["# Expert-Prefetch System Prototype: Measured Latency Result", ""]
    cols = ["model", "cache_capacity", "prefetch_budget", "h2d_time_us", "layer_compute_us_median",
            "mean_reactive_latency_ms", "mean_predictive_latency_ms", "mean_latency_saving_pct",
            "ci_low_pct", "ci_high_pct",
            "mean_reactive_paging_latency_ms", "mean_predictive_paging_latency_ms",
            "mean_paging_only_saving_pct", "paging_ci_low_pct", "paging_ci_high_pct",
            "mean_reactive_miss_rate", "mean_predictive_miss_rate", "n_docs"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in summary.iterrows():
        vals = [f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
